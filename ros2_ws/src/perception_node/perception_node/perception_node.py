"""
perception_node: RGB-D -> pose -> tracker -> ReID -> owner-3D.

Subscriptions:
  /camera/color/image_raw         sensor_msgs/Image  bgr8
  /camera/depth/image_aligned     sensor_msgs/Image  16UC1 (mm)
  /camera/color/camera_info       sensor_msgs/CameraInfo (intrinsics)
  /audio/doa                      std_msgs/Float32MultiArray  [az_deg, conf]
  /audio/wakeword                 std_msgs/String  (KWS hits drive owner bind)

Publishes:
  /perception/owner_3d            geometry_msgs/PointStamped  (mouth-3D, m, color frame)
  /perception/persons_count       std_msgs/UInt8
  /perception/owner_track_id      std_msgs/Int32  (-1 if not bound)
  /perception/owner_pixel         geometry_msgs/Point  (mouth pixel + raw depth_mm in z)
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, Int32, String, UInt8

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.backends.pose_detector import select_pose_detector  # noqa: E402
from audio_frontend.backends.reid import select_reid                    # noqa: E402
from audio_frontend.vision.gallery import OwnerGallery                  # noqa: E402
from audio_frontend.vision.mouth_estimator import estimate_mouth_3d     # noqa: E402
from audio_frontend.vision.tracker import IouTracker                    # noqa: E402


def _decode_color(msg: Image) -> Optional[np.ndarray]:
    if msg.encoding != "bgr8":
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    return buf.reshape(int(msg.height), int(msg.width), 3)


def _decode_depth(msg: Image) -> Optional[np.ndarray]:
    if msg.encoding not in ("16UC1", "mono16"):
        return None
    buf = np.frombuffer(msg.data, dtype=np.uint16)
    return buf.reshape(int(msg.height), int(msg.width))


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_aligned")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        self.declare_parameter("doa_topic", "/audio/doa")
        self.declare_parameter("wakeword_topic", "/audio/wakeword")
        self.declare_parameter("pose_backend", "auto")  # auto | cpu | hailo
        self.declare_parameter("reid_backend", "auto")
        self.declare_parameter("conf_thresh", 0.35)
        self.declare_parameter("iou_thresh", 0.55)
        self.declare_parameter("gallery_match_thresh", 0.55)
        self.declare_parameter("owner_lost_ttl_s", 8.0)
        self.declare_parameter("bind_doa_tol_deg", 25.0)
        self.declare_parameter("max_fps", 5.0)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.create_subscription(Image,
                                 str(self.get_parameter("color_topic").value),
                                 self._on_color, sensor_qos)
        self.create_subscription(Image,
                                 str(self.get_parameter("depth_topic").value),
                                 self._on_depth, sensor_qos)
        self.create_subscription(CameraInfo,
                                 str(self.get_parameter("info_topic").value),
                                 self._on_info, 1)
        self.create_subscription(Float32MultiArray,
                                 str(self.get_parameter("doa_topic").value),
                                 self._on_doa, 5)
        self.create_subscription(String,
                                 str(self.get_parameter("wakeword_topic").value),
                                 self._on_wake, 5)

        self.pub_owner_3d = self.create_publisher(
            PointStamped, "/perception/owner_3d", 5)
        self.pub_count = self.create_publisher(
            UInt8, "/perception/persons_count", 5)
        self.pub_owner_track = self.create_publisher(
            Int32, "/perception/owner_track_id", 5)
        self.pub_owner_px = self.create_publisher(
            Point, "/perception/owner_pixel", 5)

        self._lock = threading.Lock()
        self._latest_color: Optional[tuple[int, np.ndarray]] = None  # (stamp_ns, bgr)
        self._latest_depth: Optional[tuple[int, np.ndarray]] = None
        self._K: Optional[np.ndarray] = None
        self._frame_id: str = "camera_color_optical_frame"
        self._latest_doa: Optional[tuple[float, float]] = None       # (az, conf)

        self._pose = select_pose_detector(
            prefer=str(self.get_parameter("pose_backend").value))
        self._reid = select_reid(
            prefer=str(self.get_parameter("reid_backend").value))
        self._tracker = IouTracker(
            iou_thresh=float(self.get_parameter("iou_thresh").value),
            max_miss=15,
        )
        self._gallery = OwnerGallery(
            match_thresh=float(self.get_parameter("gallery_match_thresh").value),
            lost_ttl_s=float(self.get_parameter("owner_lost_ttl_s").value),
            bind_doa_tol_deg=float(self.get_parameter("bind_doa_tol_deg").value),
        )

        self._stop = threading.Event()
        self._loop_thread = threading.Thread(target=self._loop, daemon=True)
        self._loop_thread.start()

        self.get_logger().info(
            f"perception_node up: pose={self._pose.name}, reid={self._reid.name}"
        )

    # ----------------------------------------------------------------- subs
    def _on_color(self, msg: Image) -> None:
        bgr = _decode_color(msg)
        if bgr is None:
            return
        ts = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            self._latest_color = (ts, bgr)
            if msg.header.frame_id:
                self._frame_id = str(msg.header.frame_id)

    def _on_depth(self, msg: Image) -> None:
        d = _decode_depth(msg)
        if d is None:
            return
        ts = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            self._latest_depth = (ts, d)

    def _on_info(self, msg: CameraInfo) -> None:
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self._lock:
            self._K = K

    def _on_doa(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            with self._lock:
                self._latest_doa = (float(msg.data[0]), float(msg.data[1]))

    def _on_wake(self, msg: String) -> None:
        if not msg.data:
            return
        with self._lock:
            doa = self._latest_doa
            tracks = list(self._tracker.tracks.values())
            K = self._K
        if doa is None or K is None:
            self.get_logger().warning("KWS fired but no DOA / camera_info yet")
            return
        az_of = self._track_azimuth_map(tracks, K)
        bound = self._gallery.bind_on_wake(tracks, doa[0], az_of)
        self.get_logger().info(
            f"KWS '{msg.data}' DOA={doa[0]:.1f}° -> bound owner track_id={bound}"
        )

    # ---------------------------------------------------------------- helpers
    def _track_azimuth_map(self, tracks, K) -> Dict[int, float]:
        """Approximate azimuth (deg, CCW from +x) of each track from its bbox
        centroid X relative to the principal point. Used only at bind time;
        no need for depth here.
        """
        if K is None:
            return {}
        cx = float(K[0, 2]); fx = float(K[0, 0])
        out = {}
        for t in tracks:
            x_c = float((t.bbox[0] + t.bbox[2]) / 2.0)
            theta = -np.degrees(np.arctan((x_c - cx) / fx))  # negative-X => +CCW
            out[int(t.track_id)] = float(theta)
        return out

    def _publish_owner(self, stamp_ns: int, est, track_id: int) -> None:
        ps = PointStamped()
        sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
        ps.header.stamp.sec = int(sec)
        ps.header.stamp.nanosec = int(nsec)
        ps.header.frame_id = self._frame_id
        x, y, z = est.point_3d_m
        ps.point.x = float(x); ps.point.y = float(y); ps.point.z = float(z)
        self.pub_owner_3d.publish(ps)

        m_track = Int32(); m_track.data = int(track_id)
        self.pub_owner_track.publish(m_track)

        m_px = Point()
        m_px.x = float(est.pixel_xy[0]); m_px.y = float(est.pixel_xy[1])
        m_px.z = float(est.point_3d_m[2] * 1000.0)
        self.pub_owner_px.publish(m_px)

    # ------------------------------------------------------------------- loop
    def _loop(self) -> None:
        max_fps = float(self.get_parameter("max_fps").value)
        period = 1.0 / max(max_fps, 0.1)
        last = 0.0
        while not self._stop.is_set() and rclpy.ok():
            now = time.monotonic()
            if now - last < period:
                time.sleep(0.005)
                continue
            last = now

            with self._lock:
                col = self._latest_color
                dep = self._latest_depth
                K = self._K
                doa = self._latest_doa
            if col is None or dep is None or K is None:
                continue
            ts_col, bgr = col
            ts_dep, depth = dep
            if abs(ts_col - ts_dep) > 100_000_000:    # >100ms skew -> skip
                continue

            persons = self._pose.detect(bgr)
            dets = np.array([p.bbox_xyxy for p in persons], dtype=np.float32) \
                if persons else np.zeros((0, 4), dtype=np.float32)
            scores = np.array([p.confidence for p in persons], dtype=np.float32) \
                if persons else np.zeros(0, dtype=np.float32)
            kps = np.stack([p.keypoints for p in persons]) \
                if persons else np.zeros((0, 17, 3), dtype=np.float32)

            tracks = self._tracker.update(dets_xyxy=dets, scores=scores,
                                           keypoints=kps if persons else None)

            for t in tracks:
                if t.miss > 0:
                    continue
                x0, y0, x1, y1 = (max(0, int(round(t.bbox[0]))),
                                   max(0, int(round(t.bbox[1]))),
                                   max(0, int(round(t.bbox[2]))),
                                   max(0, int(round(t.bbox[3]))))
                if x1 - x0 < 20 or y1 - y0 < 40:
                    continue
                crop = bgr[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                t.embedding = self._reid.embed(crop)

            owner_id = self._gallery.step(tracks)
            self.pub_count.publish(UInt8(data=int(len(tracks))))

            owner = next((t for t in tracks if t.track_id == owner_id), None)
            if owner is None or owner.keypoints is None:
                idle = Int32(); idle.data = -1
                self.pub_owner_track.publish(idle)
                continue
            est = estimate_mouth_3d(
                owner.keypoints, tuple(owner.bbox), depth, K
            )
            if not est.valid_depth:
                continue
            self._publish_owner(ts_col, est, owner_id)

    # ------------------------------------------------------------------- exit
    def stop(self) -> None:
        self._stop.set()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
