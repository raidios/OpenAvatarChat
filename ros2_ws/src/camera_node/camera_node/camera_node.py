"""
Orbbec Gemini Pro -> ROS2 camera_node.

Pulls 640x480 BGR + hardware-D2C-aligned depth via the SDK path
(`OrbbecGemini.read_rgbd_sdk()`) and publishes:

  /camera/color/image_raw      sensor_msgs/Image  bgr8 (480, 640, 3)
  /camera/depth/image_aligned  sensor_msgs/Image  mono16 (480, 640)  uint16 mm
  /camera/color/camera_info    sensor_msgs/CameraInfo

Color intrinsics resolution order (first hit wins):
  1) `calib_npz` parameter pointing to an existing RgbdCalibration npz —
     use this for chessboard-calibrated overrides.
  2) Live read via `OrbbecGemini.get_camera_param()` (factory params from
     EEPROM). Accurate to <1 px on the SV1301S_U3 sensor and what the SDK
     itself uses for HW D2C; this is the recommended path.
  3) Pinhole approx from `hfov_deg` (default 85°). Last-resort fallback;
     fx is ~30% off the true value for Gemini Pro at 640x480, so 3D
     back-projections will be visibly distorted. Logged as a warning.

The default `calib_npz` is `config/camera_calib_factory.npz`, which the
one-shot tool `tools/dump_factory_intrinsics.py` produces by reading
EEPROM and writing it to disk; once that file exists, intrinsics load from
disk without needing to re-open the device.

Constraints baked in (per .cursor/rules/orbbec-gemini-device.mdc):
  - one process opens the device; we don't reopen on read errors
  - color is 640x480 (only D2C-supported profile)
  - color/depth share systemTimeStampUs, no message_filters needed downstream
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

REPO_ROOT = Path(__file__).resolve().parents[4]
ORBBEC_DIR = REPO_ROOT / "thirdparty" / "OrbbecSDK_v1" / "python"
sys.path.insert(0, str(ORBBEC_DIR))

from orbbec_gemini import (  # noqa: E402
    FactoryCameraParam,
    OrbbecGemini,
    OrbbecGeminiError,
    RgbdCalibration,
)


def _intrinsics_from_fov(width: int, height: int, hfov_deg: float
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Build a pinhole K + zero distortion from horizontal FOV."""
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx                                  # square pixels
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    return K, D


def _load_calib_npz(npz_path: str, width: int, height: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    cal = RgbdCalibration.load(npz_path)
    return _scale_K(cal.K_color, cal.color_size, width, height), \
        cal.dist_color.astype(np.float64).copy().reshape(-1)


def _scale_K(K_src: np.ndarray, src_size: tuple[int, int],
             dst_w: int, dst_h: int) -> np.ndarray:
    K = K_src.astype(np.float64).copy()
    sx = dst_w / float(src_size[0])
    sy = dst_h / float(src_size[1])
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def _factory_intrinsics(cam: OrbbecGemini, width: int, height: int
                        ) -> tuple[np.ndarray, np.ndarray, FactoryCameraParam]:
    """Pull factory params from the device and rescale to (width, height).

    Returns (K, D, raw_param). D uses the 5-coeff [k1,k2,p1,p2,k3] subset to
    match plumb_bob in CameraInfo.
    """
    p = cam.get_camera_param()
    K = _scale_K(p.K_color, p.color_size, width, height)
    D = p.dist_color[:5].astype(np.float64).copy()
    return K, D, p


def _image_msg(arr: np.ndarray, encoding: str, frame_id: str,
               stamp_ns: int) -> Image:
    msg = Image()
    msg.header.frame_id = frame_id
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    msg.header.stamp.sec = int(sec)
    msg.header.stamp.nanosec = int(nsec)
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    if arr.ndim == 3:
        msg.step = int(arr.shape[1] * arr.shape[2] * arr.dtype.itemsize)
    else:
        msg.step = int(arr.shape[1] * arr.dtype.itemsize)
    msg.data = arr.tobytes()
    return msg


def _camera_info_msg(K: np.ndarray, D: np.ndarray, width: int, height: int,
                     frame_id: str, stamp_ns: int) -> CameraInfo:
    ci = CameraInfo()
    ci.header.frame_id = frame_id
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    ci.header.stamp.sec = int(sec)
    ci.header.stamp.nanosec = int(nsec)
    ci.height = int(height)
    ci.width = int(width)
    ci.distortion_model = "plumb_bob"
    ci.d = D.flatten().tolist()
    ci.k = K.flatten().tolist()
    # rectified pinhole == K (no stereo rectification involved)
    ci.r = np.eye(3, dtype=np.float64).flatten().tolist()
    P = np.zeros((3, 4), dtype=np.float64)
    P[:3, :3] = K
    ci.p = P.flatten().tolist()
    return ci


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_aligned")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("color_fps", 30)
        self.declare_parameter("color_format", "mjpg")
        self.declare_parameter("color_frame_id", "camera_color_optical_frame")
        self.declare_parameter("hfov_deg", 85.0)
        # Default points to the EEPROM dump produced by
        # `tools/dump_factory_intrinsics.py`. Empty string => skip file load.
        # If the path is set but missing, we fall back to a live device read.
        default_calib = str(REPO_ROOT / "config" / "camera_calib_factory.npz")
        self.declare_parameter("calib_npz", default_calib)
        self.declare_parameter("d2c_mode", "hw")  # hw | sw | off (off => no align)
        self.declare_parameter("read_timeout_ms", 600)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_color = self.create_publisher(
            Image, str(self.get_parameter("color_topic").value), sensor_qos)
        self.pub_depth = self.create_publisher(
            Image, str(self.get_parameter("depth_topic").value), sensor_qos)
        self.pub_info = self.create_publisher(
            CameraInfo, str(self.get_parameter("camera_info_topic").value), 1)

        self._cam: Optional[OrbbecGemini] = None
        self._stop = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="camera-stream", daemon=True)
        self._frame_id = str(self.get_parameter("color_frame_id").value)
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None

    def start(self) -> None:
        self._open_camera()
        self._stream_thread.start()

    def _open_camera(self) -> None:
        cw = int(self.get_parameter("color_width").value)
        ch = int(self.get_parameter("color_height").value)
        fps = int(self.get_parameter("color_fps").value)
        fmt = str(self.get_parameter("color_format").value)
        d2c = str(self.get_parameter("d2c_mode").value)
        calib_npz = str(self.get_parameter("calib_npz").value)
        hfov = float(self.get_parameter("hfov_deg").value)

        # 1) calib_npz file (operator override / chessboard result)
        if calib_npz and Path(calib_npz).exists():
            self._K, self._D = _load_calib_npz(calib_npz, cw, ch)
            self.get_logger().info(
                f"loaded color intrinsics from {calib_npz} "
                f"(fx={self._K[0, 0]:.1f}, cx={self._K[0, 2]:.1f})"
            )
            self._calib_source = "npz"
        else:
            self._K = None  # decide after pipeline opens
            self._D = None
            self._calib_source = None
            if calib_npz:
                self.get_logger().warning(
                    f"calib_npz='{calib_npz}' does not exist; will try live "
                    f"factory params, then hfov fallback."
                )

        self._cam = OrbbecGemini(streams=["depth", "color"])
        info = self._cam.open(
            color_width=cw, color_height=ch, color_fps=fps,
            color_format=fmt, d2c=d2c,
        )
        self.get_logger().info(
            f"Gemini Pro opened: depth={info.depth_width}x{info.depth_height}@"
            f"{info.depth_fps}, color={info.color_width}x{info.color_height}@"
            f"{info.color_fps} {info.color_format}, d2c={info.d2c_mode}"
        )
        if d2c != "off" and info.d2c_mode == "off":
            self.get_logger().error(
                "requested D2C but device returned d2c=off; aligned depth "
                "will be empty. Check that color size is 640x480."
            )

        # 2) live factory params (EEPROM via SDK)
        if self._K is None:
            try:
                self._K, self._D, p = _factory_intrinsics(self._cam, cw, ch)
                self._calib_source = "factory"
                self.get_logger().info(
                    f"loaded factory intrinsics from device: "
                    f"fx={self._K[0, 0]:.1f}, cx={self._K[0, 2]:.1f} "
                    f"(calib resolution {p.color_size[0]}x{p.color_size[1]}). "
                    f"Run `tools/dump_factory_intrinsics.py` to cache to npz."
                )
            except OrbbecGeminiError as e:
                self.get_logger().warning(
                    f"factory params unavailable ({e}); falling back to "
                    f"hfov-based pinhole."
                )

        # 3) hfov fallback
        if self._K is None:
            self._K, self._D = _intrinsics_from_fov(cw, ch, hfov)
            self._calib_source = "hfov"
            self.get_logger().warning(
                f"using pinhole approx K from hfov={hfov}° "
                f"(fx={self._K[0, 0]:.1f}). 3D back-projection will be "
                f"systematically biased; run `tools/dump_factory_intrinsics.py`."
            )

    def stop(self) -> None:
        self._stop.set()
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None

    def _stream_loop(self) -> None:
        if self._cam is None:
            return
        timeout_ms = int(self.get_parameter("read_timeout_ms").value)
        miss = 0
        last_log = time.monotonic()
        n_frames = 0
        while not self._stop.is_set() and rclpy.ok():
            try:
                rgbd = self._cam.read_rgbd_sdk(timeout_ms=timeout_ms)
            except OrbbecGeminiError as e:
                self.get_logger().error(f"read_rgbd_sdk: {e}")
                time.sleep(0.1)
                continue
            if rgbd is None:
                miss += 1
                if miss >= 3:
                    self.get_logger().warning(
                        "3 consecutive read timeouts; the device may be "
                        "stalled. Per device-rule, NOT auto-reopening; "
                        "physical replug may be required."
                    )
                    miss = 0
                continue
            miss = 0
            n_frames += 1
            stamp_ns = rgbd.color.timestamp_us * 1000
            self.pub_color.publish(_image_msg(
                rgbd.color.data, "bgr8", self._frame_id, stamp_ns))
            self.pub_depth.publish(_image_msg(
                rgbd.aligned_depth, "16UC1", self._frame_id, stamp_ns))
            self.pub_info.publish(_camera_info_msg(
                self._K, self._D,
                rgbd.color.data.shape[1], rgbd.color.data.shape[0],
                self._frame_id, stamp_ns,
            ))
            now = time.monotonic()
            if now - last_log > 5.0:
                fps = n_frames / (now - last_log)
                self.get_logger().info(
                    f"camera throughput: {fps:.1f} fps "
                    f"(scale_mm_per_unit={rgbd.scale_mm_per_unit:.3f})")
                last_log = now
                n_frames = 0


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = CameraNode()
    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except OrbbecGeminiError as e:
        node.get_logger().error(f"camera open failed: {e}")
    finally:
        node.stop()
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
