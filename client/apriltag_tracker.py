"""
Marker detection and pose estimation via OpenCV ArUco API.

Default dictionary is ``DICT_5X5_250`` (classic ArUco; pair with
``generate_aruco.py`` defaults). AprilTag 36h11: pass ``aruco_dict=APRILTAG_36H11``
or ``family=tag36h11``. Uses ``cv2.aruco.ArucoDetector`` + ``solvePnP`` —
no pupil-apriltags (avoids SIGSEGV on some ARM/Pi builds).
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from camera import SharedCamera

# Half-edge object points in marker frame (Z=0), corners order matches OpenCV detectMarkers:
# id order: top-left, top-right, bottom-right, bottom-left (clockwise from TL).
_OBJECT_HALF = 0.5


def _aruco_dictionary(name: str):
    """Resolve dictionary name to cv2.aruco predefined dict."""
    key = name.strip().upper().replace("-", "_")
    legacy = {"TAG36H11", "TAG_36H11", "36H11"}
    if key in legacy:
        key = "DICT_APRILTAG_36H11"
    if not key.startswith("DICT_"):
        key = "DICT_" + key
    if not hasattr(cv2.aruco, key):
        raise ValueError(
            f"Unknown ArUco dictionary {name!r} -> {key}; "
            f"see cv2.aruco.DICT_* in OpenCV docs."
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def _detector_params(preset: str):
    p = cv2.aruco.DetectorParameters()
    if preset == "permissive":
        p.minMarkerPerimeterRate = 0.005
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 25
    elif preset == "refine":
        p.minMarkerPerimeterRate = 0.012
        try:
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except AttributeError:
            p.cornerRefinementMethod = 1
    elif preset != "default":
        raise ValueError(f"Unknown detector preset {preset!r}")
    return p


@dataclass
class TagDetection:
    """Result of a single marker detection."""

    tag_id: int
    center: Tuple[float, float]  # pixel coordinates
    corners: np.ndarray  # 4x2 pixel corners

    distance: Optional[float] = None  # metres
    angle_h: Optional[float] = None  # rad, tag to the right of optical axis -> positive
    angle_v: Optional[float] = None
    tvec: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    pose_err: Optional[float] = None  # mean reprojection error (px), if computed


class AprilTagTracker:
    """Continuously detects markers from a SharedCamera (OpenCV ArUco)."""

    def __init__(
        self,
        camera: SharedCamera,
        tag_size: float = 0.045,
        calib_file: Optional[str] = None,
        family: str = "5X5_250",
        target_fps: float = 15.0,
        aruco_dict: Optional[str] = None,
        detector_preset: str = "permissive",
    ):
        self._camera = camera
        self._tag_size = tag_size
        self._target_fps = target_fps
        self._detector_preset = detector_preset

        dict_name = aruco_dict if aruco_dict is not None else family
        self._aruco_dict = _aruco_dictionary(dict_name)
        params = _detector_params(detector_preset)
        try:
            self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, params)
        except AttributeError:
            self._detector = None  # type: ignore
            self._legacy_params = params

        self._camera_params: Optional[Tuple[float, float, float, float]] = None
        self._dist_coeffs: Optional[np.ndarray] = None
        if calib_file and os.path.isfile(calib_file):
            self._load_calibration(calib_file)

        self._detections: List[TagDetection] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_id: int = -1

        half = self._tag_size * _OBJECT_HALF
        self._obj_corners = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    # -- public API ----------------------------------------------------------

    @property
    def detections(self) -> List[TagDetection]:
        with self._lock:
            return list(self._detections)

    @property
    def has_detection(self) -> bool:
        with self._lock:
            return len(self._detections) > 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()
        print("[apriltag] Tracker started (OpenCV ArUco)")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[apriltag] Tracker stopped")

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        with self._lock:
            dets = list(self._detections)
        for d in dets:
            pts = d.corners.astype(int)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
            cx, cy = int(d.center[0]), int(d.center[1])
            label = f"id={d.tag_id}"
            if d.distance is not None:
                label += f" d={d.distance:.2f}m"
            if d.angle_h is not None:
                label += f" a={math.degrees(d.angle_h):.1f}°"
            cv2.putText(
                out,
                label,
                (cx - 40, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return out

    # -- internals -----------------------------------------------------------

    def _load_calibration(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            fx = data["fx"]
            fy = data["fy"]
            cx = data["cx"]
            cy = data["cy"]
            self._camera_params = (fx, fy, cx, cy)
            if "dist_coeffs" in data:
                self._dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
            print(f"[apriltag] Loaded calibration from {path}")
        except Exception as e:
            print(f"[apriltag] Failed to load calibration: {e}")

    def _estimate_camera_params(self, w: int, h: int):
        fx = fy = w * 0.8
        cx, cy = w / 2.0, h / 2.0
        self._camera_params = (fx, fy, cx, cy)
        print(
            f"[apriltag] Using estimated camera params: "
            f"fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}"
        )

    def _camera_matrix(self) -> np.ndarray:
        assert self._camera_params is not None
        fx, fy, cx, cy = self._camera_params
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _detect_markers(self, gray: np.ndarray):
        if self._detector is not None:
            return self._detector.detectMarkers(gray)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self._aruco_dict, parameters=self._legacy_params
        )
        return corners, ids, rejected

    def _pose_for_corners(self, img_corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """img_corners: (4, 2). Returns rvec, tvec, mean reprojection error (px)."""
        K = self._camera_matrix()
        dist = self._dist_coeffs
        if dist is None:
            dist = np.zeros((4, 1), dtype=np.float64)
        img_pts = img_corners.reshape(-1, 1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            self._obj_corners, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                self._obj_corners, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not ok:
            raise RuntimeError("solvePnP failed")

        proj, _ = cv2.projectPoints(self._obj_corners, rvec, tvec, K, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img_corners, axis=1)))
        return rvec.flatten(), tvec.flatten(), err

    def _detect_loop(self):
        interval = 1.0 / self._target_fps
        while self._running:
            t0 = time.monotonic()
            frame, fid = self._camera.get_frame_raw()
            if frame is None or fid == self._last_frame_id:
                time.sleep(0.005)
                continue
            self._last_frame_id = fid

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

            if self._camera_params is None:
                self._estimate_camera_params(w, h)

            corners, ids, _rej = self._detect_markers(gray)
            dets: List[TagDetection] = []

            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    cid = int(ids[i][0])
                    c = corners[i].reshape(4, 2)
                    center = (float(c[:, 0].mean()), float(c[:, 1].mean()))
                    det = TagDetection(tag_id=cid, center=center, corners=c)
                    try:
                        rvec, tvec, perr = self._pose_for_corners(c)
                        det.tvec = tvec
                        det.rvec = rvec
                        det.pose_err = perr
                        det.distance = float(np.linalg.norm(tvec))
                        det.angle_h = float(math.atan2(tvec[0], tvec[2]))
                        det.angle_v = float(math.atan2(tvec[1], tvec[2]))
                    except Exception:
                        pass
                    dets.append(det)

            with self._lock:
                self._detections = dets

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
