"""
相机内参标定的可复用核心。

把原 ``client/camera_calibration.py`` 里的 ``cmd_calibrate`` 算法切成"喂帧 +
状态机 + 收口"三个清晰阶段，让 CLI 与 teach server 都能复用同一份实现。

使用方式：

    session = CalibSession(rows=9, cols=6, square_mm=25.0, min_frames=15,
                           auto_interval_s=1.0, output_path=Path("config/camera_calib.json"))
    session.start()
    while not session.is_finalized():
        frame = camera.get_frame_raw()[0]
        if frame is not None:
            snap = session.feed_frame(frame)
            ui.update(snap)
        if snap.captured >= snap.min_frames:
            break
    result = session.finalize()
    print(result.rms_error)

线程模型：``CalibSession`` 内部用 ``threading.Lock`` 保护可变状态；``feed_frame``
可由后台线程频繁调用，REST 接口（``start/force_capture/finalize/abort/snapshot``）
可由 asyncio 协程调用。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class CalibStateSnapshot:
    state: str             # "idle" | "running" | "finalized" | "aborted" | "error"
    rows: int
    cols: int
    inner_w: int
    inner_h: int
    square_mm: float
    min_frames: int
    captured: int
    last_corner_count: int  # 上一帧检测到的内角点数（== inner_w * inner_h 时表示完整识别）
    last_corner_at: float   # time.monotonic 戳；0 表示从未检测到
    auto_cooldown_left: float
    auto_interval_s: float
    output_path: str
    error_msg: Optional[str] = None
    last_capture_at: float = 0.0


@dataclass
class CalibResult:
    rms_error: float
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: List[float]
    resolution: List[int]
    frame_count: int
    saved_to: str


class CalibSession:
    def __init__(
        self,
        rows: int,
        cols: int,
        square_mm: float,
        min_frames: int = 15,
        auto_interval_s: float = 1.0,
        output_path: Path | str = "config/camera_calib.json",
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.square_mm = float(square_mm)
        self.min_frames = int(min_frames)
        self.auto_interval_s = float(auto_interval_s)
        self.output_path = Path(output_path)

        self.inner_w = self.cols - 1
        self.inner_h = self.rows - 1
        square_m = self.square_mm / 1000.0
        objp = np.zeros((self.inner_h * self.inner_w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.inner_w, 0:self.inner_h].T.reshape(-1, 2) * square_m
        self._objp_template = objp

        self._lock = threading.Lock()
        self._state = "idle"
        self._obj_points: List[np.ndarray] = []
        self._img_points: List[np.ndarray] = []
        self._last_corners: Optional[np.ndarray] = None
        self._last_corner_count = 0
        self._last_corner_at = 0.0
        self._last_capture_at = 0.0
        self._frame_size: Optional[tuple] = None
        self._result: Optional[CalibResult] = None
        self._error_msg: Optional[str] = None

    # ------------------------------------------------------------------
    # external API
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self._state = "running"
            self._obj_points = []
            self._img_points = []
            self._last_corners = None
            self._last_corner_count = 0
            self._last_corner_at = 0.0
            self._last_capture_at = 0.0
            self._frame_size = None
            self._result = None
            self._error_msg = None

    def abort(self) -> None:
        with self._lock:
            if self._state == "running":
                self._state = "aborted"
            self._last_corners = None

    def is_finalized(self) -> bool:
        with self._lock:
            return self._state in ("finalized", "aborted", "error")

    def feed_frame(self, frame_bgr: np.ndarray) -> CalibStateSnapshot:
        with self._lock:
            running = self._state == "running"
        if not running:
            return self.snapshot()
        if frame_bgr is None:
            return self.snapshot()

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (self.inner_w, self.inner_h), None,
        )
        now = time.monotonic()
        with self._lock:
            self._frame_size = (gray.shape[1], gray.shape[0])
            if found and corners is not None:
                self._last_corners = corners
                self._last_corner_count = int(self.inner_w * self.inner_h)
                self._last_corner_at = now
            else:
                self._last_corners = None
                self._last_corner_count = 0

            # auto-capture：检测到角点 + 距离上次入栈足够久
            if (
                found and corners is not None
                and (self._last_capture_at == 0.0 or now - self._last_capture_at >= self.auto_interval_s)
            ):
                self._enqueue_corners_locked(gray, corners, now)
        return self.snapshot()

    def force_capture(self) -> bool:
        """在当前已检测到角点的情况下立刻把这一帧入栈；返回是否成功。

        典型用例：浏览器侧用户点 "Capture" 按钮，希望忽略 auto_interval 强制采样。
        需要事先有过一次 ``feed_frame`` 检测出角点。
        """
        with self._lock:
            if self._state != "running":
                return False
            corners = self._last_corners
            if corners is None:
                return False
            # 我们没有 grey 缓存，corners 已是 cornerSubPix 之前的角点；这里
                # 没有 grey 重新精修；和原 CLI 行为略有差异——auto-capture 走 cornerSubPix，
                # force_capture 写入原始角点也能用，但精度略低。简化设计先这么处理。
            self._obj_points.append(self._objp_template.copy())
            self._img_points.append(corners)
            self._last_capture_at = time.monotonic()
            return True

    def feed_and_subpix(self, frame_bgr: np.ndarray) -> bool:
        """force_capture 的高质量变体：现场跑 cornerSubPix 精修后入栈。"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (self.inner_w, self.inner_h), None)
        if not found or corners is None:
            return False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        with self._lock:
            if self._state != "running":
                return False
            self._frame_size = (gray.shape[1], gray.shape[0])
            self._obj_points.append(self._objp_template.copy())
            self._img_points.append(corners_refined)
            self._last_corners = corners_refined
            self._last_corner_count = int(self.inner_w * self.inner_h)
            self._last_corner_at = time.monotonic()
            self._last_capture_at = self._last_corner_at
        return True

    def finalize(self) -> Optional[CalibResult]:
        with self._lock:
            if self._state != "running":
                return self._result
            if len(self._obj_points) < 3 or self._frame_size is None:
                self._state = "error"
                self._error_msg = f"not enough frames: {len(self._obj_points)}"
                return None
            obj_points = list(self._obj_points)
            img_points = list(self._img_points)
            w, h = self._frame_size

        try:
            ret, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
                obj_points, img_points, (w, h), None, None,
            )
        except cv2.error as e:
            with self._lock:
                self._state = "error"
                self._error_msg = f"calibrateCamera failed: {e}"
            return None

        result = CalibResult(
            rms_error=float(ret),
            fx=float(mtx[0, 0]),
            fy=float(mtx[1, 1]),
            cx=float(mtx[0, 2]),
            cy=float(mtx[1, 2]),
            dist_coeffs=[float(x) for x in np.array(dist).flatten().tolist()],
            resolution=[int(w), int(h)],
            frame_count=len(obj_points),
            saved_to=str(self.output_path),
        )

        payload = {
            "fx": result.fx,
            "fy": result.fy,
            "cx": result.cx,
            "cy": result.cy,
            "dist_coeffs": result.dist_coeffs,
            "resolution": result.resolution,
            "rms_error": result.rms_error,
        }
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.output_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.output_path)
        except OSError as e:
            with self._lock:
                self._state = "error"
                self._error_msg = f"save failed: {e}"
            return result

        with self._lock:
            self._state = "finalized"
            self._result = result
        return result

    def snapshot(self) -> CalibStateSnapshot:
        now = time.monotonic()
        with self._lock:
            cd_left = max(0.0, self.auto_interval_s - (now - self._last_capture_at)) if self._last_capture_at else 0.0
            return CalibStateSnapshot(
                state=self._state,
                rows=self.rows,
                cols=self.cols,
                inner_w=self.inner_w,
                inner_h=self.inner_h,
                square_mm=self.square_mm,
                min_frames=self.min_frames,
                captured=len(self._obj_points),
                last_corner_count=self._last_corner_count,
                last_corner_at=self._last_corner_at,
                auto_cooldown_left=cd_left,
                auto_interval_s=self.auto_interval_s,
                output_path=str(self.output_path),
                error_msg=self._error_msg,
                last_capture_at=self._last_capture_at,
            )

    @property
    def captured(self) -> int:
        with self._lock:
            return len(self._obj_points)

    # ------------------------------------------------------------------
    # locked helpers
    # ------------------------------------------------------------------

    def _enqueue_corners_locked(self, gray: np.ndarray, corners: np.ndarray, now: float) -> None:
        """已在 self._lock 内部调用：精修角点后入栈。"""
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # cornerSubPix 在 lock 内调用：单次 ~几 ms，不会阻塞太久。
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        self._obj_points.append(self._objp_template.copy())
        self._img_points.append(corners_refined)
        self._last_capture_at = now
