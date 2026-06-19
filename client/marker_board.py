"""Marker-board target estimation for multi-tag tracking.

The physical board used by the robot is a 3x3 grid of marker IDs 0..8:

    0 1 2
    3 4 5
    6 7 8

Each marker is still detected independently by OpenCV. This module turns
those independent detections into one board-center target so the tracking
controller does not jump between competing marker IDs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class MarkerBoardLayout:
    tag_size_m: float = 0.045
    gap_m: float = 0.007
    rows: int = 3
    cols: int = 3
    first_id: int = 0

    @property
    def pitch_m(self) -> float:
        return self.tag_size_m + self.gap_m

    def offset_for_id(self, tag_id: int) -> Optional[tuple[float, float]]:
        idx = tag_id - self.first_id
        if idx < 0 or idx >= self.rows * self.cols:
            return None
        row = idx // self.cols
        col = idx % self.cols
        x = (col - (self.cols - 1) / 2.0) * self.pitch_m
        y = (row - (self.rows - 1) / 2.0) * self.pitch_m
        return (x, y)


@dataclass
class BoardTarget:
    tag_id: int
    distance: float
    angle_h: float
    angle_v: float
    tvec: np.ndarray
    visible_count: int
    held: bool = False


class MarkerBoardEstimator:
    def __init__(
        self,
        layout: MarkerBoardLayout,
        smoothing_alpha: float = 0.35,
        lost_timeout_s: float = 0.35,
    ):
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self._layout = layout
        self._alpha = smoothing_alpha
        self._lost_timeout_s = lost_timeout_s
        self._last_target: Optional[BoardTarget] = None
        self._last_visible_s: Optional[float] = None

    def estimate(self, detections: Iterable[object], now_s: Optional[float] = None) -> Optional[BoardTarget]:
        now = time.monotonic() if now_s is None else now_s
        centers: list[np.ndarray] = []

        for det in detections:
            tag_id = getattr(det, "tag_id", None)
            tvec = getattr(det, "tvec", None)
            if tag_id is None or tvec is None:
                continue
            offset = self._layout.offset_for_id(int(tag_id))
            if offset is None:
                continue
            vec = np.asarray(tvec, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(vec)):
                continue
            if vec[2] <= 0.0:
                continue
            # Board X is aligned with camera X for the near-frontal tracking
            # case. Vertical offset is included for distance/angle_v stability.
            tag_x, tag_y = offset
            centers.append(vec - np.array([tag_x, tag_y, 0.0], dtype=np.float64))

        if not centers:
            if (
                self._last_target is not None
                and self._last_visible_s is not None
                and now - self._last_visible_s <= self._lost_timeout_s
            ):
                return BoardTarget(
                    tag_id=self._last_target.tag_id,
                    distance=self._last_target.distance,
                    angle_h=self._last_target.angle_h,
                    angle_v=self._last_target.angle_v,
                    tvec=self._last_target.tvec.copy(),
                    visible_count=0,
                    held=True,
                )
            return None

        raw = np.median(np.vstack(centers), axis=0)
        if self._last_target is not None:
            fused = self._alpha * raw + (1.0 - self._alpha) * self._last_target.tvec
        else:
            fused = raw
        if not np.all(np.isfinite(fused)) or fused[2] <= 0.0:
            return None

        distance = float(np.linalg.norm(fused))
        angle_h = float(math.atan2(fused[0], fused[2]))
        angle_v = float(math.atan2(fused[1], fused[2]))
        target = BoardTarget(
            tag_id=-1,
            distance=distance,
            angle_h=angle_h,
            angle_v=angle_v,
            tvec=fused,
            visible_count=len(centers),
        )
        self._last_target = target
        self._last_visible_s = now
        return target
