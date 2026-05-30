"""
Static rigid transform from camera optical frame -> mic-array frame.

Conventions (right-handed, OpenCV):
  Camera optical: +x right, +y down, +z forward
  Mic array:      +x channel-0 direction, +y 90° CCW (top-view), +z up
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class MicCameraExtrinsic:
    R: np.ndarray   # (3,3) camera -> mic (DCM)
    T: np.ndarray   # (3,)  camera -> mic, meters

    @classmethod
    def identity(cls) -> "MicCameraExtrinsic":
        return cls(R=np.eye(3, dtype=np.float64),
                   T=np.zeros(3, dtype=np.float64))

    @classmethod
    def load(cls, path: str | Path) -> "MicCameraExtrinsic":
        data = np.load(str(path))
        return cls(
            R=np.asarray(data["R"], dtype=np.float64).reshape(3, 3),
            T=np.asarray(data["T"], dtype=np.float64).reshape(3),
        )

    def transform(self, p_cam_m: np.ndarray) -> np.ndarray:
        return self.R @ p_cam_m + self.T

    def to_az_el(self, p_cam_m: np.ndarray) -> Tuple[float, float, float]:
        """Return (azimuth_deg [-180..180), elevation_deg, distance_m) in
        the mic-array frame for a 3-D point given in the camera frame.
        Azimuth is 0° at +x (channel-0 direction) and increases CCW.
        """
        p = self.transform(np.asarray(p_cam_m, dtype=np.float64).reshape(3))
        x, y, z = p[0], p[1], p[2]
        d = float(np.sqrt(x * x + y * y + z * z))
        if d < 1e-6:
            return 0.0, 0.0, 0.0
        az = float(np.degrees(np.arctan2(y, x)))
        el = float(np.degrees(np.arcsin(z / d)))
        return az, el, d


def load_or_default(path: Optional[str | Path]) -> MicCameraExtrinsic:
    """Best-effort: load if file exists, else return identity. Logs when
    falling back so callers know the steering direction will be camera-frame
    azimuth, not mic-frame."""
    if path is None:
        return MicCameraExtrinsic.identity()
    p = Path(path)
    if p.exists():
        try:
            return MicCameraExtrinsic.load(p)
        except Exception:
            return MicCameraExtrinsic.identity()
    return MicCameraExtrinsic.identity()
