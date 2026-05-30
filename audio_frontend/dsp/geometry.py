"""Mic-array geometry helpers.

The M260C carries six omnidirectional MEMS arranged on a 33-mm-radius ring
(measured with calipers; replace with the value yaml if it differs). Mic 0
points towards the positive X axis (control-board front); subsequent mics are
placed at 60° increments counter-clockwise looking down on the array.

All angles are returned in radians; azimuths increase counter-clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOUND_SPEED = 343.0  # m/s at 20 °C dry air


@dataclass
class RingArray:
    """Symmetric ring of N omnidirectional mics on the XY plane.

    Attributes:
      n_mics      number of microphones
      radius_m    ring radius in meters
      ref_axis    azimuth of mic 0, in radians (default 0 = +X)
    """

    n_mics: int = 6
    radius_m: float = 0.033
    ref_axis: float = 0.0

    def positions(self) -> np.ndarray:
        """Return (n_mics, 3) Cartesian positions in the array frame."""
        thetas = self.ref_axis + 2.0 * np.pi * np.arange(self.n_mics) / self.n_mics
        x = self.radius_m * np.cos(thetas)
        y = self.radius_m * np.sin(thetas)
        z = np.zeros_like(x)
        return np.stack([x, y, z], axis=-1)

    def azimuths_deg(self) -> np.ndarray:
        return np.degrees(self.ref_axis
                          + 2.0 * np.pi * np.arange(self.n_mics) / self.n_mics)


def steering_vector(positions: np.ndarray, doa_az_rad: float,
                    doa_el_rad: float, freq_hz: np.ndarray,
                    sound_speed: float = SOUND_SPEED) -> np.ndarray:
    """Compute the far-field steering vector matrix.

    Args:
      positions   (M, 3) mic positions in meters
      doa_az_rad  scalar azimuth in radians
      doa_el_rad  scalar elevation in radians
      freq_hz     (F,) frequency bins in Hz

    Returns:
      (F, M) complex64 steering vectors d(f, m).
    """
    # unit vector pointing FROM source TO array (incident direction)
    ux = np.cos(doa_el_rad) * np.cos(doa_az_rad)
    uy = np.cos(doa_el_rad) * np.sin(doa_az_rad)
    uz = np.sin(doa_el_rad)
    direction = np.array([ux, uy, uz])
    # tau[m] = -direction . pos[m] / c    (positive when mic lies on the side
    # facing the source, i.e. earlier arrival)
    tau = -(positions @ direction) / sound_speed   # (M,)
    # steering vector exp(-j 2π f tau) (positive tau == earlier arrival ->
    # phase leads; standard MVDR convention)
    f = freq_hz.reshape(-1, 1)                      # (F, 1)
    t = tau.reshape(1, -1)                          # (1, M)
    d = np.exp(-1j * 2.0 * np.pi * f * t).astype(np.complex64)
    return d
