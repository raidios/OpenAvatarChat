"""
SRP-PHAT sound-source-localization for a 6-mic ring.

Implements the classical Steered Response Power with PHAT weighting:

  P(theta) = sum_{m,n} Re{ X_m(f) X_n*(f) / (|X_m(f) X_n*(f)| + eps)
                           * exp(j 2 pi f * (tau_m(theta) - tau_n(theta))) }

evaluated at a coarse azimuth grid. Pure numpy; no ODAS dependency. Good
enough for the 1.5m / 3° static and 6° dynamic targets in the plan KPI.

For elevation we keep a single value (default 0 rad). The plan only commits
to azimuth in the first phase.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RingArray, SOUND_SPEED, steering_vector


@dataclass
class SrpPhatConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 256
    az_grid_deg: int = 5
    f_low: float = 200.0
    f_high: float = 4000.0
    elev_rad: float = 0.0
    smoothing_alpha: float = 0.4   # exponential smoothing on the azimuth track


class SrpPhat:
    """Frame-by-frame SRP-PHAT estimator on a (T, M) float32 input."""

    def __init__(self, array: RingArray, cfg: SrpPhatConfig):
        self.array = array
        self.cfg = cfg
        self.window = np.hanning(cfg.n_fft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.sample_rate)
        self.f_mask = (self.freqs >= cfg.f_low) & (self.freqs <= cfg.f_high)
        self.f_used = self.freqs[self.f_mask]
        self.az_grid_rad = np.deg2rad(np.arange(0, 360, cfg.az_grid_deg))
        # Pre-compute per-mic steering vectors for every grid azimuth.
        positions = array.positions()
        self._d_all = np.stack([
            steering_vector(positions, az, cfg.elev_rad, self.f_used)
            for az in self.az_grid_rad
        ], axis=0)  # (G, F, M)
        self._smoothed_az_rad: float | None = None

    def reset_streaming_smoothing(self) -> None:
        """Clear the exponential azimuth smoother (streaming ``estimate``).

        Call after a known rigid-body rotation of the array (e.g. chassis
        wake-orient) so the next SRP frame is not dragged toward the
        pre-rotation direction; MVDR can then re-lock quickly on the new
        wavefront geometry.
        """
        self._smoothed_az_rad = None

    def _stft(self, x: np.ndarray) -> np.ndarray:
        """x: (T, M) float32 -> (F, M) complex64 (single frame STFT).

        Single-frame: pads / truncates ``x`` to ``cfg.n_fft`` samples and
        returns one (F, M) complex spectrum. Used by the streaming
        ``estimate()`` path which is called once every ``hop`` samples.
        For multi-frame batches use :py:meth:`_stft_batch`.
        """
        T, M = x.shape
        if T < self.cfg.n_fft:
            x = np.pad(x, ((0, self.cfg.n_fft - T), (0, 0)))
        x = x[: self.cfg.n_fft]
        windowed = x * self.window[:, None]
        spec = np.fft.rfft(windowed, axis=0).astype(np.complex64)
        return spec[self.f_mask]   # (F, M)

    def _stft_batch(self, x: np.ndarray) -> np.ndarray:
        """x: (T, M) float32 -> (W, F, M) complex64 stack of W STFT windows.

        Slides a Hann-windowed FFT of size ``n_fft`` with 50 % overlap
        across the time axis. Used by the batch SRP-PHAT path so that
        ``wake_window_doa`` / ``dominant_doa`` actually integrate
        cross-spectra over the entire voiced batch (hundreds of ms),
        not just the first 32 ms which is all ``_stft`` retains.
        """
        T, M = x.shape
        n_fft = self.cfg.n_fft
        hop = max(1, n_fft // 2)
        if T < n_fft:
            return self._stft(x)[np.newaxis, ...]
        W = 1 + (T - n_fft) // hop
        out = np.empty((W, self.f_used.size, M), dtype=np.complex64)
        win = self.window[:, None]
        for w in range(W):
            s = w * hop
            seg = x[s:s + n_fft] * win
            spec = np.fft.rfft(seg, axis=0).astype(np.complex64)
            out[w] = spec[self.f_mask]
        return out

    def estimate(
        self,
        frame: np.ndarray,
        bypass_smoothing: bool = False,
    ) -> tuple[float, float, np.ndarray]:
        """Return (azimuth_deg, peak_power, P_grid).

        ``bypass_smoothing``:
            * False (default) — apply the unit-circle exponential
              smoother (alpha = ``cfg.smoothing_alpha``) and update
              ``self._smoothed_az_rad``. Use this for the streaming
              100 ms-update path that drives MVDR's steering target;
              the smoothing prevents the beam jitter on noise frames.
            * True — return the raw argmax azimuth on this batch
              **without reading or writing** the smoother state. Use
              this for one-shot, large-batch estimators like
              ``DspPipeline.wake_window_doa`` /
              ``DspPipeline.dominant_doa``: they are already
              integrating SRP power over hundreds of ms of voiced
              audio, so smoothing against a streaming history of
              silence/noise frames would only pull the estimate
              away from the true direction. Empirically this fixes
              the "say wake word once → controller rotates only ~⅓
              of the way" failure mode, where mostly-silent buffer
              frames had been dragging the smoother off-target.
        """
        spec = self._stft(frame)                          # (F, M)
        # Auto-power for normalisation
        denom = np.abs(spec) + 1e-9                       # (F, M)
        spec_phat = spec / denom                          # (F, M) PHAT
        # P(theta) = sum_f sum_{m,n} d* . X X* d  (standard SRP-PHAT)
        # For all theta: y(theta, f) = X_phat . d(theta)*
        d_conj = np.conjugate(self._d_all)                 # (G, F, M)
        # einsum: sum over (f, m) of (g, f, m) * (f, m)
        beam = np.einsum("gfm,fm->gf", d_conj, spec_phat)  # (G, F)
        P = np.sum(np.abs(beam) ** 2, axis=1).real         # (G,)
        g = int(np.argmax(P))
        az_rad = float(self.az_grid_rad[g])
        if bypass_smoothing:
            return float(np.degrees(az_rad)), float(P[g]), P
        # exponential smoothing on the unit circle to avoid wrap discontinuities
        if self._smoothed_az_rad is None:
            self._smoothed_az_rad = az_rad
        else:
            a = self.cfg.smoothing_alpha
            ux = a * np.cos(az_rad) + (1 - a) * np.cos(self._smoothed_az_rad)
            uy = a * np.sin(az_rad) + (1 - a) * np.sin(self._smoothed_az_rad)
            self._smoothed_az_rad = float(np.arctan2(uy, ux))
            if self._smoothed_az_rad < 0:
                self._smoothed_az_rad += 2 * np.pi
        return float(np.degrees(self._smoothed_az_rad)), float(P[g]), P

    def estimate_batch(
        self, frames: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Multi-window SRP-PHAT integration over a long batch.

        Splits ``frames`` (T, M) into ``W`` half-overlapping STFT
        windows and **sums** the per-window PHAT-weighted steered
        response power across all windows before taking argmax. This
        is what classical "batch DOA over a 1.5 s utterance" actually
        looks like — every analysis frame casts an equal-weighted vote,
        and the SNR of the integrated response grows linearly with W.

        The single-frame ``estimate(frame)`` retains the streaming
        MVDR-driving behaviour (one frame per call, with smoother).
        ``estimate_batch`` is intentionally smoother-free: it never
        reads or writes ``self._smoothed_az_rad``.

        Returns ``(azimuth_deg, peak_power, P_grid)``. ``peak_power``
        is the integrated peak (≈ W * single-frame peak under
        coherent steering), so divide by ``mean(P_grid)`` if you
        want a confidence ratio comparable to ``estimate``.
        """
        T, _M = frames.shape
        if T < self.cfg.n_fft:
            return self.estimate(frames, bypass_smoothing=True)
        spec_w = self._stft_batch(frames)                  # (W, F, M)
        denom = np.abs(spec_w) + 1e-9
        spec_phat = spec_w / denom                          # (W, F, M)
        d_conj = np.conjugate(self._d_all)                  # (G, F, M)
        # beam (W, G, F) = sum_m d_conj[g, f, m] * spec_phat[w, f, m]
        beam = np.einsum("gfm,wfm->wgf", d_conj, spec_phat)
        P_w = np.sum(np.abs(beam) ** 2, axis=2).real        # (W, G)
        P = P_w.sum(axis=0)                                 # (G,)
        g = int(np.argmax(P))
        az_rad = float(self.az_grid_rad[g])
        return float(np.degrees(az_rad)), float(P[g]), P
