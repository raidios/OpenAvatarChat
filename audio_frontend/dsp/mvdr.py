"""
Minimum Variance Distortionless Response beamformer for the 6-mic ring.

Pipeline per frame:
  1. STFT each mic channel.
  2. Estimate noise covariance R_nn(f) on silence frames (gated by an
     external VAD flag passed to ``process()``).
  3. For the current target azimuth, build steering vector d(f).
  4. w(f) = R_nn^{-1}(f) d(f) / (d^H R_nn^{-1} d)
  5. y(f) = w^H X(f), inverse STFT, overlap-add into mono output.

The ``set_doa(az_deg, el_deg=0)`` method updates the steering vector at
runtime; the dynamic-MVDR controller in stage 2 calls this every 50 ms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RingArray, steering_vector


@dataclass
class MvdrConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 160               # 10 ms hop @ 16k for low latency
    initial_az_deg: float = 0.0
    initial_el_deg: float = 0.0
    diag_load: float = 1e-3      # Tikhonov regularisation
    noise_alpha: float = 0.92    # exponential smoothing for R_nn


class MvdrBeamformer:
    def __init__(self, array: RingArray, cfg: MvdrConfig):
        self.array = array
        self.cfg = cfg
        self.window = np.hanning(cfg.n_fft).astype(np.float32)
        self.freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.sample_rate)
        self.M = array.n_mics
        self.F = self.freqs.size
        self.positions = array.positions()
        self._buffer = np.zeros((cfg.n_fft, self.M), dtype=np.float32)
        self._out_buf = np.zeros(cfg.n_fft, dtype=np.float32)
        self._w = np.zeros((self.F, self.M), dtype=np.complex64)
        self._R_nn = (1e-3 * np.eye(self.M)
                      .astype(np.complex64)[None, :, :]
                      .repeat(self.F, axis=0))
        self.set_doa(cfg.initial_az_deg, cfg.initial_el_deg)

    def set_doa(self, az_deg: float, el_deg: float = 0.0) -> None:
        d = steering_vector(self.positions,
                            np.deg2rad(az_deg), np.deg2rad(el_deg),
                            self.freqs)                 # (F, M)
        self._d = d
        self._update_weights()

    def _update_weights(self) -> None:
        # w(f) = R_inv d / (d^H R_inv d)
        eye = self.cfg.diag_load * np.eye(self.M, dtype=np.complex64)[None]
        R = self._R_nn + eye
        R_inv = np.linalg.inv(R)                         # (F, M, M)
        # numerator: R_inv @ d
        num = np.einsum("fmn,fn->fm", R_inv, self._d)    # (F, M)
        # denominator: d^H @ num
        denom = np.einsum("fm,fm->f",
                          np.conjugate(self._d), num) + 1e-9  # (F,)
        self._w = (num / denom[:, None]).astype(np.complex64)

    def update_noise(self, X: np.ndarray) -> None:
        """X: (F, M) complex spectrum of the current frame.

        Adds it to R_nn as a noise sample (caller decides VAD gating).
        """
        outer = np.einsum("fm,fn->fmn", X, np.conjugate(X))   # (F, M, M)
        a = self.cfg.noise_alpha
        self._R_nn = a * self._R_nn + (1 - a) * outer
        self._update_weights()

    def process_frame(self, hop_x: np.ndarray, vad_silence: bool) -> np.ndarray:
        """hop_x: (hop, M) float32. Return (hop,) float32 beamformer output."""
        n_fft, M = self.cfg.n_fft, self.M
        # slide buffer
        self._buffer[:n_fft - self.cfg.hop] = self._buffer[self.cfg.hop:]
        self._buffer[n_fft - self.cfg.hop:] = hop_x
        windowed = self._buffer * self.window[:, None]
        X = np.fft.rfft(windowed, axis=0).astype(np.complex64)   # (F, M)
        if vad_silence:
            self.update_noise(X)
        # y(f) = w^H X
        y = np.einsum("fm,fm->f", np.conjugate(self._w), X)       # (F,)
        time_block = np.fft.irfft(y, n=n_fft).real.astype(np.float32)
        time_block *= self.window
        # overlap-add
        self._out_buf[:n_fft - self.cfg.hop] = (
            self._out_buf[self.cfg.hop:] + time_block[: n_fft - self.cfg.hop]
        )
        self._out_buf[n_fft - self.cfg.hop:] = time_block[n_fft - self.cfg.hop:]
        return self._out_buf[: self.cfg.hop].copy()

    def process(self, frames: np.ndarray, vad: np.ndarray | None = None) -> np.ndarray:
        """frames: (T, M) float32, T multiple of hop.

        vad: (T // hop,) bool array, True = silence (use frame to update R_nn).
        """
        T, M = frames.shape
        assert M == self.M, f"expected {self.M} mics, got {M}"
        hop = self.cfg.hop
        n_chunks = T // hop
        if vad is None:
            vad = np.zeros(n_chunks, dtype=bool)
        out = np.zeros(n_chunks * hop, dtype=np.float32)
        for i in range(n_chunks):
            s = i * hop
            out[s:s + hop] = self.process_frame(frames[s:s + hop], bool(vad[i]))
        return out
