"""
Single-instance acoustic echo canceller for the M260C far-field DSP chain.

Three backends:
  - SpeexAec   wraps speexdsp.EchoCanceller (MDF). Production default —
                on real voice playback through the M260C line-level ref
                (cherry.wav probe) it gives +13.8 dB ERLE on raw ch0
                and +19.2 dB when run *per-mic before MVDR*. The
                ``slapped (reset)`` warnings we hit earlier were
                triggered by the synthetic chirp probe (a known LMS
                pathological signal), not by the line-level ref.
  - NlmsAec    pure-numpy NLMS, kept as fallback / sanity reference.
                Convenient because it has no native deps; performance
                is signal-dependent (~+13 dB on chirp, only ~+5 dB on
                voice). Use only when Speex is unavailable.
  - FdafAec    pure-numpy frequency-domain adaptive filter (block FLMS,
                constrained gradient, scalar power normalisation,
                step + IR clipping). Hits +58 dB on synthetic
                white-noise + LTI IR but does not converge on real
                voice or chirp refs through the M260C; experimental,
                left in for offline study.

All three expose the same ``process_frame(near, far) -> near_clean``
contract for 10 ms float32 frames at 16 kHz.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SIZE = 160          # 10 ms @ 16 kHz
# ref(ch6/7) is line-level (speaker electrical signal direct-wired).
# For Speex MDF the filter length should comfortably exceed the room
# tail. Voice probe (tools/probe_aec_isolation.py --stimulus cherry.wav)
# hits +13.8 dB on raw ch0 and +19.2 dB per-mic-before-MVDR with the
# Speex default of 200 ms; that's the operating point.
FILTER_LEN_MS = 200

try:
    from speexdsp import EchoCanceller as _SpeexEchoCanceller   # type: ignore
    _HAVE_SPEEX = True
except Exception as exc:  # noqa: BLE001
    _HAVE_SPEEX = False
    _SPEEX_IMPORT_ERROR = exc
    logger.debug("speexdsp not importable; using NLMS fallback")


@dataclass
class AecConfig:
    sample_rate: int = 16000
    frame_size: int = FRAME_SIZE
    filter_length_samples: int = int(FILTER_LEN_MS * 16)   # 3200 = 200ms@16k
    backend: str = "auto"        # auto | speex | fdaf | nlms


class _AecBase:
    name: str
    sample_rate: int
    frame_size: int

    def reset(self) -> None: ...
    def process_frame(self, near: np.ndarray, far: np.ndarray) -> np.ndarray: ...


class SpeexAec(_AecBase):
    """Wrapper around speexdsp.EchoCanceller. Float32 in [-1, 1]."""

    def __init__(self, cfg: AecConfig):
        if not _HAVE_SPEEX:
            raise RuntimeError(f"speexdsp not available: {_SPEEX_IMPORT_ERROR!r}")
        self.cfg = cfg
        self.name = "speexdsp"
        self.sample_rate = cfg.sample_rate
        self.frame_size = cfg.frame_size
        self._ec = _SpeexEchoCanceller.create(
            cfg.frame_size, cfg.filter_length_samples, cfg.sample_rate)

    def reset(self) -> None:
        self._ec = _SpeexEchoCanceller.create(
            self.cfg.frame_size, self.cfg.filter_length_samples,
            self.cfg.sample_rate)

    def process_frame(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        n_i16 = (np.clip(near, -1, 1) * 32767).astype(np.int16).tobytes()
        f_i16 = (np.clip(far, -1, 1) * 32767).astype(np.int16).tobytes()
        out_bytes = self._ec.process(n_i16, f_i16)
        out = np.frombuffer(out_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return out


class NlmsAec(_AecBase):
    """Block-NLMS adaptive filter. CPU only, no external deps.

    The filter taps are updated once per 10ms block on the average gradient.
    Convergence is slower than Speex but ERLE typically reaches 15-20 dB on
    linear echo paths, plenty to verify wiring.
    """

    def __init__(self, cfg: AecConfig):
        self.cfg = cfg
        self.name = "nlms"
        self.sample_rate = cfg.sample_rate
        self.frame_size = cfg.frame_size
        self.taps = np.zeros(cfg.filter_length_samples, dtype=np.float32)
        self._far_buf = np.zeros(
            cfg.filter_length_samples + cfg.frame_size, dtype=np.float32)
        # Step size empirically chosen via 2-D sweep (filter length × mu)
        # against the M260C line-level ref. mu=0.2 is the sweet spot on
        # 20 ms filters; mu=0.5 (the textbook default) overshoots and
        # leaves residual at -13 dB instead of -13.4. See aec.py header.
        self.mu = 0.2
        self.eps = 1e-6

    def reset(self) -> None:
        self.taps[:] = 0.0
        self._far_buf[:] = 0.0

    def process_frame(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        L = self.cfg.filter_length_samples
        N = self.cfg.frame_size
        # slide far buffer
        self._far_buf[:-N] = self._far_buf[N:]
        self._far_buf[-N:] = far
        out = np.zeros_like(near)
        for n in range(N):
            x = self._far_buf[L + n - L: L + n][::-1]  # most recent L samples
            y_hat = float(np.dot(self.taps, x))
            e = float(near[n] - y_hat)
            norm = float(np.dot(x, x)) + self.eps
            self.taps += (self.mu * e / norm) * x
            out[n] = e
        return out


class FdafAec(_AecBase):
    """Frequency-domain adaptive filter (constrained block FLMS).

    Single-block, overlap-save layout. Block size B equals the filter
    length L; the FFT runs at 2L. Per block we do 4 rfft/irfft calls
    (~12k flops at L=320), versus NLMS's O(L*N) ≈ 51k flops/sample at
    the same length. ERLE on stationary inputs is typically within
    a couple of dB of the Wiener oracle.

    Latency: caller frames are buffered until B samples are accumulated;
    the very first ``process_frame`` returns zeros until the internal
    block fills (one-block startup delay = B samples = 20 ms by default).
    """

    def __init__(self, cfg: AecConfig):
        self.cfg = cfg
        self.name = "fdaf"
        self.sample_rate = cfg.sample_rate
        self.frame_size = cfg.frame_size

        L = max(cfg.frame_size, cfg.filter_length_samples)
        self.B = int(L)
        self.N = 2 * self.B
        self.W = np.zeros(self.N // 2 + 1, dtype=np.complex128)
        self.X_old = np.zeros(self.B, dtype=np.float32)
        # Scalar mean-power normaliser. Per-bin variants are unstable on
        # bursty / non-stationary refs (chirp, abrupt onsets) because
        # bins with near-zero power blow up; constraint adds cross-bin
        # leakage that makes the issue worse. Mean power is robust.
        self.power: float = 0.0
        # Track number of blocks processed so we can warm up the power
        # estimate before we start adapting (prevents the very first
        # block from injecting a huge step against a tiny init power).
        self._n_blocks = 0
        self._warmup_blocks = 4

        self.mu = 0.5
        self.eps = 1e-3
        self.power_alpha = 0.10
        # Per-block step magnitude clip in time domain (max |delta_w|).
        self.step_clip = 0.05
        # Hard cap on the time-domain IR magnitude. For a stable LTI
        # echo path the IR samples are typically well below 1; capping
        # at 2.0 prevents pathological non-LTI inputs (chirp with
        # near-clipping mic) from pushing the filter into a degenerate
        # divergent state while still allowing reasonable IRs.
        self.w_clip = 2.0

        self._near_buf = np.zeros(0, dtype=np.float32)
        self._far_buf = np.zeros(0, dtype=np.float32)
        self._out_buf = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self.W[:] = 0.0
        self.X_old[:] = 0.0
        self.power = 0.0
        self._n_blocks = 0
        self._near_buf = np.zeros(0, dtype=np.float32)
        self._far_buf = np.zeros(0, dtype=np.float32)
        self._out_buf = np.zeros(0, dtype=np.float32)

    def process_frame(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        if near.dtype != np.float32:
            near = near.astype(np.float32, copy=False)
        if far.dtype != np.float32:
            far = far.astype(np.float32, copy=False)
        self._near_buf = np.concatenate([self._near_buf, near])
        self._far_buf = np.concatenate([self._far_buf, far])

        while len(self._near_buf) >= self.B:
            n_blk = self._near_buf[: self.B]
            f_blk = self._far_buf[: self.B]
            self._near_buf = self._near_buf[self.B:].copy()
            self._far_buf = self._far_buf[self.B:].copy()

            # Overlap-save linear convolution. far_2L[0..B-1] = previous
            # block (X_old), far_2L[B..2B-1] = current block. Linear
            # convolution result is the LAST B samples of IFFT(W * F).
            far_2L = np.concatenate([self.X_old, f_blk])
            F = np.fft.rfft(far_2L, n=self.N)

            Y = self.W * F
            y_full = np.fft.irfft(Y, n=self.N)
            y = y_full[self.B:].astype(np.float32)
            e = n_blk - y
            self._out_buf = np.concatenate([self._out_buf, e])

            # Gradient: linear correlation of e (length B, *current*
            # samples) with far_2L. e_padded must put e in the SECOND
            # half so its alignment with far_2L matches that of y. Then
            # the first B samples of IFFT(conj(F)*E_pad) are the causal
            # gradient taps (lag 0..B-1); the trailing B are wrap-around
            # garbage and get zeroed by the constraint.
            e_padded = np.concatenate(
                [np.zeros(self.B, dtype=np.float32), e])
            E = np.fft.rfft(e_padded, n=self.N)
            phi = np.conj(F) * E
            phi_t = np.fft.irfft(phi, n=self.N)
            phi_t[self.B:] = 0.0
            phi_c = np.fft.rfft(phi_t, n=self.N)

            # Update scalar mean power and weights. The first iteration
            # snaps power to the actual signal level instead of the
            # tiny init value; subsequent iterations smooth.
            inst_p = float(np.mean(np.abs(F) ** 2))
            if self.power == 0.0:
                self.power = inst_p
            else:
                self.power = ((1 - self.power_alpha) * self.power
                              + self.power_alpha * inst_p)
            self._n_blocks += 1

            if self._n_blocks > self._warmup_blocks:
                step_freq = self.mu * phi_c / (self.power + self.eps)
                step_t = np.fft.irfft(step_freq, n=self.N)
                step_t[self.B:] = 0.0
                np.clip(step_t, -self.step_clip, self.step_clip,
                        out=step_t)
                self.W += np.fft.rfft(step_t, n=self.N)

                # Cap the time-domain IR magnitude to prevent runaway.
                w_t = np.fft.irfft(self.W, n=self.N)
                w_t[self.B:] = 0.0
                if np.max(np.abs(w_t)) > self.w_clip:
                    np.clip(w_t, -self.w_clip, self.w_clip, out=w_t)
                self.W = np.fft.rfft(w_t, n=self.N)

            self.X_old = f_blk.copy()

        n_out = len(near)
        if len(self._out_buf) >= n_out:
            out = self._out_buf[:n_out].copy()
            self._out_buf = self._out_buf[n_out:].copy()
            return out.astype(np.float32, copy=False)
        # Not enough cleaned samples yet — emit silence for the start-up
        # block. After the first B samples accumulate this branch never
        # fires again.
        return np.zeros(n_out, dtype=np.float32)


def make_aec(cfg: Optional[AecConfig] = None) -> _AecBase:
    cfg = cfg or AecConfig()
    backend = (cfg.backend or "auto").lower()
    if backend == "speex":
        return SpeexAec(cfg)
    if backend == "fdaf":
        return FdafAec(cfg)
    if backend == "nlms":
        return NlmsAec(cfg)
    if backend != "auto":
        raise ValueError(f"unknown AEC backend {backend!r}")
    # ``auto`` defaults to Speex MDF when available: voice-ref probe on
    # the M260C line-level reference puts it at +13.8 dB ERLE on raw
    # ch0 and +19.2 dB per-mic-before-MVDR. NLMS only manages +5 dB on
    # voice (chirp numbers are misleading because chirp is not
    # representative production content). Falls back to NLMS only if
    # the speexdsp native lib is missing.
    if _HAVE_SPEEX:
        return SpeexAec(cfg)
    logger.warning("speexdsp unavailable, falling back to NLMS AEC")
    return NlmsAec(cfg)
