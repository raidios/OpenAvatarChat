"""
End-to-end DSP pipeline: 8ch int16 in -> mono float32 clean out.

  in  : (T, 8) int16  (ch0..ch5 = mics, ch6, ch7 = ref)
  out : (T,)   float32 in [-1, 1]   16 kHz

Default order (aec_per_mic = True, the production path):
  1. split mics / ref
  2. AEC × n_mics:    one Speex MDF instance per mic, all sharing the
                      same line-level ref(mean ch6/7). Echo cancellation
                      runs on a *time-invariant* echo path here, before
                      MVDR weights start moving with the source. The
                      voice-ref isolation probe shows this gives +19 dB
                      ERLE versus +4 dB for the legacy MVDR-then-AEC
                      order.
  3. SRP-PHAT update DOA every cfg.doa_update_ms (default 100 ms)
  4. MVDR -> mono beam at current DOA, fed with echo-cancelled mics
  5. DNS: process(beam) -> clean

Legacy order (aec_per_mic = False): MVDR -> single-instance AEC -> DNS.
Kept for ablation; not recommended in production.

All blocks operate on the same hop = 160 (10ms @ 16k) for latency
consistency except the DNS which uses its own internal block_size and
re-buffers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# M260C channel layout -------------------------------------------------------
# The M260C is a 6-mic ring of radius 35 mm (manufacturer spec). The raw
# 8-ch stream is "hacked" out of the firmware so the SDK channel order
# does not follow the manufacturer mic numbering, and the physical
# handedness around the ring depends on which face of the PCB is up.
#
# Two unknowns to calibrate per install:
#   * mic_channel_perm: SDK ch idx 0..5 -> CCW-around-the-ring slot
#     assumed by RingArray.
#   * mic_yaw_offset_deg: where logical mic 0 (post-perm) sits in body
#     frame, used by FarfieldAudioSource to convert mic_az -> body_az.
#
# Calibration recipe (run after every remount of the array):
#   1. tap test (`make mic-tap-calibrate`) determines the perm
#      directly from physical taps -- this is the source of truth.
#   2. clean DOA recording at 4 cardinal angles
#      (`make doa-record-cardinal`) + brute-force fit
#      (`make doa-calibrate`) cross-checks the perm and pins the yaw
#      offset. The two should agree on the perm; if not, the dataset
#      was contaminated by reflections or a ~180° hemisphere flip.
#
# Historical data point #1: with the array mounted MIC-DOWN (PCB
# facing up), brute-force fits converged on a CW perm and a yaw offset
# ~180° away from the physical tap-test value. This is consistent with
# the PCB substrate + robot chassis acting as an acoustic shield over
# the mics, suppressing the direct path so floor/wall reflections
# dominate the SRP-PHAT peak. Mount the array MIC-UP (capsules facing
# the open hemisphere) to recover.
#
# Historical data point #2: post-flip the perm comes out as
# (4, 3, 5, 0, 1, 2), which is NOT a clean ring rotation -- channels
# 3 and 5 are swapped relative to the obvious "CW from ch4" pattern
# (4, 3, 2, 1, 0, 5). The array is still physically a regular hexagon
# (per the M260C datasheet); the SDK channel ordering is just
# scrambled by the firmware "hack". Tap test is the source of truth
# here; the +90° clip in the cleaned dataset validates this perm to
# within ~16° (mic +44° measured vs +60° expected at yaw=+30°).
#
# Re-run the calibration recipe above any time the array is remounted,
# repositioned, or the chassis above the ring changes.
M260C_MIC_PERM: Tuple[int, ...] = (4, 3, 5, 0, 1, 2)
M260C_RING_RADIUS_M: float = 0.035
# Body-frame yaw of logical mic 0 after applying M260C_MIC_PERM. Tap
# test confirms raw ch4 sits at body +30° (M260C "0°" mark on the
# current install with mic-up mount).
M260C_YAW_OFFSET_DEG: float = 30.0

from ..backends.denoiser import select_denoiser, Denoiser
from .aec import AecConfig, make_aec
from .geometry import RingArray
from .mvdr import MvdrBeamformer, MvdrConfig
from .srp_phat import SrpPhat, SrpPhatConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    hop_samples: int = 160              # MVDR/AEC hop = 10 ms @ 16 k
    dns_block_samples: int = 480        # DNS block; DfNet wants 30 ms minimum
    n_mics: int = 6
    n_ref: int = 2
    array_radius_m: float = M260C_RING_RADIUS_M  # M260C ring radius
    mic_channel_perm: Optional[Tuple[int, ...]] = M260C_MIC_PERM
    # if set, applied to incoming mic channels before any DSP. Default
    # is the M260C calibrated permutation (see module-level comment).
    # Set to None for an already-correctly-ordered ring.
    doa_update_ms: int = 100
    use_doa_for_mvdr: bool = True
    static_az_deg: float = 0.0          # used when use_doa_for_mvdr=False
    aec_backend: str = "auto"           # auto | speex | nlms | fdaf
    aec_filter_length_ms: int = 200     # AEC adaptive filter tail length.
                                        # Speex MDF likes the filter to
                                        # comfortably exceed the room
                                        # tail; 200 ms is the sweet
                                        # spot from the cherry probe.
    aec_per_mic: bool = True            # if True, run one AEC per mic
                                        # *before* MVDR (best ERLE).
                                        # If False, run a single AEC
                                        # *after* MVDR (legacy).
    wake_window_ms: int = 1500          # how much AEC-cleaned mic
                                        # history to retain for the
                                        # wake-event DOA estimate.
    dns_backend: str = "auto"           # auto | cpu | hailo
    dns_use_passthrough: bool = False   # if True, skip even CPU DfDenoiser
    silence_rms_threshold: float = 0.005  # below this -> treat as silence (R_nn update)


class DspPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.array = RingArray(n_mics=self.cfg.n_mics,
                               radius_m=self.cfg.array_radius_m)

        srp_cfg = SrpPhatConfig(sample_rate=self.cfg.sample_rate)
        self.srp = SrpPhat(self.array, srp_cfg)

        mvdr_cfg = MvdrConfig(
            sample_rate=self.cfg.sample_rate,
            hop=self.cfg.hop_samples,
            initial_az_deg=self.cfg.static_az_deg,
        )
        self.mvdr = MvdrBeamformer(self.array, mvdr_cfg)

        aec_cfg = AecConfig(
            sample_rate=self.cfg.sample_rate,
            backend=self.cfg.aec_backend,
            filter_length_samples=int(
                self.cfg.aec_filter_length_ms * self.cfg.sample_rate / 1000),
        )
        if self.cfg.aec_per_mic:
            self.aec_pool = [make_aec(aec_cfg)
                             for _ in range(self.cfg.n_mics)]
            self.aec = None
        else:
            self.aec_pool = None
            self.aec = make_aec(aec_cfg)

        # DNS
        if self.cfg.dns_use_passthrough:
            from ..backends.denoiser import CpuDenoiser
            self.denoiser: Denoiser = CpuDenoiser(
                block_size=self.cfg.dns_block_samples, model="passthrough")
        else:
            self.denoiser = select_denoiser(prefer=self.cfg.dns_backend,
                                            block_size=self.cfg.dns_block_samples)
        # buffer between AEC and DNS so we never call DNS with a sub-block
        self._dns_in_buf = np.zeros(0, dtype=np.float32)

        self._samples_since_doa = 0
        self._doa_update_samples = int(
            self.cfg.doa_update_ms * self.cfg.sample_rate / 1000)
        self._doa_buffer = np.zeros((self._doa_update_samples,
                                     self.cfg.n_mics), dtype=np.float32)
        self._doa_buf_len = 0
        self._last_doa_deg = self.cfg.static_az_deg
        self._external_az_deg: Optional[float] = None
        # Rolling buffer for wake_window_doa(): when KWS fires, we want
        # a high-confidence DOA over the last ~1 s of voiced AEC-cleaned
        # mic data instead of the running 100 ms estimate.
        self._wake_buffer_max = int(
            self.cfg.wake_window_ms * self.cfg.sample_rate / 1000)
        self._wake_buffer = np.zeros((0, self.cfg.n_mics),
                                     dtype=np.float32)
        self._wake_vad_buffer = np.zeros(0, dtype=bool)

    @property
    def latest_doa_deg(self) -> float:
        """The last SRP-PHAT estimate (independent of any external override)."""
        return float(self._last_doa_deg)

    def set_target_azimuth(self, az_deg: float) -> None:
        """Override MVDR steering azimuth from the perception side. The
        SRP-PHAT estimator keeps running so `latest_doa_deg` stays fresh and
        downstream code can decide which one to use. We only redirect the
        beam: the next MVDR call will use this az regardless of SRP-PHAT.
        """
        self._external_az_deg = float(az_deg)
        self.mvdr.set_doa(float(az_deg))

    def clear_target_azimuth(self) -> None:
        """Drop the external override; MVDR will follow SRP-PHAT again."""
        self._external_az_deg = None

    @staticmethod
    def _wrap180(deg: float) -> float:
        return float((float(deg) + 180.0) % 360.0 - 180.0)

    def apply_chassis_imu_yaw_delta_ccw_deg(self, delta_ccw_deg: float) -> None:
        """Shift internal DOA / MVDR steering after the chassis yaw changes.

        ``delta_ccw_deg`` is the change in IMU yaw in **degrees**, positive
        counter-clockwise viewed from above (same sign as
        ``yaw_deg_unwrapped`` deltas on this robot). The mic ring is rigid
        on the chassis, so a world-fixed acoustic source's azimuth in the
        mic-array frame advances **clockwise** by that amount, i.e. the
        CCW-positive mic azimuth **decreases** by ``delta_ccw_deg``.

        When ``set_target_azimuth`` is active (external perception override),
        this call is a no-op so we do not fight the ROS steering path.

        Also clears the SRP streaming smoother and the short DOA buffer so
        the next ``estimate()`` is not blended with pre-rotation frames.
        """
        d = float(delta_ccw_deg)
        if abs(d) < 0.05:
            return
        if self._external_az_deg is not None:
            logger.info(
                "apply_chassis_imu_yaw_delta_ccw: skipped "
                "(external MVDR override active)"
            )
            return
        mic = self._wrap180(self._last_doa_deg - d)
        self.srp.reset_streaming_smoothing()
        self._last_doa_deg = mic
        self._doa_buf_len = 0
        self.mvdr.set_doa(mic)
        logger.info(
            "apply_chassis_imu_yaw_delta_ccw: Δ=%+.2f° -> mic_az=%+.2f° "
            "(MVDR + SRP smoother reset)",
            d, mic,
        )

    def dominant_doa(
        self,
        top_k_pct: float = 50.0,
        window_ms: Optional[int] = None,
    ) -> Optional[Tuple[float, float]]:
        """SRP-PHAT over the loudest top-k% frames of the rolling buffer.

        Use this when the clip is dominated by a single loud source
        (e.g. a known speaker placed at a labelled azimuth) and there
        is no KWS event to anchor on. Per-hop RMS is computed across
        the buffered AEC-cleaned mic history, the top ``top_k_pct``
        loudest hops are kept, concatenated, and one big SRP-PHAT is
        run over the result. Stray/quieter sounds get filtered out
        because their hops fall below the percentile threshold and
        do not contribute to the cross-spectra.

        Returns ``(az_mic_deg, normalised_peak)`` or ``None`` if the
        buffer has too little material. ``normalised_peak`` is the
        SRP-PHAT peak divided by its mean across the azimuth grid
        (>2 = very confident, ~1.4 = OK, <1.2 = sketchy).
        """
        mics = self._wake_buffer
        if window_ms is not None:
            n = int(window_ms * self.cfg.sample_rate / 1000)
            mics = mics[-min(n, len(mics)):]
        if len(mics) < self.srp.cfg.n_fft:
            return None
        hop = self.cfg.hop_samples
        n_chunks = len(mics) // hop
        if n_chunks == 0:
            return None
        chunked = mics[: n_chunks * hop].reshape(
            n_chunks, hop, mics.shape[1])
        rms_per_chunk = np.sqrt(
            np.mean(chunked ** 2, axis=(1, 2)) + 1e-12)
        if rms_per_chunk.max() <= 0:
            return None
        threshold = float(np.percentile(rms_per_chunk,
                                        100.0 - float(top_k_pct)))
        keep = rms_per_chunk >= threshold
        if not keep.any():
            return None
        keep_full = np.repeat(keep, hop)
        n_full = min(len(keep_full), n_chunks * hop)
        mics_keep = mics[:n_full][keep_full[:n_full]]
        if len(mics_keep) < self.srp.cfg.n_fft:
            return None
        # Use the batch SRP-PHAT path: integrates cross-spectra across
        # all loud frames in the kept window. The single-frame
        # ``estimate`` only uses the first n_fft samples, which would
        # discard >95% of a 1 s wake utterance.
        az_deg, peak, P = self.srp.estimate_batch(mics_keep)
        norm_peak = float(peak / (np.mean(P) + 1e-12))
        return float(az_deg), norm_peak

    def wake_window_doa(
        self,
        window_ms: Optional[int] = None,
        min_voiced_ratio: float = 0.3,
    ) -> Optional[Tuple[float, float]]:
        """Compute a high-confidence DOA over the rolling wake buffer.

        Intended to be called by the KWS handler the moment a wake word
        fires. Runs SRP-PHAT *once* over the longest contiguous voiced
        slice of the rolling AEC-cleaned mic history (default 1.5 s).
        A long batch window lets the steered-response peak grow well
        above the noise floor, which gives a much more accurate single
        DOA estimate than the streaming 100 ms updates that the
        beamformer follows. The returned angle is **mic-frame**; the
        caller is responsible for converting to body frame if needed.

        Returns ``(az_mic_deg, normalised_peak)`` or ``None`` if there
        is not enough voiced material in the buffer (less than
        ``min_voiced_ratio`` of the requested window). ``normalised_peak``
        is the SRP-PHAT peak power divided by its mean across the
        azimuth grid (>1 means peak is above noise floor; >2 is
        comfortably confident, <1.2 is sketchy).
        """
        if window_ms is None:
            window_ms = self.cfg.wake_window_ms
        n = int(window_ms * self.cfg.sample_rate / 1000)
        n = min(n, len(self._wake_buffer))
        if n < self.cfg.sample_rate // 4:   # < 250 ms of audio
            return None
        mics = self._wake_buffer[-n:]
        # Align VAD chunks to the same window. Each VAD bit covers
        # ``hop_samples``; figure out how many bits cover ``n`` samples.
        hop = self.cfg.hop_samples
        vad_bits_needed = n // hop
        vad = self._wake_vad_buffer[-vad_bits_needed:] \
            if vad_bits_needed > 0 else np.zeros(0, dtype=bool)
        voiced = ~vad if vad.size > 0 else np.ones(0, dtype=bool)
        voiced_ratio = float(voiced.mean()) if voiced.size > 0 else 0.0
        if voiced_ratio < min_voiced_ratio:
            return None
        # Concatenate voiced chunks only; SRP-PHAT does not care about
        # the time order of frames, only the per-mic spectra.
        if voiced.size > 0:
            voiced_idx = np.repeat(voiced, hop)
            n_v = min(len(voiced_idx), len(mics))
            voiced_idx = voiced_idx[:n_v]
            mics_voiced = mics[:n_v][voiced_idx]
        else:
            mics_voiced = mics
        if len(mics_voiced) < self.srp.cfg.n_fft:
            return None
        # Use the batch SRP-PHAT path: integrates cross-spectra over
        # every voiced FFT frame in the wake window (~ 30+ overlapping
        # frames for a 500 ms keyword). The previous single-frame
        # estimate() truncated to the first 32 ms, which made
        # wake_window_doa() effectively reduce to "use the very first
        # voiced frame", yielding low-confidence DOAs. Bypassing the
        # smoother on top of the batch is also necessary — see
        # estimate_batch() docstring and the comment in
        # SrpPhat.estimate(bypass_smoothing=True).
        az_deg, peak, P = self.srp.estimate_batch(mics_voiced)
        norm_peak = float(peak / (np.mean(P) + 1e-12))
        return float(az_deg), norm_peak

    def process_block(self, frames_8ch: np.ndarray) -> np.ndarray:
        """Process N samples (N % hop == 0). Return (N,) float32 clean."""
        if frames_8ch.dtype == np.int16:
            x = frames_8ch.astype(np.float32) / 32768.0
        else:
            x = frames_8ch.astype(np.float32, copy=False)
        mics = x[:, : self.cfg.n_mics]
        if self.cfg.mic_channel_perm is not None:
            perm = list(self.cfg.mic_channel_perm)
            assert len(perm) == self.cfg.n_mics, \
                f"mic_channel_perm length {len(perm)} != n_mics " \
                f"{self.cfg.n_mics}"
            mics = mics[:, perm]
        refs = x[:, self.cfg.n_mics: self.cfg.n_mics + self.cfg.n_ref]
        far_mono = refs.mean(axis=1) if refs.shape[1] > 0 \
            else np.zeros(x.shape[0], dtype=np.float32)

        T = mics.shape[0]
        hop = self.cfg.hop_samples
        n_chunks = T // hop
        n_total = n_chunks * hop
        mics_for_mvdr = mics[: n_total]

        # Coarse VAD-like gating on raw mic energy. Used to update MVDR
        # noise covariance without needing the Silero VAD here (Silero
        # runs on the *output* clean stream).
        rms = np.sqrt(np.mean(mics ** 2, axis=1)) + 1e-9
        rms_chunks = rms[: n_total].reshape(n_chunks, hop).mean(axis=1)
        vad_silence = rms_chunks < self.cfg.silence_rms_threshold

        if self.cfg.aec_per_mic:
            # ---- per-mic AEC (before MVDR and before SRP-PHAT) ----
            # Each mic sees a time-invariant echo path because MVDR's
            # moving weights are downstream. Voice probe: +19.2 dB
            # ERLE on raw ch0 baseline.
            mics_clean = np.empty_like(mics_for_mvdr)
            N = self.aec_pool[0].frame_size
            far_clip = far_mono[:n_total]
            for m in range(self.cfg.n_mics):
                aec = self.aec_pool[m]
                col_in = mics_for_mvdr[:, m]
                col_out = np.empty_like(col_in)
                for s in range(0, n_total, N):
                    nn = col_in[s:s + N]
                    ff = far_clip[s:s + N]
                    if len(nn) < N:
                        pad = N - len(nn)
                        nn = np.concatenate(
                            [nn, np.zeros(pad, dtype=np.float32)])
                        ff = np.concatenate(
                            [ff, np.zeros(pad, dtype=np.float32)])
                        col_out[s:s + N - pad] = aec.process_frame(
                            nn, ff)[: N - pad]
                    else:
                        col_out[s:s + N] = aec.process_frame(nn, ff)
                mics_clean[:, m] = col_out
        else:
            mics_clean = mics_for_mvdr

        # ---- DOA update on AEC-cleaned mics ----
        # SRP-PHAT now sees mics with playback echo removed, so the
        # estimate locks onto the talker, not the speaker. Per-frame
        # the estimate is also pushed into a circular buffer so a
        # later wake_window_doa() call can recompute over a longer
        # voiced window with higher confidence.
        if self.cfg.use_doa_for_mvdr and self.cfg.aec_per_mic:
            doa_input = mics_clean
        else:
            doa_input = mics_for_mvdr
        # Keep a rolling history for wake_window_doa(). Use the AEC-
        # cleaned mics so the wake-time SRP also benefits.
        self._wake_buffer = np.concatenate(
            [self._wake_buffer, doa_input], axis=0)
        if len(self._wake_buffer) > self._wake_buffer_max:
            self._wake_buffer = self._wake_buffer[
                -self._wake_buffer_max:].copy()
        # Same for VAD bits aligned to those mic samples.
        self._wake_vad_buffer = np.concatenate(
            [self._wake_vad_buffer, vad_silence], axis=0)
        max_chunks = self._wake_buffer_max // hop
        if len(self._wake_vad_buffer) > max_chunks:
            self._wake_vad_buffer = self._wake_vad_buffer[
                -max_chunks:].copy()

        if self.cfg.use_doa_for_mvdr:
            consumed = 0
            while consumed < n_total:
                need = self._doa_update_samples - self._doa_buf_len
                take = min(need, n_total - consumed)
                self._doa_buffer[self._doa_buf_len:
                                 self._doa_buf_len + take] = \
                    doa_input[consumed: consumed + take]
                self._doa_buf_len += take
                consumed += take
                if self._doa_buf_len >= self._doa_update_samples:
                    az_deg, _peak, _ = self.srp.estimate(self._doa_buffer)
                    self._last_doa_deg = az_deg
                    if self._external_az_deg is None:
                        self.mvdr.set_doa(az_deg)
                    self._doa_buf_len = 0

        if self.cfg.aec_per_mic:
            beam = self.mvdr.process(mics_clean, vad_silence)
            out = beam
        else:
            # ---- legacy: MVDR -> single AEC ----
            beam = self.mvdr.process(mics_for_mvdr, vad_silence)
            out = np.zeros_like(beam)
            N = self.aec.frame_size
            for s in range(0, len(beam), N):
                n = beam[s:s + N]
                f = far_mono[s:s + N]
                if len(n) < N:
                    pad = N - len(n)
                    n = np.concatenate(
                        [n, np.zeros(pad, dtype=np.float32)])
                    f = np.concatenate(
                        [f, np.zeros(pad, dtype=np.float32)])
                    out[s:s + N - pad] = self.aec.process_frame(
                        n, f)[: N - pad]
                else:
                    out[s:s + N] = self.aec.process_frame(n, f)

        # ---- DNS ----
        # DNS prefers larger blocks (DfNet 30ms, DTLN 32ms). We buffer the
        # AEC output across calls so each DNS call is exactly block_size; any
        # tail is held over to the next process_block. This keeps RTF in
        # check on Pi5 (DfNet ~RTF 0.8 at 480-sample blocks vs >10 at 160).
        bs = getattr(self.denoiser, "block_size", hop)
        self._dns_in_buf = np.concatenate([self._dns_in_buf, out])
        n_full = len(self._dns_in_buf) // bs
        if n_full == 0:
            return np.zeros(0, dtype=np.float32)
        n_consume = n_full * bs
        clean = np.zeros(n_consume, dtype=np.float32)
        for i in range(n_full):
            chunk = self._dns_in_buf[i * bs: (i + 1) * bs]
            clean[i * bs: (i + 1) * bs] = self.denoiser.process(chunk)
        self._dns_in_buf = self._dns_in_buf[n_consume:].copy()
        return clean
