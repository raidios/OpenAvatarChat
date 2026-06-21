"""
Far-field 8-channel audio source for the robot client.

Drop-in replacement for the per-frame ``stream.read(n_samples, ...)`` call
that ``mic_sender`` makes on a PyAudio input stream. Internally it pulls 8
channels from the M260C live stream (``adb forward tcp:9999``), runs the
in-repo DSP pipeline (SRP-PHAT DOA → MVDR → AEC → DNS), and serves a
clean 16-kHz mono int16 PCM stream back via the same API the rest of the
client expects.

Usage (typical):

    from farfield_audio_source import FarfieldAudioSource

    src = FarfieldAudioSource()         # uses sensible defaults
    src.open()                          # starts board audio_server, connects
    pcm_bytes = src.read(1600)          # 100 ms @ 16 kHz, int16 mono
    src.close()

The ``read()`` call blocks until the requested number of samples is
available and returns ``2 * n_samples`` bytes — exactly what PyAudio's
``stream.read(n_samples, exception_on_overflow=False)`` returns. This lets
``client/chat_client.py`` use either source unchanged.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Make repo + thirdparty paths importable when this file is loaded from the
# ``client/`` directory (the same way ``client/main.py`` does).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_THIRDPARTY = _REPO_ROOT / "thirdparty" / "M2_SDK" / "live_stream" / "host"
if str(_THIRDPARTY) not in sys.path:
    sys.path.insert(0, str(_THIRDPARTY))

from live_stream import LiveStream                                      # noqa: E402
from audio_frontend.dsp.pipeline import DspPipeline, PipelineConfig     # noqa: E402

logger = logging.getLogger(__name__)


class FarfieldAudioSource:
    """Blocking 8ch → mono DSP source with a PyAudio-compatible ``read()``.

    The DSP pipeline emits clean mono in chunks aligned to the configured
    DNS block size (e.g. 128 for Hailo DTLN, 480 for DfNet); we accumulate
    those chunks into an internal float32 ring and slice off exactly
    ``n_samples`` at each ``read()``. The first call may take slightly
    longer than steady state because the pipeline must fill its analysis
    buffers before producing output — it never blocks indefinitely though,
    just waits one or two extra LiveStream periods.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        sample_rate: int = 16000,
        hop_samples: int = 160,
        n_mics: int = 6,
        n_ref: int = 2,
        array_radius_m: float = 0.033,
        aec_backend: str = "auto",
        aec_filter_length_ms: int = 200,
        aec_per_mic: bool = True,
        dns_backend: str = "auto",
        dns_use_passthrough: bool = False,
        use_doa_for_mvdr: bool = True,
        static_az_body_deg: float = 0.0,
        doa_update_ms: int = 100,
        livestream_period_ms: int = 32,
        auto_start_audio_server: bool = True,
        connect_timeout: float = 5.0,
        mic_yaw_offset_deg: float = 30.0,
        wake_spotter: "Optional[object]" = None,
        wake_doa_min_peak: float = 1.20,
        wake_doa_mid_peak: float = 1.10,
        wake_doa_consensus_deg: float = 45.0,
    ):
        """``static_az_body_deg`` and every azimuth crossing this boundary
        is interpreted in **body frame** (0° = vehicle forward). Internally
        the DSP pipeline keeps running in mic-array frame (0° = ch0
        direction); we shift by ``mic_yaw_offset_deg`` at the boundary.

        ``mic_yaw_offset_deg`` is **the body-frame azimuth of logical mic
        0** (the channel that sits at index 0 after applying the M260C
        channel permutation in ``audio_frontend.dsp.pipeline``). Empirical
        tap-test on this robot pins logical mic 0 at body +30°, so the
        default offset is +30°. With this definition:

            mic_az  = body_az - mic_yaw_offset_deg
            body_az = mic_az  + mic_yaw_offset_deg

        i.e. a source at body +30° (where mic 0 lives) reads out as
        mic_az = 0; a source at body 0° (straight ahead) reads out as
        mic_az = -30°.

        This sign convention matches ``audio_frontend.dsp.offline``
        (see comments around ``stream_body_az`` there). Earlier versions
        of this file used the opposite sign and returned a wrong
        ``latest_doa_deg``; that surfaced as the wake-orient controller
        rotating in the wrong direction (because at most user positions
        the bug flipped sign across the body forward axis).
        """
        self.mic_yaw_offset_deg = float(mic_yaw_offset_deg)
        self.wake_doa_min_peak = float(wake_doa_min_peak)
        self.wake_doa_mid_peak = float(wake_doa_mid_peak)
        self.wake_doa_consensus_deg = float(wake_doa_consensus_deg)
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.connect_timeout = connect_timeout
        self.auto_start_audio_server = auto_start_audio_server

        period_samples = max(
            1, sample_rate * livestream_period_ms // 1000)
        if period_samples % hop_samples != 0:
            period_samples = max(
                hop_samples,
                (period_samples // hop_samples) * hop_samples,
            )
        self._period_samples = period_samples
        self._period_ms = period_samples * 1000 // sample_rate

        self._pipeline = DspPipeline(PipelineConfig(
            sample_rate=sample_rate,
            hop_samples=hop_samples,
            n_mics=n_mics,
            n_ref=n_ref,
            array_radius_m=array_radius_m,
            aec_backend=aec_backend,
            aec_filter_length_ms=aec_filter_length_ms,
            aec_per_mic=aec_per_mic,
            dns_backend=dns_backend,
            dns_use_passthrough=dns_use_passthrough,
            use_doa_for_mvdr=use_doa_for_mvdr,
            static_az_deg=self._body_to_mic_az(static_az_body_deg),
            doa_update_ms=doa_update_ms,
        ))
        self._dsp_lock = threading.Lock()
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._stream: Optional[LiveStream] = None
        self._opened = False
        # Optional client-side wake-word spotter. We feed every "clean"
        # block produced by the DSP pipeline into it; on a hit it can
        # snapshot self.latest_doa_deg via the doa_snapshot_fn we set
        # below at construction time.
        self._wake_spotter = wake_spotter
        if self._wake_spotter is not None:
            try:
                # IMPORTANT: use wake_doa_snapshot_deg, not the streaming
                # latest_doa_deg. The streaming value is a 100 ms-window
                # estimate and at the moment Sherpa-ONNX fires (end of
                # keyword) the user's speech has already decayed, leading
                # to small-magnitude noise-dominated DOAs that mis-aim
                # the rotation. The wake-window estimator integrates
                # over the whole utterance instead.
                self._wake_spotter.set_doa_snapshot_fn(
                    lambda: self.wake_doa_snapshot_deg)
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Start the board audio_server (if requested) and connect TCP."""
        if self.auto_start_audio_server:
            self._ensure_board_raw_mode()
        self._stream = LiveStream(
            host=self.host,
            port=self.port,
            connect_timeout=self.connect_timeout,
            ensure_forward=False,   # board_audio_mode already applied it
        )
        self._stream.connect()
        self._opened = True
        logger.info(
            "FarfieldAudioSource open: %s:%d  period=%dms  denoiser=%s",
            self.host, self.port, self._period_ms,
            self._pipeline.denoiser.name,
        )

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._opened = False

    def _ensure_board_raw_mode(self) -> None:
        try:
            tool = _REPO_ROOT / "tools" / "board_audio_mode.py"
            python = sys.executable
            cp = subprocess.run(
                [python, str(tool), "start-raw"],
                capture_output=True, text=True, timeout=15,
            )
            if cp.returncode != 0:
                logger.warning(
                    "board_audio_mode start-raw rc=%d: %s",
                    cp.returncode, cp.stderr.strip(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("board_audio_mode bootstrap failed: %s", exc)

    # ------------------------------------------------------------------
    # PyAudio-compatible blocking read
    # ------------------------------------------------------------------
    def read(self, n_samples: int,
             exception_on_overflow: bool = False) -> bytes:
        """Block until ``n_samples`` of clean 16-kHz mono are available;
        return ``2 * n_samples`` bytes of int16 little-endian PCM.

        The ``exception_on_overflow`` argument is accepted for API parity
        with ``pyaudio.Stream.read`` but ignored — overflow only matters
        for hardware capture, and our LiveStream buffers in the kernel.
        """
        if not self._opened or self._stream is None:
            raise RuntimeError("FarfieldAudioSource.read called before open()")
        while self._out_buf.size < n_samples:
            frames = self._stream.read_frames(self._period_samples)
            with self._dsp_lock:
                clean = self._pipeline.process_block(frames)
            if clean.size:
                self._out_buf = np.concatenate([self._out_buf, clean])
                if self._wake_spotter is not None:
                    try:
                        self._wake_spotter.feed_audio(clean)
                    except Exception:
                        logger.exception("wake_spotter.feed_audio raised")
        emit = self._out_buf[:n_samples]
        self._out_buf = self._out_buf[n_samples:].copy()
        i16 = (np.clip(emit, -1.0, 1.0) * 32767.0).astype(np.int16)
        return i16.tobytes()

    # ------------------------------------------------------------------
    # introspection helpers (used by teach UI / log lines)
    # ------------------------------------------------------------------
    @property
    def is_opened(self) -> bool:
        return self._opened

    @property
    def latest_doa_mic_deg(self) -> float:
        """Raw DOA in mic-array frame (0° = ch0 direction)."""
        return float(self._pipeline.latest_doa_deg)

    @property
    def latest_doa_deg(self) -> float:
        """DOA in **body frame** (0° = vehicle forward).

        WARNING: this property is **CW-positive viewed from above**:
        ``+90°`` is a source on the robot's RIGHT, ``-90°`` on its LEFT.
        That convention is opposite the IMU yaw / ``vw`` command on the
        chassis (CCW-positive standard). It is what fell out of the
        empirical SRP-PHAT calibration: the brute-force perm fit and
        the user's labelled DOA recordings agreed on this convention,
        and the production live-doa monitor reproduces it.

        Consumers that drive the chassis (``WakeOrientController``)
        negate this value when converting to an IMU yaw delta. Other
        consumers — server-side analytics, log lines, perception_node
        bookkeeping — should treat the sign convention as documented
        and convert at the boundary if they need the standard CCW
        convention.

        This property is the **streaming** DOA, updated every
        ``doa_update_ms`` (~100 ms). Use it for visualization / live
        monitoring; for an event trigger like KWS wake, see
        :pyattr:`wake_doa_snapshot_deg`, which integrates over a
        longer voiced window for a much more confident estimate.
        """
        return self._mic_to_body_az(self._pipeline.latest_doa_deg)

    @property
    def wake_doa_snapshot_deg(self) -> Optional[float]:
        """Wake-event DOA in **body frame** (CW positive; see warning
        on :pyattr:`latest_doa_deg`).

        Computes one big SRP-PHAT over the recent voiced segment of
        the rolling 1.5 s wake buffer (``wake_window_doa``), falling
        back to a loud-frame "dominant DOA" if the VAD is too
        permissive on the AEC residual, and finally to the streaming
        ``latest_doa_deg`` if neither has enough material.

        The fallbacks matter on this platform: Sherpa-ONNX KWS fires
        ~end-of-keyword, by which time the user's voice has already
        decayed below VAD threshold for a couple of frames. Reading
        ``latest_doa_deg`` at that exact instant yielded SNR-poor
        single-frame estimates (≈ -24° instead of ±90°). The wake
        window covers the actual utterance and gives a stable
        large-magnitude DOA the rotation controller can act on.

        Returns body-frame degrees, or ``None`` if not even the
        streaming estimate is available (pipeline never warmed up).
        """
        latest_mic: Optional[float] = None
        latest_body: Optional[float] = None
        try:
            latest_mic = float(self._pipeline.latest_doa_deg)
            latest_body = self._mic_to_body_az(latest_mic)
        except Exception:
            pass

        wake_res = None
        candidates = []
        try:
            wake_res = self._pipeline.wake_window_doa()
        except Exception:
            wake_res = None
        if wake_res is None:
            for window_ms in (600, 1000, 1500):
                try:
                    dom = self._pipeline.dominant_doa(
                        top_k_pct=30.0, window_ms=window_ms)
                except Exception:
                    dom = None
                if dom is not None:
                    candidates.append((f"dominant{window_ms}", dom))
        else:
            candidates.append(("wake_window", wake_res))

        chosen = self._choose_wake_doa_candidate(candidates)
        if chosen is not None:
            source, mic_az_deg, norm_peak = chosen
            body_az_deg = self._mic_to_body_az(float(mic_az_deg))
            logging.getLogger(__name__).info(
                "wake_doa_snapshot source=%s mic=%+0.1f° body=%+0.1f° "
                "peak=%0.3f latest_mic=%s latest_body=%s offset=%+0.1f°",
                source,
                float(mic_az_deg),
                body_az_deg,
                float(norm_peak),
                "n/a" if latest_mic is None else f"{latest_mic:+0.1f}°",
                "n/a" if latest_body is None else f"{latest_body:+0.1f}°",
                self.mic_yaw_offset_deg,
            )
            return body_az_deg
        if candidates:
            parts = []
            for source, (mic_az_deg, norm_peak) in candidates:
                parts.append(
                    f"{source}:mic={float(mic_az_deg):+.1f}° "
                    f"body={self._mic_to_body_az(float(mic_az_deg)):+.1f}° "
                    f"peak={float(norm_peak):.3f}"
                )
            logging.getLogger(__name__).warning(
                "wake_doa_snapshot rejected: %s; min_peak=%0.3f "
                "mid_peak=%0.3f consensus=%0.1f°",
                "; ".join(parts),
                self.wake_doa_min_peak,
                self.wake_doa_mid_peak,
                self.wake_doa_consensus_deg,
            )
            return None
        # last resort: streaming snapshot
        if latest_body is not None:
            logging.getLogger(__name__).info(
                "wake_doa_snapshot source=latest mic=%s body=%+0.1f° offset=%+0.1f°",
                "n/a" if latest_mic is None else f"{latest_mic:+0.1f}°",
                latest_body,
                self.mic_yaw_offset_deg,
            )
            return latest_body
        return None

    def _choose_wake_doa_candidate(self, candidates):
        """Pick a wake DOA candidate or return None when confidence is weak."""
        parsed = []
        for source, res in candidates:
            mic_az_deg, norm_peak = res
            parsed.append((source, float(mic_az_deg), float(norm_peak)))
        if not parsed:
            return None
        best = max(parsed, key=lambda item: item[2])
        if best[2] >= self.wake_doa_min_peak:
            return best

        mid = [item for item in parsed if item[2] >= self.wake_doa_mid_peak]
        if len(mid) < 2:
            return None

        best_source, best_mic, best_peak = max(mid, key=lambda item: item[2])
        best_body = self._mic_to_body_az(best_mic)
        agree = []
        for item in mid:
            body = self._mic_to_body_az(item[1])
            err = abs(self._wrap180(body - best_body))
            if err <= self.wake_doa_consensus_deg:
                agree.append(item)
        if len(agree) >= 2:
            return best_source, best_mic, best_peak
        return None

    @property
    def denoiser_name(self) -> str:
        return self._pipeline.denoiser.name

    def set_target_azimuth(self, az_body_deg: float) -> None:
        """Steer MVDR toward a body-frame azimuth (0° = vehicle forward)."""
        with self._dsp_lock:
            self._pipeline.set_target_azimuth(self._body_to_mic_az(az_body_deg))

    def clear_target_azimuth(self) -> None:
        with self._dsp_lock:
            self._pipeline.clear_target_azimuth()

    def enable_wake_spotter(self, on: bool) -> None:
        """Pause/resume client-side KWS while keeping DSP and ASR audio alive."""
        spotter = self._wake_spotter
        if spotter is None:
            return
        try:
            spotter.enable(bool(on))
        except AttributeError:
            return

    def notify_chassis_imu_yaw_delta_ccw_deg(self, delta_ccw_deg: float) -> None:
        """After chassis yaw changes (CCW-positive deg), retarget MVDR/DOA.

        Keeps the beamformer aligned with a rigid mic ring on the robot when
        ``WakeOrientController`` rotates the base. Thread-safe with ``read``.
        """
        with self._dsp_lock:
            self._pipeline.apply_chassis_imu_yaw_delta_ccw_deg(
                float(delta_ccw_deg))

    # ------------------------------------------------------------------
    # frame conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap180(deg: float) -> float:
        return float((float(deg) + 180.0) % 360.0 - 180.0)

    def _body_to_mic_az(self, body_az_deg: float) -> float:
        # mic 0 sits at body +mic_yaw_offset_deg, so mic_az = body_az - offset.
        return self._wrap180(body_az_deg - self.mic_yaw_offset_deg)

    def _mic_to_body_az(self, mic_az_deg: float) -> float:
        return self._wrap180(mic_az_deg + self.mic_yaw_offset_deg)

    # ------------------------------------------------------------------
    # PyAudio.Stream-shaped no-ops so mic_sender can call them blindly
    # ------------------------------------------------------------------
    def stop_stream(self) -> None:
        return

    def start_stream(self) -> None:
        return
