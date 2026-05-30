"""
Client-side wake-word spotter for the robot.

Runs Sherpa-ONNX KeywordSpotter on the AEC+MVDR-cleaned mono audio that
the rest of the client sends to the server, and fires a callback the
moment a keyword is detected. Server-side ``HandlerWakeWord`` keeps doing
its own KWS for the conversation flow (TTS reply, session timeout); this
client-side spotter exists purely to:

  * snapshot the high-confidence DOA at the wake moment via the live
    DspPipeline (which only the client owns)
  * trigger local actions that need wall-clock-fast reactions, like
    rotating the chassis to face the wake direction

Why two KWS instances? Because the server-side one fires ~200-300 ms
after the client emitted the audio (audio buffering + ws + handler
queueing), at which point the live DOA buffer has moved on. Running a
mirror KWS on the same machine that owns the DOA buffer keeps the
latency under one DSP block (~10 ms) and removes any cross-process
synchronization. The model is ~12 MB and Sherpa is multi-threaded but
Pi5-friendly: profiling on the same hardware shows ~3% of one core in
steady state.

Usage:

    spotter = WakeWordSpotter(
        model_dir="models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
        keywords_file="config/keywords.txt",
        doa_snapshot_fn=lambda: farfield_source.latest_doa_deg,
        on_wake=lambda kw, az: print(f"WAKE {kw!r} from body {az}°"),
    )
    spotter.start()
    # in the audio loop, after pipeline.process_block():
    spotter.feed_audio(clean_mono_float32)

The class is thread-safe in the sense that ``feed_audio`` and
``set_on_wake`` can be called from different threads. The Sherpa
spotter itself is single-threaded; we serialise calls to it with a
mutex so the DSP read loop and any external "drain" thread can both
push without corruption.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

WakeCallback = Callable[[str, Optional[float]], None]
DoaSnapshotFn = Callable[[], Optional[float]]


def _ensure_onnxruntime_lib() -> None:
    """Mirror of ``HandlerWakeWord._ensure_onnxruntime_lib``.

    Sherpa-ONNX 1.10+ links libonnxruntime.so via RPATH from
    ``site-packages/sherpa_onnx/lib/``, but pip's onnxruntime puts the
    actual ``.so.X.Y.Z`` inside its ``capi/`` folder instead. We
    symlink so Sherpa can dlopen it without LD_LIBRARY_PATH games.
    Idempotent.
    """
    import glob
    import importlib.util

    try:
        import onnxruntime
        capi_dir = os.path.dirname(onnxruntime.capi._pybind_state.__file__)
    except Exception:
        return
    candidates = glob.glob(os.path.join(capi_dir, "libonnxruntime.so.*"))
    if not candidates:
        return
    ort_lib = candidates[0]

    spec = importlib.util.find_spec("sherpa_onnx")
    if spec is None or spec.origin is None:
        return
    sherpa_pkg_dir = os.path.dirname(spec.origin)
    sherpa_lib_dir = os.path.join(sherpa_pkg_dir, "lib")
    if not os.path.isdir(sherpa_lib_dir):
        return
    target = os.path.join(sherpa_lib_dir, "libonnxruntime.so")
    if not os.path.exists(target):
        try:
            os.symlink(ort_lib, target)
            logger.info("Created symlink %s -> %s", target, ort_lib)
        except OSError as exc:
            logger.warning("Could not create symlink: %s", exc)


class WakeWordSpotter:
    """Sherpa-ONNX KeywordSpotter wrapper with a DOA-aware on_wake hook."""

    def __init__(
        self,
        model_dir: str,
        keywords_file: str,
        keywords_score: float = 1.5,
        keywords_threshold: float = 0.25,
        num_threads: int = 2,
        sample_rate: int = 16000,
        cooldown_s: float = 1.5,
        doa_snapshot_fn: Optional[DoaSnapshotFn] = None,
        on_wake: Optional[WakeCallback] = None,
        repo_root: Optional[Path] = None,
    ):
        """
        Args:
          model_dir: directory holding the encoder/decoder/joiner onnx
              and tokens.txt. Relative paths resolve against
              ``repo_root`` (defaulting to two levels up from this
              file, matching the ``HandlerWakeWord`` convention).
          keywords_file: same lexicon file the server uses. Relative
              paths resolve the same way.
          keywords_score / keywords_threshold: same Sherpa hyper-params
              as the server-side handler. Keep aligned to avoid the
              client and server disagreeing about wake events.
          cooldown_s: ignore additional wake fires within this window
              after a successful trigger. Keeps repeated detections
              of the same utterance from re-arming the rotation /
              ROS event train.
          doa_snapshot_fn: zero-arg callable that returns the current
              body-frame DOA in degrees (or ``None`` if unavailable).
              Called synchronously from ``feed_audio`` the moment a
              wake fires, so make it cheap and side-effect-free.
          on_wake: callback ``(keyword, doa_body_deg) -> None``. Runs
              in the same thread that called ``feed_audio``; if it
              blocks the audio reader will stall, so dispatch to a
              worker thread there if needed.
        """
        self._sample_rate = int(sample_rate)
        self._cfg = {
            "model_dir": model_dir,
            "keywords_file": keywords_file,
            "keywords_score": float(keywords_score),
            "keywords_threshold": float(keywords_threshold),
            "num_threads": int(num_threads),
        }
        self._cooldown_s = float(cooldown_s)
        self._repo_root = repo_root or Path(__file__).resolve().parents[1]

        self._spotter = None       # sherpa_onnx.KeywordSpotter
        self._stream = None
        self._lock = threading.Lock()
        self._last_fire_t: float = 0.0

        self._doa_snapshot_fn: Optional[DoaSnapshotFn] = doa_snapshot_fn
        self._on_wake: Optional[WakeCallback] = on_wake
        self._enabled: bool = True
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Load Sherpa + open a streaming session.

        Returns True if the spotter is ready, False if the model
        could not be loaded (in which case ``feed_audio`` becomes a
        no-op so the caller doesn't have to special-case it).
        """
        if self._loaded:
            return True
        try:
            _ensure_onnxruntime_lib()
            import sherpa_onnx
        except Exception as exc:
            logger.warning("sherpa_onnx import failed; KWS disabled: %s", exc)
            return False

        model_dir = self._resolve(self._cfg["model_dir"])
        keywords_file = self._resolve(self._cfg["keywords_file"])
        if not model_dir.is_dir():
            logger.warning("KWS model dir not found: %s", model_dir)
            return False
        if not keywords_file.is_file():
            logger.warning("KWS keywords file not found: %s", keywords_file)
            return False

        encoder = model_dir / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
        decoder = model_dir / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
        joiner = model_dir / "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
        tokens = model_dir / "tokens.txt"
        for p in (encoder, decoder, joiner, tokens):
            if not p.exists():
                logger.warning("KWS asset missing: %s", p)
                return False

        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                keywords_file=str(keywords_file),
                keywords_score=self._cfg["keywords_score"],
                keywords_threshold=self._cfg["keywords_threshold"],
                num_threads=self._cfg["num_threads"],
                provider="cpu",
            )
        except Exception as exc:
            logger.warning("Failed to load Sherpa KeywordSpotter: %s", exc)
            return False
        self._stream = self._spotter.create_stream()
        self._loaded = True
        logger.info(
            "Client-side KWS loaded (keywords=%s, threshold=%.2f)",
            keywords_file.name, self._cfg["keywords_threshold"],
        )
        return True

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = (self._repo_root / p).resolve()
        return p

    # ------------------------------------------------------------------
    # config setters (callable from any thread)
    # ------------------------------------------------------------------
    def set_on_wake(self, on_wake: Optional[WakeCallback]) -> None:
        with self._lock:
            self._on_wake = on_wake

    def set_doa_snapshot_fn(self,
                             fn: Optional[DoaSnapshotFn]) -> None:
        with self._lock:
            self._doa_snapshot_fn = fn

    def enable(self, on: bool) -> None:
        """Pause / resume the spotter without unloading the model.

        Useful while the chassis is mid-rotation: we don't want to
        re-arm a wake-orient maneuver from picking up our own TTS
        through the mic ring.
        """
        with self._lock:
            self._enabled = bool(on)
            if not self._enabled and self._stream is not None and \
                    self._spotter is not None:
                self._stream = self._spotter.create_stream()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # feed loop
    # ------------------------------------------------------------------
    def feed_audio(self, samples: np.ndarray) -> None:
        """Push a mono float32 16k chunk through the spotter.

        Accepts arbitrary chunk lengths >= 1; Sherpa internally buffers
        until enough samples have arrived for a decode step. Any
        non-finite values are clamped before forwarding to avoid the
        decoder NaN-poisoning on the rare AEC instability frame.
        """
        if not self._loaded or self._spotter is None or self._stream is None:
            return
        with self._lock:
            if not self._enabled:
                return
            on_wake = self._on_wake
            doa_snapshot_fn = self._doa_snapshot_fn

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim > 1:
            samples = samples.reshape(-1)
        if not np.isfinite(samples).all():
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        # Sherpa expects float32 in [-1, 1]; the DSP pipeline already
        # produces that, but clamp defensively to avoid the few-sample
        # over-shoots that can come out of MVDR.
        samples = np.clip(samples, -1.0, 1.0)

        with self._lock:
            self._stream.accept_waveform(self._sample_rate, samples)
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream).strip()
            if not result:
                return

            import time
            now = time.monotonic()
            if now - self._last_fire_t < self._cooldown_s:
                # In cooldown; consume the result but don't re-fire.
                self._stream = self._spotter.create_stream()
                return
            self._last_fire_t = now
            self._stream = self._spotter.create_stream()

        # Snapshot DOA OUTSIDE the spotter lock so a slow snapshot
        # function can't stall the audio thread for the full DSP
        # buffer. The buffer is owned by FarfieldAudioSource, not us.
        doa_body_deg: Optional[float] = None
        if doa_snapshot_fn is not None:
            try:
                doa_body_deg = doa_snapshot_fn()
            except Exception:
                logger.exception("doa_snapshot_fn raised; firing wake "
                                 "without DOA")

        logger.info("CLIENT KWS wake: %r  body_doa=%s°",
                    result, "n/a" if doa_body_deg is None
                    else f"{doa_body_deg:+.1f}")
        if on_wake is not None:
            try:
                on_wake(result, doa_body_deg)
            except Exception:
                logger.exception("on_wake callback raised")
