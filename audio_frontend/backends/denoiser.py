"""
Denoiser backend abstraction.

Stage 0: both CpuBackend and HailoBackend are passthrough placeholders so the
node graph wires up cleanly. Stage 1A wires a real CPU baseline (DeepFilterNet
or `df` Python package). Stage 1B wires DTLN on Hailo-10H and switches `auto`
to it.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ._hailo_probe import probe_hailo

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@runtime_checkable
class Denoiser(Protocol):
    """Stateful, single-channel streaming denoiser at 16 kHz."""

    sample_rate: int
    block_size: int  # input/output block size in samples (e.g. 480 = 30ms)
    name: str

    def reset(self) -> None: ...
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process exactly ``block_size`` float32 samples, return same shape."""


class _PassthroughDenoiser:
    """Placeholder that returns the input untouched. Stage-0 default."""

    def __init__(self, block_size: int = 480, name: str = "passthrough"):
        self.sample_rate = SAMPLE_RATE
        self.block_size = block_size
        self.name = name

    def reset(self) -> None:
        return

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        return frame


class CpuDenoiser(_PassthroughDenoiser):
    """
    CPU denoiser; tries DeepFilterNet 3 first, falls back to passthrough.

    Stage 1A baseline. Public surface stays ``process(block_size) -> block_size``
    so swapping in/out a real DNS does not perturb downstream code.
    """

    def __init__(self, block_size: int = 480, model: str = "deepfilternet"):
        super().__init__(block_size=block_size, name=f"cpu/{model}")
        self._inner: Optional[object] = None
        if model == "deepfilternet":
            try:
                # Lazy import; module imports torch/onnxruntime which may be
                # absent. We keep a CpuDenoiser callable either way.
                from ..dsp.dns_cpu import DfDenoiser
                self._inner = DfDenoiser(block_size=block_size)
                self.name = self._inner.name
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"DeepFilterNet unavailable ({exc!s}); using passthrough."
                )
                self.name = "cpu/passthrough"
        elif model != "passthrough":
            logger.warning(f"unknown CpuDenoiser model={model!r}; using passthrough")

    def reset(self) -> None:
        if self._inner is not None and hasattr(self._inner, "reset"):
            self._inner.reset()

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._inner is None:
            return super().process(frame)
        return self._inner.process(frame)


class HailoDenoiser:
    """
    DTLN streaming denoiser on Hailo-10H — v7 dual-HEF protocol.

    Two HEFs are loaded and run in series for each 8 ms (128-sample) audio
    block at 16 kHz; together they form one DTLN inference step:

      p1  (frequency-mask network):
          in : mag_in       (1, 257)   |  h_in_p1_0/1   (1, 128)
                                       |  c_in_p1_0/1   (1, 128)
          out: mask_p1      (1, 257)   |  h_out_p1_0/1  (1, 128)
                                       |  c_out_p1_0/1  (1, 128)

      p2  (time-domain refine network):
          in : frame_in     (1, 512)   |  h_in_p2_0/1   (1, 128)
                                       |  c_in_p2_0/1   (1, 128)
          out: dec_dense    (1, 512)   |  h_out_p2_0/1  (1, 128)
                                       |  c_out_p2_0/1  (1, 128)

    HailoRT 5.x InferModel API is used (legacy InferVStreams returns
    HAILO_NOT_IMPLEMENTED on Hailo-10H multi-input models). HEF stream
    shapes are taken verbatim from the model: audio I/O is `[1, 1, C]`
    and state I/O is `[C]`. Names are read from
    ``models/hailo/dtln_hef_manifest.json``, which is the build-side
    artifact mapping compiler-generated stream names (``fc1``/``conv7``/
    ``precision_change*``/``ew_add*``) to ONNX semantic roles.

    Streaming protocol (DTLN native, follows real_time_processing_tf_lite.py):
      * fft_size = 512, hop = 128 (8 ms), NO analysis/synthesis window
        (training did not use a window; the model has absorbed the framing).
      * For each 128-sample input frame:
          - shift the 512-sample analysis ring buffer; append the new 128.
          - mag/phase = rfft(ring); p1 -> mask; reconstruct masked spec;
            irfft(mag*mask*e^{j*phase}) -> 512-sample time block.
          - p2 -> 512-sample refined block.
          - shift the 512-sample synthesis buffer by HOP and zero the tail;
            add the refined block directly (no window); emit the oldest
            HOP=128 samples (4-frame OLA matches DTLN reference).

    If the HEF files are missing or hailo_platform fails to load, this
    wrapper degrades gracefully to passthrough so the rest of the pipeline
    keeps running.
    """

    DEFAULT_P1_NAMES = ("dtln_p1_h10h.hef",)
    DEFAULT_P2_NAMES = ("dtln_p2_h10h.hef",)
    DEFAULT_MANIFEST_NAMES = (
        "dtln_hef_manifest.json",     # authoritative, emitted by build
        "dtln_v7.manifest.json",      # legacy stub
    )

    BLOCK_LEN = 512
    HOP = 128
    FFT = 512
    LSTM_UNITS = 128
    N_STATE_PORTS = 4   # 2 layers * (h, c) per HEF

    def __init__(self,
                 block_size: int = 128,
                 hef_p1_path: Optional[str] = None,
                 hef_p2_path: Optional[str] = None,
                 manifest_path: Optional[str] = None,
                 silence_rms: Optional[float] = None,
                 atten_floor_db: Optional[float] = None):
        probe = probe_hailo()
        if not probe.available:
            raise RuntimeError(f"Hailo unavailable: {probe.reason}")
        if block_size != self.HOP:
            raise ValueError(
                f"DTLN runs at hop={self.HOP}; got block_size={block_size}"
            )
        self.sample_rate = SAMPLE_RATE
        self.block_size = block_size

        # Silence gate + attenuation floor — suppresses the "buzz" the v7
        # quantised DTLN emits on low-energy frames where the LSTM mask
        # estimate is noisy and gets normalised back to non-zero amplitude.
        # Both can be disabled by passing 0.0 (or env override). Defaults
        # match what's needed for the M260C MVDR output level (~ -20 dBFS
        # peak / -40 dBFS RMS while speaking 1.5 m away from the array).
        self._silence_rms = float(
            silence_rms if silence_rms is not None
            else os.environ.get("DTLN_SILENCE_RMS", "0.001")
        )
        self._atten_floor_db = float(
            atten_floor_db if atten_floor_db is not None
            else os.environ.get("DTLN_ATTEN_FLOOR_DB", "12.0")
        )
        # Cache of the dry input from the previous call. The HEF emits a
        # 1-hop-delayed refined block, so the dry pair for the current
        # ``emit`` is the previous frame's input.
        self._prev_dry_frame = np.zeros(self.HOP, dtype=np.float32)

        self.hef_p1_path = self._resolve_hef(hef_p1_path,
                                              self.DEFAULT_P1_NAMES)
        self.hef_p2_path = self._resolve_hef(hef_p2_path,
                                              self.DEFAULT_P2_NAMES)
        self.manifest_path = self._resolve_manifest(manifest_path)

        if self.hef_p1_path and self.hef_p2_path:
            self.name = "hailo/dtln_v7"
        else:
            missing = []
            if not self.hef_p1_path: missing.append("p1")
            if not self.hef_p2_path: missing.append("p2")
            self.name = f"hailo/dtln(passthrough; missing {','.join(missing)})"

        self._analysis_buf = np.zeros(self.BLOCK_LEN, dtype=np.float32)
        self._synthesis_buf = np.zeros(self.BLOCK_LEN, dtype=np.float32)

        self._cim1 = None
        self._cim2 = None
        if self.hef_p1_path and self.hef_p2_path:
            try:
                self._init_hefs()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"DTLN HEF init failed ({exc!s}); falling back to passthrough"
                )
                self._cim1 = None
                self._cim2 = None
                self.name = "hailo/dtln(passthrough; init failed)"

        self.reset()

    @staticmethod
    def _resolve_hef(explicit: Optional[str],
                     default_names: tuple) -> Optional[str]:
        if explicit:
            return explicit if Path(explicit).exists() else None
        from pathlib import Path as _P
        repo = _P(__file__).resolve().parents[2]
        for name in default_names:
            p = repo / "models" / "hailo" / name
            if p.exists():
                return str(p)
        return None

    @staticmethod
    def _resolve_manifest(explicit: Optional[str]) -> Optional[str]:
        if explicit:
            return explicit if Path(explicit).exists() else None
        from pathlib import Path as _P
        repo = _P(__file__).resolve().parents[2]
        for name in HailoDenoiser.DEFAULT_MANIFEST_NAMES:
            p = repo / "models" / "hailo" / name
            if p.exists():
                return str(p)
        return None

    def _normalize_manifest_section(self, sec: dict) -> dict:
        """Convert authoritative manifest schema into the flat form the
        wrapper consumes.

        Authoritative schema (emitted by the v7 build, ``dtln_hef_manifest.json``):
            {
              "audio_input":  {"vstream": ..., "onnx": ..., "shape": ...},
              "audio_output": {"vstream": ..., "onnx": ..., "shape": ...},
              "state_loopback": [
                  {"kind": "h"|"c", "layer": int,
                   "from_output_vstream": ..., "to_input_vstream": ...}
              ]
            }

        Legacy flat schema:
            {"audio_in": str, "audio_out": str,
             "state_in": [4 strs], "state_out": [4 strs]}

        We always emit the legacy flat form, ordered by (kind, layer) so
        ``state_in[k]`` and ``state_out[k]`` share the same (kind, layer).
        """
        if "audio_input" in sec and "state_loopback" in sec:
            audio_in  = sec["audio_input"]["vstream"]
            audio_out = sec["audio_output"]["vstream"]
            # Sort loopbacks by (kind, layer) so ordering is deterministic.
            loops = sorted(sec["state_loopback"],
                           key=lambda r: (r["kind"], r["layer"]))
            state_in  = [r["to_input_vstream"]    for r in loops]
            state_out = [r["from_output_vstream"] for r in loops]
            return {
                "audio_in":  audio_in,
                "audio_out": audio_out,
                "state_in":  state_in,
                "state_out": state_out,
            }
        # Legacy flat schema -- use as-is.
        return {
            "audio_in":  sec["audio_in"],
            "audio_out": sec["audio_out"],
            "state_in":  list(sec["state_in"]),
            "state_out": list(sec["state_out"]),
        }

    def _load_manifest(self,
                       in_names_p1, out_names_p1,
                       in_names_p2, out_names_p2):
        """Resolve which actual VStream names play which semantic role.

        Hailo's compiler renames input/output VStreams when emitting a HEF
        (``dtln_v7_p1/input_layer1``, ``dtln_v7_p1/precision_change6``,
        ``dtln_v7_p1/ew_add2``…). The build produces a manifest mapping
        these auto-names to ONNX semantic roles; wrapper trusts it.
        """
        if self.manifest_path is None:
            raise RuntimeError(
                "no DTLN v7 manifest found; expected one of "
                f"{self.DEFAULT_MANIFEST_NAMES} under models/hailo/. "
                "Drop the build-side dtln_hef_manifest.json there."
            )
        with open(self.manifest_path, "r") as f:
            raw = json.load(f)
        return (
            self._normalize_manifest_section(raw["p1"]),
            self._normalize_manifest_section(raw["p2"]),
        )

    def _init_hefs(self) -> None:
        # HailoRT 5.x InferModel API path. The legacy InferVStreams path is
        # not implemented for Hailo-10H multi-input models — we hit
        # HAILO_NOT_IMPLEMENTED on activate().
        #
        # Scheduling: NONE was measured at 5 ms/frame but yielded all-zero
        # outputs when running two CIMs in turn (NONE expects a single
        # active context and times out otherwise). ROUND_ROBIN gives correct
        # outputs at ~11 ms/frame for sequential sync, but ~6.8 ms/frame
        # when both CIMs are queued via run_async() so the scheduler can
        # overlap the two contexts. We use that path with a 1-frame
        # cross-frame pipeline: frame N's p1 is queued in parallel with
        # frame N-1's p2. End-to-end latency cost: 1 hop = 8 ms.
        from hailo_platform import FormatType, VDevice  # type: ignore
        self._vdev = VDevice()

        def _make(hef_path: str):
            mdl = self._vdev.create_infer_model(hef_path)
            for s in mdl.inputs:
                s.set_format_type(FormatType.FLOAT32)
            for s in mdl.outputs:
                s.set_format_type(FormatType.FLOAT32)
            cim = mdl.configure()
            in_shape  = {s.name: list(s.shape) for s in mdl.inputs}
            out_shape = {s.name: list(s.shape) for s in mdl.outputs}
            in_names  = list(mdl.input_names)
            out_names = list(mdl.output_names)
            return mdl, cim, in_names, out_names, in_shape, out_shape

        (self._mdl1, self._cim1,
         in1_names, out1_names, self._in_shape1, self._out_shape1) = \
            _make(self.hef_p1_path)
        (self._mdl2, self._cim2,
         in2_names, out2_names, self._in_shape2, self._out_shape2) = \
            _make(self.hef_p2_path)

        logger.info(f"DTLN p1 inputs : {in1_names}")
        logger.info(f"DTLN p1 outputs: {out1_names}")
        logger.info(f"DTLN p2 inputs : {in2_names}")
        logger.info(f"DTLN p2 outputs: {out2_names}")

        self._map_p1, self._map_p2 = self._load_manifest(
            in1_names, out1_names, in2_names, out2_names)

        logger.info(f"DTLN v7 mapping (p1): {self._map_p1}")
        logger.info(f"DTLN v7 mapping (p2): {self._map_p2}")

        # Persistent bindings + pre-allocated output buffers (avoid per-frame
        # allocation overhead).
        self._bind1 = self._cim1.create_bindings()
        self._bind2 = self._cim2.create_bindings()
        self._out_bufs1 = {
            name: np.zeros(self._out_shape1[name], dtype=np.float32)
            for name in self._out_shape1
        }
        self._out_bufs2 = {
            name: np.zeros(self._out_shape2[name], dtype=np.float32)
            for name in self._out_shape2
        }
        for name, buf in self._out_bufs1.items():
            self._bind1.output(name).set_buffer(buf)
        for name, buf in self._out_bufs2.items():
            self._bind2.output(name).set_buffer(buf)

        # Pre-allocate input buffers too (one per input stream).
        self._in_bufs1 = {
            name: np.zeros(self._in_shape1[name], dtype=np.float32)
            for name in self._in_shape1
        }
        self._in_bufs2 = {
            name: np.zeros(self._in_shape2[name], dtype=np.float32)
            for name in self._in_shape2
        }
        for name, buf in self._in_bufs1.items():
            self._bind1.input(name).set_buffer(buf)
        for name, buf in self._in_bufs2.items():
            self._bind2.input(name).set_buffer(buf)

        # Unified shape lookup for _shape_input.
        self._in_shape = {**self._in_shape1, **self._in_shape2}
        self._out_shape = {**self._out_shape1, **self._out_shape2}

    def set_gate(self, silence_rms: Optional[float] = None,
                 atten_floor_db: Optional[float] = None) -> None:
        """Adjust the silence-gate / attenuation-floor on the fly.

        Useful for offline A/B benchmarks where we want to compare
        gate-on vs gate-off without re-creating the (single-instance)
        Hailo VDevice. Pass ``0.0`` to disable a behaviour, ``None`` to
        leave it untouched.
        """
        if silence_rms is not None:
            self._silence_rms = float(silence_rms)
        if atten_floor_db is not None:
            self._atten_floor_db = float(atten_floor_db)

    def reset(self) -> None:
        z = lambda: np.zeros(self.LSTM_UNITS, dtype=np.float32)
        self._states_p1 = [z() for _ in range(self.N_STATE_PORTS)]
        self._states_p2 = [z() for _ in range(self.N_STATE_PORTS)]
        self._analysis_buf.fill(0.0)
        self._synthesis_buf.fill(0.0)
        # Cross-frame pipelining: at the start of each frame we queue p1
        # with the current mag and p2 with the *previous* frame's time
        # block. ``_pending_time_block`` holds that previous time block so
        # the very first frame feeds zero (and emits zero-equivalent OLA).
        self._pending_time_block = np.zeros(self.FFT, dtype=np.float32)
        self._first_frame = True
        self._prev_dry_frame = np.zeros(self.HOP, dtype=np.float32)

    def _write_input(self, bind, name: str, in_shape, vec: np.ndarray) -> None:
        """Copy ``vec`` into the pre-allocated input buffer for ``name`` so
        that the persistent binding picks it up on the next run().

        InferModel binding buffers are real numpy arrays; mutating them in
        place is the documented zero-copy path.
        """
        shape = in_shape[name]
        flat = np.ascontiguousarray(vec, dtype=np.float32).reshape(-1)
        target_size = int(np.prod(shape))
        if flat.size != target_size:
            raise ValueError(
                f"input {name!r} expects {target_size} elements (shape {shape}); "
                f"got {flat.size}"
            )
        buf = bind.input(name).get_buffer()
        buf.reshape(-1)[:] = flat

    @staticmethod
    def _read_output(bind, name: str) -> np.ndarray:
        """Copy out a 1-D view of the binding's output buffer."""
        return np.ascontiguousarray(
            bind.output(name).get_buffer(), dtype=np.float32
        ).reshape(-1)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a HOP-sized (128-sample) audio frame.

        Cross-frame pipeline:
          - frame N: STFT -> mag_N
          - queue p1 with mag_N AND p2 with time_block_{N-1} concurrently
          - wait both
          - mask_N = p1.out;  refined_{N-1} = p2.out
          - time_block_N = iSTFT(mag_N * mask_N * exp(j*phase_N))   (cache)
          - OLA refined_{N-1}; emit HOP samples
        End-to-end this adds 1 hop of extra delay (8 ms) but keeps each
        per-frame call under the 8 ms RT budget.
        """
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32, copy=False)
        if self._cim1 is None or self._cim2 is None:
            return frame
        if frame.shape[-1] != self.HOP:
            raise ValueError(
                f"HailoDenoiser.process expects {self.HOP} samples, got {frame.shape}"
            )

        # 1) shift analysis ring + append new hop.
        self._analysis_buf[: self.BLOCK_LEN - self.HOP] = \
            self._analysis_buf[self.HOP:].copy()
        self._analysis_buf[self.BLOCK_LEN - self.HOP:] = frame.reshape(-1)

        # 2) rfft of unwindowed buffer (DTLN reference behaviour).
        spec = np.fft.rfft(self._analysis_buf, n=self.FFT)
        mag = np.abs(spec).astype(np.float32)
        phase = np.angle(spec).astype(np.float32)

        # 3) Queue p1 (current mag) and p2 (previous time block) concurrently.
        m1, m2 = self._map_p1, self._map_p2
        self._write_input(self._bind1, m1["audio_in"], self._in_shape1, mag)
        for k, name in enumerate(m1["state_in"]):
            self._write_input(self._bind1, name, self._in_shape1,
                              self._states_p1[k])
        self._write_input(self._bind2, m2["audio_in"], self._in_shape2,
                          self._pending_time_block)
        for k, name in enumerate(m2["state_in"]):
            self._write_input(self._bind2, name, self._in_shape2,
                              self._states_p2[k])
        job1 = self._cim1.run_async([self._bind1])
        job2 = self._cim2.run_async([self._bind2])
        job1.wait(1000)
        job2.wait(1000)

        # 4) p1 -> mask, then build this frame's time block (cached for N+1).
        mask = self._read_output(self._bind1, m1["audio_out"])
        for k, name in enumerate(m1["state_out"]):
            self._states_p1[k] = self._read_output(self._bind1, name)
        masked = mag * mask * np.exp(1j * phase)
        time_block_now = np.fft.irfft(masked, n=self.FFT).astype(np.float32)

        # 5) p2 -> refined block for the PREVIOUS frame.
        if self._first_frame:
            # First call: pending time block was zero, so p2's output
            # corresponds to "nothing" and is dropped (return zeros). The
            # caller now sees a 1-hop emit delay relative to input — same
            # as the DTLN reference's ~32 ms ring buffer fill behaviour.
            refined_prev = np.zeros(self.FFT, dtype=np.float32)
            self._first_frame = False
        else:
            refined_prev = self._read_output(self._bind2, m2["audio_out"])
            for k, name in enumerate(m2["state_out"]):
                self._states_p2[k] = self._read_output(self._bind2, name)

        # 6) Cache for next frame.
        self._pending_time_block = time_block_now

        # 7) Synthesis OLA: shift, zero the tail, add the refined block,
        # emit oldest HOP. (DTLN reference behaviour, no window.)
        self._synthesis_buf[: self.BLOCK_LEN - self.HOP] = \
            self._synthesis_buf[self.HOP:].copy()
        self._synthesis_buf[self.BLOCK_LEN - self.HOP:] = 0.0
        self._synthesis_buf += refined_prev
        emit = self._synthesis_buf[: self.HOP].astype(np.float32, copy=True)

        # 8) Silence-gate + attenuation-floor against quantised LSTM buzz.
        # ``emit`` is 1 hop delayed relative to the input; its dry pair is
        # the input from the *previous* call (cached in ``_prev_dry_frame``).
        # We always advance the LSTM (no skipping inference) — only the
        # output is post-processed so DTLN's internal state stays coherent.
        dry = self._prev_dry_frame
        new_dry = frame.reshape(-1).astype(np.float32, copy=True)
        if not self._first_frame:
            dry_rms = float(np.sqrt(np.mean(dry * dry))) + 1e-12
            if self._silence_rms > 0.0 and dry_rms < self._silence_rms:
                # Pure silence: bypass DNS to avoid mask-quantisation buzz.
                emit = dry.copy()
            elif self._atten_floor_db > 0.0:
                # Cap how much DNS is allowed to attenuate the dry signal.
                # When ``emit_rms`` falls more than ``floor_db`` below the
                # dry envelope, replace ``emit`` with a scaled-down copy of
                # the dry frame instead of letting buzz/quantisation noise
                # take over the output. (Boosting ``emit`` directly would
                # only amplify the artifact.)
                emit_rms = float(np.sqrt(np.mean(emit * emit))) + 1e-12
                min_emit_rms = dry_rms * (10.0 ** (-self._atten_floor_db / 20.0))
                if emit_rms < min_emit_rms and dry_rms > 1e-9:
                    emit = dry * (min_emit_rms / dry_rms)
        self._prev_dry_frame = new_dry
        return emit


def select_denoiser(prefer: str = "auto",
                    block_size: int = 480) -> Denoiser:
    """
    Returns a Denoiser. ``block_size`` is honoured for CPU; for Hailo the
    native DTLN hop (128) is used regardless because the HEFs are compiled
    for that. The pipeline reads ``denoiser.block_size`` at runtime to
    re-buffer between AEC and DNS, so this asymmetry is safe.
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        d = CpuDenoiser(block_size=block_size)
        logger.info(f"Denoiser: {d.name} (forced cpu, block={d.block_size})")
        return d
    if prefer == "hailo":
        d = HailoDenoiser()  # native 128
        logger.info(f"Denoiser: {d.name} (forced hailo, block={d.block_size})")
        return d
    if prefer != "auto":
        raise ValueError(f"unknown prefer={prefer!r}")
    probe = probe_hailo()
    if probe.available:
        try:
            d = HailoDenoiser()
            logger.info(f"Denoiser: {d.name} (auto -> hailo, block={d.block_size})")
            return d
        except Exception as e:
            logger.warning(f"Denoiser hailo init failed: {e}; falling back to cpu")
    d = CpuDenoiser(block_size=block_size)
    logger.info(
        f"Denoiser: {d.name} (auto -> cpu, hailo: {probe.reason}, "
        f"block={d.block_size})"
    )
    return d
