#!/usr/bin/env python3
"""
DNS "buzz" closed-loop regression test.

Background:
  Hailo DTLN v7 emits audible quantisation buzz on low-energy frames where
  the LSTM mask estimate is noisy and gets normalised back to non-zero
  amplitude. The denoiser ships with a silence gate + attenuation floor
  to suppress this; this test verifies they keep working as we tune
  parameters and rebuild HEFs.

What it does:
  1. Loads a known input wav (default: tests/data/cherry.wav, the same
     voice clip we use for AEC isolation).
  2. Pads with silence head/tail and inserts silence gaps (so we have
     unambiguous silence frames to measure buzz on).
  3. Runs the input offline through each enabled backend at its native
     block size:
         - passthrough           (control)
         - hailo / dtln_v7       (gate ON, default)
         - hailo / dtln_v7 NO-GATE  (gate disabled, exposes raw buzz)
         - cpu / deepfilternet   (when available)
     and saves the output wavs.
  4. Classifies frames as silence vs voice using the *input dry* RMS,
     then for each backend reports:
         - silence: output RMS p50/p99, spectral peak/mean ratio
                    (peakedness; high = tonal buzz),
                    "buzz energy" = output - input residual RMS.
         - voice  : output/input RMS ratio (preservation),
                    spectral correlation with input.
  5. PASS if hailo (gated) silence p99 RMS and buzz energy stay below
     thresholds, AND voice preservation is reasonable.

Usage:
  .venv/bin/python tests/test_dns_buzz.py                   # full sweep
  .venv/bin/python tests/test_dns_buzz.py --backend hailo   # one backend
  .venv/bin/python tests/test_dns_buzz.py --silence-rms 0.0 # disable gate
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# tests/ contains a local ``unittest/`` package that shadows stdlib's
# unittest when Python auto-injects the script's parent dir at the head
# of sys.path (which breaks scipy/numpy.testing imports). Strip it.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_HERE)]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.backends.denoiser import (   # noqa: E402
    CpuDenoiser, HailoDenoiser, _PassthroughDenoiser,
)

SR = 16000


@dataclass
class BackendResult:
    name: str
    block_size: int
    out: np.ndarray         # float32, same length as input (padded if needed)
    silence_p50_rms: float
    silence_p99_rms: float
    silence_peak_to_mean: float
    silence_buzz_rms: float
    # Transition = the 100 ms after each voice->silence edge and the
    # 100 ms before each silence->voice edge. This is where "tail buzz"
    # is actually audible — the steady-state silence after that often
    # measures fine because the gate has had time to fully damp.
    transition_p50_rms: float
    transition_p99_rms: float
    transition_peak_to_mean: float
    voice_preservation: float
    voice_spectral_corr: float
    voice_attenuation_db: float


def _load_input(path: Path, sr: int = SR) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    data, in_sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if in_sr != sr:
        g = gcd(in_sr, sr)
        data = resample_poly(data, sr // g, in_sr // g).astype(np.float32)
    return data.astype(np.float32, copy=False)


def _build_test_signal(
    voice: np.ndarray,
    head_s: float = 1.0,
    tail_s: float = 1.0,
    gap_s: float = 0.5,
    n_voice_chunks: int = 2,
    voice_peak: float = 0.3,
    sr: int = SR,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Construct silence + (voice + silence) pattern.

    Returns ``(signal, silence_intervals, voice_intervals)`` where each
    interval is ``(start_sample, end_sample)``. Knowing the ground-truth
    boundaries is much cleaner than RMS-classifying frames.
    """
    if len(voice) == 0:
        raise ValueError("empty voice clip")
    peak = float(np.max(np.abs(voice))) + 1e-9
    voice = voice * (voice_peak / peak)

    # Apply a short cosine ramp on every chunk's edges so that splicing
    # cherry.wav in the middle does not create abrupt cuts: those cuts
    # become high-frequency transients which the DTLN trains on, then
    # the OLA buffer lingers, and the buzz we measure is then partly
    # an artefact of the unnatural step rather than real production
    # buzz. 10 ms cosine fades make the splice points indistinguishable
    # from a natural voice tail.
    fade_n = int(0.010 * sr)
    chunk_len = max(1, len(voice) // max(1, n_voice_chunks))
    chunks = []
    for k in range(n_voice_chunks):
        s = k * chunk_len
        e = min(len(voice), s + chunk_len)
        c = voice[s:e].copy()
        if len(c) > 2 * fade_n:
            ramp = 0.5 * (1 - np.cos(
                np.linspace(0, np.pi, fade_n, dtype=np.float32)))
            c[:fade_n] *= ramp
            c[-fade_n:] *= ramp[::-1]
        chunks.append(c)
    head_n = int(head_s * sr)
    gap_n = int(gap_s * sr)
    tail_n = int(tail_s * sr)

    parts: List[np.ndarray] = [np.zeros(head_n, dtype=np.float32)]
    silence_intervals: List[Tuple[int, int]] = []
    voice_intervals: List[Tuple[int, int]] = []
    cursor = 0
    silence_intervals.append((cursor, cursor + head_n))
    cursor += head_n

    for i, c in enumerate(chunks):
        if i > 0:
            parts.append(np.zeros(gap_n, dtype=np.float32))
            silence_intervals.append((cursor, cursor + gap_n))
            cursor += gap_n
        parts.append(c)
        voice_intervals.append((cursor, cursor + len(c)))
        cursor += len(c)
    parts.append(np.zeros(tail_n, dtype=np.float32))
    silence_intervals.append((cursor, cursor + tail_n))

    sig = np.concatenate(parts).astype(np.float32, copy=False)
    return sig, silence_intervals, voice_intervals


def _pad_to_multiple(x: np.ndarray, m: int) -> np.ndarray:
    rem = len(x) % m
    if rem == 0:
        return x
    return np.concatenate([x, np.zeros(m - rem, dtype=np.float32)])


def _run_backend(name: str, signal: np.ndarray,
                 block_size: int,
                 process) -> Tuple[np.ndarray, float]:
    x = _pad_to_multiple(signal, block_size)
    out = np.zeros_like(x)
    t0 = time.monotonic()
    for s in range(0, len(x), block_size):
        out[s:s+block_size] = process(x[s:s+block_size])
    elapsed = time.monotonic() - t0
    rtf = elapsed / (len(x) / SR)
    print(f"  [{name:24s}] block={block_size:>4d}  "
          f"wall={elapsed:.2f}s  RTF={rtf:.2f}")
    return out[:len(signal)], rtf


def _frame_rms(x: np.ndarray, frame_len: int) -> np.ndarray:
    n = (len(x) // frame_len) * frame_len
    blocks = x[:n].reshape(-1, frame_len)
    return np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-20)


def _build_sample_mask(intervals: List[Tuple[int, int]],
                       length: int,
                       guard_ms: float = 50.0,
                       sr: int = SR) -> np.ndarray:
    """Boolean (length,) mask covering every interval, shrunk by
    ``guard_ms`` at each end. Used to extract the steady-state interior."""
    g = int(guard_ms / 1000.0 * sr)
    mask = np.zeros(length, dtype=bool)
    for s, e in intervals:
        s = max(0, s + g)
        e = min(length, e - g)
        if e > s:
            mask[s:e] = True
    return mask


def _build_transition_mask(silence_intervals: List[Tuple[int, int]],
                           voice_intervals: List[Tuple[int, int]],
                           length: int,
                           ola_skip_ms: float = 50.0,
                           transition_ms: float = 150.0,
                           sr: int = SR) -> np.ndarray:
    """Mask covering the post-OLA portion of each silence boundary.

    Skips the first ``ola_skip_ms`` after a voice end (DTLN's OLA
    buffer holds 4 hops = 32 ms of legitimate voice tail; allowing
    ~50 ms of grace gives the LSTM time to stop emitting non-zero
    output) and then captures the next ``transition_ms - ola_skip_ms``.
    Symmetric for the silence-before-voice side.

    What we expect to measure here for each backend:
      - passthrough              : 0 (silence is silence)
      - hailo gated              : ~0 once the gate has had a frame
                                    or two to detect silence
      - hailo without gate       : the steady quantisation buzz
                                    (~ -56 dBFS continuous)
    """
    a_skip = int(ola_skip_ms / 1000.0 * sr)
    a_end = int(transition_ms / 1000.0 * sr)
    if a_end <= a_skip:
        return np.zeros(length, dtype=bool)
    mask = np.zeros(length, dtype=bool)
    voice_set = set()
    for s, e in voice_intervals:
        voice_set.add(s); voice_set.add(e)
    for s, e in silence_intervals:
        if s in voice_set and s < length:
            mask[max(0, s + a_skip): min(length, s + a_end)] = True
        if e in voice_set and e > 0:
            mask[max(0, e - a_end): min(length, e - a_skip)] = True
    return mask


def _spectral_peak_to_mean(x: np.ndarray, frame_len: int,
                           min_rms: float = 1e-5) -> float:
    """Median across frames of |X[k]|.max / |X[k]|.mean (excluding DC).

    Frames with RMS below ``min_rms`` (essentially exact silence) are
    skipped — they have a degenerate spectrum (all zeros) that would
    yield 0/0 NaNs and contaminate the median. Keeping only frames that
    have *some* output amplitude is what we actually want to measure:
    "given the DNS emitted something audible, was it tonal buzz or
    well-distributed hiss?". Tonal buzz lifts a few bins to 50-500x,
    clean white-ish hiss is ~5-10x.
    """
    n = (len(x) // frame_len) * frame_len
    if n == 0:
        return float("nan")
    blocks = x[:n].reshape(-1, frame_len)
    block_rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-20)
    audible = block_rms > min_rms
    if not audible.any():
        return float("nan")
    win = np.hanning(frame_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(blocks[audible] * win, axis=1))
    spec_no_dc = spec[:, 1:]
    if spec_no_dc.size == 0:
        return float("nan")
    mean = spec_no_dc.mean(axis=1) + 1e-12
    peak = spec_no_dc.max(axis=1)
    return float(np.median(peak / mean))


def _spectral_corr_per_frame(a: np.ndarray, b: np.ndarray,
                             frame_len: int,
                             min_rms: float = 1e-4) -> float:
    n = min((len(a) // frame_len) * frame_len,
            (len(b) // frame_len) * frame_len)
    if n == 0:
        return float("nan")
    A = a[:n].reshape(-1, frame_len)
    B = b[:n].reshape(-1, frame_len)
    a_rms = np.sqrt(np.mean(A ** 2, axis=1) + 1e-20)
    b_rms = np.sqrt(np.mean(B ** 2, axis=1) + 1e-20)
    audible = (a_rms > min_rms) & (b_rms > min_rms)
    if not audible.any():
        return float("nan")
    A = A[audible]; B = B[audible]
    win = np.hanning(frame_len).astype(np.float32)
    Sa = np.abs(np.fft.rfft(A * win, axis=1))
    Sb = np.abs(np.fft.rfft(B * win, axis=1))
    Sa_c = Sa - Sa.mean(axis=1, keepdims=True)
    Sb_c = Sb - Sb.mean(axis=1, keepdims=True)
    num = (Sa_c * Sb_c).sum(axis=1)
    den = (np.sqrt((Sa_c ** 2).sum(axis=1) * (Sb_c ** 2).sum(axis=1))
           + 1e-12)
    rho = num / den
    return float(np.median(rho))


def _evaluate(name: str, block_size: int,
              dry: np.ndarray, out: np.ndarray,
              silence_intervals: List[Tuple[int, int]],
              voice_intervals: List[Tuple[int, int]]) -> BackendResult:
    frame_len = 320  # 20 ms analysis frames

    n = min(len(dry), len(out))
    dry = dry[:n]; out = out[:n]
    sil_idx = _build_sample_mask(silence_intervals, n, guard_ms=150.0)
    voi_idx = _build_sample_mask(voice_intervals, n, guard_ms=50.0)
    trans_idx = _build_transition_mask(silence_intervals, voice_intervals,
                                       n, transition_ms=100.0)

    if sil_idx.any():
        sil_dry = dry[sil_idx]
        sil_out = out[sil_idx]
        residual = sil_out - sil_dry
        buzz_rms = float(np.sqrt(np.mean(residual ** 2)) + 1e-12)
        rms_per_frame = _frame_rms(sil_out, frame_len)
        s_p50 = float(np.percentile(rms_per_frame, 50))
        s_p99 = float(np.percentile(rms_per_frame, 99))
        peak_to_mean = _spectral_peak_to_mean(sil_out, frame_len)
    else:
        s_p50 = s_p99 = buzz_rms = peak_to_mean = float("nan")

    if trans_idx.any():
        trans_out = out[trans_idx]
        # Transition windows are short (100 ms = 1600 samples) so
        # use a smaller analysis frame to avoid percentile rounding.
        t_frame = 160
        t_rms_per_frame = _frame_rms(trans_out, t_frame)
        if len(t_rms_per_frame) > 0:
            t_p50 = float(np.percentile(t_rms_per_frame, 50))
            t_p99 = float(np.percentile(t_rms_per_frame, 99))
        else:
            t_p50 = t_p99 = float("nan")
        t_peak = _spectral_peak_to_mean(trans_out, t_frame)
    else:
        t_p50 = t_p99 = t_peak = float("nan")

    if voi_idx.any():
        voi_dry = dry[voi_idx]
        voi_out = out[voi_idx]
        # Per-frame ratios (median = robust to onset/offset transients)
        v_out_rms = _frame_rms(voi_out, frame_len)
        v_dry_rms = _frame_rms(voi_dry, frame_len)
        n_frames = min(len(v_out_rms), len(v_dry_rms))
        ratios = v_out_rms[:n_frames] / (v_dry_rms[:n_frames] + 1e-12)
        preservation = float(np.clip(np.median(ratios), 0.0, 5.0))
        atten_db = -20.0 * np.log10(preservation + 1e-12)
        spec_corr = _spectral_corr_per_frame(voi_dry, voi_out, frame_len)
    else:
        preservation = atten_db = spec_corr = float("nan")

    return BackendResult(
        name=name,
        block_size=block_size,
        out=out.astype(np.float32, copy=False),
        silence_p50_rms=s_p50,
        silence_p99_rms=s_p99,
        silence_peak_to_mean=peak_to_mean,
        silence_buzz_rms=buzz_rms,
        transition_p50_rms=t_p50,
        transition_p99_rms=t_p99,
        transition_peak_to_mean=t_peak,
        voice_preservation=preservation,
        voice_spectral_corr=spec_corr,
        voice_attenuation_db=atten_db,
    )


def _save_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="tests/data/cherry.wav",
                    help="voice wav for the test stimulus.")
    ap.add_argument("--backend", type=str, default="all",
                    choices=["all", "passthrough", "cpu", "hailo",
                             "hailo-nogate"])
    ap.add_argument("--out", type=str,
                    default="tests/results/dns_buzz")
    ap.add_argument("--head-s", type=float, default=1.0)
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--gap-s", type=float, default=0.5)
    ap.add_argument("--n-voice-chunks", type=int, default=3,
                    help="how many voice chunks the input is split into. "
                         "Each gap between them is a guaranteed silence "
                         "segment (gap_s long) used for buzz measurement.")
    ap.add_argument("--voice-peak", type=float, default=0.3,
                    help="rescale the voice peak to this absolute amplitude "
                         "(matches the post-MVDR level the DNS sees).")
    ap.add_argument("--silence-rms", type=float, default=None,
                    help="override DTLN_SILENCE_RMS (set 0 to disable gate).")
    ap.add_argument("--atten-floor-db", type=float, default=None,
                    help="override DTLN_ATTEN_FLOOR_DB (set 0 to disable).")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"FAIL: input wav not found: {in_path}")
        return 2
    voice = _load_input(in_path)
    signal, silence_intervals, voice_intervals = _build_test_signal(
        voice,
        head_s=args.head_s,
        tail_s=args.tail_s,
        gap_s=args.gap_s,
        n_voice_chunks=args.n_voice_chunks,
        voice_peak=args.voice_peak,
    )
    print(f"[*] test stimulus: {in_path.name}, "
          f"{len(signal)/SR:.2f}s, peak={float(np.max(np.abs(signal))):.3f}")
    print(f"    head={args.head_s:.1f}s gap={args.gap_s:.1f}s "
          f"tail={args.tail_s:.1f}s n_voice={args.n_voice_chunks} "
          f"voice_peak={args.voice_peak:.2f}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_wav(out_dir / "input.wav", signal)

    backends_to_run: List[str] = []
    if args.backend == "all":
        backends_to_run = ["passthrough", "hailo-nogate", "hailo", "cpu"]
    else:
        backends_to_run = [args.backend]

    results: Dict[str, BackendResult] = {}
    # Hailo VDevice is single-instance per process; reuse one HailoDenoiser
    # across both gated/no-gate runs by toggling the gate on the fly.
    hailo_d: Optional[HailoDenoiser] = None

    def _get_hailo() -> HailoDenoiser:
        nonlocal hailo_d
        if hailo_d is None:
            hailo_d = HailoDenoiser(
                silence_rms=args.silence_rms,
                atten_floor_db=args.atten_floor_db,
            )
        return hailo_d

    for be in backends_to_run:
        try:
            if be == "passthrough":
                d = _PassthroughDenoiser(block_size=480, name="passthrough")
                out, _ = _run_backend(d.name, signal, d.block_size, d.process)
                r = _evaluate(d.name, d.block_size, signal, out, silence_intervals, voice_intervals)

            elif be == "cpu":
                d = CpuDenoiser(block_size=480, model="deepfilternet")
                out, _ = _run_backend(d.name, signal, d.block_size, d.process)
                r = _evaluate(d.name, d.block_size, signal, out, silence_intervals, voice_intervals)

            elif be == "hailo":
                d = _get_hailo()
                d.set_gate(silence_rms=(args.silence_rms
                                        if args.silence_rms is not None
                                        else 0.001),
                           atten_floor_db=(args.atten_floor_db
                                           if args.atten_floor_db is not None
                                           else 12.0))
                d.reset()
                label = d.name + " (gated)"
                out, _ = _run_backend(label, signal, d.block_size, d.process)
                r = _evaluate(label, d.block_size, signal, out, silence_intervals, voice_intervals)

            elif be == "hailo-nogate":
                d = _get_hailo()
                d.set_gate(silence_rms=0.0, atten_floor_db=0.0)
                d.reset()
                label = d.name + " (no gate)"
                out, _ = _run_backend(label, signal, d.block_size, d.process)
                r = _evaluate(label, d.block_size, signal, out, silence_intervals, voice_intervals)

            else:
                continue

            _save_wav(out_dir / f"out_{be}.wav", out)
            results[be] = r

        except Exception as exc:  # noqa: BLE001
            print(f"  [{be:24s}] FAIL: {exc!s}")

    if not results:
        print("FAIL: no backends ran")
        return 1

    def _fmt(v: float, w: int = 7, prec: int = 5) -> str:
        if v != v:  # nan
            return f"{'nan':>{w}}"
        return f"{v:>{w}.{prec}f}"

    print()
    print(f"{'backend':<28} {'sil p50':>8} {'sil p99':>8} {'buzz':>8} "
          f"{'p/m':>5} | {'tran p50':>8} {'tran p99':>8} {'p/m':>5} | "
          f"{'voice':>5} {'atten':>6} {'spec':>5}")
    print(f"{'':<28} {'rms':>8} {'rms':>8} {'rms':>8} "
          f"{'ratio':>5} | {'rms':>8} {'rms':>8} {'ratio':>5} | "
          f"{'keep':>5} {'dB':>6} {'corr':>5}")
    print(f"{'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 5}-+-"
          f"{'-' * 8} {'-' * 8} {'-' * 5}-+-"
          f"{'-' * 5} {'-' * 6} {'-' * 5}")
    for be, r in results.items():
        print(f"{r.name:<28} "
              f"{_fmt(r.silence_p50_rms, 8)} "
              f"{_fmt(r.silence_p99_rms, 8)} "
              f"{_fmt(r.silence_buzz_rms, 8)} "
              f"{_fmt(r.silence_peak_to_mean, 5, 1)} | "
              f"{_fmt(r.transition_p50_rms, 8)} "
              f"{_fmt(r.transition_p99_rms, 8)} "
              f"{_fmt(r.transition_peak_to_mean, 5, 1)} | "
              f"{_fmt(r.voice_preservation, 5, 3)} "
              f"{_fmt(r.voice_attenuation_db, 6, 2) if r.voice_attenuation_db != r.voice_attenuation_db else f'{r.voice_attenuation_db:>+6.2f}'} "
              f"{_fmt(r.voice_spectral_corr, 5, 3)}")

    def _le(x: float, t: float) -> bool:
        """nan-tolerant ``x <= t``: nan means 'no audible frames in this
        zone', which is a desirable outcome for silence/transition."""
        return (x != x) or x <= t

    passed = True
    if "hailo" in results:
        h = results["hailo"]
        sil_ok = (_le(h.silence_p99_rms, 1e-4)
                  and _le(h.silence_buzz_rms, 5e-4))
        trans_ok = (_le(h.transition_p99_rms, 0.01)
                    and _le(h.transition_peak_to_mean, 50.0))
        voice_ok = (h.voice_preservation > 0.3
                    and h.voice_spectral_corr > 0.5)
        if not sil_ok:
            print(f"FAIL: hailo gated steady-silence above threshold "
                  f"(p99={h.silence_p99_rms:.5f}, "
                  f"buzz={h.silence_buzz_rms:.5f})")
            passed = False
        if not trans_ok:
            print(f"FAIL: hailo gated post-OLA transition zone above "
                  f"threshold (p99={h.transition_p99_rms:.5f}, "
                  f"p/m={h.transition_peak_to_mean:.1f})")
            passed = False
        if not voice_ok:
            print(f"FAIL: hailo gated voice not preserved "
                  f"(keep={h.voice_preservation:.3f}, "
                  f"corr={h.voice_spectral_corr:.3f})")
            passed = False

    print()
    print(f"artefacts in {out_dir}/")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
