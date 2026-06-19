#!/usr/bin/env python3
"""
DOA accuracy regression on a directory of ground-truth 8-channel
recordings, plus side-by-side beamforming output for ear-level A/B.

Inputs are wav files produced by ``tools/record_doa_truth.py``: each
file is named ``body_<az>deg_<label>.wav`` (8 ch, 16 kHz int16) and
has a sidecar JSON with the body azimuth. We feed each clip through
the production ``DspPipeline`` offline twice: once dynamically to
get the streaming SRP-PHAT trace plus a one-shot ``dominant-source``
DOA over the loudest top-K% of frames, and once with MVDR locked to
that dominant DOA to produce the beamformed mono output. No KWS or
VAD assumption is made; the recording just needs to be dominated by
a single loud source at the labelled azimuth.

For each input ``body_p045deg_me.wav`` the test (re)writes:

  body_p045deg_me.bf.wav      AEC + MVDR mono, beam locked to
                              estimated DOA   <-- what production
                              produces after wake
  body_p045deg_me.ch0.wav     raw ch0 single-mic baseline, mono
                              <-- A/B against .bf

Listen to ``ch0.wav`` and ``bf.wav`` side by side to confirm
beamforming gain and noise rejection at the labelled angle.

Pass criteria are loose by default because we don't yet know how
clean the recording environment is. Tighten later as we collect more
data.

Usage:
  .venv/bin/python tests/test_doa_real.py
  .venv/bin/python tests/test_doa_real.py \
      --dataset tests/data/doa_dataset \
      --max-stream-err 25 --max-wake-err 12

Returns 0 on PASS, 1 on FAIL, 2 if the dataset directory is missing
or empty (so a fresh checkout doesn't break CI).
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_HERE)]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.dsp.offline import (   # noqa: E402
    analyze_clip, write_mono_wav,
)

SR = 16000


def _load_8ch(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if n_ch != 8:
        raise ValueError(f"{path.name}: expected 8 channels, got {n_ch}")
    arr = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch)
    return arr.astype(np.int16), sr


def _read_meta(path: Path) -> dict:
    json_path = path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    # Fallback: parse body_<az>deg_<label>.wav
    name = path.stem
    parts = name.split("_")
    body_az = None
    for tok in parts:
        if tok.endswith("deg"):
            try:
                # tokens like "p045deg" or "n090deg" or "+45deg" / "-90deg"
                core = tok[:-3]
                if core.startswith("p"):
                    body_az = float(core[1:])
                elif core.startswith("n"):
                    body_az = -float(core[1:])
                else:
                    body_az = float(core)
                break
            except ValueError:
                continue
    if body_az is None:
        raise ValueError(
            f"{path.name}: cannot parse body azimuth from filename and "
            f"no sidecar JSON found."
        )
    return {"body_az_deg": body_az}


def _wrap180(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _abs_err(est: float, truth: float) -> float:
    return abs(_wrap180(est - truth))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="tests/data/doa_dataset")
    ap.add_argument("--max-stream-err", type=float, default=25.0,
                    help="mean abs error gate for streaming DOA (deg).")
    ap.add_argument("--max-dominant-err", type=float, default=12.0,
                    help="mean abs error gate for dominant-source DOA (deg).")
    ap.add_argument("--dominant-top-k-pct", type=float, default=50.0,
                    help="percentage of loudest hops kept when computing "
                         "the dominant-source DOA.")
    ap.add_argument("--mic-yaw-offset-deg", type=float, default=30.0,
                    help="overrides the offset baked into the dataset's "
                         "JSON sidecar (only when the JSON is missing).")
    ap.add_argument("--no-write-bf", action="store_true",
                    help="skip writing per-clip .bf.wav / .ch0.wav files. "
                         "By default, every regression run refreshes them "
                         "in place so you can ear-level A/B raw vs BF.")
    args = ap.parse_args()

    ds = Path(args.dataset)
    wavs = sorted(
        p for p in ds.glob("body_*deg_*.wav")
        if not p.name.endswith(".bf.wav")
        and not p.name.endswith(".ch0.wav")
    )
    if not wavs:
        print(f"[!] no recordings found in {ds}/")
        print(f"    record some first:")
        print(f"    .venv/bin/python tools/record_doa_truth.py "
              f"--body-az 0 --duration 6 --label me")
        return 2

    print(f"[*] dataset: {ds}/  ({len(wavs)} files)  "
          f"(dominant top-{args.dominant_top_k_pct:.0f}%)")
    print()
    header = (f"{'file':<38} {'truth':>7} | "
              f"{'stream':>9} {'err':>5} | "
              f"{'dominant':>9} {'err':>5} {'pk':>5} | "
              f"{'bf-ch0':>7}")
    print(header)
    print("-" * len(header))

    stream_errs: List[float] = []
    dom_errs: List[float] = []
    dom_misses = 0
    bf_gains_db: List[float] = []

    for wav in wavs:
        try:
            sig, sr = _load_8ch(wav)
            if sr != SR:
                print(f"  [{wav.name}] FAIL: sr {sr} != {SR}")
                continue
            meta = _read_meta(wav)
            truth = float(meta["body_az_deg"])
            offset = float(meta.get("mic_yaw_offset_deg",
                                    args.mic_yaw_offset_deg))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{wav.name}] FAIL: {exc!s}")
            continue

        try:
            res = analyze_clip(
                sig, mic_yaw_offset_deg=offset, sample_rate=SR,
                dominant_top_k_pct=args.dominant_top_k_pct,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{wav.name}] FAIL pipeline: {exc!s}")
            continue

        if not args.no_write_bf:
            bf_path = wav.with_suffix(".bf.wav")
            ch0_path = wav.with_suffix(".ch0.wav")
            try:
                write_mono_wav(bf_path, res.bf_mono, sample_rate=SR)
                write_mono_wav(ch0_path, res.ch0_mono, sample_rate=SR)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{wav.name}] WARN bf write: {exc!s}")

        rms_ch0 = float(np.sqrt(np.mean(res.ch0_mono ** 2) + 1e-12))
        rms_bf = float(np.sqrt(np.mean(res.bf_mono ** 2) + 1e-12)) \
            if res.bf_mono.size else 0.0
        bf_db = 20.0 * np.log10((rms_bf + 1e-12) / (rms_ch0 + 1e-12))
        bf_gains_db.append(bf_db)

        s_err = _abs_err(res.streaming_body_doa_deg, truth)
        stream_errs.append(s_err)
        short = wav.name[:36] + ".." if len(wav.name) > 38 else wav.name
        if res.dominant_body_doa_deg is None:
            dom_misses += 1
            print(f"{short:<38} {truth:>+6.1f}° | "
                  f"{res.streaming_body_doa_deg:>+8.1f}° "
                  f"{s_err:>+4.0f}° | "
                  f"{'(none)':>9} {'-':>5} {'-':>5} | "
                  f"{bf_db:>+5.1f}dB")
        else:
            d_err = _abs_err(res.dominant_body_doa_deg, truth)
            dom_errs.append(d_err)
            print(f"{short:<38} {truth:>+6.1f}° | "
                  f"{res.streaming_body_doa_deg:>+8.1f}° "
                  f"{s_err:>+4.0f}° | "
                  f"{res.dominant_body_doa_deg:>+8.1f}° "
                  f"{d_err:>+4.0f}° "
                  f"{res.dominant_peak:>5.2f} | "
                  f"{bf_db:>+5.1f}dB")

    print()
    if stream_errs:
        s_mean = float(np.mean(stream_errs))
        s_max = float(np.max(stream_errs))
        s_p90 = float(np.percentile(stream_errs, 90))
        print(f"streaming DOA  : mean={s_mean:.1f}°  p90={s_p90:.1f}°  "
              f"max={s_max:.1f}°  (n={len(stream_errs)})")
    else:
        s_mean = s_max = s_p90 = float("nan")
        print("streaming DOA  : no data")
    if dom_errs:
        d_mean = float(np.mean(dom_errs))
        d_max = float(np.max(dom_errs))
        d_p90 = float(np.percentile(dom_errs, 90))
        print(f"dominant DOA   : mean={d_mean:.1f}°  p90={d_p90:.1f}°  "
              f"max={d_max:.1f}°  (n={len(dom_errs)}, "
              f"misses={dom_misses})")
    else:
        d_mean = d_max = d_p90 = float("nan")
        print(f"dominant DOA   : ALL MISSED ({dom_misses})")
    if bf_gains_db:
        print(f"BF gain (RMS bf - RMS ch0): "
              f"mean={float(np.mean(bf_gains_db)):+.1f} dB  "
              f"min={float(np.min(bf_gains_db)):+.1f} dB  "
              f"max={float(np.max(bf_gains_db)):+.1f} dB")
    if not args.no_write_bf:
        print(f"\nBeamformed mono and ch0 baseline written next to each "
              f"input wav. Listen with e.g.:")
        sample_name = wavs[0].stem
        print(f"  aplay {ds}/{sample_name}.ch0.wav   # raw single mic")
        print(f"  aplay {ds}/{sample_name}.bf.wav    # beamformed mono")

    passed = True
    if not stream_errs:
        print("FAIL: no streaming results")
        passed = False
    elif s_mean > args.max_stream_err:
        print(f"FAIL: streaming mean error {s_mean:.1f}° "
              f"> {args.max_stream_err:.1f}°")
        passed = False
    if not dom_errs:
        print("FAIL: dominant-source DOA produced no estimates")
        passed = False
    elif d_mean > args.max_dominant_err:
        print(f"FAIL: dominant-source mean error {d_mean:.1f}° "
              f"> {args.max_dominant_err:.1f}°")
        passed = False

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
