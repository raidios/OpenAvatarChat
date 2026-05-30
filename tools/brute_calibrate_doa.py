#!/usr/bin/env python3
"""Search the 12 physically valid channel→position permutations for a
6-mic ring (CCW/CW × 6 rotations) and find the (perm, yaw-offset) that
minimises body-frame DOA error against labelled ground-truth clips.

The ring radius is taken as 35 mm (M260C spec). For each candidate
perm we run the production pipeline once per reliable clip, read the
dominant-source mic-frame azimuth, then grid-search yaw_offset over
[-180, +180) at 0.5° resolution.

Usage:
  .venv/bin/python tools/brute_calibrate_doa.py
  .venv/bin/python tools/brute_calibrate_doa.py \
      --reliable -90 0 90 180 \
      --include-approx
  .venv/bin/python tools/brute_calibrate_doa.py --radius 0.035

The default uses the 4 cardinal angles only (which the user
confirmed are precisely measured); pass --include-approx to add the
diagonals (which are eyeballed).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_HERE)]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.dsp.pipeline import (   # noqa: E402
    DspPipeline, PipelineConfig,
)

SR = 16000
N_MICS = 6


def _ring_perms() -> List[Tuple[str, int, Tuple[int, ...]]]:
    """The 12 channel→position permutations consistent with a 6-mic ring.

    A perm specifies which raw channel sits at logical position k.
    After applying it (logical[k] = raw[perm[k]]), logical channels go
    CCW around the ring at 60° spacing, with logical-mic-0 at some
    physical angle determined by yaw_offset.
    """
    out = []
    for k in range(N_MICS):
        ccw = tuple((k + i) % N_MICS for i in range(N_MICS))
        out.append(("CCW", k, ccw))
        cw = tuple((k - i) % N_MICS for i in range(N_MICS))
        out.append(("CW", k, cw))
    return out


def _load_8ch(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch)


def _wrap180(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _abs_err(est: float, truth: float) -> float:
    return abs(_wrap180(est - truth))


def _compute_dominant_mic_az(
    sig8: np.ndarray, perm: Tuple[int, ...], radius_m: float,
) -> float:
    duration_ms = int(sig8.shape[0] * 1000 / SR)
    cfg = PipelineConfig(
        sample_rate=SR, hop_samples=160, n_mics=N_MICS, n_ref=2,
        array_radius_m=radius_m,
        mic_channel_perm=perm,
        aec_per_mic=False,         # skip AEC; faster & no echo in cal data
        use_doa_for_mvdr=True,
        wake_window_ms=max(duration_ms + 500, 2000),
        dns_use_passthrough=True,
    )
    pl = DspPipeline(cfg)
    hop = pl.cfg.hop_samples
    n_total = (sig8.shape[0] // hop) * hop
    sig = sig8[:n_total]
    block = 1600
    for s in range(0, n_total, block):
        e = min(n_total, s + block)
        chunk = sig[s:e]
        n_full = (chunk.shape[0] // hop) * hop
        if n_full > 0:
            pl.process_block(chunk[:n_full])
    dom = pl.dominant_doa(top_k_pct=50.0)
    return float(dom[0]) if dom is not None else float("nan")


def _best_offset(mic_azs: List[float], truths: List[float]
                 ) -> Tuple[float, float]:
    """body_az = mic_az + offset. Find offset minimising mean |err|."""
    grid = np.arange(-180.0, 180.0, 0.5)
    mic = np.array(mic_azs)
    truth = np.array(truths)
    body_grid = ((mic[None, :] + grid[:, None] + 180.0) % 360.0) - 180.0
    err_grid = np.abs(((body_grid - truth[None, :]) + 180.0) % 360.0
                      - 180.0)
    mean_err = err_grid.mean(axis=1)
    i = int(np.argmin(mean_err))
    return float(grid[i]), float(mean_err[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="tests/data/doa_dataset")
    ap.add_argument("--reliable", type=float, nargs="+",
                    default=[-90, 0, 90, 180])
    ap.add_argument("--include-approx", action="store_true",
                    help="also fit the 4 diagonal angles even if they "
                         "were eyeballed.")
    ap.add_argument("--radius", type=float, default=0.035,
                    help="ring radius in metres (default 0.035 = M260C "
                         "manufacturer spec).")
    args = ap.parse_args()

    ds = Path(args.dataset)
    wavs = sorted(
        p for p in ds.glob("body_*deg_*.wav")
        if not p.name.endswith(".bf.wav")
        and not p.name.endswith(".ch0.wav")
    )

    truths_to_use = set(args.reliable)
    if args.include_approx:
        truths_to_use |= {-135.0, -45.0, 45.0, 135.0}

    clips = []
    extra = []
    for w in wavs:
        meta = json.loads(w.with_suffix(".json").read_text())
        truth = float(meta["body_az_deg"])
        sig = _load_8ch(w)
        if truth in truths_to_use:
            clips.append((truth, sig, w.name))
        else:
            extra.append((truth, sig, w.name))
    if len(clips) < 2:
        print("[!] need at least 2 reliable clips")
        return 2
    print(f"[*] fitting on {len(clips)} clips: "
          f"truths = {sorted(c[0] for c in clips)}")
    if extra:
        print(f"[*] (will also evaluate on "
              f"{sorted(c[0] for c in extra)} but not include in fit)")
    print(f"[*] radius = {args.radius * 1000:.1f} mm")

    perms = _ring_perms()
    t0 = time.monotonic()
    results = []
    print(f"[*] testing {len(perms)} ring permutations...")
    for i, (sense, k, perm) in enumerate(perms):
        mic_azs = [
            _compute_dominant_mic_az(sig, perm, args.radius)
            for _, sig, _name in clips
        ]
        if any(np.isnan(mic_azs)):
            continue
        truths = [t for t, _, _ in clips]
        off, mean_err = _best_offset(mic_azs, truths)
        results.append((mean_err, off, sense, k, perm, mic_azs))
        print(f"    [{i+1:2d}/{len(perms)}] {sense} rot={k}  "
              f"perm={perm}  -> err={mean_err:.2f}°  "
              f"yaw_off={off:+.1f}°")
    print(f"[*] elapsed {time.monotonic() - t0:.1f}s")

    results.sort()
    print()
    print("=== top 5 ===")
    print(f"{'rank':>4}  {'err':>6}  {'sense':>5}  {'rot':>3}  "
          f"{'yaw':>7}  perm")
    print("-" * 70)
    for i, (err, off, sense, k, perm, _m) in enumerate(results[:5]):
        print(f"{i+1:>4}  {err:>+5.2f}°  {sense:>5}  {k:>3}  "
              f"{off:>+6.1f}°  {perm}")

    if not results:
        print("FAIL: no valid combos")
        return 1

    best = results[0]
    print()
    print(f"=== BEST ===")
    print(f"  sense    = {best[2]}")
    print(f"  rotation = {best[3]}")
    print(f"  perm     = {best[4]}")
    print(f"  yaw_off  = {best[1]:+.1f}°  (body = mic_az + yaw_off)")
    print(f"  err      = {best[0]:.2f}° (mean abs, on fit clips)")

    truths_fit = [t for t, _, _ in clips]
    print()
    print(f"per-clip breakdown (fit set):")
    print(f"{'truth':>7}  {'mic':>9}  {'body':>9}  {'err':>5}")
    for truth, mic in zip(truths_fit, best[5]):
        body = _wrap180(mic + best[1])
        err = _abs_err(body, truth)
        print(f"{truth:>+6.0f}°  {mic:>+8.1f}°  {body:>+8.1f}°  "
              f"{err:>+4.0f}°")

    if extra:
        print()
        print(f"per-clip evaluation (held-out approx clips):")
        print(f"{'truth':>7}  {'mic':>9}  {'body':>9}  {'err':>5}")
        for truth, sig, _name in extra:
            mic = _compute_dominant_mic_az(sig, best[4], args.radius)
            body = _wrap180(mic + best[1])
            err = _abs_err(body, truth)
            print(f"{truth:>+6.0f}°  {mic:>+8.1f}°  {body:>+8.1f}°  "
                  f"{err:>+4.0f}°")

    print()
    print(f"To apply, edit audio_frontend/dsp/pipeline.py:")
    print(f"  M260C_MIC_PERM = {best[4]}")
    print(f"  M260C_RING_RADIUS_M = {args.radius:.4f}")
    print(f"and pass mic_yaw_offset_deg = {best[1]:+.1f} to "
          f"FarfieldAudioSource / record_doa_truth.py / test_doa_real.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
