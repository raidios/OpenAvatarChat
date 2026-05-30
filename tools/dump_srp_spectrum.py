#!/usr/bin/env python3
"""Dump the SRP-PHAT spatial spectrum for each labelled clip.

Helps decide whether bad DOA results are because (a) the perm/offset
is wrong (peak is consistent across clips but in the wrong frame) or
(b) the data is genuinely noisy/multi-peaked (spectrum has no clear
peak or has multiple competing peaks).

For each reliable clip we:
  1. apply each candidate ring perm
  2. compute SRP-PHAT once over the full clip (loudest 50% frames)
  3. print the top-5 azimuths in mic frame plus the peak/mean ratio

If the data is clean, the truth angles 90° apart should produce
mic-frame peaks 90° apart (modulo a constant rotation = yaw_offset
and a possible sign flip = sense). If the peaks are scattered
randomly, the recording itself is not informative.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.dsp.geometry import RingArray   # noqa: E402
from audio_frontend.dsp.srp_phat import (           # noqa: E402
    SrpPhat, SrpPhatConfig,
)

SR = 16000
N_MICS = 6


def _load_8ch(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch).astype(
        np.float32) / 32768.0


def _ring_perms() -> List[Tuple[str, int, Tuple[int, ...]]]:
    out = []
    for k in range(N_MICS):
        out.append(("CCW", k, tuple((k + i) % N_MICS for i in range(N_MICS))))
        out.append(("CW", k, tuple((k - i) % N_MICS for i in range(N_MICS))))
    return out


def _select_loud_frames(mics: np.ndarray, hop: int = 160,
                        top_k_pct: float = 50.0) -> np.ndarray:
    n_chunks = mics.shape[0] // hop
    if n_chunks == 0:
        return mics
    chunked = mics[: n_chunks * hop].reshape(n_chunks, hop, mics.shape[1])
    rms = np.sqrt(np.mean(chunked ** 2, axis=(1, 2)) + 1e-12)
    thr = np.percentile(rms, 100.0 - top_k_pct)
    keep = rms >= thr
    keep_full = np.repeat(keep, hop)
    n = min(len(keep_full), n_chunks * hop)
    return mics[:n][keep_full[:n]]


def _srp_full_clip(mics_loud: np.ndarray, radius_m: float,
                   f_low: float = 200.0, f_high: float = 4000.0):
    array = RingArray(n_mics=N_MICS, radius_m=radius_m)
    cfg = SrpPhatConfig(sample_rate=SR, n_fft=1024, hop=512,
                        az_grid_deg=2,
                        f_low=f_low, f_high=f_high,
                        smoothing_alpha=1.0)   # no smoothing for offline
    srp = SrpPhat(array, cfg)
    n_fft = cfg.n_fft
    n = mics_loud.shape[0]
    if n < n_fft:
        return None
    # accumulate P over many frames so peak is sharper
    n_chunks = (n - n_fft) // (n_fft // 2) + 1
    P_total = np.zeros(len(srp.az_grid_rad))
    for i in range(n_chunks):
        s = i * (n_fft // 2)
        frame = mics_loud[s:s + n_fft]
        if frame.shape[0] < n_fft:
            break
        _, _, P = srp.estimate(frame)
        P_total += P
    grid = np.degrees(srp.az_grid_rad)
    grid_signed = ((grid + 180.0) % 360.0) - 180.0
    return grid_signed, P_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="tests/data/doa_dataset")
    ap.add_argument("--radius", type=float, default=0.035)
    ap.add_argument("--reliable-only", action="store_true",
                    help="only dump the 4 cardinal angles.")
    ap.add_argument("--perm", type=int, nargs="+", default=None,
                    help="apply this channel perm before SRP. Default "
                         "= identity (raw SDK channel order).")
    ap.add_argument("--f-low", type=float, default=200.0)
    ap.add_argument("--f-high", type=float, default=4000.0)
    args = ap.parse_args()

    ds = Path(args.dataset)
    wavs = sorted(
        p for p in ds.glob("body_*deg_*.wav")
        if not p.name.endswith(".bf.wav")
        and not p.name.endswith(".ch0.wav")
    )
    reliable = {-90.0, 0.0, 90.0, 180.0}

    perm = tuple(args.perm) if args.perm else (0, 1, 2, 3, 4, 5)
    print(f"[*] using perm = {perm}, radius = {args.radius * 1000:.1f} mm")
    print()

    for w in wavs:
        meta = json.loads(w.with_suffix(".json").read_text())
        truth = float(meta["body_az_deg"])
        if args.reliable_only and truth not in reliable:
            continue
        sig = _load_8ch(w)
        mics = sig[:, :N_MICS][:, list(perm)]
        loud = _select_loud_frames(mics)
        out = _srp_full_clip(loud, args.radius,
                              f_low=args.f_low, f_high=args.f_high)
        if out is None:
            print(f"  [{w.name}] SKIP (too short)")
            continue
        grid, P = out
        # Top 5 peaks (with at least 30° between them so we get distinct lobes)
        order = np.argsort(P)[::-1]
        peaks = []
        for idx in order:
            if all(abs(((grid[idx] - p[0]) + 180) % 360 - 180) > 30
                   for p in peaks):
                peaks.append((float(grid[idx]), float(P[idx])))
            if len(peaks) >= 5:
                break
        peak_v = peaks[0][1]
        mean_v = float(np.mean(P))
        ratio = peak_v / (mean_v + 1e-12)
        marker = "" if truth not in reliable else " (reliable)"
        print(f"=== {w.name}  truth={truth:+.0f}°{marker} ===")
        print(f"    peak/mean = {ratio:.2f}")
        for az, p in peaks:
            print(f"    az = {az:+7.1f}°    P = {p:.3f}    "
                  f"P/peak = {p / peak_v:.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
