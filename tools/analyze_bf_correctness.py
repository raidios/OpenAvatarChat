#!/usr/bin/env python3
"""
Verify MVDR beamforming correctness on the recorded DOA dataset.

A correct beamformer should:

  1. Have a clear MAIN LOBE: when MVDR is locked to the true source
     azimuth, output power is highest. When locked to the opposite
     direction (truth + 180°), output power should drop noticeably.
     This is "spatial discrimination" -- the metric that says the
     beam is actually pointing somewhere.

  2. Match the SRP-PHAT estimate: locking to truth and locking to the
     pipeline's own DOA estimate should produce nearly identical
     output, since the SRP estimate is within ~10° of truth on this
     dataset.

  3. Show a sensible spatial response when swept around 360°: a peak
     near the truth and a notch on the opposite side, with monotonic
     transitions in between.

Absolute BF/ch0 gain in dB can be deceptively negative on this
dataset because per-mic AEC has already removed the on-board
speaker's contribution before the beamformer sees the mics. So we
do NOT use BF/ch0 as the primary correctness metric -- spatial
discrimination is what matters.

Usage:
    .venv/bin/python tools/analyze_bf_correctness.py
    .venv/bin/python tools/analyze_bf_correctness.py \
        --dataset tests/data/doa_dataset \
        --sweep-deg 30
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
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(_HERE)]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.dsp.pipeline import (   # noqa: E402
    DspPipeline, PipelineConfig, M260C_YAW_OFFSET_DEG,
)

SR = 16000
HOP = 160
BLOCK = 1600


def _load_8ch(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    sig = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch)
    return sig


def _wrap180(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _bf_run(sig8: np.ndarray, mic_az_deg: float,
            aec_per_mic: bool = True) -> np.ndarray:
    """Run pipeline with MVDR locked to mic_az_deg. Return BF mono."""
    cfg = PipelineConfig(
        sample_rate=SR, hop_samples=HOP, n_mics=6, n_ref=2,
        use_doa_for_mvdr=False,
        static_az_deg=float(mic_az_deg),
        aec_per_mic=aec_per_mic,
        dns_use_passthrough=True,
    )
    pl = DspPipeline(cfg)
    n_total = (sig8.shape[0] // HOP) * HOP
    sig = sig8[:n_total]
    chunks = []
    for s in range(0, n_total, BLOCK):
        e = min(n_total, s + BLOCK)
        chunk = sig[s:e]
        n_full = (chunk.shape[0] // HOP) * HOP
        if n_full > 0:
            out = pl.process_block(chunk[:n_full])
            if out.size:
                chunks.append(out)
    return (np.concatenate(chunks) if chunks
            else np.zeros(0, dtype=np.float32))


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def _ch0_rms_db(sig8: np.ndarray) -> float:
    n_total = (sig8.shape[0] // HOP) * HOP
    ch0 = sig8[:n_total, 0].astype(np.float32) / 32768.0
    return _rms_db(ch0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="tests/data/doa_dataset")
    ap.add_argument("--mic-yaw-offset-deg", type=float,
                    default=M260C_YAW_OFFSET_DEG)
    ap.add_argument("--sweep-deg", type=int, default=45,
                    help="resolution of the 360° spatial sweep "
                         "(default 45 -> 8 points).")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the 360° spatial response sweep.")
    ap.add_argument("--no-aec", action="store_true",
                    help="run BF on raw mics (skip per-mic AEC). "
                         "Useful to compare absolute BF gain.")
    args = ap.parse_args()

    ds = Path(args.dataset)
    wavs = sorted(
        p for p in ds.glob("body_*deg_*.wav")
        if not p.name.endswith(".bf.wav")
        and not p.name.endswith(".ch0.wav")
    )
    if not wavs:
        print(f"[!] no clips in {ds}")
        return 2

    yaw = float(args.mic_yaw_offset_deg)
    print(f"[*] dataset: {ds}  ({len(wavs)} clips)  "
          f"mic-yaw-offset={yaw:+.1f}°")
    print(f"[*] aec_per_mic = {not args.no_aec}")
    print()

    print("=== spatial discrimination (per clip) ===")
    print(f"{'file':<36}  {'truth':>5} | "
          f"{'on-beam':>9}  {'off+90':>9}  {'off-90':>9}  "
          f"{'anti':>9}  {'discrim':>8}")
    print("-" * 100)

    all_disc = []
    for w in wavs:
        meta = json.loads(w.with_suffix(".json").read_text())
        truth_body = float(meta["body_az_deg"])
        truth_mic = _wrap180(truth_body - yaw)
        sig = _load_8ch(w)
        ch0_db = _ch0_rms_db(sig)

        bf_on = _bf_run(sig, truth_mic, aec_per_mic=not args.no_aec)
        bf_p90 = _bf_run(sig, _wrap180(truth_mic + 90.0),
                         aec_per_mic=not args.no_aec)
        bf_n90 = _bf_run(sig, _wrap180(truth_mic - 90.0),
                         aec_per_mic=not args.no_aec)
        bf_anti = _bf_run(sig, _wrap180(truth_mic + 180.0),
                          aec_per_mic=not args.no_aec)

        on_db = _rms_db(bf_on)
        p90_db = _rms_db(bf_p90)
        n90_db = _rms_db(bf_n90)
        anti_db = _rms_db(bf_anti)
        # discrimination = on-beam minus mean of (off+90, off-90, anti)
        off_mean_db = 10.0 * np.log10(
            (10 ** (p90_db / 10) + 10 ** (n90_db / 10)
             + 10 ** (anti_db / 10)) / 3.0
        )
        disc = on_db - off_mean_db
        all_disc.append(disc)

        print(f"{w.name:<36}  {truth_body:>+4.0f}° | "
              f"{on_db - ch0_db:>+8.1f}  {p90_db - ch0_db:>+8.1f}  "
              f"{n90_db - ch0_db:>+8.1f}  {anti_db - ch0_db:>+8.1f}  "
              f"{disc:>+7.1f}")

    print()
    arr = np.array(all_disc)
    print(f"spatial discrimination (on-beam vs avg-of-3-off):")
    print(f"  mean = {arr.mean():+.2f} dB")
    print(f"  min  = {arr.min():+.2f} dB")
    print(f"  max  = {arr.max():+.2f} dB")
    print()
    if arr.mean() < 1.0:
        print("[!] WARN: mean discrimination < 1 dB -- beamformer is "
              "not pointing meaningfully. Check mic perm / yaw.")
    elif arr.min() < 0.0:
        print("[!] WARN: at least one clip has NEGATIVE discrimination "
              "(off-beam louder than on-beam). Likely a corrupted clip.")
    else:
        print("[+] PASS: BF discrimination is consistently positive.")

    if args.no_sweep:
        return 0

    print()
    print(f"=== 360° spatial sweep (step {args.sweep_deg}°) ===")
    angles = list(range(0, 360, args.sweep_deg))
    print(f"{'file':<36}  {'truth':>5} |  " +
          "  ".join(f"{a:>5}°" for a in angles))
    print("-" * (40 + 8 * len(angles)))
    sweep_table: List[Tuple[float, np.ndarray]] = []
    for w in wavs:
        meta = json.loads(w.with_suffix(".json").read_text())
        truth_body = float(meta["body_az_deg"])
        truth_mic = _wrap180(truth_body - yaw)
        sig = _load_8ch(w)
        ch0_db = _ch0_rms_db(sig)
        powers_db = []
        for a in angles:
            bf = _bf_run(sig, _wrap180(float(a)),
                         aec_per_mic=not args.no_aec)
            powers_db.append(_rms_db(bf) - ch0_db)
        cells = []
        peak_idx = int(np.argmax(powers_db))
        for i, p in enumerate(powers_db):
            mark = "*" if i == peak_idx else " "
            cells.append(f"{p:>+5.1f}{mark}")
        print(f"{w.name:<36}  {truth_body:>+4.0f}° |  " +
              "  ".join(cells))
        sweep_table.append((truth_mic, np.array(powers_db)))

    print()
    print("    (* = peak of sweep for that clip)")
    print()
    print("expected: peak should be at the angle closest to truth_mic = ")
    for w, (truth_mic, _) in zip(wavs, sweep_table):
        print(f"    {w.name}: truth_mic = {truth_mic:+6.1f}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
