#!/usr/bin/env python3
"""Generate a broadband test signal optimal for SRP-PHAT DOA validation.

We want:
  * spectrally flat over the SRP-PHAT band (200-4000 Hz default), so
    every PHAT bin contributes roughly equally to the steered power
    instead of being dominated by speech-formant peaks at a few hundred
    Hz that smear the cross-correlation lag.
  * stationary energy across time, so any frame is "loud" and the
    dominant-source DOA logic doesn't need to chase voice bursts.
  * mono playback through ``aplay`` at 16 kHz to match the M260C
    capture rate without resampling artefacts.
  * cosine fade-in/out so we don't pollute the AEC start/end with a
    click.

Usage:
    .venv/bin/python tools/gen_doa_test_signal.py \
        --out tests/data/doa_test_noise.wav --duration 5

Listen first; should sound like a soft "shh" without any tonal beating.
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np


def _bandlimited_noise(duration_s: float, sample_rate: int,
                       f_low: float, f_high: float, seed: int = 0
                       ) -> np.ndarray:
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n).astype(np.float32)
    F = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mask = (f >= f_low) & (f <= f_high)
    F[~mask] = 0.0
    out = np.fft.irfft(F, n).astype(np.float32)
    out /= np.max(np.abs(out)) + 1e-9
    return out


def _fade(x: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    n = x.size
    nf = max(1, int(round(fade_ms / 1000.0 * sample_rate)))
    nf = min(nf, n // 4)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nf, dtype=np.float32)))
    y = x.copy()
    y[:nf] *= ramp
    y[-nf:] *= ramp[::-1]
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default="tests/data/doa_test_noise.wav")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="length in seconds (default 5).")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--f-low", type=float, default=200.0)
    ap.add_argument("--f-high", type=float, default=6000.0,
                    help="upper edge in Hz. Keep below SR/2.")
    ap.add_argument("--peak-dbfs", type=float, default=-3.0,
                    help="target peak level in dBFS (default -3 for "
                         "headroom; set -1 for max loudness).")
    ap.add_argument("--fade-ms", type=float, default=200.0,
                    help="cosine fade-in/out length.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.f_high >= args.sample_rate / 2:
        print(f"[!] clamping f_high {args.f_high} -> "
              f"{args.sample_rate / 2 - 1}")
        args.f_high = args.sample_rate / 2 - 1

    sig = _bandlimited_noise(
        args.duration, args.sample_rate,
        args.f_low, args.f_high, seed=args.seed,
    )
    sig = _fade(sig, args.sample_rate, args.fade_ms)

    target = 10 ** (args.peak_dbfs / 20.0)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * target
    pcm = np.clip(sig * 32767.0, -32768, 32767).astype("<i2")

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(args.sample_rate)
        w.writeframes(pcm.tobytes())

    rms = float(np.sqrt(np.mean(sig.astype(np.float64) ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-9)
    print(f"[*] wrote {out_path}")
    print(f"    duration  : {args.duration:.2f} s")
    print(f"    band      : {args.f_low:.0f} - {args.f_high:.0f} Hz")
    print(f"    peak      : {args.peak_dbfs:+.1f} dBFS")
    print(f"    rms       : {rms_db:+.1f} dBFS")
    print(f"    samples   : {pcm.size}")
    print(f"    fade      : {args.fade_ms:.0f} ms cos")
    print()
    print(f"play with: aplay -q {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
