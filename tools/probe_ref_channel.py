#!/usr/bin/env python3
"""
Verify M260C reference channel correlation with playback audio.

Procedure:
  1. Generate a known probe signal (chirp 200~7000 Hz, 5 seconds).
  2. Concurrently:
     - play it on the host (Pi5) speakers via aplay
     - capture 8ch from the M260C board via LiveStream
  3. Trim capture to playback duration (with adb pipeline lead-in).
  4. Compute normalized cross-correlation between
     - mean(ch6, ch7) <-> probe signal           (expected > 0.5  if ref is wired)
     - mean(ch0..ch5)  <-> probe signal           (sanity, mics also pick it up)
  5. Print verdict + lag (ms).

Pass criteria (per plan stage-0 acceptance):
  ref correlation > 0.5  AND  mics correlation > 0.3

Usage:
  .venv/bin/python tools/probe_ref_channel.py [--duration 5] [--out /tmp/ref_probe]

Prereqs:
  - adb device visible (`adb devices`).
  - audio_server running on board (`tools/board_audio_mode.py status`).
  - host has aplay + audible speakers/headphones.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "M2_SDK" / "live_stream" / "host"))

from live_stream import LiveStream, SAMPLE_RATE  # noqa: E402

MIC_CH = 6
REF_CH_START = 6


def make_chirp(duration_s: float, fs: int = SAMPLE_RATE,
               f0: float = 200.0, f1: float = 7000.0) -> np.ndarray:
    """Logarithmic chirp 200 -> 7000 Hz, mono int16."""
    t = np.linspace(0.0, duration_s, int(fs * duration_s), endpoint=False)
    k = (f1 / f0) ** (1.0 / duration_s)
    phase = 2.0 * np.pi * f0 * (k**t - 1.0) / np.log(k)
    sig = 0.6 * np.sin(phase)
    fade = int(0.05 * fs)
    win = np.ones_like(sig)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    sig = sig * win
    return (sig * 32767).astype("<i2")


def write_mono_wav(path: str, samples: np.ndarray, fs: int = SAMPLE_RATE):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(samples.tobytes())


def normalized_xcorr_fft(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Compute peak normalized cross-correlation between a and b over their
    full overlap range using FFT convolution. Returns (peak_cosine, lag_samples).

    Positive lag means b leads a (b appears in a after lag samples).
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0, 0
    # full cross-correlation:  corr[k] = sum_n a[n+k] b[n]
    corr = fftconvolve(a, b[::-1], mode="full")
    # normalize by the energies of the overlapping windows for each lag.
    # exact normalization is expensive, so we use a signal-power approximation:
    # N = max(len(a), len(b)), and divide by sqrt(<a^2> <b^2>) * N.
    n = max(len(a), len(b))
    norm = np.sqrt(np.mean(a * a) * np.mean(b * b)) * n + 1e-12
    corr_norm = corr / norm
    # search around lags in (-len(a), len(b)); index = lag + len(b) - 1
    idx_peak = int(np.argmax(np.abs(corr_norm)))
    peak = float(corr_norm[idx_peak])
    lag = idx_peak - (len(b) - 1)
    return peak, lag


def play_in_thread(wav_path: str, started_evt: threading.Event,
                   done_evt: threading.Event):
    started_evt.set()
    try:
        subprocess.run(["aplay", "-q", wav_path], check=False)
    finally:
        done_evt.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0,
                    help="probe chirp duration in seconds (default 5)")
    ap.add_argument("--out", default="/tmp/ref_probe",
                    help="dump dir for chirp wav + capture wav + report")
    ap.add_argument("--lead-ms", type=int, default=300,
                    help="extra capture before playback start (default 300ms)")
    ap.add_argument("--tail-ms", type=int, default=500,
                    help="extra capture after playback end")
    ap.add_argument("--no-forward", action="store_true",
                    help="skip adb forward (assume already set up)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    chirp = make_chirp(args.duration)
    chirp_wav = out_dir / "chirp.wav"
    write_mono_wav(str(chirp_wav), chirp)
    print(f"[*] generated probe chirp -> {chirp_wav}  ({args.duration:.1f}s)")

    capture_secs = args.duration + args.lead_ms / 1000.0 + args.tail_ms / 1000.0
    n_frames_total = int(capture_secs * SAMPLE_RATE)

    print("[*] connecting to LiveStream...")
    started = threading.Event()
    done = threading.Event()

    with LiveStream(ensure_forward=not args.no_forward) as ls:
        # First drain ~lead_ms so the playback aligns with mid-capture
        lead_frames = int(args.lead_ms / 1000.0 * SAMPLE_RATE)
        _ = ls.read_frames(lead_frames)
        # Start playback in another thread; capture continues
        t = threading.Thread(target=play_in_thread,
                             args=(str(chirp_wav), started, done),
                             daemon=True)
        t.start()
        started.wait(timeout=2.0)
        # Read the rest of the window
        play_frames = int(args.duration * SAMPLE_RATE)
        tail_frames = int(args.tail_ms / 1000.0 * SAMPLE_RATE)
        body = ls.read_frames(play_frames + tail_frames)

    samples = np.concatenate([np.zeros((lead_frames, 8), dtype=np.int16), body], axis=0)

    capture_wav = out_dir / "capture_8ch.wav"
    with wave.open(str(capture_wav), "wb") as wf:
        wf.setnchannels(8)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.astype("<i2").tobytes())
    print(f"[*] captured {samples.shape[0]/SAMPLE_RATE:.2f}s -> {capture_wav}")

    # The ref channels are typically much louder *exactly* during playback
    # (loopback-class signal) so RMS ratio ref/mic during playback also tells.
    # Now compute correlation.
    # Both ref and mic should correlate with the played chirp at some lag (chirp
    # arrives at mics through air with ~ms delay; arrives at ref via loopback
    # within microseconds).

    ref_chans = samples[:, REF_CH_START:].astype(np.float64).mean(axis=1)
    mic_chans = samples[:, :MIC_CH].astype(np.float64).mean(axis=1)

    # Pad chirp to capture length
    chirp_full = np.zeros_like(ref_chans)
    chirp_full[lead_frames: lead_frames + len(chirp)] = chirp.astype(np.float64)

    # Pure chirp template (without zero-padding) gives a sharper peak when
    # cross-correlated against a long capture; aplay has 200-1500ms startup
    # latency so we must search the full window.
    chirp_template = chirp.astype(np.float64)
    print()
    print("[*] cross-correlation analysis (FFT, full-window):")
    ref_corr, ref_lag = normalized_xcorr_fft(ref_chans, chirp_template)
    mic_corr, mic_lag = normalized_xcorr_fft(mic_chans, chirp_template)
    print(f"  ref(ch6,ch7 mean) <-> chirp:  |corr|={abs(ref_corr):+.3f}  "
          f"lag={ref_lag/SAMPLE_RATE*1000:+7.1f}ms")
    print(f"  mic(ch0..5  mean) <-> chirp:  |corr|={abs(mic_corr):+.3f}  "
          f"lag={mic_lag/SAMPLE_RATE*1000:+7.1f}ms")
    if abs(ref_corr) > 0 and abs(mic_corr) > 0:
        delta_ms = (mic_lag - ref_lag) / SAMPLE_RATE * 1000
        print(f"  mic - ref propagation delay : {delta_ms:+.1f}ms "
              f"(typical 1-30ms for in-room speaker)")
    print()

    # Per-channel RMS over the full capture (we don't know exactly when
    # playback hit each channel, so just report total energy)
    print("[*] per-channel RMS over full capture:")
    print("  ch | rms       role")
    print("  ---+---------------")
    for i in range(8):
        rms = float(np.sqrt(np.mean(samples[:, i].astype(np.float64) ** 2)))
        role = "mic" if i < MIC_CH else "ref"
        print(f"  {i}  | {rms:8.1f}   {role}")

    # Per-channel correlation with the chirp (gives a sense of which ref ch
    # actually carries the loopback)
    print()
    print("[*] per-channel correlation with chirp:")
    print("  ch |  |corr|   lag(ms)   role")
    print("  ---+-------------------------")
    for i in range(8):
        c, lag = normalized_xcorr_fft(samples[:, i].astype(np.float64),
                                      chirp_template)
        role = "mic" if i < MIC_CH else "ref"
        print(f"  {i}  | {abs(c):+.3f}   {lag/SAMPLE_RATE*1000:+7.1f}   {role}")

    # Pass/fail per plan: use absolute correlation (sign depends on speaker
    # phase / ADC inversion which is irrelevant for AEC).
    pass_ref = abs(ref_corr) > 0.5
    pass_mic = abs(mic_corr) > 0.3
    print()
    print("=" * 60)
    print(f"REF channel correlation > 0.5  : {'PASS' if pass_ref else 'FAIL'} "
          f"(|{ref_corr:+.3f}|)")
    print(f"MIC sanity correlation > 0.3   : {'PASS' if pass_mic else 'FAIL'} "
          f"(|{mic_corr:+.3f}|)")
    print("=" * 60)
    overall = pass_ref and pass_mic
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    if not pass_ref:
        print("  -> ch6/ch7 are NOT correlating with playback. Either:")
        print("     1) Speaker output not wired into the board's reference input")
        print("     2) Volume too low / ambient noise too high")
        print("     3) Channels actually swapped or unused")
        print("     Inspect capture_8ch.wav per-channel manually before trusting.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
