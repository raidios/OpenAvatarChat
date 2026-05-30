#!/usr/bin/env python3
"""
Estimate the ref->mic impulse response on the M260C.

Hypothesis (per device manual): ch6/ch7 are line-level reference, wired
straight from the speaker amplifier's input. If true, mic[t] should be
ref[t] convolved with the room IR (and a small electrical delay). The
estimated h(τ) should be a clean spike at the air-time-of-flight lag with
a short tail (< 100 ms in a normal room) and a modest dynamic range.

If instead ch6/ch7 carry processed audio (board-side AGC / NLP / hardware
echo cancellation already kicked in / mic-side feedback pickup), the
estimated IR will be diffuse, multi-modal, or unstable run-to-run.

This is a key diagnostic for AEC: a clean LTI relationship between ref
and mic is exactly what NLMS/Speex assume; if the IR is well-behaved but
Speex still resets, the bug is in our Speex wrapper / parameter pick, not
in the hardware.

Procedure:
  1. Play a 4 s log chirp through aplay, with 1 s pre/post silence.
  2. Concurrently capture 8ch via LiveStream for 6 s.
  3. Estimate h = (Ref^* * Mic) / (Ref^* * Ref + reg)  in the freq domain
     (Wiener-style; equivalent to a regularised inverse cross-correlation
     deconvolution).
  4. Report:
       - IR peak amplitude / decay time / energy in tail (50ms+)
       - mic vs ref energy ratio (in chirp window)
       - per-bin coherence: |Ref^*Mic|^2 / (|Ref|^2 |Mic|^2)
       - simple 'chirp-removed' baseline: subtract h*ref from mic, report
         residual energy (a manual NLMS/RLS would reach this asymptote).

Usage:
  .venv/bin/python tools/probe_ref_ir.py --duration 6 --out tests/results/ir_probe
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "thirdparty" / "M2_SDK" / "live_stream" / "host"))

from live_stream import LiveStream, SAMPLE_RATE  # noqa: E402


def make_chirp(duration_s: float, fs: int = SAMPLE_RATE,
               f0: float = 200.0, f1: float = 7000.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(fs * duration_s), endpoint=False)
    k = (f1 / f0) ** (1.0 / max(duration_s, 1e-6))
    phase = 2.0 * np.pi * f0 * (k**t - 1.0) / np.log(k)
    sig = np.sin(phase).astype(np.float32)
    fade = int(0.05 * fs)
    if fade > 0:
        sig[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        sig[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return sig


def write_wav(path: Path, mono: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes())


def play_async(pcm_bytes: bytes, sr: int, channels: int = 1,
               delay_s: float = 0.5) -> threading.Thread:
    def _run():
        time.sleep(delay_s)
        try:
            proc = subprocess.Popen(
                ["aplay", "-q", "-c", str(channels), "-r", str(sr),
                 "-f", "S16_LE"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            assert proc.stdin
            proc.stdin.write(pcm_bytes); proc.stdin.close()
            proc.wait(timeout=20)
        except Exception as exc:  # noqa: BLE001
            print(f"[!] aplay error: {exc}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0,
                    help="capture seconds")
    ap.add_argument("--chirp-secs", type=float, default=4.0)
    ap.add_argument("--volume", type=float, default=0.6)
    ap.add_argument("--out", type=str, default="tests/results/ir_probe")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    chirp = make_chirp(args.chirp_secs) * float(args.volume)
    pcm_bytes = (np.clip(chirp, -1, 1) * 32767).astype("<i2").tobytes()
    write_wav(out_dir / "chirp.wav", chirp, SAMPLE_RATE)

    print(f"[*] connecting LiveStream (8ch raw)...")
    stream = LiveStream(host="127.0.0.1", port=9999, ensure_forward=False)
    stream.connect()

    print(f"[*] playing chirp ({args.chirp_secs:.1f}s) + capturing "
          f"{args.duration:.1f}s")
    play_thread = play_async(pcm_bytes, SAMPLE_RATE, delay_s=0.5)
    cap = stream.read_frames(int(args.duration * SAMPLE_RATE))
    play_thread.join(timeout=2.0)
    stream.close()

    cap_f = cap.astype(np.float32) / 32768.0
    mic = cap_f[:, :6].mean(axis=1)
    ref = cap_f[:, 6:8].mean(axis=1)
    n = len(mic)
    print(f"    captured {n/SAMPLE_RATE:.2f} s; "
          f"mic RMS={np.sqrt(np.mean(mic**2)):.4f}, "
          f"ref RMS={np.sqrt(np.mean(ref**2)):.4f}, "
          f"ratio mic/ref = {np.sqrt(np.mean(mic**2))/(np.sqrt(np.mean(ref**2))+1e-9):.2f}x")

    # Find chirp window in ref (envelope > 5% of peak)
    env = np.abs(np.convolve(np.abs(ref), np.ones(800)/800, mode="same"))
    thr = env.max() * 0.05
    idx = np.where(env > thr)[0]
    if len(idx) < 16000:
        print("FAIL: no obvious chirp envelope detected in ref")
        return 1
    s, e = idx[0], idx[-1]
    print(f"    ref chirp window: {s/SAMPLE_RATE:.2f}-{e/SAMPLE_RATE:.2f}s "
          f"({(e-s)/SAMPLE_RATE:.2f}s)")
    pad = 800
    s = max(0, s - pad); e = min(n, e + pad)
    mic_w = mic[s:e]; ref_w = ref[s:e]

    # Frequency-domain Wiener IR estimate.
    L = len(mic_w)
    Nfft = 1 << int(np.ceil(np.log2(L)))
    M = np.fft.rfft(mic_w, n=Nfft)
    R = np.fft.rfft(ref_w, n=Nfft)
    eps = 1e-3 * float(np.mean(np.abs(R)**2))
    H = (np.conj(R) * M) / (np.conj(R) * R + eps)
    h = np.fft.irfft(H, n=Nfft)[:int(0.300 * SAMPLE_RATE)]   # 300 ms window

    h_abs = np.abs(h)
    peak_i = int(np.argmax(h_abs))
    peak_v = float(h_abs[peak_i])
    peak_lag_ms = peak_i / SAMPLE_RATE * 1000.0
    # Total energy in the IR, plus tail energy beyond peak+50ms.
    e_total = float(np.sum(h * h)) + 1e-12
    tail_start = min(len(h), peak_i + int(0.050 * SAMPLE_RATE))
    e_tail = float(np.sum(h[tail_start:] * h[tail_start:]))
    tail_db = 10.0 * np.log10(e_tail / e_total + 1e-12)

    # Coherence (averaged over freq).
    Sxx = np.mean(np.abs(R)**2)
    Syy = np.mean(np.abs(M)**2)
    Sxy = np.mean(np.conj(R) * M)
    coherence = np.abs(Sxy)**2 / (Sxx * Syy + 1e-12)

    # Subtract h * ref from mic; report residual.
    re_full = np.fft.irfft(R * H, n=Nfft)[:L]
    residual = mic_w - re_full[:L]
    erle_db = 10.0 * np.log10(
        (np.mean(mic_w * mic_w) + 1e-12) / (np.mean(residual * residual) + 1e-12))

    print()
    print(f"=== IR diagnosis ===")
    print(f"peak lag           : {peak_lag_ms:.2f} ms (sample {peak_i})")
    print(f"peak amplitude     : {peak_v:.4f}")
    print(f"tail energy 50ms+  : {tail_db:+.1f} dB below total")
    print(f"avg coherence      : {coherence:.3f} "
          f"({'GOOD LTI' if coherence > 0.5 else 'noisy / non-LTI'})")
    print(f"oracle ERLE (Wiener): {erle_db:+.1f} dB")
    print()
    print("Interpretation:")
    print("  - IR with sharp peak + short decay (tail < -10 dB) => clean")
    print("    line-level reference, NLMS/Speex should converge.")
    print("  - Diffuse / multi-modal / coherence < 0.3 => ref is processed")
    print("    or noisy; need to revisit board config or wiring.")
    print(f"  - This run's oracle ERLE upper bound = {erle_db:+.1f} dB.")
    print()

    # Save artefacts
    write_wav(out_dir / "mic.wav", mic_w, SAMPLE_RATE)
    write_wav(out_dir / "ref.wav", ref_w, SAMPLE_RATE)
    write_wav(out_dir / "residual.wav", residual.astype(np.float32), SAMPLE_RATE)
    np.save(out_dir / "ir.npy", h.astype(np.float32))
    print(f"artefacts in {out_dir}/")
    print(f"  ir.npy           - {len(h)}-sample float32 IR (300 ms)")
    print(f"  mic.wav, ref.wav - aligned chirp window")
    print(f"  residual.wav     - mic - h*ref  (oracle AEC residual)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
