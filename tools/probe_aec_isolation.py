#!/usr/bin/env python3
"""
AEC isolation probe.

Records 8-ch from M260C while host plays a chirp via aplay, then runs three
configurations on the captured PCM and reports ERLE for each:

  A. raw ch0  + ref(mean ch6/7)         → SpeexAec
  B. raw ch0  + ref(mean ch6/7)         → NlmsAec
  C. MVDR(ch0..5) + ref(mean ch6/7)     → SpeexAec   (current pipeline order)

If A/B knock the chirp down by 12+ dB but C does not, the bug is "MVDR
distorts the echo path before AEC sees it." If A/B also fail, the bug is in
the AEC config itself (filter length, backend pick, frame alignment).

Usage:
  .venv/bin/python tools/probe_aec_isolation.py [--duration 5]
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "M2_SDK" / "live_stream" / "host"))

from live_stream import LiveStream, SAMPLE_RATE                           # noqa: E402
from audio_frontend.dsp.aec import (                                       # noqa: E402
    AecConfig, SpeexAec, NlmsAec, FdafAec,
)
from audio_frontend.dsp.geometry import RingArray                          # noqa: E402
from audio_frontend.dsp.mvdr import MvdrBeamformer, MvdrConfig             # noqa: E402


def make_chirp(duration_s: float, fs: int = SAMPLE_RATE,
               f0: float = 200.0, f1: float = 7000.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(fs * duration_s), endpoint=False)
    k = (f1 / f0) ** (1.0 / duration_s)
    phase = 2.0 * np.pi * f0 * (k**t - 1.0) / np.log(k)
    sig = 0.6 * np.sin(phase)
    fade = int(0.05 * fs)
    win = np.ones_like(sig)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    return (sig * win).astype(np.float32)


def load_wav_stimulus(path: str, fs: int = SAMPLE_RATE,
                      target_peak: float = 0.6,
                      max_duration_s: float | None = None) -> np.ndarray:
    """Load a wav, resample to ``fs``, mono-mix, normalise peak."""
    import soundfile as sf
    from scipy.signal import resample_poly
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != fs:
        from math import gcd
        g = gcd(sr, fs)
        data = resample_poly(data, fs // g, sr // g).astype(np.float32)
    if max_duration_s is not None:
        n = int(max_duration_s * fs)
        data = data[:n]
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data * (target_peak / peak)
    fade = int(0.05 * fs)
    if len(data) > 2 * fade:
        ramp_in = np.linspace(0, 1, fade, dtype=np.float32)
        ramp_out = np.linspace(1, 0, fade, dtype=np.float32)
        data[:fade] *= ramp_in
        data[-fade:] *= ramp_out
    return data.astype(np.float32)


def write_wav(path: Path, mono: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes())


def play_signal(sig: np.ndarray, sr: int) -> threading.Thread:
    """Spawn aplay to play the given mono signal from a raw int16 stream."""
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes()
    def _run():
        proc = subprocess.Popen(
            ["aplay", "-q", "-c", "1", "-r", str(sr), "-f", "S16_LE"],
            stdin=subprocess.PIPE,
        )
        try:
            proc.stdin.write(pcm)
            proc.stdin.close()
            proc.wait(timeout=len(chirp)/sr + 5)
        except Exception:
            proc.kill()
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def erle_db(near: np.ndarray, out: np.ndarray, eps: float = 1e-9) -> float:
    n_e = float(np.mean(near**2))
    o_e = float(np.mean(out**2))
    return 10.0 * np.log10((n_e + eps) / (o_e + eps))


def run_aec(near: np.ndarray, far: np.ndarray, aec) -> np.ndarray:
    N = aec.frame_size
    out = np.zeros_like(near)
    for s in range(0, len(near) - N + 1, N):
        out[s:s+N] = aec.process_frame(near[s:s+N], far[s:s+N])
    return out


def run_mvdr(mics_8ch: np.ndarray, hop: int = 160) -> np.ndarray:
    array = RingArray(n_mics=6, radius_m=0.033)
    cfg = MvdrConfig(sample_rate=SAMPLE_RATE, hop=hop, initial_az_deg=0.0)
    mvdr = MvdrBeamformer(array, cfg)
    n = (mics_8ch.shape[0] // hop) * hop
    rms = np.sqrt(np.mean(mics_8ch[:, :6] ** 2, axis=1)) + 1e-9
    n_chunks = n // hop
    rms_chunks = rms[:n_chunks*hop].reshape(n_chunks, hop).mean(axis=1)
    vad_silence = rms_chunks < 0.005
    return mvdr.process(mics_8ch[:n, :6], vad_silence)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0,
                    help="capture duration; ignored if --stimulus given "
                         "(uses wav length instead).")
    ap.add_argument("--stimulus", type=str, default=None,
                    help="path to a mono wav (any sr) to play as the "
                         "AEC reference stimulus instead of a synthetic "
                         "log-chirp; useful for voice/music ref.")
    ap.add_argument("--stimulus-peak", type=float, default=0.6,
                    help="peak amplitude (0..1) the wav is rescaled to "
                         "before playback.")
    ap.add_argument("--out", type=str, default="tests/results/aec_iso")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    if args.stimulus:
        stim = load_wav_stimulus(args.stimulus, fs=SAMPLE_RATE,
                                 target_peak=args.stimulus_peak)
        stim_label = Path(args.stimulus).stem
        play_secs = len(stim) / SAMPLE_RATE
        print(f"[*] using voice stimulus '{args.stimulus}' "
              f"({play_secs:.2f}s, peak={float(np.max(np.abs(stim))):.3f})")
    else:
        stim = make_chirp(args.duration)
        stim_label = "chirp"
        play_secs = args.duration
    write_wav(out_dir / f"{stim_label}.wav", stim)

    print(f"[*] connecting LiveStream...")
    stream = LiveStream(host="127.0.0.1", port=9999, ensure_forward=False)
    stream.connect()

    cap_secs = play_secs + 1.0
    cap_n = int(cap_secs * SAMPLE_RATE)
    print(f"[*] capturing {cap_secs:.1f}s + playing {stim_label}")
    t0 = time.monotonic()
    play_thread = play_signal(stim, SAMPLE_RATE)
    cap = stream.read_frames(cap_n)        # (T, 8) int16
    play_thread.join(timeout=2.0)
    stream.close()
    print(f"    capture took {time.monotonic()-t0:.2f}s")

    cap_f = cap.astype(np.float32) / 32768.0
    mics = cap_f[:, :6]
    ref = cap_f[:, 6:8].mean(axis=1)
    near0 = mics[:, 0]

    # Trim leading silence / pipeline buffer using ref energy onset
    ref_rms_win = np.sqrt(np.convolve(ref**2, np.ones(800)/800, mode="same"))
    onset = int(np.argmax(ref_rms_win > 0.002)) if (ref_rms_win > 0.002).any() else 0
    onset = max(0, onset - 800)
    print(f"[*] ref onset detected at {onset/SAMPLE_RATE:.2f}s -> trimming")
    mics = mics[onset:]; ref = ref[onset:]; near0 = near0[onset:]
    n = (len(near0) // 160) * 160
    mics = mics[:n]; ref = ref[:n]; near0 = near0[:n]

    print(f"[*] near0 RMS={np.sqrt(np.mean(near0**2)):.4f}, "
          f"ref RMS={np.sqrt(np.mean(ref**2)):.4f}")

    cfg = AecConfig()
    have_speex = True
    try:
        speex_a = SpeexAec(cfg)
    except Exception as e:
        have_speex = False
        print(f"[!] Speex not available ({e!s}); skipping Speex tests")

    nlms_b = NlmsAec(cfg)

    # Sweep NLMS filter length × step size mu to chase the oracle Wiener
    # bound. ref is line-level (peak IR lag ~1 ms, tail < 50 ms) so very
    # short filters with conservative mu should win.
    print("[*] NLMS filter × mu sweep (raw ch0 + ch6/7 ref):")
    print("    " + "  ".join(f"mu={mu:.2f}" for mu in [0.1, 0.2, 0.3, 0.5]))
    for ms in [20, 30, 50, 70, 100, 150]:
        row = f"   filter={ms:>4d}ms"
        for mu in [0.1, 0.2, 0.3, 0.5]:
            c = AecConfig(filter_length_samples=ms * 16)
            ec = NlmsAec(c); ec.mu = mu
            o = run_aec(near0, ref, ec)
            e = erle_db(near0, o)
            row += f"   {e:+5.1f} dB"
        print(row)

    # FDAF sweep — same shape, frequency-domain block FLMS. Should beat
    # NLMS by ~10-15 dB on stationary signals.
    print("[*] FDAF filter × mu sweep (raw ch0 + ch6/7 ref):")
    print("    " + "  ".join(f"mu={mu:.2f}"
                              for mu in [0.2, 0.5, 0.8, 1.0]))
    for ms in [20, 30, 50, 70, 100, 150]:
        row = f"   filter={ms:>4d}ms"
        for mu in [0.2, 0.5, 0.8, 1.0]:
            c = AecConfig(filter_length_samples=ms * 16)
            ec = FdafAec(c); ec.mu = mu
            o = run_aec(near0, ref, ec)
            e = erle_db(near0, o)
            row += f"   {e:+5.1f} dB"
        print(row)

    write_wav(out_dir / "near_raw.wav", near0)
    write_wav(out_dir / "ref.wav", ref)

    if have_speex:
        out_a = run_aec(near0, ref, speex_a)
        e_a = erle_db(near0, out_a)
        print(f"[A] Speex(raw ch0 + ref)             ERLE = {e_a:+.1f} dB")
        write_wav(out_dir / "out_A_speex_raw.wav", out_a)

    out_b = run_aec(near0, ref, nlms_b)
    e_b = erle_db(near0, out_b)
    print(f"[B] NLMS (raw ch0 + ref)             ERLE = {e_b:+.1f} dB")
    write_wav(out_dir / "out_B_nlms_raw.wav", out_b)

    # Per-mic AEC then MVDR
    if have_speex:
        per_mic = np.zeros_like(mics)
        for i in range(6):
            ec = SpeexAec(cfg)
            per_mic[:, i] = run_aec(mics[:, i], ref, ec)
        beam_per_mic = run_mvdr(np.concatenate(
            [per_mic, np.zeros((per_mic.shape[0], 2), dtype=np.float32)], axis=1))
        e_d = erle_db(near0[:len(beam_per_mic)], beam_per_mic)
        print(f"[D] per-mic Speex AEC -> MVDR        ERLE = {e_d:+.1f} dB "
              f"(vs raw ch0 baseline)")
        write_wav(out_dir / "out_D_perm_aec_then_mvdr.wav", beam_per_mic)

    # Current pipeline order: MVDR -> Speex
    beam = run_mvdr(np.concatenate(
        [mics, np.zeros((mics.shape[0], 2), dtype=np.float32)], axis=1))
    far_for_beam = ref[:len(beam)]
    if have_speex:
        ec = SpeexAec(cfg)
        out_c = run_aec(beam, far_for_beam, ec)
        e_c = erle_db(beam, out_c)
        print(f"[C] MVDR -> Speex (current order)    ERLE = {e_c:+.1f} dB "
              f"(vs MVDR-only baseline)")
        write_wav(out_dir / "out_C_mvdr_then_speex.wav", out_c)
    write_wav(out_dir / "out_mvdr_only.wav", beam)

    # E/F. Same near, but ref = the *clean playback PCM* (i.e. what we'd
    # tap if we routed the AudioPlayer output buffer to AEC), aligned to
    # raw ch0 via cross-correlation. This is what a software-reference
    # AEC would see in real chat (TTS PCM piped to AEC alongside aplay).
    cc_window = stim[: min(len(stim), 2 * SAMPLE_RATE)]
    cc = np.correlate(near0, cc_window, mode="valid")
    lag = int(np.argmax(np.abs(cc)))
    print(f"[*] clean-{stim_label} alignment vs raw ch0: "
          f"lag={lag/SAMPLE_RATE*1000:.1f} ms")
    sw_ref = np.zeros_like(near0)
    chunk = min(len(stim), len(sw_ref) - lag)
    if chunk > 0:
        sw_ref[lag:lag+chunk] = stim[:chunk]

    if have_speex:
        ec = SpeexAec(cfg)
        out_e = run_aec(near0, sw_ref, ec)
        e_e = erle_db(near0, out_e)
        print(f"[E] Speex(raw ch0 + SOFTWARE ref)    ERLE = {e_e:+.1f} dB "
              f"<-- target architecture")
        write_wav(out_dir / "out_E_speex_swref.wav", out_e)

    nlms_f = NlmsAec(cfg)
    out_f = run_aec(near0, sw_ref, nlms_f)
    e_f = erle_db(near0, out_f)
    print(f"[F] NLMS (raw ch0 + SOFTWARE ref)    ERLE = {e_f:+.1f} dB")
    write_wav(out_dir / "out_F_nlms_swref.wav", out_f)

    write_wav(out_dir / "ref_software.wav", sw_ref)
    print(f"[*] artefacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
