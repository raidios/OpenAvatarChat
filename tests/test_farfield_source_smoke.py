#!/usr/bin/env python3
"""
Standalone smoke test for ``client/farfield_audio_source.py``.

Pulls ``--duration`` seconds of audio through ``FarfieldAudioSource.read()``
exactly the way ``client/chat_client.py:mic_sender`` does, dumps the
resulting mono PCM to a wav, and prints sanity stats (finite, RMS, peak,
DOA, denoiser backend). Exits non-zero only if no audio is produced or the
output is non-finite. Skips gracefully (exit 0) when the board link is
down so this can run on dev machines without the M260C attached.

Optional concurrent playback (so AEC has something to cancel without you
having to babysit aplay):
  --play-chirp                  log chirp 200~7000 Hz, fades in/out
  --play-wav <path>             arbitrary mono/stereo wav routed through aplay
  --play-volume <0..1>          gain applied before sending to aplay (default 0.6)
  --play-device <ALSA name>     aplay -D target (default: system default)
  --play-delay-ms <ms>          delay playback start vs capture start
                                (default 200; lets AEC see ~200 ms of silence
                                before echo arrives so the filter has a clean
                                anchor)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO / "tests"]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "client"))


def _make_chirp(duration_s: float, fs: int = 16000,
                f0: float = 200.0, f1: float = 7000.0) -> np.ndarray:
    """Logarithmic chirp 200 -> 7000 Hz with 50 ms cosine fades."""
    t = np.linspace(0.0, duration_s, int(fs * duration_s), endpoint=False)
    k = (f1 / f0) ** (1.0 / max(duration_s, 1e-6))
    phase = 2.0 * np.pi * f0 * (k**t - 1.0) / np.log(k)
    sig = np.sin(phase).astype(np.float32)
    fade = int(0.05 * fs)
    if fade > 0:
        sig[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        sig[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return sig


def _spawn_aplay_pcm(pcm_int16: bytes, sample_rate: int,
                     channels: int = 1, device: Optional[str] = None,
                     delay_s: float = 0.0,
                     stop_flag: Optional[threading.Event] = None) -> threading.Thread:
    """Background thread that pipes raw PCM into ``aplay``. Started
    immediately; sleeps ``delay_s`` before opening the device so the
    capture thread has a moment to settle."""
    cmd = ["aplay", "-q", "-c", str(channels), "-r", str(sample_rate),
           "-f", "S16_LE"]
    if device:
        cmd.extend(["-D", device])

    def _run():
        if delay_s > 0:
            time.sleep(delay_s)
        if stop_flag is not None and stop_flag.is_set():
            return
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[!] aplay not found in PATH; install alsa-utils to use --play-*")
            return
        try:
            assert proc.stdin is not None
            proc.stdin.write(pcm_int16)
            proc.stdin.close()
            try:
                proc.wait(timeout=len(pcm_int16) / (sample_rate * channels * 2) + 5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as exc:  # noqa: BLE001
            print(f"[!] aplay error: {exc}")
            try:
                proc.kill()
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _spawn_aplay_wav(wav_path: Path, device: Optional[str] = None,
                     delay_s: float = 0.0,
                     stop_flag: Optional[threading.Event] = None) -> threading.Thread:
    """Background thread that runs ``aplay <wav_path>`` after ``delay_s``."""
    cmd = ["aplay", "-q"]
    if device:
        cmd.extend(["-D", device])
    cmd.append(str(wav_path))

    def _run():
        if delay_s > 0:
            time.sleep(delay_s)
        if stop_flag is not None and stop_flag.is_set():
            return
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
        except FileNotFoundError:
            print("[!] aplay not found in PATH; install alsa-utils to use --play-*")
        except subprocess.TimeoutExpired:
            print("[!] aplay wav timed out")
        except Exception as exc:  # noqa: BLE001
            print(f"[!] aplay wav error: {exc}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0,
                    help="seconds to capture")
    ap.add_argument("--out", type=str,
                    default="tests/results/farfield_source_capture.wav")
    ap.add_argument("--aec-backend", type=str, default="auto",
                    choices=["auto", "speex", "nlms", "fdaf"])
    ap.add_argument("--aec-filter-ms", type=int, default=200)
    ap.add_argument("--aec-per-mic", dest="aec_per_mic",
                    action="store_true", default=True,
                    help="Run AEC per-mic before MVDR (default).")
    ap.add_argument("--aec-after-mvdr", dest="aec_per_mic",
                    action="store_false",
                    help="Legacy: single AEC after MVDR.")
    ap.add_argument("--dns-backend", type=str, default="auto",
                    choices=["auto", "cpu", "hailo"],
                    help="cpu = DeepFilterNet, hailo = DTLN HEF")
    ap.add_argument("--dns-passthrough", action="store_true",
                    help="bypass DNS entirely (still runs SRP/MVDR/AEC)")
    ap.add_argument("--mic-yaw-offset-deg", type=float, default=30.0)
    ap.add_argument("--chunk-ms", type=int, default=100,
                    help="size of each read() call in milliseconds (default 100)")
    ap.add_argument("--play-chirp", action="store_true",
                    help="play a 200~7000 Hz log chirp through aplay while "
                         "capturing (good AEC stress signal)")
    ap.add_argument("--play-wav", type=str, default=None,
                    help="play this wav file through aplay while capturing "
                         "(good for replaying TTS-style speech)")
    ap.add_argument("--play-volume", type=float, default=0.6,
                    help="gain applied to playback signal (0..1, default 0.6)")
    ap.add_argument("--play-device", type=str, default=None,
                    help="ALSA device passed to aplay -D (default: system default)")
    ap.add_argument("--play-delay-ms", type=int, default=200,
                    help="delay playback start vs capture start, in ms "
                         "(default 200 — gives AEC a clean lead-in)")
    ap.add_argument("--play-pad-s", type=float, default=1.0,
                    help="seconds of silence appended before+after the "
                         "synthesised chirp so it sits inside the capture "
                         "window without truncation; also bounds the "
                         "echo-residual lag search (default 1.0)")
    args = ap.parse_args()

    from farfield_audio_source import FarfieldAudioSource  # noqa: E402

    sample_rate = 16000
    chunk_samples = sample_rate * args.chunk_ms // 1000

    try:
        src = FarfieldAudioSource(
            aec_backend=args.aec_backend,
            aec_filter_length_ms=args.aec_filter_ms,
            aec_per_mic=args.aec_per_mic,
            dns_backend=args.dns_backend,
            dns_use_passthrough=args.dns_passthrough,
            mic_yaw_offset_deg=args.mic_yaw_offset_deg,
        )
        src.open()
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: FarfieldAudioSource open failed ({exc!s})")
        return 0

    target_samples = int(args.duration * sample_rate)
    chunks: list[bytes] = []
    doa_samples: list[tuple[float, float]] = []  # (mic_az, body_az)
    energies: list[float] = []
    samples_so_far = 0

    play_stop = threading.Event()
    play_thread: Optional[threading.Thread] = None
    play_label = "off"
    if args.play_chirp and args.play_wav:
        print("ERROR: pass either --play-chirp or --play-wav, not both")
        src.close()
        return 2
    if args.play_chirp:
        # Embed chirp inside silence so capture window comfortably contains it
        chirp_dur = max(0.5, args.duration - 2 * args.play_pad_s)
        chirp = _make_chirp(chirp_dur, fs=sample_rate) * float(args.play_volume)
        pad_n = int(args.play_pad_s * sample_rate)
        sig = np.concatenate([
            np.zeros(pad_n, dtype=np.float32),
            chirp,
            np.zeros(pad_n, dtype=np.float32),
        ])
        pcm = (np.clip(sig, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        play_thread = _spawn_aplay_pcm(
            pcm, sample_rate, channels=1,
            device=args.play_device,
            delay_s=args.play_delay_ms / 1000.0,
            stop_flag=play_stop,
        )
        play_label = (f"chirp 200-7000Hz {chirp_dur:.1f}s "
                      f"vol={args.play_volume:.2f} delay={args.play_delay_ms}ms")
    elif args.play_wav:
        wav_path = Path(args.play_wav)
        if not wav_path.exists():
            print(f"ERROR: --play-wav path does not exist: {wav_path}")
            src.close()
            return 2
        play_thread = _spawn_aplay_wav(
            wav_path, device=args.play_device,
            delay_s=args.play_delay_ms / 1000.0,
            stop_flag=play_stop,
        )
        play_label = f"wav={wav_path.name} delay={args.play_delay_ms}ms"

    if play_thread is not None:
        print(f"[*] playback         : {play_label}")

    t_wall_start = time.monotonic()
    try:
        while samples_so_far < target_samples:
            buf = src.read(chunk_samples)
            chunks.append(buf)
            samples_so_far += chunk_samples
            mic_az = src.latest_doa_mic_deg
            body_az = src.latest_doa_deg
            chunk_arr = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
            chunk_rms = float(np.sqrt(np.mean(chunk_arr ** 2))) if chunk_arr.size else 0.0
            doa_samples.append((mic_az, body_az))
            energies.append(chunk_rms)
    finally:
        wall_elapsed = time.monotonic() - t_wall_start
        play_stop.set()
        try:
            src.close()
        except Exception:
            pass
        if play_thread is not None:
            play_thread.join(timeout=2.0)

    if not chunks:
        print("FAIL: no chunks emitted")
        return 1
    pcm = b"".join(chunks)
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    finite = bool(np.isfinite(arr).all())
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))

    mic_az_arr = np.asarray([d[0] for d in doa_samples], dtype=np.float64)
    body_az_arr = np.asarray([d[1] for d in doa_samples], dtype=np.float64)
    energies_arr = np.asarray(energies, dtype=np.float64)
    energy_floor = max(rms * 0.7, 0.003)
    voiced_mask = energies_arr > energy_floor
    voiced_count = int(voiced_mask.sum())

    def _circ_median(deg: np.ndarray) -> float:
        if deg.size == 0:
            return float("nan")
        rad = np.deg2rad(deg)
        ang = np.arctan2(np.median(np.sin(rad)), np.median(np.cos(rad)))
        return float(np.rad2deg(ang))

    mic_med = _circ_median(mic_az_arr)
    body_med = _circ_median(body_az_arr)
    mic_med_voiced = _circ_median(mic_az_arr[voiced_mask]) if voiced_count else float("nan")
    body_med_voiced = _circ_median(body_az_arr[voiced_mask]) if voiced_count else float("nan")

    bins = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180], dtype=float)
    hist, _ = np.histogram(body_az_arr, bins=bins)
    hist_str = " ".join(f"{int(c):3d}" for c in hist)
    bin_labels = " ".join(f"{int(b):>+4d}" for b in bins[:-1])

    audio_secs = len(arr) / sample_rate
    rtf = wall_elapsed / max(audio_secs, 1e-9)
    print(f"chunks       : {len(chunks)} x {chunk_samples} samples")
    print(f"audio        : {len(arr)} samples = {audio_secs:.2f} s "
          f"(wall {wall_elapsed:.2f}s, RTF {rtf:.2f})")
    print(f"finite       : {finite}")
    print(f"rms          : {rms:.4f}")
    print(f"peak         : {peak:.4f}")
    print(f"denoiser     : {src.denoiser_name}")
    print(f"DOA mic frame: median {mic_med:+6.1f}° (all)  "
          f"{mic_med_voiced:+6.1f}° (voiced, n={voiced_count})")
    print(f"DOA body frame:median {body_med:+6.1f}° (all)  "
          f"{body_med_voiced:+6.1f}° (voiced, n={voiced_count})  "
          f"offset={args.mic_yaw_offset_deg:+.1f}°")
    print(f"DOA body hist (45° bins, lower edges):")
    print(f"  bins: {bin_labels}  +180")
    print(f"  cnt : {hist_str}")
    def _wrap180(d: float) -> float:
        return float((d + 180.0) % 360.0 - 180.0)

    print(f"DOA last     : mic {_wrap180(mic_az_arr[-1]):+6.1f}°  "
          f"body {_wrap180(body_az_arr[-1]):+6.1f}°")

    # Echo-residual estimate: when we know the playback signal we can
    # measure how much of it leaked through to the cleaned output. We
    # cross-correlate the playback reference against the cleaned output,
    # take energy in a window around the best lag, and compare to the
    # ref's own energy. Lower residual_db = more echo cancelled.
    ref_signal = None
    ref_label = None
    if args.play_chirp:
        chirp_dur = max(0.5, args.duration - 2 * args.play_pad_s)
        ref_signal = _make_chirp(chirp_dur, fs=sample_rate) * float(
            args.play_volume)
        ref_label = "chirp"
    elif args.play_wav:
        try:
            import soundfile as _sf
            from scipy.signal import resample_poly as _rp
            from math import gcd as _gcd
            ref_data, ref_sr = _sf.read(
                args.play_wav, dtype="float32", always_2d=False)
            if ref_data.ndim > 1:
                ref_data = ref_data.mean(axis=1)
            if ref_sr != sample_rate:
                g = _gcd(ref_sr, sample_rate)
                ref_data = _rp(ref_data, sample_rate // g, ref_sr // g)
            ref_signal = (ref_data.astype(np.float32)
                          * float(args.play_volume))
            ref_label = Path(args.play_wav).stem
        except Exception as e:
            print(f"[!] could not load play_wav for residual measurement: {e}")
    if ref_signal is not None and len(arr) > 1000:
        # Use only the leading portion of the ref that can fit inside
        # the capture (after the playback delay). For longer wavs the
        # cap is half the capture length; that gives `np.correlate` a
        # meaningful search range while still measuring residual on a
        # solid voice segment.
        max_ref_len = max(1000, len(arr) // 2)
        ref_for_corr = ref_signal[: max_ref_len]
        if len(arr) > len(ref_for_corr) + 100:
            cc = np.correlate(arr, ref_for_corr, mode="valid")
            min_lag = max(0, int(
                (args.play_delay_ms / 1000.0 - 0.05) * sample_rate))
            max_lag = int(
                (args.play_delay_ms / 1000.0 + args.play_pad_s + 1.5)
                * sample_rate)
            max_lag = min(max_lag, len(cc) - 1)
            min_lag = min(min_lag, max(0, max_lag - 1))
            cc_window = cc[min_lag: max_lag + 1]
            best = int(np.argmax(np.abs(cc_window))) + min_lag
            seg = arr[best: best + len(ref_for_corr)]
            seg_rms = float(np.sqrt(np.mean(seg * seg))) + 1e-12
            ref_rms = (float(np.sqrt(np.mean(ref_for_corr * ref_for_corr)))
                       + 1e-12)
            residual_db = 20.0 * np.log10(seg_rms / ref_rms)
            corr_norm = float(cc[best]) / (
                np.linalg.norm(seg) * np.linalg.norm(ref_for_corr) + 1e-12)
            print(f"echo residual ({ref_label}): "
                  f"lag={best/sample_rate*1000:.0f} ms  "
                  f"ref_dur={len(ref_for_corr)/sample_rate:.1f}s  "
                  f"seg/ref RMS={residual_db:+5.1f} dB  "
                  f"|corr|={abs(corr_norm):.3f}")
            print(f"               (more negative dB = more echo cancelled)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    print(f"wrote        : {out}")

    ok = finite and len(arr) > sample_rate * args.duration * 0.5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
