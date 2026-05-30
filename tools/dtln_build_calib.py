#!/usr/bin/env python3
"""Build Hailo DFC calibration sets for DTLN model_1 + model_2.

Run this on the compile machine (WSL/Linux) where you have onnxruntime
and the static ONNX files. It:

  1. Loads `audio_in` audio (16 kHz mono) — defaults to a synthetic noisy
     speech mixture if no input wav is provided.
  2. Streams the audio frame-by-frame through model_1 (the frequency-mask
     network) and harvests (input_2, input_3) for ~1500 frames.
  3. Streams through model_2 (the time-refine network) and harvests
     (input_4, input_5).
  4. Writes:
        calib_p1.npz  with keys input_2, input_3
        calib_p2.npz  with keys input_4, input_5

Usage:
  python tools/dtln_build_calib.py \
      --p1-onnx model_1_static.onnx \
      --p2-onnx model_2_static.onnx \
      --in-wav  /path/to/16k_mono_speech.wav \
      --num-frames 1500 \
      --out-dir .

If you don't have a recording yet, leave --in-wav unset; we'll synthesize a
clean speech proxy (LFM sweep + bandlimited noise mix) — quality is enough
for first-pass quantization, but for production rebuild calib with real mic
audio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SR = 16000
BLOCK_LEN = 512        # 32 ms
BLOCK_SHIFT = 128      # 8 ms
FFT_SIZE = 512


def _load_or_synth(in_wav: str | None, n_seconds: float) -> np.ndarray:
    """Return float32 mono 16 kHz numpy array."""
    n = int(n_seconds * SR)
    if in_wav and Path(in_wav).exists():
        import wave
        with wave.open(in_wav, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
        x = data[:, 0]
        if sr != SR:
            print(f"[!] resampling {sr} -> {SR}")
            from scipy.signal import resample_poly
            x = resample_poly(x, SR, sr).astype(np.float32)
        if len(x) >= n:
            return x[:n]
        rep = int(np.ceil(n / len(x)))
        return np.tile(x, rep)[:n]

    print("[*] no wav supplied -> synthesizing speech-like calibration signal")
    rng = np.random.default_rng(0)
    t = np.arange(n, dtype=np.float32) / SR
    chirp = 0.4 * np.sin(2 * np.pi * (200 + 600 * t) * t)
    bn = rng.normal(size=n).astype(np.float32)
    bn = np.convolve(bn, np.ones(64) / 64, mode="same") * 0.05
    return chirp + bn


def _frame_stft(audio: np.ndarray):
    """Generator of (mag, phase) per frame. mag shape (1,1,257)."""
    in_buf = np.zeros(BLOCK_LEN, dtype=np.float32)
    for s in range(0, len(audio) - BLOCK_SHIFT, BLOCK_SHIFT):
        in_buf[: BLOCK_LEN - BLOCK_SHIFT] = in_buf[BLOCK_SHIFT:].copy()
        in_buf[BLOCK_LEN - BLOCK_SHIFT:] = audio[s: s + BLOCK_SHIFT]
        spec = np.fft.rfft(in_buf, n=FFT_SIZE)
        mag = np.abs(spec).astype(np.float32)[None, None, :]   # (1,1,257)
        ph = np.angle(spec).astype(np.float32)
        yield in_buf.copy(), mag, ph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1-onnx", default="model_1_static.onnx")
    ap.add_argument("--p2-onnx", default="model_2_static.onnx")
    ap.add_argument("--in-wav", default=None)
    ap.add_argument("--num-frames", type=int, default=1500)
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="seconds of audio when synthesizing")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    import onnxruntime as ort

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = _load_or_synth(args.in_wav, args.seconds)
    sess1 = ort.InferenceSession(args.p1_onnx, providers=["CPUExecutionProvider"])
    sess2 = ort.InferenceSession(args.p2_onnx, providers=["CPUExecutionProvider"])
    input_names_1 = [i.name for i in sess1.get_inputs()]
    output_names_1 = [o.name for o in sess1.get_outputs()]
    input_names_2 = [i.name for i in sess2.get_inputs()]
    output_names_2 = [o.name for o in sess2.get_outputs()]
    print(f"[*] p1 IO: in={input_names_1} out={output_names_1}")
    print(f"[*] p2 IO: in={input_names_2} out={output_names_2}")

    # Hidden states carry across frames; reset to zero at the start.
    state1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
    state2 = np.zeros((1, 2, 128, 2), dtype=np.float32)

    p1_in_a, p1_in_s = [], []
    p2_in_a, p2_in_s = [], []

    n_frames = 0
    for in_buf, mag, ph in _frame_stft(audio):
        if n_frames >= args.num_frames:
            break
        p1_in_a.append(mag.copy())
        p1_in_s.append(state1.copy())
        feeds = {input_names_1[0]: mag, input_names_1[1]: state1}
        mask, state1 = sess1.run(output_names_1, feeds)

        # Reconstruct refined frame for model_2 (real DTLN runs model_1 mask
        # against magnitude, then iSTFT, then model_2 polishes the time-domain
        # frame). To keep the calib set realistic we mirror that here.
        masked_spec = mag.squeeze() * mask.squeeze() * np.exp(1j * ph)
        time_frame = np.fft.irfft(masked_spec, n=FFT_SIZE).astype(np.float32)
        time_frame = time_frame[None, None, :]  # (1, 1, 512)

        p2_in_a.append(time_frame.copy())
        p2_in_s.append(state2.copy())
        feeds2 = {input_names_2[0]: time_frame, input_names_2[1]: state2}
        _refined, state2 = sess2.run(output_names_2, feeds2)

        n_frames += 1

    # Each frame above is shape (1, 1, 257) etc. Concat over leading axis
    # gives (N, 1, 257) where N = num_frames. DFC 5.3's --calib-set-path
    # expects per-sample shape == runtime input shape, so we keep the
    # singleton dim. (DFC interprets leading axis as the calibration-sample
    # axis, so total shape (N, 1, 257) means N samples each of shape (1,257)
    # which then gets broadcast to (1,1,257) at parse time — correct.)
    p1_in_a = np.concatenate(p1_in_a, axis=0)   # (N, 1, 257)
    p1_in_s = np.concatenate(p1_in_s, axis=0)   # (N, 2, 128, 2)
    p2_in_a = np.concatenate(p2_in_a, axis=0)   # (N, 1, 512)
    p2_in_s = np.concatenate(p2_in_s, axis=0)   # (N, 2, 128, 2)

    out1 = out_dir / "calib_p1.npz"
    out2 = out_dir / "calib_p2.npz"
    np.savez(str(out1), **{input_names_1[0]: p1_in_a, input_names_1[1]: p1_in_s})
    np.savez(str(out2), **{input_names_2[0]: p2_in_a, input_names_2[1]: p2_in_s})
    print(f"[+] wrote {out1}: shapes "
          f"{input_names_1[0]}={p1_in_a.shape} {input_names_1[1]}={p1_in_s.shape}")
    print(f"[+] wrote {out2}: shapes "
          f"{input_names_2[0]}={p2_in_a.shape} {input_names_2[1]}={p2_in_s.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
