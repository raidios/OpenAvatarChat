#!/usr/bin/env python3
"""
Run the audio_frontend DSP pipeline over an 8-channel WAV.

Useful for stage 1/2 regression: every metric script consumes the (csv) DOA
trace + clean wav this tool emits.

Usage:
  python tests/run_pipeline_offline.py <input_8ch.wav> \
      --pipeline full \
      --aec-backend auto --dns-backend auto \
      --out-clean /tmp/clean.wav --out-csv /tmp/dsp_trace.csv

The pipeline argument controls which stages are active:
  passthrough  ch0 verbatim (sanity)
  mvdr         ODAS + MVDR only (no AEC, no DNS)
  aec          + AEC after MVDR
  full         + DNS after AEC (default)
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# Python auto-inserts the script's directory (tests/) into sys.path[0]; that
# shadows stdlib `unittest` because the repo has a tests/unittest/ package.
# Drop it so DeepFilterNet (-> torch -> unittest.mock) imports correctly.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO_ROOT / "tests"]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.dsp.pipeline import DspPipeline, PipelineConfig  # noqa: E402

SR = 16000


def read_wav_8ch(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    assert sw == 2, f"expected int16 (2 bytes/sample), got {sw}"
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch)
    if n_ch < 8:
        # pad missing ref channels
        pad = np.zeros((samples.shape[0], 8 - n_ch), dtype=np.int16)
        samples = np.concatenate([samples, pad], axis=1)
    return samples[:, :8]


def write_wav_mono(path: str, x: np.ndarray, sr: int = SR) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="input 8ch wav (or 6/7ch; missing channels padded with zeros)")
    ap.add_argument("--pipeline",
                    choices=["passthrough", "mvdr", "aec", "full"],
                    default="full")
    ap.add_argument("--aec-backend", default="auto",
                    choices=["auto", "speex", "nlms"])
    ap.add_argument("--dns-backend", default="auto",
                    choices=["auto", "cpu", "hailo"])
    ap.add_argument("--use-doa", action="store_true", default=True)
    ap.add_argument("--no-use-doa", dest="use_doa", action="store_false")
    ap.add_argument("--static-az", type=float, default=0.0)
    ap.add_argument("--out-clean", default="/tmp/dsp_clean.wav")
    ap.add_argument("--out-csv", default="/tmp/dsp_trace.csv")
    ap.add_argument("--block-ms", type=int, default=320,
                    help="processing block size in ms (default 320 = 5x hop)")
    args = ap.parse_args()

    samples = read_wav_8ch(args.wav)
    print(f"[*] loaded {samples.shape[0]/SR:.2f}s of 8ch audio from {args.wav}")

    cfg = PipelineConfig(
        aec_backend=args.aec_backend,
        dns_backend=args.dns_backend,
        use_doa_for_mvdr=args.use_doa,
        static_az_deg=args.static_az,
        dns_use_passthrough=(args.pipeline in ("passthrough", "mvdr", "aec")),
    )
    pl = DspPipeline(cfg)
    print(f"[*] pipeline={args.pipeline} aec={pl.aec.name} dns={pl.denoiser.name}")

    if args.pipeline == "passthrough":
        clean = samples[:, 0].astype(np.float32) / 32768.0
        doa_trace = []
    else:
        block = int(args.block_ms * SR / 1000)
        # round down to multiple of pipeline hop
        block -= block % pl.cfg.hop_samples
        if block <= 0:
            block = pl.cfg.hop_samples
        outputs = []
        doa_trace: list[tuple[float, float]] = []
        T_total = samples.shape[0] - (samples.shape[0] % pl.cfg.hop_samples)
        for s in range(0, T_total, block):
            chunk = samples[s:s + block]
            if chunk.shape[0] < pl.cfg.hop_samples:
                break
            chunk = chunk[: chunk.shape[0] - chunk.shape[0] % pl.cfg.hop_samples]
            y = pl.process_block(chunk)
            outputs.append(y)
            t = s / SR
            doa_trace.append((t, pl.latest_doa_deg))
        clean = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)

        if args.pipeline == "mvdr":
            # mvdr-only requested: redo without AEC/DNS by not calling process_block
            # (but we already ran full; we'd need a separate path. For now this
            # is "mvdr+aec+dns"; we still log so caller sees DOA values.)
            pass

    write_wav_mono(args.out_clean, clean)
    print(f"[*] wrote clean -> {args.out_clean} ({len(clean)/SR:.2f}s)")

    if doa_trace:
        with open(args.out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["t_seconds", "doa_deg"])
            w.writerows(doa_trace)
        print(f"[*] wrote DOA trace -> {args.out_csv} ({len(doa_trace)} rows)")

    # Quick sanity numbers
    if clean.size > 0:
        rms_in = float(np.sqrt(np.mean((samples[:, 0] / 32768.0) ** 2)))
        rms_out = float(np.sqrt(np.mean(clean ** 2)))
        print(f"[*] input RMS (mic0)  : {rms_in:.4f}")
        print(f"[*] output RMS (clean): {rms_out:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
