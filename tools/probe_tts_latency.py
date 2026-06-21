#!/usr/bin/env python3
"""Probe DashScope TTS model/voice latency.

Examples:
  python tools/probe_tts_latency.py --model qwen3-tts-flash --voice Cherry
  python tools/probe_tts_latency.py --model cosyvoice-v3.5-plus --voice longjielidou
"""

from __future__ import annotations

import argparse
import base64
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _audio_ms(num_int16_samples: int, sample_rate: int) -> float:
    return num_int16_samples / float(sample_rate) * 1000.0


def probe_multimodal(model: str, voice: str, text: str, sample_rate: int) -> dict:
    from dashscope import MultiModalConversation

    started = time.monotonic()
    first_audio_at: Optional[float] = None
    chunks = 0
    samples = 0
    responses = MultiModalConversation.call(
        model=model,
        text=text,
        voice=voice,
        stream=True,
    )
    for chunk in responses:
        if chunk.output is None:
            raise RuntimeError(f"empty output chunk: {chunk}")
        audio_output = (
            chunk.output.get("audio") if isinstance(chunk.output, dict)
            else getattr(chunk.output, "audio", None)
        )
        if audio_output is not None:
            audio_b64 = (
                audio_output.get("data") if isinstance(audio_output, dict)
                else getattr(audio_output, "data", None)
            )
            if audio_b64:
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                audio_bytes = base64.b64decode(audio_b64)
                samples += np.frombuffer(audio_bytes, dtype=np.int16).size
                chunks += 1
        finish_reason = (
            chunk.output.get("finish_reason") if isinstance(chunk.output, dict)
            else getattr(chunk.output, "finish_reason", None)
        )
        if finish_reason == "stop":
            break
    ended = time.monotonic()
    return {
        "provider": "multimodal",
        "first_ms": (first_audio_at - started) * 1000.0 if first_audio_at else -1.0,
        "total_ms": (ended - started) * 1000.0,
        "audio_ms": _audio_ms(samples, sample_rate),
        "chunks": chunks,
        "samples": samples,
        "error": "",
    }


def probe_tts_v2(model: str, voice: str, text: str, sample_rate: int) -> dict:
    from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

    started = time.monotonic()
    first_audio_at: Optional[float] = None
    chunks = 0
    samples = 0
    done_event = threading.Event()
    error_holder: list[Optional[str]] = [None]

    class Callback(ResultCallback):
        def on_data(self, data: bytes) -> None:
            nonlocal first_audio_at, chunks, samples
            if not data:
                return
            if first_audio_at is None:
                first_audio_at = time.monotonic()
            samples += np.frombuffer(data, dtype=np.int16).size
            chunks += 1

        def on_complete(self) -> None:
            done_event.set()

        def on_error(self, message) -> None:
            error_holder[0] = str(message)
            done_event.set()

        def on_close(self) -> None:
            done_event.set()

    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        callback=Callback(),
        format=AudioFormat.PCM_24000HZ_MONO_16BIT,
    )
    synthesizer.streaming_call(text)
    synthesizer.streaming_complete()
    completed = done_event.wait(timeout=30)
    ended = time.monotonic()
    if not completed:
        raise TimeoutError("tts_v2 synthesis did not complete within 30s")
    if error_holder[0] and samples == 0:
        raise RuntimeError(error_holder[0])
    return {
        "provider": "tts_v2",
        "first_ms": (first_audio_at - started) * 1000.0 if first_audio_at else -1.0,
        "total_ms": (ended - started) * 1000.0,
        "audio_ms": _audio_ms(samples, sample_rate),
        "chunks": chunks,
        "samples": samples,
        "error": error_holder[0] or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-tts-flash")
    parser.add_argument("--voice", default="Cherry")
    parser.add_argument("--text", default="你好，我是小黑。")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--provider", choices=["auto", "multimodal", "tts_v2"], default="auto")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = parser.parse_args()

    _load_dotenv(args.dotenv)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set")

    import dashscope

    dashscope.api_key = api_key
    provider = args.provider
    if provider == "auto":
        provider = "tts_v2" if args.model.lower().startswith("cosyvoice-") else "multimodal"

    if provider == "tts_v2":
        result = probe_tts_v2(args.model, args.voice, args.text, args.sample_rate)
    else:
        result = probe_multimodal(args.model, args.voice, args.text, args.sample_rate)

    print(
        "model={model} voice={voice} provider={provider} "
        "first_ms={first_ms:.0f} total_ms={total_ms:.0f} "
        "audio_ms={audio_ms:.0f} chunks={chunks} samples={samples} error={error}".format(
            model=args.model,
            voice=args.voice,
            **result,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
