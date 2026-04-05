#!/usr/bin/env python3
"""
WebSocket audio/video client for OpenAvatarChat RPi voice+vision system.

Captures microphone audio and camera frames, sends them to the server,
and plays back TTS audio received from the server.

Usage:
    python ws_audio_client.py --server ws://HOST:PORT/ws/chat
"""

import argparse
import asyncio
import json
import signal
import struct
import sys
import threading
from collections import deque

import cv2
import numpy as np
import pyaudio
import websockets

MSG_TYPE_AUDIO = 0x01
MSG_TYPE_VIDEO = 0x02

MIC_RATE = 16000
MIC_CHANNELS = 1
MIC_CHUNK = 1600  # 100ms at 16kHz

SPEAKER_RATE = 24000
SPEAKER_CHANNELS = 1
SPEAKER_CHUNK = 2400  # 100ms at 24kHz

VIDEO_FPS = 0.5  # send a frame every 2 seconds
JPEG_QUALITY = 60


class AudioPlayer:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=SPEAKER_CHANNELS,
            rate=SPEAKER_RATE,
            output=True,
            frames_per_buffer=SPEAKER_CHUNK,
        )
        self.buffer = deque()
        self.lock = threading.Lock()

    def enqueue(self, pcm_bytes: bytes):
        with self.lock:
            self.buffer.append(pcm_bytes)

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def play_chunk(self) -> bool:
        with self.lock:
            if not self.buffer:
                return False
            data = self.buffer.popleft()
        self.stream.write(data)
        return True

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


async def mic_sender(ws, pa_instance, stop_event: asyncio.Event):
    """Continuously read from mic and send audio frames."""
    stream = pa_instance.open(
        format=pyaudio.paInt16,
        channels=MIC_CHANNELS,
        rate=MIC_RATE,
        input=True,
        frames_per_buffer=MIC_CHUNK,
    )
    loop = asyncio.get_event_loop()
    try:
        while not stop_event.is_set():
            data = await loop.run_in_executor(None, stream.read, MIC_CHUNK, False)
            msg = bytes([MSG_TYPE_AUDIO]) + data
            await ws.send(msg)
    except Exception as e:
        if not stop_event.is_set():
            print(f"[mic] Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()


async def video_sender(ws, stop_event: asyncio.Event, camera_id: int = 0):
    """Periodically capture camera frames and send as JPEG."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[video] Camera not available, video sending disabled.")
        return

    interval = 1.0 / VIDEO_FPS
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                msg = bytes([MSG_TYPE_VIDEO]) + jpeg.tobytes()
                await ws.send(msg)
            await asyncio.sleep(interval)
    except Exception as e:
        if not stop_event.is_set():
            print(f"[video] Error: {e}")
    finally:
        cap.release()


async def ws_receiver(ws, player: AudioPlayer, stop_event: asyncio.Event):
    """Receive messages from the server and handle them."""
    try:
        async for message in ws:
            if stop_event.is_set():
                break
            if isinstance(message, bytes):
                if len(message) < 2:
                    continue
                msg_type = message[0]
                payload = message[1:]
                if msg_type == MSG_TYPE_AUDIO:
                    player.enqueue(payload)
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    if msg_type == "session_started":
                        print(f"[session] Connected: {data.get('session_id')}")
                    elif msg_type == "asr_text":
                        print(f"[ASR] {data.get('text', '')}")
                    elif msg_type == "llm_text":
                        print(f"[LLM] {data.get('text', '')}", end="", flush=True)
                    elif msg_type == "audio_end":
                        print("\n[TTS] Audio playback complete")
                except json.JSONDecodeError:
                    pass
    except websockets.exceptions.ConnectionClosed:
        if not stop_event.is_set():
            print("[ws] Connection closed by server")
    except Exception as e:
        if not stop_event.is_set():
            print(f"[ws] Receive error: {e}")
    finally:
        stop_event.set()


async def audio_playback(player: AudioPlayer, stop_event: asyncio.Event):
    """Play audio from the buffer."""
    while not stop_event.is_set():
        if not player.play_chunk():
            await asyncio.sleep(0.01)


async def run_client(server_url: str, camera_id: int = 0, enable_video: bool = True):
    """Main client logic."""
    pa_instance = pyaudio.PyAudio()
    player = AudioPlayer()
    stop_event = asyncio.Event()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    print(f"Connecting to {server_url} ...")
    try:
        async with websockets.connect(
            server_url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            print("Connected. Press Ctrl+C to exit.")
            tasks = [
                asyncio.create_task(mic_sender(ws, pa_instance, stop_event)),
                asyncio.create_task(ws_receiver(ws, player, stop_event)),
                asyncio.create_task(audio_playback(player, stop_event)),
            ]
            if enable_video:
                tasks.append(asyncio.create_task(video_sender(ws, stop_event, camera_id)))

            await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        player.close()
        pa_instance.terminate()
        print("Client stopped.")


def main():
    parser = argparse.ArgumentParser(description="OpenAvatarChat WebSocket Audio/Video Client")
    parser.add_argument(
        "--server",
        type=str,
        default="ws://localhost:8282/ws/chat",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video capture",
    )
    args = parser.parse_args()

    asyncio.run(run_client(args.server, args.camera, not args.no_video))


if __name__ == "__main__":
    main()
