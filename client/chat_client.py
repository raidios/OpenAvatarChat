"""
Chat client module -- voice + vision dialogue over WebSocket.

Provides AudioPlayer + async coroutines that the main event loop composes.

设计要点（详见 plan ``动作标签与示教前端``）：

* AudioPlayer 的播放队列是**异构** deque，元素是 ``("audio", bytes)`` 或
  ``("marker", payload_dict)``——保证音频与动作标签按发送顺序串行。
* ``play_chunk()`` 命中 marker 时调用注入的回调（``ActionDispatcher.dispatch``），
  实现"播放到该数据包时触发动作"的语义。
* 引入 ``set_motion_pending()``：MotionPlayer 起播 clip 时置 ``True``、结束置
  ``False``，仅用于 telemetry。``playback_done_event`` 只表达语音播放完成；
  动作独立播放，新的用户语音会中止动作。
* 暴露 ``MicMonitor``：把 mic 采集到的 raw int16 PCM 副本扔进 asyncio.Queue
  ，供 teach server ``/ws/mic`` 推到浏览器做实时监听；mute 期间分流到监听的
  是 raw（mute 前），而上行到 chat 服务端的是置零的 SILENCE_FRAME，互不影响。
"""

import asyncio
import json
import math
import time
import threading
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

import cv2
import numpy as np
import pyaudio
import websockets

from camera import SharedCamera

MSG_TYPE_AUDIO = 0x01
MSG_TYPE_VIDEO = 0x02

MIC_RATE = 16000
MIC_CHANNELS = 1
MIC_CHUNK = 1600  # 100 ms at 16 kHz

SPEAKER_RATE = 24000
SPEAKER_CHANNELS = 1
SPEAKER_CHUNK = 2400  # 100 ms at 24 kHz

VIDEO_FPS = 0.5
JPEG_QUALITY = 60

MUTE_TAIL_SEC = 0.3
FADE_IN_SAMPLES = 480  # 20 ms at 24 kHz

SILENCE_FRAME = bytes(MIC_CHUNK * 2)
TELEMETRY_PREFIX = "[telemetry][chat]"


# 回调签名：tags 形如 [{"name": "happy"}, ...]；phase 是 "before" / "after"。
ActionTagCallback = Callable[[List[dict], str], None]


class MicMonitor:
    """跨线程 fan-out raw mic PCM 到一个或多个监听者。

    使用 ``threading.Lock`` 保护订阅者列表，因为 ``feed`` 由 ``mic_sender``（在
    asyncio 线程的 executor 里、PyAudio blocking read）触发，而 ``subscribe`` /
    ``unsubscribe`` 通常由 teach server 的 ws 协程在主事件循环上调用。

    每个订阅者拿到的是一个 ``asyncio.Queue[bytes]``，bounded 防止慢消费者
    把内存吃光（满了直接丢最旧）。
    """

    def __init__(self, queue_size: int = 20) -> None:
        self._lock = threading.Lock()
        self._subscribers: List[Tuple[asyncio.AbstractEventLoop, "asyncio.Queue[bytes]"]] = []
        self._queue_size = queue_size

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> "asyncio.Queue[bytes]":
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subscribers.append((loop, q))
        return q

    def unsubscribe(self, q: "asyncio.Queue[bytes]") -> None:
        with self._lock:
            self._subscribers = [(l, qq) for (l, qq) in self._subscribers if qq is not q]

    def feed(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        with self._lock:
            subs = list(self._subscribers)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(self._put_drop_oldest, q, pcm_bytes)
            except RuntimeError:
                # loop 已关闭：忽略，等下一次 subscribe 重新注册。
                pass

    @staticmethod
    def _put_drop_oldest(q: "asyncio.Queue[bytes]", item: bytes) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass


class AudioPlayer:
    """24k mono PCM 播放器 + 异构标签队列。

    队列元素：``("audio", bytes)`` 或 ``("marker", payload_dict)``。
    play_chunk 每次从队首取一项；若 audio 则写 PyAudio 流；若 marker 则调用
    注入的 ``on_marker(payload)``（不阻塞主循环——回调内部应 fire-and-forget）。
    """

    def __init__(self, on_marker: Optional[Callable[[dict], None]] = None):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=SPEAKER_CHANNELS,
            rate=SPEAKER_RATE,
            output=True,
            frames_per_buffer=SPEAKER_CHUNK,
        )
        self.buffer: Deque[Tuple[str, Any]] = deque()
        self.lock = threading.Lock()
        self.audio_end_received = False
        self.playback_done_event = asyncio.Event()
        self.is_playing = False
        self.mute_until: float = 0.0
        self._need_fade_in = False
        self._on_marker = on_marker
        # MotionPlayer 起播/结束时翻转；仅用于 telemetry，不阻塞 playback_done。
        self._motion_pending = False
        self._playback_seq = 0
        self._playback_id = "idle"
        self._playback_started_at: Optional[float] = None
        self._last_audio_at: Optional[float] = None
        self._audio_frames = 0
        self._audio_bytes = 0
        self._markers_seen = 0
        self._end_received_at: Optional[float] = None

    @property
    def mic_should_mute(self) -> bool:
        return self.is_playing or time.monotonic() < self.mute_until

    # ------------------------------------------------------------------
    # producer API
    # ------------------------------------------------------------------

    def enqueue_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        with self.lock:
            if not self.is_playing:
                self._need_fade_in = True
                self._start_playback_locked()
            self.buffer.append(("audio", pcm_bytes))
            self.is_playing = True
            self._last_audio_at = time.monotonic()
            self._audio_frames += 1
            self._audio_bytes += len(pcm_bytes)
            playback_id = self._playback_id
            queue_len = len(self.buffer)
            frames = self._audio_frames
            audio_ms = self._audio_bytes / (SPEAKER_RATE * SPEAKER_CHANNELS * 2) * 1000.0
        if frames == 1 or frames % 50 == 0:
            print(
                f"{TELEMETRY_PREFIX} audio_enqueue id={playback_id} "
                f"frames={frames} queued={queue_len} audio_ms={audio_ms:.0f}",
                flush=True,
            )

    # 兼容旧名（其它代码可能引用），等价 enqueue_audio。
    enqueue = enqueue_audio

    def enqueue_marker(self, payload: dict) -> None:
        with self.lock:
            self.buffer.append(("marker", payload))
            self._markers_seen += 1
            playback_id = self._playback_id
            queue_len = len(self.buffer)
            markers = self._markers_seen
        tags = payload.get("tags") or payload.get("tail_action_tags") or []
        names = ",".join(str(t.get("name", t)) for t in tags if isinstance(t, dict)) or "-"
        print(
            f"{TELEMETRY_PREFIX} marker_enqueue id={playback_id} "
            f"markers={markers} queued={queue_len} phase={payload.get('phase', '-')} tags={names}",
            flush=True,
        )

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()
            self.audio_end_received = False
            self.is_playing = False
            self._need_fade_in = False
            playback_id = self._playback_id
            self._reset_playback_stats_locked()
        # 中断时同步重置 done event；motion_pending 由 dispatcher 在 motion 取消时清。
        self.playback_done_event.clear()
        print(f"{TELEMETRY_PREFIX} playback_clear id={playback_id}", flush=True)

    def mark_audio_end(self) -> None:
        with self.lock:
            self.audio_end_received = True
            self._end_received_at = time.monotonic()
            playback_id = self._playback_id
            queue_len = len(self.buffer)
            frames = self._audio_frames
            audio_ms = self._audio_bytes / (SPEAKER_RATE * SPEAKER_CHANNELS * 2) * 1000.0
        print(
            f"{TELEMETRY_PREFIX} audio_end id={playback_id} "
            f"queued={queue_len} frames={frames} audio_ms={audio_ms:.0f}",
            flush=True,
        )

    def set_motion_pending(self, active: bool) -> None:
        """MotionPlayer 起播/结束时调用。控制 playback_done_event 是否可以触发。"""
        with self.lock:
            old = self._motion_pending
            self._motion_pending = bool(active)
            playback_id = self._playback_id
            queue_len = len(self.buffer)
            is_playing = self.is_playing
        if old != bool(active):
            print(
                f"{TELEMETRY_PREFIX} motion_pending id={playback_id} "
                f"active={bool(active)} queued={queue_len} is_playing={is_playing}",
                flush=True,
            )

    def _start_playback_locked(self) -> None:
        self._playback_seq += 1
        self._playback_id = f"pb{self._playback_seq:06d}"
        now = time.monotonic()
        self._playback_started_at = now
        self._last_audio_at = None
        self._audio_frames = 0
        self._audio_bytes = 0
        self._markers_seen = 0
        self._end_received_at = None
        print(f"{TELEMETRY_PREFIX} playback_start id={self._playback_id}", flush=True)

    def _reset_playback_stats_locked(self) -> None:
        self._playback_id = "idle"
        self._playback_started_at = None
        self._last_audio_at = None
        self._audio_frames = 0
        self._audio_bytes = 0
        self._markers_seen = 0
        self._end_received_at = None

    def _finish_playback_locked(self) -> dict:
        now = time.monotonic()
        playback_id = self._playback_id
        started = self._playback_started_at
        end_received = self._end_received_at
        audio_ms = self._audio_bytes / (SPEAKER_RATE * SPEAKER_CHANNELS * 2) * 1000.0
        stats = {
            "id": playback_id,
            "wall_ms": (now - started) * 1000.0 if started is not None else math.nan,
            "after_audio_end_ms": (now - end_received) * 1000.0 if end_received is not None else math.nan,
            "frames": self._audio_frames,
            "audio_ms": audio_ms,
            "markers": self._markers_seen,
        }
        self._reset_playback_stats_locked()
        return stats

    # ------------------------------------------------------------------
    # consumer side (called from audio_playback task)
    # ------------------------------------------------------------------

    def play_chunk(self) -> bool:
        """从队列取出一项处理。

        Returns True 表示"取到了非 marker 内容"（即有音频写了一帧），调用者可以
        立即取下一帧；False 表示"队列空"或"刚处理完 marker"，调用者应短暂 sleep。
        """
        with self.lock:
            if not self.buffer:
                if self.audio_end_received:
                    stats = self._finish_playback_locked()
                    self.audio_end_received = False
                    self.is_playing = False
                    self.mute_until = time.monotonic() + MUTE_TAIL_SEC
                    self.playback_done_event.set()
                    print(
                        f"{TELEMETRY_PREFIX} playback_done "
                        f"id={stats['id']} wall_ms={stats['wall_ms']:.0f} "
                        f"after_audio_end_ms={stats['after_audio_end_ms']:.0f} "
                        f"audio_ms={stats['audio_ms']:.0f} frames={stats['frames']} "
                        f"markers={stats['markers']} mute_tail_ms={MUTE_TAIL_SEC * 1000:.0f}",
                        flush=True,
                    )
                return False
            kind, payload = self.buffer.popleft()
            apply_fade = False
            if kind == "audio":
                apply_fade = self._need_fade_in
                self._need_fade_in = False

        if kind == "marker":
            cb = self._on_marker
            if cb is not None:
                try:
                    cb(payload)
                except Exception as e:
                    print(f"[player] marker callback error: {e}", flush=True)
            return False

        # audio
        data = payload
        if apply_fade:
            data = self._apply_fade_in(data)
        self.stream.write(data)
        return True

    @staticmethod
    def _apply_fade_in(pcm_bytes: bytes) -> bytes:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
        fade_len = min(FADE_IN_SAMPLES, len(samples))
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        samples[:fade_len] = (samples[:fade_len].astype(np.float32) * ramp).astype(np.int16)
        return samples.tobytes()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


async def mic_sender(
    ws, pa_instance, player: AudioPlayer, stop_event: asyncio.Event,
    mic_monitor: Optional[MicMonitor] = None,
    audio_source: Optional[Any] = None,
):
    """读取麦克风 → 上行（mute 期间发 SILENCE）；同时把 raw PCM 副本喂 ``mic_monitor``。

    若传入 ``audio_source``（非 None），则使用其 ``read(n_samples)`` 接口取
    16-kHz int16 mono PCM，否则用 PyAudio 默认输入设备打开一个流。这是
    far-field DSP 前端（M260C → DspPipeline）的注入点：``audio_source`` 实
    现 ``read()`` / ``stop_stream()`` / ``start_stream()`` 即可，参见
    ``client/farfield_audio_source.py``。
    """
    if audio_source is not None:
        stream = audio_source
        opened_pa_stream = False
    else:
        stream = pa_instance.open(
            format=pyaudio.paInt16,
            channels=MIC_CHANNELS,
            rate=MIC_RATE,
            input=True,
            frames_per_buffer=MIC_CHUNK,
        )
        opened_pa_stream = True
    loop = asyncio.get_event_loop()
    last_mute = None
    last_stat_t = time.monotonic()
    sum_sq = 0.0
    samples = 0
    peak = 0
    sent_real = 0
    sent_silence = 0
    try:
        while not stop_event.is_set():
            data = await loop.run_in_executor(None, stream.read, MIC_CHUNK, False)
            mute = player.mic_should_mute
            if mute != last_mute:
                if hasattr(stream, "enable_wake_spotter"):
                    try:
                        stream.enable_wake_spotter(not mute)
                    except Exception as e:
                        print(f"[mic] wake_spotter gate error: {e}", flush=True)
                print(
                    f"[mic] mute={mute} is_playing={player.is_playing} "
                    f"mute_until_in={player.mute_until - time.monotonic():.2f}s",
                    flush=True,
                )
                last_mute = mute
            raw = data
            # 监听通道始终拿 raw（mute 前），便于调试时听到真实声音。
            if mic_monitor is not None:
                mic_monitor.feed(raw)
            if mute:
                data = SILENCE_FRAME
                sent_silence += 1
            else:
                sent_real += 1
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                if arr.size:
                    sum_sq += float(np.sum(arr * arr))
                    samples += arr.size
                    p = int(np.max(np.abs(arr)))
                    if p > peak:
                        peak = p
            msg = bytes([MSG_TYPE_AUDIO]) + data
            await ws.send(msg)

            now = time.monotonic()
            if now - last_stat_t >= 1.0:
                rms = int((sum_sq / samples) ** 0.5) if samples else 0
                pk_db = (
                    20.0 * np.log10(max(1e-6, peak / 32768.0)) if peak else float("-inf")
                )
                print(
                    f"[mic] real={sent_real} silence={sent_silence} "
                    f"rms={rms} peak_dbFS={pk_db:.1f}",
                    flush=True,
                )
                last_stat_t = now
                sum_sq = 0.0
                samples = 0
                peak = 0
                sent_real = 0
                sent_silence = 0
    except Exception as e:
        if not stop_event.is_set():
            print(f"[mic] Error: {e}")
    finally:
        try:
            stream.stop_stream()
        except Exception:
            pass
        if opened_pa_stream:
            try:
                stream.close()
            except Exception:
                pass


async def video_sender(
    ws,
    camera: SharedCamera,
    stop_event: asyncio.Event,
):
    interval = 1.0 / VIDEO_FPS
    if not camera.is_opened:
        print("[video] Camera not available, video sending disabled.")
        return
    try:
        while not stop_event.is_set():
            frame, _ = camera.get_frame()
            if frame is not None:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                msg = bytes([MSG_TYPE_VIDEO]) + jpeg.tobytes()
                await ws.send(msg)
            await asyncio.sleep(interval)
    except Exception as e:
        if not stop_event.is_set():
            print(f"[video] Error: {e}")


async def ws_receiver(
    ws,
    player: AudioPlayer,
    stop_event: asyncio.Event,
    on_server_json: Optional[Callable[[dict, Any], bool]] = None,
    expression_ui=None,
):
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
                    player.enqueue_audio(payload)
            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    if expression_ui is not None:
                        expression_ui.on_ws_json(data)
                    handled = (
                        on_server_json(data, player)
                        if on_server_json is not None else False
                    )
                    if not handled:
                        msg_type = data.get("type")
                        if msg_type == "session_started":
                            print(f"[session] Connected: {data.get('session_id')}")
                        elif msg_type == "asr_text":
                            print(f"[ASR] {data.get('text', '')}")
                        elif msg_type == "llm_text":
                            print(f"[LLM] {data.get('text', '')}", end="", flush=True)
                        elif msg_type == "audio_end":
                            player.mark_audio_end()
                        elif msg_type == "action_tag":
                            # 与 audio 同源同序进同一队列，由 play_chunk 在出队时
                            # 触发回调。payload 形如 {"tags": [...], "phase": "before"}。
                            player.enqueue_marker(data)
                except json.JSONDecodeError:
                    pass
    except websockets.exceptions.ConnectionClosed:
        if not stop_event.is_set():
            print("[ws] Connection closed by server")
    except Exception as e:
        if not stop_event.is_set():
            print(f"[ws] Receive error: {e}")
    finally:
        if expression_ui is not None:
            expression_ui.reset_session()
        stop_event.set()


async def audio_playback(player: AudioPlayer, stop_event: asyncio.Event):
    while not stop_event.is_set():
        if not player.play_chunk():
            await asyncio.sleep(0.01)


async def playback_notifier(
    ws, player: AudioPlayer, stop_event: asyncio.Event
):
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(player.playback_done_event.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        if stop_event.is_set():
            break
        player.playback_done_event.clear()
        print("\n[TTS] Playback complete, notifying server")
        try:
            await ws.send(json.dumps({"type": "playback_complete"}))
            print(f"{TELEMETRY_PREFIX} playback_complete_sent", flush=True)
        except Exception as e:
            if not stop_event.is_set():
                print(f"[playback_notifier] Error: {e}")


class ChatClient:
    """Wraps all chat-related coroutines for easy integration."""

    def __init__(
        self,
        on_action_tag: Optional[ActionTagCallback] = None,
        on_human_speech_start: Optional[Callable[[], None]] = None,
        audio_source: Optional[Any] = None,
    ):
        self.pa_instance = pyaudio.PyAudio()
        self._on_action_tag = on_action_tag
        self._on_human_speech_start = on_human_speech_start
        # 由 AudioPlayer 的 marker 回调转发：把 payload 拆成 tags + phase 后向上派发。
        self.player = AudioPlayer(on_marker=self._dispatch_marker)
        self.mic_monitor = MicMonitor()
        # 当 audio_source 非 None 时，mic_sender 会用它的 read() 取麦克风音
        # 频，而不是用 PyAudio 默认设备。far-field DSP 链路从这里注入。
        self.audio_source = audio_source
        # WS handle + the asyncio loop it was opened on, for thread-safe
        # cross-thread sends (e.g., wake_word fired from the audio
        # thread when client-side KWS hits).
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _dispatch_marker(self, payload: dict) -> None:
        if self._on_action_tag is None:
            return
        tags = payload.get("tags") or []
        phase = payload.get("phase", "before")
        if not tags:
            return
        try:
            self._on_action_tag(tags, phase)
        except Exception as e:
            print(f"[chat_client] on_action_tag error: {e}", flush=True)

    def set_action_tag_callback(self, cb: Optional[ActionTagCallback]) -> None:
        """允许在 ChatClient 构造之后再挂载回调（main.py 里 dispatcher 晚于 ChatClient 起来时用）。"""
        self._on_action_tag = cb

    def set_human_speech_start_callback(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_human_speech_start = cb

    def _notify_human_speech_start(self) -> None:
        cb = self._on_human_speech_start
        if cb is None:
            return
        try:
            cb()
        except Exception as e:
            print(f"[chat_client] on_human_speech_start error: {e}", flush=True)

    def handle_server_json(self, data: dict, player: Optional[AudioPlayer]) -> bool:
        msg_type = data.get("type")
        if msg_type == "session_started":
            print(f"[session] Connected: {data.get('session_id')}")
            return True
        if msg_type == "human_speech_start":
            print("[ASR] human speech start", flush=True)
            self._notify_human_speech_start()
            return True
        if msg_type == "asr_text":
            self._notify_human_speech_start()
            print(f"[ASR] {data.get('text', '')}")
            return True
        if msg_type == "llm_text":
            print(f"[LLM] {data.get('text', '')}", end="", flush=True)
            return True
        if msg_type == "audio_end":
            if player is not None:
                player.mark_audio_end()
            return True
        if msg_type == "action_tag":
            if player is not None:
                player.enqueue_marker(data)
            return True
        return False

    async def run(
        self,
        server_url: str,
        camera: Optional[SharedCamera],
        stop_event: asyncio.Event,
        enable_video: bool = True,
        expression_ui=None,
    ):
        print(f"[chat] Connecting to {server_url} ...")
        try:
            async with websockets.connect(
                server_url,
                max_size=10 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=60,
            ) as ws:
                print("[chat] Connected.")
                self._ws = ws
                self._loop = asyncio.get_running_loop()
                tasks = [
                    asyncio.create_task(
                        mic_sender(
                            ws, self.pa_instance, self.player, stop_event,
                            self.mic_monitor,
                            audio_source=self.audio_source,
                        )
                    ),
                    asyncio.create_task(
                        ws_receiver(
                            ws, self.player, stop_event,
                            on_server_json=self.handle_server_json,
                            expression_ui=expression_ui,
                        )
                    ),
                    asyncio.create_task(
                        audio_playback(self.player, stop_event)
                    ),
                    asyncio.create_task(
                        playback_notifier(ws, self.player, stop_event)
                    ),
                ]
                if enable_video and camera is not None and camera.is_opened:
                    tasks.append(
                        asyncio.create_task(
                            video_sender(ws, camera, stop_event)
                        )
                    )
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"[chat] Connection error: {e}")
        finally:
            self._ws = None
            self._loop = None

    def notify_wake(
        self, keyword: str, doa_body_deg: Optional[float] = None
    ) -> None:
        """Tell the server "client KWS fired"; thread-safe.

        Schedules a ``{"type": "wake_word", ...}`` JSON control
        frame on the active WebSocket, regardless of which thread
        called us. No-op if no chat connection is up. Does not
        block — drops the message if the loop is busy/closing.
        """
        ws = self._ws
        loop = self._loop
        if ws is None or loop is None or loop.is_closed():
            return
        msg = json.dumps({
            "type": "wake_word",
            "keyword": str(keyword),
            "doa_body_deg": (
                float(doa_body_deg) if doa_body_deg is not None else None
            ),
        })

        async def _send():
            try:
                await ws.send(msg)
            except Exception as e:
                print(f"[chat_client] notify_wake send failed: {e}",
                      flush=True)

        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except Exception as e:
            print(f"[chat_client] notify_wake schedule failed: {e}",
                  flush=True)

    def close(self):
        self.player.close()
        self.pa_instance.terminate()
