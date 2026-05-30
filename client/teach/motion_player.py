"""
MotionPlayer：按 clip 样本时间戳回放 cmd_vel。

设计目标：

* 单一后台线程顺序播放队列里的 clip；同时只播一个，避免抖动叠加。
* 起播前调用 ``tracking_ctl.pause()`` 让出 PI 通道、并调 ``audio_player.set_motion_pending(True)``；
  播完（或中断）调 ``tracking_ctl.resume_from_pause()`` + ``set_motion_pending(False)``。
* 与 PS2 共存：firmware arbitration 保证 PS2 优先；播放循环每拍检查
  ``serial.is_ps2_overriding``，若 True 立刻中止剩余样本并清空队列（操作员
  显然想要接管）。
* ``clip=None`` 兜底：当 ``action_registry`` 给某情绪还没绑动作 clip 时仍要
  让"动作未结束 == TTS 未结束"的语义成立。我们用一个 ~500ms 的"空 pending"
  顶上：不发任何 cmd_vel，但 ``motion_pending`` 在窗内保持 True。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Deque, Optional, TYPE_CHECKING
from collections import deque

from .clip_store import Clip

if TYPE_CHECKING:
    from serial_comm import XProtocolSerial
    from tracking_controller import TrackingController


# audio player 接口（chat_client.AudioPlayer 实例，duck-typed）：
#   set_motion_pending(active: bool) -> None
PlayerSetPending = Callable[[bool], None]


@dataclass
class _Job:
    clip: Optional[Clip]
    placeholder_ms: int   # 当 clip=None 时使用；ignored when clip is not None
    on_done: Optional[Callable[[], None]] = None


class MotionPlayer:
    def __init__(
        self,
        serial: "XProtocolSerial",
        tracking_ctl: Optional["TrackingController"] = None,
        player_set_pending: Optional[PlayerSetPending] = None,
        empty_clip_ms: int = 500,
    ):
        self._serial = serial
        self._tracking = tracking_ctl
        self._set_pending = player_set_pending
        self._empty_clip_ms = int(empty_clip_ms)

        self._lock = threading.Lock()
        self._jobs: Deque[_Job] = deque()
        self._cur_clip_id: Optional[str] = None
        self._cur_started_at: float = 0.0
        self._cur_total_ms: int = 0
        self._cur_progress_ms: int = 0
        self._abort = threading.Event()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start_thread(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="motion-player")
        self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def is_busy(self) -> bool:
        with self._lock:
            return self._cur_clip_id is not None or len(self._jobs) > 0

    def status(self) -> dict:
        with self._lock:
            return {
                "current_clip_id": self._cur_clip_id,
                "current_total_ms": self._cur_total_ms,
                "current_progress_ms": self._cur_progress_ms,
                "queued": len(self._jobs),
            }

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------

    def enqueue(self, clip: Optional[Clip], on_done: Optional[Callable[[], None]] = None) -> None:
        """把一个 clip（或 None 占位）排进播放队列。

        队列非空时 set_motion_pending(True) 持续保持，所有 job 都完成或被中断
        后才回到 False。
        """
        with self._lock:
            was_idle = self._cur_clip_id is None and not self._jobs
            self._jobs.append(_Job(clip=clip, placeholder_ms=self._empty_clip_ms, on_done=on_done))
        if was_idle and self._set_pending is not None:
            try:
                self._set_pending(True)
            except Exception as e:
                print(f"[motion_player] set_pending error: {e}", flush=True)
        self._wake.set()

    def stop_all(self) -> None:
        """立刻中止当前 clip + 清空待播队列。"""
        with self._lock:
            self._jobs.clear()
        self._abort.set()
        self._wake.set()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _pop_job(self) -> Optional[_Job]:
        with self._lock:
            if not self._jobs:
                return None
            return self._jobs.popleft()

    def _release_idle(self) -> None:
        """当 jobs 与当前 clip 都清空时执行：恢复 tracking + clear pending。"""
        if self._tracking is not None:
            try:
                self._tracking.resume_from_pause()
            except Exception as e:
                print(f"[motion_player] tracking resume error: {e}", flush=True)
        if self._set_pending is not None:
            try:
                self._set_pending(False)
            except Exception as e:
                print(f"[motion_player] clear pending error: {e}", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self._pop_job()
            if job is None:
                # 队列空：等下一次 enqueue 唤醒
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue

            # 起播前接管 cmd_vel
            if self._tracking is not None:
                try:
                    self._tracking.pause()
                except Exception as e:
                    print(f"[motion_player] tracking pause error: {e}", flush=True)

            try:
                self._play_one(job)
            finally:
                if job.on_done is not None:
                    try:
                        job.on_done()
                    except Exception as e:
                        print(f"[motion_player] on_done error: {e}", flush=True)

            # 仅当所有 jobs 都耗尽才 release（允许排队的下一个无缝衔接）。
            with self._lock:
                still_busy = bool(self._jobs)
            if not still_busy:
                self._release_idle()

    def _play_one(self, job: _Job) -> None:
        self._abort.clear()
        clip = job.clip
        if clip is None or not clip.samples:
            # 占位 pending：不下发任何 cmd_vel，但维持 motion_pending 一段时间。
            placeholder = max(job.placeholder_ms, 100)
            total_ms = placeholder
            with self._lock:
                self._cur_clip_id = None  # 仍标记 None 表示占位
                self._cur_total_ms = total_ms
                self._cur_progress_ms = 0
                self._cur_started_at = time.monotonic()
            deadline = time.monotonic() + total_ms / 1000.0
            while time.monotonic() < deadline and not self._abort.is_set() and not self._stop.is_set():
                if self._serial.is_ps2_overriding:
                    break
                with self._lock:
                    self._cur_progress_ms = int((time.monotonic() - self._cur_started_at) * 1000)
                time.sleep(0.02)
            with self._lock:
                self._cur_clip_id = None
                self._cur_total_ms = 0
                self._cur_progress_ms = 0
            return

        # 真实 clip：按 sample 时间戳节拍下发 cmd_vel。
        with self._lock:
            self._cur_clip_id = clip.id
            self._cur_total_ms = clip.duration_ms
            self._cur_progress_ms = 0
            self._cur_started_at = time.monotonic()

        try:
            for sample in clip.samples:
                if self._stop.is_set() or self._abort.is_set():
                    break
                if self._serial.is_ps2_overriding:
                    print("[motion_player] aborted: PS2 override detected", flush=True)
                    break
                target_t = self._cur_started_at + sample.t_ms / 1000.0
                sleep = target_t - time.monotonic()
                if sleep > 0:
                    # 用 wait 让 stop / abort 能及时中断 sleep
                    if self._abort.wait(timeout=sleep):
                        break
                self._serial.send_velocity(sample.vx, sample.vy, sample.vw)
                with self._lock:
                    self._cur_progress_ms = sample.t_ms
        finally:
            # 结尾连发几次零速保证停车（300ms freshness 窗内多发几次更稳）。
            for _ in range(5):
                self._serial.send_velocity(0, 0, 0)
                time.sleep(0.02)
            with self._lock:
                self._cur_clip_id = None
                self._cur_total_ms = 0
                self._cur_progress_ms = 0
