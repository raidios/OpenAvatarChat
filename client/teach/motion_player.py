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
from .motion_odometry import (
    MotionReturnParams,
    WheelImuOdometryBackend,
    compute_return_command,
)

if TYPE_CHECKING:
    from serial_comm import XProtocolSerial
    from tracking_controller import TrackingController


# audio player 接口（chat_client.AudioPlayer 实例，duck-typed）：
#   set_motion_pending(active: bool) -> None
PlayerSetPending = Callable[[bool], None]
YawDeltaCallback = Callable[[float], None]
TELEMETRY_PREFIX = "[telemetry][motion]"


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
        yaw_delta_callback: Optional[YawDeltaCallback] = None,
        empty_clip_ms: int = 500,
        motion_return_enabled: bool = True,
        motion_return_params: Optional[MotionReturnParams] = None,
    ):
        self._serial = serial
        self._tracking = tracking_ctl
        self._set_pending = player_set_pending
        self._yaw_delta_callback = yaw_delta_callback
        self._empty_clip_ms = int(empty_clip_ms)
        self._motion_return_enabled = bool(motion_return_enabled)
        self._motion_return_params = motion_return_params or MotionReturnParams()
        self._odometry = WheelImuOdometryBackend(
            serial=serial,
            stale_timeout_s=self._motion_return_params.stale_timeout_s,
        )

        self._lock = threading.Lock()
        self._jobs: Deque[_Job] = deque()
        self._cur_clip_id: Optional[str] = None
        self._cur_started_at: float = 0.0
        self._cur_total_ms: int = 0
        self._cur_progress_ms: int = 0
        self._phase: str = "idle"
        self._phase_history = []
        self._abort = threading.Event()
        self._return_after_abort = True
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
            pose = self._odometry.pose()
            return {
                "current_clip_id": self._cur_clip_id,
                "current_total_ms": self._cur_total_ms,
                "current_progress_ms": self._cur_progress_ms,
                "queued": len(self._jobs),
                "phase": self._phase,
                "pose": {
                    "x_m": pose.x_m,
                    "y_m": pose.y_m,
                    "yaw_deg": pose.yaw_deg,
                },
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
            queued = len(self._jobs)
        clip_id = clip.id if clip is not None else "empty"
        duration_ms = clip.duration_ms if clip is not None else self._empty_clip_ms
        print(
            f"{TELEMETRY_PREFIX} enqueue clip={clip_id} duration_ms={duration_ms} "
            f"queued={queued} was_idle={was_idle}",
            flush=True,
        )
        if was_idle and self._set_pending is not None:
            try:
                self._set_pending(True)
            except Exception as e:
                print(f"[motion_player] set_pending error: {e}", flush=True)
        self._wake.set()

    def stop_all(self, return_to_origin: bool = True) -> None:
        """立刻中止当前 clip + 清空待播队列。"""
        with self._lock:
            self._jobs.clear()
            self._return_after_abort = bool(return_to_origin)
        self._abort.set()
        self._wake.set()

    def set_yaw_delta_callback(self, cb: Optional[YawDeltaCallback]) -> None:
        self._yaw_delta_callback = cb

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
            pose = self._odometry.pose()
            self._phase_history.append({
                "phase": phase,
                "x_m": pose.x_m,
                "y_m": pose.y_m,
                "yaw_deg": pose.yaw_deg,
            })
            if len(self._phase_history) > 64:
                del self._phase_history[:-64]

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
        print(f"{TELEMETRY_PREFIX} release_idle", flush=True)
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
        with self._lock:
            self._return_after_abort = True
        yaw_tracker = _YawDeltaNotifier(
            serial=self._serial,
            callback=lambda d: self._notify_yaw_delta(d),
        )
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
                started_at = self._cur_started_at
            self._set_phase("playing")
            print(
                f"{TELEMETRY_PREFIX} clip_start clip=empty duration_ms={total_ms} samples=0",
                flush=True,
            )
            deadline = time.monotonic() + total_ms / 1000.0
            aborted_by_ps2 = False
            while time.monotonic() < deadline and not self._abort.is_set() and not self._stop.is_set():
                if self._serial.is_ps2_overriding:
                    aborted_by_ps2 = True
                    break
                yaw_tracker.poll()
                with self._lock:
                    self._cur_progress_ms = int((time.monotonic() - self._cur_started_at) * 1000)
                time.sleep(0.02)
            yaw_tracker.flush()
            with self._lock:
                self._cur_clip_id = None
                self._cur_total_ms = 0
                self._cur_progress_ms = 0
            self._set_phase("idle")
            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            print(
                f"{TELEMETRY_PREFIX} clip_done clip=empty elapsed_ms={elapsed_ms:.0f} "
                f"aborted={self._abort.is_set()} ps2={aborted_by_ps2}",
                flush=True,
            )
            return

        # 真实 clip：按 sample 时间戳节拍下发 cmd_vel。
        with self._lock:
            self._cur_clip_id = clip.id
            self._cur_total_ms = clip.duration_ms
            self._cur_progress_ms = 0
            self._cur_started_at = time.monotonic()
            started_at = self._cur_started_at
        self._odometry.reset_origin(now_s=started_at)
        self._set_phase("playing")
        print(
            f"{TELEMETRY_PREFIX} clip_start clip={clip.id} "
            f"duration_ms={clip.duration_ms} samples={len(clip.samples)}",
            flush=True,
        )

        aborted_by_ps2 = False
        should_return = False
        try:
            for sample in clip.samples:
                if self._stop.is_set() or self._abort.is_set():
                    break
                if self._serial.is_ps2_overriding:
                    print("[motion_player] aborted: PS2 override detected", flush=True)
                    aborted_by_ps2 = True
                    break
                target_t = self._cur_started_at + sample.t_ms / 1000.0
                sleep = target_t - time.monotonic()
                if sleep > 0:
                    # 用 wait 让 stop / abort 能及时中断 sleep
                    if self._abort.wait(timeout=sleep):
                        break
                yaw_tracker.poll()
                self._odometry.update()
                self._serial.send_velocity(sample.vx, sample.vy, sample.vw)
                yaw_tracker.poll()
                self._odometry.update()
                with self._lock:
                    self._cur_progress_ms = sample.t_ms
            should_return = (
                self._motion_return_enabled
                and not self._stop.is_set()
                and not aborted_by_ps2
                and (not self._abort.is_set() or self._return_after_abort)
                and self._odometry.healthy()
            )
        finally:
            # 结尾连发几次零速保证停车（300ms freshness 窗内多发几次更稳）。
            for _ in range(5):
                yaw_tracker.poll()
                self._odometry.update()
                self._serial.send_velocity(0, 0, 0)
                time.sleep(0.02)
            yaw_tracker.flush()
            if should_return:
                self._abort.clear()
                self._run_return_to_origin(yaw_tracker)
            with self._lock:
                self._cur_clip_id = None
                self._cur_total_ms = 0
                self._cur_progress_ms = 0
            self._set_phase("idle")
            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            print(
                f"{TELEMETRY_PREFIX} clip_done clip={clip.id} elapsed_ms={elapsed_ms:.0f} "
                f"aborted={self._abort.is_set()} ps2={aborted_by_ps2}",
                flush=True,
            )

    def _run_return_to_origin(self, yaw_tracker: "_YawDeltaNotifier") -> None:
        params = self._motion_return_params
        pose = self._odometry.pose()
        start = time.monotonic()
        print(
            f"{TELEMETRY_PREFIX} return_start "
            f"x_m={pose.x_m:.3f} y_m={pose.y_m:.3f} yaw_deg={pose.yaw_deg:.1f}",
            flush=True,
        )
        self._set_phase("returning")
        deadline = start + max(0.1, float(params.timeout_s))
        period = 1.0 / max(1.0, float(params.control_rate_hz))
        done = False
        reason = "timeout"
        while time.monotonic() < deadline and not self._stop.is_set():
            if self._abort.is_set():
                reason = "aborted"
                break
            if self._serial.is_ps2_overriding:
                reason = "ps2"
                break
            if not self._odometry.healthy():
                reason = "stale"
                break
            yaw_tracker.poll()
            pose = self._odometry.update()
            vx, vw, done = compute_return_command(pose, params)
            if done:
                reason = "done"
                break
            self._serial.send_velocity(int(vx * 1000), 0, int(vw * 1000))
            self._odometry.update()
            time.sleep(period)
        for _ in range(5):
            yaw_tracker.poll()
            self._odometry.update()
            self._serial.send_velocity(0, 0, 0)
            time.sleep(0.02)
        yaw_tracker.flush()
        pose = self._odometry.pose()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        print(
            f"{TELEMETRY_PREFIX} return_done reason={reason} elapsed_ms={elapsed_ms:.0f} "
            f"x_m={pose.x_m:.3f} y_m={pose.y_m:.3f} yaw_deg={pose.yaw_deg:.1f}",
            flush=True,
        )

    def _notify_yaw_delta(self, delta_ccw_deg: float) -> None:
        cb = self._yaw_delta_callback
        if cb is None:
            return
        try:
            cb(float(delta_ccw_deg))
        except Exception as e:
            print(f"[motion_player] yaw_delta callback error: {e}", flush=True)


class _YawDeltaNotifier:
    def __init__(
        self,
        serial: "XProtocolSerial",
        callback: YawDeltaCallback,
        threshold_deg: float = 0.5,
        min_flush_deg: float = 0.05,
    ):
        self._serial = serial
        self._callback = callback
        self._threshold_deg = float(threshold_deg)
        self._min_flush_deg = float(min_flush_deg)
        self._last = self._read_yaw()

    def _read_yaw(self) -> float:
        return float(self._serial.yaw_deg_unwrapped)

    def poll(self) -> None:
        cur = self._read_yaw()
        delta = cur - self._last
        if abs(delta) >= self._threshold_deg:
            self._callback(delta)
            self._last = cur

    def flush(self) -> None:
        cur = self._read_yaw()
        delta = cur - self._last
        if abs(delta) >= self._min_flush_deg:
            self._callback(delta)
            self._last = cur
