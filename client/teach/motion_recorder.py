"""
MotionRecorder：把 PS2 驱动下的 MCU 目标速度按 50Hz 采样成时间序列。

数据来源：``XProtocolSerial.latest_ctrl_state``，其中 ``tg_vx/tg_vy/tg_vw``
是 MCU 在 PS2 驱动时上报的"实际下发到电机"的目标速度——即操作员摇杆映射后
的结果。直接录这一份比录 raw PS2 sticks 通用：将来如果换 PS2 → 键盘示教，
只要还能在 ctrl_state 看到 PI/PS2 算过的目标速度就能复用。

线程模型：单一后台 daemon 线程，每 20ms（50Hz）拍一次；用 ``threading.Event``
驱动状态切换。所有公开方法都是线程安全的。

录制完成后调用 ``snapshot()`` 取一份当前会话的 ``ClipSample`` 列表；
``save(name, store, crop_start_ms, crop_end_ms)`` 把（可能裁剪后的）样本写入
``ClipStore``。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .clip_store import Clip, ClipSample, ClipStore

if TYPE_CHECKING:
    from serial_comm import XProtocolSerial


@dataclass
class RecorderStatus:
    state: str         # "idle" | "recording"
    sample_count: int
    duration_ms: int


class MotionRecorder:
    def __init__(self, serial: "XProtocolSerial", sample_hz: float = 50.0):
        self._serial = serial
        self._period = 1.0 / float(sample_hz)
        self._lock = threading.Lock()
        self._samples: List[ClipSample] = []
        self._start_t: float = 0.0
        self._recording = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start_thread(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="motion-recorder")
        self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()
        self._recording.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # control
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        with self._lock:
            self._samples = []
            self._start_t = time.monotonic()
        self._recording.set()

    def stop_recording(self) -> None:
        self._recording.clear()

    def is_recording(self) -> bool:
        return self._recording.is_set()

    def status(self) -> RecorderStatus:
        with self._lock:
            cnt = len(self._samples)
            dur = self._samples[-1].t_ms if self._samples else 0
        return RecorderStatus(
            state="recording" if self._recording.is_set() else "idle",
            sample_count=cnt,
            duration_ms=dur,
        )

    def snapshot(self) -> List[ClipSample]:
        """返回当前已采集样本的拷贝（无论是否还在 recording）。"""
        with self._lock:
            return list(self._samples)

    def discard(self) -> None:
        """丢弃当前会话样本，回到完全空白态。"""
        with self._lock:
            self._samples = []
            self._start_t = 0.0

    # ------------------------------------------------------------------
    # persistence helper
    # ------------------------------------------------------------------

    def save_clip(
        self,
        store: ClipStore,
        name: str,
        crop_start_ms: int = 0,
        crop_end_ms: Optional[int] = None,
    ) -> Clip:
        """把当前会话样本（可选裁剪）写入 ClipStore。

        裁剪窗口语义：保留 ``crop_start_ms <= t_ms <= crop_end_ms`` 的样本；
        保留后所有样本时间戳整体平移到从 0 开始。
        """
        with self._lock:
            src = list(self._samples)
        if not src:
            raise RuntimeError("no samples to save")
        end = crop_end_ms if crop_end_ms is not None else src[-1].t_ms
        if end < crop_start_ms:
            raise RuntimeError("crop_end < crop_start")
        kept = [s for s in src if crop_start_ms <= s.t_ms <= end]
        if not kept:
            raise RuntimeError("crop window has no samples")
        base = kept[0].t_ms
        normalized = [
            ClipSample(t_ms=s.t_ms - base, vx=s.vx, vy=s.vy, vw=s.vw)
            for s in kept
        ]
        clip = ClipStore.new_clip(name=name, samples=normalized)
        store.save(clip)
        return clip

    # ------------------------------------------------------------------
    # internal sample loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if self._recording.is_set():
                ctrl = self._serial.latest_ctrl_state
                with self._lock:
                    t_ms = int((now - self._start_t) * 1000.0)
                    # tg_vx/tg_vy/tg_vw 已经是 mm/s 与 mrad/s（X-Protocol 单位）。
                    self._samples.append(ClipSample(
                        t_ms=t_ms,
                        vx=int(ctrl.tg_vx),
                        vy=int(ctrl.tg_vy),
                        vw=int(ctrl.tg_vw),
                    ))
            next_t += self._period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # 严重落后，重新对齐基准时钟，避免持续追赶。
                next_t = time.monotonic()
