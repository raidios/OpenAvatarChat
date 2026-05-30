"""
ActionDispatcher：把 server 推送的情绪 tag 派发到 MotionPlayer + Expression override。

调用入口（由 ChatClient 的 marker 回调触发）：

    dispatcher.dispatch(tags=[{"name": "happy"}, ...], phase="before")

行为：

* 只识别 4 个情绪 key（``EMOTION_KEYS``）；未知 key 写一行警告但不报错。
* 同种情绪在相邻 marker 中连续出现自动去重（避免一次性触发两次相同 clip）。
* 多个不同情绪 tag 进入：按出现顺序串行入队到 ``MotionPlayer``——播放器自身
  保证只播一个；``set_motion_pending`` 在队列非空期间持续为 True，从而把
  "动作未结束 == TTS 未播完" 的语义跨多个 tag 维持住。
* 表情 override：每次 dispatch 立即触发，hold 期 = 当前 clip 的 duration_ms
  + 0.5s 缓冲；clip 为 None 时按 ``empty_clip_ms``（默认 500ms）。多 tag 时
  最后一个生效。
"""

from __future__ import annotations

import threading
from typing import Iterable, List, Optional, TYPE_CHECKING

from teach.action_registry import EMOTION_KEYS, ActionRegistry
from teach.motion_player import MotionPlayer

if TYPE_CHECKING:
    from teach.clip_store import ClipStore
    from expression_bridge import ExpressionUiState


_EXPR_HOLD_PADDING_MS = 500


class ActionDispatcher:
    def __init__(
        self,
        registry: ActionRegistry,
        store: "ClipStore",
        motion_player: MotionPlayer,
        expression_state: Optional["ExpressionUiState"] = None,
        empty_clip_ms: int = 500,
    ):
        self._registry = registry
        self._store = store
        self._motion = motion_player
        self._expr = expression_state
        self._empty_clip_ms = int(empty_clip_ms)
        self._lock = threading.Lock()
        self._last_emotion: Optional[str] = None

    def dispatch(self, tags: Iterable[dict], phase: str = "before") -> None:
        names = self._normalize_tags(tags)
        if not names:
            return
        for name in names:
            self._dispatch_one(name)

    def cancel_all(self) -> None:
        """用户 interrupt 时调用：清掉所有 pending motion 与 expression override。"""
        self._motion.stop_all()
        if self._expr is not None:
            self._expr.clear_override()
        with self._lock:
            self._last_emotion = None

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _normalize_tags(self, tags: Iterable[dict]) -> List[str]:
        out: List[str] = []
        for t in tags or []:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip().lower()
            if name not in EMOTION_KEYS:
                print(f"[dispatcher] ignoring unknown tag: {t}", flush=True)
                continue
            out.append(name)
        return out

    def _dispatch_one(self, name: str) -> None:
        with self._lock:
            if name == self._last_emotion:
                # 相邻去重：相同情绪连发只触发一次。
                return
            self._last_emotion = name

        binding = self._registry.get(name)
        if binding is None:
            print(f"[dispatcher] no binding for {name!r}", flush=True)
            return

        # 拉 clip（可能为 None）
        clip = None
        if binding.clip_id:
            clip = self._store.get(binding.clip_id)
            if clip is None:
                print(f"[dispatcher] {name!r} bound to missing clip {binding.clip_id}", flush=True)

        duration_ms = clip.duration_ms if (clip and clip.duration_ms) else self._empty_clip_ms
        hold_ms = duration_ms + _EXPR_HOLD_PADDING_MS

        # 1. 表情 override（立即可见）
        if self._expr is not None and binding.expression_id:
            self._expr.set_override(binding.expression_id, hold_ms=hold_ms)

        # 2. motion 入队（即使 clip 为 None 也要排一个占位 job，维持 motion_pending）
        def _on_done(emotion=name):
            # 队列清空后 MotionPlayer 自己会清 motion_pending；这里只清 last_emotion
            # 让下一次相同情绪能再触发。
            with self._lock:
                if self._last_emotion == emotion:
                    # 不强制重置——下一个不同情绪自然覆盖；同一会话连发相同情绪去重生效。
                    pass

        self._motion.enqueue(clip, on_done=_on_done)
