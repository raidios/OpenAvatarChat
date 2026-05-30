"""可嵌入主窗口的 SVG 表情播放器。"""

from __future__ import annotations

# 误用 ``python expression_player/player.py`` 时避免相对导入报错；正常应 ``python -m expression_player``。
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from expression_player.demo import main

    raise SystemExit(main())

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QByteArray, QEasingCurve, QPropertyAnimation, QTimer, Signal
from PySide6.QtWidgets import QGraphicsOpacityEffect, QVBoxLayout, QWidget

from .config import ExpressionConfig, PlayerState
from .svg_view import AspectFitSvgWidget


class ExpressionPlayer(QWidget):
    """
    在主程序中作为普通 QWidget 使用；由主程序决定全屏/布局。

    - register_expression 注册多套表情，set_expression 切换。
    - start_animation / stop_animation 控制是否按时间线轮播。
    - state() 返回当前快照；frame_changed 在每一帧切换时发出（主线程）。
    """

    frame_changed = Signal(str, str)  # expression_id, frame_key

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ExpressionPlayer")
        self.setStyleSheet("background-color: black;")

        self._svg = AspectFitSvgWidget()
        self._opacity_effect = QGraphicsOpacityEffect(self._svg)
        self._svg.setGraphicsEffect(self._opacity_effect)
        self._opacity_effect.setOpacity(1.0)
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(550)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutQuint)
        self._pending_fade = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._svg)

        self._configs: dict[str, ExpressionConfig] = {}
        # SVG 字节缓存：注册时一次性把 SVG 读成 QByteArray，避免每帧 disk I/O，
        # 也避免 PySide6 6.11 中 QSvgRenderer.load(str) 的引用计数 bug（长时间
        # 运行会让 None 的 refcount 减穿，导致 none_dealloc 致命错误）。
        self._svg_cache: dict[Path, QByteArray] = {}
        self._current_id: str | None = None
        self._step: int = 0
        self._display_index: int | None = None
        self._animating = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timer_fired)
        # stop() 后仍可能排队一次 timeout；用代次丢弃过期回调，避免切表情后把画面刷没。
        self._timer_epoch = 0
        self._timer_armed_epoch = -1

        self._fullscreen_resolver: Callable[[], bool] | None = None

    def _bump_timer_epoch(self) -> None:
        self._timer_epoch += 1

    def set_fullscreen_state_resolver(self, fn: Callable[[], bool] | None) -> None:
        """
        可选：返回顶层窗口是否全屏，便于 state().is_window_fullscreen。
        例：player.set_fullscreen_state_resolver(lambda: main_window.isFullScreen())
        """
        self._fullscreen_resolver = fn

    def register_expression(self, config: ExpressionConfig) -> None:
        for path in config.frames.values():
            self._cache_svg(path)
        self._configs[config.expression_id] = config
        if self._current_id is None:
            self.set_expression(config.expression_id, restart_timeline=True)

    def _cache_svg(self, path: Path) -> QByteArray | None:
        cached = self._svg_cache.get(path)
        if cached is not None:
            return cached
        try:
            with open(str(path.expanduser().resolve()), "rb") as f:
                data = QByteArray(f.read())
        except OSError:
            return None
        self._svg_cache[path] = data
        return data

    def unregister_expression(self, expression_id: str) -> None:
        self._configs.pop(expression_id, None)
        if self._current_id == expression_id:
            self._timer.stop()
            self._bump_timer_epoch()
            self._current_id = None
            self._step = 0
            self._display_index = None
            self._animating = False

    def list_expressions(self) -> list[str]:
        return sorted(self._configs.keys())

    def set_expression(self, expression_id: str, *, restart_timeline: bool = True) -> None:
        if expression_id not in self._configs:
            raise KeyError(f"未注册的表情: {expression_id!r}")
        prev_id = self._current_id
        was_animating = self._animating
        self._timer.stop()
        self._bump_timer_epoch()
        self._current_id = expression_id
        self._pending_fade = prev_id is not None and prev_id != expression_id
        cfg = self._configs[expression_id]
        n = len(cfg.timeline_ms)
        if not n:
            self._display_index = None
            self._step = 0
        elif restart_timeline:
            self._present_frame(0)
            self._step = 1 % n
        else:
            di = 0 if self._display_index is None else self._display_index % n
            self._present_frame(di)
            self._step = (di + 1) % n
        if was_animating:
            self._animating = True
            self._arm_timer_after_current_frame()
        else:
            self._animating = False

    def current_expression_id(self) -> str | None:
        return self._current_id

    def start_animation(self) -> None:
        if self._current_id is None:
            return
        self._animating = True
        self._arm_timer_after_current_frame()

    def stop_animation(self) -> None:
        self._animating = False
        self._timer.stop()
        self._bump_timer_epoch()

    def is_animating(self) -> bool:
        return self._animating

    def state(self) -> PlayerState:
        cfg = self._configs[self._current_id] if self._current_id else None
        frame_key: str | None = None
        t_idx = 0
        if cfg and self._display_index is not None:
            t_idx = self._display_index % len(cfg.timeline_ms)
            frame_key = cfg.timeline_ms[t_idx][0]
        elif cfg and cfg.timeline_ms:
            t_idx = 0
            frame_key = cfg.timeline_ms[0][0]

        fullscreen = False
        if self._fullscreen_resolver is not None:
            try:
                fullscreen = bool(self._fullscreen_resolver())
            except Exception:
                fullscreen = False

        return PlayerState(
            expression_id=self._current_id,
            current_frame_key=frame_key,
            timeline_index=t_idx,
            is_animating=self._animating,
            is_window_fullscreen=fullscreen,
        )

    def _current_config(self) -> ExpressionConfig | None:
        if self._current_id is None:
            return None
        return self._configs.get(self._current_id)

    def _present_frame(self, timeline_index: int) -> None:
        cfg = self._current_config()
        if cfg is None or not cfg.timeline_ms:
            return
        n = len(cfg.timeline_ms)
        idx = timeline_index % n
        key, _delay = cfg.timeline_ms[idx]
        path = cfg.frames[key]
        self._load_svg(path)
        self._display_index = idx
        eid = self._current_id or ""
        self.frame_changed.emit(eid, key)

    def _arm_timer_after_current_frame(self) -> None:
        cfg = self._current_config()
        if cfg is None or not self._animating or not cfg.timeline_ms:
            return
        if self._display_index is None:
            self._present_frame(0)
            self._step = 1 % len(cfg.timeline_ms)
        if self._timer.isActive():
            return
        idx = self._display_index if self._display_index is not None else 0
        _key, delay_ms = cfg.timeline_ms[idx % len(cfg.timeline_ms)]
        self._timer_armed_epoch = self._timer_epoch
        self._timer.start(delay_ms)

    def _on_timer_fired(self) -> None:
        if self._timer_armed_epoch != self._timer_epoch:
            return
        cfg = self._current_config()
        if cfg is None or not self._animating or not cfg.timeline_ms:
            return
        n = len(cfg.timeline_ms)
        idx = self._step % n
        key, delay_ms = cfg.timeline_ms[idx]
        self._load_svg(cfg.frames[key])
        self._display_index = idx
        self.frame_changed.emit(self._current_id or "", key)
        self._step = (idx + 1) % n
        self._timer_armed_epoch = self._timer_epoch
        self._timer.start(delay_ms)

    def _load_svg(self, path: Path) -> None:
        data = self._svg_cache.get(path) or self._cache_svg(path)
        if data is None:
            return
        self._svg.load(data)
        if self._pending_fade:
            self._pending_fade = False
            self._fade_anim.stop()
            self._opacity_effect.setOpacity(0.0)
            self._fade_anim.setStartValue(0.0)
            self._fade_anim.setEndValue(1.0)
            self._fade_anim.start()
        else:
            self._fade_anim.stop()
            self._opacity_effect.setOpacity(1.0)
