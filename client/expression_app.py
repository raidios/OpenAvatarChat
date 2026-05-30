"""
Run robot client with fullscreen expression window (Qt main thread + asyncio worker).
"""

from __future__ import annotations

import asyncio
import ctypes
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

from expression_player.window import ExpressionPlayerWindow


# PySide6 6.11.0 在很多 void 返回的 Qt 方法（QWidget.update / updateGeometry,
# QPropertyAnimation.stop, QGraphicsOpacityEffect.setOpacity, QTimer.start, …）
# 里每次调用都会多一次 ``Py_DECREF`` 到不朽单例（``None`` / ``True`` /
# ``False``）。表情播放器在每帧动画里要触发数次此类调用，约 17–20 分钟
# ``None`` refcount 被减穿，进程因 ``Fatal Python error: none_dealloc``
# 收到 SIGABRT 被 systemd 重启（也可能换成 ``bool_dealloc``）。
#
# 目前 PySide6 6.11.x ARM aarch64 wheel 没有修复，下面用 ``Py_IncRef`` 周期
# 性补偿；调用极轻量，对性能无影响。等到上游修复后可整体移除。
def _install_pyside_none_refcount_keeper(parent) -> QTimer:
    py_inc_ref = ctypes.pythonapi.Py_IncRef
    immortals = [ctypes.py_object(x) for x in (None, True, False)]

    def _bump() -> None:
        for obj in immortals:
            for _ in range(5000):
                py_inc_ref(obj)

    timer = QTimer(parent)
    timer.setInterval(5_000)
    timer.timeout.connect(_bump)
    timer.start()
    _bump()
    return timer


class _ExpressionMainWindow(ExpressionPlayerWindow):
    """Closing the window stops the asyncio worker."""

    def __init__(self, async_stop: asyncio.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self._async_stop = async_stop

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._async_stop.set()
        super().closeEvent(event)

from expression_bridge import (
    ExpressionUiState,
    TagVisibilityHold,
    expression_id_for_mode,
    load_expression_configs,
    resolve_expression_mode,
    resolve_expression_svg_dir,
)


def run_expression_ui(args, run_async_main) -> None:
    """
    Qt event loop on main thread; ``run_async_main`` is called from a worker thread
    with signature::

        async def run_async_main(stop: asyncio.Event, ui_state: ExpressionUiState, refs: dict) -> None
    """
    app = QApplication(sys.argv)
    _none_refcount_keeper = _install_pyside_none_refcount_keeper(app)
    ui_state = ExpressionUiState()
    async_stop = asyncio.Event()
    refs: dict[str, Any] = {}

    svg_dir = resolve_expression_svg_dir(args)
    configs = load_expression_configs(svg_dir)

    win = _ExpressionMainWindow(
        async_stop,
        defer_fullscreen_ms=int(args.expression_defer_fullscreen_ms),
        default_size=(1024, 600),
        close_on_escape=True,
        close_on_mouse_press=False,
    )
    player = win.player
    for cfg in configs:
        player.register_expression(cfg)
    player.set_expression("blink")
    player.start_animation()

    last_applied: list[Optional[str]] = [None]
    hold_ms = int(getattr(args, "tag_expression_holdoff_ms", 400))
    tag_hold = TagVisibilityHold(hold_ms / 1000.0)

    def sync_expression() -> None:
        # 优先级：情绪 override > 默认决策。这样 ActionDispatcher 触发的 happy/shy
        # 等情绪能压过 listen/speak/tag 等基础模式；override 过期/被清后自动回落。
        override_id = ui_state.override_snapshot()
        if override_id is not None:
            eid = override_id
        else:
            tag_tracker = refs.get("tag_tracker")
            chat_client = refs.get("chat_client")
            raw_tag = bool(tag_tracker and tag_tracker.has_detection)
            tag_visible = tag_hold.update(raw_tag, time.monotonic())
            tts = bool(chat_client and chat_client.player.is_playing)
            conn, wake, vad = ui_state.snapshot()
            mode = resolve_expression_mode(tag_visible, tts, conn, wake, vad)
            eid = expression_id_for_mode(mode)
        if eid == last_applied[0]:
            return
        last_applied[0] = eid
        try:
            if eid == "blink":
                player.stop_animation()
                player.set_expression("blink", restart_timeline=True)
                player.start_animation()
            else:
                player.stop_animation()
                player.set_expression(eid, restart_timeline=True)
        except KeyError:
            last_applied[0] = None

    timer = QTimer(win)
    timer.timeout.connect(sync_expression)
    timer.start(50)

    def asyncio_thread_main() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def runner() -> None:
            try:
                await run_async_main(async_stop, ui_state, refs)
            finally:
                QTimer.singleShot(0, app.quit)

        try:
            loop.run_until_complete(runner())
        finally:
            loop.close()

    worker = threading.Thread(target=asyncio_thread_main, daemon=True)
    worker.start()

    win.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    win.show()
    win.activateWindow()
    win.raise_()
    win.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    if not args.expression_window:
        win.request_fullscreen_after_show()
    else:
        screen = app.primaryScreen()
        if screen is not None:
            fg = win.frameGeometry()
            fg.moveCenter(screen.availableGeometry().center())
            win.move(fg.topLeft())

    app.exec()

    async_stop.set()
    worker.join(timeout=15.0)
