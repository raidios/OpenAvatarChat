"""可选顶层窗口：全屏、延迟全屏（便于 systemd / 桌面会话就绪）。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow

from .player import ExpressionPlayer


class ExpressionPlayerWindow(QMainWindow):
    """
    承载 ExpressionPlayer 的 QMainWindow。

    - defer_fullscreen_ms: 显示后延迟再 showFullScreen，避免会话未完全就绪时全屏失败（常见 100–300ms）。
    - close_on_escape: 自助机可设为 False，由主程序自行处理退出。
    """

    def __init__(
        self,
        player: ExpressionPlayer | None = None,
        *,
        close_on_escape: bool = True,
        close_on_mouse_press: bool = True,
        defer_fullscreen_ms: int = 0,
        default_size=(1024, 600),
        frameless: bool = False,
    ) -> None:
        super().__init__()
        self._player = player or ExpressionPlayer()
        self._defer_ms = max(0, int(defer_fullscreen_ms))
        self._pending_fullscreen = False
        self._did_deferred_fullscreen = False

        if frameless:
            self.setWindowFlag(Qt.FramelessWindowHint, True)

        self.setWindowTitle("Expression")
        self.setStyleSheet("background-color: black;")
        w, h = default_size
        self.resize(int(w), int(h))

        self.setCentralWidget(self._player)
        self._player.set_fullscreen_state_resolver(lambda: self.isFullScreen())

        if close_on_escape:
            for seq in ("<Escape>", "Q", "q"):
                QShortcut(QKeySequence(seq), self, activated=self.close)
        self._close_on_mouse = close_on_mouse_press

    @property
    def player(self) -> ExpressionPlayer:
        return self._player

    def request_fullscreen_after_show(self) -> None:
        """在窗口已 show() 后调用；若设置了 defer，会延迟全屏。"""
        self._pending_fullscreen = True
        if self.isVisible():
            self._apply_deferred_fullscreen()

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        if self._pending_fullscreen:
            self._apply_deferred_fullscreen()

    def _apply_deferred_fullscreen(self) -> None:
        if self._did_deferred_fullscreen:
            return
        self._did_deferred_fullscreen = True
        if self._defer_ms > 0:
            QTimer.singleShot(self._defer_ms, self.showFullScreen)
        else:
            self.showFullScreen()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self._close_on_mouse:
            self.close()
        else:
            super().mousePressEvent(event)
