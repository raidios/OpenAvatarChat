"""按比例显示 SVG，避免宽屏下非等比拉伸。"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PySide6.QtCore import QByteArray, QRectF, QSize
from PySide6.QtGui import QColor, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QSizePolicy, QWidget

SvgSource = Union[str, Path, bytes, bytearray, QByteArray]


class AspectFitSvgWidget(QWidget):
    """
    将 SVG 按原始纵横比缩放后居中绘制（letterbox），圆形不会被拉成椭圆。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._renderer = QSvgRenderer()
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def load(self, source: SvgSource) -> bool:
        """加载 SVG。

        统一通过 ``QByteArray`` 重载调用 ``QSvgRenderer.load``：PySide6 6.11 在
        ``load(str)`` 路径上存在一处 ``Py_DECREF(None)`` 引用计数错误，长时间
        运行（约 1500–2000 次加载）会触发 ``none_dealloc`` 致命错误并
        ``SIGABRT``。改走字节缓冲区可以稳定绕开该 bug。
        """
        if isinstance(source, QByteArray):
            data = source
        elif isinstance(source, (bytes, bytearray)):
            data = QByteArray(bytes(source))
        else:
            with open(str(source), "rb") as f:
                data = QByteArray(f.read())
        ok = bool(self._renderer.load(data))
        self.updateGeometry()
        self.update()
        return ok

    def _intrinsic_size(self) -> QSize:
        s = self._renderer.defaultSize()
        if s.width() > 0 and s.height() > 0:
            return s
        vb = self._renderer.viewBox()
        if vb.width() > 0 and vb.height() > 0:
            return QSize(int(vb.width()), int(vb.height()))
        return QSize(400, 300)

    def sizeHint(self) -> QSize:
        return self._intrinsic_size()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if not self._renderer.isValid():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        view = self._intrinsic_size()
        if view.width() <= 0 or view.height() <= 0:
            return
        rw, rh = self.width(), self.height()
        if rw <= 0 or rh <= 0:
            return
        scale = min(rw / view.width(), rh / view.height())
        tw = view.width() * scale
        th = view.height() * scale
        x = (rw - tw) * 0.5
        y = (rh - th) * 0.5
        self._renderer.render(painter, QRectF(x, y, tw, th))
