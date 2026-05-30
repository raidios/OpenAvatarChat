"""独立演示入口：python3 -m expression_player.demo [--svg-dir PATH]"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from .config import ExpressionConfig
from .resource_paths import default_svg_directory
from .window import ExpressionPlayerWindow


def _default_svg_dir() -> Path:
    """与客户端一致：优先 ``EXPRESSION_SVG_DIR``，否则 ``expression_player/images`` 等。"""
    return default_svg_directory()


def blink_config(svg_dir: Path) -> ExpressionConfig:
    d = svg_dir.expanduser().resolve()
    frames = {
        "open": d / "睁眼.svg",
        "micro": d / "微闭眼.svg",
        "half": d / "半睁眼.svg",
        "closed": d / "闭眼.svg",
    }
    timeline_ms = [
        ("open", 3000),
        ("micro", 50),
        ("half", 50),
        ("closed", 1000),
        ("half", 50),
        ("micro", 50),
    ]
    return ExpressionConfig(expression_id="blink", frames=frames, timeline_ms=timeline_ms)


def static_expression(expression_id: str, svg_path: Path) -> ExpressionConfig:
    """单张 SVG；演示里用 stop_animation 定格，时长仅由恢复用 QTimer 控制。"""
    return ExpressionConfig(
        expression_id=expression_id,
        frames={"main": svg_path},
        timeline_ms=[("main", 1000)],
    )


# (主键盘键, 表情 id, 文件名) — 1 键眨眼；2–5 持续显示直至按 1 或其它数字
EMOTION_HOTKEYS: list[tuple[Qt.Key, str, str]] = [
    (Qt.Key.Key_2, "happy", "开心.svg"),
    (Qt.Key.Key_3, "joy", "乐.svg"),
    (Qt.Key.Key_4, "dead", "死.svg"),
    (Qt.Key.Key_5, "frown", "皱眉.svg"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="表情全屏演示")
    p.add_argument(
        "--svg-dir",
        type=Path,
        default=_default_svg_dir(),
        help="表情 SVG 目录（须含眨眼四帧及 开心/乐/死/皱眉）；默认 expression_player/images；可设 EXPRESSION_SVG_DIR",
    )
    p.add_argument(
        "--defer-fullscreen-ms",
        type=int,
        default=int(os.environ.get("EXPRESSION_DEFER_FULLSCREEN_MS", "200")),
        help="显示后延迟全屏毫秒数，桌面自动启动时可设 100–300（与 --window 互斥）",
    )
    p.add_argument(
        "--window",
        action="store_true",
        help="不以全屏运行，使用普通可调整窗口（适合 Mac 上调试）",
    )
    args = p.parse_args(argv)

    d = args.svg_dir.expanduser().resolve()
    cfg_blink = blink_config(d)
    for path in cfg_blink.frames.values():
        if not path.is_file():
            print(f"缺少 SVG：{path}", file=sys.stderr)
            return 1
    emotion_configs: list[ExpressionConfig] = []
    for _qt_key, eid, fname in EMOTION_HOTKEYS:
        p = d / fname
        if not p.is_file():
            print(f"缺少 SVG：{p}", file=sys.stderr)
            return 1
        emotion_configs.append(static_expression(eid, p))

    app = QApplication(sys.argv)
    win = ExpressionPlayerWindow(
        defer_fullscreen_ms=args.defer_fullscreen_ms,
        default_size=(1024, 600),
    )
    player = win.player
    player.register_expression(cfg_blink)
    for ec in emotion_configs:
        player.register_expression(ec)

    def show_emotion(expression_id: str) -> None:
        # 切换时由播放器做短暂淡入；表情持续静止直至再按键
        player.stop_animation()
        player.set_expression(expression_id)

    def switch_to_blink() -> None:
        """按 1：恢复眨眼循环。"""
        player.stop_animation()
        player.set_expression("blink")
        player.start_animation()

    sc1 = QShortcut(QKeySequence(Qt.Key.Key_1), win)
    sc1.setContext(Qt.ShortcutContext.ApplicationShortcut)
    sc1.activated.connect(switch_to_blink)

    for qt_key, eid, _fname in EMOTION_HOTKEYS:
        seq = QKeySequence(qt_key)
        sc = QShortcut(seq, win)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(lambda e=eid: show_emotion(e))

    player.set_expression("blink")
    player.start_animation()
    win.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    win.show()
    win.activateWindow()
    win.raise_()
    win.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    if args.window:
        screen = app.primaryScreen()
        if screen is not None:
            fg = win.frameGeometry()
            fg.moveCenter(screen.availableGeometry().center())
            win.move(fg.topLeft())
    else:
        win.request_fullscreen_after_show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
