"""表情静态资源路径（默认使用仓库内 ``expression_player/images/*.svg``）。"""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def bundled_images_dir() -> Path:
    """随仓库分发的 SVG 目录：``expression_player/images``。"""
    return _PKG / "images"


def default_svg_directory() -> Path:
    """
    解析默认素材目录，优先级：

    1. 环境变量 ``EXPRESSION_SVG_DIR``
    2. 若 ``expression_player/images/睁眼.svg`` 存在，使用该目录（正常 clone 即可用）
    3. 常见本机路径（桌面「表情」等），便于覆盖仓库默认
    4. 仍回退到 ``bundled_images_dir()``（路径稳定，便于日志）
    """
    env = os.environ.get("EXPRESSION_SVG_DIR")
    if env:
        return Path(env).expanduser()

    bundled = bundled_images_dir()
    if (bundled / "睁眼.svg").is_file():
        return bundled

    home = Path.home()
    for c in (
        Path("/Users/luyun/Desktop/表情"),
        home / "Desktop" / "表情",
        home / "表情",
        home / "Documents" / "表情",
    ):
        if (c / "睁眼.svg").is_file():
            return c

    return bundled
