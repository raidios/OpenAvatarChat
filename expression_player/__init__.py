"""
可嵌入主程序的 SVG 表情播放器（PySide6）。

默认静态资源目录：``expression_player/images``（眨眼四帧 + 开心/乐/死/皱眉）。

集成示例（需在已存在 QApplication 后、于 Qt 主线程调用）::

    from expression_player import ExpressionConfig, ExpressionPlayer, ExpressionPlayerWindow
    from expression_player import default_svg_directory

    svg_dir = default_svg_directory()
    ...
"""

# 误用 ``python expression_player/__init__.py`` 时避免相对导入报错；正常应使用 ``python -m expression_player``。
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from expression_player.demo import main

    raise SystemExit(main())

from .config import ExpressionConfig, PlayerState

__all__ = [
    "ExpressionConfig",
    "ExpressionPlayer",
    "ExpressionPlayerWindow",
    "PlayerState",
    "bundled_images_dir",
    "default_svg_directory",
]


def __getattr__(name: str):
    if name == "ExpressionPlayer":
        from .player import ExpressionPlayer

        return ExpressionPlayer
    if name == "ExpressionPlayerWindow":
        from .window import ExpressionPlayerWindow

        return ExpressionPlayerWindow
    if name == "bundled_images_dir":
        from .resource_paths import bundled_images_dir

        return bundled_images_dir
    if name == "default_svg_directory":
        from .resource_paths import default_svg_directory

        return default_svg_directory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
