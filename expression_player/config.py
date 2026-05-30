"""表情配置与对外可见的状态快照。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ExpressionConfig:
    """一套表情：若干静态帧（SVG 路径）+ 循环时间线。"""

    expression_id: str
    frames: Mapping[str, Path]
    timeline_ms: Sequence[tuple[str, int]]

    def __post_init__(self) -> None:
        if not self.expression_id:
            raise ValueError("expression_id 不能为空")
        if not self.frames:
            raise ValueError("frames 不能为空")
        if not self.timeline_ms:
            raise ValueError("timeline_ms 不能为空")
        for key, ms in self.timeline_ms:
            if key not in self.frames:
                raise KeyError(f"时间线引用未知帧键 {key!r}，可用: {sorted(self.frames)}")
            if ms < 0:
                raise ValueError(f"延时不能为负: {key}={ms}")


@dataclass(frozen=True)
class PlayerState:
    """供主程序查询的当前状态（应在 Qt 主线程读取）。"""

    expression_id: str | None
    current_frame_key: str | None
    timeline_index: int
    is_animating: bool
    is_window_fullscreen: bool
