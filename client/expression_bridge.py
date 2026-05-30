"""
Thread-safe UI hints from WebSocket + helpers to map logical modes → expression ids.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from expression_player.config import ExpressionConfig
from expression_player.resource_paths import default_svg_directory


def _default_svg_dir() -> Path:
    """兼容旧名；等价于 ``default_svg_directory()``。"""
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
    return ExpressionConfig(
        expression_id=expression_id,
        frames={"main": svg_path},
        timeline_ms=[("main", 1000)],
    )


def resolve_expression_svg_dir(args) -> Path:
    """CLI / env: ``args.expression_svg_dir`` 或 ``EXPRESSION_SVG_DIR`` 或仓库 ``expression_player/images``。"""
    raw = getattr(args, "expression_svg_dir", None)
    if raw:
        return Path(raw).expanduser().resolve()
    env = os.environ.get("EXPRESSION_SVG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _default_svg_dir().resolve()


def expression_assets_available(svg_dir: Path) -> tuple[bool, Optional[str]]:
    try:
        load_expression_configs(svg_dir)
        return True, None
    except FileNotFoundError as e:
        return False, str(e)


def load_expression_configs(svg_dir: Path) -> List[ExpressionConfig]:
    """从目录加载 SVG 帧：眨眼循环 + 静态情绪；可选 ``听.svg`` / ``看.svg`` 用于听/tag 态。"""
    d = svg_dir.expanduser().resolve()
    cfgs: List[ExpressionConfig] = []
    cfg_blink = blink_config(d)
    for path in cfg_blink.frames.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少眨眼素材: {path}")
    cfgs.append(cfg_blink)

    emotion_files = [
        ("happy", "开心.svg"),
        ("joy", "乐.svg"),
        ("dead", "死.svg"),
        ("frown", "皱眉.svg"),
    ]
    for eid, fname in emotion_files:
        p = d / fname
        if not p.is_file():
            raise FileNotFoundError(f"缺少表情素材: {p}")
        cfgs.append(static_expression(eid, p))

    listen_p = d / "听.svg"
    if not listen_p.is_file():
        listen_p = d / "半睁眼.svg"
    if not listen_p.is_file():
        listen_p = d / "睁眼.svg"
    cfgs.append(static_expression("listen", listen_p))

    tag_p = d / "看.svg"
    if not tag_p.is_file():
        tag_p = d / "乐.svg"
    cfgs.append(static_expression("tag", tag_p))

    return cfgs


class ExpressionUiState:
    """Updated from asyncio ws thread; read from Qt main thread.

    新增的 ``override_*`` 通道用于情绪动作触发：``set_override(eid, hold_ms)``
    在 ``hold_ms`` 时间窗内强制把表情切到指定 eid，超时自动回落到既有的
    ``resolve_expression_mode`` 决策；``hold_ms = 0`` 视作无限期（直到下次
    手动 clear）。Qt 端的 ``sync_expression`` 在每次 tick 调 ``override_snapshot``
    决定优先级。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chat_connected: bool = False
        self.wake_session: bool = False
        self.vad_enabled: bool = False
        self._override_id: Optional[str] = None
        self._override_until: float = 0.0  # 0 表示已清空；正数为 time.monotonic 截止戳

    def on_ws_json(self, data: dict) -> None:
        t = data.get("type")
        if t == "session_started":
            with self._lock:
                self.chat_connected = True
        elif t == "ui_state":
            with self._lock:
                self.wake_session = bool(data.get("wake_session"))
                self.vad_enabled = bool(data.get("vad_enabled"))

    def reset_session(self) -> None:
        with self._lock:
            self.chat_connected = False
            self.wake_session = False
            self.vad_enabled = False
            # 会话结束时把 override 也清掉，避免下次会话起来仍卡在上一个情绪。
            self._override_id = None
            self._override_until = 0.0

    def snapshot(self) -> tuple[bool, bool, bool]:
        with self._lock:
            return self.chat_connected, self.wake_session, self.vad_enabled

    # ------------------------------------------------------------------
    # override channel
    # ------------------------------------------------------------------

    def set_override(self, expression_id: str, hold_ms: int) -> None:
        """从任意线程调用：在 ``hold_ms`` 内强制显示 ``expression_id``。

        ``hold_ms`` <=0 表示无截止时间（持续到下次手动清除或下次 override 覆盖）。
        多次调用以最新一次为准。
        """
        import time as _time
        until = 0.0 if hold_ms <= 0 else _time.monotonic() + hold_ms / 1000.0
        with self._lock:
            self._override_id = expression_id
            self._override_until = until

    def clear_override(self) -> None:
        with self._lock:
            self._override_id = None
            self._override_until = 0.0

    def override_snapshot(self) -> Optional[str]:
        """返回当前应用的 override expression id；过期/未设置时返回 None 并自动清理。"""
        import time as _time
        with self._lock:
            if self._override_id is None:
                return None
            if self._override_until and _time.monotonic() >= self._override_until:
                self._override_id = None
                self._override_until = 0.0
                return None
            return self._override_id


class TagVisibilityHold:
    """
    Tag 检测瞬时丢帧时仍保持「tag」表情一段时间，避免画面闪烁。
    检测到 tag 立即切到 tag；丢失后 holdoff_sec 内仍视为可见，超时再放开。
    """

    def __init__(self, holdoff_sec: float) -> None:
        self.holdoff_sec = float(holdoff_sec)
        self._effective = False
        self._lost_since: Optional[float] = None

    def update(self, raw_visible: bool, now: float) -> bool:
        if self.holdoff_sec <= 0:
            return raw_visible
        if raw_visible:
            self._lost_since = None
            self._effective = True
            return True
        if not self._effective:
            return False
        if self._lost_since is None:
            self._lost_since = now
        if now - self._lost_since >= self.holdoff_sec:
            self._effective = False
            self._lost_since = None
            return False
        return True


def resolve_expression_mode(
    tag_visible: bool,
    tts_playing: bool,
    connected: bool,
    wake_session: bool,
    vad_enabled: bool,
) -> str:
    """Return logical mode: default | listen | speak | tag."""
    if tag_visible:
        return "tag"
    if tts_playing:
        return "speak"
    if connected and wake_session and vad_enabled and not tts_playing:
        return "listen"
    return "default"


def expression_id_for_mode(mode: str) -> str:
    return {
        "default": "blink",
        "listen": "listen",
        "speak": "happy",
        "tag": "tag",
    }.get(mode, "blink")
