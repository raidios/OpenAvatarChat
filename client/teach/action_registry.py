"""
情绪动作的内存绑定视图。

把"情绪 key（happy/shy/apologize/scared）"映射到一个具体的 ``(clip_id, expression_id)``
对。``clip_id=None`` 是合法值：表示该情绪只切表情不放动作（首次启动数据为空
时的兜底）。

数据源：``data/teach_bindings.json``。前端/CLI 通过 ``update`` 修改并 ``save``。
默认绑定（当文件不存在或为空时）：

* happy     → (clip=None, expression="happy")     # 用现有 ``开心.svg``
* shy       → (clip=None, expression="listen")    # 半睁眼.svg 占位
* apologize → (clip=None, expression="frown")     # 皱眉.svg 占位
* scared    → (clip=None, expression="dead")      # 死.svg 占位

注意 expression id 是 ``expression_bridge.load_expression_configs`` 注册到 player
的 id，不是 SVG 文件名。
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional


EMOTION_KEYS = ("happy", "shy", "apologize", "scared")


@dataclass
class ActionBinding:
    clip_id: Optional[str]
    expression_id: str


DEFAULT_BINDINGS: Dict[str, ActionBinding] = {
    "happy": ActionBinding(clip_id=None, expression_id="happy"),
    "shy": ActionBinding(clip_id=None, expression_id="listen"),
    "apologize": ActionBinding(clip_id=None, expression_id="frown"),
    "scared": ActionBinding(clip_id=None, expression_id="dead"),
}


class ActionRegistry:
    def __init__(self, file_path: Path | str):
        self._path = Path(file_path).expanduser().resolve()
        self._lock = threading.Lock()
        self._bindings: Dict[str, ActionBinding] = {k: ActionBinding(**asdict(v)) for k, v in DEFAULT_BINDINGS.items()}

    @property
    def file_path(self) -> Path:
        return self._path

    def load(self) -> None:
        with self._lock:
            if not self._path.is_file():
                return
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"[action_registry] failed to load {self._path}: {e}", flush=True)
                return
            for key, raw in (data or {}).items():
                if key not in EMOTION_KEYS:
                    continue
                if not isinstance(raw, dict):
                    continue
                expr = raw.get("expression_id") or self._bindings[key].expression_id
                clip = raw.get("clip_id")
                self._bindings[key] = ActionBinding(
                    clip_id=clip if isinstance(clip, str) and clip else None,
                    expression_id=str(expr),
                )

    def save(self) -> None:
        with self._lock:
            data = {k: asdict(v) for k, v in self._bindings.items()}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)

    def snapshot(self) -> Dict[str, ActionBinding]:
        with self._lock:
            return {k: ActionBinding(**asdict(v)) for k, v in self._bindings.items()}

    def get(self, key: str) -> Optional[ActionBinding]:
        with self._lock:
            b = self._bindings.get(key)
            return None if b is None else ActionBinding(**asdict(b))

    def update(self, key: str, *, clip_id: Optional[str] = None, expression_id: Optional[str] = None) -> Optional[ActionBinding]:
        if key not in EMOTION_KEYS:
            return None
        with self._lock:
            cur = self._bindings[key]
            self._bindings[key] = ActionBinding(
                clip_id=clip_id if clip_id is not None else cur.clip_id,
                expression_id=expression_id if expression_id is not None else cur.expression_id,
            )
            return ActionBinding(**asdict(self._bindings[key]))
