"""
动作 clip 的 JSON 持久化。

文件布局：

* 默认目录 ``data/motion_clips/``，每个 clip 一个 ``<id>.json``。
* clip id 是 8 位 hex 随机串（避免和文件系统大小写敏感问题）。
* 内容示例：

  .. code-block:: json

      {
        "id": "a1b2c3d4",
        "name": "happy_spin",
        "duration_ms": 1840,
        "created_at": "2026-05-14T18:34:55Z",
        "samples": [
          {"t_ms": 0, "vx": 0, "vy": 0, "vw": 0},
          {"t_ms": 20, "vx": 200, "vy": 0, "vw": 1200},
          ...
        ]
      }

``samples`` 中 ``vx/vy/vw`` 与下位机 X-Protocol 一致——m/s × 1000 与 rad/s × 1000
（也就是 mm/s 与 mrad/s）。播放时直接喂 ``XProtocolSerial.send_velocity``。

模块对外只暴露 ``ClipStore``；不在 import 时碰文件，需要 ``load()`` 触发扫描。
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ClipSample:
    t_ms: int
    vx: int  # mm/s
    vy: int  # mm/s（差速车通常为 0，但格式保留以支持全向）
    vw: int  # mrad/s


@dataclass
class Clip:
    id: str
    name: str
    duration_ms: int
    created_at: str
    samples: List[ClipSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Clip":
        samples_raw = d.get("samples") or []
        samples = [
            ClipSample(
                t_ms=int(s.get("t_ms", 0)),
                vx=int(s.get("vx", 0)),
                vy=int(s.get("vy", 0)),
                vw=int(s.get("vw", 0)),
            )
            for s in samples_raw
        ]
        return Clip(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            duration_ms=int(d.get("duration_ms", 0)),
            created_at=str(d.get("created_at", "")),
            samples=samples,
        )


class ClipStore:
    """线程安全的内存索引 + JSON 落盘。

    所有写操作（save/delete/rename）会同步重写对应单个文件并刷新内存索引；
    读操作走内存，不命中再触发 load。
    """

    def __init__(self, base_dir: Path | str):
        self._base = Path(base_dir).expanduser().resolve()
        self._lock = threading.Lock()
        self._clips: Dict[str, Clip] = {}
        self._loaded = False

    @property
    def base_dir(self) -> Path:
        return self._base

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            self._base.mkdir(parents=True, exist_ok=True)
            self._clips = {}
            for p in self._base.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    clip = Clip.from_dict(data)
                    self._clips[clip.id] = clip
                except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"[clip_store] skip {p.name}: {e}", flush=True)
            self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def list(self) -> List[Clip]:
        self._ensure_loaded()
        with self._lock:
            return sorted(self._clips.values(), key=lambda c: c.created_at)

    def get(self, clip_id: str) -> Optional[Clip]:
        self._ensure_loaded()
        with self._lock:
            return self._clips.get(clip_id)

    def save(self, clip: Clip) -> Clip:
        self._ensure_loaded()
        with self._lock:
            self._clips[clip.id] = clip
            path = self._base / f"{clip.id}.json"
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(clip.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        return clip

    def delete(self, clip_id: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            if clip_id not in self._clips:
                return False
            self._clips.pop(clip_id, None)
            path = self._base / f"{clip_id}.json"
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return True

    def rename(self, clip_id: str, new_name: str) -> Optional[Clip]:
        self._ensure_loaded()
        with self._lock:
            clip = self._clips.get(clip_id)
            if clip is None:
                return None
            clip.name = new_name
            path = self._base / f"{clip.id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(clip.to_dict(), f, ensure_ascii=False, indent=2)
            return clip

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def new_clip(name: str, samples: List[ClipSample]) -> Clip:
        clip_id = secrets.token_hex(4)
        duration = samples[-1].t_ms if samples else 0
        return Clip(
            id=clip_id,
            name=name,
            duration_ms=duration,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            samples=samples,
        )


# 简易自检：临时目录跑一遍 save/list/delete。
def _self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = ClipStore(td)
        store.load()
        assert store.list() == []
        clip = ClipStore.new_clip("test", [
            ClipSample(t_ms=0, vx=0, vy=0, vw=0),
            ClipSample(t_ms=20, vx=200, vy=0, vw=0),
            ClipSample(t_ms=40, vx=0, vy=0, vw=0),
        ])
        store.save(clip)
        again = ClipStore(td)
        again.load()
        rt = again.get(clip.id)
        assert rt is not None and rt.name == "test"
        assert len(rt.samples) == 3 and rt.samples[1].vx == 200
        assert again.delete(clip.id)
        assert again.get(clip.id) is None
        print("clip_store self-test ok")


if __name__ == "__main__":
    _self_test()
