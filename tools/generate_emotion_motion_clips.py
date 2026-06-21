from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CLIPS = {
    "e0010001": {
        "name": "happy_bounce_wiggle_v4",
        "emotion": "happy",
        "expression_id": "happy",
        "segments": [
            (160, 620, 0, 0),
            (120, -460, 0, 0),
            (150, 580, 0, 0),
            (120, -420, 0, 0),
            (140, 520, 0, 0),
            (140, -360, 0, 0),
            (160, 160, 0, 1900),
            (160, 140, 0, -1900),
            (150, 100, 0, 1700),
            (150, 80, 0, -1700),
            (180, -160, 0, 0),
            (200, 0, 0, 0),
        ],
    },
    "e0010002": {
        "name": "shy_hesitate_hide_v7",
        "emotion": "shy",
        "expression_id": "listen",
        "segments": [
            (420, -190, 0, 620),
            (360, -70, 0, 260),
            (520, 0, 0, 0),
            (340, 80, 0, -420),
            (420, 20, 0, -220),
            (360, -50, 0, 180),
            (360, 0, 0, -160),
            (260, 0, 0, 0),
        ],
    },
    "e0010003": {
        "name": "apologize_big_headshake_v8",
        "emotion": "apologize",
        "expression_id": "frown",
        "segments": [
            (220, -140, 0, 0),
            (220, 0, 0, 2300),
            (220, 0, 0, -2300),
            (220, 0, 0, 2100),
            (220, 0, 0, -2100),
            (200, 0, 0, 1700),
            (200, 0, 0, -1700),
            (260, 90, 0, 0),
            (280, 0, 0, 0),
        ],
    },
    "e0010004": {
        "name": "scared_snap_back_v7",
        "emotion": "scared",
        "expression_id": "dead",
        "segments": [
            (220, -850, 0, 0),
            (100, 260, 0, 0),
            (160, 0, 0, 2500),
            (320, 0, 0, -2500),
            (160, 0, 0, 2500),
            (180, 0, 0, -1700),
            (440, 0, 0, 0),
            (260, 90, 0, 0),
            (240, 0, 0, 0),
        ],
    },
}


def _samples(segments: list[tuple[int, int, int, int]], step_ms: int = 20) -> list[dict]:
    out: list[dict] = []
    t_ms = 0
    out.append({"t_ms": t_ms, "vx": 0, "vy": 0, "vw": 0})
    for duration_ms, vx, vy, vw in segments:
        end_ms = t_ms + int(duration_ms)
        t_ms += step_ms
        while t_ms <= end_ms:
            out.append({"t_ms": t_ms, "vx": int(vx), "vy": int(vy), "vw": int(vw)})
            t_ms += step_ms
        t_ms = end_ms
    if out[-1]["vx"] or out[-1]["vy"] or out[-1]["vw"]:
        out.append({"t_ms": t_ms + step_ms, "vx": 0, "vy": 0, "vw": 0})
    return out


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", default="data/motion_clips")
    parser.add_argument("--bindings-file", default="data/teach_bindings.json")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    bindings_path = Path(args.bindings_file)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.backup and bindings_path.exists():
        backup = bindings_path.with_suffix(
            bindings_path.suffix + "." + datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"
        )
        shutil.copy2(bindings_path, backup)
        print(f"backup {backup}")

    bindings = {}
    if bindings_path.exists():
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))

    for clip_id, spec in CLIPS.items():
        samples = _samples(spec["segments"])
        clip = {
            "id": clip_id,
            "name": spec["name"],
            "duration_ms": samples[-1]["t_ms"],
            "created_at": created_at,
            "samples": samples,
        }
        _write_json(clips_dir / f"{clip_id}.json", clip)
        bindings[spec["emotion"]] = {
            "clip_id": clip_id,
            "expression_id": spec["expression_id"],
        }
        print(
            f"{clip_id} {spec['name']} duration_ms={clip['duration_ms']} "
            f"samples={len(samples)}"
        )

    _write_json(bindings_path, bindings)
    print(f"updated {bindings_path}")


if __name__ == "__main__":
    main()
