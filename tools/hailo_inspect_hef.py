#!/usr/bin/env python3
"""
Dump input/output VStream signatures of a HEF.

Usage:
    .venv/bin/python tools/hailo_inspect_hef.py models/hailo/dtln_p1_h10h.hef
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hef", nargs="+", help="path(s) to .hef")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    try:
        from hailo_platform import HEF  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"hailo_platform import failed: {exc}", file=sys.stderr)
        return 2

    out = {}
    for p in args.hef:
        path = Path(p).resolve()
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        hef = HEF(str(path))
        in_infos = hef.get_input_vstream_infos()
        out_infos = hef.get_output_vstream_infos()
        entry = {
            "path": str(path),
            "size_kb": round(path.stat().st_size / 1024.0, 1),
            "inputs": [
                {
                    "name": i.name,
                    "shape": list(i.shape),
                    "format_type": str(i.format.type),
                    "format_order": str(i.format.order),
                }
                for i in in_infos
            ],
            "outputs": [
                {
                    "name": o.name,
                    "shape": list(o.shape),
                    "format_type": str(o.format.type),
                    "format_order": str(o.format.order),
                }
                for o in out_infos
            ],
        }
        out[path.name] = entry

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for name, e in out.items():
            print(f"\n=== {name} ({e['size_kb']} KB) ===")
            print(f"  path: {e['path']}")
            print("  inputs:")
            for i in e["inputs"]:
                print(f"    {i['name']:30s} shape={i['shape']} "
                      f"order={i['format_order']} dtype={i['format_type']}")
            print("  outputs:")
            for i in e["outputs"]:
                print(f"    {i['name']:30s} shape={i['shape']} "
                      f"order={i['format_order']} dtype={i['format_type']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
