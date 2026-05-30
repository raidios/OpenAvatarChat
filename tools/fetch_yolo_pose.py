#!/usr/bin/env python3
"""Fetch YOLOv8n-Pose .pt and export to ONNX into models/yolo/yolov8n-pose.onnx.

Strategy:
  1. Download yolov8n-pose.pt (~7 MB) from ultralytics github release,
     trying github direct then several gh-proxy mirrors.
  2. Use ultralytics CLI/lib (already pulls torch which we have) to export
     to ONNX at imgsz=640 opset=12. Result: models/yolo/yolov8n-pose.onnx

If you don't want ultralytics installed, just scp a pre-exported .onnx
into the target path and skip this script entirely.
"""
from __future__ import annotations

import argparse
import os
import socket
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "models" / "yolo"
PT_FILE = TARGET_DIR / "yolov8n-pose.pt"
ONNX_FILE = TARGET_DIR / "yolov8n-pose.onnx"

URL_PT = ("https://github.com/ultralytics/assets/releases/download/v8.3.0/"
          "yolov8n-pose.pt")
PROXIES = [
    "https://gh-proxy.com",
    "https://hub.gitmirror.com",
    "https://github.moeyy.xyz",
]


def _download(url: str, out: Path, timeout: int = 30) -> bool:
    socket.setdefaulttimeout(timeout)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.x"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(out, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        sz = out.stat().st_size
        if sz < 1_000_000:
            raise RuntimeError(f"size {sz} too small, looks like an error page")
        return True
    except (urllib.error.URLError, socket.timeout, ssl.SSLError, RuntimeError) as e:
        print(f"    {e}")
        try:
            out.unlink()
        except FileNotFoundError:
            pass
        return False


def fetch_pt(force: bool = False) -> bool:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if PT_FILE.exists() and not force:
        print(f"[*] already have {PT_FILE} ({PT_FILE.stat().st_size/1024/1024:.1f} MB)")
        return True
    candidates = [URL_PT] + [f"{p}/{URL_PT}" for p in PROXIES]
    for url in candidates:
        print(f"[*] try {url}")
        if _download(url, PT_FILE):
            print(f"[+] {PT_FILE} ({PT_FILE.stat().st_size/1024/1024:.1f} MB)")
            return True
    return False


def export_onnx(force: bool = False) -> bool:
    if ONNX_FILE.exists() and not force:
        print(f"[*] already have {ONNX_FILE} ({ONNX_FILE.stat().st_size/1024/1024:.1f} MB)")
        return True
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not installed")
        print("    install:  uv pip install ultralytics --no-deps   "
              "(--no-deps avoids pulling torchvision-cu)")
        print("    minimal:  uv pip install \"ultralytics-thop<1\" pyyaml requests "
              "tqdm pandas seaborn pillow psutil")
        return False
    print("[*] exporting to ONNX (this may take 30-60s)")
    model = YOLO(str(PT_FILE))
    out = model.export(format="onnx", imgsz=640, opset=12, dynamic=False, simplify=False)
    src = Path(out)
    if src.resolve() != ONNX_FILE.resolve():
        src.replace(ONNX_FILE)
    print(f"[+] {ONNX_FILE} ({ONNX_FILE.stat().st_size/1024/1024:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-export", action="store_true",
                    help="just fetch .pt; you'll export manually")
    args = ap.parse_args()
    if not fetch_pt(force=args.force):
        print("[!] could not fetch .pt — fetch manually and place at:")
        print(f"    {PT_FILE}")
        return 1
    if args.skip_export:
        print("[*] --skip-export set; remember to run:")
        print("    yolo export model=models/yolo/yolov8n-pose.pt format=onnx imgsz=640 opset=12")
        return 0
    if not export_onnx(force=args.force):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
