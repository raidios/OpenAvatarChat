#!/usr/bin/env python3
"""
Manage WHEELTEC R818 board's audio service mode + adb forward.

Two modes are mutually exclusive on `hw:1,0`:
  - raw  : audio_server (8ch raw via TCP 9999)  <-- our pipeline uses this
  - demo : iFLYOS demo (1ch processed via UAC1) <-- vendor original

Subcommands:
  start-raw           switch board to raw mode + idempotent adb forward 9999
  start-demo          switch board to vendor demo mode (kills audio_server)
  status              print current mode, adb device, server pid, forward state
  ensure-forward      just (re)apply `adb forward tcp:9999 tcp:9999`

Exit codes:
  0  ok
  1  adb has no device
  2  ssh/board command failed
  3  audio_server not running after start
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Optional

ADB = shutil.which("adb") or "adb"

BOARD_USE_RAW = "/data/build/use_audio_server.sh"
BOARD_USE_DEMO = "/data/build/use_demo.sh"
DEFAULT_PORT = 9999


def _adb(*args: str, timeout: float = 5.0,
         check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ADB, *args], capture_output=True, text=True,
        timeout=timeout, check=check,
    )


def adb_has_device() -> bool:
    cp = _adb("devices")
    if cp.returncode != 0:
        return False
    lines = [l.strip() for l in cp.stdout.splitlines() if l.strip()]
    return any(l.endswith("\tdevice") for l in lines)


def adb_forward_state(port: int = DEFAULT_PORT) -> bool:
    cp = _adb("forward", "--list")
    if cp.returncode != 0:
        return False
    needle = f"tcp:{port} tcp:{port}"
    return any(needle in line for line in cp.stdout.splitlines())


def adb_forward_apply(port: int = DEFAULT_PORT) -> bool:
    cp = _adb("forward", f"tcp:{port}", f"tcp:{port}")
    return cp.returncode == 0


def board_audio_server_pid() -> Optional[int]:
    cp = _adb("shell", "ps | grep audio_server | grep -v grep")
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
    return None


def board_demo_pid() -> Optional[int]:
    cp = _adb("shell", "ps | grep -E '/data/build/demo|aikit' | grep -v grep")
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
    return None


def cmd_start_raw(args) -> int:
    if not adb_has_device():
        print("[!] no adb device", file=sys.stderr)
        return 1
    cp = _adb("shell", BOARD_USE_RAW, timeout=15)
    if cp.returncode != 0:
        print(f"[!] {BOARD_USE_RAW} failed: {cp.stderr.strip()}", file=sys.stderr)
        return 2
    # Wait for audio_server to be up
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if board_audio_server_pid() is not None:
            break
        time.sleep(0.3)
    pid = board_audio_server_pid()
    if pid is None:
        print("[!] audio_server did not start within 5s", file=sys.stderr)
        return 3
    if not adb_forward_state(args.port):
        if not adb_forward_apply(args.port):
            print(f"[!] adb forward tcp:{args.port} failed", file=sys.stderr)
            return 2
    print(f"[ok] raw mode active. audio_server pid={pid}, "
          f"forward tcp:{args.port}<->tcp:{args.port} applied")
    return 0


def cmd_start_demo(args) -> int:
    if not adb_has_device():
        print("[!] no adb device", file=sys.stderr)
        return 1
    cp = _adb("shell", BOARD_USE_DEMO, timeout=15)
    if cp.returncode != 0:
        print(f"[!] {BOARD_USE_DEMO} failed: {cp.stderr.strip()}", file=sys.stderr)
        return 2
    print("[ok] demo mode active (audio_server killed)")
    return 0


def cmd_status(args) -> int:
    info = {
        "adb_has_device": adb_has_device(),
        "audio_server_pid": board_audio_server_pid(),
        "demo_pid": board_demo_pid(),
        "forward_active": adb_forward_state(args.port),
        "port": args.port,
    }
    if info["audio_server_pid"]:
        info["mode"] = "raw"
    elif info["demo_pid"]:
        info["mode"] = "demo"
    else:
        info["mode"] = "unknown/idle"
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"adb device: {'OK' if info['adb_has_device'] else 'MISSING'}")
        print(f"mode      : {info['mode']}")
        print(f"  audio_server pid: {info['audio_server_pid']}")
        print(f"  demo pid        : {info['demo_pid']}")
        print(f"  adb forward     : "
              f"{'tcp:%d<->tcp:%d active' % (args.port, args.port) if info['forward_active'] else 'NOT applied'}")
    return 0 if info["adb_has_device"] else 1


def cmd_ensure_forward(args) -> int:
    if not adb_has_device():
        print("[!] no adb device", file=sys.stderr)
        return 1
    if adb_forward_state(args.port):
        print(f"[ok] adb forward tcp:{args.port} already applied")
        return 0
    if adb_forward_apply(args.port):
        print(f"[ok] adb forward tcp:{args.port} applied")
        return 0
    print(f"[!] adb forward tcp:{args.port} failed", file=sys.stderr)
    return 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start-raw").set_defaults(func=cmd_start_raw)
    sub.add_parser("start-demo").set_defaults(func=cmd_start_demo)
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)
    sub.add_parser("ensure-forward").set_defaults(func=cmd_ensure_forward)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
