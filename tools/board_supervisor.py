#!/usr/bin/env python3
"""
Watchdog for the WHEELTEC R818 board's audio link.

Loop:
  - check `adb devices` is non-empty
  - check audio_server process is alive on the board
  - check `adb forward tcp:9999 tcp:9999` is applied on host
  - check TCP 127.0.0.1:9999 is reachable (a tiny health probe)
  - if any check fails -> attempt remediation (reapply forward, start-raw)

Exposes a small text status line + Prometheus-style counters for log scraping:
  audio_stream_uptime_seconds <s>
  audio_stream_reconnect_count <n>
  audio_stream_last_failure   "<reason>"

CLI:
  python tools/board_supervisor.py --interval 2 --log /var/log/audio_supervisor.log

Designed to be wrapped by `systemd --user` (preferred) or run in a tmux pane.
A sample unit file is emitted with `--print-systemd-unit`.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import board_audio_mode as bam  # noqa: E402

PORT = bam.DEFAULT_PORT


SYSTEMD_UNIT = """\
[Unit]
Description=M2 board audio_server supervisor (adb-forward keepalive)
After=network.target

[Service]
Type=simple
ExecStart={python} {script} --interval 2 --log {logfile}
Restart=always
RestartSec=5
WorkingDirectory={cwd}
Environment=PATH={path}

[Install]
WantedBy=multi-user.target
"""


def tcp_probe(host: str = "127.0.0.1", port: int = PORT,
              timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            # Read 1 byte (audio_server starts streaming immediately on connect).
            s.settimeout(0.5)
            try:
                buf = s.recv(1)
                return len(buf) == 1
            except socket.timeout:
                return False
    except OSError:
        return False


class State:
    def __init__(self, log_fp=None):
        self.uptime_start = None
        self.reconnect_count = 0
        self.last_failure = ""
        self.last_status = "INIT"
        self.log_fp = log_fp

    def emit(self, msg: str):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        if self.log_fp:
            self.log_fp.write(line + "\n")
            self.log_fp.flush()

    def emit_metrics(self):
        if self.uptime_start is None:
            uptime = 0.0
        else:
            uptime = time.monotonic() - self.uptime_start
        line = json.dumps({
            "metric": "audio_stream",
            "uptime_seconds": round(uptime, 1),
            "reconnect_count": self.reconnect_count,
            "last_failure": self.last_failure,
            "status": self.last_status,
        })
        if self.log_fp:
            self.log_fp.write(line + "\n")
            self.log_fp.flush()


def remediate(state: State) -> bool:
    """Try to bring link back up. Return True if we believe it's healthy."""
    if not bam.adb_has_device():
        state.last_failure = "no adb device"
        return False
    pid = bam.board_audio_server_pid()
    if pid is None:
        state.emit("audio_server not running -> running start-raw")
        cp = subprocess.run(
            ["adb", "shell", bam.BOARD_USE_RAW],
            capture_output=True, text=True, timeout=15,
        )
        if cp.returncode != 0:
            state.last_failure = f"start-raw failed: {cp.stderr.strip()}"
            return False
        time.sleep(1.5)
        if bam.board_audio_server_pid() is None:
            state.last_failure = "start-raw did not bring up audio_server"
            return False
    if not bam.adb_forward_state(PORT):
        state.emit(f"adb forward missing -> applying tcp:{PORT}")
        if not bam.adb_forward_apply(PORT):
            state.last_failure = "adb forward apply failed"
            return False
    if not tcp_probe(port=PORT):
        state.last_failure = "tcp probe failed (forward applied but no data)"
        return False
    return True


def supervisor_loop(interval: float, log_fp) -> int:
    state = State(log_fp=log_fp)
    healthy_prev = False
    while True:
        ok = (bam.adb_has_device()
              and bam.board_audio_server_pid() is not None
              and bam.adb_forward_state(PORT)
              and tcp_probe(port=PORT))
        if ok:
            if not healthy_prev:
                state.uptime_start = time.monotonic()
                state.emit(f"link UP "
                           f"(audio_server pid={bam.board_audio_server_pid()}, "
                           f"forward tcp:{PORT})")
            state.last_status = "UP"
            healthy_prev = True
        else:
            if healthy_prev:
                state.emit("link DOWN -> attempting remediation")
            healthy_prev = False
            healed = remediate(state)
            if healed:
                state.reconnect_count += 1
                state.uptime_start = time.monotonic()
                state.emit(f"link RECOVERED (count={state.reconnect_count})")
                state.last_status = "UP"
                healthy_prev = True
            else:
                state.last_status = f"DOWN: {state.last_failure}"
        state.emit_metrics()
        time.sleep(interval)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--log", default=None,
                   help="append-only log file with status+metrics jsonl")
    p.add_argument("--print-systemd-unit", action="store_true",
                   help="print a systemd unit and exit")
    args = p.parse_args()

    if args.print_systemd_unit:
        print(SYSTEMD_UNIT.format(
            python=sys.executable,
            script=str(Path(__file__).resolve()),
            logfile=args.log or "/var/log/audio_supervisor.log",
            cwd=str(REPO_ROOT),
            path=os.environ.get("PATH", "/usr/bin:/bin"),
        ))
        return 0

    log_fp = open(args.log, "a") if args.log else None
    try:
        return supervisor_loop(args.interval, log_fp)
    except KeyboardInterrupt:
        return 0
    finally:
        if log_fp:
            log_fp.close()


if __name__ == "__main__":
    sys.exit(main())
