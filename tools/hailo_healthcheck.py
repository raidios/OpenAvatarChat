#!/usr/bin/env python3
"""
Hailo-10H health check for the audio frontend.

Verifies:
  1. /dev/h1x-0 exists and is r/w accessible.
  2. `hailortcli scan` finds pci/0000:01:00.0 (or any pci device).
  3. `hailortcli fw-control identify` reports HAILO10H + Firmware 5.3.0.
  4. `from hailo_platform import VDevice` works and a VDevice can be opened.
  5. (optional) Loading a stub HEF and running one inference; skipped if --skip-infer.
  6. Driver kernel module hailo1x_pci is loaded.
  7. Python wheels hailort + tappas-core are installed at matching version.

Outputs JSON at --report (default /tmp/hailo_healthcheck.json) and a human
markdown report on stdout.

Exit code: 0 = all green; 1 = any RED check; 2 = some YELLOW and no RED.

The Hailo facts pinned by .cursor/rules/hailo-device.mdc:
  - device node:  /dev/h1x-0  (NOT /dev/hailo0)
  - kernel mod :  hailo1x_pci (NOT hailo_pci)
  - hailort     : 5.3.0 (deb + driver + wheel must agree)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_DEVICE = "/dev/h1x-0"
EXPECTED_KMOD = "hailo1x_pci"
EXPECTED_FW_FAMILY = "HAILO10H"
EXPECTED_VERSION = "5.3.0"


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, cp.stdout, cp.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def check_device_node() -> dict:
    p = Path(EXPECTED_DEVICE)
    info = {"name": "device_node", "path": str(p)}
    if not p.exists():
        info.update(status="RED", reason=f"{p} does not exist")
        return info
    st = p.stat()
    info["mode"] = oct(st.st_mode)
    rw_ok = os.access(p, os.R_OK | os.W_OK)
    if not rw_ok:
        info.update(status="YELLOW", reason="exists but not r/w by current user")
    else:
        info.update(status="GREEN")
    return info


def check_kmod() -> dict:
    info = {"name": "kernel_module", "expected": EXPECTED_KMOD}
    rc, out, err = _run(["lsmod"])
    if rc != 0:
        info.update(status="RED", reason=f"lsmod failed: {err}")
        return info
    loaded = any(line.split()[0] == EXPECTED_KMOD
                 for line in out.splitlines() if line.strip())
    info.update(status="GREEN" if loaded else "RED",
                loaded=loaded)
    return info


def check_hailortcli_scan() -> dict:
    info = {"name": "hailortcli_scan"}
    bin_path = shutil.which("hailortcli") or "/usr/bin/hailortcli"
    if not Path(bin_path).exists():
        info.update(status="RED", reason="hailortcli not found")
        return info
    rc, out, err = _run([bin_path, "scan"], timeout=10)
    info["stdout"] = out.strip()
    if rc != 0:
        info.update(status="RED", reason=f"scan failed: {err.strip() or out.strip()}")
        return info
    if "pci/" in out:
        m = re.search(r"pci/[0-9a-f:.]+", out)
        info.update(status="GREEN", device=m.group(0) if m else None)
    else:
        info.update(status="RED", reason="no pci device reported")
    return info


def check_fw_identify() -> dict:
    info = {"name": "hailortcli_fw_identify",
            "expected_family": EXPECTED_FW_FAMILY,
            "expected_version": EXPECTED_VERSION}
    bin_path = shutil.which("hailortcli") or "/usr/bin/hailortcli"
    rc, out, err = _run([bin_path, "fw-control", "identify"], timeout=10)
    if rc != 0:
        info.update(status="RED", reason=f"fw-control identify failed: {err.strip() or out.strip()}")
        return info
    info["stdout"] = out.strip()
    family_ok = EXPECTED_FW_FAMILY in out
    ver_ok = EXPECTED_VERSION in out
    info.update(family_ok=family_ok, version_ok=ver_ok)
    if family_ok and ver_ok:
        info["status"] = "GREEN"
    elif family_ok:
        info["status"] = "YELLOW"
        info["reason"] = f"family ok, but firmware version != {EXPECTED_VERSION}"
    else:
        info["status"] = "RED"
        info["reason"] = "firmware family does not match HAILO10H"
    return info


def check_python_wheel() -> dict:
    info = {"name": "python_wheels"}
    try:
        import importlib.metadata as md  # type: ignore
        versions = {}
        for pkg in ("hailort", "hailo-tappas-core-python-binding"):
            try:
                versions[pkg] = md.version(pkg)
            except Exception as e:
                versions[pkg] = f"NOT_INSTALLED ({e})"
        info["versions"] = versions
        all_ok = all(v == EXPECTED_VERSION for v in versions.values())
        if all_ok:
            info["status"] = "GREEN"
        else:
            info["status"] = "YELLOW"
            info["reason"] = f"some pkg(s) not at {EXPECTED_VERSION}"
    except Exception as e:
        info["status"] = "RED"
        info["reason"] = f"importlib.metadata failed: {e}"
    return info


def check_vdevice_open() -> dict:
    info = {"name": "VDevice_open"}
    try:
        from hailo_platform import VDevice  # type: ignore
    except Exception as e:
        info.update(status="RED", reason=f"import hailo_platform failed: {e}")
        return info
    t0 = time.monotonic()
    try:
        with VDevice() as d:
            try:
                ids = d.get_physical_devices_ids()
            except Exception:
                ids = None
            info["physical_ids"] = ids
        info["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
        info["status"] = "GREEN"
    except Exception as e:
        info.update(status="RED", reason=f"VDevice() raised: {e!r}")
    return info


def check_inference(hef_path: str | None) -> dict:
    info = {"name": "inference_smoke", "hef": hef_path}
    if not hef_path:
        info.update(status="SKIP", reason="--hef not provided; skipping smoke inference")
        return info
    if not Path(hef_path).exists():
        info.update(status="RED", reason=f"{hef_path} not found")
        return info
    try:
        import numpy as np  # noqa: F401
        from hailo_platform import (  # type: ignore
            ConfigureParams, FormatType, HailoStreamInterface,
            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice,
        )
        with VDevice() as device:
            from hailo_platform import HEF
            hef = HEF(hef_path)
            cfg_params = ConfigureParams.create_from_hef(
                hef=hef, interface=HailoStreamInterface.PCIe)
            net_groups = device.configure(hef, cfg_params)
            ng = net_groups[0]
            input_params = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            output_params = OutputVStreamParams.make(ng, format_type=FormatType.UINT8)
            input_info = hef.get_input_vstream_infos()[0]
            shape = input_info.shape
            data = np.zeros(shape, dtype=np.uint8)[np.newaxis, ...]
            t0 = time.monotonic()
            with InferVStreams(ng, input_params, output_params) as pipeline, \
                    ng.activate():
                pipeline.infer({input_info.name: data})
            info["latency_ms"] = round((time.monotonic() - t0) * 1000, 2)
            info["status"] = "GREEN"
    except Exception as e:
        info.update(status="RED", reason=f"inference failed: {e!r}")
    return info


CHECKS_ORDER = [
    "device_node", "kernel_module", "hailortcli_scan",
    "hailortcli_fw_identify", "python_wheels", "VDevice_open",
    "inference_smoke",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="/tmp/hailo_healthcheck.json")
    ap.add_argument("--hef", default=None,
                    help="optional HEF file to run a 1-frame smoke inference")
    ap.add_argument("--skip-infer", action="store_true",
                    help="skip the inference smoke even if --hef given")
    args = ap.parse_args()

    results = [
        check_device_node(),
        check_kmod(),
        check_hailortcli_scan(),
        check_fw_identify(),
        check_python_wheel(),
        check_vdevice_open(),
        check_inference(None if args.skip_infer else args.hef),
    ]

    def _color(s: str) -> str:
        codes = {"GREEN": "\033[32m", "YELLOW": "\033[33m",
                 "RED": "\033[31m", "SKIP": "\033[90m"}
        return f"{codes.get(s, '')}{s}\033[0m" if sys.stdout.isatty() else s

    print("# Hailo-10H Health Check\n")
    print("| Check | Status | Detail |")
    print("|---|---|---|")
    n_red = n_yel = 0
    for r in results:
        status = r.get("status", "?")
        if status == "RED":
            n_red += 1
        elif status == "YELLOW":
            n_yel += 1
        detail = r.get("reason") or r.get("stdout") or json.dumps(
            {k: v for k, v in r.items() if k not in ("name", "status", "reason", "stdout")},
            ensure_ascii=False,
        )
        print(f"| {r['name']} | {_color(status)} | {detail} |")

    print()
    overall = "GREEN" if (n_red == 0 and n_yel == 0) else (
        "RED" if n_red else "YELLOW")
    print(f"OVERALL: {_color(overall)}  (red={n_red}, yellow={n_yel})")

    Path(args.report).write_text(json.dumps({
        "overall": overall, "red": n_red, "yellow": n_yel, "checks": results,
    }, indent=2))
    print(f"report -> {args.report}")
    return 0 if overall == "GREEN" else (1 if n_red else 2)


if __name__ == "__main__":
    sys.exit(main())
