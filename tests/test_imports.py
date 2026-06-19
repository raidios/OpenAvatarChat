#!/usr/bin/env python3
"""
Stage-0 code-level smoke: every new module imports cleanly without ROS2 or
Hailo hardware present. We only require numpy.

Exit code:
  0  all imports succeeded
  1  any import raised
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO_ROOT / "tests"]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "client"))
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "M2_SDK" / "live_stream" / "host"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

# (module_path, soft) - soft means a failure here is YELLOW not RED.
TARGETS = [
    # tools
    ("board_audio_mode", False),
    ("board_supervisor", False),
    # M260C live stream API (under thirdparty)
    ("live_stream", False),
    # backend abstraction
    ("audio_frontend.backends.denoiser", False),
    ("audio_frontend.backends.pose_detector", False),
    ("audio_frontend.backends.reid", False),
    ("audio_frontend.backends._hailo_probe", False),
    # the chat-engine handler (rclpy may be soft-missing)
    ("handlers.client.ros2_client.client_handler_ros2", True),
    # the audio frontend ROS2 node (rclpy might be missing)
    ("ros2_ws.src.audio_frontend.audio_frontend.audio_frontend_node", True),
    # client-side far-field DSP audio source (must always be importable)
    ("farfield_audio_source", False),
]


def main() -> int:
    failures = []
    softfails = []
    for mod_path, soft in TARGETS:
        try:
            importlib.import_module(mod_path)
            print(f"  ok   {mod_path}")
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc(limit=3)
            print(f"  FAIL {mod_path}\n{tb}")
            (softfails if soft else failures).append(mod_path)
    print()
    print(f"hard failures: {len(failures)}")
    print(f"soft failures: {len(softfails)}")
    if failures:
        print("FAIL")
        return 1
    if softfails:
        print("PASS-WITH-SOFT-FAILS")
        return 0  # soft fails don't break code-level acceptance
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
