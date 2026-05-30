#!/usr/bin/env python3
"""Stream live DOA from M260C through the production DSP pipeline.

This is the diagnostic tool you reach for when the wake-orient
controller rotates in the wrong direction or by the wrong amount: it
isolates the mic-array → DOA → body-frame conversion path, with no
KWS, no rotation control, no chassis at all.

Usage on the robot:

    .venv/bin/python tools/live_doa_monitor.py

Stand at a KNOWN physical position (e.g. 90° to the robot's left when
viewed from above). Have a continuous noise source play there (your
phone playing ``tests/data/doa_test_noise.wav`` works fine), and watch
the body-frame DOA print every ~250 ms. The expected reading is
``body +90°`` if you're on the robot's left, ``-90°`` on its right,
``+180`` / ``-180`` directly behind, ``0`` directly in front.

Conventions printed on every line:
  body_az  = streaming body-frame azimuth (CW positive viewed from
             above; +90° means a source on the robot's RIGHT, -90° on
             its LEFT). Updated every 100 ms over a single short
             window — noisy, mostly here to track motion.
  wake_az  = wake-window body-frame azimuth (same sign convention).
             Computed by integrating SRP-PHAT over the rolling 1.5 s
             VAD-positive segment, falling back to "loudest 30%" if
             the VAD is empty. This is what ``WakeWordSpotter``
             actually consumes when it fires; if you want to ground
             the rotation controller, watch this column.
  mic_az   = streaming azimuth in the MIC-ARRAY frame (0° = logical
             mic 0, which sits at body +mic_yaw_offset_deg).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--mic-yaw-offset-deg", type=float, default=30.0)
    ap.add_argument("--no-aec-per-mic", action="store_true",
                    help="run a single AEC after MVDR (legacy order). "
                         "Default uses the production per-mic-before-MVDR.")
    ap.add_argument("--print-interval-s", type=float, default=0.25)
    ap.add_argument("--no-board-bootstrap", action="store_true",
                    help="don't auto-start the board's raw-mode TCP server")
    args = ap.parse_args()

    from farfield_audio_source import FarfieldAudioSource

    src = FarfieldAudioSource(
        host=args.host,
        port=args.port,
        mic_yaw_offset_deg=args.mic_yaw_offset_deg,
        aec_per_mic=not args.no_aec_per_mic,
        dns_use_passthrough=True,           # keep BF clean for sanity
        use_doa_for_mvdr=True,
        auto_start_audio_server=not args.no_board_bootstrap,
    )
    src.open()
    print(f"# host={args.host}:{args.port}  "
          f"mic_yaw_offset={args.mic_yaw_offset_deg:+.1f}°")
    print("# columns: t   body_az_deg   wake_az_deg   mic_az_deg")

    chunk = 1600  # 100ms @ 16k
    last_print = 0.0
    t0 = time.monotonic()
    try:
        while True:
            src.read(chunk)
            now = time.monotonic()
            if now - last_print < args.print_interval_s:
                continue
            last_print = now
            body_az = src.latest_doa_deg
            mic_az = src.latest_doa_mic_deg
            wake_az = src.wake_doa_snapshot_deg
            wake_s = ("  n/a "
                      if wake_az is None else f"{wake_az:+7.1f}°")
            print(f"{now - t0:6.2f}  body={body_az:+7.1f}°   "
                  f"wake={wake_s}   mic={mic_az:+7.1f}°")
    except KeyboardInterrupt:
        print("\n# stopped by user")
    finally:
        try:
            src.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
