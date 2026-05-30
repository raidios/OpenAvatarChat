#!/usr/bin/env python3
"""Probe the chassis vw / IMU yaw polarity convention.

Sends a small known angular velocity to the MCU for a fixed time, then
prints what the IMU said the body yaw did. By running both signs in a
row you can answer two independent questions:

  1. Does ``send_velocity(0, 0, +X mrad/s)`` rotate the chassis CCW
     (= robot's left, viewed from above) or CW (right)?
  2. Does the on-board gyro_z report increasing yaw_deg when the
     chassis rotates CCW (the standard convention) or the opposite?

The wake-orient controller assumes BOTH are CCW-positive, which is the
same convention TrackingController has been using for AprilTag track-
ing. If the actual robot rotates the OTHER way for a positive vw, or
if yaw_deg moves opposite to the physical rotation, we know the issue
is in the chassis stack and not in the DSP/DOA path.

Usage on the robot (no audio/camera required):

    .venv/bin/python tools/probe_yaw_polarity.py \
        --serial-port /dev/ttyAMA0

Stand back; the robot will rotate ~0.5 rad/s for 0.7 s in one direction,
stop, then in the other. Watch which way it physically turns.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))


def _drive(serial, vw_mrps: float, dt_s: float) -> tuple[float, float]:
    """Issue ``vw_mrps`` for ``dt_s`` seconds at 50 Hz; return (start_yaw, end_yaw)."""
    rate_hz = 50.0
    n_ticks = int(dt_s * rate_hz)
    start_yaw = serial.yaw_deg_unwrapped
    deadline = time.monotonic() + dt_s
    next_tick = time.monotonic()
    for _ in range(n_ticks):
        serial.send_velocity(0, 0, int(vw_mrps))
        next_tick += 1.0 / rate_hz
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        if time.monotonic() >= deadline:
            break
    serial.send_velocity(0, 0, 0)
    time.sleep(0.4)
    end_yaw = serial.yaw_deg_unwrapped
    return start_yaw, end_yaw


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--serial-port", default="/dev/ttyAMA0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--vw-rps", type=float, default=0.5,
                    help="angular speed magnitude in rad/s (default: 0.5)")
    ap.add_argument("--duration-s", type=float, default=0.7,
                    help="how long to drive in each direction (default 0.7)")
    args = ap.parse_args()

    from serial_comm import XProtocolSerial

    serial = XProtocolSerial(args.serial_port, args.baud)
    serial.start()
    try:
        # Wait for the bias bootstrap — the gyro_z bias EMA only kicks in
        # after a few hundred ms of detected static state.
        print("waiting 2s for gyro bias to bootstrap...")
        time.sleep(2.0)
        bias = serial.gyro_z_bias_dps
        print(f"  gyro_z bias = "
              f"{bias if bias is None else f'{bias:+.2f}'} dps")
        print(f"  starting yaw = {serial.yaw_deg:+.1f}° "
              f"(unwrapped {serial.yaw_deg_unwrapped:+.1f}°)")

        for sign, label in ((+1.0, "POSITIVE"), (-1.0, "NEGATIVE")):
            input(f"\n[{label} vw test]  "
                  f"about to send vw = {sign * args.vw_rps:+.2f} rad/s "
                  f"for {args.duration_s:.1f} s.\n"
                  f"Watch which way the chassis physically rotates. "
                  f"Press ENTER when ready.")
            mrps = sign * args.vw_rps * 1000.0
            start, end = _drive(serial, mrps, args.duration_s)
            delta = end - start
            print(f"  yaw_unwrapped: {start:+.1f}° -> {end:+.1f}° "
                  f"(delta {delta:+.1f}°)")
            print(f"  vw command = {mrps:+.0f} mrad/s for {args.duration_s:.1f} s")
            expected_deg = sign * args.vw_rps * args.duration_s * 180 / 3.14159
            print(f"  expected  ~ {expected_deg:+.1f}°  if convention is "
                  f"'+vw = CCW = +yaw'")
            actual_dir = "CCW (left)" if delta > 0 else (
                "CW (right)" if delta < 0 else "no movement")
            cmd_dir = "CCW (left)" if sign > 0 else "CW (right)"
            print(f"  IMU says rotation was: {actual_dir}")
            print(f"  command intended:      {cmd_dir} "
                  f"(per the +vw=CCW assumption)")

        print("\nINTERPRETATION CHEAT SHEET")
        print("  Both POSITIVE vw → physical CCW (left) AND +yaw delta:")
        print("    convention is correct, problem must be in DSP/DOA.")
        print("  POSITIVE vw rotates physical CW (right):")
        print("    MCU vw polarity is opposite of comments; flip in")
        print("    wake_orient_controller._send_velocity (and ideally in")
        print("    tracking_controller too).")
        print("  Physical CCW but yaw delta is NEGATIVE:")
        print("    gyro_z polarity inverted; integrator in serial_comm "
              "needs a sign flip.")
    finally:
        serial.send_velocity(0, 0, 0)
        time.sleep(0.1)
        serial.stop()


if __name__ == "__main__":
    main()
