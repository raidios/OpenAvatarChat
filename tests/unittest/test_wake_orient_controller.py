from __future__ import annotations

import os
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

from wake_orient_controller import WakeOrientController, WakeOrientParams


class FakeYawSerial:
    def __init__(self) -> None:
        self.yaw_deg_unwrapped = 0.0
        self.sent = []

    @property
    def yaw_deg(self) -> float:
        return ((self.yaw_deg_unwrapped + 180.0) % 360.0) - 180.0

    def send_velocity(self, vx: int, vy: int, vw: int) -> None:
        self.sent.append((vx, vy, vw))
        if vw > 0:
            self.yaw_deg_unwrapped += 4.0
        elif vw < 0:
            self.yaw_deg_unwrapped -= 4.0


class WakeOrientControllerTest(unittest.TestCase):
    def test_default_doa_to_yaw_sign_negates_body_doa(self) -> None:
        serial = FakeYawSerial()
        ctl = WakeOrientController(
            serial=serial,
            params=WakeOrientParams(cooldown_s=0.0, min_doa_deg=0.0),
        )

        ctl.on_wake("test", 90.0)

        self.assertEqual(ctl._pending_goal, ("test", -90.0))

    def test_inverted_doa_to_yaw_sign_preserves_body_doa(self) -> None:
        serial = FakeYawSerial()
        ctl = WakeOrientController(
            serial=serial,
            params=WakeOrientParams(
                cooldown_s=0.0,
                min_doa_deg=0.0,
                doa_to_yaw_sign=1.0,
            ),
        )

        ctl.on_wake("test", 90.0)

        self.assertEqual(ctl._pending_goal, ("test", 90.0))

    def test_reports_incremental_yaw_delta_during_rotation(self) -> None:
        serial = FakeYawSerial()
        deltas = []
        ctl = WakeOrientController(
            serial=serial,
            params=WakeOrientParams(
                max_angular_speed=2.5,
                kp_angle=3.0,
                kd_angle=0.0,
                angle_deadzone_deg=2.5,
                settle_ticks=1,
                control_rate_hz=500.0,
                max_duration_s=0.25,
                min_doa_deg=0.0,
            ),
            on_yaw_delta=deltas.append,
        )
        ctl._running = True

        achieved = ctl._execute_rotation("test", 20.0)

        self.assertGreaterEqual(len(deltas), 2)
        self.assertAlmostEqual(sum(deltas), achieved, delta=0.1)
        self.assertAlmostEqual(serial.sent[-1][2], 0)


if __name__ == "__main__":
    unittest.main()
