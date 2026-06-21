from __future__ import annotations

import os
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

from teach.clip_store import Clip, ClipSample
from teach.motion_player import MotionPlayer


class FakeSerial:
    def __init__(self) -> None:
        self.yaw_deg_unwrapped = 0.0
        self.sent = []
        self.is_ps2_overriding = False

    def send_velocity(self, vx: int, vy: int, vw: int) -> None:
        self.sent.append((vx, vy, vw))
        if vw > 0:
            self.yaw_deg_unwrapped += 3.0
        elif vw < 0:
            self.yaw_deg_unwrapped -= 3.0


class MotionPlayerYawDeltaTest(unittest.TestCase):
    def test_clip_playback_reports_incremental_yaw_delta(self) -> None:
        serial = FakeSerial()
        deltas = []
        player = MotionPlayer(serial=serial, yaw_delta_callback=deltas.append)
        clip = Clip(
            id="clip1",
            name="turn",
            duration_ms=3,
            created_at="",
            samples=[
                ClipSample(t_ms=0, vx=0, vy=0, vw=500),
                ClipSample(t_ms=1, vx=0, vy=0, vw=500),
                ClipSample(t_ms=2, vx=0, vy=0, vw=500),
            ],
        )

        player._play_one(type("Job", (), {"clip": clip})())

        self.assertGreaterEqual(len(deltas), 1)
        self.assertAlmostEqual(sum(deltas), serial.yaw_deg_unwrapped, delta=0.1)
        self.assertEqual(serial.sent[-1], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
