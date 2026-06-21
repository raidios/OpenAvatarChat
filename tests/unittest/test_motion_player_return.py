from __future__ import annotations

import os
import sys
import time
import unittest
from dataclasses import dataclass, field


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

from teach.clip_store import Clip, ClipSample  # noqa: E402
from teach.motion_odometry import MotionReturnParams  # noqa: E402
from teach.motion_player import MotionPlayer  # noqa: E402


@dataclass
class FakeMcuData:
    vel_x: int = 0
    vel_y: int = 0
    vel_w: int = 0
    timestamp: float = field(default_factory=time.monotonic)


class FakeSerial:
    def __init__(self) -> None:
        self.sent = []
        self.latest_data = FakeMcuData(timestamp=time.monotonic())
        self.yaw_deg_unwrapped = 0.0
        self.is_ps2_overriding = False
        self.freeze_telemetry = False

    def send_velocity(self, vx: int, vy: int, vw: int) -> None:
        self.sent.append((vx, vy, vw))
        if not self.freeze_telemetry:
            self.latest_data = FakeMcuData(
                vel_x=int(vx),
                vel_y=int(vy),
                vel_w=int(vw),
                timestamp=time.monotonic(),
            )
        self.yaw_deg_unwrapped += (int(vw) / 1000.0) * 0.02 * 180.0 / 3.141592653589793


def make_forward_clip() -> Clip:
    return Clip(
        id="clip-forward",
        name="forward",
        duration_ms=160,
        created_at="",
        samples=[
            ClipSample(t_ms=0, vx=500, vy=0, vw=0),
            ClipSample(t_ms=40, vx=500, vy=0, vw=0),
            ClipSample(t_ms=80, vx=500, vy=0, vw=0),
            ClipSample(t_ms=120, vx=500, vy=0, vw=0),
            ClipSample(t_ms=160, vx=0, vy=0, vw=0),
        ],
    )


def make_player(serial: FakeSerial) -> MotionPlayer:
    return MotionPlayer(
        serial=serial,
        motion_return_enabled=True,
        motion_return_params=MotionReturnParams(
            max_linear_speed=0.12,
            max_angular_speed=0.35,
            position_tolerance_m=0.005,
            yaw_tolerance_rad=0.05,
            timeout_s=1.0,
            control_rate_hz=100.0,
            stale_timeout_s=0.3,
        ),
    )


class MotionPlayerReturnTest(unittest.TestCase):
    def test_real_clip_returns_to_origin_after_normal_completion(self) -> None:
        serial = FakeSerial()
        player = make_player(serial)

        player._play_one(type("Job", (), {"clip": make_forward_clip()})())

        self.assertIn("returning", [s.get("phase") for s in player._phase_history])
        self.assertEqual(player.status()["phase"], "idle")
        self.assertEqual(serial.sent[-1], (0, 0, 0))
        self.assertTrue(any(vx < 0 for vx, _vy, _vw in serial.sent))

    def test_stop_all_during_clip_requests_return_instead_of_plain_stop(self) -> None:
        serial = FakeSerial()
        player = make_player(serial)
        clip = make_forward_clip()
        original_send = serial.send_velocity

        def send_and_abort(vx: int, vy: int, vw: int) -> None:
            original_send(vx, vy, vw)
            if vx > 0 and len([v for v in serial.sent if v[0] > 0]) >= 2:
                player.stop_all()

        serial.send_velocity = send_and_abort

        player._play_one(type("Job", (), {"clip": clip})())

        self.assertIn("returning", [s.get("phase") for s in player._phase_history])
        self.assertTrue(any(vx < 0 for vx, _vy, _vw in serial.sent))

    def test_stop_all_can_abandon_return_for_manual_stop(self) -> None:
        serial = FakeSerial()
        player = make_player(serial)
        original_send = serial.send_velocity

        def send_and_abort(vx: int, vy: int, vw: int) -> None:
            original_send(vx, vy, vw)
            if vx > 0 and len([v for v in serial.sent if v[0] > 0]) >= 2:
                player.stop_all(return_to_origin=False)

        serial.send_velocity = send_and_abort

        player._play_one(type("Job", (), {"clip": make_forward_clip()})())

        self.assertNotIn("returning", [s.get("phase") for s in player._phase_history])

    def test_ps2_override_abandons_return(self) -> None:
        serial = FakeSerial()
        player = make_player(serial)
        original_send = serial.send_velocity

        def send_and_ps2(vx: int, vy: int, vw: int) -> None:
            original_send(vx, vy, vw)
            if vx > 0:
                serial.is_ps2_overriding = True

        serial.send_velocity = send_and_ps2

        player._play_one(type("Job", (), {"clip": make_forward_clip()})())

        self.assertNotIn("returning", [s.get("phase") for s in player._phase_history])
        self.assertEqual(serial.sent[-1], (0, 0, 0))

    def test_stale_telemetry_abandons_return(self) -> None:
        serial = FakeSerial()
        player = MotionPlayer(
            serial=serial,
            motion_return_enabled=True,
            motion_return_params=MotionReturnParams(
                timeout_s=1.0,
                control_rate_hz=100.0,
                stale_timeout_s=0.01,
            ),
        )
        serial.freeze_telemetry = True
        serial.latest_data = FakeMcuData(timestamp=time.monotonic() - 1.0)

        player._play_one(type("Job", (), {"clip": make_forward_clip()})())

        returns = [s for s in player._phase_history if s.get("phase") == "returning"]
        self.assertEqual(returns, [])
        self.assertEqual(serial.sent[-1], (0, 0, 0))

    def test_empty_clip_does_not_return(self) -> None:
        serial = FakeSerial()
        player = make_player(serial)

        player._play_one(type("Job", (), {"clip": None, "placeholder_ms": 100})())

        self.assertNotIn("returning", [s.get("phase") for s in player._phase_history])
        self.assertEqual(serial.sent, [])


if __name__ == "__main__":
    unittest.main()
