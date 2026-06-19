import math
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))

class _FakeCv2(types.SimpleNamespace):
    def __getattr__(self, name):
        value = object()
        setattr(self, name, value)
        return value


sys.modules.setdefault("cv2", _FakeCv2(aruco=types.SimpleNamespace()))
sys.modules.setdefault("serial", types.SimpleNamespace(Serial=object))

from tracking_controller import TrackingController, TrackingParams, TrackingState


class DummySerial:
    def __init__(self):
        self.sent = []

    def send_velocity(self, vx: int, vy: int, vw: int):
        self.sent.append((vx, vy, vw))


def _controller() -> TrackingController:
    ctl = TrackingController.__new__(TrackingController)
    ctl._serial = DummySerial()
    ctl._params = TrackingParams()
    ctl._state = TrackingState.TRACKING
    ctl._state_lock = threading.Lock()
    ctl._last_angle_error = 0.0
    ctl._last_control_time = 1.0
    ctl._last_angle_h = 0.0
    ctl._last_vw = 0.0
    return ctl


class TrackingControllerSafetyTest(unittest.TestCase):
    def test_held_board_pose_stops_instead_of_driving_stale_target(self):
        ctl = _controller()
        target = SimpleNamespace(
            held=True,
            predicted=False,
            angle_h=0.0,
            distance=2.0,
        )

        ctl._handle_tracking(True, target)

        self.assertEqual(ctl._serial.sent[-1], (0, 0, 0))
        self.assertEqual(ctl.state, TrackingState.TRACKING)

    def test_predicted_board_pose_drives_with_reduced_speed(self):
        ctl = _controller()
        target = SimpleNamespace(
            held=True,
            predicted=True,
            angle_h=0.2,
            distance=1.2,
        )

        ctl._handle_tracking(True, target)

        vx, _vy, vw = ctl._serial.sent[-1]
        self.assertGreater(vx, 0)
        self.assertLessEqual(abs(vx), 150)
        self.assertLessEqual(abs(vw), 600)

    def test_non_finite_target_stops_and_returns_to_idle(self):
        ctl = _controller()
        target = SimpleNamespace(
            held=False,
            angle_h=float("nan"),
            distance=1.0,
        )

        ctl._handle_tracking(True, target)

        self.assertEqual(ctl._serial.sent[-1], (0, 0, 0))
        self.assertEqual(ctl.state, TrackingState.IDLE)

    def test_default_speed_caps_are_conservative_for_indoor_testing(self):
        params = TrackingParams()

        self.assertLessEqual(params.max_linear_speed, 0.35)
        self.assertLessEqual(params.max_angular_speed, 0.9)
        self.assertLessEqual(params.kd_angle, 0.2)
        self.assertTrue(math.isfinite(params.max_linear_speed))
        self.assertTrue(math.isfinite(params.max_angular_speed))

    def test_angular_velocity_step_is_rate_limited(self):
        ctl = _controller()

        limited = ctl._limit_angular_step(0.8, 0.05)

        self.assertLessEqual(limited, 0.2)
        self.assertEqual(ctl._last_vw, limited)


if __name__ == "__main__":
    unittest.main()
