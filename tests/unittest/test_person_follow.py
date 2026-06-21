import math
import os
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

class _FakeCv2(types.SimpleNamespace):
    def __getattr__(self, name):
        value = object()
        setattr(self, name, value)
        return value


sys.modules.setdefault("cv2", _FakeCv2(aruco=types.SimpleNamespace()))
sys.modules.setdefault("serial", types.SimpleNamespace(Serial=object))

from person_follow import (  # noqa: E402
    PersonFollowController,
    PersonFollowState,
    PersonPerception,
    PersonTarget,
)
from tracking_controller import TrackingParams  # noqa: E402


class DummySerial:
    def __init__(self):
        self.sent = []

    def send_velocity(self, vx: int, vy: int, vw: int):
        self.sent.append((vx, vy, vw))


class DummyPerception:
    def __init__(self):
        self.owner_target = None
        self.reacquired_target = None
        self.bound = False
        self.process_calls = 0
        self.reacquire_calls = 0

    def bind_owner(self, *_args, **_kwargs):
        self.bound = self.owner_target is not None
        return self.owner_target

    def get_owner_target(self):
        return self.owner_target

    def reacquire_owner(self, *_args, **_kwargs):
        self.reacquire_calls += 1
        if self.reacquired_target is not None:
            self.owner_target = self.reacquired_target
        return self.reacquired_target

    def process_once(self):
        self.process_calls += 1
        return []


class PersonPerceptionBindingTest(unittest.TestCase):
    def test_bind_owner_refuses_when_rgbd_frame_is_missing(self):
        class NoRgbdCamera:
            def get_rgbd_frame(self):
                return None, None, 0, 0.0

        perception = PersonPerception(
            camera=NoRgbdCamera(),
            pose_detector=SimpleNamespace(name="fake", detect=lambda _bgr: []),
            start_thread=False,
        )

        self.assertIsNone(perception.bind_owner())
        self.assertEqual(perception.disabled_reason, "rgbd_unavailable")

    def test_bind_owner_picks_front_center_safe_depth_target(self):
        perception = PersonPerception(camera=None, start_thread=False)
        left = PersonTarget(
            track_id=1,
            angle_h=math.radians(-18),
            distance=1.0,
            frame_id=7,
            timestamp=time.monotonic(),
            confidence=0.9,
        )
        center = PersonTarget(
            track_id=2,
            angle_h=math.radians(3),
            distance=1.2,
            frame_id=7,
            timestamp=time.monotonic(),
            confidence=0.8,
        )
        perception._latest_targets = [left, center]
        perception._latest_frame_id = 7
        perception.process_once = lambda: [left, center]

        bound = perception.bind_owner()

        self.assertIsNotNone(bound)
        self.assertEqual(bound.track_id, 2)
        self.assertEqual(perception.owner_track_id, 2)

    def test_bind_owner_waits_past_invalid_depth_candidate(self):
        perception = PersonPerception(camera=None, start_thread=False)
        invalid = PersonTarget(
            track_id=1,
            angle_h=0.0,
            distance=0.0,
            frame_id=1,
            timestamp=time.monotonic(),
            confidence=0.8,
            valid_depth=False,
        )
        valid = PersonTarget(
            track_id=2,
            angle_h=math.radians(2),
            distance=1.0,
            frame_id=2,
            timestamp=time.monotonic(),
            confidence=0.9,
            valid_depth=True,
        )
        calls = {"n": 0}

        def process_once():
            calls["n"] += 1
            perception._latest_targets = [invalid] if calls["n"] == 1 else [valid]
            return perception._latest_targets

        perception.process_once = process_once

        bound = perception.bind_owner()

        self.assertIsNotNone(bound)
        self.assertEqual(bound.track_id, 2)
        self.assertEqual(calls["n"], 2)

    def test_reacquire_owner_to_single_valid_target(self):
        perception = PersonPerception(camera=None, start_thread=False)
        perception.owner_track_id = 1
        replacement = PersonTarget(
            track_id=2,
            angle_h=math.radians(4),
            distance=1.4,
            frame_id=3,
            timestamp=time.monotonic(),
            confidence=0.8,
            valid_depth=True,
        )
        perception._latest_targets = [replacement]

        reacquired = perception.reacquire_owner(max_age_s=0.8)

        self.assertIsNotNone(reacquired)
        self.assertEqual(reacquired.track_id, 2)
        self.assertEqual(perception.owner_track_id, 2)

    def test_reacquire_owner_refuses_ambiguous_targets(self):
        perception = PersonPerception(camera=None, start_thread=False)
        perception.owner_track_id = 1
        now = time.monotonic()
        perception._latest_targets = [
            PersonTarget(
                track_id=2,
                angle_h=math.radians(-3),
                distance=1.0,
                frame_id=3,
                timestamp=now,
                confidence=0.8,
                valid_depth=True,
            ),
            PersonTarget(
                track_id=3,
                angle_h=math.radians(5),
                distance=1.2,
                frame_id=3,
                timestamp=now,
                confidence=0.8,
                valid_depth=True,
            ),
        ]

        self.assertIsNone(perception.reacquire_owner(max_age_s=0.8))
        self.assertEqual(perception.owner_track_id, 1)


class PersonFollowControllerTest(unittest.TestCase):
    def _controller(self):
        perception = DummyPerception()
        serial = DummySerial()
        ctl = PersonFollowController.__new__(PersonFollowController)
        ctl._perception = perception
        ctl._serial = serial
        ctl._params = TrackingParams()
        ctl._state = PersonFollowState.IDLE
        ctl._state_lock = threading.Lock()
        ctl._running = False
        ctl._thread = None
        ctl._estop = False
        ctl._paused = False
        ctl._owner_bound = False
        ctl._last_target_time = 0.0
        ctl._last_angle_error = 0.0
        ctl._last_control_time = 0.0
        ctl._last_angle_h = 0.0
        ctl._last_vw = 0.0
        ctl._lost_timeout_s = 0.4
        ctl._perception_period_s = 0.2
        ctl._last_perception_time = 0.0
        return ctl, perception, serial

    def test_bind_after_wake_stays_idle_when_no_person_target(self):
        ctl, _perception, serial = self._controller()

        self.assertIsNone(ctl.bind_after_wake())

        self.assertEqual(ctl.state, PersonFollowState.IDLE)
        self.assertEqual(serial.sent[-1], (0, 0, 0))

    def test_first_visible_owner_tick_enters_tracking_without_velocity_pulse(self):
        ctl, perception, serial = self._controller()
        perception.owner_target = PersonTarget(
            track_id=4,
            angle_h=0.20,
            distance=1.0,
            frame_id=1,
            timestamp=time.monotonic(),
            confidence=0.9,
        )

        bound = ctl.bind_after_wake()
        ctl._step()

        self.assertIsNotNone(bound)
        self.assertEqual(ctl.state, PersonFollowState.TRACKING)
        self.assertEqual(serial.sent[-1], (0, 0, 0))

    def test_tracking_owner_uses_conservative_speed_caps(self):
        ctl, perception, serial = self._controller()
        ctl._owner_bound = True
        ctl._state = PersonFollowState.TRACKING
        ctl._last_control_time = time.monotonic() - 0.05
        perception.owner_target = PersonTarget(
            track_id=4,
            angle_h=0.10,
            distance=1.2,
            frame_id=2,
            timestamp=time.monotonic(),
            confidence=0.9,
        )

        ctl._step()

        vx, _vy, vw = serial.sent[-1]
        self.assertGreater(vx, 0)
        self.assertLessEqual(abs(vx), 300)
        self.assertLessEqual(abs(vw), 800)

    def test_lost_owner_timeout_stops_and_returns_idle(self):
        ctl, perception, serial = self._controller()
        ctl._owner_bound = True
        ctl._state = PersonFollowState.TRACKING
        ctl._last_target_time = time.monotonic() - 1.0
        perception.owner_target = None

        ctl._step()

        self.assertEqual(serial.sent[-1], (0, 0, 0))
        self.assertEqual(ctl.state, PersonFollowState.IDLE)

    def test_missing_owner_reacquires_before_lost_timeout(self):
        ctl, perception, serial = self._controller()
        ctl._owner_bound = True
        ctl._state = PersonFollowState.TRACKING
        ctl._last_target_time = time.monotonic()
        perception.owner_target = None
        perception.reacquired_target = PersonTarget(
            track_id=5,
            angle_h=0.0,
            distance=1.0,
            frame_id=3,
            timestamp=time.monotonic(),
            confidence=0.9,
            valid_depth=True,
        )

        ctl._step()

        self.assertEqual(perception.reacquire_calls, 1)
        self.assertTrue(ctl._owner_bound)
        self.assertEqual(ctl.state, PersonFollowState.TRACKING)
        self.assertEqual(serial.sent[-1], (0, 0, 0))

    def test_transient_invalid_owner_stops_without_unbinding(self):
        ctl, perception, serial = self._controller()
        ctl._owner_bound = True
        ctl._state = PersonFollowState.TRACKING
        ctl._last_target_time = time.monotonic()
        perception.owner_target = PersonTarget(
            track_id=4,
            angle_h=0.0,
            distance=0.0,
            frame_id=2,
            timestamp=time.monotonic(),
            confidence=0.9,
            valid_depth=False,
        )

        ctl._step()

        self.assertEqual(serial.sent[-1], (0, 0, 0))
        self.assertTrue(ctl._owner_bound)
        self.assertEqual(ctl.state, PersonFollowState.TRACKING)

    def test_person_follow_can_accept_three_meter_valid_range(self):
        ctl, _perception, _serial = self._controller()
        ctl._params.max_valid_distance = 3.0
        target = PersonTarget(
            track_id=4,
            angle_h=0.0,
            distance=2.5,
            frame_id=2,
            timestamp=time.monotonic(),
            confidence=0.9,
            valid_depth=True,
        )

        self.assertTrue(ctl._target_is_valid(target))

    def test_idle_unbound_controller_does_not_run_pose_updates(self):
        ctl, perception, _serial = self._controller()
        ctl._owner_bound = False

        ctl._step()

        self.assertEqual(perception.process_calls, 0)

    def test_bound_controller_refreshes_perception_before_driving(self):
        ctl, perception, serial = self._controller()
        ctl._owner_bound = True
        ctl._state = PersonFollowState.TRACKING
        ctl._last_control_time = time.monotonic() - 0.05
        perception.owner_target = PersonTarget(
            track_id=4,
            angle_h=0.0,
            distance=1.2,
            frame_id=2,
            timestamp=time.monotonic(),
            confidence=0.9,
        )

        ctl._step()

        self.assertEqual(perception.process_calls, 1)
        self.assertNotEqual(serial.sent[-1], (0, 0, 0))

    def test_bound_controller_rate_limits_pose_refresh(self):
        ctl, perception, _serial = self._controller()
        ctl._owner_bound = True
        ctl._state = PersonFollowState.TRACKING
        ctl._last_control_time = time.monotonic() - 0.05
        perception.owner_target = PersonTarget(
            track_id=4,
            angle_h=0.0,
            distance=1.2,
            frame_id=2,
            timestamp=time.monotonic(),
            confidence=0.9,
        )

        ctl._step()
        ctl._step()

        self.assertEqual(perception.process_calls, 1)


if __name__ == "__main__":
    unittest.main()
