from __future__ import annotations

import math
import os
import sys
import time
import unittest
from dataclasses import dataclass, field


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

from teach.motion_odometry import (  # noqa: E402
    MotionPose,
    MotionReturnParams,
    WheelImuOdometryBackend,
    compute_return_command,
)


@dataclass
class FakeMcuData:
    vel_x: int = 0
    vel_y: int = 0
    vel_w: int = 0
    timestamp: float = field(default_factory=time.monotonic)


class FakeSerial:
    def __init__(self) -> None:
        self.latest_data = FakeMcuData()
        self.yaw_deg_unwrapped = 0.0


class WheelImuOdometryBackendTest(unittest.TestCase):
    def test_integrates_encoder_translation_in_start_frame(self) -> None:
        serial = FakeSerial()
        odom = WheelImuOdometryBackend(serial=serial, stale_timeout_s=1.0)
        odom.reset_origin(now_s=10.0)

        serial.latest_data = FakeMcuData(vel_x=100, vel_y=0, vel_w=0, timestamp=10.1)
        odom.update(now_s=10.1)
        serial.latest_data = FakeMcuData(vel_x=100, vel_y=0, vel_w=0, timestamp=10.2)
        odom.update(now_s=10.2)

        pose = odom.pose()
        self.assertAlmostEqual(pose.x_m, 0.02, delta=0.002)
        self.assertAlmostEqual(pose.y_m, 0.0, delta=0.002)
        self.assertAlmostEqual(pose.yaw_rad, 0.0, delta=0.001)
        self.assertTrue(odom.healthy(now_s=10.2))

    def test_uses_imu_yaw_for_heading_and_frame_rotation(self) -> None:
        serial = FakeSerial()
        odom = WheelImuOdometryBackend(serial=serial, stale_timeout_s=1.0)
        odom.reset_origin(now_s=20.0)

        serial.yaw_deg_unwrapped = 90.0
        serial.latest_data = FakeMcuData(vel_x=1000, vel_y=0, vel_w=0, timestamp=20.1)
        odom.update(now_s=20.1)

        pose = odom.pose()
        self.assertAlmostEqual(pose.x_m, 0.0, delta=0.005)
        self.assertAlmostEqual(pose.y_m, 0.1, delta=0.005)
        self.assertAlmostEqual(pose.yaw_rad, math.radians(90), delta=0.001)

    def test_reports_unhealthy_when_telemetry_is_stale(self) -> None:
        serial = FakeSerial()
        odom = WheelImuOdometryBackend(serial=serial, stale_timeout_s=0.3)
        odom.reset_origin(now_s=30.0)
        serial.latest_data = FakeMcuData(vel_x=0, timestamp=30.0)
        odom.update(now_s=30.0)

        self.assertFalse(odom.healthy(now_s=30.5))

    def test_return_command_uses_robot_frame_error_after_yaw_crossing(self) -> None:
        pose = MotionPose(
            x_m=0.003,
            y_m=-0.045,
            yaw_rad=math.radians(41.8),
        )
        params = MotionReturnParams(
            max_linear_speed=0.12,
            max_angular_speed=0.35,
            position_tolerance_m=0.03,
            yaw_tolerance_rad=math.radians(3.0),
        )

        vx, vw, done = compute_return_command(pose, params)

        self.assertFalse(done)
        self.assertLess(vw, 0.0)
        self.assertGreaterEqual(abs(vx), 0.0)

    def test_return_command_turns_in_place_for_large_heading_error(self) -> None:
        pose = MotionPose(
            x_m=-0.322,
            y_m=-0.199,
            yaw_rad=math.radians(-31.6),
        )
        params = MotionReturnParams(
            max_linear_speed=0.12,
            max_angular_speed=0.35,
            position_tolerance_m=0.03,
            yaw_tolerance_rad=math.radians(3.0),
        )

        vx, vw, done = compute_return_command(pose, params)

        self.assertFalse(done)
        self.assertAlmostEqual(vx, 0.0, delta=1e-6)
        self.assertGreater(vw, 0.0)

    def test_return_command_can_reverse_when_origin_is_behind(self) -> None:
        pose = MotionPose(
            x_m=0.066,
            y_m=-0.029,
            yaw_rad=math.radians(-11.4),
        )
        params = MotionReturnParams(
            max_linear_speed=0.12,
            max_angular_speed=0.35,
            position_tolerance_m=0.03,
            yaw_tolerance_rad=math.radians(3.0),
        )

        vx, vw, done = compute_return_command(pose, params)

        self.assertFalse(done)
        self.assertLess(vx, 0.0)

    def test_return_command_keeps_driving_near_origin_with_diagonal_error(self) -> None:
        pose = MotionPose(
            x_m=0.038,
            y_m=-0.038,
            yaw_rad=math.radians(-2.7),
        )
        params = MotionReturnParams(
            max_linear_speed=0.12,
            max_angular_speed=0.35,
            position_tolerance_m=0.03,
            yaw_tolerance_rad=math.radians(3.0),
        )

        vx, vw, done = compute_return_command(pose, params)

        self.assertFalse(done)
        self.assertLess(vx, 0.0)

    def test_return_command_settles_in_practical_tail_tolerance(self) -> None:
        pose = MotionPose(
            x_m=0.016,
            y_m=-0.031,
            yaw_rad=math.radians(-4.6),
        )
        params = MotionReturnParams(
            max_linear_speed=0.12,
            max_angular_speed=0.35,
            position_tolerance_m=0.03,
            yaw_tolerance_rad=math.radians(3.0),
        )

        vx, vw, done = compute_return_command(pose, params)

        self.assertTrue(done)
        self.assertEqual((vx, vw), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
