from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def _wrap_pi(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class MotionPose:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)


@dataclass
class MotionReturnParams:
    max_linear_speed: float = 0.12
    max_angular_speed: float = 0.35
    position_tolerance_m: float = 0.03
    yaw_tolerance_rad: float = math.radians(3.0)
    timeout_s: float = 10.0
    control_rate_hz: float = 20.0
    stale_timeout_s: float = 0.3
    kp_position: float = 0.8
    kp_yaw: float = 1.2


@runtime_checkable
class MotionOdometryBackend(Protocol):
    def reset_origin(self, now_s: float | None = None) -> None:
        ...

    def update(self, now_s: float | None = None) -> MotionPose:
        ...

    def pose(self) -> MotionPose:
        ...

    def healthy(self, now_s: float | None = None) -> bool:
        ...


class WheelImuOdometryBackend:
    """Short-horizon relative odometry from MCU encoder velocity + IMU yaw."""

    def __init__(self, serial, stale_timeout_s: float = 0.3):
        self._serial = serial
        self._stale_timeout_s = float(stale_timeout_s)
        self._origin_yaw_deg = 0.0
        self._last_update_s: float | None = None
        self._pose = MotionPose()
        self.reset_origin()

    def reset_origin(self, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else float(now_s)
        self._origin_yaw_deg = float(getattr(self._serial, "yaw_deg_unwrapped", 0.0))
        self._last_update_s = now
        self._pose = MotionPose()

    def update(self, now_s: float | None = None) -> MotionPose:
        now = time.monotonic() if now_s is None else float(now_s)
        if self._last_update_s is None:
            self._last_update_s = now
            return self._pose

        dt = max(0.0, min(now - self._last_update_s, 0.2))
        self._last_update_s = now

        data = getattr(self._serial, "latest_data", None)
        if data is None:
            return self._pose
        vx_mps = float(getattr(data, "vel_x", 0)) / 1000.0
        vy_mps = float(getattr(data, "vel_y", 0)) / 1000.0
        yaw_now_deg = float(getattr(self._serial, "yaw_deg_unwrapped", self._origin_yaw_deg))
        yaw_rad = math.radians(yaw_now_deg - self._origin_yaw_deg)
        yaw_rad = _wrap_pi(yaw_rad)

        cy = math.cos(yaw_rad)
        sy = math.sin(yaw_rad)
        self._pose.x_m += (vx_mps * cy - vy_mps * sy) * dt
        self._pose.y_m += (vx_mps * sy + vy_mps * cy) * dt
        self._pose.yaw_rad = yaw_rad
        return self._pose

    def pose(self) -> MotionPose:
        return MotionPose(self._pose.x_m, self._pose.y_m, self._pose.yaw_rad)

    def healthy(self, now_s: float | None = None) -> bool:
        data = getattr(self._serial, "latest_data", None)
        if data is None:
            return False
        ts = float(getattr(data, "timestamp", 0.0) or 0.0)
        now = time.monotonic() if now_s is None else float(now_s)
        return ts > 0.0 and (now - ts) <= self._stale_timeout_s


class RgbdOdometryBackend:
    """Reserved hook for a future RGB-D visual odometry implementation."""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "RGB-D odometry is reserved for a later hardware-stability phase"
        )


def compute_return_command(
    pose: MotionPose,
    params: MotionReturnParams,
) -> tuple[float, float, bool]:
    """Return (vx_mps, vw_rps, done) to drive relative pose back to origin."""
    x = float(pose.x_m)
    y = float(pose.y_m)
    yaw = _wrap_pi(float(pose.yaw_rad))
    dist = math.hypot(x, y)
    yaw_ok = abs(yaw) <= params.yaw_tolerance_rad
    pos_ok = dist <= params.position_tolerance_m
    if pos_ok and yaw_ok:
        return 0.0, 0.0, True
    settle_position_m = max(
        params.position_tolerance_m,
        min(0.04, params.position_tolerance_m * 1.5),
    )
    settle_yaw_rad = max(
        params.yaw_tolerance_rad,
        min(math.radians(5.0), params.yaw_tolerance_rad * 1.7),
    )
    if dist <= settle_position_m and abs(yaw) <= settle_yaw_rad:
        return 0.0, 0.0, True

    vx = 0.0
    vw = 0.0
    if not pos_ok:
        close_range_m = max(0.12, 4.0 * params.position_tolerance_m)
        if dist <= close_range_m:
            # Close to the start pose, use body-frame error directly.  This
            # preserves the short-motion behavior verified on the car, while
            # blending final yaw back in to avoid rotating the wrong way after
            # crossing the start heading.
            dx = -x
            dy = -y
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            forward_err = dx * cy + dy * sy
            lateral_err = -dx * sy + dy * cy
            heading_err = math.atan2(lateral_err, max(abs(forward_err), 0.05))
            vx = params.kp_position * forward_err
            vx = max(-params.max_linear_speed, min(params.max_linear_speed, vx))
            if abs(lateral_err) > params.position_tolerance_m:
                vw = params.kp_yaw * (-yaw + 0.5 * heading_err)
        else:
            desired_heading = math.atan2(-y, -x)
            forward_heading_err = _wrap_pi(desired_heading - yaw)
            reverse_heading_err = _wrap_pi(desired_heading + math.pi - yaw)
            if abs(reverse_heading_err) < abs(forward_heading_err):
                drive_sign = -1.0
                heading_err = reverse_heading_err
            else:
                drive_sign = 1.0
                heading_err = forward_heading_err

            vw = params.kp_yaw * heading_err
            heading_gate_rad = math.radians(35.0)
            if abs(heading_err) <= heading_gate_rad:
                align = max(0.0, math.cos(heading_err))
                vx = drive_sign * params.kp_position * dist * align
                vx = max(-params.max_linear_speed, min(params.max_linear_speed, vx))
    if pos_ok and not yaw_ok:
        vw = -params.kp_yaw * yaw

    vw = max(-params.max_angular_speed, min(params.max_angular_speed, vw))
    return vx, vw, False
