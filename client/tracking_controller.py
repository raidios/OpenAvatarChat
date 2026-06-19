"""
AprilTag tracking state machine and motion controller.

States:
    IDLE       -- waiting for a tag to appear; motors stopped
    TRACKING   -- tag visible; PD-controlling heading and distance
"""

import enum
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional

from apriltag_tracker import AprilTagTracker
from marker_board import MarkerBoardEstimator, MarkerBoardLayout
from serial_comm import XProtocolSerial


class TrackingState(enum.Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"


@dataclass
class TrackingParams:
    tracking_distance: float = 0.6      # desired follow distance (m)
    max_linear_speed: float = 0.30      # m/s
    max_angular_speed: float = 1.2      # rad/s
    kp_angle: float = 2.5              # P gain for heading alignment
    kd_angle: float = 0.5             # D gain for heading alignment
    kp_distance: float = 0.5          # P gain for distance
    angle_deadzone: float = 0.02       # rad – ignore small angle errors
    distance_deadzone: float = 0.05    # m – ignore small distance errors
    control_rate: float = 20.0         # Hz
    board_tag_size: float = 0.045      # m, edge length of each board marker
    board_gap: float = 0.007           # m, white gap between 3x3 board markers
    board_lost_timeout: float = 0.8    # s, tolerate short detector dropouts
    board_smoothing_alpha: float = 0.35
    board_min_visible_tags: int = 3
    min_valid_distance: float = 0.15    # m
    max_valid_distance: float = 2.0     # m


class TrackingController:
    """Runs the tracking state machine in a background thread."""

    def __init__(
        self,
        tracker: AprilTagTracker,
        serial: XProtocolSerial,
        params: Optional[TrackingParams] = None,
        target_tag_id: Optional[int] = None,
    ):
        self._tracker = tracker
        self._serial = serial
        self._params = params or TrackingParams()
        self._target_tag_id = target_tag_id
        self._board_estimator: Optional[MarkerBoardEstimator] = None
        if self._target_tag_id is None:
            self._board_estimator = MarkerBoardEstimator(
                layout=MarkerBoardLayout(
                    tag_size_m=self._params.board_tag_size,
                    gap_m=self._params.board_gap,
                ),
                smoothing_alpha=self._params.board_smoothing_alpha,
                lost_timeout_s=self._params.board_lost_timeout,
                min_visible_tags=self._params.board_min_visible_tags,
            )

        self._state = TrackingState.IDLE
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._last_angle_error: float = 0.0
        self._last_control_time: float = 0.0
        self._last_angle_h: float = 0.0
        self._estop: bool = False
        # 当 MotionPlayer 回放预录 clip 时调用 pause()：跟随循环会进入 IDLE
        # 并停止下发 cmd_vel，让出控制权给 clip。clip 结束 resume() 后恢复。
        # 与 _estop 区分：pause 不是紧急停止，仅是临时让步，对外行为是"跟丢"
        # 与"用户中断"的中间态。
        self._paused: bool = False

    @property
    def state(self) -> TrackingState:
        with self._state_lock:
            return self._state

    @property
    def is_estopped(self) -> bool:
        with self._state_lock:
            return self._estop

    @property
    def is_paused(self) -> bool:
        with self._state_lock:
            return self._paused

    def pause(self):
        """暂停跟随，让 MotionPlayer 接管 cmd_vel。idempotent。"""
        with self._state_lock:
            if self._paused:
                return
            self._paused = True
            # 退到 IDLE 是为了在 resume 后从干净状态重启 PD 控制器。
            self._state = TrackingState.IDLE
        # 停一次以防上一拍刚发出的 vel 还在 300ms freshness 窗里。
        self._send_stop()
        print("[tracking] paused (motion clip in progress)")

    def resume_from_pause(self):
        """对应 pause()。命名避免和已有的 ``resume()`` 紧急停止恢复混淆。"""
        with self._state_lock:
            if not self._paused:
                return
            self._paused = False
        print("[tracking] resumed from pause")

    def emergency_stop(self):
        """Immediately stop all motors and enter IDLE. Tracking is disabled
        until ``resume()`` is called."""
        with self._state_lock:
            self._estop = True
            self._state = TrackingState.IDLE
        self._send_stop()
        print("[tracking] *** EMERGENCY STOP ***")

    def resume(self):
        """Re-enable tracking after an emergency stop."""
        with self._state_lock:
            if not self._estop:
                return
            self._estop = False
        print("[tracking] Resumed from emergency stop")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        print("[tracking] Controller started")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._send_stop()
        print("[tracking] Controller stopped")

    # -- state machine -------------------------------------------------------

    def _control_loop(self):
        dt = 1.0 / self._params.control_rate
        while self._running:
            t0 = time.monotonic()
            self._step()
            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _step(self):
        if self._estop:
            self._send_stop()
            return
        if self._paused:
            # 不下发 cmd_vel，让 MotionPlayer 独占 PI 通道。
            return

        dets = self._tracker.detections
        target = None
        if self._board_estimator is not None:
            target = self._board_estimator.estimate(dets)
        else:
            for d in dets:
                if self._target_tag_id is not None and d.tag_id != self._target_tag_id:
                    continue
                if d.distance is not None:
                    if target is None or d.distance < target.distance:
                        target = d

        tag_visible = target is not None

        with self._state_lock:
            state = self._state

        if state == TrackingState.IDLE:
            self._handle_idle(tag_visible, target)
        elif state == TrackingState.TRACKING:
            self._handle_tracking(tag_visible, target)

    def _set_state(self, new_state: TrackingState):
        with self._state_lock:
            old = self._state
            self._state = new_state
        if old != new_state:
            print(f"[tracking] {old.value} -> {new_state.value}")

    # -- IDLE ----------------------------------------------------------------

    def _handle_idle(self, tag_visible, target):
        if tag_visible:
            # CRITICAL: do NOT fire _handle_tracking() on the same tick we
            # transition into TRACKING.
            #
            # Two reasons:
            #   1. A single false-positive detection (tag visible for one
            #      frame, gone the next) would otherwise inject a one-shot
            #      cmd_vel pulse. The MCU's PI-source freshness window is
            #      300 ms, which is more than enough for the wheels to lurch
            #      noticeably (~10° of yaw at max_angular_speed).
            #   2. Even if the tag stays visible, calling _handle_tracking()
            #      with last_control_time == now makes dt ≈ 0 in the P-D
            #      controller. d_angle = (angle_err - 0) / dt then saturates
            #      to ±max_angular_speed, regardless of how mild the actual
            #      angular error is.
            #
            # Instead, seed last_angle_error from the current observation so
            # the *next* control tick (50 ms later, with finite dt) starts
            # with d_angle ≈ 0, and only acts if the tag is still there.
            self._last_angle_error = -target.angle_h
            self._last_control_time = time.monotonic()
            self._set_state(TrackingState.TRACKING)

    # -- TRACKING ------------------------------------------------------------

    def _handle_tracking(self, tag_visible, target):
        if not tag_visible:
            self._send_stop()
            angle_deg = math.degrees(self._last_angle_h)
            direction = "right" if self._last_angle_h > 0 else "left"
            print(f"[tracking] Tag lost (last seen {angle_deg:.1f}° to the {direction})")
            self._set_state(TrackingState.IDLE)
            return
        if getattr(target, "held", False):
            self._send_stop()
            return

        p = self._params
        angle_h = target.angle_h
        dist = target.distance
        if not self._target_is_valid(angle_h, dist):
            self._send_stop()
            print("[tracking] Invalid target pose; stopping")
            self._set_state(TrackingState.IDLE)
            return

        self._last_angle_h = angle_h

        now = time.monotonic()
        dt = now - self._last_control_time if self._last_control_time else 0.02
        self._last_control_time = now

        # angular control (P-D)
        # MCU convention: positive vw = turn left; tag right → angle_h > 0 → need vw < 0
        angle_err = -angle_h
        d_angle = (angle_err - self._last_angle_error) / dt if dt > 0 else 0.0
        self._last_angle_error = angle_err

        if abs(angle_err) < p.angle_deadzone:
            vw = 0.0
        else:
            vw = p.kp_angle * angle_err + p.kd_angle * d_angle
            vw = max(-p.max_angular_speed, min(p.max_angular_speed, vw))

        # distance control (P)
        dist_err = dist - p.tracking_distance
        if abs(dist_err) < p.distance_deadzone:
            vx = 0.0
        else:
            vx = p.kp_distance * dist_err
            vx = max(-p.max_linear_speed, min(p.max_linear_speed, vx))

        # reduce forward speed when turning sharply
        if abs(angle_err) > 0.3:
            vx *= 0.5

        self._send_velocity(vx, vw)

    # -- motor helpers -------------------------------------------------------

    def _send_velocity(self, vx: float, vw: float):
        """Send velocity (m/s, rad/s) -> MCU units (m/s*1000)."""
        if not (math.isfinite(vx) and math.isfinite(vw)):
            self._send_stop()
            return
        p = self._params
        vx = max(-p.max_linear_speed, min(p.max_linear_speed, vx))
        vw = max(-p.max_angular_speed, min(p.max_angular_speed, vw))
        vx_mm = int(vx * 1000)
        vw_mm = int(vw * 1000)
        self._serial.send_velocity(vx_mm, 0, vw_mm)

    def _send_stop(self):
        self._serial.send_velocity(0, 0, 0)

    def _target_is_valid(self, angle_h: float, dist: float) -> bool:
        p = self._params
        return (
            math.isfinite(angle_h)
            and math.isfinite(dist)
            and p.min_valid_distance <= dist <= p.max_valid_distance
        )
