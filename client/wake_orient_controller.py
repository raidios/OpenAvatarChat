"""
Wake-driven chassis rotation controller.

Subscribes (via the ``on_wake`` entry point) to a wake event with a
body-frame DOA, then closes the loop on the chassis IMU yaw to bring
the wake direction in front of the vehicle.

Two conventions meet here and they have OPPOSITE sign:

* DSP body-frame DOA (``FarfieldAudioSource.latest_doa_deg``):
  empirically CW-positive viewed from above, i.e. ``+90°`` = source
  on the robot's RIGHT, ``-90°`` = on its LEFT, ``180°`` = behind.
  This convention was baked in by the user's CW-labeled DOA
  calibration recordings; the live-doa monitor reproduces it 1:1.

* IMU yaw / vw command (``XProtocolSerial.yaw_deg`` and
  ``send_velocity``): standard CCW-positive (right-hand rule with
  z up). ``+vw`` rotates the chassis CCW (the robot's LEFT) and
  ``yaw_deg`` increases on CCW rotation; verified by
  ``tools/probe_yaw_polarity.py`` on this robot.

So a DOA of ``+90°`` (source on the right) corresponds to a desired
IMU yaw delta of ``-90°`` (rotate CW to face right). The controller
negates the DOA at intake to convert frames::

    goal_imu_yaw = current_imu_yaw - wake_doa_body_deg

i.e. "rotate by ``-wake_doa_body_deg``" in IMU/vw frame, so that the
source at body ±X° before rotation lands at body 0° after rotation.

While rotating:
  * any ``TrackingController`` instance is ``pause()``-d so it stops
    issuing AprilTag PD velocity commands
  * we run our own PD on (goal - current) yaw error at 20 Hz, capped
    to ``max_angular_speed`` rad/s
  * on convergence (``angle_deadzone_deg`` for ``settle_ticks`` ticks
    in a row) we stop the wheels and call ``on_complete`` (typically
    "publish ROS wake event so perception_node binds owner")

Safety knobs:
  * ``max_duration_s``: hard timeout in case the IMU misbehaves or
    the chassis is wedged. Aborts and stops.
  * ``min_doa_deg``: don't bother rotating if the wake came from
    within this body angle of forward (the BF beam already covers
    it; rotating wastes time and shakes the camera).
  * ``cooldown_s``: ignore overlapping wake events while a rotation
    is in progress.

Threading model: ``on_wake`` is called from the audio thread, so it
must return fast. We just stash the goal and ping a background worker
thread which owns the control loop. The worker uses
``XProtocolSerial.send_velocity`` (already thread-safe per its lock).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from serial_comm import XProtocolSerial
    from tracking_controller import TrackingController

logger = logging.getLogger(__name__)

WakeCompleteCallback = Callable[[str, float, float], None]
# (keyword, requested_delta_deg, achieved_delta_deg)
WakeYawDeltaCallback = Callable[[float], None]
# incremental IMU yaw delta in degrees, CCW-positive


@dataclass
class WakeOrientParams:
    max_angular_speed: float = 2.5         # rad/s
    kp_angle: float = 2.5                  # P gain on yaw error (deg)
    kd_angle: float = 0.3                  # D gain
    angle_deadzone_deg: float = 5.0        # |err| at which we declare done
    settle_ticks: int = 4                  # consecutive ticks within deadzone
    control_rate_hz: float = 20.0
    max_duration_s: float = 6.0            # hard timeout per rotation
    min_doa_deg: float = 8.0               # don't rotate for tiny offsets
    cooldown_s: float = 1.5                # ignore retriggers during rotation
    doa_to_yaw_sign: float = -1.0          # -1 keeps historical CW->CCW flip
    overshoot_brake_dps: float = 1.0       # mrad/s mapped from m/s
    # ^ above we don't actually use this; provided for future tuning


class WakeOrientController:
    """Rotate the chassis to face the wake source, using IMU yaw feedback.

    All angles are in **degrees** and CCW-positive in the body frame.
    """

    def __init__(
        self,
        serial: "XProtocolSerial",
        params: Optional[WakeOrientParams] = None,
        tracking_ctl: Optional["TrackingController"] = None,
        on_complete: Optional[WakeCompleteCallback] = None,
        on_yaw_delta: Optional[WakeYawDeltaCallback] = None,
    ):
        self._serial = serial
        self._tracking_ctl = tracking_ctl
        self._params = params or WakeOrientParams()
        self._on_complete = on_complete
        self._on_yaw_delta = on_yaw_delta

        self._goal_lock = threading.Lock()
        self._pending_goal: Optional[tuple] = None  # (keyword, delta_deg)
        self._goal_event = threading.Event()
        self._busy = threading.Event()
        self._last_fire_t: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._estopped: bool = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, name="wake_orient",
            daemon=True,
        )
        self._thread.start()
        logger.info("WakeOrientController started")

    def stop(self) -> None:
        self._running = False
        self._goal_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Belt + suspenders: make sure motors are stopped on teardown.
        try:
            self._serial.send_velocity(0, 0, 0)
        except Exception:
            pass

    def emergency_stop(self) -> None:
        """Abort any in-progress rotation and refuse new ones."""
        self._estopped = True
        try:
            self._serial.send_velocity(0, 0, 0)
        except Exception:
            pass

    def reset_estop(self) -> None:
        self._estopped = False

    # ------------------------------------------------------------------
    # external trigger (called from the audio thread)
    # ------------------------------------------------------------------
    def on_wake(self, keyword: str,
                wake_doa_body_deg: Optional[float]) -> None:
        """Wake fired; queue a rotation goal.

        The audio thread calls this; we never block here. If a
        rotation is already in progress, drop the new event (the
        previous one will see-it-through). If DOA is None we still
        log the wake but don't rotate.
        """
        if self._estopped:
            logger.info("wake ignored: e-stopped")
            return
        if wake_doa_body_deg is None:
            logger.info("wake %r: DOA unavailable, no rotation", keyword)
            return
        now = time.monotonic()
        if self._busy.is_set() or \
                now - self._last_fire_t < self._params.cooldown_s:
            logger.info("wake %r ignored: rotation in progress / cooldown",
                        keyword)
            return
        # Convert DSP body-frame DOA into the IMU/vw delta convention, then
        # wrap to (-180, 180]. The historical car calibration uses -1 here
        # (CW-positive DOA to CCW-positive yaw). Keep it configurable because
        # mic/body sign conventions are easy to invert during hardware changes.
        body_doa = float(wake_doa_body_deg)
        delta = self._params.doa_to_yaw_sign * body_doa
        delta = ((delta + 180.0) % 360.0) - 180.0
        logger.info(
            "wake %r: dsp_body_doa=%+0.1f° sign=%+0.0f -> "
            "imu_yaw_delta=%+0.1f°",
            keyword, body_doa, self._params.doa_to_yaw_sign, delta,
        )
        if abs(delta) < self._params.min_doa_deg:
            logger.info("wake %r: doa %+0.1f° within deadzone, "
                        "not rotating", keyword, delta)
            if self._on_complete is not None:
                try:
                    self._on_complete(keyword, delta, 0.0)
                except Exception:
                    logger.exception("on_complete (deadzone) raised")
            return

        self._last_fire_t = now
        with self._goal_lock:
            self._pending_goal = (keyword, delta)
        self._goal_event.set()
        logger.info("wake %r queued: rotate by %+0.1f°", keyword, delta)

    # ------------------------------------------------------------------
    # worker loop
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while self._running:
            self._goal_event.wait(timeout=0.5)
            if not self._running:
                break
            self._goal_event.clear()
            with self._goal_lock:
                goal = self._pending_goal
                self._pending_goal = None
            if goal is None:
                continue
            keyword, delta = goal
            self._busy.set()
            try:
                achieved = self._execute_rotation(keyword, delta)
            finally:
                self._busy.clear()
            if self._on_complete is not None:
                try:
                    self._on_complete(keyword, delta, achieved)
                except Exception:
                    logger.exception("on_complete raised")

    def _execute_rotation(self, keyword: str, delta_deg: float) -> float:
        """Run the PD loop until convergence or timeout. Returns the
        achieved delta in degrees (signed).
        """
        # Pause the AprilTag tracking controller so it doesn't fight us.
        paused_tracker = False
        if self._tracking_ctl is not None:
            try:
                self._tracking_ctl.pause()
                paused_tracker = True
            except Exception:
                logger.exception("tracking pause failed; proceeding")

        start_yaw = self._serial.yaw_deg
        # Use the unwrapped accumulator for tracking large rotations
        # without wrap discontinuities tripping the deadzone check.
        start_yaw_unwrapped = self._serial.yaw_deg_unwrapped
        target_unwrapped = start_yaw_unwrapped + delta_deg
        notified_yaw_unwrapped = start_yaw_unwrapped

        def notify_yaw_delta(current_unwrapped: float, *,
                             force: bool = False) -> None:
            nonlocal notified_yaw_unwrapped
            cb = self._on_yaw_delta
            if cb is None:
                return
            inc = float(current_unwrapped - notified_yaw_unwrapped)
            if not force and abs(inc) < 0.5:
                return
            if abs(inc) < 0.05:
                return
            try:
                cb(inc)
                notified_yaw_unwrapped = float(current_unwrapped)
            except Exception:
                logger.exception("on_yaw_delta raised")

        p = self._params
        dt_target = 1.0 / p.control_rate_hz
        last_err = delta_deg
        last_t = time.monotonic()
        deadline = last_t + p.max_duration_s
        settled = 0
        achieved_deg = 0.0
        reason = "timeout"

        logger.info(
            "rotate start: yaw %+0.1f° -> %+0.1f° (delta %+0.1f°)",
            start_yaw, ((start_yaw + delta_deg + 180) % 360) - 180,
            delta_deg,
        )
        while self._running and not self._estopped:
            now = time.monotonic()
            if now >= deadline:
                break
            current_unwrapped = self._serial.yaw_deg_unwrapped
            err_deg = target_unwrapped - current_unwrapped
            achieved_deg = current_unwrapped - start_yaw_unwrapped
            notify_yaw_delta(current_unwrapped)

            if abs(err_deg) <= p.angle_deadzone_deg:
                settled += 1
                if settled >= p.settle_ticks:
                    reason = "converged"
                    break
            else:
                settled = 0

            dt = max(1e-3, now - last_t)
            d_err = (err_deg - last_err) / dt
            last_err = err_deg
            last_t = now

            # PD with degree-error → rad/s. The kp/kd here are in deg
            # space then converted; the historical TrackingController
            # uses radians, but we prefer degree-space control because
            # all our DOA / target inputs are in degrees.
            vw_dps = p.kp_angle * err_deg + p.kd_angle * d_err
            vw_rps = math.radians(vw_dps)
            vw_rps = max(-p.max_angular_speed,
                         min(p.max_angular_speed, vw_rps))
            self._send_velocity(0.0, vw_rps)

            sleep = dt_target - (time.monotonic() - now)
            if sleep > 0:
                time.sleep(sleep)

        # Stop wheels regardless of how we exited.
        self._send_velocity(0.0, 0.0)
        final_unwrapped = self._serial.yaw_deg_unwrapped
        achieved_deg = final_unwrapped - start_yaw_unwrapped
        notify_yaw_delta(final_unwrapped, force=True)

        end_yaw = self._serial.yaw_deg
        logger.info(
            "rotate %s: yaw %+0.1f° (achieved %+0.1f° / %+0.1f°), "
            "took %.2fs",
            reason, end_yaw, achieved_deg, delta_deg,
            time.monotonic() - (deadline - p.max_duration_s),
        )

        if paused_tracker and self._tracking_ctl is not None:
            try:
                self._tracking_ctl.resume_from_pause()
            except Exception:
                logger.exception("tracking resume failed")

        return achieved_deg

    # ------------------------------------------------------------------
    # motor helper
    # ------------------------------------------------------------------
    def _send_velocity(self, vx_mps: float, vw_rps: float) -> None:
        # Match TrackingController._send_velocity convention: m/s -> mm/s
        # and rad/s -> mrad/s, fed into XProtocolSerial.send_velocity
        # which the firmware decodes with 0x10 scale (vel_w mrad/s).
        try:
            self._serial.send_velocity(int(vx_mps * 1000), 0,
                                       int(vw_rps * 1000))
        except Exception:
            logger.exception("send_velocity failed")
