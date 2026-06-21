"""RGB-D speaker binding and person-follow control.

This module keeps person following separate from ArUco marker following while
reusing the same conservative ``TrackingParams`` limits.  The perception side
binds one owner after wake-orient completes; the controller only drives after
that bind succeeds.
"""
from __future__ import annotations

import enum
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tracking_controller import TrackingParams

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = logging.getLogger(__name__)


@dataclass
class PersonTarget:
    track_id: int
    angle_h: float
    distance: float
    frame_id: int
    timestamp: float
    confidence: float = 0.0
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    valid_depth: bool = True
    held: bool = False
    predicted: bool = False
    rule: str = ""


@dataclass
class PersonPerceptionParams:
    max_fps: float = 5.0
    hfov_deg: float = 85.0
    bind_max_age_s: float = 0.8
    min_bind_distance: float = 0.45
    max_bind_distance: float = 2.0
    front_preference_deg: float = 35.0


class PersonPerception:
    """Pose + RGB-D depth pipeline for wake-time speaker binding."""

    def __init__(
        self,
        camera,
        params: Optional[PersonPerceptionParams] = None,
        pose_detector=None,
        calib_file: Optional[str] = "config/camera_calib.json",
        start_thread: bool = True,
    ):
        self._camera = camera
        self._params = params or PersonPerceptionParams()
        self._pose = pose_detector
        self._calib_file = calib_file
        self._tracker = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_targets: List[PersonTarget] = []
        self._latest_frame_id: int = 0
        self._latest_frame_time: float = 0.0
        self._last_bind_stats: dict = {}
        self.owner_track_id: Optional[int] = None
        self.disabled_reason: Optional[str] = None
        if start_thread:
            self.start()

    def start(self) -> None:
        if self._running:
            return
        self._ensure_backends()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="person_perception", daemon=True,
        )
        self._thread.start()
        pose_name = getattr(self._pose, "name", "unknown")
        print(f"[person] perception started pose={pose_name}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[person] perception stopped")

    def _ensure_backends(self) -> None:
        if self._pose is None:
            from audio_frontend.backends.pose_detector import select_pose_detector
            self._pose = select_pose_detector(prefer="auto")
        if self._tracker is None:
            from audio_frontend.vision.tracker import IouTracker
            self._tracker = IouTracker(iou_thresh=0.30, max_miss=15)

    def _loop(self) -> None:
        period = 1.0 / max(0.1, self._params.max_fps)
        while self._running:
            t0 = time.monotonic()
            try:
                self.process_once()
            except Exception:
                logger.exception("person perception step failed")
            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def get_owner_target(self) -> Optional[PersonTarget]:
        with self._lock:
            if self.owner_track_id is None:
                return None
            for target in self._latest_targets:
                if target.track_id == self.owner_track_id:
                    return target
        return None

    def reacquire_owner(self, max_age_s: Optional[float] = None) -> Optional[PersonTarget]:
        """Recover owner id after a short tracker-id change in single-person scenes."""
        with self._lock:
            candidates = self._valid_bind_candidates_locked(max_age_s=max_age_s)
            if len(candidates) != 1:
                return None
            target = candidates[0]
            old_id = self.owner_track_id
            self.owner_track_id = int(target.track_id)
        print(f"[person] owner reacquired track={target.track_id} old_track={old_id}")
        return target

    def has_owner(self) -> bool:
        with self._lock:
            return self.owner_track_id is not None

    def bind_owner(self, wake_doa_body_deg: Optional[float] = None) -> Optional[PersonTarget]:
        """Bind the current front/center safe target as the speaker owner."""
        del wake_doa_body_deg  # After wake-orient, visual binding is front-biased.
        deadline = time.monotonic() + max(0.2, self._params.bind_max_age_s)
        candidates: List[PersonTarget] = []
        while True:
            self.process_once()
            with self._lock:
                candidates = self._valid_bind_candidates_locked()
            if candidates or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        with self._lock:
            candidates = self._valid_bind_candidates_locked()
        if not candidates:
            if self.disabled_reason is None:
                self.disabled_reason = "no_bind_candidate"
            print(
                f"[person] bind failed reason={self.disabled_reason} "
                f"stats={self._last_bind_stats}"
            )
            return None

        def score(t: PersonTarget) -> float:
            angle_deg = abs(math.degrees(t.angle_h))
            front_penalty = angle_deg / max(1.0, self._params.front_preference_deg)
            center_penalty = abs(t.pixel_x - 0.5)
            return front_penalty + center_penalty - 0.15 * float(t.confidence)

        best = min(candidates, key=score)
        with self._lock:
            self.owner_track_id = int(best.track_id)
        self.disabled_reason = None
        print(
            f"[person] owner bound track={best.track_id} "
            f"dist={best.distance:.2f}m angle={math.degrees(best.angle_h):+.1f}°"
        )
        return best

    def _valid_bind_candidates_locked(self, max_age_s: Optional[float] = None) -> List[PersonTarget]:
        now = time.monotonic()
        age_s = self._params.bind_max_age_s if max_age_s is None else float(max_age_s)
        return [
            t for t in self._latest_targets
            if (
                t.valid_depth
                and now - t.timestamp <= age_s
                and self._params.min_bind_distance <= t.distance <= self._params.max_bind_distance
            )
        ]

    def process_once(self) -> List[PersonTarget]:
        self._ensure_backends()
        if self._camera is None or not hasattr(self._camera, "get_rgbd_frame"):
            self._set_disabled("rgbd_unavailable")
            return []
        color, depth_mm, frame_id, frame_time = self._camera.get_rgbd_frame()
        if color is None or depth_mm is None:
            self._set_disabled("rgbd_unavailable")
            return []
        self.disabled_reason = None

        persons = self._pose.detect(color)
        dets = (
            np.array([p.bbox_xyxy for p in persons], dtype=np.float32)
            if persons else np.zeros((0, 4), dtype=np.float32)
        )
        scores = (
            np.array([p.confidence for p in persons], dtype=np.float32)
            if persons else np.zeros(0, dtype=np.float32)
        )
        keypoints = np.stack([p.keypoints for p in persons]) if persons else None
        tracks = self._tracker.update(dets, scores, keypoints)

        K = self._camera_matrix(color.shape[1], color.shape[0])
        targets: List[PersonTarget] = []
        from audio_frontend.vision.mouth_estimator import estimate_mouth_3d
        for tr in tracks:
            if getattr(tr, "miss", 0) > 0 or tr.keypoints is None:
                continue
            est = estimate_mouth_3d(
                tr.keypoints,
                tuple(float(v) for v in tr.bbox),
                np.asarray(depth_mm),
                K,
            )
            x_m, _y_m, z_m = est.point_3d_m
            distance = float(z_m)
            valid = bool(est.valid_depth and math.isfinite(distance) and distance > 0)
            if valid:
                angle_h = math.atan2(float(x_m), distance)
            else:
                angle_h = 0.0
            targets.append(PersonTarget(
                track_id=int(tr.track_id),
                angle_h=angle_h,
                distance=distance,
                frame_id=int(frame_id),
                timestamp=float(frame_time or time.monotonic()),
                confidence=float(getattr(tr, "score", 0.0)),
                pixel_x=float(est.pixel_xy[0]) / max(1.0, color.shape[1] - 1),
                pixel_y=float(est.pixel_xy[1]) / max(1.0, color.shape[0] - 1),
                valid_depth=valid,
                rule=est.rule,
            ))
        with self._lock:
            self._latest_targets = targets
            self._latest_frame_id = int(frame_id)
            self._latest_frame_time = float(frame_time or time.monotonic())
            valid_targets = sum(1 for t in targets if t.valid_depth)
            self._last_bind_stats = {
                "frame_id": int(frame_id),
                "persons": int(len(persons)),
                "tracks": int(len(tracks)),
                "targets": int(len(targets)),
                "valid_depth": int(valid_targets),
            }
            if self.owner_track_id is not None and all(
                t.track_id != self.owner_track_id for t in targets
            ):
                # Keep the owner id; the follow controller handles lost timeout.
                pass
        return targets

    def _set_disabled(self, reason: str) -> None:
        self.disabled_reason = reason
        with self._lock:
            self._latest_targets = []
        print(f"[person] person_tracking_disabled reason={reason}")

    def _camera_matrix(self, width: int, height: int) -> np.ndarray:
        data = self._load_calib()
        if data is not None:
            fx = float(data["fx"])
            fy = float(data["fy"])
            cx = float(data["cx"])
            cy = float(data["cy"])
            res = data.get("resolution")
            if isinstance(res, list) and len(res) == 2 and res[0] and res[1]:
                sx = float(width) / float(res[0])
                sy = float(height) / float(res[1])
                fx, cx = fx * sx, cx * sx
                fy, cy = fy * sy, cy * sy
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        fx = (width / 2.0) / math.tan(math.radians(self._params.hfov_deg) / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _load_calib(self) -> Optional[dict]:
        if not self._calib_file:
            return None
        try:
            with open(self._calib_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if all(k in data for k in ("fx", "fy", "cx", "cy")):
                return data
        except OSError:
            return None
        except Exception:
            logger.exception("failed to load camera calibration %s", self._calib_file)
        return None


class PersonFollowState(enum.Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"


class PersonFollowController:
    """Conservative differential-drive follow controller for one bound person."""

    def __init__(
        self,
        perception: PersonPerception,
        serial,
        params: Optional[TrackingParams] = None,
        lost_timeout_s: float = 0.8,
        perception_max_fps: float = 5.0,
    ):
        self._perception = perception
        self._serial = serial
        self._params = params or TrackingParams()
        self._state = PersonFollowState.IDLE
        self._state_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._estop = False
        self._paused = False
        self._owner_bound = False
        self._last_target_time = 0.0
        self._last_angle_error = 0.0
        self._last_control_time = 0.0
        self._last_angle_h = 0.0
        self._last_vw = 0.0
        self._lost_timeout_s = float(lost_timeout_s)
        self._perception_period_s = 1.0 / max(0.1, float(perception_max_fps))
        self._last_perception_time = 0.0

    @property
    def state(self) -> PersonFollowState:
        with self._state_lock:
            return self._state

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, name="person_follow", daemon=True,
        )
        self._thread.start()
        print("[person] follow controller started")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._send_stop()
        print("[person] follow controller stopped")

    def pause(self) -> None:
        with self._state_lock:
            self._paused = True
            self._state = PersonFollowState.IDLE
        self._send_stop()

    def resume_from_pause(self) -> None:
        with self._state_lock:
            self._paused = False

    def emergency_stop(self) -> None:
        self._estop = True
        with self._state_lock:
            self._state = PersonFollowState.IDLE
        self._send_stop()

    def reset_estop(self) -> None:
        self._estop = False

    def bind_after_wake(self, wake_doa_body_deg: Optional[float] = None) -> Optional[PersonTarget]:
        target = self._perception.bind_owner(wake_doa_body_deg)
        if target is None:
            self._owner_bound = False
            with self._state_lock:
                self._state = PersonFollowState.IDLE
            self._send_stop()
            return None
        self._owner_bound = True
        self._last_target_time = float(target.timestamp)
        self._last_angle_error = -float(target.angle_h)
        self._last_angle_h = float(target.angle_h)
        self._last_control_time = time.monotonic()
        self._last_perception_time = 0.0
        self._send_stop()
        return target

    def _control_loop(self) -> None:
        dt = 1.0 / max(1.0, self._params.control_rate)
        while self._running:
            t0 = time.monotonic()
            self._step()
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def _step(self) -> None:
        if self._estop:
            self._send_stop()
            return
        if self._paused or not self._owner_bound:
            return

        now = time.monotonic()
        if now - self._last_perception_time >= self._perception_period_s:
            try:
                self._perception.process_once()
                self._last_perception_time = now
            except Exception:
                logger.exception("person follow perception refresh failed")
        target = self._perception.get_owner_target()
        if target is None:
            target = self._perception.reacquire_owner(max_age_s=self._lost_timeout_s)
            if target is not None:
                self._last_target_time = float(target.timestamp or now)
                self._send_stop()
                return
            if self._last_target_time and now - self._last_target_time > self._lost_timeout_s:
                print("[person] owner lost -> IDLE")
                self._owner_bound = False
                with self._state_lock:
                    self._state = PersonFollowState.IDLE
                self._send_stop()
            return
        self._last_target_time = float(target.timestamp or now)

        with self._state_lock:
            state = self._state
        if state == PersonFollowState.IDLE:
            self._last_angle_error = -float(target.angle_h)
            self._last_control_time = now
            with self._state_lock:
                self._state = PersonFollowState.TRACKING
            self._send_stop()
            print("[person] IDLE -> TRACKING")
            return

        if not self._target_is_valid(target):
            print(f"[person] invalid owner target; holding stop reason={self._invalid_target_reason(target)}")
            self._send_stop()
            return

        self._drive_target(target)

    def _drive_target(self, target: PersonTarget) -> None:
        p = self._params
        now = time.monotonic()
        dt = now - self._last_control_time if self._last_control_time else 0.02
        dt = max(0.001, min(dt, 0.2))
        self._last_control_time = now

        angle_h = float(target.angle_h)
        self._last_angle_h = angle_h
        angle_err = -angle_h
        d_angle = (angle_err - self._last_angle_error) / dt if dt > 0 else 0.0
        self._last_angle_error = angle_err

        if abs(angle_err) < p.angle_deadzone:
            vw = 0.0
        else:
            vw = p.kp_angle * angle_err + p.kd_angle * d_angle
            vw = max(-p.max_angular_speed, min(p.max_angular_speed, vw))

        dist_err = float(target.distance) - p.tracking_distance
        if abs(dist_err) < p.distance_deadzone:
            vx = 0.0
        else:
            vx = p.kp_distance * dist_err
            vx = max(-p.max_linear_speed, min(p.max_linear_speed, vx))
        if abs(angle_err) > 0.3:
            vx *= 0.5
        if target.predicted:
            vx *= p.predicted_speed_scale
            vw *= p.predicted_speed_scale

        vw = self._limit_angular_step(vw, dt)
        self._send_velocity(vx, vw)

    def _target_is_valid(self, target: PersonTarget) -> bool:
        p = self._params
        return (
            bool(target.valid_depth)
            and math.isfinite(float(target.angle_h))
            and math.isfinite(float(target.distance))
            and p.min_valid_distance <= float(target.distance) <= p.max_valid_distance
        )

    def _invalid_target_reason(self, target: PersonTarget) -> str:
        p = self._params
        try:
            angle = float(target.angle_h)
            dist = float(target.distance)
        except Exception:
            return "bad_numeric"
        if not bool(target.valid_depth):
            return f"invalid_depth dist={dist:.2f}m"
        if not math.isfinite(angle):
            return "nonfinite_angle"
        if not math.isfinite(dist):
            return "nonfinite_distance"
        if dist < p.min_valid_distance:
            return f"too_close dist={dist:.2f}m min={p.min_valid_distance:.2f}m"
        if dist > p.max_valid_distance:
            return f"too_far dist={dist:.2f}m max={p.max_valid_distance:.2f}m"
        return f"unknown dist={dist:.2f}m angle={math.degrees(angle):+.1f}deg"

    def _send_velocity(self, vx: float, vw: float) -> None:
        if not (math.isfinite(vx) and math.isfinite(vw)):
            self._send_stop()
            return
        p = self._params
        vx = max(-p.max_linear_speed, min(p.max_linear_speed, vx))
        vw = max(-p.max_angular_speed, min(p.max_angular_speed, vw))
        self._serial.send_velocity(int(vx * 1000), 0, int(vw * 1000))

    def _send_stop(self) -> None:
        self._last_vw = 0.0
        self._serial.send_velocity(0, 0, 0)

    def _limit_angular_step(self, vw: float, dt: float) -> float:
        p = self._params
        max_delta = max(0.0, p.max_angular_accel) * max(0.0, dt)
        if max_delta <= 0:
            self._last_vw = vw
            return vw
        delta = max(-max_delta, min(max_delta, vw - self._last_vw))
        limited = self._last_vw + delta
        self._last_vw = limited
        return limited
