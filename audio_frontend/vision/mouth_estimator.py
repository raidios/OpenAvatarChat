"""
Mouth-3D estimator: turn 17 COCO keypoints + aligned depth into a 3-D point
in the camera frame.

Per the plan, the ground-robot view often loses the face, so we have a
priority chain:

  1. Both nose AND ears visible (kp 0/3/4) >0.4 conf  -> nose pixel itself.
  2. Both shoulders (kp 5/6) >0.4 conf and at least one of mouth area
     keypoints (0/1/2/3/4) reasonably confident          -> shoulder midpoint
                                                           + anatomical offset.
  3. Only shoulders (kp 5/6)                           -> shoulder midpoint
                                                           + larger offset.
  4. Hips (kp 11/12) only                              -> upward extrapolation
                                                           assuming 1.6 m height.
  5. Bbox top edge                                     -> last resort.

For each candidate pixel, depth is sampled from the aligned depth map at a
5x5 median; if the median is invalid (zero) we expand to 11x11 then 21x21.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# COCO 17-keypoint layout
KP_NOSE = 0
KP_LEYE = 1
KP_REYE = 2
KP_LEAR = 3
KP_REAR = 4
KP_LSHOULDER = 5
KP_RSHOULDER = 6
KP_LELBOW = 7
KP_RELBOW = 8
KP_LWRIST = 9
KP_RWRIST = 10
KP_LHIP = 11
KP_RHIP = 12
KP_LKNEE = 13
KP_RKNEE = 14
KP_LANKLE = 15
KP_RANKLE = 16

DEFAULT_CONF_THRESH = 0.4
DEFAULT_SHOULDER_TO_MOUTH_M = 0.22
DEFAULT_BODY_HEIGHT_M = 1.65


@dataclass
class MouthEstimate:
    pixel_xy: Tuple[float, float]
    point_3d_m: Tuple[float, float, float]   # (X, Y, Z) in meters, camera frame
    rule: str                                 # one of nose/shoulder/hip/bbox
    valid_depth: bool
    confidence: float                         # 0..1 (1 best, 0 fallback)


def _median_depth_at(depth_mm: np.ndarray, x: float, y: float,
                     win_sizes=(2, 5, 10)) -> float:
    """Return median valid depth (mm) around (x, y); 0 if no valid sample."""
    H, W = depth_mm.shape
    xi = int(round(x))
    yi = int(round(y))
    for r in win_sizes:
        x0, y0 = max(0, xi - r), max(0, yi - r)
        x1, y1 = min(W, xi + r + 1), min(H, yi + r + 1)
        patch = depth_mm[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size >= 3:
            return float(np.median(valid))
    return 0.0


def _backproject(K: np.ndarray, x_px: float, y_px: float,
                 z_mm: float) -> Tuple[float, float, float]:
    """Pinhole back-projection. Returns (X, Y, Z) in meters."""
    if z_mm <= 0:
        return (0.0, 0.0, 0.0)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z_m = z_mm / 1000.0
    X = (x_px - cx) / fx * z_m
    Y = (y_px - cy) / fy * z_m
    return (float(X), float(Y), float(z_m))


def estimate_mouth_3d(
    keypoints: np.ndarray,            # (17, 3)
    bbox_xyxy: Tuple[float, float, float, float],
    depth_mm: np.ndarray,             # (H, W) float / int, in mm
    K_color: np.ndarray,              # (3, 3) intrinsics of the color camera
    conf_thresh: float = DEFAULT_CONF_THRESH,
    shoulder_to_mouth_offset_m: float = DEFAULT_SHOULDER_TO_MOUTH_M,
    body_height_m: float = DEFAULT_BODY_HEIGHT_M,
) -> MouthEstimate:
    kp = keypoints.astype(np.float32, copy=False)
    fy = K_color[1, 1]

    # ---- rule 1: face anchor (nose visible AND face cluster confident) ----
    if (kp[KP_NOSE, 2] >= conf_thresh and
            (kp[KP_LEAR, 2] >= conf_thresh or kp[KP_REAR, 2] >= conf_thresh)):
        x, y = float(kp[KP_NOSE, 0]), float(kp[KP_NOSE, 1])
        z_mm = _median_depth_at(depth_mm, x, y)
        # nose -> mouth: ~3 cm down in image plane @ ~1m -> tiny pixel shift
        # (we keep the nose as the anchor; downstream beamformer is OK with cm-level offset)
        return MouthEstimate(
            pixel_xy=(x, y),
            point_3d_m=_backproject(K_color, x, y, z_mm),
            rule="nose_face_visible",
            valid_depth=z_mm > 0,
            confidence=0.95 if z_mm > 0 else 0.6,
        )

    # ---- rule 2: shoulders + at least one face cue ----
    s_ok = (kp[KP_LSHOULDER, 2] >= conf_thresh and
            kp[KP_RSHOULDER, 2] >= conf_thresh)
    face_cue = any(kp[i, 2] >= 0.25
                   for i in (KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR))
    if s_ok and face_cue:
        sx = float((kp[KP_LSHOULDER, 0] + kp[KP_RSHOULDER, 0]) / 2.0)
        sy = float((kp[KP_LSHOULDER, 1] + kp[KP_RSHOULDER, 1]) / 2.0)
        z_mm = _median_depth_at(depth_mm, sx, sy)
        # shift up by anatomical offset (in meters) reprojected to pixels
        pixel_dy = (shoulder_to_mouth_offset_m * fy / max(z_mm / 1000.0, 0.3))
        my = sy - pixel_dy
        return MouthEstimate(
            pixel_xy=(sx, my),
            point_3d_m=_backproject(K_color, sx, my, z_mm),
            rule="shoulders_with_face_cue",
            valid_depth=z_mm > 0,
            confidence=0.80 if z_mm > 0 else 0.5,
        )

    # ---- rule 3: shoulders only, no face ----
    if s_ok:
        sx = float((kp[KP_LSHOULDER, 0] + kp[KP_RSHOULDER, 0]) / 2.0)
        sy = float((kp[KP_LSHOULDER, 1] + kp[KP_RSHOULDER, 1]) / 2.0)
        z_mm = _median_depth_at(depth_mm, sx, sy)
        # offset same as rule 2 but mark lower confidence
        pixel_dy = (shoulder_to_mouth_offset_m * fy / max(z_mm / 1000.0, 0.3))
        my = sy - pixel_dy
        return MouthEstimate(
            pixel_xy=(sx, my),
            point_3d_m=_backproject(K_color, sx, my, z_mm),
            rule="shoulders_only",
            valid_depth=z_mm > 0,
            confidence=0.65 if z_mm > 0 else 0.4,
        )

    # ---- rule 4: hips only -> assume 1.6m height -> mouth at ~1.5m above ground ----
    h_ok = (kp[KP_LHIP, 2] >= conf_thresh and
            kp[KP_RHIP, 2] >= conf_thresh)
    if h_ok:
        hx = float((kp[KP_LHIP, 0] + kp[KP_RHIP, 0]) / 2.0)
        hy = float((kp[KP_LHIP, 1] + kp[KP_RHIP, 1]) / 2.0)
        z_mm = _median_depth_at(depth_mm, hx, hy)
        # Hip is ~ 0.5 of height, mouth at ~ 0.93 of height -> 0.43*height above hip.
        delta_y_m = 0.43 * body_height_m
        pixel_dy = (delta_y_m * fy / max(z_mm / 1000.0, 0.3))
        my = hy - pixel_dy
        return MouthEstimate(
            pixel_xy=(hx, my),
            point_3d_m=_backproject(K_color, hx, my, z_mm),
            rule="hips_height_prior",
            valid_depth=z_mm > 0,
            confidence=0.45 if z_mm > 0 else 0.25,
        )

    # ---- rule 5: bbox top + 0.10m down inside box ----
    x0, y0, x1, y1 = bbox_xyxy
    tx = float((x0 + x1) / 2.0)
    ty = float(y0 + 0.10 * (y1 - y0))
    z_mm = _median_depth_at(depth_mm, tx, ty)
    return MouthEstimate(
        pixel_xy=(tx, ty),
        point_3d_m=_backproject(K_color, tx, ty, z_mm),
        rule="bbox_top",
        valid_depth=z_mm > 0,
        confidence=0.30 if z_mm > 0 else 0.15,
    )
