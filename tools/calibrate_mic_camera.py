#!/usr/bin/env python3
"""
Calibrate the rigid transform between the mic ring center frame and the
color-camera optical frame.

Why we need it:
  perception_node publishes owner-3D in `camera_color_optical_frame`. The
  MVDR steering vector is built in the mic-array frame. We need a static
  4x4 transform `T_mc` (mic <- camera) that maps a 3-D point from the camera
  frame into the mic-array frame, then take its azimuth/elevation.

Procedure (in-situ rigid mount -- the actual deployed scenario):
  Both the mic ring and the camera are bolted to the same chassis, so
  their relative pose is FIXED. Calibration here just measures that
  one fixed value, NOT a multi-pose joint estimation. We use multiple
  snapshots only to average out the ~0.3-1 px solvePnP noise.

  1) WITHOUT moving anything, stick a known-size aruco tag on the mic
     ring such that:
        - the tag's CENTER sits at the mic ring's geometric center
        - the tag's +X axis points along the ring's +X axis
          (= the channel-0 direction = body +30° on this robot)
     A printed DICT_4X4_50 marker, ~100 mm side, glued flat on the
     ring's top face works. Note the `marker_size_m` and `marker_id`.
  2) Run this tool. Camera and mic stay bolted to the chassis; you do
     NOT walk the camera around the marker. Just keep the tag in
     view (which is the camera's only normal viewing angle of it
     anyway) and press SPACE 8 times. The tool averages out the
     per-frame solvePnP noise.
     If `--no-display` is set the tool auto-snaps once per second
     until `--min-snaps` is hit.
  3) Output: `tests/data/extrinsics/mic_camera_extrinsic.npz`
        R   (3,3)   rotation camera_optical -> mic
        T   (3,)    translation in METERS
        rms_px      per-snapshot reprojection RMS (px)

Sanity check after calibration:
  Apply `R` to the body-forward vector (1, 0, 0) (post-axis-conversion
  to camera optical convention: +z_cam = forward, +x_cam = right,
  +y_cam = down). The result's azimuth in the mic frame should be
  approximately `mic_yaw_offset_deg` (= +30° on this robot, calibrated
  separately via the DOA dataset). If it's not within ~5° of +30°,
  the marker is not aligned to ch0; redo the sticker.

Usage:
  .venv/bin/python tools/calibrate_mic_camera.py \\
      --marker-size 0.10 --marker-id 0 \\
      --out tests/data/extrinsics/mic_camera_extrinsic.npz

  Headless / scripted (auto-snaps every second, stops at min-snaps):
  .venv/bin/python tools/calibrate_mic_camera.py \\
      --marker-size 0.10 --marker-id 0 \\
      --no-display --min-snaps 8

If you'd rather skip vision entirely: measure (x, y, z) of the camera
optical center and the mic ring center on the chassis with a ruler,
plug them in along with mic_yaw_offset_deg=+30°, and write the same
npz format manually. Precision is ~2 cm / 1° versus ~0.5 cm / 0.5°
for the vision route, which is plenty for far-field MVDR steering.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "thirdparty" / "OrbbecSDK_v1" / "python"))

try:
    import cv2
    from cv2 import aruco
except ImportError as e:
    print("[!] opencv-python with the aruco module is required", file=sys.stderr)
    raise SystemExit(2) from e

from orbbec_gemini import OrbbecGemini, OrbbecGeminiError  # noqa: E402


def _build_K_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-size", type=float, default=0.10,
                    help="aruco tag side length in meters (default 0.10)")
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    ap.add_argument("--hfov-deg", type=float, default=85.0)
    ap.add_argument("--out", type=str,
                    default="tests/data/extrinsics/mic_camera_extrinsic.npz")
    ap.add_argument("--min-snaps", type=int, default=8)
    ap.add_argument("--no-display", action="store_true",
                    help="run headless; auto-snap once per second")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = OrbbecGemini(streams=["depth", "color"])
    info = cam.open(color_width=640, color_height=480, color_fps=30,
                     color_format="mjpg", d2c="hw")
    Wc, Hc = info.color_width, info.color_height
    K = _build_K_from_fov(Wc, Hc, args.hfov_deg)
    D = np.zeros(5, dtype=np.float64)

    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, args.dictionary))
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)

    marker_size = float(args.marker_size)
    half = marker_size / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)

    rs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    rmss: List[float] = []

    print(f"[*] camera open at {Wc}x{Hc}, marker_id={args.marker_id}, "
          f"size={marker_size}m. SPACE = snap, ESC = finish")
    last_snap = 0.0
    try:
        while True:
            rgbd = cam.read_rgbd_sdk(timeout_ms=200)
            if rgbd is None:
                continue
            bgr = rgbd.color.data
            corners, ids, _ = detector.detectMarkers(bgr)
            ok = ids is not None and args.marker_id in ids.flatten()

            display = bgr.copy()
            if ok:
                aruco.drawDetectedMarkers(display, corners, ids)
                idx = int(np.where(ids.flatten() == args.marker_id)[0][0])
                img_pts = corners[idx][0].astype(np.float64)
                rc, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if rc:
                    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
                    rms = float(np.sqrt(((proj.reshape(-1, 2) - img_pts) ** 2).sum(1).mean()))
                    cv2.putText(display, f"rms={rms:.2f} px snaps={len(rs)}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    auto_snap = (args.no_display
                                 and time.monotonic() - last_snap > 1.0
                                 and rms < 1.5)
                    snap_now = False
                    if not args.no_display:
                        cv2.imshow("calibrate_mic_camera", display)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:    # ESC
                            break
                        if key == ord(' '):
                            snap_now = True
                    if auto_snap:
                        snap_now = True
                    if snap_now:
                        R, _ = cv2.Rodrigues(rvec)
                        rs.append(R)
                        ts.append(tvec.flatten())
                        rmss.append(rms)
                        print(f"  snap #{len(rs)} t={tvec.flatten()} rms={rms:.2f}")
                        last_snap = time.monotonic()
            elif not args.no_display:
                cv2.imshow("calibrate_mic_camera", display)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if args.no_display and len(rs) >= args.min_snaps:
                break
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        cam.close()

    if len(rs) < args.min_snaps:
        print(f"[!] only collected {len(rs)} snapshots; need >= {args.min_snaps}",
              file=sys.stderr)
        return 1

    # Average rotations using quaternion mean (small-angle Karcher mean is overkill
    # for a static rig). Use SVD orthogonal Procrustes.
    R_avg = np.mean(np.array(rs), axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        Vt[2, :] *= -1
        R_avg = U @ Vt
    T_avg = np.median(np.array(ts), axis=0)

    np.savez(
        str(out_path), R=R_avg, T=T_avg,
        rms_px=np.array(rmss), num_snaps=int(len(rs)),
        marker_size_m=marker_size, marker_id=int(args.marker_id),
    )
    print(f"[+] wrote {out_path}")
    print(f"    snapshots used : {len(rs)}")
    print(f"    median rms     : {np.median(rmss):.2f} px")
    print(f"    max rms        : {np.max(rmss):.2f} px")
    print(f"    T (m)          : [{T_avg[0]:+.3f}, {T_avg[1]:+.3f}, {T_avg[2]:+.3f}]")
    print(f"    R (rad, rpy)   : "
          f"{cv2.Rodrigues(R_avg)[0].flatten().tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
