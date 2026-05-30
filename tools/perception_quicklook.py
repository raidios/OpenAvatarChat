#!/usr/bin/env python3
"""
Quick-look perception pipeline tester (no ROS2 needed).

Opens the Orbbec Gemini Pro, runs YOLO-pose + tracker + mouth estimator,
and prints per-track summary every second so you can sanity-check that:
  - depth + color stream OK
  - pose detector finds people
  - mouth-3D estimate is reasonable

Usage:
  .venv/bin/python tools/perception_quicklook.py --duration 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thirdparty" / "OrbbecSDK_v1" / "python"))

from orbbec_gemini import OrbbecGemini, OrbbecGeminiError  # noqa: E402

from audio_frontend.backends.pose_detector import select_pose_detector  # noqa: E402
from audio_frontend.vision.mouth_estimator import estimate_mouth_3d     # noqa: E402
from audio_frontend.vision.tracker import IouTracker                    # noqa: E402


def _build_K_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--hfov-deg", type=float, default=85.0)
    ap.add_argument("--max-fps", type=float, default=5.0)
    ap.add_argument("--save", type=str, default="",
                    help="optional path to save BGR overlay frames as MJPEG")
    args = ap.parse_args()

    cam = OrbbecGemini(streams=["depth", "color"])
    info = cam.open(color_width=640, color_height=480, color_fps=30,
                     color_format="mjpg", d2c="hw")
    Wc, Hc = info.color_width, info.color_height
    K = _build_K_from_fov(Wc, Hc, args.hfov_deg)
    pose = select_pose_detector(prefer="auto")
    tracker = IouTracker(iou_thresh=0.30, max_miss=15)
    print(f"[*] camera open at {Wc}x{Hc} d2c={info.d2c_mode} pose={pose.name}")

    period = 1.0 / max(args.max_fps, 0.1)
    last = 0.0
    t_end = time.monotonic() + args.duration
    n_frames = 0
    n_persons = 0
    last_log = time.monotonic()

    writer = None
    if args.save:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.save, fourcc, args.max_fps, (Wc, Hc))
        except ImportError:
            print("[!] save requested but opencv not available")

    try:
        while time.monotonic() < t_end:
            now = time.monotonic()
            if now - last < period:
                time.sleep(0.005)
                continue
            last = now
            try:
                rgbd = cam.read_rgbd_sdk(timeout_ms=300)
            except OrbbecGeminiError as e:
                print(f"[!] {e}")
                continue
            if rgbd is None:
                continue
            bgr = rgbd.color.data.copy()
            depth_mm = rgbd.aligned_depth.astype(np.float32) * rgbd.scale_mm_per_unit
            persons = pose.detect(bgr)
            n_frames += 1
            n_persons += len(persons)

            dets = np.array([p.bbox_xyxy for p in persons], dtype=np.float32) \
                if persons else np.zeros((0, 4), dtype=np.float32)
            scores = np.array([p.confidence for p in persons], dtype=np.float32) \
                if persons else np.zeros(0, dtype=np.float32)
            kps = np.stack([p.keypoints for p in persons]) if persons else None
            tracks = tracker.update(dets, scores, kps)

            if writer is not None or now - last_log > 1.0:
                line = f"frame={n_frames} tracks={len(tracks)}: "
                for t in tracks:
                    if t.keypoints is not None and t.miss == 0:
                        est = estimate_mouth_3d(t.keypoints, tuple(t.bbox),
                                                depth_mm, K)
                        line += (f"id={t.track_id} {est.rule} "
                                 f"({est.point_3d_m[0]:+.2f}, {est.point_3d_m[1]:+.2f}, "
                                 f"{est.point_3d_m[2]:+.2f}) m  ")
                if now - last_log > 1.0:
                    print(line)
                    last_log = now
                if writer is not None:
                    import cv2
                    for t in tracks:
                        if t.miss > 0: continue
                        x0, y0, x1, y1 = (int(t.bbox[0]), int(t.bbox[1]),
                                            int(t.bbox[2]), int(t.bbox[3]))
                        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(bgr, f"id={t.track_id}", (x0, y0 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        if t.keypoints is not None:
                            est = estimate_mouth_3d(t.keypoints, tuple(t.bbox),
                                                     depth_mm, K)
                            mx, my = int(est.pixel_xy[0]), int(est.pixel_xy[1])
                            cv2.circle(bgr, (mx, my), 6, (0, 0, 255), -1)
                            cv2.putText(bgr, f"{est.point_3d_m[2]:.1f}m",
                                        (mx + 8, my), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 1)
                    writer.write(bgr)
    finally:
        cam.close()
        if writer is not None:
            writer.release()
            print(f"[+] saved overlay to {args.save}")

    print(f"\n[+] processed {n_frames} frames, avg persons/frame "
          f"{n_persons / max(n_frames, 1):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
