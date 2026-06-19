#!/usr/bin/env python3
"""Read-only marker-board observer.

Starts the camera and ArUco tracker, fuses the 3x3 marker board into a
single target, and prints a compact status line. It never opens the serial
port and cannot command the chassis.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Iterable, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIENT_DIR = os.path.join(REPO_ROOT, "client")
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from marker_board import BoardTarget, MarkerBoardEstimator, MarkerBoardLayout


def format_observation(detections: Iterable[object], target: Optional[BoardTarget]) -> str:
    ids = sorted(int(getattr(d, "tag_id")) for d in detections)
    ids_text = ",".join(str(i) for i in ids) if ids else "-"
    prefix = f"tags={len(ids)} ids={ids_text}"
    if target is None:
        return f"{prefix} board LOST"
    angle_deg = math.degrees(target.angle_h)
    return (
        f"{prefix} board dist={target.distance:.2f}m "
        f"angle={angle_deg:.1f}deg held={target.held}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observe 3x3 marker-board pose without motor control")
    parser.add_argument("--camera", type=str, default="auto")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--tag-size", type=float, default=0.045)
    parser.add_argument("--marker-board-gap", type=float, default=0.007)
    parser.add_argument("--calib-file", type=str, default=None)
    parser.add_argument("--aruco-dict", type=str, default="5X5_250")
    parser.add_argument("--aruco-preset", type=str, default="permissive",
                        choices=["permissive", "default", "refine"])
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Print rate in Hz (default: 5)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Stop after N seconds; 0 means run until Ctrl+C")
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--lost-timeout", type=float, default=0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from camera import SharedCamera
    from apriltag_tracker import AprilTagTracker

    camera = SharedCamera(
        camera_id=args.camera,
        width=args.camera_width,
        height=args.camera_height,
    )
    tracker = AprilTagTracker(
        camera=camera,
        tag_size=args.tag_size,
        calib_file=args.calib_file,
        aruco_dict=args.aruco_dict,
        detector_preset=args.aruco_preset,
    )
    estimator = MarkerBoardEstimator(
        layout=MarkerBoardLayout(tag_size_m=args.tag_size, gap_m=args.marker_board_gap),
        smoothing_alpha=args.smoothing_alpha,
        lost_timeout_s=args.lost_timeout,
    )

    interval = 1.0 / max(0.1, args.fps)
    started = time.monotonic()
    camera.start()
    tracker.start()
    print("[observe] marker-board observer started (read-only, no serial)")
    try:
        while True:
            dets = tracker.detections
            target = estimator.estimate(dets)
            print(format_observation(dets, target), flush=True)
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        camera.stop()
        print("[observe] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
