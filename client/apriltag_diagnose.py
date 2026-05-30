#!/usr/bin/env python3
"""
AprilTag (tag36h11) recognition diagnostics.

Checks the most common root causes of "hard to detect" when calibration RMS looks fine:
  - Capture resolution != calibration resolution (intrinsics wrong for this stream)
  - Tag too small / too blurry in the image (quad_decimate, focus, motion)
  - OpenCV DetectorParameters presets (permissive vs default vs subpixel refine)
  - Lighting / clipping (histogram)

Usage (on the Pi, from the client directory):

  uv run python apriltag_diagnose.py
  uv run python apriltag_diagnose.py --seconds 8 --save-dir /tmp/tagdiag
  uv run python apriltag_diagnose.py --image /path/to/snapshot.jpg --save-dir /tmp/tagdiag
  uv run python apriltag_diagnose.py --calib-file config/camera_calib.json --camera-width 640 --camera-height 480

Detection uses **OpenCV contrib** ``DICT_APRILTAG_36h11`` only (three parameter presets).
``pupil-apriltags`` is **not** used here: on some Raspberry Pi builds it **SIGSEGV (exit 139)**
even with ``estimate_tag_pose=False`` (crash inside ``apriltag_detector_detect``).

Interpretation (printed at the end as well):
  - If ONLY more-permissive OpenCV presets detect: tag is small / low contrast — move closer, improve print, or use ``main.py --aruco-preset permissive`` (default).
  - If NOTHING detects: wrong family, damaged print, extreme glare, or tag occupies very few pixels — take a still with --image and inspect saved PNG.
  - If calib resolution mismatch: recalibrate at the exact width/height the running app uses, or fix V4L size.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# OpenCV contrib required (same as client). pupil-apriltags omitted — decode path SIGSEGV on some ARM.


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _gray_stats(gray: np.ndarray) -> Tuple[float, float, float, float]:
    """mean, std, p5, p95 of uint8 gray"""
    flat = gray.reshape(-1)
    p5, p95 = np.percentile(flat, (5, 95))
    return float(flat.mean()), float(flat.std()), float(p5), float(p95)


# Three OpenCV ArUco parameter presets (same dictionary, different sensitivity).
OPENCV_SUITE: List[Tuple[str, str]] = [
    ("A_opencv permissive", "permissive"),
    ("B_opencv default", "default"),
    ("C_opencv refine_subpix", "refine"),
]


def _opencv_aruco_dict():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _opencv_detector_params(preset: str):
    p = cv2.aruco.DetectorParameters()
    if preset == "permissive":
        p.minMarkerPerimeterRate = 0.005
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 25
    elif preset == "refine":
        p.minMarkerPerimeterRate = 0.012
        try:
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except AttributeError:
            p.cornerRefinementMethod = 1
    # "default": leave OpenCV defaults
    return p


def _opencv_detect_ids(gray: np.ndarray, preset: str) -> Tuple[Optional[np.ndarray], Any]:
    """Returns (ids, corners) from OpenCV AprilTag 36h11."""
    aruco_dict = _opencv_aruco_dict()
    params = _opencv_detector_params(preset)
    try:
        det = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _rej = det.detectMarkers(gray)
    except AttributeError:
        corners, ids, _rej = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
    return ids, corners


def _opencv_apriltag36_report(gray: np.ndarray) -> str:
    try:
        ids, _ = _opencv_detect_ids(gray, "default")
    except Exception as e:
        return f"unavailable ({e})"
    if ids is None or len(ids) == 0:
        return "0 markers"
    idlist = sorted(int(i[0]) for i in ids)
    return f"{len(ids)} marker(s), ids={idlist}"


def _crosscheck_frames(gray: np.ndarray, label: str = "frame") -> None:
    print(f"\n[cross-check] {label} ({gray.shape[1]}x{gray.shape[0]})")
    print(f"  OpenCV DICT_APRILTAG_36h11 (default params):  {_opencv_apriltag36_report(gray)}")
    h, w = gray.shape[:2]
    up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    ids_up, _ = _opencv_detect_ids(up, "permissive")
    n_up = 0 if ids_up is None else len(ids_up)
    print(
        f"  OpenCV 2x upscaled (permissive):  {n_up} marker(s) "
        "(if >0 while 1x was 0 → tag likely too small in image)"
    )
    mn, sd, p5, p95 = _gray_stats(gray)
    print(
        f"  gray mean={mn:.1f} std={sd:.1f} p5={p5:.0f} p95={p95:.0f} "
        "(extreme p5/p95 → clipping/glare)"
    )


def _load_calib(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError as e:
        print(f"[calib] Cannot read {path}: {e}")
        return None


def _check_calib_vs_frame(calib: dict, w: int, h: int) -> None:
    if "resolution" not in calib:
        print(
            "[calib] JSON has no 'resolution' field — cannot verify it matches this camera mode.\n"
            "        Re-run:  python camera_calibration.py calibrate -o <same.json>\n"
            "        (saved JSON will include resolution.)"
        )
        return
    cw, ch = int(calib["resolution"][0]), int(calib["resolution"][1])
    if abs(cw - w) > 2 or abs(ch - h) > 2:
        print(
            f"[calib] *** MISMATCH ***  Calibration was at {cw}x{ch}, "
            f"current frames are {w}x{h}.\n"
            "        fx,fy,cx,cy are scaled to wrong pixels → pose wrong; "
            "fix by using the SAME WxH as calibration or recalibrate."
        )
    else:
        print(f"[calib] Resolution matches: frame {w}x{h} vs calib {cw}x{ch} ✓")


@dataclass
class ConfigStats:
    hits: int = 0
    frames: int = 0
    margins: List[float] = field(default_factory=list)
    hams: List[int] = field(default_factory=list)


def run_on_frame_opencv(gray: np.ndarray, print_lines: bool) -> Dict[str, ConfigStats]:
    stats: Dict[str, ConfigStats] = {}
    for label, preset in OPENCV_SUITE:
        st = ConfigStats(frames=1)
        try:
            ids, _corners = _opencv_detect_ids(gray, preset)
        except Exception as e:
            print(f"    {label}: ERROR {e}")
            stats[label] = st
            continue
        n = 0 if ids is None else len(ids)
        if n > 0:
            st.hits = 1
            for _ in range(n):
                st.margins.append(1.0)
                st.hams.append(0)
        if print_lines:
            if n == 0:
                print(f"    {label}: 0 markers")
            else:
                idlist = sorted(int(ids[j][0]) for j in range(n))
                print(f"    {label}: ids={idlist}")
        stats[label] = st
    return stats


def merge_stats(acc: Dict[str, ConfigStats], inc: Dict[str, ConfigStats]) -> None:
    for k, st in inc.items():
        if k not in acc:
            acc[k] = ConfigStats()
        acc[k].frames += st.frames
        acc[k].hits += st.hits
        acc[k].margins.extend(st.margins)
        acc[k].hams.extend(st.hams)


def draw_tags_opencv(gray: np.ndarray, preset: str = "default") -> np.ndarray:
    ids, corners = _opencv_detect_ids(gray, preset)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
    return bgr


def main():
    p = argparse.ArgumentParser(description="AprilTag recognition diagnostics")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--calib-file", type=str, default=None)
    p.add_argument("--tag-size", type=float, default=0.06, help="Unused (reserved); diagnose is detection-only")
    p.add_argument("--seconds", type=float, default=5.0, help="Live capture duration")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=use --seconds only)")
    p.add_argument("--image", type=str, default=None, help="Analyze one image instead of camera")
    p.add_argument("--save-dir", type=str, default=None, help="Save gray + annotated PNGs")
    args = p.parse_args()

    calib = _load_calib(args.calib_file) if args.calib_file else None
    if args.calib_file and calib is None:
        sys.exit(1)

    print(
        "[diagnose] engine=OpenCV DICT_APRILTAG_36h11 (3 presets). "
        "pupil-apriltags not used (decode can SIGSEGV on Pi).",
        flush=True,
    )

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Cannot read image: {args.image}")
            sys.exit(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        print(f"[frame] {args.image}  size {w}x{h}")
        if calib:
            _check_calib_vs_frame(calib, w, h)
        lv = _laplacian_var(gray)
        mn, sd, p5, p95 = _gray_stats(gray)
        print(
            f"[image] Laplacian var={lv:.1f} (higher=sharper; typical webcam >50–150 if ok)\n"
            f"        gray mean={mn:.1f} std={sd:.1f} p5={p5:.0f} p95={p95:.0f} "
            f"(p5~0 or p95~255 suggests clipping / glare)"
        )
        print("[detectors] single frame (OpenCV):")
        run_on_frame_opencv(gray, print_lines=True)
        _crosscheck_frames(gray, os.path.basename(args.image))

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.save_dir, "input_gray.png"), gray)
            ids, _ = _opencv_detect_ids(gray, "default")
            if ids is not None and len(ids) > 0:
                cv2.imwrite(
                    os.path.join(args.save_dir, "annotated_opencv_default.png"),
                    draw_tags_opencv(gray, "default"),
                )
        _print_interpretation_hints()
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    acc: Dict[str, ConfigStats] = {}
    laps: List[float] = []
    means: List[float] = []
    p5s: List[float] = []
    p95s: List[float] = []
    t_end = time.monotonic() + args.seconds
    n = 0
    best_gray: Optional[np.ndarray] = None
    best_lv = -1.0

    print(
        f"[live] camera={args.camera} request {args.camera_width}x{args.camera_height} "
        f"for ~{args.seconds}s (Ctrl+C to stop early)\n"
        f"       Point the tag steady in view; script compares 3 detector configs per frame.\n"
        f"       Tip: stop main.py / other apps using the same camera, or reads may fail/hang.\n"
        f"       Uses OpenCV only (no pupil-apriltags in this tool).",
        flush=True,
    )

    hb_state = {"n": 0, "phase": "starting", "fail_reads": 0}
    hb_stop = threading.Event()
    t_loop_start = time.monotonic()

    def _heartbeat():
        while not hb_stop.wait(2.0):
            elapsed = time.monotonic() - t_loop_start
            msg = (
                f"[live] heartbeat {elapsed:.1f}s  frames={hb_state['n']}  "
                f"phase={hb_state['phase']}"
            )
            if hb_state["phase"] == "read_failed" and hb_state["fail_reads"] > 80:
                msg += "  (camera not delivering — close other /dev/video users?)"
            print(msg, flush=True)

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    read_fail_streak = 0
    warned_busy = False

    try:
        while time.monotonic() < t_end:
            if args.max_frames and n >= args.max_frames:
                break
            hb_state["phase"] = "cap_read"
            ok, frame = cap.read()
            if not ok:
                read_fail_streak += 1
                hb_state["fail_reads"] = read_fail_streak
                hb_state["phase"] = "read_failed"
                if read_fail_streak == 1:
                    print("[live] cap.read() failed once (retrying)...", flush=True)
                if not warned_busy and read_fail_streak >= 100:
                    warned_busy = True
                    print(
                        "[live] Many failed reads — is `main.py` or another process "
                        "using the same camera? Stop it and retry.",
                        flush=True,
                    )
                time.sleep(0.02)
                continue
            read_fail_streak = 0
            hb_state["fail_reads"] = 0
            hb_state["phase"] = "processing"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if n == 0:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[live] Actual capture: {w}x{h}  (V4L reports {aw}x{ah})", flush=True)
                if calib:
                    _check_calib_vs_frame(calib, w, h)
                print("[live] First frame OK — running 3 OpenCV detector presets per frame.", flush=True)

            lv = _laplacian_var(gray)
            laps.append(lv)
            mn, _sd, p5, p95 = _gray_stats(gray)
            means.append(mn)
            p5s.append(p5)
            p95s.append(p95)
            if lv > best_lv:
                best_lv = lv
                best_gray = gray.copy()

            hb_state["phase"] = "opencv_detect"
            st = run_on_frame_opencv(gray, print_lines=False)
            merge_stats(acc, st)
            n += 1
            hb_state["n"] = n
            if n % 15 == 0:
                hit_a = acc[OPENCV_SUITE[0][0]].hits
                hit_b = acc[OPENCV_SUITE[1][0]].hits
                print(
                    f"  ... {n} frames  "
                    f"A_hits={hit_a} B_hits={hit_b}  "
                    f"lap_var~{statistics.median(laps):.0f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[live] stopped early", flush=True)
    finally:
        hb_stop.set()

    cap.release()

    if n == 0:
        print(
            "\nNo frames captured. Typical causes:\n"
            "  • Another program still has the camera open (e.g. main.py --display).\n"
            "  • Wrong --camera index.\n"
            "  • USB camera unplugged or permissions.",
            flush=True,
        )
        sys.exit(1)

    print(f"\n[summary] processed {n} frames\n")
    for name in [x[0] for x in OPENCV_SUITE]:
        st = acc[name]
        rate = 100.0 * st.hits / max(st.frames, 1)
        print(f"  {name}\n    detection_frame_rate={rate:.1f}%")

    print(
        f"\n[focus] Laplacian variance: min={min(laps):.0f} max={max(laps):.0f} "
        f"median={statistics.median(laps):.0f}"
    )
    if means:
        print(
            f"[exposure] median over frames: gray_mean={statistics.median(means):.1f} "
            f"p5={statistics.median(p5s):.0f} p95={statistics.median(p95s):.0f}"
        )

    all_zero = all(acc[k].hits == 0 for k in acc)
    if best_gray is not None:
        _crosscheck_frames(best_gray, "best_focus (sharpest captured frame)")

    if args.save_dir and best_gray is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        path_g = os.path.join(args.save_dir, "best_focus_gray.png")
        cv2.imwrite(path_g, best_gray)
        print(f"[save] sharpest frame (by Laplacian) → {path_g}")
        ids, _ = _opencv_detect_ids(best_gray, "default")
        if ids is not None and len(ids) > 0:
            cv2.imwrite(
                os.path.join(args.save_dir, "best_focus_annotated_opencv.png"),
                draw_tags_opencv(best_gray, "default"),
            )
        print(
            "\n[next] Open the PNG in an image viewer, zoom in: is the tag sharp and "
            "~40+ px per side? Re-run:\n"
            "  uv run python client/apriltag_diagnose.py --image /tmp/tagdiag/best_focus_gray.png"
        )

    if all_zero:
        _print_zero_detection_hints()

    _print_interpretation_hints()


def _print_zero_detection_hints() -> None:
    print(
        "\n*** All OpenCV presets saw 0 markers — not a decimate-tuning issue. Checklist: ***\n"
        "  1) Tag filled enough of the frame? Wrong --camera index or tag out of FOV?\n"
        "  2) Printable must be tag36h11 — use client/generate_tags.py.\n"
        "  3) main.py tracking uses the same OpenCV DICT_APRILTAG_36h11 path as this tool (no pupil-apriltags).\n"
        "  4) Contrast, focus, flat paper, larger print / closer distance.\n"
    )


def _print_interpretation_hints() -> None:
    print(
        "\n--- How to read this ---\n"
        "• A (permissive) >> B: tag small or low contrast — move closer, improve print, flatter mount.\n"
        "• All zero: checklist above; inspect saved PNG zoomed.\n"
        "• Calib mismatch: fix before trusting distance in other apps; this tool only checks resolution.\n"
        "• Low Laplacian + motion: reduce speed or increase light; check lens focus.\n"
        "• p5≈0 & p95≈255: overexposure/glare — diffuse light, matte paper.\n"
    )


if __name__ == "__main__":
    main()
