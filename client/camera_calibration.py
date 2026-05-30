#!/usr/bin/env python3
"""
Camera calibration tool with two sub-commands:

    python camera_calibration.py generate  -- create a printable chessboard image
    python camera_calibration.py calibrate -- run interactive calibration
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from opencv_gui import (
    ensure_local_display,
    opencv_highgui_available,
    print_display_failure_help,
)
from teach.calib_session import CalibSession

# Paper sizes in mm (width, height) -- portrait orientation
PAPER_SIZES = {
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
    "letter": (215.9, 279.4),
}

PAPER_MARGIN_MM = 8.0  # safe non-printable margin for most printers


def _make_paper_canvas(paper: str, dpi: int):
    """Create a white canvas matching the paper size at the given DPI."""
    pw_mm, ph_mm = PAPER_SIZES[paper.lower()]
    px_per_mm = dpi / 25.4
    canvas_w = int(round(pw_mm * px_per_mm))
    canvas_h = int(round(ph_mm * px_per_mm))
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
    return canvas, pw_mm, ph_mm, px_per_mm


def _save_with_dpi(path: str, img: np.ndarray, dpi: int):
    """Save image as PNG with embedded DPI metadata (pHYs chunk)."""
    try:
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img.save(path, dpi=(dpi, dpi))
    except ImportError:
        cv2.imwrite(path, img)


# ---------------------------------------------------------------------------
#  Sub-command: generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    """Generate a printable chessboard calibration pattern on A4 paper."""
    rows = args.rows
    cols = args.cols
    square_mm = args.square_size
    dpi = args.dpi
    paper = args.paper
    output = args.output

    if paper.lower() not in PAPER_SIZES:
        print(f"Error: unknown paper size '{paper}'. Use: {', '.join(PAPER_SIZES)}")
        sys.exit(1)

    canvas, pw_mm, ph_mm, px_per_mm = _make_paper_canvas(paper, dpi)
    canvas_h, canvas_w = canvas.shape[:2]

    sq_px = int(round(square_mm * px_per_mm))

    pattern_w = cols * sq_px
    pattern_h = rows * sq_px

    usable_w_mm = pw_mm - 2 * PAPER_MARGIN_MM
    usable_h_mm = ph_mm - 2 * PAPER_MARGIN_MM

    board_w_mm = cols * square_mm
    board_h_mm = rows * square_mm

    if board_w_mm > usable_w_mm or board_h_mm > usable_h_mm:
        max_sq_w = usable_w_mm / cols
        max_sq_h = usable_h_mm / rows
        max_sq = min(max_sq_w, max_sq_h)
        print(f"Error: {cols}x{rows} grid with {square_mm}mm squares "
              f"({board_w_mm:.0f}x{board_h_mm:.0f}mm) does not fit "
              f"{paper.upper()} ({pw_mm:.0f}x{ph_mm:.0f}mm).")
        print(f"  Max square size for this grid: {max_sq:.1f}mm")
        sys.exit(1)

    # center the pattern on the page
    off_x = (canvas_w - pattern_w) // 2
    off_y = (canvas_h - pattern_h) // 2

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = off_x + c * sq_px
                y0 = off_y + r * sq_px
                canvas[y0: y0 + sq_px, x0: x0 + sq_px] = 0

    inner_corners_w = cols - 1
    inner_corners_h = rows - 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, sq_px / 120.0)
    thickness = max(1, int(font_scale * 2))

    # top label
    label = (
        f"Chessboard {cols}x{rows}  "
        f"inner corners: {inner_corners_w}x{inner_corners_h}  "
        f"square: {square_mm}mm"
    )
    label_y = off_y - int(10 * px_per_mm)
    if label_y < int(5 * px_per_mm):
        label_y = int(5 * px_per_mm)
    cv2.putText(canvas, label, (off_x, label_y),
                font, font_scale, 0, thickness)

    # verification ruler (50 mm) below the pattern
    ruler_len_px = int(50 * px_per_mm)
    ruler_y = off_y + pattern_h + int(8 * px_per_mm)
    if ruler_y > canvas_h - int(3 * px_per_mm):
        ruler_y = canvas_h - int(3 * px_per_mm)
    ruler_x0 = off_x
    ruler_x1 = ruler_x0 + ruler_len_px
    ruler_th = max(1, int(px_per_mm * 0.3))
    cv2.line(canvas, (ruler_x0, ruler_y), (ruler_x1, ruler_y), 0, ruler_th)
    cv2.line(canvas, (ruler_x0, ruler_y - 8), (ruler_x0, ruler_y + 8), 0, ruler_th)
    cv2.line(canvas, (ruler_x1, ruler_y - 8), (ruler_x1, ruler_y + 8), 0, ruler_th)
    cv2.putText(canvas, "50 mm", (ruler_x0 + 5, ruler_y - 12),
                font, font_scale * 0.7, 0, max(1, thickness))

    # print instruction at bottom
    note = f"Print on {paper.upper()} at 100% scale (no fit-to-page). Verify 50mm ruler."
    note_y = canvas_h - int(3 * px_per_mm)
    cv2.putText(canvas, note, (off_x, note_y),
                font, font_scale * 0.6, 120, max(1, thickness))

    _save_with_dpi(output, canvas, dpi)
    print(f"Saved chessboard pattern to {output}")
    print(f"  Paper: {paper.upper()} ({pw_mm:.0f} x {ph_mm:.0f} mm)")
    print(f"  Grid: {cols} x {rows} squares  ({inner_corners_w}x{inner_corners_h} inner corners)")
    print(f"  Square size: {square_mm} mm")
    print(f"  Image: {canvas_w} x {canvas_h} px  ({dpi} DPI)")
    print(f"  IMPORTANT: Print at 100% / actual size (do NOT use fit-to-page).")


# ---------------------------------------------------------------------------
#  Sub-command: calibrate
# ---------------------------------------------------------------------------

def cmd_calibrate(args):
    """Interactive camera calibration using a chessboard pattern.

    底层算法委托给 ``teach.calib_session.CalibSession``——同一份代码也被
    teach server 的 web 标定流程复用。CLI 在这里负责开摄像头、可选 OpenCV
    GUI 显示、循环条件和 stdout 进度。
    """
    inner_w = args.cols - 1
    inner_h = args.rows - 1
    min_frames = args.min_frames
    camera_id = args.camera
    output = args.output
    use_display = getattr(args, "display", True)

    if use_display:
        ensure_local_display()
        if not opencv_highgui_available():
            print_display_failure_help("Falling back to headless calibration (auto-capture).")
            use_display = False

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_id}")
        sys.exit(1)

    capture_interval = 1.0
    session = CalibSession(
        rows=args.rows,
        cols=args.cols,
        square_mm=args.square_size,
        min_frames=min_frames,
        auto_interval_s=capture_interval,
        output_path=Path(output),
    )
    session.start()

    print(f"Calibrating with {inner_w}x{inner_h} inner corners, "
          f"square={args.square_size}mm")
    print(
        f"Auto-capture when the full chessboard is visible; "
        f"{capture_interval:.0f}s minimum between captures (SSH-friendly, no SPACE)."
    )
    if use_display:
        print("Press 'q' in the window to finish early.")
    else:
        print("Press Ctrl+C to stop.")

    last_captured_seen = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            snap = session.feed_frame(frame)
            if snap.captured > last_captured_seen:
                last_captured_seen = snap.captured
                print(f"  Captured frame {snap.captured}")

            key = -1
            if use_display:
                display = frame.copy()
                # 复绘角点：feed_frame 已经做过一次 findChessboardCorners，
                # 为了不重复检测我们这里再快速找一次（高分辨率下 10ms 量级）。
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, (inner_w, inner_h), None)
                if found:
                    cv2.drawChessboardCorners(display, (inner_w, inner_h), corners, found)
                status = f"Captured: {snap.captured}/{min_frames}"
                if found:
                    cd = snap.auto_cooldown_left
                    status += f"  next in {cd:.1f}s" if cd > 0 else "  (auto-capture)"
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Calibration", display)
                key = cv2.waitKey(30) & 0xFF
            else:
                time.sleep(0.03)

            if snap.captured >= min_frames:
                break
            if key == ord("q"):
                break
    finally:
        cap.release()
        if use_display:
            cv2.destroyAllWindows()

    result = session.finalize()
    snap = session.snapshot()
    if result is None:
        print(f"Calibration failed: {snap.error_msg or 'unknown error'}")
        sys.exit(1)
    print(f"Computed calibration from {result.frame_count} frames.")
    print(f"Calibration saved to {result.saved_to}")
    print(f"  fx={result.fx:.2f}  fy={result.fy:.2f}")
    print(f"  cx={result.cx:.2f}  cy={result.cy:.2f}")
    print(f"  RMS reprojection error: {result.rms_error:.4f}")


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration tool"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- generate -----------------------------------------------------------
    gen = sub.add_parser("generate", help="Generate a printable chessboard image")
    gen.add_argument("--rows", type=int, default=9,
                     help="Number of rows of squares (default: 9)")
    gen.add_argument("--cols", type=int, default=6,
                     help="Number of columns of squares (default: 6)")
    gen.add_argument("--square-size", type=float, default=25.0,
                     help="Square side length in mm (default: 25)")
    gen.add_argument("--paper", type=str, default="a4",
                     help="Paper size: a4, a3, letter (default: a4)")
    gen.add_argument("--dpi", type=int, default=300,
                     help="Output DPI (default: 300)")
    gen.add_argument("-o", "--output", default="chessboard.png",
                     help="Output file path (default: chessboard.png)")

    # -- calibrate ----------------------------------------------------------
    cal = sub.add_parser("calibrate", help="Run interactive camera calibration")
    cal.add_argument("--rows", type=int, default=9,
                     help="Rows of squares on the printed board (default: 9)")
    cal.add_argument("--cols", type=int, default=6,
                     help="Columns of squares on the printed board (default: 6)")
    cal.add_argument("--square-size", type=float, default=25.0,
                     help="ACTUAL printed square side length in mm -- "
                          "measure with a ruler after printing (default: 25)")
    cal.add_argument("--camera", type=int, default=0,
                     help="Camera device ID (default: 0)")
    cal.add_argument("--min-frames", type=int, default=15,
                     help="Minimum frames to capture (default: 15)")
    cal.add_argument("--display", action="store_true", default=False,
                     help="Show live camera view on local screen (for SSH use)")
    cal.add_argument("-o", "--output", default="config/camera_calib.json",
                     help="Output calibration file (default: config/camera_calib.json)")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)


if __name__ == "__main__":
    main()
