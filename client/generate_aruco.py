#!/usr/bin/env python3
"""
Generate printable classic ArUco square markers on A4 / A3 / Letter.

Uses OpenCV ``cv2.aruco.generateImageMarker`` (opencv-contrib). Output canvas
matches paper size at the given DPI so printing at 100% scale preserves
physical ``--size`` (marker edge length in mm).

Examples:

    # One marker, default DICT_5X5_250 ID 0, 45 mm on A4
    python generate_aruco.py --ids 0

    # Sheet: IDs 0–11, three columns
    python generate_aruco.py --dict 5X5_250 --ids 0-11 --size 40 --cols 3

    # Match main.py default tracking dict (AprilTag 36h11 bit pattern)
    python generate_aruco.py --dict APRILTAG_36H11 --ids 0 --size 60

For tag36h11-only workflows you can still use ``generate_tags.py`` (cell-aligned
scaling); this script is aimed at classic ArUco dictionaries (4x4, 5x5, …).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_CLIENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CLIENT_DIR not in sys.path:
    sys.path.insert(0, _CLIENT_DIR)

import cv2
import numpy as np

from generate_tags import PAPER_MARGIN_MM, PAPER_SIZES, parse_ids, save_with_dpi


def _normalize_dict_attr(name: str) -> str:
    key = name.strip().upper().replace("-", "_")
    if not key.startswith("DICT_"):
        key = "DICT_" + key
    if not hasattr(cv2.aruco, key):
        names = sorted(
            a for a in dir(cv2.aruco) if a.startswith("DICT_") and a.isupper()
        )
        print(f"Error: unknown dictionary {name!r} -> {key}")
        print(f"Examples: {', '.join(names[:8])} ...")
        sys.exit(1)
    return key


def _get_dictionary(name: str):
    attr = _normalize_dict_attr(name)
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, attr)), attr


def _dictionary_num_markers(d) -> int:
    bl = getattr(d, "bytesList", None)
    if bl is not None and hasattr(bl, "shape") and len(bl.shape) >= 1:
        return int(bl.shape[0])
    return 0


def _marker_bitmap(aruco_dict, tag_id: int, side_px: int) -> np.ndarray:
    side_px = max(int(side_px), 8)
    try:
        return cv2.aruco.generateImageMarker(aruco_dict, tag_id, side_px)
    except AttributeError:
        img = np.zeros((side_px, side_px), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, tag_id, side_px, img)
        return img


def _render_marker(
    aruco_dict,
    tag_id: int,
    size_mm: float,
    px_per_mm: float,
) -> np.ndarray:
    tag_px = max(int(round(size_mm * px_per_mm)), 8)
    return _marker_bitmap(aruco_dict, tag_id, tag_px)


def _validate_ids(tag_ids: list, max_id: int, dict_label: str):
    bad = [i for i in tag_ids if i < 0 or i > max_id]
    if bad:
        print(
            f"Error: invalid IDs for {dict_label}: {bad}. "
            f"Valid range is 0..{max_id} ({max_id + 1} markers)."
        )
        sys.exit(1)


def generate_single_on_paper(
    aruco_dict,
    dict_label: str,
    tag_id: int,
    size_mm: float,
    dpi: int,
    paper: str,
) -> np.ndarray:
    pw_mm, ph_mm = PAPER_SIZES[paper]
    px_per_mm = dpi / 25.4
    canvas_w = int(round(pw_mm * px_per_mm))
    canvas_h = int(round(ph_mm * px_per_mm))
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    tag_img = _render_marker(aruco_dict, tag_id, size_mm, px_per_mm)
    tag_px = tag_img.shape[0]

    if size_mm > min(pw_mm, ph_mm) - 2 * PAPER_MARGIN_MM:
        print(f"Error: marker {size_mm}mm does not fit {paper.upper()} paper.")
        sys.exit(1)

    off_x = (canvas_w - tag_px) // 2
    off_y = (canvas_h - tag_px) // 2
    canvas[off_y : off_y + tag_px, off_x : off_x + tag_px] = tag_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, px_per_mm * 1.2 / 10)
    th = max(1, int(fs * 2))

    label = f"{dict_label}  ID: {tag_id}  edge: {size_mm}mm"
    label_y = off_y + tag_px + int(8 * px_per_mm)
    cv2.putText(canvas, label, (off_x, label_y), font, fs, 0, th)

    ruler_len = int(50 * px_per_mm)
    ruler_y = label_y + int(8 * px_per_mm)
    ruler_th = max(1, int(px_per_mm * 0.3))
    cv2.line(canvas, (off_x, ruler_y), (off_x + ruler_len, ruler_y), 0, ruler_th)
    cv2.line(canvas, (off_x, ruler_y - 6), (off_x, ruler_y + 6), 0, ruler_th)
    cv2.line(
        canvas,
        (off_x + ruler_len, ruler_y - 6),
        (off_x + ruler_len, ruler_y + 6),
        0,
        ruler_th,
    )
    cv2.putText(
        canvas,
        "50 mm",
        (off_x + 5, ruler_y - 10),
        font,
        fs * 0.7,
        0,
        max(1, th),
    )

    note = f"Print on {paper.upper()} at 100% scale (no fit-to-page). Verify 50mm ruler."
    note_y = canvas_h - int(3 * px_per_mm)
    cv2.putText(canvas, note, (off_x, note_y), font, fs * 0.6, 120, max(1, th))

    return canvas


def generate_sheet_on_paper(
    aruco_dict,
    dict_label: str,
    tag_ids: list,
    size_mm: float,
    dpi: int,
    cols: int,
    paper: str,
) -> np.ndarray:
    pw_mm, ph_mm = PAPER_SIZES[paper]
    px_per_mm = dpi / 25.4
    canvas_w = int(round(pw_mm * px_per_mm))
    canvas_h = int(round(ph_mm * px_per_mm))
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    usable_w = pw_mm - 2 * PAPER_MARGIN_MM
    usable_h = ph_mm - 2 * PAPER_MARGIN_MM - 15

    gap_mm = max(5.0, size_mm * 0.15)

    max_cols = int((usable_w + gap_mm) / (size_mm + gap_mm))
    if cols > max_cols:
        cols = max_cols
    if cols < 1:
        print(f"Error: marker {size_mm}mm too large for {paper.upper()} paper.")
        sys.exit(1)
    rows = math.ceil(len(tag_ids) / cols)
    max_rows = int((usable_h + gap_mm) / (size_mm + gap_mm))
    if rows > max_rows:
        print(
            f"Warning: only {max_rows * cols} of {len(tag_ids)} markers fit on "
            f"one {paper.upper()} sheet. Truncating."
        )
        tag_ids = tag_ids[: max_rows * cols]
        rows = max_rows

    tag_px = _render_marker(aruco_dict, tag_ids[0], size_mm, px_per_mm).shape[0]
    gap_px = int(round(gap_mm * px_per_mm))

    grid_w = cols * tag_px + (cols - 1) * gap_px
    grid_h = rows * tag_px + (rows - 1) * gap_px
    off_x = (canvas_w - grid_w) // 2
    off_y = (canvas_h - grid_h) // 2 - int(5 * px_per_mm)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.3, px_per_mm * 0.8 / 10)
    th = max(1, int(fs * 1.5))

    for i, tid in enumerate(tag_ids):
        r, c = divmod(i, cols)
        tag_img = _render_marker(aruco_dict, tid, size_mm, px_per_mm)
        tp = tag_img.shape[0]
        y0 = off_y + r * (tp + gap_px)
        x0 = off_x + c * (tp + gap_px)
        canvas[y0 : y0 + tp, x0 : x0 + tp] = tag_img
        cv2.putText(
            canvas,
            f"ID:{tid}",
            (x0, y0 + tp + int(3 * px_per_mm)),
            font,
            fs * 0.8,
            80,
            max(1, th),
        )

    dash_len = int(2 * px_per_mm)
    for ci in range(1, cols):
        x = off_x + ci * (tag_px + gap_px) - gap_px // 2
        for y in range(off_y, off_y + grid_h, dash_len * 2):
            y2 = min(y + dash_len, off_y + grid_h)
            cv2.line(canvas, (x, y), (x, y2), 200, 1)
    for ri in range(1, rows):
        y = off_y + ri * (tag_px + gap_px) - gap_px // 2
        for x in range(off_x, off_x + grid_w, dash_len * 2):
            x2 = min(x + dash_len, off_x + grid_w)
            cv2.line(canvas, (x, y), (x2, y), 200, 1)

    info_y = off_y + grid_h + int(10 * px_per_mm)
    cv2.putText(
        canvas,
        f"{dict_label}  edge: {size_mm}mm  IDs: {tag_ids[0]}-{tag_ids[-1]}",
        (off_x, info_y),
        font,
        fs,
        0,
        th,
    )
    note = f"Print on {paper.upper()} at 100% scale (no fit-to-page)."
    note_y = canvas_h - int(3 * px_per_mm)
    cv2.putText(canvas, note, (off_x, note_y), font, fs * 0.6, 120, max(1, th))

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Generate printable ArUco square markers on paper (OpenCV contrib)"
    )
    parser.add_argument(
        "--dict",
        dest="aruco_dict",
        type=str,
        default="5X5_250",
        help="OpenCV predefined dict name, e.g. 5X5_250, 4X4_50, 6X6_250, "
        "APRILTAG_36H11 (default: 5X5_250)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="0",
        help="Marker IDs: 0, 0-5, or 0,3,7",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=45.0,
        help="Physical marker edge length in mm (black square outer size, default: 45)",
    )
    parser.add_argument(
        "--paper",
        type=str,
        default="a4",
        help="Paper: a4, a3, letter (default: a4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=0,
        help="Columns per sheet (0 = one marker per page)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aruco",
        help="Output path prefix (default: aruco)",
    )

    args = parser.parse_args()
    paper = args.paper.lower()
    if paper not in PAPER_SIZES:
        print(f"Error: unknown paper '{paper}'. Use: {', '.join(PAPER_SIZES)}")
        sys.exit(1)

    aruco_dict, dict_label = _get_dictionary(args.aruco_dict)
    n_markers = _dictionary_num_markers(aruco_dict)
    if n_markers <= 0:
        print("Error: could not read marker count from dictionary.")
        sys.exit(1)
    max_id = n_markers - 1

    tag_ids = parse_ids(args.ids)
    _validate_ids(tag_ids, max_id, dict_label)

    if args.cols > 0:
        sheet = generate_sheet_on_paper(
            aruco_dict,
            dict_label,
            tag_ids,
            args.size,
            args.dpi,
            args.cols,
            paper,
        )
        out_path = args.output + "_sheet.png"
        save_with_dpi(out_path, sheet, args.dpi)
        print(f"Saved marker sheet ({len(tag_ids)} markers) to {out_path}")
    else:
        for tid in tag_ids:
            img = generate_single_on_paper(
                aruco_dict, dict_label, tid, args.size, args.dpi, paper
            )
            out_path = f"{args.output}_id{tid}.png"
            save_with_dpi(out_path, img, args.dpi)
            print(f"Saved marker ID {tid} to {out_path}")

    pw, ph = PAPER_SIZES[paper]
    print(f"Dictionary: {dict_label}, paper: {paper.upper()} ({pw:.0f} x {ph:.0f} mm), DPI: {args.dpi}")
    print("Use the same --aruco-dict in main.py as here when tracking.")
    print("Print at 100% / actual size; measure the printed edge for --tag-size (metres).")


if __name__ == "__main__":
    main()
