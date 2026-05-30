#!/usr/bin/env python3
"""
Generate printable AprilTag images (tag36h11 family) on A4 paper.

All output images match the target paper size exactly, so printing at
100% / actual size gives the correct physical tag dimensions.

Usage examples:
    # Single tag, 60 mm, on A4
    python generate_tags.py --ids 0 --size 60

    # Multiple tags on one A4 sheet
    python generate_tags.py --ids 0-5 --size 80 --cols 3

    # Specific IDs
    python generate_tags.py --ids 0,3,7 --size 100
"""

import argparse
import math
import sys

import cv2
import numpy as np

TAG36H11_CELLS = 10

PAPER_SIZES = {
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
    "letter": (215.9, 279.4),
}

PAPER_MARGIN_MM = 8.0


def save_with_dpi(path: str, img: np.ndarray, dpi: int):
    """Save image as PNG with embedded DPI metadata."""
    try:
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img.save(path, dpi=(dpi, dpi))
    except ImportError:
        cv2.imwrite(path, img)


def _get_tag_image_cv(tag_id: int, cells: int = TAG36H11_CELLS) -> np.ndarray:
    """Render a single tag as a binary (0/255) image using OpenCV ArUco."""
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag_img = cv2.aruco.generateImageMarker(aruco_dict, tag_id, cells)
        return tag_img
    except Exception:
        pass

    try:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        tag_img = np.zeros((cells, cells), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, tag_id, cells, tag_img)
        return tag_img
    except Exception as e:
        print(f"Error: cannot generate tag {tag_id}: {e}")
        print("Make sure opencv-contrib-python is installed.")
        sys.exit(1)


def _render_tag(tag_id: int, size_mm: float, px_per_mm: float) -> np.ndarray:
    """Render a tag at the exact physical size, returning the tag pixels only."""
    tag_px = int(round(size_mm * px_per_mm))
    cell_px = tag_px // TAG36H11_CELLS
    tag_px = cell_px * TAG36H11_CELLS
    raw = _get_tag_image_cv(tag_id, TAG36H11_CELLS)
    return cv2.resize(raw, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)


def generate_single_on_paper(
    tag_id: int,
    size_mm: float,
    dpi: int,
    paper: str,
) -> np.ndarray:
    """Place a single tag centered on a paper-sized canvas."""
    pw_mm, ph_mm = PAPER_SIZES[paper]
    px_per_mm = dpi / 25.4
    canvas_w = int(round(pw_mm * px_per_mm))
    canvas_h = int(round(ph_mm * px_per_mm))
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    tag_img = _render_tag(tag_id, size_mm, px_per_mm)
    tag_px = tag_img.shape[0]

    if size_mm > min(pw_mm, ph_mm) - 2 * PAPER_MARGIN_MM:
        print(f"Error: tag {size_mm}mm does not fit {paper.upper()} paper.")
        sys.exit(1)

    # center tag
    off_x = (canvas_w - tag_px) // 2
    off_y = (canvas_h - tag_px) // 2
    canvas[off_y: off_y + tag_px, off_x: off_x + tag_px] = tag_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, px_per_mm * 1.2 / 10)
    th = max(1, int(fs * 2))

    # label below tag
    label = f"tag36h11  ID: {tag_id}  size: {size_mm}mm"
    label_y = off_y + tag_px + int(8 * px_per_mm)
    cv2.putText(canvas, label, (off_x, label_y), font, fs, 0, th)

    # verification ruler (50 mm)
    ruler_len = int(50 * px_per_mm)
    ruler_y = label_y + int(8 * px_per_mm)
    ruler_th = max(1, int(px_per_mm * 0.3))
    cv2.line(canvas, (off_x, ruler_y), (off_x + ruler_len, ruler_y), 0, ruler_th)
    cv2.line(canvas, (off_x, ruler_y - 6), (off_x, ruler_y + 6), 0, ruler_th)
    cv2.line(canvas, (off_x + ruler_len, ruler_y - 6),
             (off_x + ruler_len, ruler_y + 6), 0, ruler_th)
    cv2.putText(canvas, "50 mm", (off_x + 5, ruler_y - 10),
                font, fs * 0.7, 0, max(1, th))

    # print instruction
    note = f"Print on {paper.upper()} at 100% scale (no fit-to-page). Verify 50mm ruler."
    note_y = canvas_h - int(3 * px_per_mm)
    cv2.putText(canvas, note, (off_x, note_y), font, fs * 0.6, 120, max(1, th))

    return canvas


def generate_sheet_on_paper(
    tag_ids: list,
    size_mm: float,
    dpi: int,
    cols: int,
    paper: str,
) -> np.ndarray:
    """Arrange multiple tags on a paper-sized canvas."""
    pw_mm, ph_mm = PAPER_SIZES[paper]
    px_per_mm = dpi / 25.4
    canvas_w = int(round(pw_mm * px_per_mm))
    canvas_h = int(round(ph_mm * px_per_mm))
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    usable_w = pw_mm - 2 * PAPER_MARGIN_MM
    usable_h = ph_mm - 2 * PAPER_MARGIN_MM - 15  # room for labels at bottom

    gap_mm = max(5.0, size_mm * 0.15)

    # how many fit?
    max_cols = int((usable_w + gap_mm) / (size_mm + gap_mm))
    if cols > max_cols:
        cols = max_cols
    if cols < 1:
        print(f"Error: tag {size_mm}mm too large for {paper.upper()} paper.")
        sys.exit(1)
    rows = math.ceil(len(tag_ids) / cols)
    max_rows = int((usable_h + gap_mm) / (size_mm + gap_mm))
    if rows > max_rows:
        print(f"Warning: only {max_rows * cols} of {len(tag_ids)} tags fit on "
              f"one {paper.upper()} sheet. Truncating.")
        tag_ids = tag_ids[: max_rows * cols]
        rows = max_rows

    tag_px = _render_tag(tag_ids[0], size_mm, px_per_mm).shape[0]
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
        tag_img = _render_tag(tid, size_mm, px_per_mm)
        tp = tag_img.shape[0]
        y0 = off_y + r * (tp + gap_px)
        x0 = off_x + c * (tp + gap_px)
        canvas[y0: y0 + tp, x0: x0 + tp] = tag_img
        # small ID label below each tag
        cv2.putText(canvas, f"ID:{tid}", (x0, y0 + tp + int(3 * px_per_mm)),
                    font, fs * 0.8, 80, max(1, th))

    # dashed cut lines between tags
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

    # bottom labels
    info_y = off_y + grid_h + int(10 * px_per_mm)
    cv2.putText(canvas, f"tag36h11  size: {size_mm}mm  IDs: {tag_ids[0]}-{tag_ids[-1]}",
                (off_x, info_y), font, fs, 0, th)
    note = f"Print on {paper.upper()} at 100% scale (no fit-to-page)."
    note_y = canvas_h - int(3 * px_per_mm)
    cv2.putText(canvas, note, (off_x, note_y), font, fs * 0.6, 120, max(1, th))

    return canvas


def parse_ids(ids_str: str) -> list:
    """Parse '0-5', '0,3,7', or '0' into a list of ints."""
    result = []
    for part in ids_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate printable AprilTag (tag36h11) images on paper"
    )
    parser.add_argument(
        "--ids", type=str, default="0",
        help="Tag IDs: single (0), range (0-5), or comma-separated (0,3,7)"
    )
    parser.add_argument(
        "--size", type=float, default=60.0,
        help="Physical tag size in mm (default: 60)"
    )
    parser.add_argument(
        "--paper", type=str, default="a4",
        help="Paper size: a4, a3, letter (default: a4)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)"
    )
    parser.add_argument(
        "--cols", type=int, default=0,
        help="Columns per sheet (0=one tag per page, default: 0)"
    )
    parser.add_argument(
        "-o", "--output", default="apriltag",
        help="Output path prefix (default: apriltag)"
    )

    args = parser.parse_args()
    paper = args.paper.lower()
    if paper not in PAPER_SIZES:
        print(f"Error: unknown paper size '{paper}'. Use: {', '.join(PAPER_SIZES)}")
        sys.exit(1)

    tag_ids = parse_ids(args.ids)

    if args.cols > 0:
        sheet = generate_sheet_on_paper(
            tag_ids, args.size, args.dpi, args.cols, paper
        )
        out_path = args.output + "_sheet.png"
        save_with_dpi(out_path, sheet, args.dpi)
        print(f"Saved tag sheet ({len(tag_ids)} tags) to {out_path}")
    else:
        for tid in tag_ids:
            img = generate_single_on_paper(
                tid, args.size, args.dpi, paper
            )
            out_path = f"{args.output}_id{tid}.png"
            save_with_dpi(out_path, img, args.dpi)
            print(f"Saved tag ID {tid} to {out_path}")

    pw, ph = PAPER_SIZES[paper]
    print(f"Paper: {paper.upper()} ({pw:.0f} x {ph:.0f} mm), DPI: {args.dpi}")
    print(f"IMPORTANT: Print at 100% / actual size (do NOT use fit-to-page).")
    print(f"After printing, measure the tag/ruler to get the actual size for --tag-size.")


if __name__ == "__main__":
    main()
