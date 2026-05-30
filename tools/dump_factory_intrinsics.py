#!/usr/bin/env python3
"""Dump the factory-calibrated intrinsics + D2C extrinsic of the Orbbec
Gemini Pro to an `RgbdCalibration` npz, ready for `camera_node` and the
software-align fallback in `align_depth_to_color`.

The device stores both color and depth intrinsics + the depth-to-color
rigid transform in EEPROM at the factory; this is enough for sub-pixel
metric back-projection (Orbbec's own `getCameraParam()` is what the SDK's
HW D2C aligner uses internally). Running a chessboard calibration on top
of this typically gains <0.5 px and is unnecessary unless the lens has
been physically swapped.

Usage
-----
    .venv/bin/python tools/dump_factory_intrinsics.py
    # writes config/camera_calib_factory.npz

Two passes are run:
  1) d2c="off" — pulls the raw stereo extrinsic (depth->color) saved in
     the device. This is what we ship to disk.
  2) d2c="hw"  — sanity-print the post-align color K (should match raw
     color K within sub-pixel rounding) so we can confirm the SDK is
     happy with HW alignment on this particular device.

Both passes use `color_width=640, color_height=480, color_format='mjpg'`
because that is the only profile that pairs with HW D2C on Gemini Pro.
The intrinsics are valid AT that resolution; `camera_node._load_calib_npz`
already scales fx/fy/cx/cy to the requested stream geometry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ORBBEC_DIR = REPO_ROOT / "thirdparty" / "OrbbecSDK_v1" / "python"
sys.path.insert(0, str(ORBBEC_DIR))

from orbbec_gemini import (  # noqa: E402
    FactoryCameraParam,
    OrbbecGemini,
    OrbbecGeminiError,
)


def _print_param(tag: str, p: FactoryCameraParam) -> None:
    print(f"\n[{tag}] is_d2c_aligned={p.is_d2c_aligned}  "
          f"is_mirrored={p.is_mirrored}")
    K_c = p.K_color
    K_d = p.K_depth
    print(f"  color size = {p.color_size[0]}x{p.color_size[1]}")
    print(f"    fx={K_c[0,0]:.3f} fy={K_c[1,1]:.3f} "
          f"cx={K_c[0,2]:.3f} cy={K_c[1,2]:.3f}")
    print(f"    dist[k1,k2,p1,p2,k3,k4,k5,k6]="
          f"{np.array2string(p.dist_color, precision=5, suppress_small=True)}")
    print(f"  depth size = {p.depth_size[0]}x{p.depth_size[1]}")
    print(f"    fx={K_d[0,0]:.3f} fy={K_d[1,1]:.3f} "
          f"cx={K_d[0,2]:.3f} cy={K_d[1,2]:.3f}")
    print(f"    dist[k1,k2,p1,p2,k3,k4,k5,k6]="
          f"{np.array2string(p.dist_depth, precision=5, suppress_small=True)}")
    print("  R_d2c =")
    print("    " + np.array2string(p.R_d2c, precision=5,
                                   suppress_small=True).replace("\n", "\n    "))
    print(f"  T_d2c (mm) = "
          f"{np.array2string(p.T_d2c, precision=3, suppress_small=True)}")
    # Quick HFOV/VFOV sanity check — useful when debugging old npz files
    fovx = 2.0 * np.degrees(np.arctan(p.color_size[0] / (2.0 * K_c[0, 0])))
    fovy = 2.0 * np.degrees(np.arctan(p.color_size[1] / (2.0 * K_c[1, 1])))
    print(f"  color HFOV ≈ {fovx:.1f}°   VFOV ≈ {fovy:.1f}°")


def _read_param(d2c: str, color_w: int, color_h: int) -> FactoryCameraParam:
    cam = OrbbecGemini(streams=["depth", "color"])
    try:
        info = cam.open(
            color_width=color_w, color_height=color_h, color_fps=30,
            color_format="mjpg", d2c=d2c,
        )
        # `getCameraParam` requires the pipeline to be running; the open()
        # path probes a few frames already, so it is by the time we get here.
        # Pull a single frame just to make sure the pipeline is stable.
        _ = cam.read_color_sdk(timeout_ms=1000) if d2c == "off" \
            else cam.read_rgbd_sdk(timeout_ms=1000)
        param = cam.get_camera_param()
        print(f"  device: {info.name}  serial={info.serial}  "
              f"firmware={info.firmware}")
        return param
    finally:
        cam.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "config" / "camera_calib_factory.npz",
                    help="output npz path (default: %(default)s)")
    ap.add_argument("--color-width", type=int, default=640)
    ap.add_argument("--color-height", type=int, default=480)
    ap.add_argument("--skip-hw-sanity", action="store_true",
                    help="skip the HW-D2C sanity probe")
    args = ap.parse_args()

    print("[1/2] reading raw factory params (d2c=off)...")
    raw = _read_param("off", args.color_width, args.color_height)
    _print_param("raw", raw)

    if not args.skip_hw_sanity:
        print("\n[2/2] reading post-align params (d2c=hw) for sanity...")
        try:
            aligned = _read_param("hw", args.color_width, args.color_height)
            _print_param("hw-aligned", aligned)
            dK = np.linalg.norm(aligned.K_color - raw.K_color)
            print(f"\n  ‖K_color(hw) - K_color(raw)‖ = {dK:.4f} (≈0 expected)")
            if not aligned.is_d2c_aligned:
                print("  WARN: SDK reports is_d2c_aligned=False even though "
                      "we opened with d2c='hw'.")
        except OrbbecGeminiError as e:
            print(f"  WARN: HW D2C sanity probe failed: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    calib = raw.to_rgbd_calibration()
    calib.save(args.out)
    print(f"\nwrote {args.out}")
    print("\nUse with camera_node by passing:")
    print(f"  --ros-args -p calib_npz:={args.out}")


if __name__ == "__main__":
    main()
