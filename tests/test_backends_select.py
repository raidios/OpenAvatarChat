#!/usr/bin/env python3
"""
Stage-0 code-level smoke: backend selection works in all 3 modes.

  prefer="cpu"    -> always CPU
  prefer="hailo"  -> RuntimeError if Hailo missing (honest failure)
  prefer="auto"   -> Hailo if probable, else CPU; never raises

Also exercises encode/process/detect on stub backends to catch shape bugs.

Exit:
  0  all asserts passed
  1  any failure
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO_ROOT / "tests"]
sys.path.insert(0, str(REPO_ROOT))

from audio_frontend.backends import (  # noqa: E402
    select_denoiser, select_pose_detector, select_reid,
)
from audio_frontend.backends._hailo_probe import probe_hailo  # noqa: E402


def expect(cond: bool, msg: str) -> bool:
    print(("  ok " if cond else " FAIL ") + msg)
    return cond


def main() -> int:
    ok = True
    probe = probe_hailo()
    print(f"hailo probe: available={probe.available}, "
          f"reason={probe.reason!r}, version={probe.hailort_version}")
    print()

    # cpu
    print("== forced cpu ==")
    d = select_denoiser("cpu", block_size=480)
    p = select_pose_detector("cpu")
    r = select_reid("cpu")
    ok &= expect(d.name.startswith("cpu/"), f"denoiser cpu name = {d.name}")
    ok &= expect(p.name.startswith("cpu/"), f"pose cpu name = {p.name}")
    ok &= expect(r.name.startswith("cpu/"), f"reid cpu name = {r.name}")

    # hailo (must not fail mysteriously: either GREEN or honest RuntimeError)
    print("\n== forced hailo ==")
    try:
        dh = select_denoiser("hailo", block_size=512)
        ok &= expect(dh.name.startswith("hailo/"), f"hailo denoiser = {dh.name}")
    except RuntimeError as e:
        ok &= expect(not probe.available,
                     f"hailo unavailable as probed; raise was honest: {e}")

    # auto
    print("\n== auto ==")
    da = select_denoiser("auto", block_size=512)
    pa = select_pose_detector("auto")
    ra = select_reid("auto")
    if probe.available:
        ok &= expect("hailo" in da.name, f"auto denoiser -> {da.name}")
    else:
        ok &= expect("cpu" in da.name, f"auto denoiser -> {da.name}")

    # functional smoke
    print("\n== functional smoke ==")
    block = np.random.randn(d.block_size).astype(np.float32) * 0.1
    out = d.process(block)
    ok &= expect(out.shape == block.shape, f"denoiser shape preserved {block.shape}")
    ok &= expect(out.dtype == np.float32, f"denoiser dtype = {out.dtype}")

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = p.detect(rgb)
    ok &= expect(isinstance(persons, list), "pose detect returns list")

    crop = np.random.randint(0, 255, size=(128, 64, 3), dtype=np.uint8)
    emb = r.encode(crop)
    ok &= expect(emb.shape == (256,) and emb.dtype == np.float32,
                 f"reid embed shape={emb.shape} dtype={emb.dtype}")
    norm = float(np.linalg.norm(emb))
    ok &= expect(0.99 < norm < 1.01, f"reid embed unit-norm: {norm:.4f}")

    # determinism: same crop -> same embedding
    emb2 = r.encode(crop)
    cos = float(np.dot(emb, emb2))
    ok &= expect(cos > 0.999, f"reid same crop -> cos~1: {cos:.4f}")

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
