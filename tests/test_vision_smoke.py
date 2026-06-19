#!/usr/bin/env python3
"""Stage 2 vision smoke: synthetic-image pose detect + tracker + mouth + gallery.

Runs entirely on CPU and without any camera/HEF; loads the ONNX from
models/yolo/yolov8n-pose.onnx. If the ONNX is missing, the test marks
the YOLO step SKIP and verifies the rest of the wiring with a synthetic
detection.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO / "tests"]
sys.path.insert(0, str(REPO))

from audio_frontend.vision.gallery import OwnerGallery  # noqa: E402
from audio_frontend.vision.mouth_estimator import (  # noqa: E402
    KP_LSHOULDER, KP_NOSE, KP_RSHOULDER, estimate_mouth_3d,
)
from audio_frontend.vision.tracker import IouTracker, Track  # noqa: E402
from audio_frontend.backends.pose_detector import select_pose_detector  # noqa: E402

_FAILED = []


def expect(cond: bool, msg: str) -> bool:
    print(f"  {'ok' if cond else 'FAIL'} {msg}")
    if not cond:
        _FAILED.append(msg)
    return cond


def main() -> int:
    print("== mouth estimator (synthetic) ==")
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
    H, W = 480, 640
    depth = np.full((H, W), 1500, dtype=np.uint16)        # 1.5 m flat
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[KP_NOSE] = (320, 200, 0.9)
    kp[3] = (300, 200, 0.7)
    kp[4] = (340, 200, 0.7)
    kp[KP_LSHOULDER] = (280, 280, 0.9)
    kp[KP_RSHOULDER] = (360, 280, 0.9)
    bbox = (260, 180, 380, 420)
    est = estimate_mouth_3d(kp, bbox, depth, K)
    expect(est.rule == "nose_face_visible",
           f"face-visible rule (got {est.rule})")
    expect(0.4 < est.point_3d_m[2] < 1.6,
           f"depth ~ 1.5 m (got {est.point_3d_m[2]:.2f})")

    kp[KP_NOSE, 2] = 0.0; kp[3, 2] = 0.0; kp[4, 2] = 0.0
    est2 = estimate_mouth_3d(kp, bbox, depth, K)
    expect(est2.rule == "shoulders_only", f"shoulders fallback (got {est2.rule})")
    expect(est2.pixel_xy[1] < (kp[KP_LSHOULDER, 1] + kp[KP_RSHOULDER, 1]) / 2.0,
           "shoulder->mouth shifts up in image plane")

    print("\n== tracker IoU consistency ==")
    tr = IouTracker(iou_thresh=0.30, max_miss=5)
    box1 = np.array([[100, 100, 200, 300]], dtype=np.float32)
    out1 = tr.update(box1)
    expect(len(out1) == 1, "one track after first frame")
    tid1 = out1[0].track_id
    box2 = np.array([[110, 105, 210, 305]], dtype=np.float32)
    out2 = tr.update(box2)
    expect(any(t.track_id == tid1 for t in out2),
           f"track id stable after small motion (kept {tid1})")
    out_empty = tr.update(np.zeros((0, 4), dtype=np.float32))
    expect(any(t.track_id == tid1 and t.miss == 1 for t in out_empty),
           "track survives 1 missed frame")

    print("\n== owner gallery binding ==")
    g = OwnerGallery(bind_doa_tol_deg=20.0, match_thresh=0.7)
    rng = np.random.default_rng(0)
    emb_a = rng.normal(size=128).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a)
    emb_b = rng.normal(size=128).astype(np.float32)
    emb_b /= np.linalg.norm(emb_b)
    t_a = Track(track_id=1, bbox=np.array([0, 0, 1, 1], dtype=np.float32),
                embedding=emb_a)
    t_b = Track(track_id=2, bbox=np.array([0, 0, 1, 1], dtype=np.float32),
                embedding=emb_b)
    az = {1: 30.0, 2: 200.0}
    bound = g.bind_on_wake([t_a, t_b], doa_deg=25.0, azimuth_of_track=az)
    expect(bound == 1, f"DOA-best track wins binding (got {bound})")
    own = g.step([t_a, t_b])
    expect(own == 1, f"step returns owner = {own} (expect 1)")
    t_b.embedding = emb_a + 0.05 * rng.normal(size=128).astype(np.float32)
    own2 = g.step([t_b])
    expect(own2 == 2, f"owner re-binds when only similar track present (got {own2})")

    print("\n== CPU YOLO-pose load ==")
    det = select_pose_detector(prefer="cpu")
    has_model = "no_model" not in det.name
    print(f"  pose detector = {det.name} (model present: {has_model})")
    if has_model:
        canvas = np.full((480, 640, 3), 100, dtype=np.uint8)
        canvas[:120, :, :] = 30
        canvas[120:330, 220:420, :] = (180, 180, 180)
        canvas[330:, :, :] = 60
        t0 = time.perf_counter()
        ppl = det.detect(canvas)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  inference latency: {dt:.1f} ms; persons: {len(ppl)}")
        expect(dt < 2000, f"pose inference < 2s wall time on synthetic (got {dt:.0f} ms)")

    print("\n" + ("FAIL" if _FAILED else "PASS"))
    return 1 if _FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
