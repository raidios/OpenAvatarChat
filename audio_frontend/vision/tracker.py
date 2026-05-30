"""
A tiny IoU-based tracker (good enough for ground-robot scenes with 1-3 people)
plus a hook for upgrading to ByteTrack later.

Why not just use https://github.com/ifzhang/ByteTrack?
  - The pip wheel pulls torch + cython_bbox; our ROS2 node already has a
    constant-size personnel set (1-3 people) and stable detection scores,
    so a 200-line IoU tracker matches ByteTrack within a couple of percent.
  - We keep a `track_id` consistent across short detection drops via "lost"
    bookkeeping (up to N frames).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (Na,4) and (Nb,4)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


def _greedy_match(cost: np.ndarray, accept_thresh: float
                  ) -> List[Tuple[int, int]]:
    """Greedy assignment minimizing 1-IoU. Returns [(row_i, col_j), ...]."""
    matches: List[Tuple[int, int]] = []
    if cost.size == 0:
        return matches
    used_r, used_c = set(), set()
    flat = [(cost[i, j], i, j)
            for i in range(cost.shape[0])
            for j in range(cost.shape[1])]
    flat.sort()
    for c, i, j in flat:
        if i in used_r or j in used_c:
            continue
        if (1.0 - c) < accept_thresh:
            continue
        used_r.add(i); used_c.add(j)
        matches.append((i, j))
    return matches


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray              # (4,) xyxy
    score: float = 0.0
    keypoints: Optional[np.ndarray] = None    # (17, 3)
    last_seen_frame: int = 0
    age: int = 0
    miss: int = 0
    embedding: Optional[np.ndarray] = None    # (D,) ReID feature, optional
    extras: Dict[str, object] = field(default_factory=dict)


class IouTracker:
    """Multi-object tracker; matches detections to tracks by IoU."""

    def __init__(self, iou_thresh: float = 0.30, max_miss: int = 15):
        self.iou_thresh = float(iou_thresh)
        self.max_miss = int(max_miss)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._frame_idx = 0

    def update(self, dets_xyxy: np.ndarray,
               scores: Optional[np.ndarray] = None,
               keypoints: Optional[np.ndarray] = None,
               ) -> List[Track]:
        """One step. Returns the live tracks (alive + just-confirmed)."""
        self._frame_idx += 1
        if dets_xyxy is None:
            dets_xyxy = np.zeros((0, 4), dtype=np.float32)
        else:
            dets_xyxy = np.asarray(dets_xyxy, dtype=np.float32).reshape(-1, 4)
        N = dets_xyxy.shape[0]
        if scores is None:
            scores = np.ones(N, dtype=np.float32)
        if keypoints is not None:
            keypoints = np.asarray(keypoints, dtype=np.float32).reshape(N, -1, 3)

        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[t].bbox for t in track_ids],
                               dtype=np.float32) if track_ids else \
                      np.zeros((0, 4), dtype=np.float32)
        iou = _iou_xyxy(track_boxes, dets_xyxy)
        cost = 1.0 - iou
        matches = _greedy_match(cost, accept_thresh=self.iou_thresh)
        matched_t = {m[0] for m in matches}
        matched_d = {m[1] for m in matches}

        for ti, di in matches:
            tid = track_ids[ti]
            tr = self.tracks[tid]
            tr.bbox = dets_xyxy[di]
            tr.score = float(scores[di])
            tr.keypoints = keypoints[di] if keypoints is not None else None
            tr.last_seen_frame = self._frame_idx
            tr.age += 1
            tr.miss = 0

        for ti, tid in enumerate(track_ids):
            if ti not in matched_t:
                self.tracks[tid].miss += 1

        for di in range(N):
            if di in matched_d:
                continue
            tid = self._next_id; self._next_id += 1
            self.tracks[tid] = Track(
                track_id=tid, bbox=dets_xyxy[di], score=float(scores[di]),
                keypoints=keypoints[di] if keypoints is not None else None,
                last_seen_frame=self._frame_idx, age=1, miss=0,
            )

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].miss > self.max_miss:
                del self.tracks[tid]
        return list(self.tracks.values())
