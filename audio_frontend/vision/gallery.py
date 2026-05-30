"""
Owner Re-ID gallery.

Lifecycle:
  - Empty until KWS fires.
  - On KWS, the gallery binds to the track whose mouth-3D azimuth best
    matches the audio DOA (within `bind_doa_tol_deg`). Snapshots of that
    track's ReID embeddings are stored.
  - On every subsequent frame, all live tracks are scored against the
    gallery via cosine similarity. The owner gets the track with the best
    score (must clear `match_thresh`).
  - The gallery decays over time: snapshots older than `snapshot_ttl_s`
    are pruned, and if the owner is unseen for `lost_ttl_s` we drop them.

The whole thing is intentionally pure-Python; the tight loop cost is
~50 us / frame for ≤ 10 snapshots.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _Snap:
    embedding: np.ndarray
    timestamp: float


@dataclass
class Gallery:
    snapshots: List[_Snap] = field(default_factory=list)
    track_id: Optional[int] = None
    bound_at: float = 0.0
    last_seen: float = 0.0


class OwnerGallery:
    """Manages a single 'owner' identity across frames."""

    def __init__(self,
                 max_snapshots: int = 8,
                 snapshot_ttl_s: float = 60.0,
                 lost_ttl_s: float = 8.0,
                 bind_doa_tol_deg: float = 25.0,
                 match_thresh: float = 0.55,
                 update_cooldown_s: float = 0.5):
        self.max_snapshots = int(max_snapshots)
        self.snapshot_ttl_s = float(snapshot_ttl_s)
        self.lost_ttl_s = float(lost_ttl_s)
        self.bind_doa_tol_deg = float(bind_doa_tol_deg)
        self.match_thresh = float(match_thresh)
        self.update_cooldown_s = float(update_cooldown_s)
        self.gallery = Gallery()
        self._last_update = 0.0

    def is_bound(self) -> bool:
        return self.gallery.track_id is not None and bool(self.gallery.snapshots)

    def reset(self) -> None:
        self.gallery = Gallery()
        self._last_update = 0.0

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 0 or nb <= 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _prune_snapshots(self, now: float) -> None:
        self.gallery.snapshots = [
            s for s in self.gallery.snapshots
            if now - s.timestamp <= self.snapshot_ttl_s
        ]

    def bind_on_wake(self,
                     tracks: List["TrackLike"],            # noqa: F821 - duck type
                     doa_deg: float,
                     azimuth_of_track: Dict[int, float],
                     ) -> Optional[int]:
        """Pick the live track best matching `doa_deg` and bind it."""
        if not tracks:
            return None
        best = None
        best_err = self.bind_doa_tol_deg
        for t in tracks:
            az = azimuth_of_track.get(t.track_id)
            if az is None:
                continue
            err = abs((az - doa_deg + 540.0) % 360.0 - 180.0)
            if err < best_err:
                best, best_err = t, err
        if best is None:
            return None
        now = time.monotonic()
        self.gallery = Gallery(
            snapshots=[_Snap(np.asarray(best.embedding, dtype=np.float32), now)]
            if best.embedding is not None else [],
            track_id=int(best.track_id),
            bound_at=now,
            last_seen=now,
        )
        self._last_update = now
        return int(best.track_id)

    def step(self, tracks: List["TrackLike"]) -> Optional[int]:  # noqa: F821
        """Score and pick the owner among `tracks`. Returns the owner track id
        or None if no convincing match. Also self-cleans the gallery.
        """
        now = time.monotonic()
        if not self.gallery.snapshots:
            return None

        self._prune_snapshots(now)
        if now - self.gallery.last_seen > self.lost_ttl_s and self.gallery.track_id is not None:
            self.gallery.track_id = None  # owner lost; keep snapshots

        if not tracks:
            return self.gallery.track_id

        scores: List[Tuple[int, float, np.ndarray]] = []
        for t in tracks:
            if t.embedding is None:
                continue
            best = max((self._cos(t.embedding, s.embedding)
                        for s in self.gallery.snapshots), default=0.0)
            scores.append((int(t.track_id), best, t.embedding))
        if not scores:
            return self.gallery.track_id

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score, best_emb = scores[0]
        if best_score >= self.match_thresh:
            self.gallery.track_id = best_id
            self.gallery.last_seen = now
            if (now - self._last_update >= self.update_cooldown_s
                    and best_score < 0.95):
                self.gallery.snapshots.append(_Snap(best_emb.copy(), now))
                if len(self.gallery.snapshots) > self.max_snapshots:
                    self.gallery.snapshots.pop(0)
                self._last_update = now
            return best_id
        return self.gallery.track_id  # keep stale id (might come back)
