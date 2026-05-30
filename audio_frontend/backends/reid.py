"""
Re-identification embedding backend abstraction.

Output: 256-D unit-norm float32 vector for any HxWx3 person crop. Stage 0
returns deterministic random vectors so cosine matching exercises the wiring
without skewing real similarity values.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ._hailo_probe import probe_hailo

logger = logging.getLogger(__name__)

EMBED_DIM = 256


@runtime_checkable
class ReIDEncoder(Protocol):
    name: str
    embed_dim: int

    def encode(self, crop: np.ndarray) -> np.ndarray:
        """crop: HxWx3 uint8. Returns (embed_dim,) float32 unit-norm."""


def _stub_embed(crop: np.ndarray) -> np.ndarray:
    """Deterministic stub embedding for stage-0 wiring.

    Hashes the crop's down-sampled bytes -> seeds a PRNG -> returns a unit
    vector. Same crop yields same vector; different crops get different
    vectors (cosine similarity ~0 between distinct people, ~1 with itself).
    """
    if crop.size == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    small = crop[::4, ::4]
    h = hashlib.sha1(small.tobytes()).digest()
    seed = int.from_bytes(h[:8], "little", signed=False) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    n = np.linalg.norm(v) + 1e-9
    return v / n


class CpuReIDEncoder:
    def __init__(self, model: str = "osnet_x0_25"):
        self.name = f"cpu/{model}"
        self.embed_dim = EMBED_DIM
        self.model = model

    def encode(self, crop: np.ndarray) -> np.ndarray:
        return _stub_embed(crop)


class HailoReIDEncoder:
    def __init__(self, model: str = "osnet_x0_25",
                 hef_path: Optional[str] = None):
        probe = probe_hailo()
        if not probe.available:
            raise RuntimeError(f"Hailo unavailable: {probe.reason}")
        self.name = f"hailo/{model}"
        self.embed_dim = EMBED_DIM
        self.model = model
        self.hef_path = hef_path

    def encode(self, crop: np.ndarray) -> np.ndarray:
        return _stub_embed(crop)


def select_reid(prefer: str = "auto",
                cpu_model: str = "osnet_x0_25",
                hailo_model: str = "osnet_x0_25",
                hef_path: Optional[str] = None) -> ReIDEncoder:
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        e = CpuReIDEncoder(model=cpu_model)
        logger.info(f"ReIDEncoder: {e.name} (forced cpu)")
        return e
    if prefer == "hailo":
        e = HailoReIDEncoder(model=hailo_model, hef_path=hef_path)
        logger.info(f"ReIDEncoder: {e.name} (forced hailo)")
        return e
    if prefer != "auto":
        raise ValueError(f"unknown prefer={prefer!r}")
    probe = probe_hailo()
    if probe.available:
        try:
            e = HailoReIDEncoder(model=hailo_model, hef_path=hef_path)
            logger.info(f"ReIDEncoder: {e.name} (auto -> hailo)")
            return e
        except Exception as exc:
            logger.warning(f"ReID hailo init failed: {exc}; falling back to cpu")
    e = CpuReIDEncoder(model=cpu_model)
    logger.info(f"ReIDEncoder: {e.name} (auto -> cpu, hailo: {probe.reason})")
    return e
