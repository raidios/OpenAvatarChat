"""
CPU deep-noise-suppression wrapper (DeepFilterNet 3 + passthrough fallback).

Adapter signature matches ``audio_frontend.backends.denoiser.Denoiser``:
  process(frame: float32[N]) -> float32[N]   # N = block_size

Model resolution order (first match wins):
  1. explicit ``model_base_dir`` argument
  2. env var DF_MODEL_DIR
  3. ``$REPO/models/dfnet3/DeepFilterNet3``  (recommended; pre-downloaded)
  4. default DfNet 0.5.6 download path (fetches from github -> may fail in CN)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL = REPO_ROOT / "models" / "dfnet3" / "DeepFilterNet3"


def _resolve_model_dir(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get("DF_MODEL_DIR")
    if env:
        return env
    if DEFAULT_LOCAL_MODEL.exists():
        return str(DEFAULT_LOCAL_MODEL)
    return None  # let DfNet try its default download URL


class DfDenoiser:
    """DeepFilterNet 3 wrapper at 16 kHz."""

    def __init__(self, block_size: int = 480,
                 model_base_dir: Optional[str] = None):
        try:
            from df.enhance import enhance, init_df    # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"DeepFilterNet not installed: {exc}. "
                "Install with `uv pip install deepfilternet`."
            ) from exc
        self.sample_rate = SAMPLE_RATE
        self.block_size = block_size
        self.name = "cpu/deepfilternet3"
        self._enhance = enhance
        mdir = _resolve_model_dir(model_base_dir)
        try:
            if mdir:
                self.model, self.df_state, _ = init_df(
                    model_base_dir=mdir, post_filter=True, log_file=None,
                )
            else:
                self.model, self.df_state, _ = init_df(
                    post_filter=True, log_file=None,
                )
        except Exception as exc:
            raise RuntimeError(
                f"DfNet init_df failed: {exc}. "
                f"Try pre-downloading DeepFilterNet3.zip into {DEFAULT_LOCAL_MODEL.parent} "
                "or set DF_MODEL_DIR."
            ) from exc

    def reset(self) -> None:
        # df handles state internally per call; no explicit reset needed for
        # short blocks (each block is independent).
        pass

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        # df expects (B, T) tensor at 48k; it handles resampling internally,
        # but we feed 16k since the model is 16k-trained as well.
        import torch
        x = torch.from_numpy(frame[None, :])
        y = self._enhance(self.model, self.df_state, x)
        if hasattr(y, "cpu"):
            y = y.cpu().numpy()
        return y.squeeze().astype(np.float32)


def make_cpu_denoiser(block_size: int = 480, prefer_passthrough: bool = False):
    """Factory: try DfDenoiser; if missing or prefer_passthrough -> stub."""
    if prefer_passthrough:
        from ..backends.denoiser import CpuDenoiser
        return CpuDenoiser(block_size=block_size)
    try:
        return DfDenoiser(block_size=block_size)
    except RuntimeError as e:
        logger.warning(f"DeepFilterNet unavailable: {e}; using passthrough")
        from ..backends.denoiser import CpuDenoiser
        return CpuDenoiser(block_size=block_size)
