"""
Pluggable backends. Each backend module exposes:
  - a ``Protocol`` describing the runtime contract
  - one ``CpuBackend`` implementation (always available, baseline / fallback)
  - one ``HailoBackend`` implementation (uses /dev/h1x-0 when probable)
  - a ``select_backend(prefer="auto")`` factory that probes Hailo availability
    and returns whichever is requested / probable.

prefer="auto" semantics:
  - try HailoBackend(); on success return it (logged "selected hailo")
  - on failure (driver missing, HEF absent, version mismatch) -> CpuBackend
prefer="cpu"     -> always CpuBackend
prefer="hailo"   -> always HailoBackend (raises if unavailable)
"""
from .denoiser import Denoiser, select_denoiser  # noqa: F401
from .pose_detector import PoseDetector, select_pose_detector  # noqa: F401
from .reid import ReIDEncoder, select_reid  # noqa: F401
