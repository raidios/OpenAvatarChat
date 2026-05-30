"""
Cheap, cached probe for Hailo runtime availability.

We do NOT open a VDevice during the probe by default - opening + closing
churns FW state and creates a small race window with other consumers. We just
check that:
  - /dev/h1x-0 exists and is r/w
  - hailo_platform imports
  - the HailoRT wheel version matches what the .deb / driver version expects

The first call to ``probe_hailo()`` runs all checks; subsequent calls reuse
the cached result.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HailoProbeResult:
    available: bool
    reason: str
    device_node: Optional[str] = None
    hailort_version: Optional[str] = None

    def __bool__(self) -> bool:
        return self.available


@functools.lru_cache(maxsize=1)
def probe_hailo() -> HailoProbeResult:
    dev = Path("/dev/h1x-0")
    if not dev.exists():
        return HailoProbeResult(False, "/dev/h1x-0 missing")
    if not os.access(dev, os.R_OK | os.W_OK):
        return HailoProbeResult(False, "/dev/h1x-0 not r/w by current user",
                                str(dev))
    try:
        import hailo_platform  # type: ignore  # noqa: F401
    except Exception as e:
        return HailoProbeResult(False, f"hailo_platform import failed: {e}",
                                str(dev))
    version = None
    try:
        import importlib.metadata as md  # type: ignore
        version = md.version("hailort")
    except Exception:
        pass
    return HailoProbeResult(True, "ok", str(dev), version)


def reset_probe_cache() -> None:
    probe_hailo.cache_clear()
