"""
Person + pose detector backend abstraction.

Output contract is COCO 17-keypoints + bounding box. Stage 0 returns empty
detections so the wiring works end-to-end without pulling 50MB+ of model
files. Stage 2A swaps in YOLOv8n-Pose ONNX (CPU). Stage 2B swaps in YOLO11n-Pose
HEF (Hailo).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ._hailo_probe import probe_hailo

logger = logging.getLogger(__name__)


@dataclass
class Person:
    bbox_xyxy: Tuple[float, float, float, float]
    keypoints: np.ndarray   # (17, 3) -> (x, y, conf)
    confidence: float       # whole-person score
    track_id: Optional[int] = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class PoseDetector(Protocol):
    name: str
    input_size: Tuple[int, int]  # (h, w)

    def detect(self, rgb: np.ndarray) -> List[Person]:
        """rgb: HxWx3 uint8 BGR or RGB (model decides).

        Returns 0+ Person objects. Stage 0 returns []."""


class CpuPoseDetector:
    """Stage-2A: YOLOv8n-Pose via onnxruntime. Falls back to empty list when
    no ONNX model is present so node wiring keeps working in CI."""

    def __init__(self, model: str = "yolov8n_pose",
                 model_path: Optional[str] = None,
                 conf_thresh: float = 0.35,
                 iou_thresh: float = 0.55,
                 num_threads: int = 4):
        from ..vision.yolo_pose import CpuYoloPose
        self._impl = CpuYoloPose(
            model_path=model_path, conf_thresh=conf_thresh,
            iou_thresh=iou_thresh, num_threads=num_threads,
        )
        self.input_size = (self._impl._input_size, self._impl._input_size)
        self.model = model
        self.name = self._impl.name if self._impl.model_path else f"cpu/{model}_no_model"

    def detect(self, bgr: np.ndarray) -> List[Person]:
        out = []
        for d in self._impl.detect(bgr):
            out.append(Person(
                bbox_xyxy=tuple(d["bbox"]),
                keypoints=d["keypoints"],
                confidence=float(d["score"]),
            ))
        return out


class HailoPoseDetector:
    """Hailo YOLO11n-Pose. HEF must be supplied; loader is lazy so the import
    succeeds even when the HEF hasn't landed yet."""

    def __init__(self, model: str = "yolo11n_pose",
                 hef_path: Optional[str] = None,
                 conf_thresh: float = 0.35,
                 iou_thresh: float = 0.55):
        probe = probe_hailo()
        if not probe.available:
            raise RuntimeError(f"Hailo unavailable: {probe.reason}")
        self.name = f"hailo/{model}"
        self.input_size = (640, 640)
        self.model = model
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        from pathlib import Path as _P
        from ..vision.yolo_pose import REPO_ROOT as _R
        candidates = (
            [_P(hef_path)] if hef_path else
            [_R / "models" / "hailo" / "yolo11n-pose_h10h.hef",
             _R / "models" / "hailo" / "yolov11n-pose_h10h.hef"]
        )
        self.hef_path = next((p for p in candidates if p.exists()), None)
        self._sess = None
        if self.hef_path is not None:
            try:
                self._init_hef()
            except Exception as e:
                logger.warning(f"HailoPose HEF load failed ({e}); will return empty detections")
                self._sess = None

    def _init_hef(self) -> None:
        from hailo_platform import (HEF, ConfigureParams, FormatType,
                                     HailoStreamInterface, InferVStreams,
                                     InputVStreamParams, OutputVStreamParams,
                                     VDevice)
        self._vdev = VDevice()
        self._hef = HEF(str(self.hef_path))
        cfg = ConfigureParams.create_from_hef(
            hef=self._hef, interface=HailoStreamInterface.PCIe)
        self._ng = self._vdev.configure(self._hef, cfg)[0]
        self._in_params = InputVStreamParams.make_from_network_group(
            self._ng, format_type=FormatType.FLOAT32)
        self._out_params = OutputVStreamParams.make_from_network_group(
            self._ng, format_type=FormatType.FLOAT32)

    def detect(self, bgr: np.ndarray) -> List[Person]:
        # Decoding YOLO11-pose output requires the same NMS/letterbox pipeline
        # as the CPU path; keep the door open for that wiring once the HEF
        # lands so we can plug it in without restructuring this class.
        return []


def select_pose_detector(prefer: str = "auto",
                         cpu_model: str = "yolov8n_pose",
                         hailo_model: str = "yolo11n_pose",
                         hef_path: Optional[str] = None) -> PoseDetector:
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        d = CpuPoseDetector(model=cpu_model)
        logger.info(f"PoseDetector: {d.name} (forced cpu)")
        return d
    if prefer == "hailo":
        d = HailoPoseDetector(model=hailo_model, hef_path=hef_path)
        logger.info(f"PoseDetector: {d.name} (forced hailo)")
        return d
    if prefer != "auto":
        raise ValueError(f"unknown prefer={prefer!r}")
    probe = probe_hailo()
    if probe.available:
        try:
            d = HailoPoseDetector(model=hailo_model, hef_path=hef_path)
            logger.info(f"PoseDetector: {d.name} (auto -> hailo)")
            return d
        except Exception as e:
            logger.warning(f"PoseDetector hailo init failed: {e}; falling back to cpu")
    d = CpuPoseDetector(model=cpu_model)
    logger.info(f"PoseDetector: {d.name} (auto -> cpu, hailo: {probe.reason})")
    return d
