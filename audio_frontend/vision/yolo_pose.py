"""
CPU YOLO-pose backend (onnxruntime).

Why YOLOv8n-Pose:
  - Single ONNX file, ~12 MB, no torch dependency.
  - 17 COCO keypoints per person — exactly what mouth_estimator needs.
  - On RPi5 4 threads: ~4-6 fps at 640x480 (acceptable for 5 fps target).

Public API (matches the `PoseDetector` protocol in audio_frontend.backends):

  class CpuYoloPose:
    def detect(self, bgr) -> list[dict]
        # each dict: {'bbox': (x0,y0,x1,y1), 'score': float,
        #             'keypoints': (17, 3) array of (x, y, conf)}

The class auto-resolves the model path: explicit > $YOLO_POSE_ONNX env >
``models/yolo/yolov8n-pose.onnx`` > ``models/yolo/yolov11n-pose.onnx``.
If no ONNX is present, ``detect()`` returns an empty list and the
attribute ``self.model_path`` is None.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = [
    REPO_ROOT / "models" / "yolo" / "yolov8n-pose.onnx",
    REPO_ROOT / "models" / "yolo" / "yolov11n-pose.onnx",
    REPO_ROOT / "models" / "yolo" / "yolo11n-pose.onnx",
]


def _resolve_model(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("YOLO_POSE_ONNX")
    if env and Path(env).exists():
        return Path(env)
    for c in DEFAULT_CANDIDATES:
        if c.exists():
            return c
    return None


def _letterbox(img: np.ndarray, target: int = 640
               ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = img.shape[:2]
    s = target / max(h, w)
    new_w, new_h = int(round(w * s)), int(round(h * s))
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("opencv-python is required for letterbox") from e
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target - new_w
    pad_h = target - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, s, (left, top)


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    out = np.zeros_like(box)
    out[..., 0] = cx - w / 2
    out[..., 1] = cy - h / 2
    out[..., 2] = cx + w / 2
    out[..., 3] = cy + h / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float
         ) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        keep_idx = np.where(iou <= iou_thr)[0]
        order = order[keep_idx + 1]
    return np.array(keep, dtype=np.int64)


class CpuYoloPose:
    """YOLO-pose ONNX runner."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 conf_thresh: float = 0.35,
                 iou_thresh: float = 0.55,
                 num_threads: int = 4):
        self.model_path = _resolve_model(model_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.name = "cpu/yolov8n-pose"
        self._sess = None
        self._input_name = None
        self._input_size = 640
        if self.model_path is None:
            logger.warning(
                "no YOLO-pose ONNX found; CpuYoloPose will return empty detections. "
                "Place an ONNX in %s to enable.", DEFAULT_CANDIDATES[0],
            )
            return
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            logger.warning("onnxruntime not installed; pose disabled")
            self.model_path = None
            return
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = int(num_threads)
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=sess_opts,
        )
        self._input_name = self._sess.get_inputs()[0].name
        sh = self._sess.get_inputs()[0].shape
        if isinstance(sh[-1], int) and sh[-1] > 0:
            self._input_size = int(sh[-1])
        logger.info("loaded YOLO-pose ONNX %s (input %dx%d)",
                    self.model_path.name, self._input_size, self._input_size)

    def detect(self, bgr: np.ndarray) -> List[Dict[str, object]]:
        if self._sess is None:
            return []
        img, scale, (pad_x, pad_y) = _letterbox(bgr, self._input_size)
        rgb = img[..., ::-1].astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None]   # NCHW
        out = self._sess.run(None, {self._input_name: x})[0]
        # YOLOv8/11 pose output: (1, 56, 8400) = (1, 4 + 1 + 17*3, N)
        if out.shape[1] == 56:
            preds = out[0].T   # (N, 56)
        elif out.shape[2] == 56:
            preds = out[0]
        else:
            logger.warning("unexpected YOLO-pose output shape %s", out.shape)
            return []
        boxes_xywh = preds[:, :4]
        obj_scores = preds[:, 4]
        kps = preds[:, 5:].reshape(-1, 17, 3)
        keep = obj_scores >= self.conf_thresh
        if not np.any(keep):
            return []
        boxes_xywh = boxes_xywh[keep]
        obj_scores = obj_scores[keep]
        kps = kps[keep]
        boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
        idx = _nms(boxes_xyxy, obj_scores, self.iou_thresh)

        results: List[Dict[str, object]] = []
        H, W = bgr.shape[:2]
        for i in idx:
            b = boxes_xyxy[i].copy()
            b[[0, 2]] = (b[[0, 2]] - pad_x) / scale
            b[[1, 3]] = (b[[1, 3]] - pad_y) / scale
            b[[0, 2]] = np.clip(b[[0, 2]], 0, W - 1)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, H - 1)
            kp = kps[i].copy()
            kp[..., 0] = (kp[..., 0] - pad_x) / scale
            kp[..., 1] = (kp[..., 1] - pad_y) / scale
            results.append({
                "bbox": (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                "score": float(obj_scores[i]),
                "keypoints": kp.astype(np.float32, copy=False),
            })
        return results
