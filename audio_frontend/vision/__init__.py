"""
Vision-side helpers for the far-field audio frontend.

Modules:
  mouth_estimator   3-D mouth-point estimation from keypoints + depth.
  tracker           lightweight IoU-based tracker (ByteTrack-style).
  gallery           session-scoped Re-ID owner gallery with DOA gating.
  yolo_pose         CPU pose estimator (YOLOv8n-Pose ONNX with onnxruntime).
"""
