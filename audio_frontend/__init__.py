"""
audio_frontend
==============

In-process Python library used by the ROS2 audio_frontend_node + camera_node
to do all DSP/CV heavy lifting. Built around three pluggable interfaces so we
can swap CPU <-> Hailo backends at runtime without changing the node logic.

Submodules:
  backends.denoiser       deep noise suppression (DeepFilterNet / DTLN)
  backends.pose_detector  YOLOv8n-Pose / YOLO11n-Pose
  backends.reid           OSNet body re-identification
"""
