"""
Teach / debug subsystem for the robot client.

子模块：

- ``clip_store``: 动作 clip 的 JSON 持久化（CRUD）。
- ``motion_recorder``: 后台线程，订阅串口的 ctrl_state.tg_v* 录制 50Hz cmd_vel 时序。
- ``motion_player``: 后台线程，按时间戳回放 clip；起止时切换 ``AudioPlayer.set_motion_pending``
  与 ``TrackingController.pause/resume_from_pause``。
- ``action_registry``: 情绪→(clip, expression) 绑定的内存视图，从 ``data/teach_bindings.json`` 加载。
- ``calib_session``: 相机内参标定的可复用核心（同时被 ``camera_calibration.py`` CLI 与 teach server 使用）。
- ``camera_stream``: MJPEG ``multipart/x-mixed-replace`` StreamingResponse。
- ``server``: FastAPI app；提供 REST + ``/ws/state`` + ``/ws/mic`` + 静态 UI。

所有子模块对外都暴露纯 Python class/函数，不在 import 时启动后台线程；启动
由 ``client/main.py`` 显式控制。
"""
