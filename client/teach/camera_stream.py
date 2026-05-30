"""
MJPEG ``multipart/x-mixed-replace`` 视频流。

浏览器侧用法：``<img src="/api/camera/stream?overlay=1&fps=10">`` 即可直接
显示。每帧之间用 ``--frame\r\n`` 边界分隔；和 Motion JPEG 流的标准实现一致。

关键约束：

* Pi5 上 ``cv2.imencode('.jpg', quality=60)`` 单帧 ~20-40ms（取决于分辨率），
  10fps 时 CPU 占用约 20-40%。fps 上限默认锁到 15；用户可通过 query 参数
  调低保护其它任务。
* 与 chat_client 的 ``video_sender`` 共享同一个 ``SharedCamera``——后者通过
  ``get_frame()`` / ``get_frame_raw()`` 接口加锁拉帧，不会冲突。
* 每个连接独立采样循环、独立 throttle，多浏览器并发也只重叠帧抓取，不重叠
  编码。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

import cv2
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

if TYPE_CHECKING:
    from camera import SharedCamera


_BOUNDARY = "frame"


async def _gen_frames(
    request: Request,
    camera: "SharedCamera",
    overlay: bool,
    fps: float,
    jpeg_quality: int,
):
    interval = 1.0 / max(1.0, fps)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    while True:
        if await request.is_disconnected():
            return
        frame, _ = camera.get_frame() if overlay else camera.get_frame_raw()
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            await asyncio.sleep(0.05)
            continue
        payload = jpg.tobytes()
        # 这里手动拼边界比 web framework 提供的 `multipart` 更直观（也是 Motion
        # JPEG 的事实标准）。注意：行结束符必须是 CRLF，否则部分浏览器会拒
        # 显示。
        chunk = (
            b"--" + _BOUNDARY.encode("ascii") + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n"
            b"\r\n" + payload + b"\r\n"
        )
        yield chunk
        await asyncio.sleep(interval)


def make_mjpeg_response(
    request: Request,
    camera: "SharedCamera",
    overlay: bool = False,
    fps: float = 10.0,
    jpeg_quality: int = 60,
) -> StreamingResponse:
    # 限幅：超出 15fps 拒绝（保护 Pi5 CPU 与其它任务）。
    fps = max(1.0, min(15.0, float(fps)))
    return StreamingResponse(
        _gen_frames(request, camera, overlay, fps, jpeg_quality),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )


def snapshot_jpeg(
    camera: "SharedCamera",
    overlay: bool = False,
    jpeg_quality: int = 80,
) -> Optional[bytes]:
    frame, _ = camera.get_frame() if overlay else camera.get_frame_raw()
    if frame is None:
        return None
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return jpg.tobytes()


def make_snapshot_response(
    camera: "SharedCamera",
    overlay: bool = False,
    jpeg_quality: int = 80,
) -> Response:
    payload = snapshot_jpeg(camera, overlay=overlay, jpeg_quality=jpeg_quality)
    if payload is None:
        return Response(status_code=503, content=b"camera not ready")
    return Response(content=payload, media_type="image/jpeg")
