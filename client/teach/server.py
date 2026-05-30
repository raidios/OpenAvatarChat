"""
Teach / debug FastAPI 服务。

跟 chat_client 在同一进程的 asyncio loop 上运行（``main.py`` 用 Uvicorn task
和 chat tasks 一起 gather）。所有共享资源通过 ``app.state`` 注入，避免全局
变量；这样 lifetime 完全由 ``main.py`` 控制。

期望 ``app.state`` 上的属性：

- ``camera``: ``SharedCamera``
- ``serial``: ``XProtocolSerial``
- ``tracking_ctl``: ``TrackingController``（可选；None 时禁用 track 相关 api）
- ``motion_recorder``: ``MotionRecorder``
- ``motion_player``: ``MotionPlayer``
- ``clip_store``: ``ClipStore``
- ``action_registry``: ``ActionRegistry``
- ``expression_state``: ``ExpressionUiState``（可选；None 时禁用 expression api）
- ``chat_client``: 可能 None（用于查 chat 链路状态、暴露 mic_monitor）
- ``calib_output_path``: ``Path`` —— 标定结果落盘路径
- ``loop``: ``asyncio.AbstractEventLoop`` —— 主事件循环句柄（供后台线程
  call_soon_threadsafe 用）

各路接口的实现里面没有进一步的同步层；底层数据结构（MotionRecorder /
MotionPlayer / ClipStore / ActionRegistry / CalibSession / MicMonitor）都
是线程安全的。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .calib_session import CalibSession
from .camera_stream import make_mjpeg_response, make_snapshot_response
from .clip_store import ClipSample


# ----------------------------------------------------------------------
# request / response models
# ----------------------------------------------------------------------

class SaveClipBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    crop_start_ms: int = Field(default=0, ge=0)
    crop_end_ms: Optional[int] = Field(default=None, ge=0)


class RenameClipBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class BindingBody(BaseModel):
    clip_id: Optional[str] = None
    expression_id: Optional[str] = None


class SayBody(BaseModel):
    text: str = Field(..., min_length=1)


class CalibStartBody(BaseModel):
    rows: int = Field(default=9, ge=3, le=30)
    cols: int = Field(default=6, ge=3, le=30)
    square_mm: float = Field(default=25.0, gt=0.0, le=200.0)
    min_frames: int = Field(default=15, ge=3, le=200)
    auto_interval_s: float = Field(default=1.0, ge=0.1, le=10.0)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _telemetry_snapshot(state) -> Dict[str, Any]:
    serial = state.serial
    mcu = serial.latest_data if serial is not None else None
    ps2 = serial.latest_ps2 if serial is not None else None
    ctrl = serial.latest_ctrl_state if serial is not None else None
    out: Dict[str, Any] = {}
    if mcu is not None:
        out["mcu"] = {
            "acc": [mcu.acc_x, mcu.acc_y, mcu.acc_z],
            "gyro": [mcu.gyro_x, mcu.gyro_y, mcu.gyro_z],
            "vel": [mcu.vel_x, mcu.vel_y, mcu.vel_w],
            "bat_voltage": mcu.bat_voltage / 100.0,
            "yaw_deg": serial.yaw_deg,
            "yaw_deg_unwrapped": serial.yaw_deg_unwrapped,
            "gyro_z_bias_dps": serial.gyro_z_bias_dps,
            "timestamp": mcu.timestamp,
        }
    if ps2 is not None:
        out["ps2"] = {
            "mode": ps2.mode,
            "btn1": ps2.btn1,
            "btn2": ps2.btn2,
            "rjoy_lr": ps2.rjoy_lr,
            "rjoy_ud": ps2.rjoy_ud,
            "ljoy_lr": ps2.ljoy_lr,
            "ljoy_ud": ps2.ljoy_ud,
        }
    if ctrl is not None:
        out["ctrl"] = {
            "source": ctrl.source,
            "source_name": ctrl.source_name,
            "flags": ctrl.flags,
            "flags_str": ctrl.flags_str(),
            "tg_vx": ctrl.tg_vx,
            "tg_vy": ctrl.tg_vy,
            "tg_vw": ctrl.tg_vw,
        }
    return out


def _full_status(state) -> Dict[str, Any]:
    recorder = state.motion_recorder
    player = state.motion_player
    chat = state.chat_client
    expr = state.expression_state
    s: Dict[str, Any] = {
        "telemetry": _telemetry_snapshot(state),
        "recorder": recorder.status().__dict__ if recorder is not None else None,
        "player": player.status() if player is not None else None,
        "chat_connected": bool(chat is not None and chat.player.is_playing or (expr and expr.snapshot()[0])),
    }
    if expr is not None:
        connected, wake, vad = expr.snapshot()
        s["chat"] = {
            "connected": connected,
            "wake_session": wake,
            "vad_enabled": vad,
            "override_expression": expr.override_snapshot(),
        }
    calib = getattr(state, "calib_session", None)
    if calib is not None:
        s["calib"] = calib.snapshot().__dict__
    else:
        s["calib"] = None
    return s


# ----------------------------------------------------------------------
# app factory
# ----------------------------------------------------------------------

def create_app(static_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="OpenAvatarChat Teach UI")

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def get_status(request: Request):
        return JSONResponse(_full_status(request.app.state))

    # ------------------------------------------------------------------
    # clips
    # ------------------------------------------------------------------

    @app.get("/api/clips")
    async def list_clips(request: Request):
        store = request.app.state.clip_store
        return [
            {
                "id": c.id,
                "name": c.name,
                "duration_ms": c.duration_ms,
                "created_at": c.created_at,
                "sample_count": len(c.samples),
            }
            for c in store.list()
        ]

    @app.get("/api/clips/{clip_id}")
    async def get_clip(clip_id: str, request: Request):
        store = request.app.state.clip_store
        clip = store.get(clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        return clip.to_dict()

    @app.post("/api/clips/record/start")
    async def record_start(request: Request):
        recorder = request.app.state.motion_recorder
        if recorder is None:
            raise HTTPException(status_code=503, detail="recorder not available")
        recorder.start_recording()
        return {"ok": True, "status": recorder.status().__dict__}

    @app.post("/api/clips/record/stop")
    async def record_stop(request: Request):
        recorder = request.app.state.motion_recorder
        if recorder is None:
            raise HTTPException(status_code=503, detail="recorder not available")
        recorder.stop_recording()
        return {"ok": True, "status": recorder.status().__dict__}

    @app.get("/api/clips/record/preview")
    async def record_preview(request: Request):
        recorder = request.app.state.motion_recorder
        if recorder is None:
            raise HTTPException(status_code=503, detail="recorder not available")
        samples = recorder.snapshot()
        return {
            "duration_ms": samples[-1].t_ms if samples else 0,
            "sample_count": len(samples),
            "samples": [
                {"t_ms": s.t_ms, "vx": s.vx, "vy": s.vy, "vw": s.vw}
                for s in samples
            ],
        }

    @app.post("/api/clips/record/discard")
    async def record_discard(request: Request):
        recorder = request.app.state.motion_recorder
        if recorder is None:
            raise HTTPException(status_code=503, detail="recorder not available")
        recorder.discard()
        return {"ok": True}

    @app.post("/api/clips/save")
    async def save_clip(body: SaveClipBody, request: Request):
        state = request.app.state
        recorder = state.motion_recorder
        store = state.clip_store
        if recorder is None or store is None:
            raise HTTPException(status_code=503, detail="recorder/store not available")
        try:
            clip = recorder.save_clip(
                store,
                name=body.name,
                crop_start_ms=body.crop_start_ms,
                crop_end_ms=body.crop_end_ms,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "ok": True,
            "id": clip.id,
            "name": clip.name,
            "duration_ms": clip.duration_ms,
        }

    @app.post("/api/clips/{clip_id}/play")
    async def play_clip(clip_id: str, request: Request):
        store = request.app.state.clip_store
        player = request.app.state.motion_player
        if player is None:
            raise HTTPException(status_code=503, detail="motion player not available")
        clip = store.get(clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        player.enqueue(clip)
        return {"ok": True}

    @app.post("/api/clips/stop")
    async def stop_clip(request: Request):
        player = request.app.state.motion_player
        if player is None:
            raise HTTPException(status_code=503, detail="motion player not available")
        player.stop_all()
        return {"ok": True}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str, request: Request):
        store = request.app.state.clip_store
        ok = store.delete(clip_id)
        if not ok:
            raise HTTPException(status_code=404, detail="clip not found")
        return {"ok": True}

    @app.patch("/api/clips/{clip_id}")
    async def rename_clip(clip_id: str, body: RenameClipBody, request: Request):
        store = request.app.state.clip_store
        c = store.rename(clip_id, body.name)
        if c is None:
            raise HTTPException(status_code=404, detail="clip not found")
        return {"ok": True, "id": c.id, "name": c.name}

    # ------------------------------------------------------------------
    # bindings
    # ------------------------------------------------------------------

    @app.get("/api/bindings")
    async def get_bindings(request: Request):
        reg = request.app.state.action_registry
        snap = reg.snapshot()
        return {k: {"clip_id": v.clip_id, "expression_id": v.expression_id} for k, v in snap.items()}

    @app.post("/api/bindings/{key}")
    async def set_binding(key: str, body: BindingBody, request: Request):
        reg = request.app.state.action_registry
        # 允许把 clip_id 主动设为空字符串以解绑 clip。
        clip_id: Optional[str] = body.clip_id
        if clip_id == "":
            clip_id = None
        b = reg.update(key, clip_id=clip_id, expression_id=body.expression_id)
        if b is None:
            raise HTTPException(status_code=400, detail=f"unknown emotion key: {key}")
        reg.save()
        return {"key": key, "clip_id": b.clip_id, "expression_id": b.expression_id}

    @app.post("/api/bindings/{key}/trigger")
    async def trigger_binding(key: str, request: Request):
        dispatcher = getattr(request.app.state, "dispatcher", None)
        if dispatcher is None:
            raise HTTPException(status_code=503, detail="dispatcher not available")
        dispatcher.dispatch([{"name": key}], phase="before")
        return {"ok": True}

    # ------------------------------------------------------------------
    # expression debug
    # ------------------------------------------------------------------

    @app.post("/api/expression/{eid}")
    async def override_expression(eid: str, request: Request, hold_ms: int = 3000):
        expr = request.app.state.expression_state
        if expr is None:
            raise HTTPException(status_code=503, detail="expression state not available")
        expr.set_override(eid, hold_ms=hold_ms)
        return {"ok": True, "expression_id": eid, "hold_ms": hold_ms}

    @app.delete("/api/expression")
    async def clear_expression(request: Request):
        expr = request.app.state.expression_state
        if expr is None:
            raise HTTPException(status_code=503, detail="expression state not available")
        expr.clear_override()
        return {"ok": True}

    # ------------------------------------------------------------------
    # say (debug)
    # ------------------------------------------------------------------

    @app.post("/api/say")
    async def debug_say(body: SayBody, request: Request):
        dispatcher = getattr(request.app.state, "dispatcher", None)
        if dispatcher is None:
            raise HTTPException(status_code=503, detail="dispatcher not available")
        # 直接走标签解析 + 派发，不经过服务端：
        # 1) 用 action_tags 解析出情绪 key
        # 2) 把它们逐个 dispatch 到本地的 motion + expression 通道
        # 这样不需要 chat 链路也能调试 4 个情绪的端到端效果。
        from handlers.common.action_tags import extract_action_tags
        try:
            _clean, tags = extract_action_tags(body.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not tags:
            return {"ok": True, "dispatched": []}
        payload = [{"name": t.name} for t in tags]
        dispatcher.dispatch(payload, phase="before")
        return {"ok": True, "dispatched": payload}

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------

    @app.get("/api/camera/snapshot")
    async def camera_snapshot(request: Request, overlay: int = 0):
        camera = request.app.state.camera
        if camera is None or not camera.is_opened:
            raise HTTPException(status_code=503, detail="camera not available")
        return make_snapshot_response(camera, overlay=bool(overlay))

    @app.get("/api/camera/stream")
    async def camera_stream(request: Request, overlay: int = 0, fps: float = 10.0):
        camera = request.app.state.camera
        if camera is None or not camera.is_opened:
            raise HTTPException(status_code=503, detail="camera not available")
        return make_mjpeg_response(request, camera, overlay=bool(overlay), fps=fps)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    @app.get("/api/calib/current")
    async def calib_current(request: Request):
        path = getattr(request.app.state, "calib_output_path", None)
        if path is None:
            raise HTTPException(status_code=503, detail="calib_output_path not configured")
        if not path.is_file():
            return {"exists": False, "path": str(path)}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"exists": True, "path": str(path), "data": data}

    @app.post("/api/calib/start")
    async def calib_start(body: CalibStartBody, request: Request):
        state = request.app.state
        existing = getattr(state, "calib_session", None)
        if existing is not None and not existing.is_finalized():
            raise HTTPException(status_code=409, detail="calibration already running")
        sess = CalibSession(
            rows=body.rows,
            cols=body.cols,
            square_mm=body.square_mm,
            min_frames=body.min_frames,
            auto_interval_s=body.auto_interval_s,
            output_path=state.calib_output_path,
        )
        sess.start()
        state.calib_session = sess
        # 起一个后台 task，把摄像头 raw frame 喂给 session。
        cancel = asyncio.Event()
        state.calib_cancel = cancel
        state.calib_task = asyncio.create_task(_calib_feed_loop(state, sess, cancel))
        return {"ok": True, "snapshot": sess.snapshot().__dict__}

    @app.post("/api/calib/capture")
    async def calib_capture(request: Request):
        sess: Optional[CalibSession] = getattr(request.app.state, "calib_session", None)
        if sess is None or sess.is_finalized():
            raise HTTPException(status_code=409, detail="no calibration in progress")
        camera = request.app.state.camera
        if camera is None or not camera.is_opened:
            raise HTTPException(status_code=503, detail="camera not available")
        frame, _ = camera.get_frame_raw()
        if frame is None:
            raise HTTPException(status_code=503, detail="camera frame not ready")
        ok = sess.feed_and_subpix(frame)
        return {"ok": bool(ok), "snapshot": sess.snapshot().__dict__}

    @app.post("/api/calib/finish")
    async def calib_finish(request: Request):
        state = request.app.state
        sess: Optional[CalibSession] = getattr(state, "calib_session", None)
        if sess is None:
            raise HTTPException(status_code=409, detail="no calibration session")
        # 通知 feed loop 退出
        cancel = getattr(state, "calib_cancel", None)
        if cancel is not None:
            cancel.set()
        result = sess.finalize()
        snap = sess.snapshot()
        return {
            "ok": result is not None,
            "snapshot": snap.__dict__,
            "result": result.__dict__ if result is not None else None,
        }

    @app.post("/api/calib/abort")
    async def calib_abort(request: Request):
        state = request.app.state
        sess: Optional[CalibSession] = getattr(state, "calib_session", None)
        if sess is None:
            return {"ok": True, "noop": True}
        cancel = getattr(state, "calib_cancel", None)
        if cancel is not None:
            cancel.set()
        sess.abort()
        return {"ok": True, "snapshot": sess.snapshot().__dict__}

    # ------------------------------------------------------------------
    # /ws/state
    # ------------------------------------------------------------------

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket):
        await websocket.accept()
        state = websocket.app.state
        last_status_json: Optional[str] = None
        try:
            while True:
                snap = _full_status(state)
                payload = {"type": "state", "ts": time.time(), **snap}
                serialized = json.dumps(payload, ensure_ascii=False, default=str)
                if serialized != last_status_json:
                    await websocket.send_text(serialized)
                    last_status_json = serialized
                await asyncio.sleep(0.2)  # 5 Hz
        except WebSocketDisconnect:
            return
        except Exception as e:
            print(f"[teach.ws_state] error: {e}", flush=True)

    # ------------------------------------------------------------------
    # /ws/mic
    # ------------------------------------------------------------------

    @app.websocket("/ws/mic")
    async def ws_mic(websocket: WebSocket):
        state = websocket.app.state
        chat = getattr(state, "chat_client", None)
        if chat is None:
            await websocket.close(code=1011, reason="chat not running")
            return
        monitor = chat.mic_monitor
        if monitor is None:
            await websocket.close(code=1011, reason="mic monitor not available")
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        q = monitor.subscribe(loop)
        # 元数据
        await websocket.send_text(json.dumps({
            "type": "mic_info",
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm_s16le",
            "chunk_ms": 100,
        }))
        last_stat = time.monotonic()
        sum_sq = 0.0
        n_samples = 0
        peak = 0
        try:
            while True:
                pcm = await q.get()
                await websocket.send_bytes(pcm)
                # 顺手算 rms / peak：用 memoryview 解 int16 减少 numpy 开销
                import numpy as np
                arr = np.frombuffer(pcm, dtype=np.int16)
                if arr.size:
                    sum_sq += float(np.sum(arr.astype(np.float32) ** 2))
                    n_samples += arr.size
                    p = int(np.max(np.abs(arr)))
                    if p > peak:
                        peak = p
                now = time.monotonic()
                if now - last_stat >= 1.0:
                    rms = int((sum_sq / n_samples) ** 0.5) if n_samples else 0
                    peak_dbfs = (
                        20.0 * np.log10(max(1e-6, peak / 32768.0)) if peak else float("-inf")
                    )
                    muted = bool(chat.player.mic_should_mute)
                    await websocket.send_text(json.dumps({
                        "type": "mic_stats",
                        "rms": rms,
                        "peak_dbfs": None if peak_dbfs == float("-inf") else peak_dbfs,
                        "muted": muted,
                    }))
                    last_stat = now
                    sum_sq = 0.0
                    n_samples = 0
                    peak = 0
        except WebSocketDisconnect:
            return
        except Exception as e:
            print(f"[teach.ws_mic] error: {e}", flush=True)
        finally:
            monitor.unsubscribe(q)

    # ------------------------------------------------------------------
    # static UI
    # ------------------------------------------------------------------

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


# ----------------------------------------------------------------------
# background tasks
# ----------------------------------------------------------------------

async def _calib_feed_loop(state, sess: CalibSession, cancel: asyncio.Event) -> None:
    """50ms 一拍喂 raw 帧给 CalibSession，直到 finalize / abort / 数量达成。"""
    try:
        while not cancel.is_set() and not sess.is_finalized():
            camera = state.camera
            if camera is not None and camera.is_opened:
                frame, _ = camera.get_frame_raw()
                if frame is not None:
                    sess.feed_frame(frame)
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        pass
