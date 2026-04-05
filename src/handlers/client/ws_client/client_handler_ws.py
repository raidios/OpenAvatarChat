import asyncio
import json
from typing import Dict, Optional, Any, Union, Tuple
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.common.handler_base import HandlerDataInfo, HandlerDetail, HandlerBaseInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalSourceType, ChatSignalType
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, DataBundle, \
    VariableSize

MSG_TYPE_AUDIO = 0x01
MSG_TYPE_VIDEO = 0x02


class WsClientSessionDelegate(ClientSessionDelegate):
    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
        }
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }
        self.quit = asyncio.Event()

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None
        try:
            data = await asyncio.wait_for(data_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None
        return data

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None, samplerate: Optional[int] = None,
                 loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        data_bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            data_bundle.add_meta('human_text_end', True)
            data_bundle.add_meta('speech_id', str(uuid4()))
            data_bundle.set_main_data(data)
        else:
            return
        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        self.data_submitter.submit(chat_data)

    def get_timestamp(self):
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        if signal.source_type == ChatSignalSourceType.CLIENT and signal.type == ChatSignalType.END:
            self.shared_states.enable_vad = True

    def clear_data(self):
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                data_queue.get_nowait()

    async def _ws_send_loop(self, websocket: WebSocket):
        while not self.quit.is_set():
            try:
                chat_data: Optional[ChatData] = await self.get_data(EngineChannelType.AUDIO, timeout=0.05)
                if chat_data is not None and chat_data.data is not None:
                    audio = chat_data.data.get_main_data()
                    if audio is not None:
                        audio = audio.squeeze()
                        pcm_bytes = (audio * 32767).astype(np.int16).tobytes()
                        await websocket.send_bytes(bytes([MSG_TYPE_AUDIO]) + pcm_bytes)

                    speech_end = chat_data.data.get_meta("avatar_speech_end", False)
                    if speech_end:
                        await websocket.send_text(json.dumps({"type": "audio_end"}))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.opt(exception=True).error(f"Error in ws send loop: {e}")
                break

            try:
                text_data: Optional[ChatData] = await self.get_data(EngineChannelType.TEXT, timeout=0.01)
                if text_data is not None and text_data.data is not None:
                    text = text_data.data.get_main_data()
                    if text and isinstance(text, str):
                        msg_type = "asr_text" if text_data.type == ChatDataType.HUMAN_TEXT else "llm_text"
                        await websocket.send_text(json.dumps({"type": msg_type, "text": text}))
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    async def _ws_recv_loop(self, websocket: WebSocket):
        while not self.quit.is_set():
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                self.quit.set()
                break
            except Exception as e:
                logger.opt(exception=True).error(f"Error receiving WS message: {e}")
                self.quit.set()
                break

            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                if len(raw) < 2:
                    continue
                msg_type = raw[0]
                payload = raw[1:]
                if msg_type == MSG_TYPE_AUDIO:
                    audio_int16 = np.frombuffer(payload, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32767.0
                    self.put_data(EngineChannelType.AUDIO, audio_float)
                elif msg_type == MSG_TYPE_VIDEO:
                    frame_array = cv2.imdecode(
                        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame_array is not None:
                        self.put_data(EngineChannelType.VIDEO, frame_array)
            elif "text" in msg and msg["text"]:
                try:
                    ctrl = json.loads(msg["text"])
                    ctrl_type = ctrl.get("type")
                    if ctrl_type == "interrupt":
                        self.clear_data()
                    elif ctrl_type == "end_speech":
                        signal = ChatSignal(
                            source_type=ChatSignalSourceType.CLIENT,
                            type=ChatSignalType.END,
                        )
                        self.emit_signal(signal)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON control message from client")

    async def serve_websocket(self, websocket: WebSocket):
        recv_task = asyncio.create_task(self._ws_recv_loop(websocket))
        send_task = asyncio.create_task(self._ws_send_loop(websocket))
        await asyncio.gather(recv_task, send_task, return_exceptions=True)
        recv_task.cancel()
        send_task.cancel()


class WsClientConfigModel(HandlerBaseConfigModel, BaseModel):
    pass


class WsClientContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[WsClientConfigModel] = None
        self.client_session_delegate: Optional[WsClientSessionDelegate] = None


class ClientHandlerWs(ClientHandlerBase):
    def __init__(self):
        super().__init__()
        self.engine_config = None
        self.handler_config = None
        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=WsClientConfigModel,
            client_session_delegate_class=WsClientSessionDelegate,
        )

    def _prepare_definitions(self):
        audio_def = DataBundleDefinition()
        audio_def.add_entry(DataBundleEntry.create_audio_entry("mic_audio", 1, 16000))
        audio_def.lockdown()
        self.output_bundle_definitions[EngineChannelType.AUDIO] = audio_def

        video_def = DataBundleDefinition()
        video_def.add_entry(DataBundleEntry.create_framed_entry(
            "camera_video", [VariableSize(), VariableSize(), VariableSize(), 3], 0, 1
        ))
        video_def.lockdown()
        self.output_bundle_definitions[EngineChannelType.VIDEO] = video_def

        text_def = DataBundleDefinition()
        text_def.add_entry(DataBundleEntry.create_text_entry("human_text"))
        text_def.lockdown()
        self.output_bundle_definitions[EngineChannelType.TEXT] = text_def

    def load(self, engine_config: ChatEngineConfigModel,
             handler_config: Optional[HandlerBaseConfigModel] = None):
        self.engine_config = engine_config
        self.handler_config = handler_config
        self._prepare_definitions()

    def on_setup_app(self, app: FastAPI, ui: Optional[Any] = None, parent_block: Optional[Any] = None):
        @app.websocket("/ws/chat")
        async def ws_chat(websocket: WebSocket):
            await websocket.accept()
            session_id = str(uuid4())
            logger.info(f"WebSocket client connected, session_id={session_id}")

            try:
                session_delegate = self.handler_delegate.start_session(
                    session_id=session_id, timestamp_base=16000
                )
                await websocket.send_text(json.dumps({
                    "type": "session_started", "session_id": session_id
                }))
                await session_delegate.serve_websocket(websocket)
            except Exception as e:
                logger.opt(exception=True).error(f"WebSocket session error: {e}")
            finally:
                try:
                    self.handler_delegate.stop_session(session_id)
                except Exception:
                    pass
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.close()
                logger.info(f"WebSocket session {session_id} ended")

        @app.get("/")
        async def root():
            return {"status": "ok", "message": "OpenAvatarChat WebSocket Server"}

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        context = WsClientContext(session_context.session_info.session_id)
        context.config = handler_config
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def on_setup_session_delegate(self, session_context: SessionContext, handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
        handler_context = WsClientContext.__cast(handler_context) if False else handler_context
        session_delegate.timestamp_generator = session_context.get_timestamp
        session_delegate.data_submitter = handler_context.data_submitter
        session_delegate.input_data_definitions = self.output_bundle_definitions
        session_delegate.shared_states = session_context.shared_states
        handler_context.client_session_delegate = session_delegate

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        inputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(type=ChatDataType.AVATAR_AUDIO),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(type=ChatDataType.AVATAR_TEXT),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(type=ChatDataType.HUMAN_TEXT),
        }
        outputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
                definition=self.output_bundle_definitions[EngineChannelType.AUDIO],
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
                definition=self.output_bundle_definitions[EngineChannelType.VIDEO],
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=self.output_bundle_definitions[EngineChannelType.TEXT],
            ),
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        ctx = context
        if not hasattr(ctx, 'client_session_delegate') or ctx.client_session_delegate is None:
            return
        channel = inputs.type.channel_type
        data_queue = ctx.client_session_delegate.output_queues.get(channel)
        if data_queue is not None:
            data_queue.put_nowait(inputs)

    def destroy_context(self, context: HandlerContext):
        if hasattr(context, 'client_session_delegate') and context.client_session_delegate is not None:
            context.client_session_delegate.quit.set()
