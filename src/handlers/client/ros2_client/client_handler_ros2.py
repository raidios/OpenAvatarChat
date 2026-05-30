"""
ROS2 client handler for OpenAvatarChat.

Bridges audio frontend ROS2 topics into the existing chat-engine pipeline,
in place of `ws_client` for headless robot deployments.

Subscriptions (all in `audio_frontend_node`'s namespace by default):
  /audio/clean         std_msgs/Float32MultiArray  16kHz mono float32 in [-1,1]
                                                   (or audio_common_msgs/AudioStamped if installed)
  /audio/wakeword      std_msgs/String             "wake" / "farewell" events
  /audio/vad           std_msgs/Bool               speech / silence transitions
  /person/owner_3d     geometry_msgs/PointStamped  (consumed downstream)

Audio is fanned into `ChatDataType.MIC_AUDIO` exactly the way ws_client does.
TTS playback (avatar_audio) is currently routed back to ALSA via aplay; the
robot speaker is handled by the host. The handler exposes a tiny FastAPI route
`/ros2/status` for liveness probes.

We intentionally tolerate `import rclpy` failure: when ROS2 is not installed
the handler logs a clear error and stays inert (rest of the chat engine still
loads). This is the path of stage 0 where we may run on a dev box first.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate
from chat_engine.common.engine_channel_type import EngineChannelType
from chat_engine.common.handler_base import HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalSourceType, ChatSignalType
from chat_engine.data_models.runtime_data.data_bundle import (
    DataBundle, DataBundleDefinition, DataBundleEntry, VariableSize,
)


_RCLPY_IMPORT_ERROR: Optional[BaseException] = None
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Bool, Float32MultiArray, String
    _HAVE_RCLPY = True
except Exception as exc:  # noqa: BLE001
    _RCLPY_IMPORT_ERROR = exc
    _HAVE_RCLPY = False


class Ros2ClientConfigModel(HandlerBaseConfigModel, BaseModel):
    audio_clean_topic: str = Field(default="/audio/clean")
    wakeword_topic: str = Field(default="/audio/wakeword")
    vad_topic: str = Field(default="/audio/vad")
    sample_rate: int = Field(default=16000)
    chunk_samples: int = Field(default=512,
                               description="samples per put_data() call (32ms @ 16k = 512)")
    auto_start_session: bool = Field(default=True,
                                     description="open one persistent robot session at startup")
    tts_playback: bool = Field(
        default=True,
        description="when True, locally play avatar audio via simpleaudio/aplay; "
                    "when False, downstream consumer is expected (e.g. external speaker)",
    )
    tts_aplay_device: Optional[str] = Field(
        default=None,
        description="optional --device argument for aplay (e.g. 'plughw:0,0')",
    )


@dataclass
class _ChunkBuf:
    """Concatenate float32 audio chunks until we have at least target_samples."""
    target_samples: int
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        x = x.astype(np.float32, copy=False).reshape(-1)
        self.buf = np.concatenate([self.buf, x])
        out = []
        n = self.target_samples
        while len(self.buf) >= n:
            out.append(self.buf[:n].copy())
            self.buf = self.buf[n:]
        return out


class Ros2ClientSessionDelegate(ClientSessionDelegate):
    """ROS2-backed session delegate; mirrors WsClientSessionDelegate's contract."""

    def __init__(self):
        self.timestamp_generator = None
        self.data_submitter = None
        self.shared_states = None
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
        }
        self.quit = threading.Event()

    async def get_data(self, modality: EngineChannelType,
                       timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        q = self.output_queues.get(modality)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def put_data(self, modality: EngineChannelType,
                 data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None,
                 samplerate: Optional[int] = None, loopback: bool = False):
        if timestamp is None:
            timestamp = self.get_timestamp()
        if self.data_submitter is None:
            return
        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)
        if chat_data_type is None or definition is None:
            return
        bundle = DataBundle(definition)
        if modality == EngineChannelType.AUDIO:
            bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            bundle.add_meta('human_text_end', True)
            bundle.add_meta('speech_id', str(uuid4()))
            bundle.set_main_data(data)
        else:
            return
        self.data_submitter.submit(ChatData(
            source="client", type=chat_data_type, data=bundle, timestamp=timestamp,
        ))

    def get_timestamp(self) -> Tuple[int, int]:
        return self.timestamp_generator()

    def emit_signal(self, signal: ChatSignal):
        if signal.source_type == ChatSignalSourceType.CLIENT \
                and signal.type == ChatSignalType.END:
            if self.shared_states is not None and self.shared_states.wake_session_active:
                self.shared_states.enable_vad = True

    def clear_data(self):
        for q in self.output_queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break


class Ros2ClientContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[Ros2ClientConfigModel] = None
        self.client_session_delegate: Optional[Ros2ClientSessionDelegate] = None


class _Ros2BridgeNode:
    """Owns rclpy node + spin thread; routes incoming topics to a delegate."""

    def __init__(self, cfg: Ros2ClientConfigModel,
                 delegate: Ros2ClientSessionDelegate):
        if not _HAVE_RCLPY:
            raise RuntimeError(f"rclpy import failed: {_RCLPY_IMPORT_ERROR!r}")
        self.cfg = cfg
        self.delegate = delegate
        self._chunk_buf = _ChunkBuf(target_samples=cfg.chunk_samples)
        self._stop = threading.Event()

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("openavatarchat_ros2_client")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.node.create_subscription(
            Float32MultiArray, cfg.audio_clean_topic, self._on_clean, sensor_qos,
        )
        self.node.create_subscription(
            String, cfg.wakeword_topic, self._on_wakeword, 10,
        )
        self.node.create_subscription(Bool, cfg.vad_topic, self._on_vad, 10)
        self._thread = threading.Thread(
            target=self._spin, name="ros2-client-spin", daemon=True,
        )
        self._thread.start()
        logger.info(f"ros2_client subscribed to {cfg.audio_clean_topic}, "
                    f"{cfg.wakeword_topic}, {cfg.vad_topic}")

    def _spin(self):
        try:
            while not self._stop.is_set() and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
        except Exception as e:  # noqa: BLE001
            logger.opt(exception=True).error(f"ros2 spin loop crashed: {e}")

    def _on_clean(self, msg: "Float32MultiArray"):
        try:
            x = np.asarray(msg.data, dtype=np.float32)
            for chunk in self._chunk_buf.push(x):
                self.delegate.put_data(EngineChannelType.AUDIO, chunk)
        except Exception:
            logger.opt(exception=True).debug("on_clean error")

    def _on_wakeword(self, msg: "String"):
        logger.debug(f"wakeword event: {msg.data!r}")
        # The KWS handler in chat-engine consumes audio + wakeword decisions
        # internally; this topic is informational. We could trigger session
        # reset here in future. For stage 0 it is just logged.

    def _on_vad(self, msg: "Bool"):
        # End-of-speech transition: emit END signal to mirror ws_client behavior.
        if not msg.data and self.delegate.shared_states is not None \
                and self.delegate.shared_states.wake_session_active:
            self.delegate.emit_signal(ChatSignal(
                source_type=ChatSignalSourceType.CLIENT,
                type=ChatSignalType.END,
            ))

    def shutdown(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass


class ClientHandlerRos2(ClientHandlerBase):
    """ROS2 client handler. Single persistent robot session."""

    def __init__(self):
        super().__init__()
        self.engine_config: Optional[ChatEngineConfigModel] = None
        self.handler_config: Optional[Ros2ClientConfigModel] = None
        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self._bridge: Optional[_Ros2BridgeNode] = None
        self._persistent_session_id: Optional[str] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=Ros2ClientConfigModel,
            client_session_delegate_class=Ros2ClientSessionDelegate,
        )

    def _prepare_definitions(self):
        sr = self.handler_config.sample_rate if self.handler_config else 16000
        audio_def = DataBundleDefinition()
        audio_def.add_entry(DataBundleEntry.create_audio_entry("mic_audio", 1, sr))
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
        self.handler_config = handler_config or Ros2ClientConfigModel()
        self._prepare_definitions()
        if not _HAVE_RCLPY:
            logger.warning(
                "rclpy is not importable: %s. The ros2_client handler will be "
                "registered but cannot bridge any topic until rclpy is installed "
                "(set up ROS2 Jazzy and `source /opt/ros/jazzy/setup.bash`).",
                _RCLPY_IMPORT_ERROR,
            )

    def on_setup_app(self, app: FastAPI, ui: Optional[Any] = None,
                     parent_block: Optional[Any] = None):
        @app.get("/ros2/status")
        async def ros2_status():
            return {
                "rclpy_available": _HAVE_RCLPY,
                "session": self._persistent_session_id,
                "topics": {
                    "audio_clean": self.handler_config.audio_clean_topic,
                    "wakeword": self.handler_config.wakeword_topic,
                    "vad": self.handler_config.vad_topic,
                },
            }

        if self.handler_config.auto_start_session and _HAVE_RCLPY:
            self._open_persistent_session()

    def _open_persistent_session(self):
        if self._persistent_session_id is not None:
            return
        sid = f"robot-{uuid4()}"
        try:
            delegate = self.handler_delegate.start_session(
                session_id=sid, timestamp_base=self.handler_config.sample_rate,
            )
            self._bridge = _Ros2BridgeNode(self.handler_config, delegate)
            self._persistent_session_id = sid
            logger.info(f"ros2_client persistent session opened: {sid}")
        except Exception as e:  # noqa: BLE001
            logger.opt(exception=True).error(f"failed to open ros2 session: {e}")

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        ctx = Ros2ClientContext(session_context.session_info.session_id)
        ctx.config = handler_config or self.handler_config
        return ctx

    def start_context(self, session_context: SessionContext,
                      handler_context: HandlerContext):
        pass

    def on_setup_session_delegate(self, session_context: SessionContext,
                                  handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
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
        if not hasattr(context, "client_session_delegate") \
                or context.client_session_delegate is None:
            return
        channel = inputs.type.channel_type
        # For now we only locally play avatar audio (TTS) so the headless robot
        # produces sound; subsequent stages may publish AVATAR_AUDIO to a ROS2
        # speaker_node instead.
        if channel == EngineChannelType.AUDIO and inputs.type == ChatDataType.AVATAR_AUDIO:
            audio = inputs.data.get_main_data()
            if audio is not None and self.handler_config.tts_playback:
                self._play_audio(audio.squeeze())
        # Mirror the ws_client convention: also place on output queue so
        # downstream consumers (e.g. UI) can observe.
        q = context.client_session_delegate.output_queues.get(channel)
        if q is not None:
            q.put_nowait(inputs)

    def _play_audio(self, audio: np.ndarray):
        """Best-effort TTS playback to default ALSA device.

        Stage 0 just shells out to aplay; we will replace this with a proper
        speaker node + duplex mixing in stage 1 once AEC is online.
        """
        try:
            import shutil
            import subprocess
            import tempfile
            import wave
            audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            sr = self.handler_config.sample_rate
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wpath = f.name
            with wave.open(wpath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_i16.tobytes())
            aplay = shutil.which("aplay") or "aplay"
            cmd = [aplay, "-q"]
            if self.handler_config.tts_aplay_device:
                cmd.extend(["-D", self.handler_config.tts_aplay_device])
            cmd.append(wpath)
            subprocess.Popen(cmd)
        except Exception:
            logger.opt(exception=True).debug("aplay launch failed")

    def destroy_context(self, context: HandlerContext):
        if hasattr(context, "client_session_delegate") \
                and context.client_session_delegate is not None:
            context.client_session_delegate.quit.set()
        if self._bridge is not None:
            self._bridge.shutdown()
            self._bridge = None
