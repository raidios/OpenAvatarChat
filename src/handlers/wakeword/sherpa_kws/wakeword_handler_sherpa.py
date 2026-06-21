import base64
import os
import threading
import time
from abc import ABC
from typing import Dict, List, Optional, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext, SharedStates
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class WakeWordConfig(HandlerBaseConfigModel, BaseModel):
    model_dir: str = Field(default="models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
    keywords_file: str = Field(default="config/keywords.txt")
    keywords_score: float = Field(default=1.5)
    keywords_threshold: float = Field(default=0.25)
    num_threads: int = Field(default=2)
    sample_rate: int = Field(default=16000)
    wake_reply_text: str = Field(default="你好，有什么可以帮你的？")
    farewell_reply_text: str = Field(default="好的，再见！")
    farewell_keywords: List[str] = Field(default=["再见", "拜拜", "结束", "退下"])
    session_timeout: float = Field(default=15.0)
    tts_model_name: str = Field(default="qwen3-tts-flash")
    tts_voice: str = Field(default="Cherry")
    tts_sample_rate: int = Field(default=24000)
    wake_reply_failsafe_extra_s: float = Field(default=3.0)
    # Same as QwenTTS; WakeWord loads earlier in yaml so we must set dashscope explicitly.
    api_key: Optional[str] = Field(default=None)
    # When True, do NOT run KWS on the incoming MIC_AUDIO stream; rely on
    # the client to detect the keyword (it owns the M260C audio + DOA
    # buffer) and trigger a wake session via a ``{"type": "wake_word"}``
    # WebSocket control message. The handler then performs the same
    # session_active + wake-reply playback path it would have run on a
    # local KWS hit. The Sherpa model and pre-rendered TTS audio are
    # still loaded (we need the wake_audio buffer), so the only cost
    # we save is per-frame KWS decoding (~1 CPU thread on Pi5).
    external_wake_only: bool = Field(default=False)


class WakeWordContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.stream = None
        self.shared_states = None
        self.idle_since: Optional[float] = None
        self.wake_listen_timer: Optional[threading.Timer] = None


class HandlerWakeWord(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.keyword_spotter = None
        self.config: Optional[WakeWordConfig] = None
        self.wake_audio: Optional[np.ndarray] = None
        self.farewell_audio: Optional[np.ndarray] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=WakeWordConfig,
        )

    @staticmethod
    def _ensure_onnxruntime_lib():
        """Create a symlink in sherpa_onnx/lib/ so it can find libonnxruntime.so via its RPATH."""
        import glob
        import importlib.util
        try:
            import onnxruntime
            capi_dir = os.path.dirname(onnxruntime.capi._pybind_state.__file__)
        except Exception:
            return
        candidates = glob.glob(os.path.join(capi_dir, "libonnxruntime.so.*"))
        if not candidates:
            return
        ort_lib = candidates[0]

        spec = importlib.util.find_spec("sherpa_onnx")
        if spec is None or spec.origin is None:
            return
        sherpa_pkg_dir = os.path.dirname(spec.origin)
        sherpa_lib_dir = os.path.join(sherpa_pkg_dir, "lib")
        if not os.path.isdir(sherpa_lib_dir):
            return
        target = os.path.join(sherpa_lib_dir, "libonnxruntime.so")
        if not os.path.exists(target):
            try:
                os.symlink(ort_lib, target)
                logger.info(f"Created symlink {target} -> {ort_lib}")
            except OSError as e:
                logger.warning(f"Could not create symlink: {e}")

    def _ensure_dashscope_api_key(self) -> None:
        """Align with QwenTTS: get_default_api_key() uses dashscope.api_key, not live os.environ."""
        import dashscope

        if self.config is None:
            return
        cfg_key = (self.config.api_key or "").strip()
        if cfg_key:
            dashscope.api_key = cfg_key
        elif os.environ.get("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    def _synthesize_audio(self, text: str) -> Optional[np.ndarray]:
        """Use dashscope TTS to pre-generate audio for a text prompt."""
        self._ensure_dashscope_api_key()
        try:
            if (self.config.tts_model_name or "").lower().startswith("cosyvoice-"):
                from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

                chunks = []
                done_event = threading.Event()
                error_holder: List[Optional[str]] = [None]

                class WakeTTSCallback(ResultCallback):
                    def on_data(self, data: bytes) -> None:
                        if data:
                            audio_int16 = np.frombuffer(data, dtype=np.int16)
                            if audio_int16.size:
                                chunks.append(audio_int16.astype(np.float32) / 32767.0)

                    def on_complete(self) -> None:
                        done_event.set()

                    def on_error(self, message) -> None:
                        error_holder[0] = str(message)
                        done_event.set()

                    def on_close(self) -> None:
                        done_event.set()

                synthesizer = SpeechSynthesizer(
                    model=self.config.tts_model_name,
                    voice=self.config.tts_voice,
                    callback=WakeTTSCallback(),
                    format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                )
                synthesizer.streaming_call(text)
                synthesizer.streaming_complete()
                completed = done_event.wait(timeout=30)
                if not completed:
                    logger.error(f"Wake TTS timeout for model={self.config.tts_model_name}")
                if error_holder[0]:
                    logger.error(f"Wake TTS error: {error_holder[0]}")
                if chunks:
                    return np.concatenate(chunks)
                return None

            from dashscope import MultiModalConversation
            responses = MultiModalConversation.call(
                model=self.config.tts_model_name,
                text=text,
                voice=self.config.tts_voice,
                stream=True,
            )
            chunks = []
            for chunk in responses:
                if chunk.output is None:
                    logger.error(f"TTS pre-generation error: {chunk}")
                    break
                audio_output = chunk.output.get("audio") if isinstance(chunk.output, dict) else getattr(chunk.output, "audio", None)
                if audio_output is not None:
                    audio_b64 = audio_output.get("data") if isinstance(audio_output, dict) else getattr(audio_output, "data", None)
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32767.0
                        chunks.append(audio_float)
                finish_reason = chunk.output.get("finish_reason") if isinstance(chunk.output, dict) else getattr(chunk.output, "finish_reason", None)
                if finish_reason == "stop":
                    break
            if chunks:
                return np.concatenate(chunks)
        except Exception as e:
            logger.opt(exception=True).error(f"TTS pre-generation failed for '{text}': {e}")
        return None

    def _submit_pregenerated_audio(self, context: WakeWordContext, audio: np.ndarray, speech_id: str):
        """Submit pre-generated audio chunks through AVATAR_AUDIO output."""
        sample_rate = self.config.tts_sample_rate
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, sample_rate))

        chunk_size = sample_rate // 10  # 100ms chunks
        offset = 0
        while offset < len(audio):
            end = min(offset + chunk_size, len(audio))
            chunk = audio[offset:end]
            output = DataBundle(definition)
            output.set_main_data(chunk[np.newaxis, ...])
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            offset = end

        end_output = DataBundle(definition)
        end_output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        end_output.add_meta("avatar_speech_end", True)
        end_output.add_meta("speech_id", speech_id)
        context.submit_data(end_output)

    def _cancel_wake_listen_failsafe(self, context: WakeWordContext) -> None:
        t = context.wake_listen_timer
        if t is not None:
            t.cancel()
            context.wake_listen_timer = None

    def _schedule_wake_listen_failsafe(self, context: WakeWordContext, ss: SharedStates) -> None:
        """If the client never sends playback_complete, re-enable VAD after wake TTS duration."""
        self._cancel_wake_listen_failsafe(context)
        if self.wake_audio is None or self.config is None:
            return
        delay = (
            len(self.wake_audio) / float(self.config.tts_sample_rate)
            + 0.35
            + float(self.config.wake_reply_failsafe_extra_s)
        )

        def _on_timeout() -> None:
            try:
                if not ss.wake_session_active or ss.enable_vad:
                    return
                logger.warning(
                    "Wake reply did not receive playback_complete in time; enabling VAD (server failsafe)"
                )
                ss.enable_vad = True
                ss.last_interaction_time = time.monotonic()
            finally:
                context.wake_listen_timer = None

        context.wake_listen_timer = threading.Timer(delay, _on_timeout)
        context.wake_listen_timer.daemon = True
        context.wake_listen_timer.start()

    def load(self, engine_config: ChatEngineConfigModel,
             handler_config: Optional[HandlerBaseConfigModel] = None):
        self.config = cast(WakeWordConfig, handler_config)
        if not isinstance(self.config, WakeWordConfig):
            self.config = WakeWordConfig()

        # Skip loading the Sherpa KWS model entirely when the client
        # drives wake events; the handler still owns session state +
        # TTS replies but no longer runs CPU-bound keyword decoding.
        if not self.config.external_wake_only:
            self._ensure_onnxruntime_lib()
            import sherpa_onnx

            model_dir = self.config.model_dir
            if not os.path.isabs(model_dir):
                from engine_utils.directory_info import DirectoryInfo
                model_dir = os.path.join(DirectoryInfo.get_project_dir(), model_dir)

            keywords_file = self.config.keywords_file
            if not os.path.isabs(keywords_file):
                from engine_utils.directory_info import DirectoryInfo
                keywords_file = os.path.join(DirectoryInfo.get_project_dir(), keywords_file)

            encoder = os.path.join(model_dir, "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
            decoder = os.path.join(model_dir, "decoder-epoch-13-avg-2-chunk-16-left-64.onnx")
            joiner = os.path.join(model_dir, "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
            tokens = os.path.join(model_dir, "tokens.txt")

            if not os.path.exists(encoder):
                logger.warning(f"WakeWord model not found at {model_dir}. "
                               f"Please download using scripts/download_kws_model.sh")
                return

            if not os.path.exists(keywords_file):
                logger.warning(f"Keywords file not found: {keywords_file}")
                return

            self.keyword_spotter = sherpa_onnx.KeywordSpotter(
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                tokens=tokens,
                keywords_file=keywords_file,
                keywords_score=self.config.keywords_score,
                keywords_threshold=self.config.keywords_threshold,
                num_threads=self.config.num_threads,
                provider="cpu",
            )
            logger.info("WakeWord KeywordSpotter loaded successfully")
        else:
            logger.info("WakeWord: external_wake_only=True, "
                        "skipping local KWS model load (client drives wakes)")

        if self.config.wake_reply_text:
            logger.info(f"Pre-generating wake reply audio: '{self.config.wake_reply_text}'")
            self.wake_audio = self._synthesize_audio(self.config.wake_reply_text)
            if self.wake_audio is not None:
                logger.info(f"Wake reply audio ready ({len(self.wake_audio)} samples)")
            else:
                logger.warning("Failed to pre-generate wake reply audio")

        if self.config.farewell_reply_text:
            logger.info(f"Pre-generating farewell reply audio: '{self.config.farewell_reply_text}'")
            self.farewell_audio = self._synthesize_audio(self.config.farewell_reply_text)
            if self.farewell_audio is not None:
                logger.info(f"Farewell reply audio ready ({len(self.farewell_audio)} samples)")
            else:
                logger.warning("Failed to pre-generate farewell reply audio")

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        context = WakeWordContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if self.config and self.config.farewell_keywords:
            context.shared_states.farewell_keywords = self.config.farewell_keywords
        # external_wake_only: no per-session decoder stream needed.
        if self.keyword_spotter is not None and not (
            self.config and self.config.external_wake_only
        ):
            context.stream = self.keyword_spotter.create_stream()
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        sample_rate = self.config.tts_sample_rate if self.config else 24000
        audio_def = DataBundleDefinition()
        audio_def.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, sample_rate))

        inputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=audio_def,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def _end_wake_session(self, context: WakeWordContext, play_farewell: bool = True):
        """End the current wake session and optionally play farewell audio."""
        self._cancel_wake_listen_failsafe(context)
        logger.info("Ending wake session")
        context.shared_states.wake_session_active = False
        context.shared_states.enable_vad = False
        context.shared_states.farewell_pending = False
        context.idle_since = None
        if not play_farewell:
            return
        if self.farewell_audio is None and self.config and self.config.farewell_reply_text:
            self.farewell_audio = self._synthesize_audio(self.config.farewell_reply_text)
        if self.farewell_audio is not None:
            speech_id = f"farewell-{context.session_id}"
            self._submit_pregenerated_audio(context, self.farewell_audio, speech_id)

    def _start_wake_session(self, context: "WakeWordContext",
                            keyword: str) -> None:
        """Common path for both KWS-detected and externally-driven wakes.

        Marks the session active, keeps VAD off until the client
        finishes wake-reply playback, and submits the pre-generated
        wake reply audio.
        """
        ss = context.shared_states
        logger.info(f"Wake word detected: {keyword}")
        ss.wake_session_active = True
        # Keep VAD off until wake reply finishes on the client (playback_complete).
        # Forcing VAD here caused immediate "speech end" + empty ASR while the user
        # was still listening to the wake prompt.
        ss.enable_vad = False
        ss.last_interaction_time = time.monotonic()
        context.idle_since = None

        if self.wake_audio is None and self.config.wake_reply_text:
            self.wake_audio = self._synthesize_audio(self.config.wake_reply_text)
        if self.wake_audio is not None:
            speech_id = f"wake-{context.session_id}"
            self._submit_pregenerated_audio(context, self.wake_audio, speech_id)
            self._schedule_wake_listen_failsafe(context, ss)
        else:
            ss.enable_vad = True

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(WakeWordContext, context)
        if inputs.type != ChatDataType.MIC_AUDIO:
            return

        ss = context.shared_states
        if ss is None:
            return

        # Handle farewell triggered by ASR
        if ss.farewell_pending:
            self._end_wake_session(context, play_farewell=True)
            return

        # During active wake session: manage timeout
        if ss.wake_session_active:
            if ss.enable_vad:
                now = time.monotonic()
                if context.idle_since is None:
                    context.idle_since = now
                elif now - context.idle_since > self.config.session_timeout:
                    logger.info(f"Wake session timeout ({self.config.session_timeout}s)")
                    self._end_wake_session(context, play_farewell=True)
            else:
                context.idle_since = None
                ss.last_interaction_time = time.monotonic()
            return

        # Not in wake session: listen for an externally-driven wake event
        # first (client-side KWS), then fall back to local KWS unless
        # external_wake_only is set.
        ext = ss.external_wake_event
        if ext is not None:
            ss.external_wake_event = None
            keyword = ext[0] if isinstance(ext, (tuple, list)) else str(ext)
            doa = ext[1] if isinstance(ext, (tuple, list)) and len(ext) > 1 else None
            logger.info(
                f"External wake from client: keyword={keyword!r} doa={doa}"
            )
            self._start_wake_session(context, keyword)
            return

        if self.config and self.config.external_wake_only:
            return  # don't run local KWS at all
        if self.keyword_spotter is None or context.stream is None:
            return

        audio = inputs.data.get_main_data()
        if audio is None:
            return

        audio = audio.squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32767.0

        context.stream.accept_waveform(self.config.sample_rate, audio)

        while self.keyword_spotter.is_ready(context.stream):
            self.keyword_spotter.decode_stream(context.stream)

        result = self.keyword_spotter.get_result(context.stream)
        if result and len(result) > 0:
            self._start_wake_session(context, result.strip())

    def destroy_context(self, context: HandlerContext):
        self._cancel_wake_listen_failsafe(cast(WakeWordContext, context))
