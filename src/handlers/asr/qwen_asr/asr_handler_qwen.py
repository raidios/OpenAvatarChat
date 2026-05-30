import os
import threading
from abc import ABC
from typing import Dict, List, Optional, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class QwenASRConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen3-asr-flash")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY", ""))
    language_hints: List[str] = Field(default=["zh", "en"])
    sample_rate: int = Field(default=16000)


class QwenASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[QwenASRConfig] = None
        self.output_audios: List[np.ndarray] = []
        self.shared_states = None


class HandlerQwenASR(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.config: Optional[QwenASRConfig] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=QwenASRConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel,
             handler_config: Optional[HandlerBaseConfigModel] = None):
        if isinstance(handler_config, QwenASRConfig):
            self.config = handler_config
        else:
            self.config = QwenASRConfig()

        if self.config.api_key:
            import dashscope
            dashscope.api_key = self.config.api_key
        elif 'DASHSCOPE_API_KEY' in os.environ:
            import dashscope
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

        logger.info(f"QwenASR handler loaded with model: {self.config.model_name}")

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        context = QwenASRContext(session_context.session_info.session_id)
        if isinstance(handler_config, QwenASRConfig):
            context.config = handler_config
        else:
            context.config = self.config
        context.shared_states = session_context.shared_states
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("human_text"))

        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(QwenASRContext, context)

        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return

        audio = inputs.data.get_main_data()
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if audio is not None:
            audio = audio.squeeze()
            context.output_audios.append(audio)

        speech_end = inputs.data.get_meta("human_speech_end", False)
        if not speech_end:
            return

        if not context.output_audios:
            if context.shared_states is not None and context.shared_states.wake_session_active:
                logger.info("ASR: empty audio buffer, re-enabling VAD")
                context.shared_states.enable_vad = True
            return

        full_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()

        if full_audio.dtype == np.float32:
            pcm_int16 = (full_audio * 32767).astype(np.int16)
        else:
            pcm_int16 = full_audio.astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        results: List[str] = []
        done_event = threading.Event()
        error_holder: List[Optional[str]] = [None]

        class ASRCallback(RecognitionCallback):
            def on_event(self, result: RecognitionResult):
                sentence = result.get_sentence()
                if sentence and sentence.get("text"):
                    if sentence.get("end_time") is not None or sentence.get("is_sentence_end"):
                        results.append(sentence["text"])

            def on_complete(self):
                done_event.set()

            def on_error(self, result: RecognitionResult):
                error_holder[0] = str(result)
                done_event.set()

        callback = ASRCallback()
        sample_rate = context.config.sample_rate

        recognition = Recognition(
            model=context.config.model_name,
            format="pcm",
            sample_rate=sample_rate,
            language_hints=context.config.language_hints,
            callback=callback,
        )

        try:
            recognition.start()

            chunk_size = sample_rate * 2  # 1 second of int16 audio
            offset = 0
            while offset < len(pcm_bytes):
                end = min(offset + chunk_size, len(pcm_bytes))
                recognition.send_audio_frame(pcm_bytes[offset:end])
                offset = end

            recognition.stop()
            done_event.wait(timeout=30)
        except Exception as e:
            logger.opt(exception=True).error(f"QwenASR recognition error: {e}")
            if context.shared_states is not None and context.shared_states.wake_session_active:
                context.shared_states.enable_vad = True
            return

        if error_holder[0]:
            logger.error(f"QwenASR callback error: {error_holder[0]}")
            if context.shared_states is not None and context.shared_states.wake_session_active:
                context.shared_states.enable_vad = True
            return

        output_text = "".join(results).strip()
        logger.info(f"QwenASR result: {output_text}")

        if not output_text:
            if context.shared_states is not None:
                if context.shared_states.wake_session_active:
                    context.shared_states.enable_vad = True
            return

        if context.shared_states is not None and context.shared_states.wake_session_active:
            farewell_keywords = context.shared_states.farewell_keywords or []
            if any(kw in output_text for kw in farewell_keywords):
                logger.info(f"Farewell keyword detected in: {output_text}")
                context.shared_states.farewell_pending = True
                return

        output = DataBundle(output_definition)
        output.set_main_data(output_text)
        output.add_meta("human_text_end", False)
        output.add_meta("speech_id", speech_id)
        yield output

        end_output = DataBundle(output_definition)
        end_output.set_main_data("")
        end_output.add_meta("human_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
