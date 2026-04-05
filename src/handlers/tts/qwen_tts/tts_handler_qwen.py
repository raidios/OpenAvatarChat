import base64
import os
import re
from abc import ABC
from typing import Dict, Optional, cast

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


class QwenTTSConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen3-tts-flash")
    voice: str = Field(default="Cherry")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY", ""))
    sample_rate: int = Field(default=24000)


class QwenTTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[QwenTTSConfig] = None
        self.input_text: str = ""


class HandlerQwenTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.config: Optional[QwenTTSConfig] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=QwenTTSConfig,
        )

    def load(self, engine_config: ChatEngineConfigModel,
             handler_config: Optional[HandlerBaseConfigModel] = None):
        if isinstance(handler_config, QwenTTSConfig):
            self.config = handler_config
        else:
            self.config = QwenTTSConfig()

        if self.config.api_key:
            import dashscope
            dashscope.api_key = self.config.api_key
        elif 'DASHSCOPE_API_KEY' in os.environ:
            import dashscope
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

        logger.info(f"QwenTTS handler loaded with model: {self.config.model_name}, voice: {self.config.voice}")

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        context = QwenTTSContext(session_context.session_info.session_id)
        if isinstance(handler_config, QwenTTSConfig):
            context.config = handler_config
        else:
            context.config = self.config
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        sample_rate = self.config.sample_rate if self.config else 24000
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, sample_rate))

        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        from dashscope import MultiModalConversation

        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(QwenTTSContext, context)

        if inputs.type != ChatDataType.AVATAR_TEXT:
            return

        text = inputs.data.get_main_data()
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += text

        text_end = inputs.data.get_meta("avatar_text_end", False)
        if not text_end:
            return

        full_text = context.input_text.strip()
        context.input_text = ""

        if not full_text:
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            output.add_meta("avatar_speech_end", True)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            return

        logger.info(f"QwenTTS synthesizing: {full_text[:50]}...")

        try:
            responses = MultiModalConversation.call(
                model=context.config.model_name,
                text=full_text,
                voice=context.config.voice,
                stream=True,
            )

            for chunk in responses:
                if chunk.output is None:
                    logger.error(f"QwenTTS error: {chunk}")
                    break

                audio_output = chunk.output.get("audio") if isinstance(chunk.output, dict) else getattr(chunk.output, "audio", None)
                if audio_output is not None:
                    audio_b64 = audio_output.get("data") if isinstance(audio_output, dict) else getattr(audio_output, "data", None)
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32767.0
                        audio_float = audio_float[np.newaxis, ...]

                        output = DataBundle(output_definition)
                        output.set_main_data(audio_float)
                        output.add_meta("avatar_speech_end", False)
                        output.add_meta("speech_id", speech_id)
                        context.submit_data(output)

                finish_reason = chunk.output.get("finish_reason") if isinstance(chunk.output, dict) else getattr(chunk.output, "finish_reason", None)
                if finish_reason == "stop":
                    break

        except Exception as e:
            logger.opt(exception=True).error(f"QwenTTS synthesis error: {e}")

        end_output = DataBundle(output_definition)
        end_output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        end_output.add_meta("avatar_speech_end", True)
        end_output.add_meta("speech_id", speech_id)
        context.submit_data(end_output)
        logger.info("QwenTTS speech end")

    def destroy_context(self, context: HandlerContext):
        pass
