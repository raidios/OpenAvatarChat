import os
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


class WakeWordConfig(HandlerBaseConfigModel, BaseModel):
    model_dir: str = Field(default="models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
    keywords_file: str = Field(default="config/keywords.txt")
    keywords_score: float = Field(default=1.5)
    keywords_threshold: float = Field(default=0.25)
    num_threads: int = Field(default=2)
    sample_rate: int = Field(default=16000)


class WakeWordContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.stream = None
        self.shared_states = None


class HandlerWakeWord(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.keyword_spotter = None
        self.config: Optional[WakeWordConfig] = None

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

    def load(self, engine_config: ChatEngineConfigModel,
             handler_config: Optional[HandlerBaseConfigModel] = None):
        self._ensure_onnxruntime_lib()
        import sherpa_onnx

        self.config = cast(WakeWordConfig, handler_config)
        if not isinstance(self.config, WakeWordConfig):
            self.config = WakeWordConfig()

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

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        context = WakeWordContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if self.keyword_spotter is not None:
            context.stream = self.keyword_spotter.create_stream()
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        inputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
            )
        }
        return HandlerDetail(inputs=inputs, outputs={})

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(WakeWordContext, context)
        if self.keyword_spotter is None or context.stream is None:
            return
        if context.shared_states is not None and context.shared_states.enable_vad:
            return
        if inputs.type != ChatDataType.MIC_AUDIO:
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
            logger.info(f"Wake word detected: {result}")
            if context.shared_states is not None:
                context.shared_states.enable_vad = True

    def destroy_context(self, context: HandlerContext):
        pass
