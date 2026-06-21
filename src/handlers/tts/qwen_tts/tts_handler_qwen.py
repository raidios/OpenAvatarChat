"""
Qwen TTS handler —— 流式按句合成 + 情绪标签锚点。

行为约定（详见 plan ``动作标签与示教前端``）：

* 每次拿到 LLM 增量 chunk 时，先用 ``extract_action_tags`` 剥离情绪标签，得到
  纯文本 ``clean_chunk`` 与 ``AnchoredTag`` 列表。标签的局部 offset 会被转换
  为"全局 absolute offset"（自首字符以来的累计 clean 字符数），追加到
  ``context.pending_tags``。
* 累积的 ``clean_buffer`` 跑标点切句正则
  ``re.split(r'(?<=[,.~!?，。！？])', clean_buffer)``——规则与 cosyvoice /
  edgetts 一致——切出完整句立即调用 ``MultiModalConversation.call`` 合成。
* 每句的第一帧 PCM 在 ``DataBundle`` 上挂 meta ``action_tags=[{"name": ...}]``，
  其余帧不挂；末包 (``avatar_speech_end=True``) 携带 ``tail_action_tags``。
* 整段 clean 文本为空只剩标签时，不调 TTS，但仍下发一个零长度 audio 包带
  ``tail_action_tags + avatar_speech_end``。

设计上保持"标签剥离始终在断句之前"，这样将来给 cosyvoice/edgetts 同款接入
也只需要把它们入口处的 ``input_text +=`` 改成"先 extract，再 split"，断句
正则本身不变。
"""

import base64
import os
import re
import threading
import time
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
from handlers.common.action_tags import AnchoredTag, extract_action_tags


# 与 cosyvoice / edgetts 同款，避免不同 handler 朗读断句不一致。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[,.~!?，。！？])")
TELEMETRY_PREFIX = "[telemetry][tts]"


def _split_stream_carry(text: str) -> tuple[str, str]:
    """流式 LLM 可能在 ``[`` 与 ``]`` 之间切断 chunk。

    若 ``text`` 末尾存在「最后一个 ``[`` 之后直到串尾都没有 ``]``」的片段，
    则把该 ``[`` 起的后缀整体 **carry** 到下一 chunk，避免 ``[hap`` + ``py]``
    被当成正文送进 TTS。

    Returns:
        ``(process_now, carry_to_next)``。
    """
    if not text:
        return "", ""
    last_open = text.rfind("[")
    if last_open < 0:
        return text, ""
    if "]" in text[last_open:]:
        return text, ""
    return text[:last_open], text[last_open:]


class QwenTTSConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen3-tts-flash")
    voice: str = Field(default="Cherry")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY", ""))
    sample_rate: int = Field(default=24000)
    sentence_timeout_s: float = Field(default=3.0)


class QwenTTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[QwenTTSConfig] = None
        # 流式断句缓冲：累计的剥离后纯文本。每次切出完整句就把"碎尾"留这里。
        self.clean_buffer: str = ""
        # 已合成的累计 clean 字符数；用于把 tag 的 absolute offset 翻译成
        # 相对当前 clean_buffer 的本地偏移。
        self.chars_consumed: int = 0
        # 待挂接的 tag（absolute offset，自首字符算起）。
        self.pending_tags: List[AnchoredTag] = []
        # 出现过任何音频包的标志，决定空文本路径要不要下发零长度音频包。
        self.any_audio_emitted: bool = False
        self.shared_states = None
        # 跨 chunk 的未闭合 ``[...`` 缓冲，见 ``_split_stream_carry``。
        self._tts_carry: str = ""
        self._active_speech_id: Optional[str] = None
        self._speech_started_at: Optional[float] = None
        self._sentence_seq: int = 0
        self._speech_audio_frames: int = 0
        self._speech_audio_ms: float = 0.0


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
        context.shared_states = session_context.shared_states
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

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tag_meta(tags: List[AnchoredTag]) -> List[Dict[str, str]]:
        # 输出顺序按 offset 保留；offset 字段对客户端无意义，不传。
        return [{"name": t.name} for t in sorted(tags, key=lambda t: t.offset)]

    def _ensure_speech_telemetry(self, context: QwenTTSContext, speech_id: str) -> None:
        if context._active_speech_id == speech_id and context._speech_started_at is not None:
            return
        context._active_speech_id = speech_id
        context._speech_started_at = time.monotonic()
        context._sentence_seq = 0
        context._speech_audio_frames = 0
        context._speech_audio_ms = 0.0
        logger.info(f"{TELEMETRY_PREFIX} speech_start speech_id={speech_id}")

    def _append_clean_from_raw(self, context: QwenTTSContext, raw_piece: str) -> None:
        """对一段已确定「不会在中间切断标签」的原文做 extract 并写入 clean_buffer。"""
        if not raw_piece:
            return
        clean_chunk, local_tags = extract_action_tags(raw_piece)
        base_offset = context.chars_consumed + len(context.clean_buffer)
        for t in local_tags:
            context.pending_tags.append(AnchoredTag(name=t.name, offset=base_offset + t.offset))
        context.clean_buffer += clean_chunk

    def _ingest_chunk(self, context: QwenTTSContext, chunk_text: str) -> None:
        """剥离标签 + 把 absolute offset 的 tag 追加到 pending（含跨 chunk 标签拼接）。"""
        combined = context._tts_carry + (chunk_text or "")
        process_now, carry = _split_stream_carry(combined)
        context._tts_carry = carry
        self._append_clean_from_raw(context, process_now)

    def _take_complete_sentences(self, context: QwenTTSContext) -> List[str]:
        """从 clean_buffer 里切出完整句（带句尾标点），剩余碎尾保留。"""
        parts = _SENTENCE_SPLIT_RE.split(context.clean_buffer)
        if len(parts) <= 1:
            return []
        # parts[-1] 是断句正则切完后剩下的"尾"（可能为空字符串）；前面都是完整句。
        complete = [s for s in parts[:-1] if s]
        context.clean_buffer = parts[-1]
        return complete

    def _pop_tags_for_sentence(
        self, context: QwenTTSContext, sentence_len: int
    ) -> List[AnchoredTag]:
        """取出所有偏移落在该句区间 [chars_consumed, chars_consumed+sentence_len) 的 tag。"""
        start = context.chars_consumed
        end = start + sentence_len
        keep: List[AnchoredTag] = []
        out: List[AnchoredTag] = []
        for t in context.pending_tags:
            if start <= t.offset < end:
                out.append(t)
            else:
                keep.append(t)
        context.pending_tags = keep
        return out

    @staticmethod
    def _uses_tts_v2(model_name: str) -> bool:
        return (model_name or "").lower().startswith("cosyvoice-")

    def _synthesize_sentence_tts_v2(
        self,
        context: QwenTTSContext,
        sentence: str,
        prefix_tags: List[AnchoredTag],
        output_definition: DataBundleDefinition,
        speech_id: str,
        sentence_started_at: float,
        sentence_idx: int,
    ) -> tuple[Optional[float], int, float]:
        from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

        first_audio_at: Optional[float] = None
        first_frame = True
        audio_frames = 0
        audio_ms = 0.0
        done_event = threading.Event()
        error_holder: List[Optional[str]] = [None]

        class CosyVoiceCallback(ResultCallback):
            def on_data(self, data: bytes) -> None:
                nonlocal first_audio_at, first_frame, audio_frames, audio_ms
                if not data:
                    return
                now = time.monotonic()
                if first_audio_at is None:
                    first_audio_at = now
                    logger.info(
                        f"{TELEMETRY_PREFIX} sentence_first_audio speech_id={speech_id} "
                        f"idx={sentence_idx} first_ms={(now - sentence_started_at) * 1000.0:.0f}"
                    )
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                if audio_int16.size == 0:
                    return
                audio_frames += 1
                audio_ms += audio_int16.size / float(context.config.sample_rate) * 1000.0
                audio_float = audio_int16.astype(np.float32) / 32767.0

                output = DataBundle(output_definition)
                output.set_main_data(audio_float[np.newaxis, ...])
                output.add_meta("avatar_speech_end", False)
                output.add_meta("speech_id", speech_id)
                if first_frame and prefix_tags:
                    output.add_meta("action_tags", self._make_tag_meta(prefix_tags))
                context.submit_data(output)
                context.any_audio_emitted = True
                first_frame = False

            def on_complete(self) -> None:
                done_event.set()

            def on_error(self, message) -> None:
                error_holder[0] = str(message)
                done_event.set()

            def on_close(self) -> None:
                done_event.set()

        callback = CosyVoiceCallback()
        synthesizer = SpeechSynthesizer(
            model=context.config.model_name,
            voice=context.config.voice,
            callback=callback,
            format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        )
        synthesizer.streaming_call(sentence)
        synthesizer.streaming_complete()
        completed = done_event.wait(timeout=30)
        if not completed:
            logger.error(
                f"QwenTTS tts_v2 timeout: model={context.config.model_name} "
                f"voice={context.config.voice} sentence={sentence!r}"
            )
        if error_holder[0]:
            logger.error(f"QwenTTS tts_v2 error: {error_holder[0]}")
        return first_audio_at, audio_frames, audio_ms

    def _synthesize_sentence_multimodal(
        self,
        context: QwenTTSContext,
        sentence: str,
        prefix_tags: List[AnchoredTag],
        output_definition: DataBundleDefinition,
        speech_id: str,
        sentence_started_at: float,
        sentence_idx: int,
    ) -> tuple[Optional[float], int, float]:
        first_audio_at: Optional[float] = None
        first_frame = True
        audio_frames = 0
        audio_ms = 0.0
        done_event = threading.Event()
        cancelled = threading.Event()
        error_holder: List[Optional[BaseException]] = [None]
        progress_lock = threading.Lock()
        last_progress_at = [sentence_started_at]

        def worker() -> None:
            nonlocal first_audio_at, first_frame, audio_frames, audio_ms
            try:
                from dashscope import MultiModalConversation

                responses = MultiModalConversation.call(
                    model=context.config.model_name,
                    text=sentence,
                    voice=context.config.voice,
                    stream=True,
                )

                for chunk in responses:
                    if cancelled.is_set():
                        break
                    if chunk.output is None:
                        logger.error(f"QwenTTS error chunk: {chunk}")
                        break

                    audio_output = (
                        chunk.output.get("audio") if isinstance(chunk.output, dict)
                        else getattr(chunk.output, "audio", None)
                    )
                    if audio_output is not None:
                        audio_b64 = (
                            audio_output.get("data") if isinstance(audio_output, dict)
                            else getattr(audio_output, "data", None)
                        )
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            now = time.monotonic()
                            if first_audio_at is None:
                                first_audio_at = now
                                logger.info(
                                    f"{TELEMETRY_PREFIX} sentence_first_audio speech_id={speech_id} "
                                    f"idx={sentence_idx} first_ms={(now - sentence_started_at) * 1000.0:.0f}"
                                )
                            with progress_lock:
                                last_progress_at[0] = now
                            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                            audio_frames += 1
                            audio_ms += audio_int16.size / float(context.config.sample_rate) * 1000.0
                            audio_float = audio_int16.astype(np.float32) / 32767.0
                            audio_float = audio_float[np.newaxis, ...]

                            if cancelled.is_set():
                                break
                            output = DataBundle(output_definition)
                            output.set_main_data(audio_float)
                            output.add_meta("avatar_speech_end", False)
                            output.add_meta("speech_id", speech_id)
                            if first_frame and prefix_tags:
                                output.add_meta(
                                    "action_tags",
                                    self._make_tag_meta(prefix_tags),
                                )
                            context.submit_data(output)
                            context.any_audio_emitted = True
                            first_frame = False

                    finish_reason = (
                        chunk.output.get("finish_reason") if isinstance(chunk.output, dict)
                        else getattr(chunk.output, "finish_reason", None)
                    )
                    if finish_reason == "stop":
                        break
            except BaseException as e:
                error_holder[0] = e
            finally:
                done_event.set()

        thread = threading.Thread(
            target=worker,
            name=f"qwen-tts-{speech_id}-{sentence_idx}",
            daemon=True,
        )
        thread.start()

        timeout_s = max(float(context.config.sentence_timeout_s), 0.1)
        while not done_event.is_set():
            with progress_lock:
                idle_for = time.monotonic() - last_progress_at[0]
            remaining = timeout_s - idle_for
            if remaining <= 0:
                cancelled.set()
                logger.error(
                    f"QwenTTS multimodal idle timeout after {timeout_s:.1f}s: "
                    f"model={context.config.model_name} voice={context.config.voice} "
                    f"sentence={sentence!r}"
                )
                break
            done_event.wait(timeout=min(remaining, 0.1))

        if done_event.is_set() and error_holder[0] is not None:
            raise error_holder[0]

        return first_audio_at, audio_frames, audio_ms

    def _synthesize_sentence(
        self,
        context: QwenTTSContext,
        sentence: str,
        prefix_tags: List[AnchoredTag],
        output_definition: DataBundleDefinition,
        speech_id: str,
        chars_advance: int,
    ) -> None:
        """调 dashscope MultiModalConversation 合成单句，并下发对应 PCM 包。

        ``sentence`` 一般为 ``.strip()`` 后的文本供 TTS 朗读；``chars_advance``
        必须等于**断句切出的原始片段长度**（含首尾空白），以便与
        ``_pop_tags_for_sentence`` 里用的锚点区间一致。
        """
        context._sentence_seq += 1
        sentence_idx = context._sentence_seq
        sentence_started_at = time.monotonic()
        first_audio_at: Optional[float] = None
        audio_frames = 0
        audio_ms = 0.0
        tag_names = [t.name for t in prefix_tags]
        logger.info(f"QwenTTS synth sentence: {sentence!r} tags={[t.name for t in prefix_tags]}")
        logger.info(
            f"{TELEMETRY_PREFIX} sentence_start speech_id={speech_id} "
            f"idx={sentence_idx} model={context.config.model_name} voice={context.config.voice} "
            f"chars={len(sentence)} tags={tag_names}"
        )
        try:
            if self._uses_tts_v2(context.config.model_name):
                first_audio_at, audio_frames, audio_ms = self._synthesize_sentence_tts_v2(
                    context=context,
                    sentence=sentence,
                    prefix_tags=prefix_tags,
                    output_definition=output_definition,
                    speech_id=speech_id,
                    sentence_started_at=sentence_started_at,
                    sentence_idx=sentence_idx,
                )
            else:
                first_audio_at, audio_frames, audio_ms = self._synthesize_sentence_multimodal(
                    context=context,
                    sentence=sentence,
                    prefix_tags=prefix_tags,
                    output_definition=output_definition,
                    speech_id=speech_id,
                    sentence_started_at=sentence_started_at,
                    sentence_idx=sentence_idx,
                )

        except Exception as e:
            logger.opt(exception=True).error(f"QwenTTS synthesis error: {e}")
            if context.shared_states is not None and context.shared_states.wake_session_active:
                context.shared_states.enable_vad = True
        finally:
            context.chars_consumed += chars_advance
            context._speech_audio_frames += audio_frames
            context._speech_audio_ms += audio_ms
            elapsed_ms = (time.monotonic() - sentence_started_at) * 1000.0
            first_ms = (
                (first_audio_at - sentence_started_at) * 1000.0
                if first_audio_at is not None else -1.0
            )
            logger.info(
                f"{TELEMETRY_PREFIX} sentence_done speech_id={speech_id} idx={sentence_idx} "
                f"elapsed_ms={elapsed_ms:.0f} first_ms={first_ms:.0f} "
                f"audio_ms={audio_ms:.0f} frames={audio_frames}"
            )

    def _flush_complete_sentences(
        self,
        context: QwenTTSContext,
        output_definition: DataBundleDefinition,
        speech_id: str,
    ) -> None:
        for sent in self._take_complete_sentences(context):
            tags = self._pop_tags_for_sentence(context, len(sent))
            stripped = sent.strip()
            if not stripped:
                # 还是要把 chars_consumed 推进，但跳过合成。tag 如果落在纯空白句
                # 上很罕见——保险起见把它们挂到下一句的 pending（已经在
                # _pop_tags_for_sentence 里被取出，这里再放回去）。
                for t in tags:
                    context.pending_tags.append(t)
                context.chars_consumed += len(sent)
                continue
            self._synthesize_sentence(
                context, stripped, tags, output_definition, speech_id,
                chars_advance=len(sent),
            )

    def _emit_end_packet(
        self,
        context: QwenTTSContext,
        output_definition: DataBundleDefinition,
        speech_id: str,
    ) -> None:
        """统一发末包：携带 tail_action_tags + avatar_speech_end=True。"""
        end_output = DataBundle(output_definition)
        end_output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        end_output.add_meta("avatar_speech_end", True)
        end_output.add_meta("speech_id", speech_id)
        if context.pending_tags:
            end_output.add_meta(
                "tail_action_tags",
                self._make_tag_meta(context.pending_tags),
            )
        tail_names = [t.name for t in context.pending_tags]
        context.submit_data(end_output)
        speech_started = context._speech_started_at
        wall_ms = (
            (time.monotonic() - speech_started) * 1000.0
            if speech_started is not None else -1.0
        )
        logger.info(
            f"{TELEMETRY_PREFIX} speech_end speech_id={speech_id} "
            f"wall_ms={wall_ms:.0f} audio_ms={context._speech_audio_ms:.0f} "
            f"frames={context._speech_audio_frames} sentences={context._sentence_seq} "
            f"tail_tags={tail_names}"
        )
        logger.info(f"QwenTTS speech end (tail_tags={tail_names})")
        context.pending_tags = []
        context.clean_buffer = ""
        context.chars_consumed = 0
        context.any_audio_emitted = False
        context._tts_carry = ""
        context._active_speech_id = None
        context._speech_started_at = None
        context._sentence_seq = 0
        context._speech_audio_frames = 0
        context._speech_audio_ms = 0.0

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(QwenTTSContext, context)

        if inputs.type != ChatDataType.AVATAR_TEXT:
            return

        text = inputs.data.get_main_data()
        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id
        self._ensure_speech_telemetry(context, speech_id)

        if text:
            self._ingest_chunk(context, text)
            # 增量阶段也尝试切句，先把完整句送去合成。
            self._flush_complete_sentences(context, output_definition, speech_id)

        text_end = inputs.data.get_meta("avatar_text_end", False)
        if not text_end:
            return

        # 流结束：把未闭合的 carry 拼回正文（若仍无 ``]``，整段当普通字处理）
        if context._tts_carry:
            self._append_clean_from_raw(context, context._tts_carry)
            context._tts_carry = ""

        # 末尾 flush：把碎尾当最后一句合（哪怕没结尾标点）。
        tail = context.clean_buffer
        context.clean_buffer = ""
        if tail.strip():
            tags = self._pop_tags_for_sentence(context, len(tail))
            self._synthesize_sentence(
                context, tail.strip(), tags, output_definition, speech_id,
                chars_advance=len(tail),
            )
        else:
            # 末尾全是空白：把 chars_consumed 推齐，让 pending_tags 里末尾偏移
            # 的 tag 落入 tail_action_tags 范畴。
            context.chars_consumed += len(tail)

        if not context.any_audio_emitted and not context.pending_tags:
            # 极端情况：LLM 整段没东西可读，也没有标签。原来的实现会下发一个
            # 240 帧的零长度包并提示 wake-session 重新开 VAD；保持此行为。
            logger.info("TTS: empty input text, skipping synthesis")
            if context.shared_states is not None and context.shared_states.wake_session_active:
                context.shared_states.enable_vad = True

        self._emit_end_packet(context, output_definition, speech_id)

    def destroy_context(self, context: HandlerContext):
        pass
