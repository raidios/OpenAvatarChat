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
        from dashscope import MultiModalConversation

        first_frame = True
        logger.info(f"QwenTTS synth sentence: {sentence!r} tags={[t.name for t in prefix_tags]}")
        try:
            responses = MultiModalConversation.call(
                model=context.config.model_name,
                text=sentence,
                voice=context.config.voice,
                stream=True,
            )

            for chunk in responses:
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
                        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_int16.astype(np.float32) / 32767.0
                        audio_float = audio_float[np.newaxis, ...]

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

        except Exception as e:
            logger.opt(exception=True).error(f"QwenTTS synthesis error: {e}")
            if context.shared_states is not None and context.shared_states.wake_session_active:
                context.shared_states.enable_vad = True
        finally:
            context.chars_consumed += chars_advance

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
        logger.info(f"QwenTTS speech end (tail_tags={tail_names})")
        context.pending_tags = []
        context.clean_buffer = ""
        context.chars_consumed = 0
        context.any_audio_emitted = False
        context._tts_carry = ""

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
