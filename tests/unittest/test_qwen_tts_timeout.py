import sys
import threading
import types
import unittest

import numpy as np

from chat_engine.common.handler_base import HandlerDataInfo
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import (
    DataBundle,
    DataBundleDefinition,
    DataBundleEntry,
)
from handlers.tts.qwen_tts.tts_handler_qwen import (
    HandlerQwenTTS,
    QwenTTSConfig,
    QwenTTSContext,
)


class _Submitter:
    def __init__(self):
        self.items = []

    def submit(self, data):
        self.items.append(data)


class _BlockingResponses:
    def __iter__(self):
        return self

    def __next__(self):
        threading.Event().wait(60)
        raise StopIteration


class QwenTTSTimeoutTest(unittest.TestCase):
    def setUp(self):
        self._old_dashscope = sys.modules.get("dashscope")
        dashscope = types.ModuleType("dashscope")

        class MultiModalConversation:
            @staticmethod
            def call(**_kwargs):
                return _BlockingResponses()

        dashscope.MultiModalConversation = MultiModalConversation
        sys.modules["dashscope"] = dashscope

    def tearDown(self):
        if self._old_dashscope is None:
            sys.modules.pop("dashscope", None)
        else:
            sys.modules["dashscope"] = self._old_dashscope

    def test_handle_emits_speech_end_when_qwen_stream_stalls(self):
        handler = HandlerQwenTTS()
        context = QwenTTSContext("session-1")
        context.config = QwenTTSConfig(sentence_timeout_s=0.05)
        context.data_submitter = _Submitter()

        text_def = DataBundleDefinition()
        text_def.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        text_bundle = DataBundle(text_def)
        text_bundle.set_main_data("或者跟我聊聊天呀。")
        text_bundle.add_meta("avatar_text_end", True)
        text_bundle.add_meta("speech_id", "speech-timeout")

        audio_def = DataBundleDefinition()
        audio_def.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        output_defs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=audio_def,
            )
        }

        thread = threading.Thread(
            target=handler.handle,
            args=(
                context,
                ChatData(type=ChatDataType.AVATAR_TEXT, data=text_bundle),
                output_defs,
            ),
            daemon=True,
        )
        thread.start()
        thread.join(0.5)

        self.assertFalse(thread.is_alive(), "TTS handler did not return after sentence timeout")
        self.assertTrue(context.data_submitter.items)
        last = context.data_submitter.items[-1]
        self.assertTrue(last.get_meta("avatar_speech_end"))
        audio = last.get_main_data()
        self.assertIsInstance(audio, np.ndarray)


if __name__ == "__main__":
    unittest.main()
