from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from handlers.wakeword.sherpa_kws.wakeword_handler_sherpa import (  # noqa: E402
    HandlerWakeWord,
    WakeWordConfig,
    WakeWordContext,
)


class FakeSharedStates:
    wake_session_active = True
    enable_vad = False
    last_interaction_time = 0.0


class FakeTimer:
    instances = []

    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.daemon = False
        FakeTimer.instances.append(self)

    def start(self):
        return None

    def cancel(self):
        return None


class WakeWordFailsafeDelayTest(unittest.TestCase):
    def test_failsafe_delay_includes_configured_extra_margin(self) -> None:
        handler = HandlerWakeWord()
        handler.config = WakeWordConfig(
            tts_sample_rate=24000,
            wake_reply_failsafe_extra_s=3.0,
        )
        handler.wake_audio = np.zeros(24000 * 2, dtype=np.float32)
        context = WakeWordContext("test-session")
        ss = FakeSharedStates()

        FakeTimer.instances.clear()
        with patch("threading.Timer", FakeTimer):
            handler._schedule_wake_listen_failsafe(context, ss)

        self.assertEqual(len(FakeTimer.instances), 1)
        self.assertAlmostEqual(FakeTimer.instances[0].delay, 5.35)


if __name__ == "__main__":
    unittest.main()
