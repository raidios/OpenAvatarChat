from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))


class _FakeStream:
    def write(self, _data):
        pass

    def stop_stream(self):
        pass

    def close(self):
        pass


class _FakePyAudio:
    paInt16 = 8

    class PyAudio:
        def open(self, **_kwargs):
            return _FakeStream()

        def terminate(self):
            pass


sys.modules.setdefault("pyaudio", _FakePyAudio)
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("websockets", types.SimpleNamespace(exceptions=types.SimpleNamespace(ConnectionClosed=Exception)))
sys.modules.setdefault("camera", types.SimpleNamespace(SharedCamera=object))

from chat_client import AudioPlayer, ChatClient, SILENCE_FRAME, mic_sender  # noqa: E402


class _FakeWs:
    def __init__(self, stop_event):
        self.sent = []
        self._stop_event = stop_event

    async def send(self, data):
        self.sent.append(data)
        if len(self.sent) >= 2:
            self._stop_event.set()


class _FakeAudioSource:
    def __init__(self):
        self.gates = []

    def read(self, *_args):
        return SILENCE_FRAME

    def enable_wake_spotter(self, on):
        self.gates.append(bool(on))

    def stop_stream(self):
        pass

    def close(self):
        pass


class _FakeMutePlayer:
    is_playing = False
    mute_until = 0.0

    def __init__(self):
        self._states = [True, False]
        self._idx = 0

    @property
    def mic_should_mute(self):
        state = self._states[min(self._idx, len(self._states) - 1)]
        self.is_playing = state
        self._idx += 1
        return state


class ChatClientMotionInterruptTest(unittest.TestCase):
    def test_playback_done_does_not_wait_for_motion_pending(self) -> None:
        player = AudioPlayer()
        player.set_motion_pending(True)
        player.mark_audio_end()

        self.assertFalse(player.play_chunk())

        self.assertTrue(player.playback_done_event.is_set())
        self.assertFalse(player.is_playing)

    def test_human_speech_event_requests_action_cancel(self) -> None:
        cancelled = []
        client = ChatClient.__new__(ChatClient)
        client._on_human_speech_start = lambda: cancelled.append(True)

        handled = client.handle_server_json({"type": "human_speech_start"}, None)

        self.assertTrue(handled)
        self.assertEqual(cancelled, [True])

    def test_asr_text_requests_action_cancel_as_late_fallback(self) -> None:
        cancelled = []
        client = ChatClient.__new__(ChatClient)
        client._on_human_speech_start = lambda: cancelled.append(True)

        handled = client.handle_server_json({"type": "asr_text", "text": "你好"}, None)

        self.assertTrue(handled)
        self.assertEqual(cancelled, [True])

    def test_mic_sender_gates_wake_spotter_while_muted(self) -> None:
        async def run_once():
            stop_event = asyncio.Event()
            source = _FakeAudioSource()
            ws = _FakeWs(stop_event)
            await mic_sender(
                ws,
                None,
                _FakeMutePlayer(),
                stop_event,
                audio_source=source,
            )
            return source.gates

        self.assertEqual(asyncio.run(run_once()), [False, True])


if __name__ == "__main__":
    unittest.main()
