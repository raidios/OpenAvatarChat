from __future__ import annotations

import os
import sys
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

fake_live_stream = types.ModuleType("live_stream")
fake_live_stream.LiveStream = object
sys.modules.setdefault("live_stream", fake_live_stream)

from farfield_audio_source import FarfieldAudioSource


class FakePipeline:
    latest_doa_deg = 0.0

    def __init__(self, wake_res=None, dom_res=None, dom_by_window=None) -> None:
        self._wake_res = wake_res
        self._dom_res = dom_res
        self._dom_by_window = dom_by_window or {}

    def wake_window_doa(self):
        return self._wake_res

    def dominant_doa(self, top_k_pct=30.0, window_ms=None):
        if window_ms in self._dom_by_window:
            return self._dom_by_window[window_ms]
        return self._dom_res


def make_source(fake_pipeline: FakePipeline, min_peak: float = 1.20):
    src = object.__new__(FarfieldAudioSource)
    src._pipeline = fake_pipeline
    src.mic_yaw_offset_deg = 30.0
    src.wake_doa_min_peak = min_peak
    src.wake_doa_mid_peak = 1.10
    src.wake_doa_consensus_deg = 45.0
    return src


class FarfieldWakeDoaConfidenceTest(unittest.TestCase):
    def test_rejects_low_confidence_dominant_snapshot(self) -> None:
        src = make_source(FakePipeline(dom_res=(260.0, 1.05)))

        self.assertIsNone(src.wake_doa_snapshot_deg)

    def test_accepts_high_confidence_dominant_snapshot(self) -> None:
        src = make_source(FakePipeline(dom_res=(225.0, 1.23)))

        self.assertAlmostEqual(src.wake_doa_snapshot_deg, -105.0)

    def test_accepts_mid_confidence_consensus(self) -> None:
        src = make_source(FakePipeline(dom_by_window={
            600: (60.0, 1.11),
            1000: (65.0, 1.12),
            1500: (70.0, 1.10),
        }))

        self.assertAlmostEqual(src.wake_doa_snapshot_deg, 95.0)

    def test_rejects_mid_confidence_disagreement(self) -> None:
        src = make_source(FakePipeline(dom_by_window={
            600: (60.0, 1.11),
            1000: (220.0, 1.12),
            1500: (300.0, 1.10),
        }))

        self.assertIsNone(src.wake_doa_snapshot_deg)

    def test_applies_gate_to_wake_window_snapshot_too(self) -> None:
        src = make_source(FakePipeline(wake_res=(60.0, 1.10)))

        self.assertIsNone(src.wake_doa_snapshot_deg)


if __name__ == "__main__":
    unittest.main()
