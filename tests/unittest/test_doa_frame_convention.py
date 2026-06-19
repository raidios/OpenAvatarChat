"""Regression test for the body↔mic azimuth conversion convention.

Both ``audio_frontend.dsp.offline.analyze_clip`` (validated by the
8-angle DOA accuracy test) and ``client.farfield_audio_source`` need to
agree on which direction the offset rotates. The empirical tap-test
pins logical mic 0 at body +``mic_yaw_offset_deg``, so:

    mic_az = body_az - offset
    body_az = mic_az + offset

Earlier versions of ``farfield_audio_source`` had the signs flipped,
which made the wake-orient controller rotate in the wrong direction
for sources roughly behind / beside the robot (the bug shifts the
reported azimuth by 2*offset, i.e. flips sign across the body forward
axis once the source is past ±60°).

This test pins the contract from both directions.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))


def _angle_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    """Compare two azimuths modulo 360°; ±180° are the same angle."""
    diff = abs(((a - b) + 180.0) % 360.0 - 180.0)
    return diff < tol


@pytest.mark.parametrize(
    "body_az,offset,expected_mic_az",
    [
        (0.0,    30.0,  -30.0),    # source forward → 30° CW of mic 0
        (30.0,   30.0,    0.0),    # source at mic 0
        (90.0,   30.0,   60.0),    # source 90° CCW of forward
        (-90.0,  30.0, -120.0),    # source 90° CW of forward
        (180.0,  30.0,  150.0),    # source behind
        (-150.0, 30.0,  180.0),    # behind-right wraps to ±180
        (60.0,   45.0,   15.0),    # different offset
    ],
)
def test_body_to_mic_matches_offline_pipeline_convention(
    body_az, offset, expected_mic_az
):
    """``body_to_mic`` must be the inverse of the analytic offline mapping."""
    from farfield_audio_source import FarfieldAudioSource

    src = FarfieldAudioSource.__new__(FarfieldAudioSource)
    src.mic_yaw_offset_deg = offset
    got = src._body_to_mic_az(body_az)
    assert _angle_eq(got, expected_mic_az), \
        f"body→mic for {body_az}° (offset {offset}°) gave {got}, want {expected_mic_az}"


@pytest.mark.parametrize(
    "mic_az,offset,expected_body_az",
    [
        (0.0,    30.0,   30.0),    # mic 0 sits at body +offset
        (-30.0,  30.0,    0.0),    # 30° CW of mic 0 == body forward
        (60.0,   30.0,   90.0),    # 60° CCW of mic 0
        (-120.0, 30.0,  -90.0),    # 120° CW of mic 0
        (150.0,  30.0,  180.0),    # 150° CCW → behind
        (180.0,  30.0, -150.0),    # wraps to -150° (CCW-positive)
    ],
)
def test_mic_to_body_matches_offline_pipeline_convention(
    mic_az, offset, expected_body_az
):
    from farfield_audio_source import FarfieldAudioSource

    src = FarfieldAudioSource.__new__(FarfieldAudioSource)
    src.mic_yaw_offset_deg = offset
    got = src._mic_to_body_az(mic_az)
    assert _angle_eq(got, expected_body_az), \
        f"mic→body for {mic_az}° (offset {offset}°) gave {got}, want {expected_body_az}"


def test_body_mic_round_trip_is_identity():
    """body→mic→body must reproduce the original (modulo 360°) for any az."""
    from farfield_audio_source import FarfieldAudioSource

    src = FarfieldAudioSource.__new__(FarfieldAudioSource)
    src.mic_yaw_offset_deg = 30.0
    for body in (-180.0, -90.0, -30.0, 0.0, 30.0, 90.0, 179.9):
        rt = src._mic_to_body_az(src._body_to_mic_az(body))
        # _wrap180 returns (-180, 180]; -180 wraps to 180, accept either.
        diff = abs(rt - body)
        if diff > 180.0:
            diff = abs(diff - 360.0)
        assert diff < 1e-6, f"round-trip body={body} → {rt}"


def test_offline_and_farfield_agree_on_offset_30():
    """Cross-check: offline.analyze_clip uses ``body = mic + offset``.

    The farfield class must produce the same body azimuth when fed the
    same mic_az / offset, otherwise the production wake-orient pipeline
    sees a different DOA than the validated accuracy test does.
    """
    from farfield_audio_source import FarfieldAudioSource

    src = FarfieldAudioSource.__new__(FarfieldAudioSource)
    src.mic_yaw_offset_deg = 30.0
    for mic_az in (-150, -90, -30, 0, 30, 60, 90, 150, 179):
        offline_body = ((mic_az + 30.0 + 180.0) % 360.0) - 180.0
        farfield_body = src._mic_to_body_az(float(mic_az))
        assert _angle_eq(farfield_body, offline_body), (
            f"mic_az={mic_az}: offline says body={offline_body}, "
            f"farfield says body={farfield_body}"
        )
