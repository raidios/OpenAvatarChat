#!/usr/bin/env python3
"""Interactive tap-based channel-to-physical-position calibration.

When the M260C raw stream is hacked out (no manufacturer SDK), the
SDK channel index for each physical mic is unknown. The simplest way
to recover that mapping is to tap physically next to each mic in a
known sequence and read which channel saw the loudest pulse.

A finger-tap close (~1 cm) to one mic produces a signal ~20-30 dB
louder on that mic than on the other ring mics (35-70 mm away). So
the channel with the largest peak amplitude during a detected tap is
unambiguously the one closest to your finger.

Workflow:

  1. Decide which physical mic on the ring is at body 0° (vehicle
     forward) -- or some other reference angle of your choice. Call
     this "position 0". The remaining 5 positions are then 60°
     apart, going CCW (counter-clockwise viewed from above).
  2. Run:
        .venv/bin/python tools/calibrate_mic_tap.py
     The tool prompts: "Tap mic at body 0° ... waiting for tap"
  3. Tap that mic with your fingernail. The tool prints the detected
     channel and prompts for the next position.
  4. Move CCW 60° on the ring and tap again. Repeat for 6 positions.
  5. The tool prints the final permutation and writes
     audio_frontend/dsp/mic_layout_calibration.json.

Use a fingernail or a pen tip; clean strikes work best. Keep the
robot in a quiet room (or just keep ambient sound below ~0.05 RMS).

The output `mic_channel_perm` should be plugged into
``audio_frontend/dsp/pipeline.py:M260C_MIC_PERM``. The
``mic_yaw_offset_deg`` matches the body angle you chose for
position 0 (default 0°).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(
    REPO_ROOT / "thirdparty" / "M2_SDK" / "live_stream" / "host"))

from live_stream import LiveStream, SAMPLE_RATE   # noqa: E402

N_MICS = 6
PERIOD_MS = 32                # iter_frames period
TAP_WINDOW_MS = 80            # peak measurement window centred on peak
COOLDOWN_MS = 800             # ignore further taps for this long
NOISE_FLOOR_WINDOW_MS = 1500  # how much history to use for noise est


class TapDetector:
    """Streaming tap detector.

    Keeps a rolling background noise floor estimate (per-channel max
    abs) and reports a tap when any mic channel's instantaneous peak
    is more than ``trigger_db`` above the floor. Returns the channel
    index with the highest peak in a short window centred on the
    triggering frame.
    """

    def __init__(self, trigger_db: float = 18.0,
                 cooldown_ms: int = COOLDOWN_MS,
                 tap_window_ms: int = TAP_WINDOW_MS,
                 noise_window_ms: int = NOISE_FLOOR_WINDOW_MS,
                 sample_rate: int = SAMPLE_RATE):
        self.trigger_lin = 10.0 ** (trigger_db / 20.0)
        self.tap_window = int(tap_window_ms * sample_rate / 1000)
        self.cooldown = int(cooldown_ms * sample_rate / 1000)
        self.history = deque(maxlen=int(
            noise_window_ms * sample_rate / 1000))
        self.cooldown_left = 0
        # Lookahead buffer to capture full tap waveform after trigger
        self.lookahead = []
        self.lookahead_remaining = 0
        self.trigger_origin = None    # tuple(buf, idx_in_buf) when armed

    def _max_per_chan(self, chunk: np.ndarray) -> np.ndarray:
        return np.max(np.abs(chunk[:, :N_MICS]), axis=0)

    def _noise_floor(self) -> np.ndarray:
        if not self.history:
            return np.zeros(N_MICS, dtype=np.float32) + 1e-4
        # robust: median of per-frame max-abs over history
        h = np.array(self.history)             # (T, N_MICS)
        return np.maximum(1e-4,
                          np.median(h, axis=0) * 1.4826)

    def update(self, chunk: np.ndarray
               ) -> Optional[Tuple[int, np.ndarray, float]]:
        """Feed one period of int16 8-ch (T, 8) audio.
        Returns (peak_channel, peak_per_channel, peak_db_above_floor)
        if a tap completes inside this chunk; otherwise None.
        """
        x = chunk.astype(np.float32) / 32768.0
        per_chan = self._max_per_chan(x)
        # Update rolling history (one entry per call) before checking
        self.history.append(per_chan.copy())
        if self.cooldown_left > 0:
            self.cooldown_left -= chunk.shape[0]
            self.lookahead.clear()
            self.lookahead_remaining = 0
            self.trigger_origin = None
            return None

        floor = self._noise_floor()
        # Already collecting lookahead from a previous trigger?
        if self.lookahead_remaining > 0:
            self.lookahead.append(x[:, :N_MICS])
            self.lookahead_remaining -= chunk.shape[0]
            if self.lookahead_remaining <= 0:
                full = np.concatenate(self.lookahead, axis=0)
                peaks = np.max(np.abs(full), axis=0)
                ch = int(np.argmax(peaks))
                db_above = 20.0 * np.log10(
                    peaks[ch] / max(floor[ch], 1e-6))
                self.cooldown_left = self.cooldown
                self.lookahead.clear()
                self.lookahead_remaining = 0
                self.trigger_origin = None
                return ch, peaks, db_above
            return None

        # Look for new trigger inside this chunk.
        ratio = per_chan / np.maximum(floor, 1e-6)
        if np.max(ratio) >= self.trigger_lin:
            self.lookahead = [x[:, :N_MICS]]
            self.lookahead_remaining = self.tap_window
            self.trigger_origin = (0, int(np.argmax(ratio)))
            self.lookahead_remaining -= chunk.shape[0]
            if self.lookahead_remaining <= 0:
                full = self.lookahead[0]
                peaks = np.max(np.abs(full), axis=0)
                ch = int(np.argmax(peaks))
                db_above = 20.0 * np.log10(
                    peaks[ch] / max(floor[ch], 1e-6))
                self.cooldown_left = self.cooldown
                self.lookahead.clear()
                self.lookahead_remaining = 0
                self.trigger_origin = None
                return ch, peaks, db_above
        return None


def _format_peaks(peaks: np.ndarray, peak_ch: int) -> str:
    parts = []
    pmax = float(np.max(peaks))
    for i, p in enumerate(peaks):
        rel_db = 20.0 * np.log10(p / max(pmax, 1e-9))
        marker = " <-" if i == peak_ch else "   "
        parts.append(f"  ch{i}: {p:.3f}  ({rel_db:+5.1f} dB){marker}")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-body-az", type=float, default=0.0,
                    help="body-frame azimuth of position 0 (the first "
                         "mic you tap). Default 0° (vehicle forward).")
    ap.add_argument("--ccw", action="store_true", default=True,
                    help="walk CCW around the ring (default).")
    ap.add_argument("--cw", dest="ccw", action="store_false",
                    help="walk CW around the ring.")
    ap.add_argument("--trigger-db", type=float, default=18.0,
                    help="trigger when any mic's peak exceeds the "
                         "rolling noise floor by this many dB (default "
                         "18 dB; raise if the room is noisy).")
    ap.add_argument("--out", type=str,
                    default="audio_frontend/dsp/mic_layout_calibration.json")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    args = ap.parse_args()

    direction = +1 if args.ccw else -1
    sense_str = "CCW" if args.ccw else "CW"

    print(f"=== M260C tap-test channel calibration ===")
    print(f"  6 mics on a 35 mm ring; tap each mic in {sense_str} order")
    print(f"  starting at body {args.start_body_az:+.0f}°")
    print(f"  trigger threshold: {args.trigger_db:.0f} dB above noise")
    print(f"  use a fingernail or pen tip for clean strikes")
    print()

    stream = LiveStream(host=args.host, port=args.port,
                        ensure_forward=False)
    stream.connect()
    print("[*] streaming, ambient noise calibrating...")

    # Warm up the noise floor for ~1 second before accepting taps.
    detector = TapDetector(trigger_db=args.trigger_db)
    iter_frames = stream.iter_frames(period_ms=PERIOD_MS)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        chunk = next(iter_frames)
        detector.update(chunk)
    print("[*] ready. Tap each mic when prompted.")
    print()

    taps: List[Tuple[int, int, np.ndarray, float]] = []
    body_angles: List[float] = []
    seen: set = set()
    duplicates = []

    try:
        for k in range(N_MICS):
            body_az = args.start_body_az + direction * k * 60.0
            body_az = ((body_az + 180.0) % 360.0) - 180.0
            body_angles.append(body_az)
            print(f"[#{k+1}/{N_MICS}] tap mic at body {body_az:+.0f}°  ", end="",
                  flush=True)
            t_start = time.monotonic()
            while True:
                chunk = next(iter_frames)
                result = detector.update(chunk)
                if result is not None:
                    ch, peaks, db_above = result
                    taps.append((k, ch, peaks, db_above))
                    print(f"-> ch{ch}  (+{db_above:.1f} dB above floor)")
                    if ch in seen:
                        duplicates.append((k, ch))
                    seen.add(ch)
                    print(_format_peaks(peaks, ch))
                    print()
                    break
                if time.monotonic() - t_start > 30.0:
                    print("[!] timeout (no tap detected). Aborting.")
                    return 1
    finally:
        stream.close()

    perm = [None] * N_MICS
    for k, ch, _peaks, _db in taps:
        if perm[k] is None:
            perm[k] = ch

    print("=== summary ===")
    for k, body_az, ch in zip(range(N_MICS), body_angles,
                               (t[1] for t in taps)):
        print(f"  position {k} (body {body_az:+5.0f}°)  -> ch{ch}")
    print()
    if duplicates:
        print(f"[!] WARNING: ch{[d[1] for d in duplicates]} fired twice. "
              f"You may have tapped the wrong mic or two mics are wired "
              f"to the same ADC input. Re-run if needed.")
    if any(p is None for p in perm) or len(set(perm)) != N_MICS:
        print(f"[!] perm is not a valid permutation: {perm}")
        return 1
    perm = tuple(perm)

    yaw_offset = float(args.start_body_az)
    print(f"=== calibration ===")
    print(f"  M260C_MIC_PERM    = {perm}")
    print(f"  mic_yaw_offset_deg = {yaw_offset:+.1f}")
    print(f"  sense (your tapping direction) = {sense_str}")
    print()
    print(f"To apply, edit audio_frontend/dsp/pipeline.py:")
    print(f"  M260C_MIC_PERM = {perm}")
    if args.ccw:
        print(f"(your tapping order matches the CCW convention used by")
        print(f" RingArray; no further sign change needed.)")
    else:
        print(f"WARNING: you tapped in CW order. If the rest of the "
              f"pipeline assumes CCW (default), feed the perm reversed:")
        rev = (perm[0],) + tuple(perm[N_MICS - i] for i in range(1, N_MICS))
        print(f"  M260C_MIC_PERM = {rev}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "M260C_MIC_PERM": list(perm),
        "mic_yaw_offset_deg": yaw_offset,
        "tap_sense": sense_str,
        "tap_strength_db_above_floor": [float(t[3]) for t in taps],
        "duplicates_warning": [list(d) for d in duplicates],
    }, indent=2))
    print(f"\n[*] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
