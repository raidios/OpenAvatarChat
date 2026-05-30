#!/usr/bin/env python3
"""Reverse-engineer the M260C channel→physical-mic-position mapping
from the labelled DOA dataset.

Idea: for each labelled recording (body_az known), the mic that's
physically closest to the source should have (a) the highest RMS
energy (in real life mics are not perfectly omni; the side-on mic
also benefits from the housing baffle) and (b) the earliest
acoustic arrival relative to the others. Cross-correlating each
mic against ch0 yields a (per-mic) sample-domain time delay; given
known geometry we can predict the expected per-mic delay for each
ground-truth angle, then find the mic-position permutation +
sign convention that best fits the data.

We DON'T trust the mic_yaw_offset_deg or the assumed CCW-from-ch0
ordering until they're validated this way.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SR = 16000
N_MICS = 6
RADIUS_M = 0.033
SOUND_SPEED = 343.0


def _load_8ch(path: Path):
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    arr = np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch)
    return arr.astype(np.float32) / 32768.0, sr


def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Sub-sample lag of b vs a via parabolic fit on integer-lag xcorr."""
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    R = np.fft.irfft(A * np.conj(B), nfft)
    R = np.concatenate([R[-max_lag:], R[:max_lag + 1]])
    k = int(np.argmax(R))
    if 0 < k < len(R) - 1:
        y0, y1, y2 = R[k - 1], R[k], R[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return (k - max_lag) + float(delta)


def _expected_delay_samples(az_body_deg: float, mic_pos_deg: float,
                             sr: int = SR, radius_m: float = RADIUS_M):
    """Expected arrival delay (samples) at a mic placed at mic_pos_deg
    in body frame, for a far-field source at body azimuth az_body_deg.
    Positive delay = mic receives the wavefront LATER than the array
    centre. Reference is the array centre (so cross-correlation between
    two mics m1, m2 should give delay_m2 - delay_m1).
    """
    # Mic position vector
    th_mic = np.deg2rad(mic_pos_deg)
    mx = radius_m * np.cos(th_mic)
    my = radius_m * np.sin(th_mic)
    # Source direction (unit vector pointing AT the source from origin)
    th_src = np.deg2rad(az_body_deg)
    sx = np.cos(th_src)
    sy = np.sin(th_src)
    # Wave-front arrival at mic relative to array centre: mic at +s direction
    # gets it earlier (negative delay). delay = -mic.dot(s) / c
    delay_s = -(mx * sx + my * sy) / SOUND_SPEED
    return delay_s * sr


def main() -> int:
    ds = REPO_ROOT / "tests/data/doa_dataset"
    wavs = sorted(
        p for p in ds.glob("body_*deg_*.wav")
        if not p.name.endswith(".bf.wav")
        and not p.name.endswith(".ch0.wav")
    )
    if not wavs:
        print(f"no recordings in {ds}/")
        return 2

    print(f"[*] {len(wavs)} clips in {ds}/")
    print()

    # 1) Per-clip per-channel RMS  -> which mic is closest to the source
    print("=== per-channel RMS (ch0..ch7) ===")
    print(f"{'clip':<32} {'truth':>7} | "
          + " ".join(f"ch{i}".rjust(7) for i in range(8))
          + "  | argmax")
    rms_table = {}
    for w in wavs:
        sig, sr = _load_8ch(w)
        meta_path = w.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        truth = float(meta.get("body_az_deg", 0))
        rms = np.sqrt(np.mean(sig ** 2, axis=0) + 1e-12)
        rms_table[truth] = rms
        argmax = int(np.argmax(rms[:N_MICS]))
        cells = " ".join(f"{r:7.4f}" for r in rms)
        print(f"{w.name[:32]:<32} {truth:>+6.0f}° | {cells}  | ch{argmax}")
    print()

    # 2) Per-clip cross-correlation: ch_m vs ch0, in samples. Compare to
    #    expected delays for several candidate mic-position layouts.
    print("=== measured per-channel lag vs ch0 (samples) ===")
    print(f"{'clip':<32} {'truth':>7} | " +
          " ".join(f"ch{i}".rjust(7) for i in range(N_MICS)))
    measured_lags = {}
    for w in wavs:
        sig, _ = _load_8ch(w)
        meta = json.loads(w.with_suffix(".json").read_text())
        truth = float(meta["body_az_deg"])
        # Use the loudest 50% of frames for the cross-correlation, same
        # idea as dominant_doa - reduces influence of stray noise.
        hop = 160
        n_chunks = sig.shape[0] // hop
        chunked = sig[: n_chunks * hop, :N_MICS].reshape(n_chunks, hop, N_MICS)
        ch_rms = np.sqrt(np.mean(chunked ** 2, axis=(1, 2)))
        thr = np.percentile(ch_rms, 50.0)
        keep = ch_rms >= thr
        keep_full = np.repeat(keep, hop)
        n = min(len(keep_full), n_chunks * hop)
        sig_kept = sig[:n, :N_MICS][keep_full[:n]]
        if len(sig_kept) < 1024:
            continue
        max_lag = int(np.ceil(2 * RADIUS_M / SOUND_SPEED * SR)) + 2  # ~3-4
        lags = np.zeros(N_MICS)
        for m in range(N_MICS):
            lags[m] = _xcorr_lag(sig_kept[:, 0], sig_kept[:, m], max_lag)
        measured_lags[truth] = lags
        cells = " ".join(f"{l:+7.3f}" for l in lags)
        print(f"{w.name[:32]:<32} {truth:>+6.0f}° | {cells}")
    print()

    # 3) Two-stage fit:
    #    a) reliable-only fit: use ONLY {-90, 0, +90, 180} (user said these
    #       four cardinal angles are well-measured) and search over both
    #       senses (CW/CCW), continuous rotation 0..360° (1° steps), and
    #       radius 0.020..0.050 m. We get the best-fit array geometry
    #       free of bias from the approximate diagonal labels.
    #    b) fix the geometry from (a), then evaluate MAD against ALL 8
    #       angles to estimate diagonal-label noise.
    reliable_truths = [-90.0, 0.0, 90.0, 180.0]
    reliable = {t: measured_lags[t] for t in reliable_truths
                if t in measured_lags}
    if len(reliable) < 4:
        print(f"[!] only {len(reliable)} reliable angles available, "
              f"layout fit may be noisy")

    base_ccw = np.array([0, 60, 120, 180, 240, 300], dtype=float)
    radii = np.arange(0.012, 0.050 + 1e-9, 0.0005)
    rotations = np.arange(0.0, 360.0, 1.0)

    def _layout_for(sense: str, rot_deg: float) -> np.ndarray:
        if sense == "CCW":
            return (base_ccw + rot_deg) % 360.0
        return (-base_ccw + rot_deg) % 360.0

    def _expected_lag_relative(layout, az_body, radius):
        d = np.array([_expected_delay_samples(az_body, layout[m],
                                              radius_m=radius)
                      for m in range(N_MICS)])
        return d - d[0]

    def _mad(layout, radius, table) -> float:
        residuals = []
        for truth, lags in table.items():
            exp_rel = _expected_lag_relative(layout, truth, radius)
            residuals.append(np.abs(lags - exp_rel))
        return float(np.mean(np.concatenate(residuals)))

    best_mad = float("inf")
    best_sense = best_rot = best_radius = None
    for sense in ("CCW", "CW"):
        for rot in rotations:
            layout = _layout_for(sense, rot)
            for r in radii:
                mad = _mad(layout, r, reliable)
                if mad < best_mad:
                    best_mad = mad
                    best_sense = sense
                    best_rot = rot
                    best_radius = r

    print("=== free-rotation fit on RELIABLE angles only "
          f"({sorted(reliable.keys())}) ===")
    print(f"  sense    = {best_sense}")
    print(f"  rotation = {best_rot:.1f}°  (pos of channel 0 in body frame)")
    print(f"  radius   = {best_radius * 1000:.2f} mm")
    print(f"  MAD      = {best_mad:.3f} samples "
          f"({best_mad / SR * 1e6:.1f} µs path)")

    best_layout = _layout_for(best_sense, best_rot)
    print(f"  ch positions in body frame: "
          f"{[round(x, 1) for x in best_layout]}")

    print()
    print("=== same geometry, evaluated on ALL angles ===")
    print(f"{'truth':>7}   {'MAD':>6}")
    for truth, lags in measured_lags.items():
        exp_rel = _expected_lag_relative(best_layout, truth, best_radius)
        mad = float(np.mean(np.abs(lags - exp_rel)))
        marker = " (reliable)" if truth in reliable else " (approx)"
        print(f"{truth:>+6.0f}°   {mad:6.3f}{marker}")

    # Channel permutation that maps physical channels into a CCW,
    # ref_axis=0° layout that RingArray expects.
    target = (np.arange(N_MICS) * 60.0) % 360.0
    perm = []
    for tgt in target:
        diffs = np.abs(((best_layout - tgt) + 180.0) % 360.0 - 180.0)
        perm.append(int(np.argmin(diffs)))

    yaw_residual = best_rot - perm[0] * 60.0
    if best_sense == "CW":
        yaw_residual = best_rot - 0.0   # ch perm[0] sits at best_rot deg

    # The simpler statement: where does logical-mic-0 (i.e. raw_mics[perm[0]])
    # sit in body frame after the permutation? It sits at body angle
    # closest to 0°, by construction: best_layout[perm[0]] ~= 0°.
    actual_pos_of_mic0 = best_layout[perm[0]]
    actual_pos_of_mic0 = float(((actual_pos_of_mic0 + 180.0) % 360.0) - 180.0)
    print()
    print(f"=> reorder channels:  fed_mics[k] = raw_mics[perm[k]],  "
          f"perm = {perm}")
    print(f"   after permutation: logical mic 0 sits at body "
          f"{actual_pos_of_mic0:+.1f}°")
    print(f"   so mic_yaw_offset_deg should be {actual_pos_of_mic0:+.1f}°")
    print(f"   (recall: yaw_offset = body azimuth of logical mic 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
