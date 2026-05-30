#!/usr/bin/env python3
"""Solve per-channel mic position on the 35-mm ring directly from
labelled DOA recordings, without brute-forcing all 720 permutations.

The acoustic delay at mic m for a far-field source at body angle B is

    tau_m(B) = -R cos(theta_m - B) / c
            = -R/c [cos(theta_m) cos(B) + sin(theta_m) sin(B)]

so the measured cross-correlation lag between mic m and the reference
channel (ch0) is

    lag(m, k) = tau_m(B_k) - tau_0(B_k)
              = -R/c [(u_m - u_0) cos(B_k) + (v_m - v_0) sin(B_k)]

where u_m = cos(theta_m), v_m = sin(theta_m). With K labelled body
angles {B_k} we get K equations per (m=1..5) in 2 unknowns
(a_m, b_m) = (u_m-u_0, v_m-v_0), solved by least-squares.

We then fix the gauge (i.e. recover the absolute positions u_0, v_0)
by enforcing the ring constraint |p_m| = 1 for all m. With 6
constraints on 2 unknowns it's overdetermined; we minimise the
residual sum of squared (|p_m|^2 - 1).

Outputs the per-channel body-frame angle theta_m, and from those
derives the channel-to-physical permutation that orders mics CCW
on the ring with mic 0 at the smallest positive angle.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SR = 16000
N_MICS = 6
RADIUS_M = 0.035
SOUND_SPEED = 343.0


def _load_8ch(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        n_ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").reshape(-1, n_ch).astype(
        np.float32) / 32768.0


def _gccphat_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 5,
                 sr: int = SR, f_low: float = 1500.0,
                 f_high: float = 4500.0) -> float:
    """GCC-PHAT lag of b vs a, restricted to a frequency band.

    The PHAT weighting whitens the cross-spectrum so the integration
    is over phase only (each frequency bin contributes equally). We
    further restrict to ``[f_low, f_high]`` to suppress the low band
    where room reflections dominate and the high band where spatial
    aliasing kicks in. For a 35-mm ring, the spatial-aliasing limit
    is ~5 kHz, and direct-path-to-reverberant ratio improves with
    frequency, so 1.5--4.5 kHz is the sweet spot.

    Returns sub-sample lag via parabolic fit.
    """
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= f_low) & (freqs <= f_high)
    cross = A * np.conj(B)
    cross[~band] = 0.0
    cross[band] /= (np.abs(cross[band]) + 1e-9)
    R = np.fft.irfft(cross, nfft)
    R = np.concatenate([R[-max_lag:], R[:max_lag + 1]])
    k = int(np.argmax(R))
    if 0 < k < len(R) - 1:
        y0, y1, y2 = R[k - 1], R[k], R[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return (k - max_lag) + float(delta)


def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> float:
    return _gccphat_lag(a, b, max_lag)


def _select_loud_frames(mics: np.ndarray, hop: int = 160,
                        top_k_pct: float = 50.0) -> np.ndarray:
    n_chunks = mics.shape[0] // hop
    chunked = mics[: n_chunks * hop].reshape(n_chunks, hop, mics.shape[1])
    rms = np.sqrt(np.mean(chunked ** 2, axis=(1, 2)) + 1e-12)
    thr = np.percentile(rms, 100.0 - top_k_pct)
    keep = rms >= thr
    keep_full = np.repeat(keep, hop)
    n = min(len(keep_full), n_chunks * hop)
    return mics[:n][keep_full[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str,
                    default="tests/data/doa_dataset")
    ap.add_argument("--reliable", type=float, nargs="+",
                    default=[-90, 0, 90, 180])
    ap.add_argument("--include-approx", action="store_true")
    ap.add_argument("--ref-channel", type=int, default=0,
                    help="reference channel for cross-correlation lags.")
    ap.add_argument("--max-lag-samples", type=int, default=5,
                    help="search window for xcorr (samples). 5 covers "
                         "the full diameter 2R/c = 3.27 samples plus "
                         "noise margin.")
    args = ap.parse_args()

    ds = Path(args.dataset)
    wavs = sorted(
        p for p in ds.glob("body_*deg_*.wav")
        if not p.name.endswith(".bf.wav")
        and not p.name.endswith(".ch0.wav")
    )
    truths_to_use = set(args.reliable)
    if args.include_approx:
        truths_to_use |= {-135.0, -45.0, 45.0, 135.0}

    clips = []
    for w in wavs:
        meta = json.loads(w.with_suffix(".json").read_text())
        truth = float(meta["body_az_deg"])
        if truth in truths_to_use:
            sig = _load_8ch(w)
            clips.append((truth, sig, w.name))
    if len(clips) < 3:
        print(f"[!] need at least 3 clips, got {len(clips)}")
        return 2
    print(f"[*] {len(clips)} clips: truths = "
          f"{sorted(c[0] for c in clips)}")
    print(f"[*] reference channel = ch{args.ref_channel}")
    print()

    ref = args.ref_channel
    other_mics = [m for m in range(N_MICS) if m != ref]

    # Step 1: measure lags vs ref channel.
    print("=== measured xcorr lags vs ch{} (samples) ===".format(ref))
    print(f"{'truth':>7}  " + "  ".join(f"ch{m}".rjust(8)
                                         for m in other_mics))
    lag_table: Dict[float, List[float]] = {}
    for truth, sig, _name in clips:
        mics = sig[:, :N_MICS]
        loud = _select_loud_frames(mics)
        lags = []
        for m in other_mics:
            lags.append(_xcorr_lag(loud[:, ref], loud[:, m],
                                   args.max_lag_samples))
        lag_table[truth] = lags
        cells = "  ".join(f"{l:+8.3f}" for l in lags)
        print(f"{truth:>+6.0f}°  {cells}")
    print()

    # Step 2: per-mic least-squares fit for (a_m, b_m) = (u_m-u_ref, v_m-v_ref).
    R_over_c_samples = RADIUS_M / SOUND_SPEED * SR  # ~1.633 samples
    truths = np.array([c[0] for c in clips])
    B_rad = np.deg2rad(truths)
    # Design matrix: each row k = [-(R/c)*cos B_k, -(R/c)*sin B_k]
    A = np.stack([-R_over_c_samples * np.cos(B_rad),
                  -R_over_c_samples * np.sin(B_rad)], axis=1)  # (K, 2)

    a_b: Dict[int, Tuple[float, float]] = {}   # per-mic (a_m, b_m)
    print(f"=== per-mic fit (a_m=u_m-u_ref, b_m=v_m-v_ref), residual ===")
    print(f"  ch  |    a_m       b_m   |  residual")
    for idx, m in enumerate(other_mics):
        y = np.array([lag_table[t][idx] for t in truths])
        sol, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
        a_m, b_m = float(sol[0]), float(sol[1])
        a_b[m] = (a_m, b_m)
        if residuals.size > 0:
            res = float(np.sqrt(residuals[0] / len(y)))
        else:
            pred = A @ sol
            res = float(np.sqrt(np.mean((y - pred) ** 2)))
        print(f"  ch{m} | {a_m:+8.4f}  {b_m:+8.4f}  | {res:.3f} samples")
    a_b[ref] = (0.0, 0.0)
    print()

    # Step 3: recover (u_ref, v_ref) by enforcing ring constraint
    # |p_m| = 1 for all m. We have p_m = (a_m + u_ref, b_m + v_ref).
    # |p_m|^2 = (a_m + u_ref)^2 + (b_m + v_ref)^2 = 1
    # Expanding: a_m^2 + b_m^2 + 2 a_m u_ref + 2 b_m v_ref + u_ref^2 + v_ref^2 = 1
    # Since |p_ref|^2 = u_ref^2 + v_ref^2 = 1 too, we get:
    # 2 a_m u_ref + 2 b_m v_ref = -(a_m^2 + b_m^2)
    # i.e.  M @ [u_ref, v_ref]^T = -d  where M_m = [2 a_m, 2 b_m]
    # and d_m = a_m^2 + b_m^2
    M = []
    d = []
    for m in range(N_MICS):
        if m == ref:
            continue
        a_m, b_m = a_b[m]
        M.append([2.0 * a_m, 2.0 * b_m])
        d.append(a_m ** 2 + b_m ** 2)
    M = np.array(M)
    d = np.array(d)
    # Least squares for (u_ref, v_ref):
    sol, _residuals, _rank, _ = np.linalg.lstsq(M, -d, rcond=None)
    u_ref, v_ref = float(sol[0]), float(sol[1])
    norm = float(np.sqrt(u_ref ** 2 + v_ref ** 2))
    print(f"=== gauge fix: (u_ref, v_ref) = ({u_ref:+.4f}, {v_ref:+.4f}), "
          f"|p_ref| = {norm:.4f} ===")
    if abs(norm - 1.0) > 0.2:
        print(f"[!] |p_ref| should be ~1.0 if mic 0 is on the ring; "
              f"got {norm:.3f}. Either ref channel isn't on the ring "
              f"(e.g. it's the bottom mic) or the data is too noisy.")
    print()

    # Step 4: per-mic absolute positions and angles.
    print(f"=== per-channel body-frame mic positions ===")
    print(f"  ch  |   u_m       v_m   |  |p_m|  |  theta_m (body deg)")
    angles = {}
    for m in range(N_MICS):
        a_m, b_m = a_b[m]
        u_m = u_ref + a_m
        v_m = v_ref + b_m
        n = float(np.sqrt(u_m ** 2 + v_m ** 2))
        ang = float(np.degrees(np.arctan2(v_m, u_m)))
        # Normalise to ring (project to unit circle in case of noise)
        if n > 1e-6:
            u_m /= n
            v_m /= n
        angles[m] = ang
        print(f"  ch{m} | {u_m:+7.4f}  {v_m:+7.4f}  |  {n:.3f}  |  "
              f"{ang:+7.1f}°")
    print()

    # Step 5: derive perm. For a CCW ring with mic 0 at smallest angle,
    # we sort channels by angle and that gives perm such that
    # logical_mic[k] = raw_channel[perm[k]] sits at angles increasing CCW.
    sorted_chs = sorted(range(N_MICS), key=lambda m: angles[m] % 360.0)
    perm = tuple(sorted_chs)
    sorted_angles = [angles[m] % 360.0 for m in sorted_chs]
    spacings = [(sorted_angles[(i + 1) % 6] - sorted_angles[i]) % 360.0
                for i in range(6)]
    print(f"=== sorted channels (CCW, mic 0 at smallest angle) ===")
    print(f"  perm = {perm}")
    print(f"  angles after perm: " +
          ", ".join(f"{a:+.1f}°" for a in sorted_angles))
    print(f"  spacings (should all be ~60°): " +
          ", ".join(f"{s:.1f}°" for s in spacings))
    spacing_err = max(abs(s - 60.0) for s in spacings)
    print(f"  max |spacing - 60°| = {spacing_err:.1f}°")
    print()

    # mic_yaw_offset = angle of logical mic 0 in body frame
    yaw = sorted_angles[0]
    yaw = ((yaw + 180.0) % 360.0) - 180.0
    print(f"=== suggested calibration ===")
    print(f"  M260C_MIC_PERM = {perm}")
    print(f"  mic_yaw_offset_deg = {yaw:+.1f}")
    print()

    # Sanity check: predict per-clip lags using the recovered geometry
    # and compare to measurements.
    print("=== sanity check: predicted vs measured lags ===")
    R_over_c = R_over_c_samples
    for truth, _sig, _name in clips:
        b_vec = np.array([np.cos(np.deg2rad(truth)),
                          np.sin(np.deg2rad(truth))])
        cells = []
        for idx, m in enumerate(other_mics):
            a_m, b_m = a_b[m]
            pred = -R_over_c * (a_m * b_vec[0] + b_m * b_vec[1])
            meas = lag_table[truth][idx]
            cells.append(f"{meas:+5.2f}/{pred:+5.2f}")
        print(f"{truth:>+6.0f}°  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
