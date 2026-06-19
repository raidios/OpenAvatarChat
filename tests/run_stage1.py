#!/usr/bin/env python3
"""
Stage-1 acceptance: runs the offline pipeline against every wav in dataset
and produces a markdown summary + JSON of metrics.

Required dataset layout:
  <dataset>/01_silence/raw_8ch.wav
  <dataset>/01_silence/gt.yaml         # optional
  <dataset>/04_far_noise/raw_8ch.wav
  ...

Each scenario can ship an optional gt.yaml describing:
  scenario: 04_far_noise
  expected_events: 5            # for KWS
  ground_truth_doa: [...]       # for DOA
  single_talk: [...]            # for AEC ERLE
  double_talk: [...]            # for AEC ERLE

The script runs `tests/run_pipeline_offline.py` once per scenario, then calls
the relevant metric scripts. Missing scenarios are skipped, not failed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Avoid tests/unittest/ shadowing stdlib unittest.
sys.path[:] = [p for p in sys.path if Path(p).resolve() != REPO / "tests"]

SCENARIOS = [
    "01_silence", "02_near_quiet", "03_far_quiet", "04_far_noise",
    "05_aec_echo_only", "06_aec_double_talk", "07_doa_static",
    "08_doa_moving", "09_multi_talk",
]


def run_pipeline(wav: Path, out_dir: Path, backend: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_clean = out_dir / "clean.wav"
    out_csv = out_dir / "doa.csv"
    cmd = [
        sys.executable, str(REPO / "tests" / "run_pipeline_offline.py"),
        str(wav), "--pipeline", "full",
        "--aec-backend", "auto",
        "--dns-backend", backend,
        "--out-clean", str(out_clean),
        "--out-csv", str(out_csv),
    ]
    subprocess.run(cmd, check=True)
    return out_clean, out_csv


def call_metric(script: Path, **kwargs) -> dict:
    args = [sys.executable, str(script)]
    for k, v in kwargs.items():
        if v is None:
            continue
        args.extend([f"--{k.replace('_', '-')}", str(v)])
    cp = subprocess.run(args, capture_output=True, text=True)
    if cp.returncode != 0:
        return {"error": cp.stderr.strip(), "stdout": cp.stdout.strip()}
    out_json = None
    for k, v in kwargs.items():
        if k == "out_json":
            out_json = v
    if out_json and Path(out_json).exists():
        return json.loads(Path(out_json).read_text())
    return {"stdout": cp.stdout}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(REPO / "tests" / "data" / "dataset_v1"))
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "cpu", "hailo"])
    ap.add_argument("--out", default=str(REPO / "tests" / "results" / "stage1.json"))
    args = ap.parse_args()

    ds = Path(args.dataset)
    out_root = Path(args.out).parent / f"stage1_runs_{args.backend}"
    summary = {"backend": args.backend, "scenarios": {}}

    if not ds.exists():
        print(f"[!] dataset not found: {ds}\n"
              "    record one with `make record DATASET=dataset_v1` "
              "(see tools/record_dataset.py)", file=sys.stderr)
        Path(args.out).write_text(json.dumps(
            {"backend": args.backend, "status": "no_dataset"}, indent=2))
        return 0

    for sc in SCENARIOS:
        sc_dir = ds / sc
        wav = sc_dir / "raw_8ch.wav"
        if not wav.exists():
            print(f"[skip] {sc}: missing {wav}")
            continue
        gt = sc_dir / "gt.yaml"
        run_dir = out_root / sc
        clean, doa_csv = run_pipeline(wav, run_dir, args.backend)
        sc_summary = {"clean": str(clean), "doa_csv": str(doa_csv)}

        if "07_doa" in sc and gt.exists():
            sc_summary["doa"] = call_metric(
                REPO / "tests" / "metrics" / "doa.py",
                csv=doa_csv, gt=gt,
                out_json=run_dir / "doa_metric.json",
            )
        if "05_aec" in sc or "06_aec" in sc:
            sc_summary["aec"] = call_metric(
                REPO / "tests" / "metrics" / "aec.py",
                near=wav, clean=clean,
                segments=gt if gt.exists() else None,
                out_json=run_dir / "aec_metric.json",
            )
        if sc in ("02_near_quiet", "04_far_noise"):
            sc_summary["dns"] = call_metric(
                REPO / "tests" / "metrics" / "dns.py",
                clean=clean, noisy=wav, no_dnsmos=False,
                out_json=run_dir / "dns_metric.json",
            )
        if sc in ("03_far_quiet", "04_far_noise"):
            sc_summary["kws"] = call_metric(
                REPO / "tests" / "metrics" / "kws.py",
                clean=clean,
                gt=gt if gt.exists() else None,
                out_json=run_dir / "kws_metric.json",
            )
        summary["scenarios"][sc] = sc_summary

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[*] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
