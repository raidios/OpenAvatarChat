#!/usr/bin/env python3
"""
DTLN HEF setup helper.

Three modes:
  --check                    Check models/hailo/dtln*.hef presence, dump shapes.
  --compile-instructions     Print the (manual) Hailo Dataflow Compiler steps
                             needed to produce a HAILO10H DTLN HEF from the
                             upstream ONNX. Requires an x86 dev box with the
                             Hailo SDK; cannot run on the RPi5.
  --download-onnx <dir>      Pull DTLN-norm-40h.onnx + DTLN-aec.onnx into <dir>
                             (uses git/curl; needs network).

The runtime DTLN inference on Hailo is implemented in
``audio_frontend.backends.denoiser:HailoDenoiser`` and expects the HEF to live
at ``models/hailo/dtln_h10h.hef`` by default.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEF_DIR = REPO / "models" / "hailo"

INSTRUCTIONS = """
DTLN -> HAILO10H HEF compilation (run on x86 with Hailo SDK >= 5.3.0)

  1. Clone DTLN sources:
     git clone https://github.com/breizhn/DTLN
     cd DTLN
     # Use the streaming model with both LSTM hidden states exposed:
     python convert_weights_to_onnx.py --weights pretrained_model/model.h5 \\
         --target dtln_streaming.onnx --statefull True

  2. Verify ONNX inputs/outputs:
     python -c "
     import onnx
     m = onnx.load('dtln_streaming.onnx')
     print([(i.name, i.type.tensor_type.shape) for i in m.graph.input])
     print([(o.name, o.type.tensor_type.shape) for o in m.graph.output])
     "
     # Expect 2 inputs (audio + state) and 2 outputs (audio + state).

  3. Calibration set (5-10 clean+noisy pairs from VCTK or DNS dataset):
     python prepare_calib.py --dataset DNS-Challenge --out calib_set.npy

  4. Hailo Model Zoo / Dataflow Compiler (HAILO10H target!):
     hailo parser onnx dtln_streaming.onnx --hw-arch hailo10h
     hailo optimize dtln_streaming.har --calib-set-path calib_set.npy \\
         --hw-arch hailo10h
     hailo compiler dtln_streaming_optimized.har --hw-arch hailo10h \\
         -o dtln_h10h.hef

  5. Copy:
     scp dtln_h10h.hef pi5:{hef_dir}/dtln_h10h.hef

  6. Verify on the device:
     hailortcli benchmark --hef {hef_dir}/dtln_h10h.hef
     # Expected: < 5 ms per inference on HAILO10H

  7. Run the offline pipeline with --dns-backend hailo to confirm:
     {repo}/.venv/bin/python tests/run_pipeline_offline.py <wav> \\
         --pipeline full --dns-backend hailo
"""


def cmd_check() -> int:
    HEF_DIR.mkdir(parents=True, exist_ok=True)
    hefs = sorted(HEF_DIR.glob("dtln*.hef"))
    if not hefs:
        print(f"no DTLN HEFs in {HEF_DIR}")
        print("run with --compile-instructions to learn how to build one.")
        return 1
    for hef in hefs:
        print(f"== {hef.name} ({hef.stat().st_size/1e6:.2f} MB)")
        bin_path = shutil.which("hailortcli") or "/usr/bin/hailortcli"
        cp = subprocess.run([bin_path, "parse-hef", str(hef)],
                            capture_output=True, text=True, timeout=10)
        print(cp.stdout.strip())
    return 0


def cmd_compile_instructions() -> int:
    print(INSTRUCTIONS.format(hef_dir=HEF_DIR, repo=REPO))
    return 0


def cmd_download_onnx(dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    repo = "https://github.com/breizhn/DTLN.git"
    work = dst / "DTLN"
    if not work.exists():
        subprocess.run(["git", "clone", "--depth", "1", repo, str(work)],
                       check=True)
    print(f"DTLN clone at {work}; follow --compile-instructions to build HEF")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compile-instructions", action="store_true")
    ap.add_argument("--download-onnx", default=None)
    args = ap.parse_args()
    if args.check:
        return cmd_check()
    if args.compile_instructions:
        return cmd_compile_instructions()
    if args.download_onnx:
        return cmd_download_onnx(Path(args.download_onnx))
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
