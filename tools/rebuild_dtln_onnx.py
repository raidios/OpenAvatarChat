#!/usr/bin/env python3
"""
Rebuild DTLN as 2 sub-models with explicit LSTM state I/O (using DTLN_model.py's
seperation_kernel_with_states + unroll=True), load weights from a .h5, and
export each sub-model to a clean ONNX via modern tf2onnx.

Why this exists:
  - The keras2onnx-era model_1.onnx in pretrained_model/ has missing dtype
    fields on LSTM weight tensors → DFC 5.3 parse errors.
  - The keras2onnx-era model_2.onnx hits an internal "_split is not in list"
    bug in DFC 5.3.
  - Re-exporting with tf2onnx (current generation) and unroll=True yields ONNX
    files made of Dense+Mul+Add+Sigmoid+Tanh — no native LSTM op, no
    metadata gaps. Hailo DFC eats those cleanly.

Run from the cloned DTLN repo root (where DTLN_model.py lives):

    python rebuild_dtln_onnx.py \
        --weights pretrained_model/model.h5 \
        --out-prefix dtln_v2 \
        --opset 14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="pretrained_model/model.h5",
                    help="DTLN weights .h5 (default: pretrained_model/model.h5)")
    ap.add_argument("--out-prefix", default="dtln_v2",
                    help="output filename prefix (writes <prefix>_p1.onnx, "
                         "<prefix>_p2.onnx)")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--block-len", type=int, default=512)
    args = ap.parse_args()

    repo = Path(".").resolve()
    if not (repo / "DTLN_model.py").exists():
        print(f"[!] DTLN_model.py not found in {repo}; run from the DTLN repo root",
              file=sys.stderr)
        return 1
    sys.path.insert(0, str(repo))

    import tensorflow as tf
    import tf2onnx
    import onnx
    from tensorflow.keras.layers import Conv1D, Input, Multiply
    from tensorflow.keras.models import Model

    from DTLN_model import DTLN_model, InstantLayerNormalization  # type: ignore

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[!] weights file missing: {weights_path}", file=sys.stderr)
        return 1
    norm_stft = "_norm_" in weights_path.name
    print(f"[*] weights={weights_path} norm_stft={norm_stft}")

    # 1) build the stateful full DTLN to load weights from the .h5 cleanly.
    m = DTLN_model()
    m.blockLen = args.block_len
    m.build_DTLN_model_stateful(norm_stft=norm_stft)
    m.model.load_weights(str(weights_path))
    n_layer = m.numLayer
    n_units = m.numUnits
    n_freq = m.blockLen // 2 + 1
    n_block = m.blockLen
    n_enc = m.encoder_size
    print(f"[*] numLayer={n_layer} numUnits={n_units} blockLen={n_block} "
          f"encoder_size={n_enc}")

    # 2) build sub-model 1 with explicit state I/O (unroll=True via
    # seperation_kernel_with_states).
    mag = Input(batch_shape=(1, 1, n_freq), name="mag_in")
    s1_in = Input(batch_shape=(1, n_layer, n_units, 2), name="state_in_1")
    if norm_stft:
        mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
    else:
        mag_norm = mag
    mask1, s1_out = m.seperation_kernel_with_states(
        n_layer, n_freq, mag_norm, s1_in
    )
    model_1 = Model(inputs=[mag, s1_in], outputs=[mask1, s1_out],
                    name="dtln_p1")

    # 3) build sub-model 2.
    frame_in = Input(batch_shape=(1, 1, n_block), name="frame_in")
    s2_in = Input(batch_shape=(1, n_layer, n_units, 2), name="state_in_2")
    enc = Conv1D(n_enc, 1, strides=1, use_bias=False)(frame_in)
    enc_norm = InstantLayerNormalization()(enc)
    mask2, s2_out = m.seperation_kernel_with_states(
        n_layer, n_enc, enc_norm, s2_in
    )
    est = Multiply()([enc, mask2])
    dec = Conv1D(n_block, 1, padding="causal", use_bias=False)(est)
    model_2 = Model(inputs=[frame_in, s2_in], outputs=[dec, s2_out],
                    name="dtln_p2")

    # 4) split weights between the two sub-models exactly like
    # create_tf_lite_model does.
    if norm_stft:
        n_elem_1 = 2 + n_layer * 3 + 2
    else:
        n_elem_1 = n_layer * 3 + 2
    weights = m.model.get_weights()
    print(f"[*] full model weights: {len(weights)} tensors; "
          f"first {n_elem_1} -> p1, rest -> p2")
    model_1.set_weights(weights[:n_elem_1])
    model_2.set_weights(weights[n_elem_1:])

    # 5) export to ONNX via tf2onnx (modern, opset 14+).
    spec_1 = (
        tf.TensorSpec((1, 1, n_freq), tf.float32, name="mag_in"),
        tf.TensorSpec((1, n_layer, n_units, 2), tf.float32, name="state_in_1"),
    )
    spec_2 = (
        tf.TensorSpec((1, 1, n_block), tf.float32, name="frame_in"),
        tf.TensorSpec((1, n_layer, n_units, 2), tf.float32, name="state_in_2"),
    )
    out_p1 = f"{args.out_prefix}_p1.onnx"
    out_p2 = f"{args.out_prefix}_p2.onnx"
    print(f"[*] exporting {out_p1} (opset {args.opset})")
    onnx_model_1, _ = tf2onnx.convert.from_keras(
        model_1, input_signature=spec_1, opset=args.opset)
    onnx.save(onnx_model_1, out_p1)
    print(f"[*] exporting {out_p2} (opset {args.opset})")
    onnx_model_2, _ = tf2onnx.convert.from_keras(
        model_2, input_signature=spec_2, opset=args.opset)
    onnx.save(onnx_model_2, out_p2)

    print()
    for path in (out_p1, out_p2):
        m_ = onnx.load(path)
        print(f"=== {path} ===")
        print("  inputs:")
        for x in m_.graph.input:
            s = [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]
            print(f"    {x.name:25s} shape={s}")
        print("  outputs:")
        for x in m_.graph.output:
            s = [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]
            print(f"    {x.name:25s} shape={s}")
        ops = sorted({n.op_type for n in m_.graph.node})
        has_lstm = "LSTM" in ops
        print(f"  ops: {ops}")
        print(f"  native LSTM op present? {'YES (bad)' if has_lstm else 'no (good)'}")
        size_kb = Path(path).stat().st_size / 1024
        print(f"  size: {size_kb:.0f} KB")

    print(f"\n[+] done. Try parsing now:")
    print(f"  hailo parser onnx {out_p1} --hw-arch hailo10h \\")
    print(f"      --start-node-names mag_in state_in_1 \\")
    print(f"      --end-node-names <see ONNX output names above> -y")
    return 0


if __name__ == "__main__":
    sys.exit(main())
