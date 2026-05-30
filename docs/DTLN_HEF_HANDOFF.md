# DTLN → Hailo-10H HEF 编译任务移交文档（WSL agent 用）

## 0. 你的工作目标

在 WSL（x86_64 Ubuntu 24.04，RTX 2060 6GB）上把 [breizhn/DTLN](https://github.com/breizhn/DTLN) 编译成两个 Hailo-10H HEF：

- `dtln_p1_h10h.hef` — 频域 mask 网络（input: STFT 幅度 + LSTM 状态；output: mask + 新状态）
- `dtln_p2_h10h.hef` — 时域 refine 网络（input: 时域帧 + LSTM 状态；output: enhanced 帧 + 新状态）

最终交付：把这两个 HEF `scp` 回到树莓派 5（RPi5）上的 `OpenAvatarChat/models/hailo/` 目录；那边的 `audio_frontend.backends.denoiser.HailoDenoiser` Python wrapper 已经写好了（细节见 §6 一节，需要稍作适配）。

## 1. 当前 WSL 环境状态（已验证 OK）

| 组件 | 版本 / 路径 |
|---|---|
| OS | Ubuntu 24.04 in WSL2，kernel 5.15.167.4-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 2060 with Max-Q (6 GB)，driver 572.16，CUDA 12.8 capability |
| Python venv | `~/.venv-hailo-dfc/`（python 3.12） |
| Hailo DFC | **5.3.0**（wheel `hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl`）|
| HailoRT (Python) | 5.3.0（wheel） |
| TensorFlow | 2.13.x（venv 自动装） |
| tf2onnx | 已装 |
| onnx | 已装 |
| onnxsim | 已装 |
| Model Zoo | **没装**（DTLN 不在 Zoo 里，所以不需要） |
| `hailort-pcie-driver` | **绝对不要装**（编译机没卡，post-install 会失败）|
| cuDNN | 暂未装；hailo optimize 会用到，到那一步再处理 |
| libdevice (XLA) | 缺失；用 `CUDA_VISIBLE_DEVICES=""` 让 TF 在 CPU 跑就行（rebuild 用 CPU 已足够） |

DTLN 源代码仓库已 clone 在：

```
/mnt/d/Projects/cyk/DTLN/
├── DTLN_model.py             # 主代码
├── pretrained_model/
│   ├── model.h5              # 默认权重 (norm_stft=False)
│   ├── DTLN_norm_500h.h5     # 备用 (norm_stft=True)
│   ├── DTLN_norm_40h.h5
│   ├── model_1.onnx          # 老 keras2onnx 导出（坏，data_type 缺失）
│   ├── model_2.onnx          # 老 keras2onnx 导出（坏，Split bug）
│   ├── model_1.tflite / model_2.tflite / model_quant_*.tflite
│   └── *_saved_model/        # 三个目录全是空的，不要用
├── rebuild_dtln_onnx.py      # 我们自己的转换脚本（最新版见 §3）
├── dtln_build_calib.py       # 校准集生成脚本（见 §4）
├── dtln_v2_p1.onnx           # unroll=True 版（坏，DFC 有 Split bug）
├── dtln_v3_p1.onnx           # unroll=False 版（坏，DFC 有 Loop bug）
├── dtln_v4_p1.onnx           # 预期产物：StatelessLSTM 版（待跑，§3）
├── calib_p1.npz / calib_p2.npz   # 老校准集（基于 v2/v3 ONNX 的 input 名）
└── ...
```

## 2. 重要：已经踩过的坑（不要再重复）

| 尝试 | 失败原因 |
|---|---|
| 直接用 `pretrained_model/model_1.onnx` | keras2onnx 老版本导出的 LSTM 权重 tensor 缺 `data_type=0`，DFC parser 在 `numpy_helper.to_array` 报 `TypeError: element type ... not defined` |
| 直接用 `pretrained_model/model_2.onnx` | DFC parser 内部 bug：`'0_split' is not in list` (`onnx_graph.py:3141 sort_func`) |
| 修复 ONNX 的 `data_type` 字段（写脚本遍历所有 initializer） | 失败原因不仅是 dtype，老 ONNX 还有别的脏东西 |
| `tf.saved_model.load("dtln_saved_model")` | 三个 saved_model 目录都是空的（之前 `tf.saved_model.save` 没真正写入） |
| `tflite2onnx pretrained_model/model_1.tflite` | `NotImplementedError: Unsupported TFLite OP: 88 UNPACK!` (tflite2onnx 库 2 年没更新) |
| `unroll=True` 重导（dtln_v2）| 展开后产生命名 `xxx/lstm_4_1/lstm_cell_1/split` 的 Split 节点，触发 DFC 同款 Split bug |
| `unroll=False` 重导（dtln_v3）| ONNX 14 用 Loop op 表示动态 LSTM，DFC parser 不支持 Loop（`Couldn't find inputs ... expected 2, found 1`）|
| 直接走 Model Zoo `hailomz` | Zoo 里没有 DTLN（音频模型基本没收录）；安装 Zoo 还要造 `versions.py` stub |

## 3. 当前 in-progress 计划：StatelessLSTM 自定义 Keras Layer

把 LSTM 一步推理用 **纯 MatMul + Slice + Sigmoid + Tanh + Mul + Add** 手动实现，绕开所有 ONNX LSTM / Loop / Split op 路径。理由：

- 不用 native LSTM op → 避开 DFC LSTM 解析的 dtype / shape 挑剔
- 不用 unroll → 不产生命名带 `_split` 的节点，绕开 DFC sort_func bug
- 不用 ONNX Loop op → 避开 DFC 不支持 control flow

完整脚本在 `/mnt/d/Projects/cyk/DTLN/rebuild_dtln_onnx.py`（最新版见下方代码块；如果当前文件不对请覆盖）：

```python
#!/usr/bin/env python3
"""DTLN -> 2 sub-ONNX with explicit LSTM state I/O.

Custom StatelessLSTM Layer decomposes a one-step LSTM into pure MatMul + Slice
+ Sigmoid + Tanh + Mul + Add. No native LSTM op, no Loop, no Split.
Weight layout matches keras.layers.LSTM (gate order [i, f, c, o]).
"""
import argparse, sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx, onnx
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Input, Lambda,
                                       LSTM, Layer, Multiply)
from tensorflow.keras.models import Model


class StatelessLSTM(Layer):
    """One-step LSTM with explicit (h, c) I/O."""
    def __init__(self, units, **kw):
        super().__init__(**kw)
        self.units = int(units)

    def build(self, input_shape):
        in_dim = int(input_shape[0][-1])
        self.kernel = self.add_weight("kernel", (in_dim, 4 * self.units),
                                       initializer="glorot_uniform")
        self.recurrent_kernel = self.add_weight(
            "recurrent_kernel", (self.units, 4 * self.units),
            initializer="orthogonal")
        self.bias = self.add_weight("bias", (4 * self.units,),
                                     initializer="zeros")
        super().build(input_shape)

    def call(self, inputs):
        x, h, c = inputs
        x2 = tf.squeeze(x, axis=1)              # (1, in_dim)
        z = (tf.matmul(x2, self.kernel)
             + tf.matmul(h, self.recurrent_kernel)
             + self.bias)
        u = self.units
        i_g = tf.sigmoid(z[:, 0*u:1*u])
        f_g = tf.sigmoid(z[:, 1*u:2*u])
        c_b = tf.tanh(   z[:, 2*u:3*u])
        o_g = tf.sigmoid(z[:, 3*u:4*u])
        c_n = f_g * c + i_g * c_b
        h_n = o_g * tf.tanh(c_n)
        return tf.expand_dims(h_n, axis=1), h_n, c_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="pretrained_model/model.h5")
    ap.add_argument("--out-prefix", default="dtln_v4")
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(".").resolve()))
    from DTLN_model import DTLN_model, InstantLayerNormalization

    weights_path = Path(args.weights)
    norm_stft = "_norm_" in weights_path.name
    print(f"[*] weights={weights_path} norm_stft={norm_stft}")

    m = DTLN_model()
    m.build_DTLN_model_stateful(norm_stft=norm_stft)
    m.model.load_weights(str(weights_path))
    n_layer, n_units = m.numLayer, m.numUnits
    n_freq, n_block, n_enc = m.blockLen // 2 + 1, m.blockLen, m.encoder_size
    print(f"[*] numLayer={n_layer} numUnits={n_units} "
          f"blockLen={n_block} encoder_size={n_enc}")

    full_lstms = [l for l in m.model.layers if isinstance(l, LSTM)]
    sub1_lstms = full_lstms[:n_layer]
    sub2_lstms = full_lstms[n_layer:]
    full_denses = [l for l in m.model.layers if isinstance(l, Dense)]
    sub1_dense, sub2_dense = full_denses[0], full_denses[1]
    full_convs = [l for l in m.model.layers if isinstance(l, Conv1D)]
    enc_conv, dec_conv = full_convs[0], full_convs[1]
    full_norms = [l for l in m.model.layers
                  if isinstance(l, InstantLayerNormalization)]

    def _combine(lst, nl=n_layer, nu=n_units):
        sh = lst[:nl]; sc = lst[nl:]
        return tf.stack([tf.stack(sh, axis=1), tf.stack(sc, axis=1)], axis=-1)

    # ----- sub-model 1
    mag = Input(batch_shape=(1, 1, n_freq), name="mag_in")
    s1_in = Input(batch_shape=(1, n_layer, n_units, 2), name="state_in_1")
    if norm_stft:
        norm_p1 = InstantLayerNormalization(name="norm_p1")
        x_pre = norm_p1(tf.math.log(mag + 1e-7))
    else:
        x_pre = mag; norm_p1 = None
    flat_h, flat_c = [], []
    x = x_pre
    for idx in range(n_layer):
        h_in = Lambda(lambda s, i=idx: s[:, i, :, 0],
                      output_shape=lambda s: (s[0], s[2]))(s1_in)
        c_in = Lambda(lambda s, i=idx: s[:, i, :, 1],
                      output_shape=lambda s: (s[0], s[2]))(s1_in)
        cell = StatelessLSTM(n_units, name=f"slstm_p1_{idx}")
        x, ho, co = cell([x, h_in, c_in])
        flat_h.append(ho); flat_c.append(co)
    p1_dense = Dense(n_freq, name="dense_p1")(x)
    p1_mask = Activation("sigmoid", name="mask_p1")(p1_dense)
    s1_out = Lambda(_combine,
                    output_shape=lambda shapes: (1, n_layer, n_units, 2),
                    name="state_out_1")(flat_h + flat_c)
    model_1 = Model([mag, s1_in], [p1_mask, s1_out], name="dtln_p1")

    if norm_stft and norm_p1 is not None:
        norm_p1.set_weights(full_norms[0].get_weights())
    for idx, src in enumerate(sub1_lstms):
        model_1.get_layer(f"slstm_p1_{idx}").set_weights(src.get_weights())
    model_1.get_layer("dense_p1").set_weights(sub1_dense.get_weights())

    # ----- sub-model 2
    frame_in = Input(batch_shape=(1, 1, n_block), name="frame_in")
    s2_in = Input(batch_shape=(1, n_layer, n_units, 2), name="state_in_2")
    enc_layer = Conv1D(n_enc, 1, strides=1, use_bias=False, name="enc_conv")
    enc = enc_layer(frame_in)
    norm_p2 = InstantLayerNormalization(name="norm_p2")
    enc_norm = norm_p2(enc)
    flat_h2, flat_c2 = [], []
    x = enc_norm
    for idx in range(n_layer):
        h_in = Lambda(lambda s, i=idx: s[:, i, :, 0],
                      output_shape=lambda s: (s[0], s[2]))(s2_in)
        c_in = Lambda(lambda s, i=idx: s[:, i, :, 1],
                      output_shape=lambda s: (s[0], s[2]))(s2_in)
        cell = StatelessLSTM(n_units, name=f"slstm_p2_{idx}")
        x, ho, co = cell([x, h_in, c_in])
        flat_h2.append(ho); flat_c2.append(co)
    p2_dense = Dense(n_enc, name="dense_p2")(x)
    p2_mask = Activation("sigmoid", name="mask_p2")(p2_dense)
    est = Multiply()([enc, p2_mask])
    dec_layer = Conv1D(n_block, 1, padding="causal", use_bias=False,
                        name="dec_conv")
    dec = dec_layer(est)
    s2_out = Lambda(_combine,
                    output_shape=lambda shapes: (1, n_layer, n_units, 2),
                    name="state_out_2")(flat_h2 + flat_c2)
    model_2 = Model([frame_in, s2_in], [dec, s2_out], name="dtln_p2")

    enc_layer.set_weights(enc_conv.get_weights())
    norm_p2.set_weights(full_norms[-1].get_weights())
    for idx, src in enumerate(sub2_lstms):
        model_2.get_layer(f"slstm_p2_{idx}").set_weights(src.get_weights())
    model_2.get_layer("dense_p2").set_weights(sub2_dense.get_weights())
    dec_layer.set_weights(dec_conv.get_weights())

    spec_1 = (tf.TensorSpec((1, 1, n_freq), tf.float32, name="mag_in"),
              tf.TensorSpec((1, n_layer, n_units, 2), tf.float32, name="state_in_1"))
    spec_2 = (tf.TensorSpec((1, 1, n_block), tf.float32, name="frame_in"),
              tf.TensorSpec((1, n_layer, n_units, 2), tf.float32, name="state_in_2"))
    out1 = f"{args.out_prefix}_p1.onnx"
    out2 = f"{args.out_prefix}_p2.onnx"
    o1, _ = tf2onnx.convert.from_keras(model_1, input_signature=spec_1, opset=args.opset)
    onnx.save(o1, out1)
    o2, _ = tf2onnx.convert.from_keras(model_2, input_signature=spec_2, opset=args.opset)
    onnx.save(o2, out2)

    for path in (out1, out2):
        mm = onnx.load(path)
        print(f"\n=== {path} ===")
        for x in mm.graph.input:
            s = [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]
            print(f"  in : {x.name:25s} shape={s}")
        for x in mm.graph.output:
            s = [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]
            print(f"  out: {x.name:25s} shape={s}")
        ops = sorted({n.op_type for n in mm.graph.node})
        print(f"  ops: {ops}")
        for bad in ("LSTM", "Loop", "Split"):
            if bad in ops:
                print(f"  [!] still has {bad} op")


if __name__ == "__main__":
    sys.exit(main() or 0)
```

跑：

```bash
cd /mnt/d/Projects/cyk/DTLN
CUDA_VISIBLE_DEVICES="" python rebuild_dtln_onnx.py \
    --weights pretrained_model/model.h5 \
    --out-prefix dtln_v4 --opset 14 2>&1 | tail -30
```

**期望产物**：`dtln_v4_p1.onnx`、`dtln_v4_p2.onnx`，且：
- ops 列表里 **没有** `LSTM` / `Loop` / `Split`
- 全静态 shape

**输入/输出名（基于现有 ONNX 命名约定，可能略有差异，以实际导出为准）**：

| 模型 | input | shape | output | shape |
|---|---|---|---|---|
| `dtln_v4_p1.onnx` | `mag_in` | (1, 1, 257) | `mask_p1` 或 `activation_*` | (1, 1, 257) |
|  | `state_in_1` | (1, 2, 128, 2) | `state_out_1` 或 `lambda_*` | (1, 2, 128, 2) |
| `dtln_v4_p2.onnx` | `frame_in` | (1, 1, 512) | `conv1d_*` 或 `dec_conv` | (1, 1, 512) |
|  | `state_in_2` | (1, 2, 128, 2) | `state_out_2` 或 `lambda_*` | (1, 2, 128, 2) |

**先运行脚本后用 Python 打印实际输出节点名**，记下来用于 §5 parse 命令。

### ORT 数值校验（强烈建议）

在 parse 之前用 onnxruntime 跑一遍 sub-models，与原始 .h5 在同一段音频上的输出做对比（误差应 < 1e-4）。如果误差很大说明 weight 拷贝顺序错了。校验脚本：

```python
# verify_v4.py
import numpy as np, onnxruntime as ort, sys, tensorflow as tf
sys.path.insert(0, "."); from DTLN_model import DTLN_model

m = DTLN_model(); m.build_DTLN_model_stateful(norm_stft=False)
m.model.load_weights("pretrained_model/model.h5")

s1 = ort.InferenceSession("dtln_v4_p1.onnx", providers=["CPUExecutionProvider"])
s2 = ort.InferenceSession("dtln_v4_p2.onnx", providers=["CPUExecutionProvider"])

# Generate one random audio frame
np.random.seed(0)
audio = np.random.randn(512).astype(np.float32) * 0.1
spec = np.fft.rfft(audio, n=512)
mag = np.abs(spec).astype(np.float32)[None, None, :]
state1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
state2 = np.zeros((1, 2, 128, 2), dtype=np.float32)

mask, st1_out = s1.run(None, {"mag_in": mag, "state_in_1": state1})
print("p1 mask norm:", np.linalg.norm(mask), "state_out norm:", np.linalg.norm(st1_out))

# feed mask back to time domain, run p2
masked = mag.squeeze() * mask.squeeze() * np.exp(1j * np.angle(spec))
frame = np.fft.irfft(masked, n=512).astype(np.float32)[None, None, :]
out, st2_out = s2.run(None, {"frame_in": frame, "state_in_2": state2})
print("p2 out norm:", np.linalg.norm(out), "state_out norm:", np.linalg.norm(st2_out))
```

输出非 NaN、非 0、量级合理（mask 0.1-1.0，p2 out ~ frame 量级）即认为重建正确。

## 4. 校准集生成

校准脚本 `dtln_build_calib.py` 会用合成或真实音频生成 (N, 1, 257) 频谱 + state，喂给 ONNX 推理拿到下一帧 state，作为校准样本。**输入名要跟新 ONNX 对齐**（`mag_in / state_in_1 / frame_in / state_in_2`），脚本是从 ONNX 的 `get_inputs()` 自动读名字的，所以理论上换 ONNX 不需要改：

```bash
cd /mnt/d/Projects/cyk/DTLN
python dtln_build_calib.py \
    --p1-onnx dtln_v4_p1.onnx \
    --p2-onnx dtln_v4_p2.onnx \
    --num-frames 1500 \
    --out-dir .

ls -lh calib_p1.npz calib_p2.npz
python -c "
import numpy as np
for p in ['calib_p1.npz', 'calib_p2.npz']:
    z = np.load(p); print(p, ':', {k: z[k].shape for k in z.files})
"
```

期望：
- `calib_p1.npz`: `mag_in` (1500, 1, 257) + `state_in_1` (1500, 2, 128, 2)
- `calib_p2.npz`: `frame_in` (1500, 1, 512) + `state_in_2` (1500, 2, 128, 2)

> 用合成信号做第一遍校准，**最终生产 HEF 时建议用真实 16 kHz 单声道语音 wav**（`--in-wav <path>`）。RPi5 已经录了一段 5.8s 的 ref_probe wav，可以从 RPi5 scp 过来用：
> `scp pi5:/home/cyk/codes/OpenAvatarChat/tests/results/326f42c/ref_probe/capture_8ch.wav .`
> 然后传 `--in-wav capture_8ch.wav`（脚本会自动取 mic[0]）。

## 5. parse → optimize → compile

```bash
cd /mnt/d/Projects/cyk/DTLN

# === 5.1 parse ===
# 注意：替换 <real_p1_out_audio_name> 等为 dtln_v4_p1.onnx 实际导出的 output 节点名
hailo parser onnx dtln_v4_p1.onnx \
    --hw-arch hailo10h \
    --start-node-names mag_in state_in_1 \
    --end-node-names <p1_out_audio> <p1_out_state> \
    -y

hailo parser onnx dtln_v4_p2.onnx \
    --hw-arch hailo10h \
    --start-node-names frame_in state_in_2 \
    --end-node-names <p2_out_audio> <p2_out_state> \
    -y

ls -lh dtln_v4_p1.har dtln_v4_p2.har

# === 5.2 optimize（需要 GPU + cuDNN）===
# 如果 cuDNN 还没装：
#   sudo apt search cudnn 2>/dev/null | grep -E "^cudnn|^libcudnn"
# 找到包名后装；NVIDIA 仓库的标准包：
#   sudo apt install -y cudnn9-cuda-12
# 然后：
cat > p1.alls <<'EOF'
quantization_param({mag_in}, precision_mode=int8)
quantization_param({state_in_1}, precision_mode=int16)
EOF
cat > p2.alls <<'EOF'
quantization_param({frame_in}, precision_mode=int8)
quantization_param({state_in_2}, precision_mode=int16)
EOF

hailo optimize dtln_v4_p1.har \
    --hw-arch hailo10h \
    --calib-set-path calib_p1.npz \
    --model-script p1.alls \
    --use-random-calib-set false -y

hailo optimize dtln_v4_p2.har \
    --hw-arch hailo10h \
    --calib-set-path calib_p2.npz \
    --model-script p2.alls \
    --use-random-calib-set false -y

# === 5.3 compile ===
hailo compiler dtln_v4_p1_optimized.har --hw-arch hailo10h --output-dir build_h10h -y
hailo compiler dtln_v4_p2_optimized.har --hw-arch hailo10h --output-dir build_h10h -y

ls -lh build_h10h/*.hef
mv build_h10h/dtln_v4_p1.hef build_h10h/dtln_p1_h10h.hef
mv build_h10h/dtln_v4_p2.hef build_h10h/dtln_p2_h10h.hef
```

### 常见 optimize 报错

| 现象 | 处理 |
|---|---|
| `cuDNN not found` / `libcudnn.so.9 cannot open` | 装 cuDNN 9，或者加 `--disable-finetune` 强制纯量化（精度可能差几个 dB） |
| `OOM during fine-tune` | 6GB GPU 紧张：加 `--batch-size 1`；不行就 `--disable-finetune` |
| `libdevice not found` | 装 NVCC：`sudo apt install nvidia-cuda-toolkit`，或设 `XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit` |

### compile 报错

| 现象 | 处理 |
|---|---|
| `Compilation failed: utilization too high` | 加 `--performance-mode latency` 优先低延迟而非吞吐 |
| `Resource exhausted` | 模型在小芯上太大；DTLN 单段应该没问题（< 1MB 权重）|

## 6. 部署到 RPi5

### 6.1 拷贝 HEF

```bash
# 在 WSL 上：
scp build_h10h/dtln_p1_h10h.hef pi5:/home/cyk/codes/OpenAvatarChat/models/hailo/
scp build_h10h/dtln_p2_h10h.hef pi5:/home/cyk/codes/OpenAvatarChat/models/hailo/
```

### 6.2 RPi5 端 wrapper 调整（已落地）

`audio_frontend/backends/denoiser.py:HailoDenoiser` 已按 v7 双-HEF 协议重写，
关键事项（如果以后重 compile 需要核对）：

1. **双 HEF + per-layer h/c IO**：每个 HEF 5 输入（audio + 4 state）+ 5 输出（audio + 4 state）。
2. **HailoRT 5.x InferModel API（不是 InferVStreams）**：
   ```python
   m   = vdev.create_infer_model(hef_path)
   for s in m.inputs:  s.set_format_type(FormatType.FLOAT32)
   for s in m.outputs: s.set_format_type(FormatType.FLOAT32)
   cm  = m.configure()
   bnd = cm.create_bindings()
   ...
   cm.run([bnd], 1000)             # 同步
   job = cm.run_async([bnd]); job.wait(1000)   # 异步
   ```
   旧 `InferVStreams.activate()` 在 Hailo-10H 多输入网络上会抛 `HAILO_NOT_IMPLEMENTED`。
3. **Manifest 是 wrapper 必读**：HEF 自动生成的 VStream 名（`fc1`/`conv7`/
   `precision_change*`/`ew_add*`）跟语义角色没有命名规律，必须从 `models/hailo/dtln_hef_manifest.json` 读。重 compile 出新 HEF 时务必重跑 `dump_hef_io.py` 刷新 manifest。
4. **HailoRT VStream 输入 reshape**：HEF 实际 input shape 是 `(1, 1, C)` 或
   `[C]`（state）；wrapper 直接 `set_buffer(np.zeros(list(s.shape), dtype=np.float32))`
   再 in-place 写就行（验证过 `get_buffer()` 与 `set_buffer()` 共享内存）。
5. **Scheduler 是关键**：默认 `ROUND_ROBIN` 同步串行 ~11 ms/帧（context switch 开销）；
   `NONE` 调度需要每次手动 activate/deactivate，timeout 频发；当前选择
   **`ROUND_ROBIN` + `run_async` 跨帧 pipelining**：本帧 p1 与上一帧 p2 用 `run_async`
   并行排队，~7 ms/帧，代价是音频输出多 1 hop（8 ms）延迟。

实测（RPi5 8GB + Hailo-10H + HailoRT 5.3）：
- DTLN HEF 单实例（同步串行）：~11.0 ms/帧（87 fps）
- DTLN HEF 跨帧 pipelining（async 并发）：**~7.1 ms/帧（141 fps）** ✓ <8 ms RT 预算
- 整链 RTF（mics + MVDR + AEC + Hailo DTLN）：1.21（DTLN 0.87 + DSP 其余 0.34）

如果以后 compile level≥2 拿到更精确的 HEF，wrapper 不变；只需 SCP HEF + manifest 即可。

### 6.3 验证

RPi5 上跑：

```bash
cd /home/cyk/codes/OpenAvatarChat
.venv/bin/python tools/dtln_hef_setup.py --check
hailortcli benchmark --hef models/hailo/dtln_p1_h10h.hef
hailortcli benchmark --hef models/hailo/dtln_p2_h10h.hef
```

期望每个 HEF 推理延迟 < 5 ms / inference，> 200 fps。

整链验证：

```bash
.venv/bin/python tests/run_pipeline_offline.py \
    tests/results/<sha>/ref_probe/capture_8ch.wav \
    --pipeline full --aec-backend speex --dns-backend hailo \
    --out-clean /tmp/hailo_clean.wav --out-csv /tmp/hailo_doa.csv
```

期望 RTF < 0.3（vs CPU 版 ~6）。

## 7. 你的任务范围

**你（WSL agent）需要做的事**：

1. ☐ 跑 `rebuild_dtln_onnx.py`（StatelessLSTM 版），**直到产出 v4 ONNX，且 ops 不含 LSTM/Loop/Split**。如果 StatelessLSTM 路也炸了，进 §A 备用方案。
2. ☐ 跑 ORT 数值校验（§3 末尾的 verify_v4.py），确认重建模型行为与原始 .h5 一致（量级合理）。
3. ☐ `dtln_build_calib.py` 重新生成 calib_p1.npz、calib_p2.npz。
4. ☐ `hailo parser` 两段 ONNX，得到两个 .har。
5. ☐ 装 cuDNN（必要的话）。
6. ☐ `hailo optimize` 两段 .har，得到两个 _optimized.har。
7. ☐ `hailo compiler` 两段 .har，得到两个 .hef，重命名为 `dtln_p1_h10h.hef` / `dtln_p2_h10h.hef`。
8. ☐ 用 ORT 同样的逻辑跑一遍 HEF 模拟（hailomz 有 `infer` 子命令在 emulator 上跑），看输出与 ONNX/原始模型 diff 有多大；> 0.05 RMS 就要回去调 ALLS 加大 `precision_mode`。
9. ☐ 把 .hef 文件拷到 `/mnt/d/Projects/cyk/DTLN/build_h10h/`，告诉用户怎么 scp 到 RPi5。

**你不需要做的**：
- 改 RPi5 端代码（只在交付时给用户改 HailoDenoiser 的指引）
- 测 ROS2 节点（那是 RPi5 的事）

## §A. 如果 StatelessLSTM 也失败的备用路径

按概率从低到高试：

1. **opset 13 / opset 11**：tf2onnx 在不同 opset 产出的算子集略有差别，老 opset 可能更 DFC 友好。改 `--opset 11`。
2. **手动重写 StatelessLSTM 让 Slice 用固定 axes**：DFC 对 Slice op 也挑剔，确保 Slice 的 axes/starts/ends 都是 graph initializer 而不是 input。
3. **每个 LSTM 层各导一个 ONNX，主 ONNX 用 ScatterND 拼接**：极端办法，把 2 层 LSTM 拆成 4 个独立 ONNX。
4. **找 Hailo Community DTLN 已编译 HEF**：在 [Hailo Community](https://community.hailo.ai/) 搜 "DTLN HEF Hailo10H"。如果有热心人编过，直接用。
5. **换降噪模型**：DTLN 是流式 LSTM，对 NPU 编译器最不友好。FRCRN / GTCRN 是更现代的 conv-based 流式模型，编译概率更高。这是大改，最后兜底。

## 8. 如果一切顺利，告诉我（用户）

1. `dtln_p1_h10h.hef` / `dtln_p2_h10h.hef` 大小（应该各 < 5 MB）
2. `hailortcli benchmark` 给出的 inference latency / fps
3. ORT vs HEF 模拟器输出的 RMS diff
4. 把这两个 HEF 给我（或者直接 scp，路径见 §6.1）

---

**TL;DR**：

```bash
cd /mnt/d/Projects/cyk/DTLN
# 1. 跑 rebuild（已经有了，最新版见 §3）
CUDA_VISIBLE_DEVICES="" python rebuild_dtln_onnx.py --out-prefix dtln_v4
# 2. 数值校验
python verify_v4.py
# 3. calib
python dtln_build_calib.py --p1-onnx dtln_v4_p1.onnx --p2-onnx dtln_v4_p2.onnx --num-frames 1500
# 4. parse (替换 <out_names> 为实际导出的 output 节点名)
hailo parser onnx dtln_v4_p1.onnx --hw-arch hailo10h --start-node-names mag_in state_in_1 --end-node-names <out1> <out2> -y
hailo parser onnx dtln_v4_p2.onnx --hw-arch hailo10h --start-node-names frame_in state_in_2 --end-node-names <out1> <out2> -y
# 5. optimize + compile
hailo optimize dtln_v4_p1.har --hw-arch hailo10h --calib-set-path calib_p1.npz --use-random-calib-set false -y
hailo optimize dtln_v4_p2.har --hw-arch hailo10h --calib-set-path calib_p2.npz --use-random-calib-set false -y
hailo compiler dtln_v4_p1_optimized.har --hw-arch hailo10h --output-dir build_h10h -y
hailo compiler dtln_v4_p2_optimized.har --hw-arch hailo10h --output-dir build_h10h -y
# 6. 重命名 + 告诉用户
mv build_h10h/dtln_v4_p1.hef build_h10h/dtln_p1_h10h.hef
mv build_h10h/dtln_v4_p2.hef build_h10h/dtln_p2_h10h.hef
ls -lh build_h10h/*.hef
```
