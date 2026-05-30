# DTLN → Hailo-10H HEF 编译详细步骤

**目标产物**：`models/hailo/dtln_h10h.hef`，~3-5 MB，被 `audio_frontend.backends.denoiser.HailoDenoiser` 加载，运行时单帧 32 ms，推理 < 5 ms，CPU 占用 ≤ 3%。

**前置事实（2026 / Hailo SW Suite 5.3 时代）**：
- 编译机：**x86_64 Linux** + **NVIDIA GPU**（Pascal/Turing/Ampere）+ CUDA 12.5.1。Dataflow Compiler 不支持 aarch64 / RPi5；optimize 阶段的量化 fine-tune 需要 GPU。
- 部署机（本机 RPi5）：仅做 HEF 部署 + 推理，编译完的 HEF `scp` 过来即可。
- 版本对齐：编译机的 Dataflow Compiler **5.3.0**（注意：Hailo 在 2026/04 把 DFC 从 3.x 体系直接跳到与 HailoRT 同步的 5.3，**不再有 `.run` 安装器，只发 `.whl`**），与本机 HailoRT 5.3.0 / driver / Python wheel 5.3.0 严格对齐。
- 操作系统：Ubuntu 22.04 或 24.04；Python 3.10/3.11/3.12 都支持（DFC 5.3 wheel 是 `py3-none-linux_x86_64`，跨 Python 版本通用）。

---

## 0. 选模型

DTLN 有几个版本，按优先级选：

| 模型 | 仓库 | 输入 | 优点 | 风险 |
|---|---|---|---|---|
| **DTLN (statefull, 推荐)** | [breizhn/DTLN](https://github.com/breizhn/DTLN) | 32 ms × 1 ch | LSTM × 2 + Conv1D，对 H10H 友好；社区已验证 | LSTM hidden state 必须做成显式输入/输出 |
| DTLN-aec | [breizhn/DTLN-aec](https://github.com/breizhn/DTLN-aec) | 32 ms 近端 + 32 ms 远端 | 一次性替代 SpeexDSP+DfNet | 训练数据更挑剔，国内网络拿模型权重慢 |
| DeepFilterNet 3 | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | 30 ms | 质量最优 | GRU + ERB band 较复杂，H10H 编译可行性不保证；建议 DTLN 跑通后再尝试 |

第一期就 **DTLN statefull**。

---

## 1. 编译机环境准备（一次性）

注册 Hailo Developer Zone（需要邮箱审核，工作日通常半天）后，登录到 Software Downloads 页面，可以选两条路径之一。**强烈推荐路径 A**（一次性把版本对齐问题全部消掉）。

### 路径 A：Hailo AI Software Suite Docker（推荐）

```bash
# 下载（约 12 GB，国内挂梯子或者 Hailo CN 镜像）
# - hailo_ai_sw_suite_2026-04_docker.tar.gz       (DFC 5.3 + HailoRT 5.3 + Model Zoo + TAPPAS)
# - hailo_ai_sw_suite_2026-04.zip                 (启动脚本与样例)

# 加载镜像
docker load -i hailo_ai_sw_suite_2026-04_docker.tar.gz

# 解压启动脚本
unzip hailo_ai_sw_suite_2026-04.zip
cd hailo_ai_sw_suite_2026-04

# 起一个开发容器；workdir 会自动挂载在容器的 /local/workspace
./hailo_ai_sw_suite_docker_run.sh

# 容器里校验
hailo --version            # 期望 5.3.0
hailortcli --version       # 5.3.0
hailomz --version          # Model Zoo 同步 5.3
nvidia-smi                 # 必须能看到 GPU；optimize 跑 fine-tune 必需
```

容器内已装好 DFC、HailoRT、Model Zoo、TAPPAS、CUDA 12.5.1 toolchain，省 30 分钟环境调试。

### 路径 B：纯 wheel 自己装

适合不能用 Docker、或者 GPU 驱动是 host 既有的场景。

> **关键澄清：编译机绝对不要装 `hailort-pcie-driver_*.deb`**。
> 这个 deb 是 Linux 内核模块，post-install 会编译并加载 PCIe 驱动给 Hailo 卡用；
> 没卡（或 WSL2、虚拟机里）会 `Failed. Exited with status 2` 报错。
> 编译机只需要 DFC wheel + 用户态 hailort（可选），HEF 输出不依赖任何驱动。
> PCIe driver 只在**部署机（你的 RPi5，插了 H10H 那台）**装；那台已经装过，不用再动。

```bash
# (1) 装 NVIDIA driver + CUDA 12.5.1（按 Hailo 文档要求；非本文重点）
nvidia-smi
nvcc --version  # 12.5.x

# (2) （可选）只装 HailoRT 用户态 lib，方便编译完用 hailortcli 看 HEF 元信息
#     注意：只装这一个 deb，不要装 hailort-pcie-driver*.deb
sudo apt install ./hailort_5.3.0_amd64.deb
# 装完会有 hailortcli + libhailort.so，没有内核模块依赖

# (3) 干净 venv + DFC wheel（编译核心，唯一必装项）
python3.11 -m venv ~/.venv-hailo-dfc
source ~/.venv-hailo-dfc/bin/activate
pip install --upgrade pip
pip install hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl
# hailort Python wheel 也只装用户态部分（不会拖驱动）
pip install hailort-5.3.0-cp311-cp311-linux_x86_64.whl   # cp310/cp311/cp312 都有

# (4) Model Zoo（提供 hailomz CLI）
git clone https://github.com/hailo-ai/hailo_model_zoo.git
cd hailo_model_zoo
pip install -e .

# (5) 校验
hailo --version            # 5.3.0
hailomz --help
hailortcli --version       # 也是 5.3.0；hailortcli scan 在没卡的机器上会报 No Hailo devices found，那是对的
```

> **如果 dev zone 找不到 .run 文件**：那是对的。Hailo 在 2026/04 之后只发 wheel，名字形如
> `hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl`（注意是 `py3-none`，不再有 cp310/cp311 之分）。
> 看到的 "5.3 wheel" 就是它，放心装。

> **WSL2 用户额外注意**：WSL2 不暴露宿主机 GPU 到容器内时，optimize 阶段会失败。
> 解决：(a) 用 Windows 端的 WSL2-cuda 启用 GPU passthrough，`nvidia-smi` 在 WSL 里要能看到卡；
> (b) 或者干脆不用 WSL，直接装 Ubuntu 22.04/24.04 双系统；
> (c) 或者租一台带 GPU 的云主机临时跑一次（Hailo 的 optimize fine-tune 通常 10-20 分钟就完）。

---

## 2. 拿 DTLN 源码 + 训练好的 ONNX

```bash
git clone https://github.com/breizhn/DTLN.git
cd DTLN

# 把 .h5 模型转成 statefull ONNX（关键！）
# 这一步会把 LSTM 的 hidden state 暴露为模型 input/output，
# 否则 HEF 没法跑流式（Hailo HEF 必须 stateless）。
python convert_weights_to_onnx.py \
    --weights pretrained_model/model.h5 \
    --target dtln_streaming.onnx \
    --statefull True

# 校验 ONNX 输入/输出
python - <<'EOF'
import onnx
m = onnx.load("dtln_streaming.onnx")
print("inputs:")
for i in m.graph.input:
    s = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
    print(f"  {i.name:20s} shape={s}")
print("outputs:")
for o in m.graph.output:
    s = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name:20s} shape={s}")
EOF
```

期望输出：

```
inputs:
  audio_in            shape=[1, 1, 512, 1]
  state_h_in          shape=[1, 2, 128, 2]   # 2 layers × 128 units × (h, c)
outputs:
  audio_out           shape=[1, 1, 512, 1]
  state_h_out         shape=[1, 2, 128, 2]
```

> 真实 shape 视 DTLN 版本而异；记下来后面校准/编译都要用。如果 LSTM 不是 2 layer × 128，更新 `audio_frontend/backends/denoiser.py:HailoDenoiser.__init__` 里的 `lstm_units` / `lstm_layers`。

---

## 3. 准备校准集（quantization 必需，~50 段）

```bash
mkdir -p calib_set
# 从 DNS-Challenge 或 VCTK 拿 50 段 16 kHz 单声道、3-10 秒的混合（带噪+干净）
# 简单做法：直接拿 dataset_v1 的 raw_8ch.wav 取 mic[0]：
python - <<'EOF'
import numpy as np, wave, os
src = "tests/data/dataset_v1/04_far_noise/raw_8ch.wav"
with wave.open(src, "rb") as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
samples = np.frombuffer(raw, "<i2").reshape(-1, 8)
mic = samples[:, 0].astype(np.float32) / 32768.0
# 切成 50 个 1 秒段
import os; os.makedirs("calib_set", exist_ok=True)
for i in range(50):
    start = i * 16000
    seg = mic[start:start + 16000]
    if len(seg) < 16000: break
    np.save(f"calib_set/seg_{i:03d}.npy", seg)
print("written", i, "segments")
EOF

# 把 50 段拼成 calib.npy (50, 16000)，再切 32ms 块
python - <<'EOF'
import numpy as np, glob
segs = [np.load(p) for p in sorted(glob.glob("calib_set/seg_*.npy"))]
arr = np.stack(segs).astype(np.float32)
# DTLN 看 32 ms = 512 samples 块；每段 16000 / 512 = 31 块
blocks = arr.reshape(-1, 512)[:1500]    # 1500 块够校准
blocks = blocks.reshape(-1, 1, 512, 1)  # 匹配 audio_in 的 NCHW
np.save("calib_audio.npy", blocks)
# state 输入用 0 即可（流式校准取代表性中间状态太复杂，用 0 也能拿到合理量化）
state = np.zeros((blocks.shape[0], 2, 128, 2), dtype=np.float32)
np.save("calib_state.npy", state)
print("calib_audio:", blocks.shape, "calib_state:", state.shape)
EOF
```

---

## 4. parse → optimize → compile (Dataflow Compiler 5.3)

DFC 5.3 把命令行简化了，全部通过 `hailo` 子命令；如果 Model Zoo 里有 DTLN，直接用 `hailomz` 一行。下面是 from-scratch 路径（不依赖 Model Zoo）。

```bash
# (1) parse: ONNX -> HAR
hailo parser onnx dtln_streaming.onnx \
    --hw-arch hailo10h \
    --start-node-names audio_in state_h_in \
    --end-node-names audio_out state_h_out \
    -y

# 产物：dtln_streaming.har（默认与输入同名）

# (2) optimize: 量化 + fine-tune（GPU 必需）
#     5.3 用 ALLS（Algo Layer Script）描述量化策略；最简单的 NMS-free 模型可以用 default
cat > dtln.alls <<'EOF'
quantization_param({audio_in}, precision_mode=int8)
quantization_param({state_h_in}, precision_mode=int16)
EOF

hailo optimize dtln_streaming.har \
    --hw-arch hailo10h \
    --calib-set-path calib_audio.npy \
    --model-script dtln.alls \
    --use-random-calib-set false \
    -y
# 产物：dtln_streaming_optimized.har

# (3) compile: optimized HAR -> HEF
hailo compiler dtln_streaming_optimized.har \
    --hw-arch hailo10h \
    --output-dir build_h10h \
    -y

# 产物：
ls -lh build_h10h/dtln_streaming.hef
```

**多输入校准注意事项**：DFC 5.3 的 `--calib-set-path` 默认接受单个 npy，多输入模型需要把所有输入打包成 `calib_set.npz`：

```python
import numpy as np
audio  = np.load("calib_audio.npy")     # (N, 1, 512, 1)
state  = np.load("calib_state.npy")     # (N, 2, 128, 2)
np.savez("dtln_calib.npz", audio_in=audio, state_h_in=state)
```

然后用 `--calib-set-path dtln_calib.npz` 即可（key 必须与 ONNX input name 完全一致）。

---

## 5. 部署到 RPi5

```bash
# 在编译机：
scp build_h10h/dtln_streaming.hef pi5:/home/cyk/codes/OpenAvatarChat/models/hailo/dtln_h10h.hef

# 在 RPi5：
cd /home/cyk/codes/OpenAvatarChat

# (1) 看 HEF 是否被设备认识 + 单帧 latency
.venv/bin/python tools/dtln_hef_setup.py --check
hailortcli benchmark --hef models/hailo/dtln_h10h.hef
# 期望：< 5 ms / inference, > 200 fps

# (2) 用我们的 wrapper 加载并跑一段
.venv/bin/python - <<'EOF'
import numpy as np
from audio_frontend.backends.denoiser import HailoDenoiser
d = HailoDenoiser(block_size=512)
print("loaded:", d.name, "hef:", d.hef_path)
y = d.process(np.zeros(512, dtype=np.float32))
print("output shape:", y.shape, "min/max:", y.min(), y.max())
EOF

# (3) 整链 + Hailo DNS
.venv/bin/python tests/run_pipeline_offline.py \
    tests/results/<sha>/ref_probe/capture_8ch.wav \
    --pipeline full --aec-backend speex --dns-backend hailo \
    --out-clean /tmp/hailo_clean.wav --out-csv /tmp/hailo_doa.csv

# (4) 测 RTF（Hailo 后端目标 < 0.3）
time .venv/bin/python tests/run_pipeline_offline.py ...
```

---

## 6. 状态张量名（HailoDenoiser 自动探测）

`audio_frontend/backends/denoiser.py` 的 `HailoDenoiser._init_hef()` 已经写了**按名字探测**逻辑：

```python
self._in_audio_name  = next(i.name for i in inputs  if "audio" in i.name.lower())
self._in_state_name  = next(i.name for i in inputs  if "state" in i.name.lower() or "lstm" in i.name.lower())
self._out_audio_name = next(o.name for o in outputs if "audio" in o.name.lower())
self._out_state_name = next(o.name for o in outputs if "state" in o.name.lower() or "lstm" in o.name.lower())
```

如果你的 ONNX 命名不带 `audio` / `state` / `lstm` 关键字（例如 `input.1` / `input.2`），有两种处理：

1. **改 ONNX 命名**：在 step 2 转 ONNX 后用 `onnx.helper` 重命名（推荐）。
2. **改 wrapper**：在 `HailoDenoiser` 构造时显式传 `in_audio_name=...` 等，覆盖自动探测。

---

## 7. 常见踩坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `hailo parser` 报 "Unsupported op: LSTM" | DTLN 用了原生 LSTM op，部分 DFC 版本不直接支持，需要 unroll | 转 ONNX 时加 `--unroll-lstm` 或先用 `onnxsim` 把动态 batch 变成静态 |
| optimize 后 HAR 体积 ~50 MB | 校准集太小或量化没收敛 | 校准集扩到 200 段，optimize 加 `--full-precision-tail True` |
| HEF 在 RPi5 加载报 "architecture mismatch" | 编译时 `--hw-arch` 写成了 `hailo8` / `hailo15h` | **必须** `hailo10h`，否则无救 |
| 推理输出全 0 | LSTM 状态张量没正确 wire；模型把 `state_h_out` 当 dummy 输出丢了 | 检查 `parse-hef` 输出的 stream 名字，确认 `out_state_name` 真的能拿到 hidden state |
| RTF > 1 | 模型可能没量化为 INT8，跑 FP16 太慢 | optimize 阶段确认量化模式：`--quantization-method symmetric_per_layer` |
| 启动后第一帧延迟 100ms+ | NetworkGroup activate 是惰性的 | 在启动钩里手动调 `network_group.activate()` 一次预热 |

---

## 8. 替代路径：直接用 Model Zoo

Hailo 官方 Model Zoo 里有 DTLN 预编译的 H8 HEF：

```
git clone https://github.com/hailo-ai/hailo_model_zoo
cd hailo_model_zoo
hailomz info dtln       # 查看支持架构
hailomz compile dtln --hw-arch hailo10h     # 自动跑 parse/optimize/compile
```

如果 Model Zoo 已经提供 `dtln_h10h.hef`，可以**完全跳过本文档 step 2-4**。第一次试时建议先看一下 Model Zoo 是否更新了 H10H 版本。

---

## 9. 验收清单

把 `models/hailo/dtln_h10h.hef` 拷过来后，跑：

```bash
make test-stage1 BACKEND=hailo
```

通过条件（同 plan §8.4 stage 1 验收表）：
- [ ] HEF 加载成功（`tools/dtln_hef_setup.py --check` 全绿）
- [ ] DNSMOS-OVRL（Hailo） 不低于 CPU 版 -0.1
- [ ] Pipeline RTF < 0.3
- [ ] CPU DNS 占用 ≤ 3%（top）
- [ ] 端到端音频前端延迟比 CPU 版 ≤ -5 ms
