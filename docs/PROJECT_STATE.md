# 项目状态与实际运行覆盖（smart-car 分支长期记忆）

> 本文件是 `smart-car` 分支的"长期记忆"，记录**车上实际启动了什么、实际跑到哪些代码、哪些代码是建好但没被部署链路用到**。新 session 开工前应先读本文 + [TODO.md](TODO.md) + [../CLAUDE.md](../CLAUDE.md)。
> 维护约定：每当部署命令、config、handler 组合发生变化时更新本文。最后更新：2026-05-30。

## 1. 实际启动点（systemd，车上 power-on 自启）

两套 systemd **user** 服务（linger 开启，开机自启，已验证 +22s/+46s 拉起）：

```
# openavatarchat.service  (WantedBy=default.target)
.venv/bin/python src/demo.py --config config/chat_rpi_voice.yaml

# openavatarchat-client.service  (WantedBy=gnome-session.target, 需图形自动登录 DISPLAY=:0)
.venv/bin/python client/main.py --expression \
    --server ws://127.0.0.1:8282/ws/chat \
    --serial-port /dev/ttyAMA0 --tag-size 0.045 --calib-file config/camera_calib.json \
    --farfield --farfield-mic-yaw-offset-deg 30 \
    --wake-kws-keywords config/keywords.txt --wake-kws-threshold 0.25 \
    --wake-orient-min-deg 8 --wake-orient-deadzone-deg 5
```

## 2. 服务端实际覆盖：**handler 由 config 决定**（关键）

OpenAvatarChat 服务端**只加载 `config/chat_rpi_voice.yaml` 的 `handler_configs` 里列出的 handler**，`src/handlers/` 下其它 handler 一律不加载。当前 config 加载链：

| Handler | 模块 | 实际状态 |
|---|---|---|
| `WsClient` | `client/ws_client/client_handler_ws` | ✅ 传输层，与机器人客户端走 WebSocket |
| `WakeWord` | `wakeword/sherpa_kws/...` | ⚠️ `external_wake_only: true` → **不加载 KWS 模型、不在服务端解码**；只保留会话态、唤醒应答 TTS、告别、超时（15s）。KWS 实际在客户端做 |
| `SileroVad` | `vad/silerovad/...` | ✅ |
| `QwenASR` | `asr/qwen_asr/...` | ✅ 云端 `fun-asr-realtime`（DashScope） |
| `LLMOpenAICompatible` | `llm/openai_compatible/...` | ✅ 云端 `qwen3.5-flash`，带视频输入，system_prompt=`config/system_prompt.txt` |
| `QwenTTS` | `tts/qwen_tts/...` | ✅ 云端 `qwen3-tts-flash`，voice=Cherry |

- **完全没被加载**（代码在仓库但当前部署不走）：`rtc_client`、`h5_rendering_client`(LAM)、`ros2_client`、`minicpm`、`qwen_omni`、`dify`、`sensevoice`、`bailian_tts`、`cosyvoice`。
- **依赖**：ASR/LLM/TTS 全是 **Aliyun DashScope 云调用** → 需联网 + `.env` 的 `DASHSCOPE_API_KEY`（无 GPU 需求）。本机只跑 VAD/KWS 这类轻量本地推理。

## 3. 客户端实际覆盖（`client/main.py::run()`，按上面的 flag）

未传 `--no-*`，所以以下模块**全部启用**：

| 模块 | 文件 | 状态 |
|---|---|---|
| 相机 | `camera.py`（`--camera auto`→首个 V4L2 或 Orbbec SDK color） | ✅ |
| 串口 X-Protocol | `serial_comm.py` ↔ 固件 UART4 | ✅ |
| AprilTag/ArUco 跟踪 | `apriltag_tracker.py` + `tracking_controller.py` | ✅ |
| 远场音频 | `farfield_audio_source.py` + `audio_frontend/` | ✅（DNS 子级默认旁路，见下） |
| 唤醒转向 | `wake_word_spotter.py` + `wake_orient_controller.py`（客户端 sherpa KWS + IMU 闭环转向） | ✅ |
| 示教/动作 | `teach/*` + `action_dispatcher.py` + `data/` | ✅（即便 `--no-teach-ui` 也保留 player+dispatcher） |
| 表情脸 | `--expression` → `expression_app.py` + `expression_player/`（Qt 全屏） | ✅ |
| 聊天 | `chat_client.py` + `ws_audio_client.py` | ✅ |
| Teach 调试 UI | `teach/server.py` + `teach_ui/`（FastAPI :8080） | ✅ |

**远场 DSP 实际链路**（`audio_frontend/dsp/`）：每路 mic 先 `aec.py`（**默认 Speex MDF**，auto 时；speexdsp 缺失才退 NLMS）→ `srp_phat.py` DOA → `mvdr.py` 波束 → **DNS 默认旁路** → 16kHz mono PCM。坐标系：DSP 全程 mic frame，与 body frame 差 `mic_yaw_offset_deg=+30°`。

## 4. 建好但**当前部署链路用不到**（dormant）

- `audio_frontend/dsp/dns_cpu.py`（DeepFilterNet CPU）、`audio_frontend/backends/denoiser.py`（Hailo DTLN）：**DNS 整级默认关闭**（DTLN v7 量化 buzz），需 `--farfield-dns-on` 才启用。详见 [TODO.md](TODO.md)。
- `audio_frontend/vision/*`（yolo_pose / reid / mouth_estimator / gallery）：stage2 感知，只在 `make test-stage2-smoke` 和 `tools/perception_quicklook.py` 跑过，**未接入部署中的客户端**。
- `ros2_ws/`（audio_frontend / camera / perception 三个节点）+ `src/handlers/client/ros2_client/`：**完全未部署**（无 systemd 单元，config 也没挂 ros2_client）。
- `tools/*`（标定/探针/诊断）、`tests/*`：手动运行，非自启链路。
- 固件 `Project/` 编译产物：已 gitignore，源码用 Keil AC5 编译（GB2312 编码）。

## 5. 仓库/远端状态

- `origin` = `raidios/OpenAvatarChat`（fork，可推）；`upstream` = `HumanAIGC-Engineering/OpenAvatarChat`（只读参考）。`main` 跟 `upstream/main`；`smart-car` 跟 `origin/smart-car`。fork 点 `93c7c4b`(#209)。
- **密钥审计（2026-05-30）**：无泄漏。`.env` 未被跟踪且在 `.gitignore`；车端 tracked 代码无硬编码密钥（key 走 `os.getenv("DASHSCOPE_API_KEY")`）。
- 厂商 SDK（`thirdparty/`）、模型权重（`*.hef/*.onnx/*.pt`）、`tests/data|metrics|results` **只在树莓派上**，不在 git。
