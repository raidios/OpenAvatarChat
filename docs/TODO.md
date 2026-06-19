# 未竟事项 / Plan B / 待回顾方向（smart-car）

> 这是从代码、文档、Cursor 规则里挖出来的**有价值但未完成 / 卡住 / 切了 Plan B / 临时废弃**的事项，供工作告一段落时回顾、决定下一步方向。
> 状态标记：🔴 卡住/阻塞　🟡 进行中或部分可用　🟢 已有可用兜底（优化项）　⚪ 想法/备选。
> 与运行现状对照见 [PROJECT_STATE.md](PROJECT_STATE.md)。最后更新：2026-06-19。

## 0. 🟡 当前已知问题（用户实测确认，优先诊断 —— 下一步重点）

四大能力都跑通了，但带三个待诊断问题。**下个 session 方向 = 先测全现状、建基线**，诊断方法见 [EVAL_PLAN.md](EVAL_PLAN.md)。

- **0.1 VAD "有时"唤醒困难（接麦阵后）**🟡 现象仍在，但"DSP 电平过低"已**证伪**。
  - 受控测量（有声起止框定窗口，2026-05-31 00:18，**确认有语音**）：真实语音 peak **-20~-28 dBFS**（rms 222~377），远高于环境噪声 ~-45；Silero `max_prob=0.95~0.96`（阈值 0.25 绰绰有余）；ASR 出文本、LLM 回复、TTS 出声——**整条对话闭环本次跑通**。→ 之前那组 -45dBFS 是**环境噪声不是语音**，"DSP 输出电平过低"不成立（教训：未确认有语音就拿历史 `[mic]` 数值下结论是错的）。
  - 新线索（更可能的间歇性根因）：**playback↔mic 门控时序**。日志见 `[mic] mute=True is_playing=True mute_until_in=-1393.29s`（疑 stale/异常时间戳）+ 服务端 `Wake reply did not receive playback_complete in time; enabling VAD (server failsafe)`。怀疑唤醒应答/TTS 播放与 VAD 重新使能的握手在某些情况下错位 → 偶发吞掉语音。**需抓一次失败实例**对比成功/失败时的 mute/playback 状态。
  - TODO：在下一次车上复现时同时抓客户端与服务端状态时间线：客户端 `is_playing` / `mute_until` / `mic_should_mute` / `playback_done_event` / `playback_complete` 发送时间；服务端 `wake_session_active` / `enable_vad` / `playback_complete` 接收时间 / wake failsafe 触发时间。目标是确认失败发生在客户端未发、服务端未收、还是 mute/VAD 状态时序错位。
  - 相关：`client/ws_audio_client.py`（mute/playback）、`wakeword/sherpa_kws/...`（wake reply + playback_complete 握手）、`vad/silerovad`、`client_handler_ws.py`。
- **0.2 ArUco/marker board 跟随卡顿/迟滞**✅ 已关闭（2026-06-19 用户确认）。
  - 2026-06-19 实机结论：单 tag/多 tag 直接选最近目标会在 3x3 marker 板上频繁换目标；暗环境下自动曝光/动态降帧导致运动模糊，检测掉点明显；固定曝光测试后检测稳定性显著改善，但固定曝光不适合对话视觉长期使用。
  - 已实现：默认把 ID 0-8 的 3x3 marker board 聚合成单一目标（45mm tag、约 7mm gap），要求至少 2 个 tag 才刷新 board pose；短暂丢失时先限速预测，再进入非驱动 hold；目标 pose 非有限/距离越界会立即停车；控制器线速度/角速度限幅，并降低角速度 D 项、加角速度变化率限制，避免 `vw` 数值尖峰。
  - 已实现：默认服务模板改走 `--camera-mode rgbd-sdk --camera orbbec`，从 Orbbec SDK 同一 pipeline 读取 color+depth，并为后续检测框取 aligned depth 预留 `get_rgbd_frame()`。UVC 启动参数 `--camera-fps`、`--camera-auto-exposure auto|manual`、`--camera-disable-dynamic-framerate`、`--camera-exposure-time`、`--camera-gain` 仅作为 `--camera-mode color` 的兜底/调试路径。
  - 关闭依据：RGB-D SDK 模式 + 3x3 marker board 聚合 + 2 tag 刷新 + 短时预测 + 控制器限速/角速度变化率限制后，短电机测试未再出现危险前冲；第二轮 20s 窗口未见 `Tag lost -> IDLE` 抖动，用户认可 marker 追踪问题可关闭。
- **0.3 情绪 tag 与回答相关性低**🟡：表情/动作触发正常，但情绪标签和回答内容语义相关性不高。疑 prompt 与 VLM/LLM 输出策略契合度。当前链路里 `config/system_prompt.txt` 要求模型在回答中直接插入 `[happy]`/`[shy]`/`[apologize]`/`[scared]`；`data/teach_bindings.json` 与 `client/action_dispatcher.py` 也只认这 4 个。该问题可先脱离小车做本地/WSL 半在线闭环：固定 system prompt + 多轮文本/可选图片样本 → 调 DashScope compatible LLM/VLM → 解析 action tags → 与人工期望标签对比，A/B prompt 后再决定是否部署到车上。
- **0.4 🔴 安全：DashScope api_key 在 journald 明文留存**：服务端 `chat_engine/core/handler_manager.py:register_handler` 在 INFO 日志里**打印了完整 handler config，包含 `api_key='sk-…'`**（QwenASR/LLM 从 `.env` 注入的 key 被原样打到 journal）。key 本身**不在 git**（`.env` 已忽略），但 journald 明文可读、且会随日志外泄。**处理**：① 轮换该 DashScope key（已在日志/调试过程中暴露）；② `register_handler` 日志脱敏已在本地实现并加 unittest（`tests/unittest/test_handler_manager_redaction.py`），车端待部署并重启服务后生效。

## 1. 🟡 DNS 降噪（DTLN → Hailo-10H）—— HEF 已成、因 buzz 默认关

**现状**：远场链路里 DNS 整级**默认旁路**（`client/main.py` 默认 `--farfield-dns-passthrough`）。原因：DTLN v7 HEF 有间歇性量化 buzz，比它带来的 +10dB 回声残余抑制更扰人；AEC+MVDR 已给 -7dB 残余，日常室内够用。

**踩坑史 & 进度**（详见 [DTLN_HEF_HANDOFF.md](DTLN_HEF_HANDOFF.md) + [DTLN_HEF_COMPILATION.md](DTLN_HEF_COMPILATION.md)）：
- 在 WSL（x86 + RTX2060 + DFC 5.3）上编 DTLN HEF，反复撞 DFC 的 LSTM/Loop/Split parser bug；老 keras2onnx 导出的 ONNX 全坏。
- 最后的 in-progress 方案：**StatelessLSTM**（把一步 LSTM 拆成纯 MatMul+Slice+Sigmoid+Tanh，绕开原生 LSTM/Loop/Split op），脚本 `tools/rebuild_dtln_onnx.py` / `tools/dtln_build_calib.py`。
- **HEF 最终编出来了并在车上跑过**（用户确认，`models/hailo/`）；RPi5 端 wrapper `audio_frontend/backends/denoiser.py:HailoDenoiser` 按双-HEF v7 协议写好，实测跨帧 pipelining ~7.1ms/帧（< 8ms 预算）。**唯一拦路的是量化 buzz**，不是编译。

**下一步选项**：
1. 修 DTLN 量化 buzz：扩校准集 / 提 `precision_mode` / 真实语音校准（`make test-stage1 BACKEND=hailo` 验收，目标 DNSMOS 不低于 CPU -0.1、RTF<0.3）。
2. ⚪ 换降噪模型：FRCRN / GTCRN（conv-based 流式，对 NPU 编译更友好）或 DeepFilterNet3。DTLN 的流式 LSTM 对编译器最不友好，是大改兜底。
3. ⚪ 找 Hailo Community 已编好的 DTLN H10H HEF 直接用。

**相关文件**：`audio_frontend/dsp/dns_cpu.py`、`audio_frontend/backends/denoiser.py`、`tools/dtln_*.py`、`tools/rebuild_dtln_onnx.py`、`models/hailo/`（树莓派上）。

## 2. 🔴 ROS2 集成（建好但完全没部署）

**现状**：`ros2_ws/src/` 有 `audio_frontend_node` / `camera_node` / `perception_node` 三个 ament_python 节点，`src/handlers/client/ros2_client/` 有 rclpy 订阅 handler；但**没有 systemd 单元启动 ros2_ws，config 也没挂 ros2_client**——整条 ROS2 链路当前是死的。

**阻塞点**（[decisions/ROS2_DISTRO.md](decisions/ROS2_DISTRO.md)）：决定用 **ROS2 Jazzy**（Ubuntu 24.04 上的 LTS），但 Jazzy 默认 **Python 3.12**，而项目 `.venv` 是 **3.11**（hailort wheel 是 cp311），rclpy 与 .venv 不直接兼容。
- Plan A（首选）：把 `.venv` 重建为 3.12，需先确认 hailort 5.3.0 有 cp312 wheel（或本地编译）。
- 🟡 Plan B：ROS2 节点跑 Jazzy/3.12，handler 通过 ZeroMQ/socket bridge 通信，不用 rclpy。
- 尚未实际安装 Jazzy 验证，架构按"节点+handler 同跑 Jazzy/3.12"假设先行。

**下一步**：先在树莓派装 Jazzy（apt 源在文档里）+ 验证 hailort cp312，决定 A 还是 B；再决定 ROS2 是否真要进部署链路（当前单进程客户端已经能跑，ROS2 主要为将来接底盘/雷达/统一消息）。

## 3. 🟡 感知 stage2（视觉）——建好、smoke 过、未接入

`audio_frontend/vision/`（YOLO pose、re-id、mouth_estimator、gallery）+ `make test-stage2-smoke` 能跑合成图，但**没接进部署中的客户端**，依赖 ROS2 `perception_node` + Orbbec 深度。
- 相关：`tools/perception_quicklook.py`（live RGB-D pose+mouth-3D）、`tools/calibrate_mic_camera.py`（aruco mic↔camera 外参）、`tools/fetch_yolo_pose.py`（取 yolov8n-pose 导 ONNX）。
- **下一步**：明确感知的产品用途（人脸朝向/距离驱动表情或跟随？），再决定走 ROS2 还是直接进客户端线程。

## 4. 🟡 Orbbec Gemini Pro 深度相机（链路通，受限）

- **USB 卡在 2.0（480M）**：想上 USB3（5000M）需同时满足 USB3 micro-B 线 + 直插 Pi5 蓝口 + 中间不过 USB2 hub（当前过了两层）。不影响功能，只影响多流不丢帧。
- **D2C 硬件对齐仅在 color=640×480 可用**（出厂标定限制）；要高分辨率 RGB-D 得走 cv2 UVC 路径 + chessboard 标定（`align_depth_to_color`），且 SDK 路径与 cv2 路径**互斥**（抢同一 USB 设备，切换要 `uvc_rebind.sh`）。
- 约束：**只能用 OrbbecSDK v1**（Gemini 不在 v2 支持列表），自写 `libobgemini.so` ctypes shim，**勿** `import pyorbbecsdk`。详见 `.cursor/rules/orbbec-gemini-device.mdc`。
- **下一步**：深度目前没进部署链路（客户端用 color/V4L2 做 ArUco）。若要做 3D 跟随/距离，先标定 + 定 SDK-vs-cv2 路径。

## 5. ⚪ 固件预留 / 已移除功能

- 帧 `0x51`（IMU 校准）、`0x5f`（AKM 舵机零偏）**已定义但未实现**（`.cursor/rules/.../communication-protocol.mdc`）。
- **FN1 陀螺仪闭环巡线已从仲裁路径移除**（原作者"开机即自动巡线"与"Pi 显式控制"冲突）。若要恢复，须加独立的"组合键+蜂鸣提示"启用守护，别再让它做默认兜底。
- **UART5（PC12/PD2）空闲可用**，作为 Pi 第二通道或扩展（`.cursor/rules/.../hardware-pinmap.mdc`）。
- 舵机（TIM5/TIM8，6 路）、SBUS、WS2812 RGB、超声波等驱动已实现，部分未初始化，可按需启用。

## 6. ⚪ 标定 / 调参类待办

- **DOA 标定**：`make doa-record/-cardinal/-octagonal` → `make doa-calibrate`（暴力拟合 perm+yaw）→ `make doa-test`。`mic_yaw_offset_deg=+30` 是实测值；若重装麦阵需重跑。
- **mic↔channel 映射**：`make mic-tap-calibrate`（敲击法识别通道→麦位）。8ch 是从固件"hack"出来的，通道序由 tap test 为准。
- **相机内参**：`make dump-camera-intrinsics`（读 Gemini 出厂 K + D2C 到 `config/camera_calib_factory.npz`）。

## 7. 🟢 文档/规则待校正（已发现的过时项）

- `.cursor/rules/m260c-mic-array.mdc` 称 `make_aec(auto)` 默认 NLMS（"直到 Speex reset 问题查清"）——**已过时**：`audio_frontend/dsp/aec.py` 现在 auto **默认 Speex MDF**（per-mic-before-MVDR 实测 +19.2dB ERLE），Speex reset 问题应已随 per-mic 顺序解决。回顾时更新该规则。

---

### 回顾用速记
- 能跑的最小闭环已经在车上自启并验证：**远场语音(AEC+MVDR) → 客户端KWS唤醒+转向 → 云端ASR/LLM/TTS → 表情+动作**。
- 三块"建好没启用"的能力按价值排序：**DNS 降噪(1) > ROS2 化(2) > 视觉感知(3)**。
- 没有安全/密钥遗留问题。
