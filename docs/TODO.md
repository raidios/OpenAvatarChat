# 未竟事项 / Plan B / 待回顾方向（smart-car）

> 这是从代码、文档、Cursor 规则里挖出来的**有价值但未完成 / 卡住 / 切了 Plan B / 临时废弃**的事项，供工作告一段落时回顾、决定下一步方向。
> 状态标记：🔴 卡住/阻塞　🟡 进行中或部分可用　🟢 已有可用兜底（优化项）　⚪ 想法/备选。
> 与运行现状对照见 [PROJECT_STATE.md](PROJECT_STATE.md)。最后更新：2026-05-30。

## 0. 🟡 当前已知问题（用户实测确认，优先诊断 —— 下一步重点）

四大能力都跑通了，但带三个待诊断问题。**下个 session 方向 = 先测全现状、建基线**，诊断方法见 [EVAL_PLAN.md](EVAL_PLAN.md)。

- **0.1 VAD 唤醒困难（接麦阵后）**🟢根因已定位（车上实测 2026-05-31）：**远场 DSP 输出电平过低**。客户端 `[mic]` 日志：说话段 peak 仅 **-43~-47 dBFS**、静音 -51~-52 dBFS，绝对电平极小（rms 21~31 / 32768），区分度只有 ~5-8 dB；Silero `speaking_threshold=0.25` 难稳定触发。无麦阵时走 PyAudio 单麦电平正常 → 印证是 DSP 链路电平问题，非麦阵硬件。**修复方向**：DSP 末级加 makeup gain / AGC（或对 MVDR 输出归一化）把语音抬到合理电平；降阈值只是权宜。相关：`audio_frontend/dsp/`（MVDR/末级增益）、`farfield_audio_source.py`、`vad/silerovad` 参数。
- **0.2 ArUco 跟踪卡顿/迟滞**🟢根因已定位（车上实测）：**marker 检测严重间歇**——静止 tag 也是"检测~1s → Tag lost → IDLE → ~7s 后重获"反复横跳，控制器每次丢失即 `TRACKING→IDLE` 发零速停车 → 一顿一顿。**机制**：客户端单进程，远场 DSP 线程 **99.9% CPU（GIL-bound）**，4 核虽有空闲但 **Python GIL** 抢占，相机/视觉线程被饿 → 丢帧 → 检测断续。相机=`/dev/video0` USB2.0 640×480（非 Orbbec），fps 未知可能不稳。原"300ms cmd_vel 超时"假设是次要放大器，主因是检测丢帧。**修复方向**：① 跟踪状态加 hold/debounce（短暂丢失保持上一速度，类似已有的 `--tag-expression-holdoff-ms`）；② 降 DSP 的 GIL 占用（DSP 移子进程 / C 扩展 / 降处理帧率），给视觉线程让出 GIL；③ 核实相机 fps，必要时换相机或调检测参数。相关：`apriltag_tracker.py`、`tracking_controller.py`、`audio_frontend/dsp/pipeline.py`、`camera.py`。
- **0.3 情绪 tag 与回答相关性低**🟡：表情/动作触发正常，但情绪标签和回答内容语义相关性不高。疑 prompt 与模型契合度。看 `config/system_prompt.txt` 对情绪 tag 的约束、`action_tags` 白名单一致性，A/B 调 prompt。
- **0.4 🔴 安全：DashScope api_key 在 journald 明文留存**：服务端 `chat_engine/core/handler_manager.py:register_handler` 在 INFO 日志里**打印了完整 handler config，包含 `api_key='sk-…'`**（QwenASR/LLM 从 `.env` 注入的 key 被原样打到 journal）。key 本身**不在 git**（`.env` 已忽略），但 journald 明文可读、且会随日志外泄。**处理**：① 轮换该 DashScope key（已在日志/调试过程中暴露）；② 在 `register_handler` 日志里对 `api_key`/secret 字段脱敏（如 `sk-…****`）。

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
