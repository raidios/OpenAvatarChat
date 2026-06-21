# 未竟事项 / Plan B / 待回顾方向（smart-car）

> 这是从代码、文档、Cursor 规则里挖出来的**有价值但未完成 / 卡住 / 切了 Plan B / 临时废弃**的事项，供工作告一段落时回顾、决定下一步方向。
> 状态标记：🔴 卡住/阻塞　🟡 进行中或部分可用　🟢 已有可用兜底（优化项）　⚪ 想法/备选。
> 与运行现状对照见 [PROJECT_STATE.md](PROJECT_STATE.md)。最后更新：2026-06-19。

## 0. 🟡 当前已知问题（用户实测确认，优先诊断 —— 下一步重点）

四大能力都跑通了，但带三个待诊断问题。**下个 session 方向 = 先测全现状、建基线**，诊断方法见 [EVAL_PLAN.md](EVAL_PLAN.md)。

- **0.1 VAD "有时"唤醒困难（接麦阵后）**🟡 现象仍在，但"DSP 电平过低"已**证伪**。
  - 受控测量（有声起止框定窗口，2026-05-31 00:18，**确认有语音**）：真实语音 peak **-20~-28 dBFS**（rms 222~377），远高于环境噪声 ~-45；Silero `max_prob=0.95~0.96`（阈值 0.25 绰绰有余）；ASR 出文本、LLM 回复、TTS 出声——**整条对话闭环本次跑通**。→ 之前那组 -45dBFS 是**环境噪声不是语音**，"DSP 输出电平过低"不成立（教训：未确认有语音就拿历史 `[mic]` 数值下结论是错的）。
  - 新线索（更可能的间歇性根因）：**playback↔mic 门控时序**。日志见 `[mic] mute=True is_playing=True mute_until_in=-1393.29s`（疑 stale/异常时间戳）+ 服务端 `Wake reply did not receive playback_complete in time; enabling VAD (server failsafe)`。怀疑唤醒应答/TTS 播放与 VAD 重新使能的握手在某些情况下错位 → 偶发吞掉语音。**需抓一次失败实例**对比成功/失败时的 mute/playback 状态。
  - 2026-06-19 多轮实机观察新增线索：未见 VAD 长时间卡死；延迟更像来自 **TTS/playback/action 门控**。一轮短回复中 `QwenTTS speech end` 比第二句 synth 晚约 15.7s；另一次 ASR 偶发慢约 4.1s；含 `[happy]` 的回答触发了 6.8s motion clip，当前 `playback_complete` 会等动作队列清空后才发送，容易造成"答完但暂时不听下一句"的体感。
  - 已加并部署 telemetry：`[telemetry][asr]` 记录 ASR 音频长度、start/send/stop/wait、first-event/final/complete 耗时；`[telemetry][tts]` 记录 TTS speech/sentence/first-audio/speech-end 耗时；`[telemetry][chat]` 记录客户端 audio queue/audio_end/motion_pending/playback_done/playback_complete；`[telemetry][action]` 记录情绪 tag → clip 映射；`[telemetry][motion]` 记录 clip enqueue/start/done/release。
  - 2026-06-20 实机复现：用户体感“最后半分钟唤醒看到小车转向，但是没有回答”。日志确认客户端 KWS/DOA/唤醒转向正常，服务端 LLM 仍出字，但 `qwen3-tts-flash` 的某句 `MultiModalConversation.call(..., stream=True)` 卡在流式 generator 内：`sentence_start idx=2` 后没有 `sentence_first_audio/sentence_done/speech_end`，直到 Wake session 15s 超时插入 farewell，后续 TTS worker 也被占住。已修复：Qwen multimodal TTS 增加**句级 idle timeout**，默认 3s；语义是“3s 内既没有结束，也没有新的音频帧”才取消该句，并继续发统一 `audio_end`，避免整条对话链路无回答。回归测试：`tests/unittest/test_qwen_tts_timeout.py`。
  - 2026-06-20 修复后复测：同句 `qwen3-tts-flash/Cherry` 探针 5 次首包 334-412ms、总耗时 575-649ms；2 分钟多轮对话未再出现 TTS idle timeout 或“转向但无回答”。ASR 样本多在 0.42-0.73s。本轮仍观察到动作门控主导的体感延迟：`[happy]` clip 约 6.8s、`[shy]` clip 约 9.9s，客户端当前会等 motion queue 清空再发 `playback_complete`，导致答完后 9-10s 才重新听下一句。
  - 2026-06-20 已修复动作门控体感延迟：`playback_complete` 现在只表示 TTS 音频播放结束，不再等待 motion queue；动作独立播放。服务端 `WsClient` 订阅 VAD 输出的 `HUMAN_AUDIO`，在 `human_speech_start` meta 出现时向客户端发 `{"type":"human_speech_start"}`；客户端收到该事件（或作为较晚兜底的 `asr_text`）会调用 `ActionDispatcher.cancel_all()` 中止当前动作。实机复测：`happy` 动作开始后 `playback_complete` 立即发送，下一句人声触发后 motion `aborted=True`，动作停止。
  - 2026-06-20 新增唤醒转向异常线索：用户观察到少数唤醒转向不对，甚至完全转反；最新反馈为三次里“只有最后一次对”。已加 `wake_doa_snapshot` 日志，打印 wake-window/dominant/latest 的 mic/body 角度和 offset。实测样本里前两次 `body_doa` 为负侧且旋转到正 yaw 后用户判断反向；最后一次接近 180°，信息量较弱。已把 DOA→IMU yaw 符号做成 `--wake-orient-doa-to-yaw-sign normal|inverted` 可配置项，默认保持历史 `delta=-DOA`。下一步需要用户站在明确左/右侧做带位置标签的 A/B 复测，区分“控制符号反了”还是“wake DOA 侧别/候选不稳定”。
  - TTS 备选：当前 `QwenTTS` handler 在 `model_name=cosyvoice-*` 时可走 DashScope `tts_v2 SpeechSynthesizer`，保留 action tag 剥离与 telemetry；wake reply 预生成也支持 `cosyvoice-*`。`tools/probe_tts_latency.py` 可 A/B `qwen3-tts-flash/Cherry` 与 CosyVoice v3。官方音色列表确认每个 CosyVoice 模型只支持自己的 voice 集合，`cosyvoice-v3-flash` 常用有效候选包括 `longanyang`、`longanhuan`、`longfei_v3`、`longyingxun_v3`、`longjielidou_v3`；实测延迟仍是 `qwen3-tts-flash/Cherry` 最低，默认配置暂不切换。
  - TODO：在下一次车上复现时同时抓客户端与服务端状态时间线：客户端 `is_playing` / `mute_until` / `mic_should_mute` / `playback_done_event` / `playback_complete` 发送时间；服务端 `wake_session_active` / `enable_vad` / `playback_complete` 接收时间 / wake failsafe 触发时间。目标是确认失败发生在客户端未发、服务端未收、还是 mute/VAD 状态时序错位。
  - 相关：`client/ws_audio_client.py`（mute/playback）、`wakeword/sherpa_kws/...`（wake reply + playback_complete 握手）、`vad/silerovad`、`client_handler_ws.py`。
- **0.2 ArUco/marker board 跟随卡顿/迟滞**✅ 已关闭（2026-06-19 用户确认）。
  - 2026-06-19 实机结论：单 tag/多 tag 直接选最近目标会在 3x3 marker 板上频繁换目标；暗环境下自动曝光/动态降帧导致运动模糊，检测掉点明显；固定曝光测试后检测稳定性显著改善，但固定曝光不适合对话视觉长期使用。
  - 已实现：默认把 ID 0-8 的 3x3 marker board 聚合成单一目标（45mm tag、约 7mm gap），要求至少 2 个 tag 才刷新 board pose；短暂丢失时先限速预测，再进入非驱动 hold；目标 pose 非有限/距离越界会立即停车；控制器线速度/角速度限幅，并降低角速度 D 项、加角速度变化率限制，避免 `vw` 数值尖峰。
  - 2026-06-20 RGB-D 恢复与回退：本次“前端画面几乎静止”最终定位为 USB 线/接头链路不稳；换线后 `0511` RGB 与 `0614` depth/control 同时枚举，depth-only C++/Python probe 与 SDK D2C probe 均通过。后续实测 CPU YOLO-pose + 底盘运动 + 多外设负载下出现黑屏/断网风险，且 depth/control 枚举仍有不稳定迹象；当前默认部署回退到 `--camera-mode color --camera /dev/video0`，先保障 RGB 前端画面和 marker 跟随。
  - 关闭依据：RGB-D SDK 模式 + 3x3 marker board 聚合 + 2 tag 刷新 + 短时预测 + 控制器限速/角速度变化率限制后，短电机测试未再出现危险前冲；第二轮 20s 窗口未见 `Tag lost -> IDLE` 抖动，用户认可 marker 追踪问题可关闭。
- **0.2.1 RGB-D 说话人跟随**🟡 代码已实现但默认关闭，待 NPU/算力/供电方案明确后再恢复实机验证。
  - 入口：客户端 KWS 唤醒后先走 `WakeOrientController` 转向声源；转向完成回调触发 RGB-D 视觉绑定 owner；绑定成功后才进入 `PersonFollowController`，绑定失败保持原地并打印 `wake bind failed; staying idle`。
  - 感知：`client/person_follow.py` 复用 `audio_frontend.backends.pose_detector.select_pose_detector()`、`IouTracker`、`estimate_mouth_3d()`；必须有 `SharedCamera.get_rgbd_frame()` 的 color+aligned depth，缺 depth 时记录 `person_tracking_disabled reason=rgbd_unavailable`，不进入人跟随。
  - 控制：复用 `TrackingParams` 的限速、死区、安全距离、角速度变化率限制；第一帧 owner 可见时只从 IDLE 进入 TRACKING 并发零速，避免单帧误检造成速度脉冲；owner 丢失超时停车回 IDLE。
  - 模式调整：当前已回退为 RGB/color 默认，marker 跟随默认启用；wake-driven person following 需显式 `--person-follow` 且配合 `--camera-mode rgbd-sdk --camera orbbec`。当前小车在桌上或低电量时，任何 marker/person 跟随实机测试前必须先征得用户许可。
  - 本地验证：`python tests\unittest\test_person_follow.py` 通过；`python -m py_compile client\person_follow.py client\main.py` 通过。
- **0.3 情绪 tag 与回答相关性低**🟡：表情/动作触发正常，但情绪标签和回答内容语义相关性不高。当前链路里 `config/system_prompt.txt` 要求模型在回答中直接插入 `[happy]`/`[shy]`/`[apologize]`/`[scared]`；`data/teach_bindings.json` 与 `client/action_dispatcher.py` 也只认这 4 个。
  - 2026-06-19 本地进展：已建立 `tests/fixtures/action_tag_eval_cases.jsonl`（44 条，含正例/负例/视觉上下文）与 `tools/eval_action_tags.py`，评测时显式发送 `enable_thinking=false`。旧 prompt + `qwen3.7-plus` 在真实 no-thinking 口径下 `29/44=65.9%`，主要问题是普通配合/视觉确认/技术问答误触发 `[happy]`/`[shy]`。围绕 flash 模型收紧 prompt 的"默认不打标签"和负例边界后，`qwen3.6-flash` 达到 `44/44=100.0%`；对照：`qwen3.7-plus` 为 `42/44=95.5%`，`qwen3-vl-flash` 为 `32/44=72.7%`（明显更爱乱加 `[happy]`）。`[scaed]` 近似拼写已在 `src/handlers/common/action_tags.py` 加窄容错。
  - 延迟结论：`qwen3.7-plus` 若不传/开启 thinking 会 10s 级首字；根部 `enable_thinking=false` 或 OpenAI SDK `extra_body={"enable_thinking": False}` 后，车端首字约 0.8-1.2s。车端同 prompt 三题快测：`qwen3-vl-flash` 首字均值约 310ms、`qwen3.6-flash` 约 391ms、`qwen3.7-plus` 约 782ms。DashScope 当前模型列表没有 `qwen3.6-vl-flash` 这个名字，但 `qwen3.6-flash` 实测可接受当前 `image_url` 输入。
  - 下一步：同步到车端后建议先试 `qwen3.6-flash`（当前评测准确率最高且车端首字约 0.4s），做 10-20 轮真实对话抽样；若现场复杂视觉理解明显不够，再 A/B `qwen3.7-plus`。
- **0.3.1 情绪动作结束/中断后的位姿漂移**✅ 已验证第一版。
  - 现象：手动录制动作经常不回到起点；VAD/human_speech_start 中断动作时，也可能停在动作中途，导致小车位姿逐轮漂移。
  - 已实现第一版：`MotionPlayer` 默认 `--motion-return`，真实 clip 播放期间用 `WheelImuOdometryBackend` 估计相对位姿，正常结束或 VAD 中断后进入低速 `returning` 回到动作起点；空 clip 不归位；PS2 接管、Teach UI 手动 stop、telemetry stale、服务退出均停车并放弃归位。2026-06-21 车端验证：`happy` 连续 3 次 `return_done reason=done`；尾端 practical settle 可避免几厘米/几度内来回摆动；`scared` 在 10s 默认超时下 `return_done reason=done`，从约 `x=-35.3cm,y=-15.2cm,yaw=-38.5deg` 回到约 `x=-0.6cm,y=-2.5cm,yaw=4.2deg`。
  - 供电风险：`scared` 长动作/归位压力测试后出现 Pi 黑屏断网、红灯常亮，可按红灯旁 reset 键恢复；重启后 `vcgencmd get_throttled=0x0`，日志没有明确 undervoltage，符合瞬时压降导致 SoC/USB/网络挂死但未可靠留日志的模式。复测大动作前应先改善供电余量，或增加 Pi 欠压/节流 telemetry 与动作压力测试记录。
  - RGB-D 里程计不进第一版：当前 depth/control 枚举和算力/供电余量仍不适合作为默认依赖；后续可在 Orbbec RGB-D 稳定后接 `RgbdOdometryBackend` 或 ROS2 odom + EKF，作为 IMU/编码器的低频校正源。
  - 本地验证：`tests/unittest/test_motion_odometry.py`、`tests/unittest/test_motion_player_return.py`、`tests/unittest/test_client_defaults.py`。
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
