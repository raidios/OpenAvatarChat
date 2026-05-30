# 整体测试与基线评估计划（smart-car）

> 目标：车开机后**系统性跑通现状、建立可复现 baseline 指标**，并诊断三个已知问题（VAD 唤醒、ArUco 卡顿、情绪 tag 相关性）。方向由用户定：**先测全现状、建基线**。
> 配套：现状见 [PROJECT_STATE.md](PROJECT_STATE.md)，待办见 [TODO.md](TODO.md)。最后更新：2026-05-30。
> ⚠️ 大部分项需**车开机 + 联网**（车现在充电中）。结果建议写到 `tests/results/<sha>/`，可自动化的用 `make test-regress` + `make bless-baseline` 固化到 `tests/results/baseline.json`。

## 0. 开机前置自检（先确认底座健康，5 分钟）

```bash
ssh cyk@cykly.home.net
cd /home/cyk/codes/OpenAvatarChat
systemctl --user status openavatarchat openavatarchat-client   # 两个都 active
journalctl --user -u openavatarchat -n 30 --no-pager           # 服务端无异常
.venv/bin/python -c "import os;print('KEY set:', bool(os.getenv('DASHSCOPE_API_KEY')))"  # 在 .env 环境下
.venv/bin/python -c "from speexdsp import EchoCanceller; print('speexdsp OK')"  # 否则 AEC 静默退 NLMS！
ls -lh models/hailo/*.hef 2>/dev/null                          # DNS HEF 是否在
ping -c1 dashscope.aliyuncs.com                                # 联网
vcgencmd measure_temp; vcgencmd get_throttled                  # 温度/降频
```
记录：git sha、config 名、各 backend（aec/dns）、温度。

---

## A. 🔴 重点：VAD 唤醒困难诊断（接麦阵后）

**现象**：换麦阵前对话良好；接 M260C 远场链路后 VAD 有时唤不起来。
**两个假设**：(a) 麦阵 DSP 输出电平偏低 → Silero VAD 概率达不到 `speaking_threshold=0.25`；(b) 麦阵实现/链路本身问题。

**诊断步骤**：
1. **量 DSP 输出电平**：`make farfield-voice-smoke`（或 `farfield-source-smoke`）录 wav，看说话段 RMS/peak dBFS（client 已打 `[mic] rms/peak_dbFS`）。对比"说话 vs 静音"区分度；若说话段 peak < -25dBFS 基本就是电平问题。
2. **看 VAD 概率 vs 阈值**：在 `vad/silerovad` 临时打印 Silero 概率序列，对照 `speaking_threshold=0.25` / `start_delay=2048` / `end_delay=8000`。试探性降阈值或加增益，看是否解决。
3. **A/B 定位 (a) vs (b)**：同一句话，分别喂 ① 麦阵 DSP 输出、② 直接单麦（不带 DSP）给 VAD，比较触发率 → 区分是 DSP 电平还是麦阵链路。
4. 检查 DSP 末级是否需要 AGC/归一化（`audio_frontend/dsp/` 末级、MVDR 输出增益）。
**baseline 指标**：VAD 触发率（N 次说话成功唤醒数）、首字延迟、误唤醒率/小时。

## B. 🔴 重点：ArUco 跟踪卡顿/迟滞诊断

**现象**：方向/角度对、能驱动底盘，但动作不连贯、明显卡断。疑检测不连续/频繁丢失。
**强假设**：检测丢失 → 客户端停发 `0x50 cmd_vel` → 固件 **300ms 超时进 IDLE 刹车** → 一顿一顿。

**诊断步骤**：
1. **Profile 检测连续性**（在 `apriltag_tracker.py` 加计数/计时，或加一个 profile 脚本）：检测命中率、平均/最长连续丢失帧数、每帧检测耗时、相机实际帧率。
2. **确认相机**：`--camera auto` 实际选了哪个（V4L2 还是 Orbbec SDK color）、分辨率（640×480）、检测预设 `permissive`。低帧率/曝光导致运动模糊→丢失？
3. **控制环时序**：`tracking_controller.py` 发 `cmd_vel` 的频率；丢失时是否频繁 pause；**对照固件 `0x12` 仲裁源**（`--ps2-debug` 能打）——看是否频繁跌进 `AX_CTRL_SRC_IDLE`（证实/证伪 300ms 超时假设）。
4. 若证实超时假设：客户端在短暂丢失时**保持/平滑上一帧 cmd_vel**（< 300ms 内续发），或固件侧延长/平滑；同时降低丢失率（提帧率、调检测预设、tag 尺寸/光照）。
**baseline 指标**：检测命中率、最长连续丢失、控制指令间隔、检测→运动端到端延迟、IDLE 跌落次数/分钟。

## C. 🔴 重点：情绪 tag 与回答相关性

**现象**：表情/动作触发正常，但情绪标签与回答语义相关性低。疑 prompt 与模型契合度。

**诊断步骤**：
1. 采一批对话（10–20 轮），记录 LLM 输出的情绪 tag vs 回答文本，人工标"相关/无关"。
2. 审 `config/system_prompt.txt` 对情绪 tag 的指令是否明确；`src/handlers/common/action_tags.py` 白名单与 prompt 描述是否一致。
3. A/B：改 prompt（明确"按回答情绪选 tag"+给例子）/ 调温度 / 试更强模型；重测相关率。
**baseline 指标**：tag-内容相关率（抽样人工评，%）。

---

## D. 远场音频质量基线（半离线）

```bash
make farfield-aec-smoke      # chirp ref，报 AEC 回声残余 ERLE
make farfield-voice-smoke    # cherry.wav ref，真实语音 ERLE（注意 A/B 的 after-mvdr 对照）
make doa-octagonal           # 录 8 角真值（车置房间中央，源距 0.5m）
make doa-test                # DOA 角度误差汇总
make live-doa                # 实时 DOA 目视（左+90/右-90/前0/后180）
```
指标：ERLE(dB)、DOA 平均/最大角误差。**先确认 speexdsp 生效**（否则 ERLE 是 NLMS 的）。

## E. 唤醒转向基线
```bash
make probe-yaw-polarity      # 先确认 vw 极性与 IMU yaw 方向一致
make wake-test               # 唤醒→转向（不连 chat）
```
指标：转向角误差、收敛时间、超调、误触发。

## F. DNS（HEF 已成、buzz 待解）
```bash
.venv/bin/python tools/dtln_hef_setup.py --check
hailortcli benchmark --hef models/hailo/dtln_p1_h10h.hef   # 延迟/fps
make test-stage1 BACKEND=hailo   # vs BACKEND=cpu vs 默认 off：DNSMOS / RTF
```
- 复现 buzz：客户端加 `--farfield-dns-on` 录一段，谱图 + 听感量化 buzz。
- 决策依据：buzz 能否靠扩校准集 / 提 `precision_mode` / 真实语音校准压下去，否则换模型（FRCRN/GTCRN）。

## G. 实时性 / 资源基线（默认配置 DNS off）
```bash
.venv/bin/python tests/run_pipeline_offline.py tests/results/<sha>/ref_probe/capture_8ch.wav \
    --pipeline full --aec-backend speex --out-clean /tmp/clean.wav   # 计时算 RTF
top -b -n3 | head -20        # 跑对话时 CPU%
vcgencmd measure_temp        # 温度
```
指标：整链 RTF、各级耗时占比、Pi5 CPU%/温度峰值、内存、续航（满电→欠压时长，client log 有 `bat=xx.xxV`）。

## H. 安全兜底回归
- PS2：SELECT 急停、START 恢复（`client/main.py` 已绑）。
- `cmd_vel` 300ms 超时停车；`--ps2-debug` 看 `0x12` 仲裁源切换。
- 确认 PS2 接管时客户端按建议停发 cmd_vel。

---

## 落盘
- 自动化项：`make test-regress` → `make bless-baseline`（固化到 `tests/results/baseline.json`）。
- 人工/半自动项（对话触发率、ArUco profile、情绪相关率）：结果记到 `tests/results/<sha>/` 或追加到本文"基线快照"小节，供回归对比。
