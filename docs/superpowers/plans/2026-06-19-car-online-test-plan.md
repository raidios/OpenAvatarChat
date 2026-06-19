# Car Online Test Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the car is online, collect enough synchronized telemetry to diagnose intermittent VAD wake/listen failures and ArUco marker-follow stutter without guessing.

**Architecture:** Start with read-only health checks and journald capture. If existing logs are insufficient, add a small opt-in telemetry mode that writes JSONL events for client playback/mic mute, server wake/VAD gating, ArUco detections, tracking state changes, and serial arbitration. Keep telemetry disabled by default.

**Tech Stack:** Raspberry Pi Ubuntu, systemd user services, Python 3.11 `.venv` on the car, Makefile smoke targets, journald, optional JSONL telemetry under `tests/results/<sha>/`.

---

## Operating Rules

- Run car-side commands from `/home/cyk/codes/OpenAvatarChat`.
- Keep movement tests low-speed, with the robot lifted or in a clear area unless the step explicitly needs floor motion.
- Do not tune thresholds during evidence capture. First capture successful and failed cases with the same config.
- Windows checkout path is `D:\Projects\cyk\OpenAvatarChat`; WSL path is `/mnt/d/Projects/cyk/OpenAvatarChat`.
- The checked-out `.venv/bin/python` under WSL is not executable as Linux Python in this workspace; use car-side `.venv/bin/python` for hardware tests.

## Task 1: Online Health Snapshot

**Files:**
- Read: `docs/PROJECT_STATE.md`
- Read: `docs/TODO.md`
- Output: `tests/results/<sha>/online_health.txt` on the car

- [ ] **Step 1: Connect and identify the deployed revision**

Run on the car:

```bash
ssh cyk@cykly.home.net
cd /home/cyk/codes/OpenAvatarChat
sha=$(git rev-parse --short HEAD)
mkdir -p "tests/results/$sha"
{
  date -Is
  git status --short --branch
  git log --oneline -n 5
} | tee "tests/results/$sha/online_health.txt"
```

Expected: branch is `smart-car`; services may already be running.

- [ ] **Step 2: Check services and core environment**

Run on the car:

```bash
cd /home/cyk/codes/OpenAvatarChat
sha=$(git rev-parse --short HEAD)
{
  systemctl --user status openavatarchat openavatarchat-client --no-pager
  journalctl --user -u openavatarchat -n 80 --no-pager
  journalctl --user -u openavatarchat-client -n 120 --no-pager
  .venv/bin/python -c "import os; print('KEY set:', bool(os.getenv('DASHSCOPE_API_KEY')))"
  .venv/bin/python -c "from speexdsp import EchoCanceller; print('speexdsp OK')"
  ping -c1 dashscope.aliyuncs.com
  vcgencmd measure_temp
  vcgencmd get_throttled
} | tee -a "tests/results/$sha/online_health.txt"
```

Expected: both services active, DashScope reachable, `speexdsp OK`, no throttling flags indicating undervoltage/throttle.

## Task 2: Passive Log Capture

**Files:**
- Output: `tests/results/<sha>/server-live.log`
- Output: `tests/results/<sha>/client-live.log`
- Output: `tests/results/<sha>/resource-live.log`

- [ ] **Step 1: Start log tails before touching the robot**

Run on the car in separate shells or tmux panes:

```bash
cd /home/cyk/codes/OpenAvatarChat
sha=$(git rev-parse --short HEAD)
journalctl --user -fu openavatarchat --no-pager | tee "tests/results/$sha/server-live.log"
```

```bash
cd /home/cyk/codes/OpenAvatarChat
sha=$(git rev-parse --short HEAD)
journalctl --user -fu openavatarchat-client --no-pager | tee "tests/results/$sha/client-live.log"
```

```bash
cd /home/cyk/codes/OpenAvatarChat
sha=$(git rev-parse --short HEAD)
while true; do
  date -Is
  vcgencmd measure_temp
  vcgencmd get_throttled
  ps -eo pid,psr,pcpu,pmem,comm,args --sort=-pcpu | head -20
  sleep 2
done | tee "tests/results/$sha/resource-live.log"
```

Expected: client log shows camera/tracking/farfield startup; resource log shows whether a Python thread/process is saturating a core.

## Task 3: VAD Wake Failure Reproduction

**Files:**
- Observe: `client/ws_audio_client.py`
- Observe: `src/handlers/wakeword/sherpa_kws/wakeword_handler_sherpa.py`
- Observe: `src/handlers/client/ws_client/client_handler_ws.py`
- Output: `tests/results/<sha>/vad_trials.md`

- [ ] **Step 1: Run five normal wake-dialog trials**

User action:

```text
For each trial:
1. Say wake word from normal position.
2. Wait for wake reply audio to finish.
3. Immediately say a short fixed phrase, for example "今天天气怎么样".
4. Say whether the robot responded, ignored the phrase, or responded late.
```

Observer notes in `vad_trials.md`:

```markdown
| trial | time | wake reply heard | user phrase accepted | delay | notes |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
```

Expected evidence to look for:

- Server line: `Client playback complete, re-enabling VAD`
- Server warning: `Wake reply did not receive playback_complete in time; enabling VAD (server failsafe)`
- VAD lines: `Start of human speech`, `End of human speech`, `VAD debug status=... max_prob=...`
- Client-side mute/playback clues, if present.

- [ ] **Step 2: If one trial fails, immediately repeat a controlled follow-up**

User action:

```text
Stay in the same position. Repeat the wake word and fixed phrase once more.
Do not change volume or distance.
```

Observer decision:

- If the retry works and failed trial had wake failsafe warning: prioritize playback_complete/mute telemetry.
- If the retry also fails and VAD max_prob stays high: prioritize enable_vad/session state telemetry.
- If VAD max_prob is low only in failure: re-check farfield audio path, but do not resurrect the old "level too low" hypothesis without confirmed speech windows.

## Task 4: Marker Follow Stutter Reproduction

**Files:**
- Observe: `client/apriltag_tracker.py`
- Observe: `client/tracking_controller.py`
- Observe: `client/camera.py`
- Output: `tests/results/<sha>/tracking_trials.md`

- [ ] **Step 1: Static marker continuity test**

User action:

```text
Hold the marker still at the normal tracking distance for 60 seconds.
Keep lighting and marker angle steady.
Do not let the robot move if it is unsafe; lift wheels or disable motor power if needed.
```

Observer notes:

```markdown
| window | tag visible pattern | Tag lost count | longest apparent loss | tracking state pattern | notes |
|---|---|---:|---:|---|---|
| 0-20s |  |  |  |  |  |
| 20-40s |  |  |  |  |  |
| 40-60s |  |  |  |  |  |
```

Expected evidence:

- Repeated `[tracking] Tag lost ...`
- Repeated `TRACKING -> IDLE`
- Long gaps before detections return even while marker is static.

- [ ] **Step 2: Repeat with chat/farfield isolated if needed**

If static marker loss is frequent during full deployment, compare with tracking-only mode:

```bash
cd /home/cyk/codes/OpenAvatarChat
.venv/bin/python client/main.py \
  --no-chat --no-teach-ui \
  --server ws://127.0.0.1:8282/ws/chat \
  --serial-port /dev/ttyAMA0 \
  --tag-size 0.045 \
  --calib-file config/camera_calib.json \
  --ps2-debug
```

Expected:

- If tracking-only is smooth but full deployment stutters, the DSP/chat workload starvation hypothesis gets stronger.
- If tracking-only still stutters, prioritize camera selection/fps/exposure/tag detection parameters.

## Task 5: Optional Telemetry Patch

**Files:**
- Modify: `client/main.py`
- Modify: `client/ws_audio_client.py`
- Modify: `client/apriltag_tracker.py`
- Modify: `client/tracking_controller.py`
- Modify: `src/handlers/client/ws_client/client_handler_ws.py`
- Modify: `src/handlers/wakeword/sherpa_kws/wakeword_handler_sherpa.py`
- Create: `client/telemetry.py`
- Create: `tests/test_telemetry_jsonl.py`

- [ ] **Step 1: Only implement if passive logs are insufficient**

Trigger condition:

```text
We saw at least one VAD failure or marker stutter, but journald does not show
enough timing/state detail to decide which component boundary failed.
```

- [ ] **Step 2: Add opt-in JSONL telemetry**

Implementation shape:

```text
Add --telemetry-jsonl PATH to client/main.py.
When provided, emit one JSON object per line with:
  t_monotonic, t_wall, component, event, and event-specific fields.
Keep disabled by default.
```

Client events:

```text
audio.enqueue
audio.playback_done_event_set
audio.playback_complete_send
audio.mic_muted_sampled
aruco.frame
aruco.detected
aruco.no_detection
tracking.state_change
tracking.tag_lost
tracking.cmd_vel
```

Server events can go to normal logger first if plumbing a shared JSONL path into server config is too invasive:

```text
wake.external_event
wake.vad_disabled_for_reply
wake.failsafe_scheduled
wake.failsafe_fired
ws.playback_complete_received
ws.enable_vad_true
vad.debug_window
```

- [ ] **Step 3: Validate telemetry locally before car deployment**

On WSL, do not use the checked-out `.venv/bin/python`. Use a Linux venv or system Python:

```bash
cd /mnt/d/Projects/cyk/OpenAvatarChat
python3 -m venv /tmp/oac-telemetry-venv
/tmp/oac-telemetry-venv/bin/python -m pip install -e .
/tmp/oac-telemetry-venv/bin/python tests/test_telemetry_jsonl.py
```

Expected: telemetry test writes valid JSONL and preserves monotonic timestamps.

## Task 6: Decision Table After Capture

**Files:**
- Output: `tests/results/<sha>/diagnosis.md`

- [ ] **Step 1: Classify VAD result**

Write one of:

```markdown
## VAD Diagnosis

- Classification: client did not send playback_complete | server did not receive playback_complete | VAD stayed disabled | mic muted too long | true low VAD probability | inconclusive
- Evidence:
  - Client:
  - Server:
  - VAD:
- Next change:
```

- [ ] **Step 2: Classify marker stutter result**

Write one of:

```markdown
## Marker Diagnosis

- Classification: detection starvation under full workload | camera/fps/exposure problem | controller too eager to stop on brief loss | serial arbitration/300ms freshness problem | inconclusive
- Evidence:
  - Detection:
  - Tracking controller:
  - Resource usage:
  - Serial/PS2:
- Next change:
```

- [ ] **Step 3: Pick exactly one first fix**

Allowed first fixes:

- VAD: add precise playback/mute/VAD telemetry if not already added; otherwise fix the confirmed boundary only.
- Marker: add short tag-loss hold/debounce in `tracking_controller.py` if detection gaps are brief; if gaps are multi-second under load, reduce DSP GIL pressure or isolate DSP before tuning controller.

Do not combine VAD and marker fixes in the same commit.
