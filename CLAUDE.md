# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of upstream **OpenAvatarChat** (a general conversational digital-human system) adapted to run a **physical smart car / robot**: a Raspberry Pi 5 "upper computer" running the voice+vision dialog stack, driving an STM32F103 "lower computer" (chassis/IMU/motors) over a serial protocol. The robot-specific code (`client/`, `audio_frontend/`, `ros2_ws/`, `tools/`, `firmware/`, `expression_player/`, robot configs) lives only on this fork; the dialog **server** is largely upstream.

> **Read first each session:** `docs/PROJECT_STATE.md` (what actually runs on the car — startup → which handlers/config are live vs. built-but-dormant, plus user-confirmed runtime status), `docs/TODO.md` (unfinished / blocked / Plan-B work + current known issues), and `docs/EVAL_PLAN.md` (test & baseline plan for verifying real runtime capability). Keep them updated as the deployment changes.

## Branch model — do not pollute upstream

Two remotes: **`upstream`** = `HumanAIGC-Engineering/OpenAvatarChat` (the real digital-human project, read-only reference); **`origin`** = `raidios/OpenAvatarChat` (the fork — push here).

- **`main`** tracks `upstream/main` = pristine upstream digital-human project. Keep it clean; never commit car work here. `git pull` brings clean upstream updates.
- **`smart-car`** = all robot/car work (the normal working branch); tracks `origin/smart-car` on the fork. Fork point from upstream is `93c7c4b` (#209).
- Push car work only to the fork (`git push origin smart-car`), never to `upstream`/`main`.

## Runtime topology (two processes)

The system is **two cooperating processes**, connected by a WebSocket:

1. **Dialog server** — `src/demo.py --config config/chat_rpi_voice.yaml`, listens on `:8282`. This is the upstream chat engine: a config-declared **pipeline of handlers** loaded from `src/handlers/` (see `src/chat_engine/`). For the car the pipeline is: `WsClient` (transport) → `WakeWord` (sherpa KWS, here in `external_wake_only` mode) → `SileroVad` → `QwenASR` → `LLMOpenAICompatible` → `QwenTTS`. ASR/LLM/TTS are **cloud calls to Aliyun DashScope** (Qwen), so it needs network + an API key, not a GPU.
2. **Robot client** — `client/main.py`, connects to the server's WebSocket and bridges all hardware: camera, far-field mic DSP, chassis serial, wake-driven rotation, teach/motion playback, expression face. Each subsystem is an **independently toggleable module** wired together in `client/main.py::run()` (read this file first to understand the client). Flags like `--no-chat`, `--no-tracking`, `--no-teach-ui`, `--farfield` turn modules on/off, which is how the `make` smoke targets isolate one subsystem.

Hardware-dependent code only works on the Pi (`ssh cyk@cykly.home.net`, repo at `/home/cyk/codes/OpenAvatarChat`). A Windows/dev checkout can edit and run pure-Python tests but cannot exercise the mic array, Hailo, camera, or chassis.

## Commands

Python runs through the uv-managed venv (`.venv/bin/python`, Python 3.11 on the Pi). `make help` lists all targets.

```bash
# Server (dialog engine)
.venv/bin/python src/demo.py --config config/chat_rpi_voice.yaml

# Robot client — full deployment (what the systemd unit runs)
.venv/bin/python client/main.py --farfield --server ws://127.0.0.1:8282/ws/chat \
    --serial-port /dev/ttyAMA0 --calib-file config/camera_calib.json \
    --farfield-mic-yaw-offset-deg 30 --wake-kws-keywords config/keywords.txt
# Or via make (far-field client without chassis-required extras):
make client-up-farfield

# Code-level tests (no hardware) — standalone scripts, not pytest
make test-stage0          # imports + backend-selection
make test-stage1-smoke    # SRP-PHAT/MVDR/AEC on synthetic signals
make test-stage2-smoke    # pose/tracker/mouth-3D on synthetic image
# Run a single test directly:
.venv/bin/python tests/test_stage1_pipeline.py

# Hardware acceptance (on the Pi, devices attached)
make test-stage0-hardware # M260C board mode + Hailo health
make test-stage1 BACKEND=cpu|hailo   # AEC+DOA+DNS+KWS against tests/data/<DATASET>
make test-regress         # all stages vs tests/results/baseline.json
make bless-baseline       # promote current run as new baseline

# Bench/diagnostic targets (on the Pi)
make wake-test            # M260C → DSP → KWS('小黑') → IMU rotation, no chat
make live-doa             # live body-frame DOA monitor
make doa-record AZ=+45    # record DOA ground truth; doa-test / doa-calibrate
make probe-yaw-polarity   # drive +/-0.5 rad/s, verify chassis & IMU yaw sign
```

Firmware (`firmware/OpenCTR_B20V11_2WD_IMU_V1.00.230912/`) builds **only in Keil MDK-ARM v5.36 with ARM Compiler 5** (not AC6), project `Project/xproject.uvprojx`, flashed via J-Link SWD, output `Project/Objects/xproject.hex`. Build artifacts are gitignored. Sources are **GB2312/GBK encoded** (AC5 limitation) with Chinese comments — preserve the encoding when editing.

## Deployment (on the car)

Autostart is via **systemd user units** (`~/.config/systemd/user/`) with linger enabled, so they start on power-up without login:
- `openavatarchat.service` → the server (`WantedBy=default.target`).
- `openavatarchat-client.service` → `client/main.py` (`WantedBy=gnome-session.target`, needs the graphical autologin for `DISPLAY=:0` + the expression window).

Template + the full ExecStart flag set: `scripts/openavatarchat-client.service.example`. The server reads `/home/cyk/codes/OpenAvatarChat/.env`, which must contain `DASHSCOPE_API_KEY` (consumed in `llm_handler_openai_compatible.py`; missing key fails handler init). Server also needs the sherpa KWS model under `models/` (download scripts in `scripts/`).

## Client subsystem notes

- **Far-field audio (`--farfield`, `client/farfield_audio_source.py` + `audio_frontend/`)**: pulls **raw 8-channel** from the M260C ring array (the board's `audio_server` over TCP `:9999`; `tools/board_audio_mode.py` flips the board into raw mode — do **not** read the 1-ch UAC1 ALSA gadget). DSP order is **per-mic AEC → SRP-PHAT DOA → MVDR beamform → (DNS, bypassed by default)** → clean 16 kHz mono PCM into the chat WebSocket. DNS is off by default (DTLN quantization buzz); enable with `--farfield-dns-on`.
- **Wake-orient (`client/wake_word_spotter.py` + `wake_orient_controller.py`)**: client-side sherpa KWS spots the wake word, gets the talker DOA, then closes an IMU loop to rotate the chassis toward the source. The server's `WakeWord` handler runs in `external_wake_only=true` (no server-side KWS decode) and is notified by the client.
- **Tracking (`apriltag_tracker.py` + `tracking_controller.py`)**: OpenCV ArUco/AprilTag follow with differential-drive control.
- **Teach/motion (`client/teach/`, `data/motion_clips/`)**: record/playback chassis motion clips, bound to dialog emotion tags via `action_dispatcher.py` + `data/teach_bindings.json`. A FastAPI debug UI (`client/teach_ui/`, default `:8080`) exposes calibration/recording; motion playback stays active even with `--no-teach-ui`.
- **Expression (`--expression`, `expression_player/`)**: fullscreen Qt/PySide6 face; asyncio runs in a worker thread. Falls back to windowless mode if SVG assets are missing.

## Critical cross-cutting facts

- **Coordinate frames**: all DSP (SRP-PHAT, MVDR, `extrinsic.npz`) works in **mic frame** (0° = ch0). The **body frame** (0° = vehicle forward) differs by `mic_yaw_offset_deg` (physically +30° on this robot). Convert mic↔body only at module boundaries (`FarfieldAudioSource`, ROS2 perception, teach UI, motion control) — never inside the DSP.
- **Serial / X-Protocol (`client/serial_comm.py` ↔ firmware `Driver/ax_uart4.c`)**: framed protocol over `/dev/ttyAMA0` ↔ STM32 **UART4** at 115200. Firmware runs a **50 Hz input arbiter** (`Robot/ax_robot.c`): PS2 gamepad > Pi `cmd_vel` (0x50, 300 ms freshness) > APP > IDLE-brake. The MCU reports the active source each tick (frame `0x12`); when PS2 has control the client should pause sending velocity. IMU `yaw_deg` drifts and is **odometry reference only** — it never drives motion.
- **Hardware platform** (Pi 5 + Hailo-10H NPU, M260C 6+2 mic array on a WHEELTEC R818 board, Orbbec Gemini Pro depth cam, STM32F103): the detailed, load-bearing device facts (device nodes, SDK selection, USB quirks, self-check commands) are in the Cursor rules — **read these before touching the corresponding subsystem**:
  - `.cursor/rules/hailo-device.mdc` — Hailo-10H, HailoRT 5.3.0, `/dev/h1x-0` (not `/dev/hailo0`).
  - `.cursor/rules/m260c-mic-array.mdc` — mic array link, raw-8ch vs demo mode, ch6/ch7 AEC reference, mic-yaw offset.
  - `.cursor/rules/orbbec-gemini-device.mdc` — depth camera, OrbbecSDK v1 only (not v2), SDK-vs-cv2 RGB trade-off.
  - `firmware/**/.cursor/rules/*.mdc` — STM32 project overview, **X-Protocol communication-protocol**, pin map, coding conventions.
  - `docs/decisions/ROS2_DISTRO.md` — ROS2 **Jazzy** (Python 3.12) vs the 3.11 venv tension for `ros2_ws/` and the `ros2_client` handler.
- **Vendored SDKs & weights are not in git**: `thirdparty/` (Orbbec/M260C SDKs), model weights (`*.hef/*.onnx/*.pt`), and `tests/data|metrics|results` live only on the Pi. Code that needs them must degrade gracefully when absent (the `--farfield`/hardware paths already do).
- **Hailo adaptations must stay optional/probed** — `/dev/h1x-0`, the cp311 venv, and specific device PIDs describe *this* machine, not every deploy target; keep them behind capability detection so changes remain upstream-mergeable.
