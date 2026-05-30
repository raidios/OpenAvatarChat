# OpenAvatarChat farfield audio frontend - dev convenience targets.
#
# Stage acceptance:
#   make test-stage0           # link probe + Hailo health + ROS2 bridge smoke
#   make test-stage1           # AEC + DOA + DNS + KWS (CPU and Hailo backends)
#   make test-stage1 BACKEND=cpu
#   make test-stage1 BACKEND=hailo
#   make test-stage2           # camera + person 3D + ReID
#   make test-regress          # all stages, compare against baseline
#   make bless-baseline        # promote latest results as new baseline
#
# Dataset:
#   make record DATASET=dataset_v1   # interactive recording

PY := .venv/bin/python
TOOLS := tools
TESTS := tests
RESULTS_DIR := tests/results
SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nosha)
RUN_DIR := $(RESULTS_DIR)/$(SHA)
BACKEND ?= auto
DATASET ?= dataset_v1

.PHONY: help test-stage0 test-stage0-hardware test-stage1 test-stage1-smoke \
        test-stage2 test-stage2-smoke fetch-yolo-pose calibrate-mic-camera \
        dump-camera-intrinsics \
        test-regress bless-baseline record clean-results dirs board-status \
        board-start-raw board-supervisor hailo-check probe-ref \
        farfield-source-smoke farfield-aec-smoke farfield-voice-smoke \
        dns-buzz-test doa-record doa-test mic-tap-calibrate \
        client-up-farfield wake-test live-doa probe-yaw-polarity

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN{FS=":.*##"}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

dirs:
	@mkdir -p $(RUN_DIR)

# ---- stage 0 ----------------------------------------------------------------

test-stage0: dirs ## stage 0 acceptance (no hardware required for code-only checks)
	@echo "==> stage 0: code-level checks"
	$(PY) $(TESTS)/test_imports.py
	$(PY) $(TESTS)/test_backends_select.py
	@echo
	@echo "Code-level stage 0 GREEN. Run hardware checks with:"
	@echo "  make test-stage0-hardware"

test-stage1-smoke: dirs ## stage 1 code-level smoke (synthetic signals only)
	@echo "==> stage 1 smoke: SRP-PHAT + MVDR + AEC + pipeline RTF"
	$(PY) $(TESTS)/test_stage1_pipeline.py

test-stage2-smoke: dirs ## stage 2 code-level smoke (synthetic image only)
	@echo "==> stage 2 smoke: pose + tracker + mouth-3D + gallery"
	$(PY) $(TESTS)/test_vision_smoke.py

# ---- stage 2 helpers --------------------------------------------------------

fetch-yolo-pose: ## fetch yolov8n-pose.pt and export ONNX into models/yolo/
	$(PY) $(TOOLS)/fetch_yolo_pose.py

calibrate-mic-camera: ## interactive aruco-based mic<->camera extrinsic calibration
	$(PY) $(TOOLS)/calibrate_mic_camera.py \
		--out tests/data/extrinsics/mic_camera_extrinsic.npz

dump-camera-intrinsics: ## read Gemini Pro factory K + D2C from EEPROM to config/camera_calib_factory.npz
	$(PY) $(TOOLS)/dump_factory_intrinsics.py \
		--out config/camera_calib_factory.npz

perception-quicklook: ## live RGB-D pose+mouth-3D quicklook (no ROS2 required)
	$(PY) $(TOOLS)/perception_quicklook.py --duration 30

test-stage0-hardware: dirs ## stage 0 acceptance with attached M260C + Hailo
	@echo "==> stage 0: hardware checks"
	$(PY) $(TOOLS)/board_audio_mode.py status --json | tee $(RUN_DIR)/board_status.json
	$(PY) $(TOOLS)/hailo_healthcheck.py --report $(RUN_DIR)/hailo_health.json --skip-infer
	@echo
	@echo "Optional: run probe-ref to verify ch6/ch7 reference channels:"
	@echo "  make probe-ref"

probe-ref: dirs ## verify ch6/ch7 ref channels correlate with TTS playback
	$(PY) $(TOOLS)/probe_ref_channel.py --duration 5 --out $(RUN_DIR)/ref_probe

board-status: ## show R818 audio_server + adb forward state
	$(PY) $(TOOLS)/board_audio_mode.py status

board-start-raw: ## switch board to raw mode + apply adb forward
	$(PY) $(TOOLS)/board_audio_mode.py start-raw

board-supervisor: ## run watchdog (Ctrl-C to stop)
	$(PY) $(TOOLS)/board_supervisor.py --interval 2 \
		--log /tmp/audio_supervisor.log

hailo-check: dirs ## one-shot hailo health (writes JSON report)
	$(PY) $(TOOLS)/hailo_healthcheck.py --report $(RUN_DIR)/hailo_health.json --skip-infer

# ---- stage 1 + 2 ------------------------------------------------------------

test-stage1: dirs ## stage 1 acceptance (BACKEND=cpu|hailo|auto)
	@echo "==> stage 1: BACKEND=$(BACKEND)"
	$(PY) $(TESTS)/run_stage1.py --backend $(BACKEND) \
		--dataset $(TESTS)/data/$(DATASET) --out $(RUN_DIR)/stage1_$(BACKEND).json

test-stage2: dirs ## stage 2 acceptance (BACKEND=cpu|hailo|auto)
	@echo "==> stage 2: BACKEND=$(BACKEND)"
	$(PY) $(TESTS)/run_stage2.py --backend $(BACKEND) \
		--dataset $(TESTS)/data/$(DATASET) --out $(RUN_DIR)/stage2_$(BACKEND).json

# ---- regression -------------------------------------------------------------

test-regress: test-stage0 ## run all stages, compare to baseline
	-$(MAKE) test-stage1 BACKEND=cpu
	-$(MAKE) test-stage1 BACKEND=hailo
	-$(MAKE) test-stage2 BACKEND=cpu
	-$(MAKE) test-stage2 BACKEND=hailo
	$(PY) $(TESTS)/regress_compare.py --run $(RUN_DIR) \
		--baseline $(RESULTS_DIR)/baseline.json

bless-baseline: ## promote latest run as the new baseline
	$(PY) $(TESTS)/bless_baseline.py --run $(RUN_DIR) \
		--baseline $(RESULTS_DIR)/baseline.json

# ---- dataset ----------------------------------------------------------------

record: ## interactive dataset_vX recorder
	$(PY) $(TOOLS)/record_dataset.py --dataset $(DATASET)

clean-results:
	rm -rf $(RUN_DIR)

# ---- ROS2-free dialog bring-up ----------------------------------------------
# Audio DSP frontend lives on the *client* side (M260C -> DSP -> mono PCM
# over WebSocket). The server keeps the existing chat_rpi_voice config; only
# the client gets the --farfield flag.

farfield-source-smoke: dirs ## 6s offline run of M260C->DSP->mono via FarfieldAudioSource
	$(PY) $(TESTS)/test_farfield_source_smoke.py --duration 6 \
		--out $(RUN_DIR)/farfield_source_capture.wav

farfield-aec-smoke: dirs ## 8s capture w/ chirp auto-played; reports AEC echo residual
	$(PY) $(TESTS)/test_farfield_source_smoke.py --duration 8 \
		--play-chirp --play-volume 0.6 \
		--out $(RUN_DIR)/farfield_aec_chirp.wav
	@echo ""
	@echo "==> A/B against passthrough DNS (isolates AEC behaviour):"
	$(PY) $(TESTS)/test_farfield_source_smoke.py --duration 8 \
		--play-chirp --play-volume 0.6 --dns-passthrough \
		--out $(RUN_DIR)/farfield_aec_chirp_no_dns.wav

farfield-voice-smoke: dirs ## 8s capture w/ voice (cherry.wav) auto-played; AEC perf on real ref
	$(PY) $(TESTS)/test_farfield_source_smoke.py --duration 8 \
		--play-wav $(TESTS)/data/cherry.wav --play-volume 0.6 \
		--out $(RUN_DIR)/farfield_voice_per_mic.wav
	@echo ""
	@echo "==> A/B legacy MVDR-then-AEC order (expect ~14 dB worse ERLE):"
	$(PY) $(TESTS)/test_farfield_source_smoke.py --duration 8 \
		--play-wav $(TESTS)/data/cherry.wav --play-volume 0.6 \
		--aec-after-mvdr \
		--out $(RUN_DIR)/farfield_voice_after_mvdr.wav

dns-buzz-test: dirs ## offline closed-loop test of DNS silence buzz; voice + silence stimulus
	$(PY) $(TESTS)/test_dns_buzz.py \
		--input $(TESTS)/data/cherry.wav \
		--out $(RUN_DIR)/dns_buzz

# ---- DOA ground-truth dataset & accuracy regression -------------------------
# Workflow (no KWS / VAD assumed; DOA from dominant source by RMS):
#   1. mark body 0° (vehicle forward) and a fixed radius (~1.5 m) on the floor
#   2. place a loud speaker at the labelled body angle, start playing a
#      reference clip (cherry.wav, music, sweep, whatever); ambient/stray
#      noise is OK as long as it's clearly quieter than the source
#   3. record from that angle:
#        make doa-record AZ=+45
#      The tool captures 6 s of 8-ch via the running board_audio_server
#      and writes raw 8-ch + .bf.wav (beamformed mono) + .ch0.wav baseline
#      into tests/data/doa_dataset/.
#   4. repeat for each angle (0, +45, +90, +135, +180, -135, -90, -45)
#   5. run `make doa-test` for aggregate accuracy + regenerated BF wavs.

mic-tap-calibrate: ## interactive tap-test to identify channel→mic-position mapping
	@echo "==> tap each mic CCW starting at body 0° (or pass START=...)"
	$(PY) $(TOOLS)/calibrate_mic_tap.py \
		$(if $(START),--start-body-az $(START)) \
		$(if $(CW),--cw)

doa-test-signal: ## regenerate the band-limited noise wav used for DOA recording
	$(PY) $(TOOLS)/gen_doa_test_signal.py \
		--out tests/data/doa_test_noise.wav \
		--duration $(if $(DURATION),$(DURATION),5)

doa-record: ## record one ground-truth clip (AZ=<deg>, optional LABEL=<name>)
	@if [ -z "$(AZ)" ]; then \
		echo "usage: make doa-record AZ=<deg> [LABEL=<name>] [DURATION=6] [PLAY_WAV=...]"; \
		echo "  AZ is body-frame azimuth in degrees, CCW positive (0 = vehicle forward)."; \
		echo "  PLAY_WAV defaults to tests/data/doa_test_noise.wav (band-limited noise,"; \
		echo "  spectrally flat over the SRP band; better than speech for DOA truth)."; \
		exit 2; \
	fi
	$(PY) $(TOOLS)/record_doa_truth.py \
		--body-az $(AZ) \
		--duration $(if $(DURATION),$(DURATION),6) \
		--label $(if $(LABEL),$(LABEL),noise) \
		--play-wav $(if $(PLAY_WAV),$(PLAY_WAV),tests/data/doa_test_noise.wav)

doa-record-cardinal: ## record the 4 cardinal angles (0/+90/+180/-90) in one run
	@echo "==> recording 4 cardinal DOA clips."
	@echo "    Source = speaker @ 0.5 m from mic ring, mic-array height."
	@echo "    Robot in room CENTER (>= 1 m from any wall) to minimise reflections."
	@for az in 0 90 180 -90; do \
		read -p "    Move source to body $$az° and press Enter (Ctrl-C to abort) " _ ; \
		$(PY) $(TOOLS)/record_doa_truth.py \
			--body-az $$az --duration 5 \
			--label noise --no-confirm \
			--play-wav tests/data/doa_test_noise.wav || exit $$? ; \
	done
	@echo "==> done. run 'make doa-calibrate' to re-fit perm + yaw"

doa-record-octagonal: ## record all 8 angles (0/+45/+90/+135/+180/-135/-90/-45)
	@echo "==> recording 8 DOA clips (cardinals + diagonals)."
	@echo "    Source = speaker @ 0.5 m from mic ring, mic-array height."
	@echo "    Robot in room CENTER; if some angle is close to a wall,"
	@echo "    rotate the WHOLE setup so all 8 placements have ~equal clearance."
	@echo "    Tip: mark the 8 positions on the floor with tape before starting."
	@for az in 0 45 90 135 180 -135 -90 -45; do \
		read -p "    Move source to body $$az° and press Enter (Ctrl-C to abort) " _ ; \
		$(PY) $(TOOLS)/record_doa_truth.py \
			--body-az $$az --duration 5 \
			--label noise --no-confirm \
			--play-wav tests/data/doa_test_noise.wav || exit $$? ; \
	done
	@echo "==> done. 'make doa-test' for accuracy summary, "
	@echo "    'make doa-calibrate' to re-fit perm + yaw if errors are high."

doa-calibrate: ## brute-force perm/yaw against the recorded dataset
	$(PY) $(TOOLS)/brute_calibrate_doa.py --include-approx

doa-test: dirs ## offline DOA regression on the recorded dataset
	$(PY) $(TESTS)/test_doa_real.py \
		--dataset $(TESTS)/data/doa_dataset

doa-bf-check: ## verify MVDR beamforming spatial discrimination
	$(PY) $(TOOLS)/analyze_bf_correctness.py \
		--dataset tests/data/doa_dataset \
		$(if $(SWEEP_DEG),--sweep-deg $(SWEEP_DEG)) \
		$(if $(NO_AEC),--no-aec)

client-up-farfield: ## start client/main.py with --farfield (M260C->DSP->ws)
	@echo "==> starting robot client with far-field DSP source"
	$(PY) client/main.py --farfield \
		--server $(if $(SERVER),$(SERVER),ws://127.0.0.1:8282/ws/chat) \
		$(if $(NO_VIDEO),--no-video) \
		$(if $(NO_TRACKING),--no-tracking) \
		$(if $(NO_TEACH_UI),--no-teach-ui)

probe-yaw-polarity: ## drive +0.5/-0.5 rad/s, observe direction + IMU yaw
	@echo "==> probe yaw / vw polarity (no audio; chassis only)"
	$(PY) $(TOOLS)/probe_yaw_polarity.py \
		--serial-port $(if $(SERIAL),$(SERIAL),/dev/ttyAMA0)

live-doa: ## stream live body-frame DOA (no KWS, no rotation; pure DSP)
	@echo "==> live DOA monitor: speak/play from a known direction,"
	@echo "    expect body=+90° at robot's left, -90° at right,"
	@echo "    180° behind, 0° in front."
	$(PY) $(TOOLS)/live_doa_monitor.py

wake-test: ## standalone wake->rotate test (no chat/tracking/teach UI)
	@echo "==> wake-test: M260C -> DSP -> KWS('小黑') -> IMU rotation"
	@echo "    serial=$(if $(SERIAL),$(SERIAL),/dev/ttyAMA0)"
	@echo "    keywords=$(if $(KEYWORDS),$(KEYWORDS),config/keywords.txt)"
	@echo "    threshold=$(if $(THRESHOLD),$(THRESHOLD),0.25)"
	@echo "    say the wake word (default: 小黑) at any direction;"
	@echo "    Ctrl-C to stop."
	$(PY) client/main.py \
		--farfield \
		--no-chat --no-tracking --no-teach-ui \
		--serial-port $(if $(SERIAL),$(SERIAL),/dev/ttyAMA0) \
		--wake-kws-keywords $(if $(KEYWORDS),$(KEYWORDS),config/keywords.txt) \
		--wake-kws-threshold $(if $(THRESHOLD),$(THRESHOLD),0.25)
