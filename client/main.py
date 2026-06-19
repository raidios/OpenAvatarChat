#!/usr/bin/env python3
"""
Robot client main entry point.

Integrates optional modules:
  - Chat (voice + vision dialogue over WebSocket)
  - ArUco marker tracking (camera + differential-drive control)
  - Serial communication with the MCU (X-Protocol)

Usage examples:

    # Chat only (original behaviour)
    python main.py --server ws://host:8282/ws/chat

    # Tracking only (no chat server)
    python main.py --no-chat --serial-port /dev/ttyAMA0

    # Both chat and tracking
    python main.py --server ws://host:8282/ws/chat --serial-port /dev/ttyAMA0

    # Tracking with calibrated camera (default dict: 5X5_250)
    python main.py --no-chat --serial-port /dev/ttyAMA0 \
                   --calib-file config/camera_calib.json --tag-size 0.045

    # Fullscreen expression + chat + tracking (after desktop login)
    python main.py --expression --server ws://127.0.0.1:8282/ws/chat \\
                   --serial-port /dev/ttyAMA0 --calib-file config/camera_calib.json
"""

import argparse
import asyncio
import logging
import math
import signal
import sys
import os
import threading
import time
from typing import Any, Dict, Optional

# Surface wake_word_spotter / wake_orient_controller INFO logs (and any
# other library logging) on stdout so the wake-test prints are useful.
# Idempotent: calling basicConfig twice is a no-op.
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(name)s] %(message)s",
)

# Add client directory to path so modules can be imported directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from camera import SharedCamera
from opencv_gui import (
    ensure_local_display,
    opencv_highgui_available,
    print_display_failure_help,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robot client: chat + ArUco marker tracking"
    )

    # -- camera --------------------------------------------------------------
    cam = parser.add_argument_group("Camera")
    cam.add_argument("--camera", type=str, default="auto",
                     help="Camera: 'auto' (first V4L2 capture-capable /dev/videoN, "
                          "else Orbbec SDK if lib present), 'orbbec' (Gemini Pro SDK "
                          "color-only), index (e.g. 0), or path (e.g. /dev/video0). "
                          "Note: Orbbec RGB via SDK uses 640x480@30 MJPG on this device.")
    cam.add_argument("--camera-width", type=int, default=640)
    cam.add_argument("--camera-height", type=int, default=480)

    # -- chat ----------------------------------------------------------------
    chat = parser.add_argument_group("Chat")
    chat.add_argument("--server", type=str,
                      default="ws://localhost:8282/ws/chat",
                      help="WebSocket server URL")
    chat.add_argument("--no-chat", action="store_true",
                      help="Disable chat (voice + vision dialogue)")
    chat.add_argument("--no-video", action="store_true",
                      help="Disable video sending to chat server")

    # -- far-field audio -----------------------------------------------------
    ff = parser.add_argument_group(
        "Far-field audio (M260C 6+2ch → DSP → mono)",
        "Replaces the default PyAudio mic with a LiveStream-fed DSP "
        "frontend (SRP-PHAT DOA → MVDR → AEC → DNS) that emits clean "
        "16-kHz mono int16 PCM into the chat WebSocket.",
    )
    ff.add_argument("--farfield", action="store_true",
                    help="Enable the far-field DSP audio source")
    ff.add_argument("--farfield-host", type=str, default="127.0.0.1",
                    help="LiveStream TCP host (default: 127.0.0.1)")
    ff.add_argument("--farfield-port", type=int, default=9999,
                    help="LiveStream TCP port (default: 9999, matches "
                         "tools/board_audio_mode.py)")
    ff.add_argument("--farfield-aec", type=str, default="auto",
                    choices=["auto", "speex", "nlms", "fdaf"],
                    help="AEC backend (default: auto -> Speex MDF; "
                         "voice-ref probe gives +19.2 dB ERLE per-mic-"
                         "before-MVDR, falls back to NLMS only if "
                         "speexdsp native lib is missing)")
    ff.add_argument("--farfield-aec-filter-ms", type=int, default=200,
                    help="AEC adaptive filter tail length in ms. "
                         "Default 200 (Speex MDF wants comfortable "
                         "head-room over the room tail; voice-ref "
                         "probe peaks here). Drop to 20-50 for NLMS.")
    ff.add_argument("--farfield-aec-per-mic",
                    dest="farfield_aec_per_mic",
                    action="store_true", default=True,
                    help="Run one AEC instance per mic before MVDR. "
                         "Voice-ref probe: +19.2 dB ERLE, vs +4.5 dB "
                         "for the legacy MVDR-then-AEC order. Default.")
    ff.add_argument("--farfield-aec-after-mvdr",
                    dest="farfield_aec_per_mic",
                    action="store_false",
                    help="Legacy: single AEC after MVDR. Cheaper "
                         "(1 instance vs 6) but ~14 dB worse ERLE.")
    ff.add_argument("--farfield-dns", type=str, default="auto",
                    choices=["auto", "cpu", "hailo"],
                    help="DNS backend when DNS is enabled: auto (hailo if "
                         "HEFs+driver present, else cpu DfNet) | cpu "
                         "(DeepFilterNet on CPU) | hailo (DTLN HEF). "
                         "Currently NO-OP unless --farfield-dns-on is "
                         "passed: by default the DNS stage is bypassed "
                         "because DTLN v7 emits intermittent quantisation "
                         "buzz that bothers users more than the +10 dB "
                         "extra echo-residual reduction it offers. AEC + "
                         "MVDR alone give -7 dB residual on voice ref, "
                         "which is sufficient for ASR / KWS in normal "
                         "indoor conditions.")
    ff.add_argument("--farfield-dns-on", dest="farfield_dns_passthrough",
                    action="store_false", default=True,
                    help="Re-enable the DNS stage (default is bypassed). "
                         "Use in noisy environments where the residual "
                         "stationary noise hurts ASR more than the buzz.")
    ff.add_argument("--farfield-dns-passthrough",
                    dest="farfield_dns_passthrough",
                    action="store_true",
                    help="(Default) Skip DNS entirely.")
    ff.add_argument("--farfield-no-doa", action="store_true",
                    help="Disable DOA-driven MVDR steering (use static azimuth)")
    ff.add_argument("--farfield-static-az", type=float, default=0.0,
                    help="MVDR target azimuth in degrees, BODY frame "
                         "(0° = vehicle forward, CCW positive). Used when "
                         "--farfield-no-doa. Default: 0")
    ff.add_argument("--farfield-mic-yaw-offset-deg", type=float, default=30.0,
                    help="Yaw of mic-array 0° (ch0 direction) relative to "
                         "vehicle forward, CCW positive viewed from above. "
                         "Physical install on this robot has the array "
                         "rotated 30° CCW of the chassis heading; override "
                         "if the array is remounted. Default: 30")
    ff.add_argument("--farfield-no-board-bootstrap", action="store_true",
                    help="Skip running tools/board_audio_mode.py start-raw "
                         "before connecting (use if board is already in raw mode)")

    # -- wake orient (KWS-driven chassis rotation) ---------------------------
    wo = parser.add_argument_group("Wake-driven orient")
    wo.add_argument("--no-wake-orient", action="store_true",
                    help="Disable client-side KWS + chassis rotation on wake. "
                         "Server-side KWS keeps running for the conversation "
                         "flow regardless.")
    wo.add_argument("--wake-kws-model-dir", type=str,
                    default="models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
                    help="Sherpa KWS model directory (mirrors server config).")
    wo.add_argument("--wake-kws-keywords", type=str,
                    default="config/keywords.txt",
                    help="Sherpa KWS keywords lexicon file (mirrors server).")
    wo.add_argument("--wake-kws-threshold", type=float, default=0.25,
                    help="Sherpa keywords_threshold (default 0.25, server value).")
    wo.add_argument("--wake-orient-min-deg", type=float, default=8.0,
                    help="Don't bother rotating for wake DOAs within this "
                         "many degrees of body forward (default 8°).")
    wo.add_argument("--wake-orient-deadzone-deg", type=float, default=5.0,
                    help="Stop rotating once |yaw error| is this small for "
                         "settle_ticks ticks in a row (default 5°).")
    wo.add_argument("--wake-orient-max-omega", type=float, default=2.5,
                    help="Cap rotation speed in rad/s (default 2.5).")
    wo.add_argument("--wake-orient-timeout", type=float, default=6.0,
                    help="Hard timeout per rotation in seconds (default 6).")

    # -- tracking ------------------------------------------------------------
    trk = parser.add_argument_group("Marker tracking (OpenCV ArUco)")
    trk.add_argument("--no-tracking", action="store_true",
                     help="Disable marker tracking")
    trk.add_argument("--tag-size", type=float, default=0.045,
                     help="ACTUAL printed tag edge length in metres "
                          "(default: 0.045 = 45 mm, matches generate_aruco.py --size 45)")
    trk.add_argument("--aruco-dict", type=str, default="5X5_250",
                     help="OpenCV dict (default: 5X5_250). Others: 4X4_50, "
                          "APRILTAG_36H11 (tag36h11 prints), ... (cv2.aruco.DICT_*)")
    trk.add_argument("--aruco-preset", type=str, default="permissive",
                     choices=["permissive", "default", "refine"],
                     help="DetectorParameters preset (default: permissive, robust on Pi)")
    trk.add_argument("--calib-file", type=str, default=None,
                     help="Camera calibration JSON file")
    trk.add_argument("--target-tag-id", type=int, default=None,
                     help="Only track this tag ID (default: any)")
    trk.add_argument("--tracking-distance", type=float, default=0.6,
                     help="Desired follow distance in metres (default: 0.6)")
    trk.add_argument("--marker-board-gap", type=float, default=0.007,
                     help="Gap between markers in the 3x3 board, in metres "
                          "(default: 0.007)")
    trk.add_argument("--marker-board-lost-timeout", type=float, default=0.35,
                     help="Seconds to hold the last board pose across short "
                          "detector dropouts (default: 0.35)")
    trk.add_argument("--marker-board-smoothing-alpha", type=float, default=0.35,
                     help="Low-pass alpha for board pose smoothing; 1 disables "
                          "smoothing (default: 0.35)")

    # -- serial --------------------------------------------------------------
    ser = parser.add_argument_group("Serial (MCU)")
    ser.add_argument("--serial-port", type=str, default=None,
                     help="Serial port device (e.g. /dev/ttyAMA0)")
    ser.add_argument("--serial-baud", type=int, default=115200,
                     help="Serial baud rate (default: 115200)")
    ser.add_argument("--ps2-debug", action="store_true",
                     help="Periodically log PS2 sticks/buttons + arbitration state "
                          "(source, flags, target velocity) reported by the MCU. "
                          "Useful when there is no debug serial on the MCU side.")
    ser.add_argument("--ps2-debug-interval", type=float, default=0.5,
                     help="Seconds between --ps2-debug log lines (default: 0.5)")

    # -- teach UI ------------------------------------------------------------
    teach = parser.add_argument_group("Teach / Debug UI")
    teach.add_argument("--no-teach-ui", action="store_true",
                       help="Disable the on-board teach/debug FastAPI server")
    teach.add_argument("--teach-ui-port", type=int, default=8080,
                       help="Port for the teach/debug UI (default: 8080)")
    teach.add_argument("--teach-ui-bind", type=str, default="0.0.0.0",
                       help="Bind address for the teach/debug UI (default: 0.0.0.0; "
                            "use 127.0.0.1 to restrict to localhost)")
    teach.add_argument("--motion-data-dir", type=str, default="data/motion_clips",
                       help="Directory for motion clip JSON files (default: data/motion_clips)")
    teach.add_argument("--teach-bindings-file", type=str, default="data/teach_bindings.json",
                       help="Path to emotion-to-clip+expression binding JSON")
    teach.add_argument("--calib-output", type=str, default="config/camera_calib.json",
                       help="Path the teach UI writes calibration result to "
                            "(default: config/camera_calib.json)")

    # -- debug display -------------------------------------------------------
    dbg = parser.add_argument_group("Debug display")
    dbg.add_argument("--display", action="store_true",
                     help="Show live camera view with debug overlay on local screen")
    dbg.add_argument("--display-fps", type=float, default=15.0,
                     help="Debug display refresh rate (default: 15)")
    dbg.add_argument("--expression", action="store_true",
                     help="Show fullscreen expression window (Qt); asyncio runs in a worker thread")
    dbg.add_argument("--expression-svg-dir", type=str, default=None,
                     help="SVG 目录（眨眼四帧 + 开心/乐/死/皱眉）；默认仓库 expression_player/images 或 EXPRESSION_SVG_DIR")
    dbg.add_argument("--expression-defer-fullscreen-ms", type=int, default=250,
                     help="Delay before fullscreen (ms) after window show, for desktop session")
    dbg.add_argument("--expression-window", action="store_true",
                     help="With --expression: use a normal window instead of fullscreen")
    dbg.add_argument("--tag-expression-holdoff-ms", type=int, default=800,
                     help="With --expression: after tag lost, keep tag expression this long (ms) "
                          "to reduce flicker from unstable detection (default: 800, 0 = off)")

    return parser


def _console_log_loop(
    stop_flag: threading.Event,
    tag_tracker=None,
    tracking_ctl=None,
    serial_conn=None,
    interval: float = 1.0,
):
    """Background thread: prints ArUco and serial status to stdout periodically."""
    last_state = None
    last_estopped = False
    last_det_str = ""

    while not stop_flag.is_set():
        lines = []

        if tag_tracker is not None:
            dets = tag_tracker.detections
            if dets:
                parts = []
                for d in dets:
                    s = f"id={d.tag_id}"
                    if d.distance is not None:
                        s += f" dist={d.distance:.3f}m"
                    if d.angle_h is not None:
                        s += f" angle_h={math.degrees(d.angle_h):.1f}°"
                    if d.angle_v is not None:
                        s += f" angle_v={math.degrees(d.angle_v):.1f}°"
                    parts.append(s)
                det_str = " | ".join(parts)
                if det_str != last_det_str:
                    lines.append(f"[aruco] Detected: {det_str}")
                    last_det_str = det_str
            else:
                if last_det_str:
                    lines.append("[aruco] No tag detected")
                    last_det_str = ""

        if tracking_ctl is not None:
            state = tracking_ctl.state
            estopped = tracking_ctl.is_estopped
            state_str = f"{state.value} [E-STOP]" if estopped else state.value
            if state != last_state or estopped != last_estopped:
                lines.append(f"[tracking] State: {state_str}")
                last_state = state
                last_estopped = estopped

        if serial_conn is not None:
            data = serial_conn.latest_data
            yaw = serial_conn.yaw_deg
            bias_dps = serial_conn.gyro_z_bias_dps
            bias_str = (
                f"bias={bias_dps:+.2f}dps" if bias_dps is not None else "bias=--"
            )
            lines.append(
                f"[serial] yaw={yaw:+6.1f}° {bias_str} "
                f"bat={data.bat_voltage / 100.0:.2f}V "
                f"vel=({data.vel_x},{data.vel_y},{data.vel_w})"
            )

        for line in lines:
            print(line)

        stop_flag.wait(interval)


def _debug_display_loop(
    camera: SharedCamera,
    stop_flag: threading.Event,
    fps: float,
    tag_tracker=None,
    tracking_ctl=None,
    serial_conn=None,
):
    """Background thread that shows a live debug window on the local screen."""
    win_name = "RobotCar Debug"
    interval = 1.0 / fps
    frame_count = 0
    fps_timer = time.monotonic()
    display_fps = 0.0

    while not stop_flag.is_set():
        t0 = time.monotonic()
        frame, _ = camera.get_frame_raw()
        if frame is None:
            time.sleep(0.05)
            continue

        # marker overlay
        if tag_tracker is not None:
            frame = tag_tracker.draw_detections(frame)

        h, w = frame.shape[:2]

        # --- HUD overlay ---
        y = 20
        line_h = 22
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5
        th = 1

        def put(text, color=(0, 255, 0)):
            nonlocal y
            # dark background strip for readability
            cv2.rectangle(frame, (0, y - 15), (w, y + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (5, y), font, fs, color, th)
            y += line_h

        # FPS counter
        frame_count += 1
        elapsed_fps = t0 - fps_timer
        if elapsed_fps >= 1.0:
            display_fps = frame_count / elapsed_fps
            frame_count = 0
            fps_timer = t0
        put(f"FPS: {display_fps:.1f}")

        # Tracking state
        if tracking_ctl is not None:
            state = tracking_ctl.state
            state_colors = {
                "IDLE": (180, 180, 180),
                "TRACKING": (0, 255, 0),
            }
            color = state_colors.get(state.value, (255, 255, 255))
            put(f"State: {state.value}", color)

        # Tag detection info
        if tag_tracker is not None:
            dets = tag_tracker.detections
            if dets:
                for d in dets:
                    info = f"Tag {d.tag_id}"
                    if d.distance is not None:
                        info += f"  dist={d.distance:.2f}m"
                    if d.angle_h is not None:
                        info += f"  angle={math.degrees(d.angle_h):.1f}deg"
                    put(info)
            else:
                put("No tag detected", (100, 100, 100))

        # MCU / serial info
        if serial_conn is not None:
            data = serial_conn.latest_data
            yaw = serial_conn.yaw_deg
            bias_dps = serial_conn.gyro_z_bias_dps
            bias_txt = f"{bias_dps:+.2f}dps" if bias_dps is not None else "--"
            put(f"Yaw: {yaw:+6.1f}deg  Bias: {bias_txt}")
            put(f"Bat: {data.bat_voltage / 100.0:.1f}V  "
                f"Vel: vx={data.vel_x} vy={data.vel_y} vw={data.vel_w}")

        # key hint at bottom
        cv2.rectangle(frame, (0, h - 20), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Press 'q' to quit", (5, h - 5),
                    font, 0.4, (150, 150, 150), 1)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_flag.set()
            break

        elapsed = time.monotonic() - t0
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cv2.destroyAllWindows()


async def run(
    args,
    *,
    stop_event: Optional[asyncio.Event] = None,
    expression_ui=None,
    output_refs: Optional[Dict[str, Any]] = None,
    skip_signal_handlers: bool = False,
):
    stop_event = stop_event if stop_event is not None else asyncio.Event()
    thread_stop = threading.Event()
    loop = asyncio.get_event_loop()

    def _signal_handler():
        stop_event.set()
        thread_stop.set()

    if not skip_signal_handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    need_camera = (
        (not args.no_chat and not args.no_video)
        or (not args.no_tracking)
        or args.display
    )

    use_display = bool(args.display)
    if use_display:
        ensure_local_display()
        if not opencv_highgui_available():
            print_display_failure_help("Debug window disabled.")
            use_display = False

    # -- camera --------------------------------------------------------------
    camera = None
    if need_camera:
        cam_id = args.camera
        if cam_id != "auto" and cam_id.isdigit():
            cam_id = int(cam_id)
        camera = SharedCamera(cam_id, args.camera_width, args.camera_height)
        camera.start()
        if not camera.is_opened:
            print("[main] Warning: camera unavailable")

    # -- serial --------------------------------------------------------------
    serial_conn = None
    if args.serial_port:
        from serial_comm import XProtocolSerial
        serial_conn = XProtocolSerial(args.serial_port, args.serial_baud)
        serial_conn.start()
        if args.ps2_debug:
            serial_conn.enable_debug_log(interval=args.ps2_debug_interval)

    # -- apriltag tracker ----------------------------------------------------
    tag_tracker = None
    tracking_ctl = None
    if not args.no_tracking and camera is not None:
        from apriltag_tracker import AprilTagTracker
        tag_tracker = AprilTagTracker(
            camera=camera,
            tag_size=args.tag_size,
            calib_file=args.calib_file,
            aruco_dict=args.aruco_dict,
            detector_preset=args.aruco_preset,
        )
        tag_tracker.start()

        if camera is not None:
            camera.set_overlay_callback(tag_tracker.draw_detections)

        if serial_conn is not None:
            from tracking_controller import TrackingController, TrackingParams
            params = TrackingParams(
                tracking_distance=args.tracking_distance,
                board_tag_size=args.tag_size,
                board_gap=args.marker_board_gap,
                board_lost_timeout=args.marker_board_lost_timeout,
                board_smoothing_alpha=args.marker_board_smoothing_alpha,
            )
            tracking_ctl = TrackingController(
                tracker=tag_tracker,
                serial=serial_conn,
                params=params,
                target_tag_id=args.target_tag_id,
            )
            tracking_ctl.start()
        else:
            print("[main] No serial port -- tracking is vision-only (no motor control)")

    if output_refs is not None:
        output_refs["tag_tracker"] = tag_tracker
        output_refs["tracking_ctl"] = tracking_ctl
        output_refs["serial_conn"] = serial_conn
        output_refs["camera"] = camera

    # -- PS2 button handling via serial callback --------------------------------
    if serial_conn is not None:
        from serial_comm import (
            PS2_BTN_SELECT, PS2_BTN_START, PS2_BTN_UP, PS2_BTN_DOWN,
            PS2_BTN_LEFT, PS2_BTN_RIGHT, PS2_BTN_JOY_L, PS2_BTN_JOY_R,
            PS2_BTN_L1, PS2_BTN_L2, PS2_BTN_R1, PS2_BTN_R2,
            PS2_BTN_A, PS2_BTN_B, PS2_BTN_X, PS2_BTN_Y,
        )

        _BTN1_NAMES = [
            (PS2_BTN_SELECT, "SELECT"), (PS2_BTN_JOY_L, "JOY_L"),
            (PS2_BTN_JOY_R, "JOY_R"),  (PS2_BTN_START,  "START"),
            (PS2_BTN_UP,    "UP"),      (PS2_BTN_RIGHT,  "RIGHT"),
            (PS2_BTN_DOWN,  "DOWN"),    (PS2_BTN_LEFT,   "LEFT"),
        ]
        _BTN2_NAMES = [
            (PS2_BTN_L2, "L2"), (PS2_BTN_R2, "R2"),
            (PS2_BTN_L1, "L1"), (PS2_BTN_R1, "R1"),
            (PS2_BTN_Y,  "Y"),  (PS2_BTN_B,  "B"),
            (PS2_BTN_A,  "A"),  (PS2_BTN_X,  "X"),
        ]

        def _on_ps2_change(old_ps2, new_ps2):
            pressed = []
            released = []
            for mask, name in _BTN1_NAMES:
                was = old_ps2.btn1_pressed(mask)
                now_ = new_ps2.btn1_pressed(mask)
                if now_ and not was:
                    pressed.append(name)
                elif was and not now_:
                    released.append(name)
            for mask, name in _BTN2_NAMES:
                was = old_ps2.btn2_pressed(mask)
                now_ = new_ps2.btn2_pressed(mask)
                if now_ and not was:
                    pressed.append(name)
                elif was and not now_:
                    released.append(name)

            if pressed:
                print(f"[ps2] Pressed:  {', '.join(pressed)}")
            if released:
                print(f"[ps2] Released: {', '.join(released)}")

            # SELECT rising edge → emergency stop
            if "SELECT" in pressed:
                if tracking_ctl is not None:
                    tracking_ctl.emergency_stop()
                serial_conn.send_velocity(0, 0, 0)
                print("[main] E-STOP triggered by PS2 SELECT")

            # START rising edge → resume from e-stop
            if "START" in pressed:
                if tracking_ctl is not None and tracking_ctl.is_estopped:
                    tracking_ctl.resume()
                    print("[main] Tracking resumed by PS2 START")

        serial_conn.on_ps2_change(_on_ps2_change)

    # -- debug display -------------------------------------------------------
    display_thread = None
    if use_display and camera is not None:
        display_thread = threading.Thread(
            target=_debug_display_loop,
            args=(camera, thread_stop, args.display_fps,
                  tag_tracker, tracking_ctl, serial_conn),
            daemon=True,
        )
        display_thread.start()
        print("[main] Debug display started (press 'q' in window to quit)")

    # -- console logger (when no display) ------------------------------------
    console_log_thread = None
    if not use_display and (tag_tracker is not None or serial_conn is not None):
        console_log_thread = threading.Thread(
            target=_console_log_loop,
            args=(thread_stop, tag_tracker, tracking_ctl, serial_conn),
            kwargs={"interval": 1.0},
            daemon=True,
        )
        console_log_thread.start()
        print("[main] Console logging started (ArUco + serial)")

    # -- teach (clip store + recorder + player + action registry) -----------
    # 即使 --no-teach-ui，也保留 motion player + dispatcher，这样 chat 链路上
    # 收到的情绪 tag 仍能触发动作回放；只是不再起 FastAPI server。
    enable_teach_ui = not args.no_teach_ui
    motion_recorder = None
    motion_player = None
    clip_store = None
    action_registry = None
    action_dispatcher = None
    if serial_conn is not None:
        from teach.clip_store import ClipStore
        from teach.motion_recorder import MotionRecorder
        from teach.motion_player import MotionPlayer
        from teach.action_registry import ActionRegistry
        from action_dispatcher import ActionDispatcher

        from pathlib import Path as _Path
        clip_store = ClipStore(_Path(args.motion_data_dir))
        clip_store.load()
        action_registry = ActionRegistry(_Path(args.teach_bindings_file))
        action_registry.load()
        motion_recorder = MotionRecorder(serial_conn)
        motion_recorder.start_thread()
        motion_player = MotionPlayer(
            serial=serial_conn,
            tracking_ctl=tracking_ctl,
            player_set_pending=None,  # 等 chat_client 起来后再 wire
        )
        motion_player.start_thread()
        action_dispatcher = ActionDispatcher(
            registry=action_registry,
            store=clip_store,
            motion_player=motion_player,
            expression_state=expression_ui,
        )

    # -- wake-word spotter (client-side KWS, gated on far-field audio) ------
    # Built BEFORE FarfieldAudioSource so we can wire it via constructor;
    # the spotter exposes set_on_wake() so we attach the rotation callback
    # AFTER the rotation controller is built (down below). Decoupled from
    # `--no-chat`: we want bench/CLI smoke tests of the wake pipeline to
    # work without a chat server (the audio drain loop further down
    # auto-pumps the DSP pipeline in that case).
    wake_spotter = None
    if not args.no_wake_orient and args.farfield:
        try:
            from wake_word_spotter import WakeWordSpotter
            spotter = WakeWordSpotter(
                model_dir=args.wake_kws_model_dir,
                keywords_file=args.wake_kws_keywords,
                keywords_threshold=args.wake_kws_threshold,
            )
            if spotter.start():
                wake_spotter = spotter
                print("[main] Client-side WakeWordSpotter loaded "
                      f"(threshold={args.wake_kws_threshold:.2f})")
            else:
                print("[main] WakeWordSpotter.start() returned False; "
                      "wake-orient disabled (model assets missing?).")
        except Exception as exc:
            print(f"[main] WakeWordSpotter init failed: {exc}; "
                  "wake-orient disabled.")

    # -- far-field audio source ---------------------------------------------
    farfield_source = None
    if args.farfield:
        try:
            from farfield_audio_source import FarfieldAudioSource
            farfield_source = FarfieldAudioSource(
                host=args.farfield_host,
                port=args.farfield_port,
                aec_backend=args.farfield_aec,
                aec_filter_length_ms=args.farfield_aec_filter_ms,
                aec_per_mic=args.farfield_aec_per_mic,
                dns_backend=args.farfield_dns,
                dns_use_passthrough=args.farfield_dns_passthrough,
                use_doa_for_mvdr=not args.farfield_no_doa,
                static_az_body_deg=args.farfield_static_az,
                mic_yaw_offset_deg=args.farfield_mic_yaw_offset_deg,
                auto_start_audio_server=not args.farfield_no_board_bootstrap,
                wake_spotter=wake_spotter,
            )
            farfield_source.open()
            print(
                f"[main] far-field audio source up: {args.farfield_host}:"
                f"{args.farfield_port} aec={args.farfield_aec} "
                f"dns={farfield_source.denoiser_name} "
                f"mic_yaw_offset={args.farfield_mic_yaw_offset_deg:+.1f}°"
            )
        except Exception as exc:
            print(f"[main] FAILED to open far-field audio source: {exc}")
            print("[main] Falling back to default PyAudio input.")
            farfield_source = None

    # -- wake-orient controller (IMU-closed rotation on wake) ---------------
    # Built AFTER far-field source so a failed audio-source init disables
    # the whole pipeline cleanly. Requires both spotter + serial; if serial
    # is absent we still attach a logging-only on_wake so wake events show
    # up in stdout (handy for bench runs without the chassis).
    wake_orient_ctl = None
    if wake_spotter is not None and farfield_source is not None:
        # Composite on_wake: forward to (a) the chassis rotator (if
        # serial present) AND (b) the chat_client to notify the server
        # so its WakeWord handler can start the wake session without
        # having to re-run KWS on the same audio. The chat_client is
        # built later in this function — we resolve it lazily from
        # ``output_refs`` (which the closure captures by reference).
        def _composite_on_wake(kw, az):
            cc = (output_refs or {}).get("chat_client")
            if cc is not None:
                try:
                    cc.notify_wake(kw, az)
                except Exception as e:
                    print(f"[main] notify_wake failed: {e}")
            inner = output_refs.get("_wake_inner") if output_refs else None
            if inner is not None:
                try:
                    inner(kw, az)
                except Exception as e:
                    print(f"[main] inner on_wake failed: {e}")

        if output_refs is None:
            output_refs = {}

        if serial_conn is not None:
            try:
                from wake_orient_controller import (
                    WakeOrientController, WakeOrientParams,
                )
                params = WakeOrientParams(
                    max_angular_speed=float(args.wake_orient_max_omega),
                    angle_deadzone_deg=float(args.wake_orient_deadzone_deg),
                    max_duration_s=float(args.wake_orient_timeout),
                    min_doa_deg=float(args.wake_orient_min_deg),
                )
                def _on_wake_complete(
                    kw,
                    req,
                    ach,
                    _serial=serial_conn,
                    _ff=farfield_source,
                ):
                    # Print the post-rotation IMU yaw too so the operator
                    # can sanity-check the closed-loop directly.
                    print(
                        f"[wake-orient] {kw!r}: requested {req:+0.1f}°, "
                        f"achieved {ach:+0.1f}°  "
                        f"(yaw_now={_serial.yaw_deg:+.1f}°)"
                    )
                    # Mic ring is rigid on the chassis: rotate MVDR / DOA by
                    # −Δyaw (CCW-positive IMU) so the beam stays on the
                    # world-fixed talker after wake-orient.
                    if _ff is not None and abs(float(ach)) >= 0.5:
                        try:
                            _ff.notify_chassis_imu_yaw_delta_ccw_deg(
                                float(ach))
                        except Exception as e:
                            print(
                                f"[main] farfield BF yaw notify failed: {e}"
                            )

                wake_orient_ctl = WakeOrientController(
                    serial=serial_conn,
                    params=params,
                    tracking_ctl=tracking_ctl,
                    on_complete=_on_wake_complete,
                )
                wake_orient_ctl.start()
                output_refs["_wake_inner"] = wake_orient_ctl.on_wake
                wake_spotter.set_on_wake(_composite_on_wake)
                print("[main] Wake-orient enabled "
                      "(client KWS + IMU-closed rotation + server notify)")
            except Exception as exc:
                print(f"[main] WakeOrientController init failed: {exc}; "
                      "logging wake events only.")
                output_refs["_wake_inner"] = (
                    lambda kw, az, _src=str(exc): print(
                        f"[wake] {kw!r} body_doa={az}° "
                        f"(rotation disabled: {_src})"
                    )
                )
                wake_spotter.set_on_wake(_composite_on_wake)
        else:
            output_refs["_wake_inner"] = (
                lambda kw, az: print(
                    f"[wake] {kw!r} body_doa={az}° "
                    f"(no serial port -> rotation skipped)"
                )
            )
            wake_spotter.set_on_wake(_composite_on_wake)
            print("[main] Client-side KWS active "
                  "(no serial port; rotation skipped)")

    if output_refs is not None:
        output_refs["wake_spotter"] = wake_spotter
        output_refs["wake_orient_ctl"] = wake_orient_ctl

    # -- far-field drain (when no chat consumer) ----------------------------
    # `FarfieldAudioSource.read()` is what advances the DSP pipeline (and
    # therefore the wake-spotter feed). Normally ChatClient drives this in
    # its capture loop. When `--no-chat` is set we have to pump it
    # ourselves so the wake-orient pipeline still runs end-to-end. One
    # 100ms chunk (1600 samples @ 16k) keeps latency low without burning
    # CPU on syscalls.
    farfield_drain_thread = None
    if farfield_source is not None and args.no_chat:
        def _farfield_drain():
            chunk = 1600  # 100 ms @ 16 kHz
            while not thread_stop.is_set() and not stop_event.is_set():
                try:
                    farfield_source.read(chunk)
                except Exception as exc:
                    print(f"[farfield-drain] read failed: {exc}")
                    time.sleep(0.1)
        farfield_drain_thread = threading.Thread(
            target=_farfield_drain, name="farfield_drain", daemon=True,
        )
        farfield_drain_thread.start()
        print("[main] Far-field drain thread started "
              "(no chat consumer -> pumping DSP+KWS locally)")

    # -- chat ----------------------------------------------------------------
    chat_client = None
    if not args.no_chat:
        from chat_client import ChatClient
        chat_client = ChatClient(
            on_action_tag=(
                (lambda tags, phase, _d=action_dispatcher: _d.dispatch(tags, phase=phase))
                if action_dispatcher is not None else None
            ),
            audio_source=farfield_source,
        )
        # 现在 chat_client 起来了，把 set_motion_pending 接到 motion_player 上。
        if motion_player is not None:
            motion_player._set_pending = chat_client.player.set_motion_pending

    if output_refs is not None:
        output_refs["chat_client"] = chat_client

    # -- teach FastAPI server -----------------------------------------------
    teach_server_task = None
    if enable_teach_ui and serial_conn is not None:
        from teach.server import create_app
        from pathlib import Path as _Path
        import uvicorn

        static_dir = _Path(__file__).resolve().parent / "teach_ui"
        teach_app = create_app(static_dir=static_dir if static_dir.is_dir() else None)
        st = teach_app.state
        st.camera = camera
        st.serial = serial_conn
        st.tracking_ctl = tracking_ctl
        st.motion_recorder = motion_recorder
        st.motion_player = motion_player
        st.clip_store = clip_store
        st.action_registry = action_registry
        st.dispatcher = action_dispatcher
        st.expression_state = expression_ui
        st.chat_client = chat_client
        st.calib_output_path = _Path(args.calib_output)
        st.calib_session = None
        st.calib_cancel = None
        st.calib_task = None
        st.loop = loop

        uv_config = uvicorn.Config(
            teach_app,
            host=args.teach_ui_bind,
            port=args.teach_ui_port,
            log_level="warning",
            lifespan="off",  # we own the lifecycle
        )
        uv_server = uvicorn.Server(uv_config)

        async def _serve_teach():
            try:
                await uv_server.serve()
            except Exception as e:
                print(f"[teach] uvicorn error: {e}", flush=True)

        async def _watch_stop_for_teach():
            await stop_event.wait()
            uv_server.should_exit = True

        teach_server_task = asyncio.create_task(_serve_teach())
        asyncio.create_task(_watch_stop_for_teach())
        print(f"[main] Teach UI on http://{args.teach_ui_bind}:{args.teach_ui_port}/")
    elif enable_teach_ui and serial_conn is None:
        print("[main] --no-teach-ui not set but no serial port — teach UI requires serial; skipped.")

    # -- main loop -----------------------------------------------------------
    tasks = []

    if chat_client is not None:
        tasks.append(asyncio.create_task(
            chat_client.run(
                server_url=args.server,
                camera=camera,
                stop_event=stop_event,
                enable_video=not args.no_video,
                expression_ui=expression_ui,
            )
        ))

    if teach_server_task is not None:
        tasks.append(teach_server_task)

    if not tasks:
        print("[main] Running in tracking-only mode. Press Ctrl+C to exit.")
        # poll thread_stop so display 'q' key also terminates
        while not stop_event.is_set() and not thread_stop.is_set():
            await asyncio.sleep(0.2)
    else:
        # also watch thread_stop in case display window closes
        async def _watch_thread_stop():
            while not thread_stop.is_set():
                await asyncio.sleep(0.2)
            stop_event.set()

        tasks.append(asyncio.create_task(_watch_thread_stop()))
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- cleanup -------------------------------------------------------------
    print("[main] Shutting down ...")
    thread_stop.set()
    if display_thread is not None:
        display_thread.join(timeout=3.0)
    if console_log_thread is not None:
        console_log_thread.join(timeout=2.0)
    if farfield_drain_thread is not None:
        farfield_drain_thread.join(timeout=2.0)
    if motion_player is not None:
        motion_player.stop_thread()
    if motion_recorder is not None:
        motion_recorder.stop_thread()
    # Stop wake-orient before tracking so a mid-rotation shutdown still
    # leaves the wheels at zero (the controller's stop() drains the goal
    # queue and writes one final zero-velocity frame).
    if wake_orient_ctl is not None:
        wake_orient_ctl.stop()
    if tracking_ctl is not None:
        tracking_ctl.stop()
    if tag_tracker is not None:
        tag_tracker.stop()
    if serial_conn is not None:
        for _ in range(10):
            serial_conn.send_velocity(0, 0, 0)
            time.sleep(0.02)
        serial_conn.stop()
    if camera is not None:
        camera.stop()
    if chat_client is not None:
        chat_client.close()
    if farfield_source is not None:
        try:
            farfield_source.close()
        except Exception:
            pass
    print("[main] Done.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.no_chat and args.no_tracking and not args.farfield:
        print("Error: --no-chat + --no-tracking with no other workload "
              "(--farfield) specified, nothing to do.")
        sys.exit(1)

    if args.expression and args.display:
        print("Error: --expression and --display both use a screen; pick one.")
        sys.exit(1)

    if args.expression:
        from expression_bridge import expression_assets_available, resolve_expression_svg_dir

        svg_dir = resolve_expression_svg_dir(args)
        ok, err = expression_assets_available(svg_dir)
        if not ok:
            print(f"[expression] {err}")
            print(
                "[expression] 素材目录无效，已回退为无表情窗口模式。"
                "请将 SVG 放入仓库 expression_player/images（或 EXPRESSION_SVG_DIR / --expression-svg-dir）；"
                "systemd 可去掉 ExecStart 中的 --expression。"
            )
            asyncio.run(run(args))
        else:
            from expression_app import run_expression_ui

            async def _expr_main(stop, ui_state, refs):
                await run(
                    args,
                    stop_event=stop,
                    expression_ui=ui_state,
                    output_refs=refs,
                    skip_signal_handlers=True,
                )

            run_expression_ui(args, _expr_main)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
