"""
X-Protocol serial communication with the robot MCU.

Protocol (variable-length frame):
    AA 55 | LEN | CMD | DATA... | CHECKSUM
    header  len   cmd   payload   sum & 0xFF

- LEN   = len(DATA) + 5  (header=2 + len=1 + cmd=1 + checksum=1)
- DATA  is big-endian
- CHECKSUM = (sum of all preceding bytes) & 0xFF

Frame codes (defined in firmware ax_uart4.c):
    0x10  MCU->Pi  comprehensive data (20 B)
    0x11  MCU->Pi  PS2 joystick state (7 B)
    0x12  MCU->Pi  active control source (1 B) - see CTRL_SRC_*
    0x50  Pi->MCU  velocity command   (6 B)
    0x51  Pi->MCU  IMU calibrate      (1 B)
    0x5F  Pi->MCU  servo offset       (2 B)
"""

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import serial


FRAME_HEADER = bytes([0xAA, 0x55])

CMD_MCU_DATA = 0x10
CMD_PS2_DATA = 0x11
CMD_CTRL_STATE = 0x12
CMD_SET_VELOCITY = 0x50
CMD_IMU_CALIBRATE = 0x51
CMD_SERVO_OFFSET = 0x5F

# Control-source IDs reported by the MCU (CMD 0x12 payload).
# Mirrors AX_CTRL_SRC_* in firmware/.../Driver/ax_uart4.h.
CTRL_SRC_IDLE = 0   # nothing is driving the wheels
CTRL_SRC_PS2  = 1   # PS2 pad has overridden everything
CTRL_SRC_PI   = 2   # this host's cmd_vel via 0x50
CTRL_SRC_APP  = 3   # Bluetooth APP (USART2)
CTRL_SRC_FN1  = 4   # line-following / autonomous fallback

CTRL_SRC_NAMES = {
    CTRL_SRC_IDLE: "IDLE",
    CTRL_SRC_PS2:  "PS2",
    CTRL_SRC_PI:   "PI",
    CTRL_SRC_APP:  "APP",
    CTRL_SRC_FN1:  "FN1",
}

# Bitmask flags packed into the 0x12 frame's second byte.
# Mirrors AX_CTRL_FLAG_* in firmware/.../Driver/ax_uart4.h.
CTRL_FLAG_PS2_ACTIVE   = 0x01   # PS2_IsActive() returned 1 this tick
CTRL_FLAG_PS2_OVERRIDE = 0x02   # inside the 800ms grace window
CTRL_FLAG_PS2_WARMUP   = 0x04   # warm-up window elapsed (PS2 may grab)
CTRL_FLAG_PI_LIVE      = 0x08   # Pi cmd_vel within 300ms freshness
CTRL_FLAG_PS2_GLITCH   = 0x10   # all four sticks pinned to 0x00 / 0xFF
CTRL_FLAG_PS2_HOLD0    = 0x20   # override active but sticks released -> brake

# MPU6050 gyroscope scale: +-500 deg/s mapped to int16
_GYRO_SCALE = 500.0 / 32768.0

# --- Online gyro_z bias estimation -----------------------------------------
# The MCU's boot-time calibration (10 samples × ~50 ms in main.c::Start_Task)
# leaves a residual bias on cheap MPU6050 modules that easily reaches tens of
# dps. We estimate the leftover bias on the Pi side using the encoder-measured
# body velocity as a static-state detector; while the wheels are not turning
# the body cannot rotate (rigid differential drive, no slip), so any non-zero
# gyro_z is bias and gets folded into a slow EMA.

# Static-state thresholds. vel_x/y are mm/s, vel_w is mrad/s (firmware 0x10
# payload). Encoder quantisation puts the noise floor below these values.
_STATIC_VEL_MM_S = 5
_STATIC_W_MRAD_S = 5

# Wait this many consecutive static frames before touching the bias estimate.
# At 50 Hz, 50 frames = 1 s. Filters out brief pauses between motion segments.
_STATIC_FRAMES_REQUIRED = 50

# EMA learning rate for the bias. At 50 Hz, alpha=0.005 → time constant ≈ 4 s;
# bias converges to within 1% of the true mean after ~20 s of stillness.
_BIAS_EMA_ALPHA = 0.005

# Reject samples whose residual (after subtracting current bias) exceeds this
# magnitude in dps. Protects the estimator if someone physically rotates the
# robot while the wheels happen to read zero (caster slip, manual handling).
_BIAS_SANITY_DPS = 20.0

# PS2 button masks — btn1
PS2_BTN_SELECT  = 0x01
PS2_BTN_JOY_L   = 0x02
PS2_BTN_JOY_R   = 0x04
PS2_BTN_START   = 0x08
PS2_BTN_UP      = 0x10
PS2_BTN_RIGHT   = 0x20
PS2_BTN_DOWN    = 0x40
PS2_BTN_LEFT    = 0x80

# PS2 button masks — btn2
PS2_BTN_L2 = 0x01
PS2_BTN_R2 = 0x02
PS2_BTN_L1 = 0x04
PS2_BTN_R1 = 0x08
PS2_BTN_Y  = 0x10
PS2_BTN_B  = 0x20
PS2_BTN_A  = 0x40
PS2_BTN_X  = 0x80


@dataclass
class MCUData:
    """Parsed telemetry from the MCU (CMD 0x10)."""
    acc_x: int = 0
    acc_y: int = 0
    acc_z: int = 0
    gyro_x: int = 0
    gyro_y: int = 0
    gyro_z: int = 0
    vel_x: int = 0    # m/s * 1000
    vel_y: int = 0
    vel_w: int = 0    # rad/s * 1000
    bat_voltage: int = 0  # voltage * 100
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class PS2Data:
    """Parsed PS2 joystick state from MCU (CMD 0x11)."""
    mode: int = 0
    btn1: int = 0
    btn2: int = 0
    rjoy_lr: int = 0x80   # 0x00=left, 0xFF=right, 0x80=center
    rjoy_ud: int = 0x80   # 0x00=up,   0xFF=down,  0x80=center
    ljoy_lr: int = 0x80
    ljoy_ud: int = 0x80
    timestamp: float = field(default_factory=time.monotonic)

    def btn1_pressed(self, mask: int) -> bool:
        return bool(self.btn1 & mask)

    def btn2_pressed(self, mask: int) -> bool:
        return bool(self.btn2 & mask)


@dataclass
class CtrlState:
    """Arbitration / control-source state from MCU (CMD 0x12).

    Populated from either the legacy 1-byte payload (only `source` is valid) or
    the extended 8-byte payload that includes diagnostic flags and the live
    target velocity. Use `flags_str()` for a compact human-readable rendering.
    """
    source: int = CTRL_SRC_IDLE
    flags: int = 0
    tg_vx: int = 0          # mm/s, signed
    tg_vy: int = 0
    tg_vw: int = 0          # mrad/s, signed
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def source_name(self) -> str:
        return CTRL_SRC_NAMES.get(self.source, f"0x{self.source:02X}")

    @property
    def ps2_active(self) -> bool:
        return bool(self.flags & CTRL_FLAG_PS2_ACTIVE)

    @property
    def ps2_override(self) -> bool:
        return bool(self.flags & CTRL_FLAG_PS2_OVERRIDE)

    @property
    def ps2_warmup_done(self) -> bool:
        return bool(self.flags & CTRL_FLAG_PS2_WARMUP)

    @property
    def pi_live(self) -> bool:
        return bool(self.flags & CTRL_FLAG_PI_LIVE)

    @property
    def ps2_glitch(self) -> bool:
        return bool(self.flags & CTRL_FLAG_PS2_GLITCH)

    @property
    def ps2_hold0(self) -> bool:
        return bool(self.flags & CTRL_FLAG_PS2_HOLD0)

    def flags_str(self) -> str:
        names = []
        if self.ps2_active:      names.append("ACT")
        if self.ps2_override:    names.append("OVR")
        if self.ps2_hold0:       names.append("HOLD0")
        if self.ps2_glitch:      names.append("GLITCH")
        if self.pi_live:         names.append("PI_LIVE")
        if not self.ps2_warmup_done: names.append("WARMUP")
        return "|".join(names) if names else "-"


# Callback type: called when PS2 button edges are detected
PS2Callback = Callable[["PS2Data", "PS2Data"], None]

# Callback type: called when the MCU reports a new active control source.
# Signature: callback(old_source: int, new_source: int)
CtrlSrcCallback = Callable[[int, int], None]


class XProtocolSerial:
    """Manages serial communication with the robot MCU using X-Protocol."""

    def __init__(self, port: str, baudrate: int = 115200):
        self._port = port
        self._baudrate = baudrate
        self._ser: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_data = MCUData()
        self._latest_ps2 = PS2Data()
        self._yaw_deg: float = 0.0
        self._last_gyro_time: Optional[float] = None

        # Online gyro_z bias estimation (raw int16 LSB units; None until the
        # first stable static window has been observed).
        self._gyro_z_bias_lsb: Optional[float] = None
        self._static_frames: int = 0

        self._latest_ctrl: CtrlState = CtrlState()

        self._ps2_callbacks: List[PS2Callback] = []
        self._ctrl_src_callbacks: List[CtrlSrcCallback] = []

        self._debug_log_thread: Optional[threading.Thread] = None
        self._debug_log_interval: float = 0.0

    # -- public properties ---------------------------------------------------

    @property
    def latest_data(self) -> MCUData:
        with self._lock:
            return self._latest_data

    @property
    def latest_ps2(self) -> PS2Data:
        with self._lock:
            return self._latest_ps2

    @property
    def yaw_deg(self) -> float:
        """Integrated yaw, wrapped to [-180, 180)°.

        Internally we keep an unwrapped accumulator for callers that need
        absolute heading drift (`yaw_deg_unwrapped`); display paths almost
        always want the wrapped value to avoid 6-digit nonsense after a few
        hours of static drift.
        """
        with self._lock:
            return ((self._yaw_deg + 180.0) % 360.0) - 180.0

    @property
    def yaw_deg_unwrapped(self) -> float:
        """Raw integrated yaw, never wrapped. Useful for trajectory odom."""
        with self._lock:
            return self._yaw_deg

    @property
    def gyro_z_bias_dps(self) -> Optional[float]:
        """Estimated gyro_z zero-bias in dps, or None if not yet bootstrapped.

        Becomes non-None once the car has been static for at least
        `_STATIC_FRAMES_REQUIRED` consecutive 0x10 frames (~1 s at 50 Hz).
        """
        with self._lock:
            if self._gyro_z_bias_lsb is None:
                return None
            return self._gyro_z_bias_lsb * _GYRO_SCALE

    def reset_yaw(self):
        with self._lock:
            self._yaw_deg = 0.0
            self._last_gyro_time = None

    def reset_gyro_bias(self):
        """Forget the learned gyro_z bias and start over.

        Call after physically reseating the IMU, swapping boards, or whenever
        you suspect the current bias estimate has been corrupted (e.g. someone
        manually rotated the chassis while the wheels were locked).
        """
        with self._lock:
            self._gyro_z_bias_lsb = None
            self._static_frames = 0

    def on_ps2_change(self, callback: PS2Callback):
        """Register a callback invoked when PS2 button state changes.

        callback(old_ps2, new_ps2) is called from the receive thread.
        """
        self._ps2_callbacks.append(callback)

    @property
    def control_source(self) -> int:
        """Most recent active input source reported by the MCU (CMD 0x12).

        Returns one of CTRL_SRC_*. Use this to detect when the PS2 pad has
        taken over and stop sending cmd_vel until the user releases it.
        """
        with self._lock:
            return self._latest_ctrl.source

    @property
    def latest_ctrl_state(self) -> CtrlState:
        """Most recent full arbitration state (source + flags + target vel)."""
        with self._lock:
            return self._latest_ctrl

    @property
    def is_ps2_overriding(self) -> bool:
        """True iff the wheels are currently being driven by the PS2 pad."""
        with self._lock:
            return self._latest_ctrl.source == CTRL_SRC_PS2

    def on_ctrl_source_change(self, callback: CtrlSrcCallback):
        """Register a callback invoked when the active control source changes.

        callback(old_source, new_source) is called from the receive thread on
        every transition (e.g. PI -> PS2 when the user grabs the pad,
        PS2 -> PI when the pad has been idle long enough).
        """
        self._ctrl_src_callbacks.append(callback)

    # -- debug log -----------------------------------------------------------

    def enable_debug_log(self, interval: float = 1.0):
        """Start a background thread that prints PS2 + arbitration state at
        a fixed cadence. Useful when no MCU debug serial is available; pairs
        with the 8-byte 0x12 frame to expose the full arbitration state to
        user-space log output.

        Idempotent: a second call updates the interval but keeps one thread.
        """
        with self._lock:
            self._debug_log_interval = max(0.05, float(interval))
            if self._debug_log_thread is not None and self._debug_log_thread.is_alive():
                return
            self._debug_log_thread = threading.Thread(
                target=self._debug_log_loop, daemon=True, name="serial-ps2-debug"
            )
            self._debug_log_thread.start()

    def _debug_log_loop(self):
        while self._running:
            with self._lock:
                ps2 = self._latest_ps2
                ctrl = self._latest_ctrl
                data = self._latest_data
                interval = self._debug_log_interval

            # bat_voltage is V*100 (3S Li-ion: <10.65 V = 40%, <10.12 V = 20%,
            # <9.84 V = 10% -> firmware suspends Robot_Task after ~5 s).
            vbat = data.bat_voltage / 100.0 if data.bat_voltage else 0.0
            vbat_warn = ""
            if data.bat_voltage:
                if data.bat_voltage < 984:
                    vbat_warn = " !!CRITICAL!!"
                elif data.bat_voltage < 1012:
                    vbat_warn = " !LOW!"
                elif data.bat_voltage < 1065:
                    vbat_warn = " (40%)"

            # Stale telemetry detection: if 0x10 frames stopped arriving (e.g.
            # Robot_Task got suspended by the low-voltage cutoff), bat_voltage
            # will keep its last value but data.timestamp won't refresh.
            stale = ""
            if data.timestamp:
                age = time.monotonic() - data.timestamp
                if age > 1.0:
                    stale = f" [STALE {age:.1f}s]"

            print(
                f"[ps2-dbg] src={ctrl.source_name:<4} "
                f"flags={ctrl.flags_str():<24} "
                f"tg=({ctrl.tg_vx:+5d},{ctrl.tg_vy:+5d},{ctrl.tg_vw:+6d}) "
                f"vbat={vbat:5.2f}V{vbat_warn} "
                f"mode=0x{ps2.mode:02X} "
                f"btn1=0x{ps2.btn1:02X} btn2=0x{ps2.btn2:02X} "
                f"L=({ps2.ljoy_lr:3d},{ps2.ljoy_ud:3d}) "
                f"R=({ps2.rjoy_lr:3d},{ps2.rjoy_ud:3d})"
                f"{stale}"
            )
            time.sleep(interval)

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        if self._running:
            return
        try:
            self._ser = serial.Serial(
                self._port, self._baudrate, timeout=0.05
            )
        except serial.SerialException as e:
            print(f"[serial] Failed to open {self._port}: {e}")
            return
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[serial] Started ({self._port} @ {self._baudrate})")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        print("[serial] Stopped")

    # -- send commands -------------------------------------------------------

    def send_velocity(self, vx_mm: int, vy_mm: int, vw_mm: int):
        """Send velocity command.  Units: m/s * 1000 (int16)."""
        data = struct.pack(">hhh", int(vx_mm), int(vy_mm), int(vw_mm))
        self._send_frame(CMD_SET_VELOCITY, data)

    def send_imu_calibrate(self):
        self._send_frame(CMD_IMU_CALIBRATE, bytes([0x01]))

    def send_servo_offset(self, offset: int):
        data = struct.pack(">h", int(offset))
        self._send_frame(CMD_SERVO_OFFSET, data)

    # -- internal ------------------------------------------------------------

    def _send_frame(self, cmd: int, data: bytes):
        frame_len = len(data) + 5
        frame = bytearray(FRAME_HEADER)
        frame.append(frame_len)
        frame.append(cmd)
        frame.extend(data)
        checksum = sum(frame) & 0xFF
        frame.append(checksum)
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                try:
                    self._ser.write(frame)
                except serial.SerialException as e:
                    print(f"[serial] Write error: {e}")

    def _recv_loop(self):
        buf = bytearray()
        rx_total = 0
        frame_ok = 0
        frame_bad = 0
        last_report = time.monotonic()
        REPORT_INTERVAL = 10.0

        while self._running:
            if self._ser is None or not self._ser.is_open:
                time.sleep(0.1)
                continue
            try:
                chunk = self._ser.read(64)
            except serial.SerialException:
                time.sleep(0.1)
                continue
            if not chunk:
                now = time.monotonic()
                if now - last_report >= REPORT_INTERVAL:
                    print(f"[serial] RX stats: {rx_total} bytes, "
                          f"{frame_ok} frames OK, {frame_bad} bad")
                    if rx_total == 0:
                        print("[serial] WARNING: no data received from MCU")
                    last_report = now
                continue

            rx_total += len(chunk)
            if rx_total <= len(chunk):
                print(f"[serial] First data received: "
                      f"{chunk[:min(20, len(chunk))].hex(' ')}")

            buf.extend(chunk)
            ok, bad = self._parse_buffer(buf)
            frame_ok += ok
            frame_bad += bad

            now = time.monotonic()
            if now - last_report >= REPORT_INTERVAL:
                print(f"[serial] RX stats: {rx_total} bytes, "
                      f"{frame_ok} frames OK, {frame_bad} bad")
                last_report = now

    def _parse_buffer(self, buf: bytearray) -> tuple:
        ok = 0
        bad = 0
        while len(buf) >= 5:
            idx = buf.find(FRAME_HEADER)
            if idx < 0:
                buf.clear()
                return ok, bad
            if idx > 0:
                del buf[:idx]
            if len(buf) < 3:
                return ok, bad
            frame_len = buf[2]
            if frame_len < 5 or frame_len > 60:
                bad += 1
                del buf[:2]
                continue
            if len(buf) < frame_len:
                return ok, bad
            expected_checksum = sum(buf[: frame_len - 1]) & 0xFF
            if buf[frame_len - 1] != expected_checksum:
                bad += 1
                del buf[:2]
                continue
            cmd = buf[3]
            data = bytes(buf[4: frame_len - 1])
            del buf[:frame_len]
            ok += 1
            self._handle_frame(cmd, data)
        return ok, bad

    def _handle_frame(self, cmd: int, data: bytes):
        if cmd == CMD_MCU_DATA and len(data) == 20:
            vals = struct.unpack(">hhhhhhhhhH", data)
            now = time.monotonic()
            with self._lock:
                d = MCUData(
                    acc_x=vals[0], acc_y=vals[1], acc_z=vals[2],
                    gyro_x=vals[3], gyro_y=vals[4], gyro_z=vals[5],
                    vel_x=vals[6], vel_y=vals[7], vel_w=vals[8],
                    bat_voltage=vals[9],
                    timestamp=now,
                )
                self._latest_data = d

                # Static-state detection from encoder-measured body velocity.
                # vel_x/y are mm/s, vel_w is mrad/s (per firmware 0x10 payload).
                is_static = (
                    abs(d.vel_x) <= _STATIC_VEL_MM_S
                    and abs(d.vel_y) <= _STATIC_VEL_MM_S
                    and abs(d.vel_w) <= _STATIC_W_MRAD_S
                )
                if is_static:
                    self._static_frames += 1
                    if self._static_frames >= _STATIC_FRAMES_REQUIRED:
                        if self._gyro_z_bias_lsb is None:
                            # Bootstrap from the first stable sample, then EMA
                            # takes over on subsequent frames.
                            self._gyro_z_bias_lsb = float(d.gyro_z)
                        else:
                            residual_dps = (
                                (d.gyro_z - self._gyro_z_bias_lsb) * _GYRO_SCALE
                            )
                            if abs(residual_dps) <= _BIAS_SANITY_DPS:
                                self._gyro_z_bias_lsb = (
                                    (1.0 - _BIAS_EMA_ALPHA) * self._gyro_z_bias_lsb
                                    + _BIAS_EMA_ALPHA * d.gyro_z
                                )
                else:
                    self._static_frames = 0

                bias_lsb = (
                    self._gyro_z_bias_lsb if self._gyro_z_bias_lsb is not None else 0.0
                )
                gyro_z_dps = (d.gyro_z - bias_lsb) * _GYRO_SCALE
                if self._last_gyro_time is not None:
                    dt = now - self._last_gyro_time
                    self._yaw_deg += gyro_z_dps * dt
                self._last_gyro_time = now

        elif cmd == CMD_PS2_DATA and len(data) == 7:
            now = time.monotonic()
            new_ps2 = PS2Data(
                mode=data[0],
                btn1=data[1],
                btn2=data[2],
                rjoy_lr=data[3],
                rjoy_ud=data[4],
                ljoy_lr=data[5],
                ljoy_ud=data[6],
                timestamp=now,
            )
            with self._lock:
                old_ps2 = self._latest_ps2
                self._latest_ps2 = new_ps2

            btns_changed = (old_ps2.btn1 != new_ps2.btn1
                            or old_ps2.btn2 != new_ps2.btn2)
            if btns_changed:
                for cb in self._ps2_callbacks:
                    try:
                        cb(old_ps2, new_ps2)
                    except Exception as e:
                        print(f"[serial] PS2 callback error: {e}")

        elif cmd == CMD_CTRL_STATE and len(data) >= 1:
            new_src = data[0]
            if len(data) >= 8:
                flags, tg_vx, tg_vy, tg_vw = struct.unpack(">Bhhh", data[1:8])
            else:
                flags, tg_vx, tg_vy, tg_vw = 0, 0, 0, 0
            now = time.monotonic()
            new_state = CtrlState(
                source=new_src,
                flags=flags,
                tg_vx=tg_vx,
                tg_vy=tg_vy,
                tg_vw=tg_vw,
                timestamp=now,
            )
            with self._lock:
                old_src = self._latest_ctrl.source
                self._latest_ctrl = new_state

            if old_src != new_src:
                old_name = CTRL_SRC_NAMES.get(old_src, f"0x{old_src:02X}")
                new_name = CTRL_SRC_NAMES.get(new_src, f"0x{new_src:02X}")
                print(f"[serial] Control source: {old_name} -> {new_name}")
                for cb in self._ctrl_src_callbacks:
                    try:
                        cb(old_src, new_src)
                    except Exception as e:
                        print(f"[serial] ctrl-src callback error: {e}")
