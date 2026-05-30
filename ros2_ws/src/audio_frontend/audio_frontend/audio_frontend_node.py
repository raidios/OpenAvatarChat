"""
Stage-0 skeleton of the audio_frontend ROS2 node.

Reads 8ch from the M260C / R818 board via LiveStream (TCP 9999 over `adb forward`)
and republishes single-channel mono float32 16kHz to /audio/clean.

Stage-0 takes mic[0] verbatim. Stages 1-2 will plug ODAS / MVDR / AEC / DNS in
front of the publish call without changing topic shape or QoS.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, Float32MultiArray, String

REPO_ROOT = Path(__file__).resolve().parents[4]
LIVESTREAM_DIR = REPO_ROOT / "thirdparty" / "M2_SDK" / "live_stream" / "host"
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(LIVESTREAM_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from live_stream import LiveStream, MIC_CH, REF_CH, SAMPLE_RATE  # noqa: E402

try:
    import board_audio_mode as bam  # type: ignore  # noqa: E402
    _HAVE_BAM = True
except Exception:
    _HAVE_BAM = False

try:
    sys.path.insert(0, str(REPO_ROOT))
    from audio_frontend.dsp.pipeline import DspPipeline, PipelineConfig  # noqa: E402
    from audio_frontend.dsp.extrinsic import load_or_default              # noqa: E402
    _HAVE_DSP = True
except Exception as _dsp_exc:
    _HAVE_DSP = False
    _DSP_IMPORT_ERROR = _dsp_exc


def ensure_board_raw_mode(node: "AudioFrontendNode") -> None:
    """Best-effort: confirm board is in raw mode + adb forward up. Not fatal on failure."""
    if not _HAVE_BAM:
        return
    try:
        if not bam.adb_has_device():
            node.get_logger().error("adb device missing; LiveStream will fail to connect")
            return
        if bam.board_audio_server_pid() is None:
            node.get_logger().warning("audio_server not running on board; running start-raw")
            subprocess.run(
                ["adb", "shell", bam.BOARD_USE_RAW],
                capture_output=True, text=True, timeout=15,
            )
            time.sleep(1.5)
        if not bam.adb_forward_state(bam.DEFAULT_PORT):
            bam.adb_forward_apply(bam.DEFAULT_PORT)
    except Exception as exc:
        node.get_logger().warning(f"ensure_board_raw_mode soft-failure: {exc}")


class AudioFrontendNode(Node):
    """
    Parameters (declared via `--ros-args -p ...`):
      period_ms      (int)    audio loop quantum, default 32 (matches ALSA period)
      passthrough    (bool)   if True, publish mic[0] verbatim (stage 0). If False,
                              run ODAS+MVDR+AEC+DNS pipeline (stage 1+ wiring).
      ensure_board   (bool)   try to bring board to raw mode at startup
      reconnect_max_s (float) max backoff between reconnect attempts
    """

    def __init__(self):
        super().__init__("audio_frontend_node")
        self.declare_parameter("period_ms", 32)
        self.declare_parameter("passthrough", True)
        self.declare_parameter("ensure_board", True)
        self.declare_parameter("reconnect_max_s", 5.0)
        self.declare_parameter("audio_clean_topic", "/audio/clean")
        self.declare_parameter("vad_topic", "/audio/vad")
        self.declare_parameter("wakeword_topic", "/audio/wakeword")
        self.declare_parameter("doa_topic", "/audio/doa")
        self.declare_parameter("aec_backend", "auto")          # auto|speex|nlms
        self.declare_parameter("dns_backend", "auto")          # auto|cpu|hailo
        self.declare_parameter("array_radius_m", 0.033)
        self.declare_parameter("owner_3d_topic", "/perception/owner_3d")
        self.declare_parameter("mic_camera_extrinsic_npz",
                               str(REPO_ROOT / "tests" / "data" / "extrinsics"
                                   / "mic_camera_extrinsic.npz"))
        # Tracker config: how long after last owner_3d update we keep using it
        # before falling back to SRP-PHAT DOA.
        self.declare_parameter("owner_freshness_s", 1.5)
        # EMA coefficient for azimuth smoothing (0=instant, 1=frozen).
        self.declare_parameter("azimuth_ema_alpha", 0.65)

        self.period_ms = int(self.get_parameter("period_ms").value)
        self.passthrough = bool(self.get_parameter("passthrough").value)
        self.ensure_board = bool(self.get_parameter("ensure_board").value)
        self.reconnect_max_s = float(self.get_parameter("reconnect_max_s").value)
        self._pipeline: Optional["DspPipeline"] = None  # noqa: F821

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.pub_clean = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("audio_clean_topic").value),
            sensor_qos,
        )
        # Reserved for stage 1; declared early so that ROS2 graph includes them
        # from the very first run.
        self.pub_vad = self.create_publisher(
            Bool, str(self.get_parameter("vad_topic").value), 10,
        )
        self.pub_wakeword = self.create_publisher(
            String, str(self.get_parameter("wakeword_topic").value), 10,
        )
        # geometry_msgs is heavy; fall back to Float32MultiArray for DOA so the
        # node has zero hard ROS2 message dep beyond std_msgs.
        self.pub_doa = self.create_publisher(
            Float32MultiArray, str(self.get_parameter("doa_topic").value), 10,
        )

        if not self.passthrough:
            if not _HAVE_DSP:
                raise RuntimeError(
                    f"DSP pipeline import failed: {_DSP_IMPORT_ERROR!r}"
                )
            cfg = PipelineConfig(
                aec_backend=str(self.get_parameter("aec_backend").value),
                dns_backend=str(self.get_parameter("dns_backend").value),
                array_radius_m=float(self.get_parameter("array_radius_m").value),
            )
            self._pipeline = DspPipeline(cfg)
            self.get_logger().info(
                f"DSP pipeline online: aec={self._pipeline.aec.name}, "
                f"dns={self._pipeline.denoiser.name}"
            )

        # Owner 3D subscription (optional; falls back to SRP-PHAT DOA).
        self._extrinsic = load_or_default(
            str(self.get_parameter("mic_camera_extrinsic_npz").value)
            if _HAVE_DSP else None
        )
        self._owner_lock = threading.Lock()
        self._owner_az_deg: Optional[float] = None
        self._owner_az_stamp: float = 0.0
        self._owner_freshness_s = float(self.get_parameter("owner_freshness_s").value)
        self._azimuth_ema_alpha = float(self.get_parameter("azimuth_ema_alpha").value)
        self._smoothed_az: Optional[float] = None
        self.create_subscription(
            PointStamped,
            str(self.get_parameter("owner_3d_topic").value),
            self._on_owner_3d,
            10,
        )

        self._stop = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="audio-frontend-stream", daemon=True,
        )

    def start(self):
        if self.ensure_board:
            ensure_board_raw_mode(self)
        self._stream_thread.start()

    def stop(self):
        self._stop.set()

    def _stream_loop(self):
        backoff = 0.5
        while not self._stop.is_set() and rclpy.ok():
            try:
                self.get_logger().info(
                    f"opening LiveStream (period_ms={self.period_ms}, "
                    f"passthrough={self.passthrough})"
                )
                with LiveStream(ensure_forward=True) as ls:
                    backoff = 0.5
                    n_total = 0
                    t0 = time.monotonic()
                    for frames in ls.iter_frames(period_ms=self.period_ms):
                        if self._stop.is_set():
                            break
                        self._on_frames(frames)
                        n_total += frames.shape[0]
                        if (time.monotonic() - t0) > 30.0:
                            self.get_logger().info(
                                f"throughput: {n_total / SAMPLE_RATE:.1f}s of audio in "
                                f"{(time.monotonic() - t0):.1f}s wall time"
                            )
                            t0 = time.monotonic()
                            n_total = 0
            except Exception as e:
                self.get_logger().warning(
                    f"LiveStream loop error: {e!r} - reconnecting in {backoff:.1f}s"
                )
                self._stop.wait(backoff)
                backoff = min(self.reconnect_max_s, backoff * 1.7)
                if self.ensure_board and _HAVE_BAM:
                    try:
                        if not bam.board_audio_server_pid():
                            subprocess.run(["adb", "shell", bam.BOARD_USE_RAW],
                                           capture_output=True, text=True, timeout=15)
                            time.sleep(1.0)
                        if not bam.adb_forward_state(bam.DEFAULT_PORT):
                            bam.adb_forward_apply(bam.DEFAULT_PORT)
                    except Exception:
                        pass

    def _on_owner_3d(self, msg: PointStamped) -> None:
        if self._pipeline is None:
            return
        try:
            p = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
            az_deg, _, dist = self._extrinsic.to_az_el(p)
        except Exception:
            return
        if dist < 0.2 or dist > 6.0:
            return
        with self._owner_lock:
            self._owner_az_deg = az_deg
            self._owner_az_stamp = time.monotonic()

    def _resolve_steering_az(self) -> Optional[float]:
        """Pick steering azimuth. owner_3d wins when fresh; else SRP-PHAT DOA;
        smooth with EMA so the beam doesn't chatter."""
        if self._pipeline is None:
            return None
        with self._owner_lock:
            owner_az = self._owner_az_deg
            owner_age = time.monotonic() - self._owner_az_stamp
        if owner_az is not None and owner_age < self._owner_freshness_s:
            target = owner_az
            source = "owner3d"
        else:
            target = float(self._pipeline.latest_doa_deg)
            source = "doa"
        if self._smoothed_az is None:
            self._smoothed_az = target
        else:
            a = self._azimuth_ema_alpha
            err = (target - self._smoothed_az + 540.0) % 360.0 - 180.0
            self._smoothed_az = (self._smoothed_az + (1.0 - a) * err) % 360.0
        self._last_steering_source = source
        return self._smoothed_az

    def _on_frames(self, frames_8ch: np.ndarray):
        """frames_8ch: (T, 8) int16  with T = period_ms * 16 (e.g. 512 @ 32ms)."""
        if self.passthrough or self._pipeline is None:
            mono = frames_8ch[:, 0].astype(np.float32) / 32768.0
        else:
            steer = self._resolve_steering_az()
            if steer is not None:
                self._pipeline.set_target_azimuth(steer)
            mono = self._pipeline.process_block(frames_8ch)
            if mono.size > 0:
                doa_msg = Float32MultiArray()
                doa_msg.data = [
                    float(self._pipeline.latest_doa_deg),
                    float(self._smoothed_az if self._smoothed_az is not None else 0.0),
                    1.0 if getattr(self, "_last_steering_source", "doa") == "owner3d" else 0.0,
                ]
                self.pub_doa.publish(doa_msg)

        # Coarse RMS-based VAD (heuristic, only published so other ROS2 nodes
        # can react; the real VAD lives in the OpenAvatarChat handler chain
        # downstream of /audio/clean).
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        speaking = rms > 0.01
        if not hasattr(self, "_last_vad") or self._last_vad != speaking:
            vad_msg = Bool()
            vad_msg.data = speaking
            self.pub_vad.publish(vad_msg)
            self._last_vad = speaking

        msg = Float32MultiArray()
        msg.data = mono.tolist()
        self.pub_clean.publish(msg)


def main(argv=None):
    rclpy.init(args=argv)
    node = AudioFrontendNode()
    node.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
