"""
Shared camera module.

Provides thread-safe access to a single camera device for multiple consumers
(e.g. video sender, AprilTag tracker).

Supports two backends:
  * V4L2 (cv2.VideoCapture) for any UVC USB camera at ``/dev/videoN``
  * Orbbec SDK (Gemini Pro depth camera) when no V4L2 capture node exists
    or when ``--camera orbbec`` is requested explicitly.

Auto-detect prefers V4L2; falls back to Orbbec when no /dev/videoN device
exposes the Video Capture capability (e.g. Pi 5 with only pispbe / rpivid
nodes plus an Orbbec accessed via the SDK).
"""

import atexit
import glob
import os
import sys
import threading
import time
from typing import Callable, Optional, Tuple, Union

import cv2
import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_ORBBEC_PY_DIR = os.path.join(
    _REPO_ROOT, "thirdparty", "OrbbecSDK_v1", "python"
)
if _ORBBEC_PY_DIR not in sys.path:
    sys.path.insert(0, _ORBBEC_PY_DIR)


# ----------------------------------------------------------------------
# Orbbec backend: cv2.VideoCapture-compatible wrapper around OrbbecGemini
# ----------------------------------------------------------------------

class _OrbbecBackend:
    """Tiny ``cv2.VideoCapture``-shaped shim around the Orbbec SDK.

    Exposes ``isOpened``, ``read``, ``release``, ``set`` so the rest of
    ``SharedCamera`` doesn't need to know whether the underlying source
    is V4L2 or the Orbbec SDK. Color-only; ``read()`` returns the BGR
    frame the SDK produced (matching ``cv2.VideoCapture.read``).
    """

    def __init__(self, width: int = 0, height: int = 0, fps: int = 30):
        self._width = int(width or 0)
        self._height = int(height or 0)
        self._fps = int(fps or 30)
        self._dev = None
        self._open()

    def _open(self) -> None:
        try:
            from orbbec_gemini import OrbbecGemini  # type: ignore
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] Orbbec SDK import failed: {exc}")
            self._dev = None
            return
        # Gemini Pro hardware D2C only pairs with 640x480@30 color, so
        # we lock the color stream there even if SharedCamera asked for
        # something different. The captured frame is downstream-resized
        # by AprilTag / chat consumers if needed (cheap on this Pi).
        cw = self._width if self._width else 640
        ch = self._height if self._height else 480
        cfps = self._fps if self._fps else 30
        dev = None
        try:
            dev = OrbbecGemini()
            dev.open(
                streams=("color",),
                color_width=cw, color_height=ch, color_fps=cfps,
                color_format="mjpg",
                d2c="off",  # color-only path
            )
            self._dev = dev
            print(f"[camera] Orbbec opened (color only, "
                  f"{cw}x{ch}@{cfps})")
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] Orbbec open failed: {exc}")
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass
            self._dev = None

    def isOpened(self) -> bool:        # noqa: N802 (cv2 API)
        return self._dev is not None

    def read(self):
        if self._dev is None:
            return False, None
        try:
            frame = self._dev.read_color_sdk(timeout_ms=200)
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] Orbbec read_color_sdk error: {exc}")
            return False, None
        if frame is None:
            return False, None
        bgr = getattr(frame, "bgr", None)
        if bgr is None:
            arr = getattr(frame, "data", None)
            if arr is None:
                return False, None
            bgr = arr
        return True, bgr

    def release(self) -> None:
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    def set(self, prop, value) -> bool:  # noqa: A003 (cv2 API)
        # Only width/height are settable up-front; ignore runtime sets.
        return True

    def grab(self) -> bool:  # noqa: N802 (cv2 API)
        """No-op: ``SharedCamera._capture_loop`` calls ``grab`` on shutdown."""
        return self._dev is not None


def _v4l2_has_capture_capability(dev: str) -> bool:
    """True iff ``dev`` exposes a Video Capture interface.

    On Pi 5 the kernel exposes a long list of /dev/videoN nodes for
    rpivid (HW codec, no capture) and the pispbe ISP backend (internal
    bookkeeping, also not a real capture endpoint). Without filtering,
    ``cv2.VideoCapture`` happily binds to /dev/video19 (rpivid) and
    then fails to grab frames forever. We use the V4L2 sysfs hint to
    pick only nodes that advertise ``Video Capture`` support.
    """
    idx = dev.replace("/dev/video", "")
    if not idx.isdigit():
        return False
    name_path = f"/sys/class/video4linux/video{idx}/name"
    if not os.path.exists(name_path):
        return False
    try:
        with open(name_path) as f:
            name = f.read().strip()
    except OSError:
        return False
    if not name:
        return False
    # rpivid / pispbe / dw100 / ... are decoder/ISP nodes, not cameras.
    bad_prefixes = ("rpivid", "pispbe", "dw100", "rkvenc", "rkvdec")
    if name.startswith(bad_prefixes):
        return False
    return True


def _open_video_capture(
    cam_arg: Union[int, str],
    width: int = 0,
    height: int = 0,
) -> Union[cv2.VideoCapture, _OrbbecBackend]:
    """Open camera by V4L2 path or Orbbec SDK.

    ``cam_arg`` may be:
      * int → V4L2 index
      * "/dev/videoN" → V4L2 path
      * "orbbec" → Orbbec SDK (width/height passed to SDK ``open``)
    """
    if isinstance(cam_arg, str) and cam_arg.lower().startswith("orbbec"):
        return _OrbbecBackend(width=int(width), height=int(height))
    v4l2 = getattr(cv2, "CAP_V4L2", None)
    if isinstance(cam_arg, str) and v4l2 is not None:
        return cv2.VideoCapture(cam_arg, v4l2)
    return cv2.VideoCapture(cam_arg)


def _find_first_camera() -> str:
    """Return the device path of the first real V4L2 video-capture device,
    or the special string ``"orbbec"`` if none is present and the Orbbec
    SDK is available.
    """
    for dev in sorted(glob.glob("/dev/video*")):
        if not _v4l2_has_capture_capability(dev):
            continue
        idx = dev.replace("/dev/video", "")
        try:
            with open(f"/sys/class/video4linux/video{idx}/name") as f:
                name = f.read().strip()
        except OSError:
            name = "?"
        print(f"[camera] Auto-detected {dev} ({name})")
        return dev
    # No real V4L2 capture device found. If the Orbbec SDK shared lib is
    # present, prefer it. Otherwise fall back to /dev/video0 so the
    # error message is self-describing.
    so = os.path.join(_ORBBEC_PY_DIR, "libobgemini.so")
    if os.path.exists(so):
        print("[camera] No V4L2 capture device found; "
              "falling back to Orbbec SDK")
        return "orbbec"
    print("[camera] No V4L2 capture device found and "
          "Orbbec SDK not available")
    return "/dev/video0"


class SharedCamera:
    """Thread-safe wrapper around cv2.VideoCapture.

    A background thread continuously grabs frames. Consumers call
    ``get_frame()`` to obtain the latest captured frame without blocking
    the camera pipeline.

    ``camera_id`` accepts an int index, a device path string like
    ``/dev/video1``, or ``"auto"`` to pick the first available camera.
    """

    def __init__(
        self,
        camera_id: Union[int, str] = "auto",
        width: int = 640,
        height: int = 480,
    ):
        if camera_id == "auto":
            camera_id = _find_first_camera()
        self._camera_id = camera_id
        self._width = width
        self._height = height

        self._cap: Optional[Union[cv2.VideoCapture, _OrbbecBackend]] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

        self._overlay_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None

        atexit.register(self._atexit_release)

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._width, self._height

    def set_overlay_callback(self, cb: Optional[Callable[[np.ndarray], np.ndarray]]):
        """Register a callback that annotates frames before they are served.

        The callback receives a copy of the raw frame and should return the
        annotated version.  Set to ``None`` to disable.
        """
        self._overlay_callback = cb

    def start(self, retries: int = 3, retry_delay: float = 1.0):
        if self._running:
            return

        cam_arg = self._camera_id

        for attempt in range(retries):
            self._cap = _open_video_capture(
                cam_arg, width=self._width, height=self._height,
            )
            if self._cap.isOpened():
                break
            self._cap.release()
            self._cap = None
            if attempt < retries - 1:
                print(f"[camera] Open attempt {attempt + 1} failed, "
                      f"retrying in {retry_delay}s ...")
                time.sleep(retry_delay)

        if self._cap is None or not self._cap.isOpened():
            print(f"[camera] Failed to open camera {self._camera_id} "
                  f"after {retries} attempts")
            if self._cap is not None:
                self._cap.release()
            self._cap = None
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._running = True
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[camera] Started (device={self._camera_id}, "
              f"{self._width}x{self._height})")

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        print("[camera] Stopped")

    def _atexit_release(self):
        """Ensure camera is released even on unclean exit."""
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def get_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """Return (frame, frame_id).  ``frame`` is None when unavailable."""
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    def get_frame_raw(self) -> Tuple[Optional[np.ndarray], int]:
        """Return the raw frame without overlay, useful for vision tasks."""
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    def _capture_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
        if self._cap is not None and self._cap.isOpened():
            self._cap.grab()
