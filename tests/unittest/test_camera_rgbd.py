import os
import sys
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))
sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_V4L2=200,
        VideoWriter_fourcc=lambda *args: 0,
    ),
)


class _FakeColor:
    data = [[1, 2, 3]]


class _FakeRgbd:
    color = _FakeColor()

    def aligned_depth_mm(self):
        return [[1000.0, 0.0, 1015.0]]


class _FakeOrbbecGemini:
    last_open_kwargs = None

    def open(self, **kwargs):
        _FakeOrbbecGemini.last_open_kwargs = kwargs

    def read_rgbd_sdk(self, timeout_ms=600):
        return _FakeRgbd()

    def read_color_sdk(self, timeout_ms=200):
        return _FakeColor()

    def close(self):
        pass


sys.modules["orbbec_gemini"] = types.SimpleNamespace(
    OrbbecGemini=_FakeOrbbecGemini,
)

from camera import SharedCamera, _OrbbecBackend


class CameraRgbdTest(unittest.TestCase):
    def test_orbbec_rgbd_backend_opens_depth_color_with_hardware_d2c(self):
        backend = _OrbbecBackend(width=640, height=480, fps=30, rgbd=True)

        self.assertTrue(backend.isOpened())
        self.assertEqual(
            _FakeOrbbecGemini.last_open_kwargs["streams"],
            ("depth", "color"),
        )
        self.assertEqual(_FakeOrbbecGemini.last_open_kwargs["d2c"], "hw")

    def test_orbbec_rgbd_read_returns_color_and_caches_aligned_depth(self):
        backend = _OrbbecBackend(width=640, height=480, fps=30, rgbd=True)

        ok, color = backend.read()

        self.assertTrue(ok)
        self.assertEqual(color, [[1, 2, 3]])
        self.assertEqual(backend.get_aligned_depth_mm(), [[1000.0, 0.0, 1015.0]])

    def test_shared_camera_rgbd_snapshot_keeps_color_api_shape(self):
        cam = SharedCamera.__new__(SharedCamera)
        cam._lock = __import__("threading").Lock()
        cam._frame = [[1, 2, 3]]
        cam._aligned_depth_mm = [[1000.0]]
        cam._frame_id = 7
        cam._frame_time = 12.5

        color, depth, frame_id, frame_time = cam.get_rgbd_frame()

        self.assertEqual(color, [[1, 2, 3]])
        self.assertEqual(depth, [[1000.0]])
        self.assertEqual(frame_id, 7)
        self.assertEqual(frame_time, 12.5)


if __name__ == "__main__":
    unittest.main()
