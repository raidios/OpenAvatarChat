import os
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))


class _FakeCv2(types.SimpleNamespace):
    def VideoWriter_fourcc(self, *args):
        return 0


sys.modules.setdefault(
    "cv2",
    _FakeCv2(
        CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
    ),
)

from camera import V4L2Controls, _apply_v4l2_controls


class CameraControlsTest(unittest.TestCase):
    def test_apply_v4l2_controls_sets_auto_exposure_and_framerate(self):
        controls = V4L2Controls(
            auto_exposure="auto",
            disable_dynamic_framerate=True,
        )

        with mock.patch("camera.shutil.which", return_value="/usr/bin/v4l2-ctl"):
            with mock.patch("camera.subprocess.run") as run:
                _apply_v4l2_controls("/dev/video0", controls)

        args = [call.args[0] for call in run.call_args_list]
        self.assertIn(
            ["/usr/bin/v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=exposure_dynamic_framerate=0"],
            args,
        )
        self.assertIn(
            ["/usr/bin/v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=auto_exposure=3"],
            args,
        )

    def test_apply_v4l2_controls_ignores_non_v4l2_camera(self):
        controls = V4L2Controls(auto_exposure="manual", exposure_time_absolute=80)

        with mock.patch("camera.subprocess.run") as run:
            _apply_v4l2_controls("orbbec", controls)

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
