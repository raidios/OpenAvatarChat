import os
import sys
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLIENT_DIR = os.path.join(REPO_ROOT, "client")
sys.path.insert(0, CLIENT_DIR)


sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault(
    "camera",
    types.SimpleNamespace(
        SharedCamera=object,
        V4L2Controls=object,
    ),
)
sys.modules.setdefault(
    "opencv_gui",
    types.SimpleNamespace(
        ensure_local_display=lambda: None,
        opencv_highgui_available=lambda: False,
        print_display_failure_help=lambda *_args, **_kwargs: None,
    ),
)

import main  # noqa: E402


class ClientDefaultArgsTest(unittest.TestCase):
    def test_car_runtime_defaults_use_rgb_marker_and_no_person_follow(self):
        args = main.build_parser().parse_args([])

        self.assertEqual(args.camera_mode, "color")
        self.assertTrue(args.marker_tracking)
        self.assertFalse(args.person_follow)
        self.assertTrue(args.motion_return)
        self.assertAlmostEqual(args.motion_return_max_linear_speed, 0.12)
        self.assertAlmostEqual(args.motion_return_max_angular_speed, 0.35)
        self.assertAlmostEqual(args.motion_return_position_tolerance, 0.03)
        self.assertAlmostEqual(args.motion_return_yaw_tolerance_deg, 3.0)
        self.assertAlmostEqual(args.motion_return_timeout, 10.0)
        self.assertAlmostEqual(args.motion_return_stale_timeout, 0.3)

    def test_can_explicitly_enable_person_follow_rgbd_mode(self):
        args = main.build_parser().parse_args(
            ["--camera-mode", "rgbd-sdk", "--camera", "orbbec", "--person-follow"]
        )

        self.assertEqual(args.camera_mode, "rgbd-sdk")
        self.assertEqual(args.camera, "orbbec")
        self.assertTrue(args.person_follow)

    def test_can_disable_marker_tracking_for_bench_runs(self):
        args = main.build_parser().parse_args(["--no-marker-tracking"])

        self.assertFalse(args.marker_tracking)

    def test_can_disable_motion_return_for_bench_runs(self):
        args = main.build_parser().parse_args(["--no-motion-return"])

        self.assertFalse(args.motion_return)


if __name__ == "__main__":
    unittest.main()
