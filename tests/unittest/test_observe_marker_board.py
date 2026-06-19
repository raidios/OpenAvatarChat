import os
import sys
import unittest
from types import SimpleNamespace


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from observe_marker_board import format_observation


class ObserveMarkerBoardTest(unittest.TestCase):
    def test_formats_visible_board_target(self):
        target = SimpleNamespace(distance=0.623, angle_h=-0.01745, held=False, predicted=False)
        line = format_observation([SimpleNamespace(tag_id=2), SimpleNamespace(tag_id=0)], target)

        self.assertEqual(line, "tags=2 ids=0,2 board dist=0.62m angle=-1.0deg held=False predicted=False")

    def test_formats_held_board_target(self):
        target = SimpleNamespace(distance=0.7, angle_h=0.05236, held=True, predicted=True)
        line = format_observation([], target)

        self.assertEqual(line, "tags=0 ids=- board dist=0.70m angle=3.0deg held=True predicted=True")

    def test_formats_lost_board_target(self):
        line = format_observation([], None)

        self.assertEqual(line, "tags=0 ids=- board LOST")


if __name__ == "__main__":
    unittest.main()
