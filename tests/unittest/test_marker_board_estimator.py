import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client"))

from marker_board import MarkerBoardEstimator, MarkerBoardLayout


def _det(tag_id: int, x: float, z: float):
    tvec = np.array([x, 0.0, z], dtype=np.float64)
    return SimpleNamespace(
        tag_id=tag_id,
        center=(0.0, 0.0),
        corners=np.zeros((4, 2), dtype=np.float64),
        distance=float(np.linalg.norm(tvec)),
        angle_h=float(math.atan2(x, z)),
        tvec=tvec,
    )


class MarkerBoardEstimatorTest(unittest.TestCase):
    def test_fuses_multiple_visible_tags_into_board_center(self):
        layout = MarkerBoardLayout(tag_size_m=0.045, gap_m=0.007)
        estimator = MarkerBoardEstimator(layout=layout, smoothing_alpha=1.0)

        # Board center is 1m straight ahead. Visible detections come from
        # top-left, center, and bottom-right markers of the 3x3 board.
        detections = [
            _det(0, -0.052, 1.0),
            _det(4, 0.0, 1.0),
            _det(8, 0.052, 1.0),
        ]

        board = estimator.estimate(detections)

        self.assertIsNotNone(board)
        self.assertEqual(board.tag_id, -1)
        self.assertAlmostEqual(board.angle_h, 0.0, places=6)
        self.assertAlmostEqual(board.distance, 1.0, places=6)

    def test_ignores_missing_and_unknown_tags(self):
        layout = MarkerBoardLayout(tag_size_m=0.045, gap_m=0.007)
        estimator = MarkerBoardEstimator(
            layout=layout,
            smoothing_alpha=1.0,
            min_visible_tags=1,
        )

        board = estimator.estimate([
            _det(99, 0.0, 0.5),
            _det(5, 0.052, 0.8),
        ])

        self.assertIsNotNone(board)
        self.assertAlmostEqual(board.angle_h, 0.0, places=6)
        self.assertAlmostEqual(board.distance, 0.8, places=6)

    def test_reuses_recent_board_pose_during_short_dropouts(self):
        estimator = MarkerBoardEstimator(
            layout=MarkerBoardLayout(tag_size_m=0.045, gap_m=0.007),
            smoothing_alpha=1.0,
            lost_timeout_s=0.35,
            min_visible_tags=1,
        )

        first = estimator.estimate([_det(4, 0.1, 1.0)], now_s=10.0)
        held = estimator.estimate([], now_s=10.2)
        lost = estimator.estimate([], now_s=10.5)

        self.assertIsNotNone(first)
        self.assertIsNotNone(held)
        self.assertAlmostEqual(held.angle_h, first.angle_h, places=6)
        self.assertIsNone(lost)

    def test_ignores_non_finite_pose_candidates(self):
        estimator = MarkerBoardEstimator(
            layout=MarkerBoardLayout(tag_size_m=0.045, gap_m=0.007),
            smoothing_alpha=1.0,
            min_visible_tags=1,
        )

        board = estimator.estimate([
            _det(4, float("nan"), 1.0),
            _det(5, 0.052, 1.0),
        ])

        self.assertIsNotNone(board)
        self.assertTrue(math.isfinite(board.angle_h))
        self.assertAlmostEqual(board.angle_h, 0.0, places=6)

    def test_default_requires_three_visible_tags_for_fresh_update(self):
        estimator = MarkerBoardEstimator(
            layout=MarkerBoardLayout(tag_size_m=0.045, gap_m=0.007),
            smoothing_alpha=1.0,
        )

        weak = estimator.estimate([
            _det(4, 0.0, 1.0),
            _det(5, 0.052, 1.0),
        ], now_s=1.0)
        strong = estimator.estimate([
            _det(3, -0.052, 1.0),
            _det(4, 0.0, 1.0),
            _det(5, 0.052, 1.0),
        ], now_s=1.2)
        held = estimator.estimate([
            _det(4, 0.0, 1.0),
        ], now_s=1.4)

        self.assertIsNone(weak)
        self.assertIsNotNone(strong)
        self.assertIsNotNone(held)
        self.assertTrue(held.held)

    def test_predicts_short_dropouts_from_recent_board_velocity(self):
        estimator = MarkerBoardEstimator(
            layout=MarkerBoardLayout(tag_size_m=0.045, gap_m=0.007),
            smoothing_alpha=1.0,
            lost_timeout_s=0.8,
            prediction_timeout_s=0.3,
        )

        first = estimator.estimate([
            _det(3, -0.052, 1.0),
            _det(4, 0.0, 1.0),
            _det(5, 0.052, 1.0),
        ], now_s=1.0)
        second = estimator.estimate([
            _det(3, -0.032, 1.0),
            _det(4, 0.020, 1.0),
            _det(5, 0.072, 1.0),
        ], now_s=1.1)
        predicted = estimator.estimate([], now_s=1.2)
        held_after_prediction = estimator.estimate([], now_s=1.5)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(predicted)
        self.assertTrue(predicted.held)
        self.assertTrue(predicted.predicted)
        self.assertAlmostEqual(predicted.tvec[0], 0.040, places=6)
        self.assertIsNotNone(held_after_prediction)
        self.assertTrue(held_after_prediction.held)
        self.assertFalse(held_after_prediction.predicted)


if __name__ == "__main__":
    unittest.main()
