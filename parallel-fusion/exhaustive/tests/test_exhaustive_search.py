from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from exhaustive_search import (  # noqa: E402
    DatasetScores,
    evaluate_grid_point,
    exhaustive_search,
    step_to_interval_count,
)


class ExhaustiveSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DatasetScores(
            dataset="toy",
            sample_ids=["real-a", "real-b", "fake-a", "fake-b"],
            labels=np.array([0, 0, 1, 1], dtype=np.int8),
            scores=np.array(
                [
                    [0.05, 0.70, 0.40],
                    [0.15, 0.65, 0.45],
                    [0.85, 0.30, 0.55],
                    [0.95, 0.35, 0.60],
                ],
                dtype=float,
            ),
        )

    def test_step_point_counts(self) -> None:
        result = exhaustive_search(self.data, step=0.05)
        self.assertEqual(step_to_interval_count(0.05), 20)
        self.assertEqual(result.alpha_combination_count, 231)
        self.assertEqual(len(result.points), 4_851)

    def test_finds_zero_error_grid_point(self) -> None:
        result = exhaustive_search(self.data, step=0.05)
        self.assertEqual(result.best.false_accepts, 0)
        self.assertEqual(result.best.false_rejections, 0)
        self.assertEqual(result.best.objective, 0.0)

    def test_threshold_equality_is_fake(self) -> None:
        weights = np.array([1.0, 0.0, 0.0])
        *_, predictions = evaluate_grid_point(self.data, weights, beta=0.05)
        self.assertEqual(int(predictions[0]), 1)


if __name__ == "__main__":
    unittest.main()
