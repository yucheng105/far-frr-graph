from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from milp_solver import FusionSample, solve_parallel_fusion  # noqa: E402


class ParallelFusionMilpTests(unittest.TestCase):
    def test_finds_zero_error_solution(self) -> None:
        rows = [
            ("real-a", 0, 0.05, 0.70, 0.40),
            ("real-b", 0, 0.15, 0.65, 0.45),
            ("fake-a", 1, 0.85, 0.30, 0.55),
            ("fake-b", 1, 0.95, 0.35, 0.60),
        ]
        samples = [
            FusionSample(
                dataset="toy",
                sample_id=sample_id,
                label=label,
                xception_score=xception,
                ucf_score=ucf,
                sbi_score=sbi,
                source_row=index + 2,
            )
            for index, (sample_id, label, xception, ucf, sbi) in enumerate(rows)
        ]

        result = solve_parallel_fusion(samples)

        self.assertTrue(result.optimal)
        self.assertAlmostEqual(
            result.alpha_xception + result.alpha_ucf + result.alpha_sbi,
            1.0,
            places=7,
        )
        self.assertEqual(result.false_accepts, 0)
        self.assertEqual(result.false_rejections, 0)
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.predictions.tolist(), [0, 0, 1, 1])
        self.assertTrue(all(result.fusion_scores[result.predictions == 0] < result.beta))
        self.assertTrue(all(result.fusion_scores[result.predictions == 1] >= result.beta))


if __name__ == "__main__":
    unittest.main()
