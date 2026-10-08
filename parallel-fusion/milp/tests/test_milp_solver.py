from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from milp_solver import (  # noqa: E402
    FusionSample,
    _build_indicator_constraints,
    build_fusion_seed,
    solve_parallel_fusion,
)


class ParallelFusionMilpTests(unittest.TestCase):
    def make_samples(self) -> list[FusionSample]:
        rows = [
            ("real-a", 0, 0.05, 0.70, 0.40),
            ("real-b", 0, 0.15, 0.65, 0.45),
            ("fake-a", 1, 0.85, 0.30, 0.55),
            ("fake-b", 1, 0.95, 0.35, 0.60),
        ]
        return [
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

    def assert_indicator_feasibility(
        self,
        *,
        score: float,
        beta: float,
        prediction: int,
        epsilon: float,
        expected: bool,
    ) -> None:
        scores = np.array([[score, score, score]], dtype=float)
        matrix, lower, upper = _build_indicator_constraints(scores, epsilon)
        # alpha_X=1 fixes q to score. The last variable is the fixed objective
        # offset and does not participate in the indicator constraints.
        variables = np.array([1.0, 0.0, 0.0, beta, prediction, 1.0])
        values = matrix @ variables
        feasible = bool(
            np.all(values >= lower - 1e-12)
            and np.all(values <= upper + 1e-12)
        )
        self.assertEqual(feasible, expected)

    def test_strict_indicator_boundaries_have_no_label_ambiguity(self) -> None:
        epsilon = 1e-6
        beta = 0.5

        # At q == beta, only Fake is feasible.
        self.assert_indicator_feasibility(
            score=beta,
            beta=beta,
            prediction=0,
            epsilon=epsilon,
            expected=False,
        )
        self.assert_indicator_feasibility(
            score=beta,
            beta=beta,
            prediction=1,
            epsilon=epsilon,
            expected=True,
        )

        # At q == beta - epsilon, only Real is feasible.
        self.assert_indicator_feasibility(
            score=beta - epsilon,
            beta=beta,
            prediction=0,
            epsilon=epsilon,
            expected=True,
        )
        self.assert_indicator_feasibility(
            score=beta - epsilon,
            beta=beta,
            prediction=1,
            epsilon=epsilon,
            expected=False,
        )

        # The open margin contains no solver-selectable label.
        for prediction in (0, 1):
            self.assert_indicator_feasibility(
                score=beta - epsilon / 2,
                beta=beta,
                prediction=prediction,
                epsilon=epsilon,
                expected=False,
            )

    def test_finds_zero_error_solution(self) -> None:
        samples = self.make_samples()

        epsilon = 1e-6
        result = solve_parallel_fusion(samples, epsilon=epsilon)

        self.assertTrue(result.optimal)
        self.assertAlmostEqual(
            result.alpha_xception + result.alpha_ucf + result.alpha_sbi,
            1.0,
            places=7,
        )
        self.assertEqual(result.false_accepts, 0)
        self.assertEqual(result.false_rejections, 0)
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.termination_reason, "optimal")
        self.assertIsNotNone(result.best_bound)
        self.assertAlmostEqual(result.best_bound, 0.0, places=7)
        self.assertEqual(result.predictions.tolist(), [0, 0, 1, 1])
        self.assertTrue(
            all(
                result.fusion_scores[result.predictions == 0]
                <= result.beta - epsilon + 1e-7
            )
        )
        self.assertTrue(all(result.fusion_scores[result.predictions == 1] >= result.beta))

    def test_grid_seed_adds_incumbent_cutoff(self) -> None:
        samples = self.make_samples()
        seed = build_fusion_seed(
            samples,
            alpha_xception=1.0,
            alpha_ucf=0.0,
            alpha_sbi=0.0,
            beta=0.5,
        )

        result = solve_parallel_fusion(samples, grid_seed=seed)

        self.assertEqual(seed.objective, 0.0)
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.grid_start_objective, 0.0)
        self.assertEqual(result.solution_source, "milp")

    def test_rejects_grid_seed_inside_epsilon_margin(self) -> None:
        samples = [
            FusionSample("toy", "real", 0, 0.4999995, 0.4999995, 0.4999995, 2),
            FusionSample("toy", "fake", 1, 0.5, 0.5, 0.5, 3),
        ]

        with self.assertRaisesRegex(ValueError, "not feasible under epsilon"):
            build_fusion_seed(
                samples,
                alpha_xception=1.0,
                alpha_ucf=0.0,
                alpha_sbi=0.0,
                beta=0.5,
                epsilon=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
