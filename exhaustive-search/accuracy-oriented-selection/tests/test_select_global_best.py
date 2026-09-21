"""Tests for global accuracy-oriented selection."""

from pathlib import Path
import sys
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from select_global_best import select_global_rows  # noqa: E402


class GlobalSelectionTests(unittest.TestCase):
    def test_selects_lowest_frr_across_permutations(self):
        sources = [
            (
                Path("b_permutation.csv"),
                [
                    {"detector": "b", "FAR Limit": "0.005", "FAR": "0.004", "FRR": "0.7"},
                    {"detector": "b", "FAR Limit": "0.010", "FAR": "0.009", "FRR": "0.5"},
                ],
            ),
            (
                Path("a_permutation.csv"),
                [
                    {"detector": "a", "FAR Limit": "0.005", "FAR": "0.003", "FRR": "0.6"},
                    {"detector": "a", "FAR Limit": "0.010", "FAR": "0.008", "FRR": "0.6"},
                ],
            ),
        ]

        selected = select_global_rows(sources)

        self.assertEqual(selected[next(iter(sorted(selected)))]["detector"], "a")
        self.assertEqual(selected[max(selected)]["detector"], "b")

    def test_equal_frr_prefers_lower_far(self):
        sources = [
            (
                Path("first.csv"),
                [{"detector": "first", "FAR Limit": "0.010", "FAR": "0.009", "FRR": "0.5"}],
            ),
            (
                Path("second.csv"),
                [{"detector": "second", "FAR Limit": "0.010", "FAR": "0.008", "FRR": "0.5"}],
            ),
        ]

        selected = select_global_rows(sources)

        self.assertEqual(selected[max(selected)]["detector"], "second")

    def test_rejects_far_over_limit(self):
        sources = [
            (
                Path("invalid.csv"),
                [{"FAR Limit": "0.005", "FAR": "0.006", "FRR": "0.5"}],
            )
        ]

        with self.assertRaisesRegex(ValueError, "violates FAR <= FAR Limit"):
            select_global_rows(sources)


if __name__ == "__main__":
    unittest.main()
