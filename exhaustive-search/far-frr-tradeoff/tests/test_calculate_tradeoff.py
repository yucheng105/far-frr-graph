"""Tests for FAR/FRR trade-off selection."""

from decimal import Decimal
from pathlib import Path
import sys
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from calculate_tradeoff import (  # noqa: E402
    build_far_limits,
    output_fieldnames,
    select_tradeoff_rows,
)


class TradeoffSelectionTests(unittest.TestCase):
    def test_selects_lowest_frr_without_exceeding_limit(self):
        step = Decimal("0.005")
        limits = build_far_limits(step)
        rows = [
            {"id": "first", "FAR": "0.002", "FRR": "0.8"},
            {"id": "worse", "FAR": "0.005", "FRR": "0.9"},
            {"id": "next", "FAR": "0.006", "FRR": "0.7"},
            {"id": "later", "FAR": "0.012", "FRR": "0.6"},
        ]

        selected = select_tradeoff_rows(rows, limits, step)
        by_limit = {limit: row["id"] for limit, row in selected}

        self.assertNotIn(Decimal("0.000"), by_limit)
        self.assertEqual(by_limit[Decimal("0.005")], "first")
        self.assertEqual(by_limit[Decimal("0.010")], "next")
        self.assertEqual(by_limit[Decimal("0.015")], "later")

    def test_equal_frr_prefers_lower_far_then_earlier_row(self):
        step = Decimal("0.005")
        limits = build_far_limits(step)
        rows = [
            {"id": "higher-far", "FAR": "0.004", "FRR": "0.5"},
            {"id": "lower-far-first", "FAR": "0.003", "FRR": "0.5"},
            {"id": "lower-far-later", "FAR": "0.003", "FRR": "0.5"},
        ]

        selected = select_tradeoff_rows(rows, limits, step)

        self.assertEqual(selected[0][0], Decimal("0.005"))
        self.assertEqual(selected[0][1]["id"], "lower-far-first")

    def test_output_header_inserts_far_limit_before_far(self):
        source = ["Threshold", "FAR", "FRR"]

        self.assertEqual(
            output_fieldnames(source),
            ["Threshold", "FAR Limit", "FAR", "FRR"],
        )


if __name__ == "__main__":
    unittest.main()
