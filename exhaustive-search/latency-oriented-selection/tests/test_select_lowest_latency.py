"""Tests for latency-oriented selection."""

from decimal import Decimal
from pathlib import Path
import sys
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from select_lowest_latency import (  # noqa: E402
    permutation_latency,
    select_lowest_latency_rows,
)


TIMES = {
    "sbi": Decimal("100"),
    "ucf": Decimal("20"),
    "xception": Decimal("10"),
}


def make_row(
    detector: str,
    far_limit: str,
    far: str,
    frr: str,
) -> dict[str, str]:
    row = {
        "sbi_stage": "-1",
        "ucf_stage": "-1",
        "xception_stage": "-1",
        "FAR Limit": far_limit,
        "FAR": far,
        "FRR": frr,
    }
    row[f"{detector}_stage"] = "1"
    return row


class LatencySelectionTests(unittest.TestCase):
    def test_permutation_latency_sums_used_detectors(self):
        row = make_row("xception", "0.010", "0.009", "0.8")
        row["ucf_stage"] = "2"

        latency = permutation_latency(row, TIMES, Path("source.csv"), 2)

        self.assertEqual(latency, Decimal("30"))

    def test_lower_latency_wins_even_with_higher_frr(self):
        sources = [
            (
                Path("slow.csv"),
                [make_row("sbi", "0.010", "0.009", "0.1")],
            ),
            (
                Path("fast.csv"),
                [make_row("xception", "0.010", "0.009", "0.9")],
            ),
        ]

        selected = select_lowest_latency_rows(sources, TIMES)
        row, latency = selected[Decimal("0.010")]

        self.assertEqual(row["xception_stage"], "1")
        self.assertEqual(latency, Decimal("10"))

    def test_equal_latency_prefers_lower_frr(self):
        sources = [
            (
                Path("first.csv"),
                [make_row("xception", "0.010", "0.009", "0.7")],
            ),
            (
                Path("second.csv"),
                [make_row("xception", "0.010", "0.008", "0.6")],
            ),
        ]

        selected = select_lowest_latency_rows(sources, TIMES)
        row, _ = selected[Decimal("0.010")]

        self.assertEqual(row["FRR"], "0.6")


if __name__ == "__main__":
    unittest.main()
