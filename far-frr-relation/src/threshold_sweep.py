"""Calculate FAR and FRR over an evenly spaced threshold sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "exhaustive-search"
    / "threshold-sweep"
    / "input-csvs"
    / "sbi"
    / "scores_visual_sbi_v0.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "exhaustive-search"
    / "threshold-sweep"
    / "results"
    / "1000"
    / "sbi"
    / "threshold_sweep_sbi_faceforensics.csv"
)


def read_scores(csv_path: Path) -> tuple[list[float], list[float]]:
    real_scores: list[float] = []
    fake_scores: list[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"label", "fake_score"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for row in reader:
            if row.get("status", "ok").strip().lower() != "ok":
                continue

            try:
                label = int(row["label"])
                score = float(row["fake_score"])
            except (TypeError, ValueError):
                continue

            if label == 0:
                real_scores.append(score)
            elif label == 1:
                fake_scores.append(score)

    if not real_scores:
        raise ValueError(f"{csv_path} has no valid real samples (label 0)")
    if not fake_scores:
        raise ValueError(f"{csv_path} has no valid fake samples (label 1)")

    return real_scores, fake_scores


def format_threshold(threshold: float) -> str:
    return f"{threshold:.3f}".rstrip("0").rstrip(".")


def calculate_sweep(
    real_scores: list[float], fake_scores: list[float], steps: int
):
    for index in range(1, steps + 1):
        threshold = index / steps
        false_accepts = sum(score < threshold for score in fake_scores)
        false_rejects = sum(score >= threshold for score in real_scores)
        yield {
            "Threshold": format_threshold(threshold),
            "FAR": round(false_accepts / len(fake_scores), 6),
            "FRR": round(false_rejects / len(real_scores), 6),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate a threshold sweep from detector score rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    real_scores, fake_scores = read_scores(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Threshold", "FAR", "FRR"])
        writer.writeheader()
        writer.writerows(calculate_sweep(real_scores, fake_scores, args.steps))

    print(
        f"Wrote {args.output} ({args.steps} thresholds; "
        f"{len(real_scores)} real, {len(fake_scores)} fake samples)"
    )


if __name__ == "__main__":
    main()
