#!/usr/bin/env python3
"""Run exhaustive grid search for parallel fusion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from exhaustive_search import (
    DatasetScores,
    ExhaustiveResult,
    evaluate_grid_point,
    exhaustive_search,
    read_fusion_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PARALLEL_FUSION_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

DATASET_FILES = {
    "Celeb-DF-v2": "Celeb-DF-v2_fusion.csv",
    "DFDC": "DFDC_fusion.csv",
    "FaceForensics++": "FaceForensics++_fusion.csv",
}

GRID_COLUMNS = [
    "dataset",
    "alpha_xception",
    "alpha_ucf",
    "alpha_sbi",
    "beta",
    "false_accepts",
    "false_rejections",
    "n_real",
    "n_fake",
    "far",
    "frr",
    "far_plus_frr",
]

SUMMARY_COLUMNS = [
    "dataset",
    "step",
    "interval_count",
    "grid_value_count",
    "alpha_combination_count",
    "evaluated_combination_count",
    "best_tie_count",
    "alpha_xception",
    "alpha_ucf",
    "alpha_sbi",
    "beta",
    "false_accepts",
    "false_rejections",
    "n_real",
    "n_fake",
    "far",
    "frr",
    "far_plus_frr",
]

PREDICTION_COLUMNS = [
    "dataset",
    "sample_index",
    "sample_id",
    "label(ground_truth)",
    "xception_fake_score",
    "ucf_fake_score",
    "sbi_fake_score",
    "fusion_score",
    "prediction",
    "error_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively search fusion weights and threshold on a fixed grid."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_FILES],
        default="all",
        help="Dataset to evaluate (default: all).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Grid step that divides 1 exactly (default: 0.05).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PARALLEL_FUSION_DIR,
        help="Directory containing the merged *_fusion.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exhaustive-search results.",
    )
    return parser.parse_args()


def atomic_write_csv(path: Path, columns: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def format_number(value: float) -> str:
    return format(value, ".17g")


def format_grid_value(value: float) -> str:
    return format(value, ".12g")


def grid_rows(data: DatasetScores, result: ExhaustiveResult):
    n_real = int(np.count_nonzero(data.labels == 0))
    n_fake = int(np.count_nonzero(data.labels == 1))
    for point in result.points:
        yield {
            "dataset": data.dataset,
            "alpha_xception": format_grid_value(point.alpha_xception),
            "alpha_ucf": format_grid_value(point.alpha_ucf),
            "alpha_sbi": format_grid_value(point.alpha_sbi),
            "beta": format_grid_value(point.beta),
            "false_accepts": point.false_accepts,
            "false_rejections": point.false_rejections,
            "n_real": n_real,
            "n_fake": n_fake,
            "far": format_number(point.far),
            "frr": format_number(point.frr),
            "far_plus_frr": format_number(point.objective),
        }


def summary_row(data: DatasetScores, result: ExhaustiveResult) -> dict[str, object]:
    best = result.best
    n_real = int(np.count_nonzero(data.labels == 0))
    n_fake = int(np.count_nonzero(data.labels == 1))
    return {
        "dataset": data.dataset,
        "step": format_grid_value(result.step),
        "interval_count": result.interval_count,
        "grid_value_count": result.interval_count + 1,
        "alpha_combination_count": result.alpha_combination_count,
        "evaluated_combination_count": len(result.points),
        "best_tie_count": result.best_tie_count,
        "alpha_xception": format_grid_value(best.alpha_xception),
        "alpha_ucf": format_grid_value(best.alpha_ucf),
        "alpha_sbi": format_grid_value(best.alpha_sbi),
        "beta": format_grid_value(best.beta),
        "false_accepts": best.false_accepts,
        "false_rejections": best.false_rejections,
        "n_real": n_real,
        "n_fake": n_fake,
        "far": format_number(best.far),
        "frr": format_number(best.frr),
        "far_plus_frr": format_number(best.objective),
    }


def prediction_rows(data: DatasetScores, result: ExhaustiveResult):
    best = result.best
    weights = np.array(
        [best.alpha_xception, best.alpha_ucf, best.alpha_sbi], dtype=float
    )
    *_, fusion_scores, predictions = evaluate_grid_point(data, weights, best.beta)
    for sample_index, (sample_id, label, source_scores, fusion_score, prediction) in enumerate(
        zip(
            data.sample_ids,
            data.labels,
            data.scores,
            fusion_scores,
            predictions,
            strict=True,
        ),
        start=1,
    ):
        label_value = int(label)
        prediction_value = int(prediction)
        if label_value == 0 and prediction_value == 1:
            error = "false_accept"
        elif label_value == 1 and prediction_value == 0:
            error = "false_reject"
        else:
            error = "correct"
        yield {
            "dataset": data.dataset,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "label(ground_truth)": label_value,
            "xception_fake_score": format_number(float(source_scores[0])),
            "ucf_fake_score": format_number(float(source_scores[1])),
            "sbi_fake_score": format_number(float(source_scores[2])),
            "fusion_score": format_number(float(fusion_score)),
            "prediction": prediction_value,
            "error_type": error,
        }


def print_result(result: ExhaustiveResult) -> None:
    best = result.best
    print(f"Dataset: {result.dataset}")
    print(
        f"  Grid: {result.alpha_combination_count} alpha combinations x "
        f"{result.interval_count + 1} beta values = {len(result.points)} rows"
    )
    print(
        "  Alpha: "
        f"Xception={best.alpha_xception:.2f}, "
        f"UCF={best.alpha_ucf:.2f}, SBI={best.alpha_sbi:.2f}"
    )
    print(f"  Beta: {best.beta:.2f}")
    print(f"  FAR: {best.far:.10f}")
    print(f"  FRR: {best.frr:.10f}")
    print(f"  FAR+FRR: {best.objective:.10f}")
    print(f"  Objective ties: {result.best_tie_count}")
    print()


def main() -> None:
    args = parse_args()
    selected_datasets = (
        list(DATASET_FILES) if args.dataset == "all" else [args.dataset]
    )

    summaries: list[dict[str, object]] = []
    for dataset in selected_datasets:
        data = read_fusion_csv(args.input_dir / DATASET_FILES[dataset])
        if data.dataset != dataset:
            raise ValueError(
                f"Input contains dataset {data.dataset!r}, expected {dataset!r}"
            )
        result = exhaustive_search(data, step=args.step)

        atomic_write_csv(
            args.output_dir / f"{dataset}_exhaustive_grid.csv",
            GRID_COLUMNS,
            grid_rows(data, result),
        )
        atomic_write_csv(
            args.output_dir / f"{dataset}_best_predictions.csv",
            PREDICTION_COLUMNS,
            prediction_rows(data, result),
        )
        summaries.append(summary_row(data, result))
        print_result(result)

    atomic_write_csv(
        args.output_dir / "exhaustive_best_summary.csv",
        SUMMARY_COLUMNS,
        summaries,
    )
    print(f"Summary: {args.output_dir / 'exhaustive_best_summary.csv'}")


if __name__ == "__main__":
    main()
