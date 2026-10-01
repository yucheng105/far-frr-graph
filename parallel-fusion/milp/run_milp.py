#!/usr/bin/env python3
"""Run the parallel-fusion MILP for one or all datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from milp_solver import FusionResult, FusionSample, read_fusion_csv, solve_parallel_fusion


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "milp-results"

DATASET_FILES = {
    "Celeb-DF-v2": "Celeb-DF-v2_fusion.csv",
    "DFDC": "DFDC_fusion.csv",
    "FaceForensics++": "FaceForensics++_fusion.csv",
}

SUMMARY_COLUMNS = [
    "dataset",
    "alpha_xception",
    "alpha_ucf",
    "alpha_sbi",
    "beta",
    "far",
    "frr",
    "far_plus_frr",
    "false_accepts",
    "false_rejections",
    "n_real",
    "n_fake",
    "optimal",
    "solver_status",
    "mip_gap",
    "mip_node_count",
    "solver_message",
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
        description="Optimize Xception/UCF/SBI fusion weights and threshold with MILP."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_FILES],
        default="all",
        help="Dataset to optimize (default: all).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing *_fusion.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for summary and per-sample result CSVs.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Strict margin for q < beta when prediction is Real (default: 1e-6).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional HiGHS time limit in seconds.",
    )
    parser.add_argument(
        "--mip-rel-gap",
        type=float,
        default=None,
        help="Optional relative MIP gap in [0, 1].",
    )
    parser.add_argument(
        "--solver-log",
        action="store_true",
        help="Show the HiGHS solver log.",
    )
    return parser.parse_args()


def atomic_write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def error_type(label: int, prediction: int) -> str:
    if label == 0 and prediction == 1:
        return "false_accept"
    if label == 1 and prediction == 0:
        return "false_reject"
    return "correct"


def prediction_rows(
    samples: list[FusionSample], result: FusionResult
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_index, (sample, score, prediction) in enumerate(
        zip(samples, result.fusion_scores, result.predictions, strict=True),
        start=1,
    ):
        predicted_label = int(prediction)
        rows.append(
            {
                "dataset": sample.dataset,
                "sample_index": sample_index,
                "sample_id": sample.sample_id,
                "label(ground_truth)": sample.label,
                "xception_fake_score": format(sample.xception_score, ".17g"),
                "ucf_fake_score": format(sample.ucf_score, ".17g"),
                "sbi_fake_score": format(sample.sbi_score, ".17g"),
                "fusion_score": format(float(score), ".17g"),
                "prediction": predicted_label,
                "error_type": error_type(sample.label, predicted_label),
            }
        )
    return rows


def summary_row(result: FusionResult) -> dict[str, object]:
    return {
        "dataset": result.dataset,
        "alpha_xception": format(result.alpha_xception, ".17g"),
        "alpha_ucf": format(result.alpha_ucf, ".17g"),
        "alpha_sbi": format(result.alpha_sbi, ".17g"),
        "beta": format(result.beta, ".17g"),
        "far": format(result.far, ".17g"),
        "frr": format(result.frr, ".17g"),
        "far_plus_frr": format(result.objective, ".17g"),
        "false_accepts": result.false_accepts,
        "false_rejections": result.false_rejections,
        "n_real": result.n_real,
        "n_fake": result.n_fake,
        "optimal": result.optimal,
        "solver_status": result.solver_status,
        "mip_gap": "" if result.mip_gap is None else format(result.mip_gap, ".17g"),
        "mip_node_count": "" if result.mip_node_count is None else result.mip_node_count,
        "solver_message": result.solver_message,
    }


def print_result(result: FusionResult) -> None:
    print(f"Dataset: {result.dataset}")
    print("Optimal weights" if result.optimal else "Best feasible weights found")
    print(f"  Xception: {result.alpha_xception:.10f}")
    print(f"  UCF:      {result.alpha_ucf:.10f}")
    print(f"  SBI:      {result.alpha_sbi:.10f}")
    print(f"Threshold:  {result.beta:.10f}")
    print("Performance")
    print(f"  FAR:      {result.far:.10f} ({result.false_accepts}/{result.n_real})")
    print(f"  FRR:      {result.frr:.10f} ({result.false_rejections}/{result.n_fake})")
    print(f"  Objective:{result.objective:.10f}")
    print(f"  Optimal:  {result.optimal}")
    if result.mip_gap is not None:
        print(f"  MIP gap:  {result.mip_gap:.6%}")
    print()


def main() -> None:
    args = parse_args()
    selected_datasets = (
        list(DATASET_FILES) if args.dataset == "all" else [args.dataset]
    )

    summary_rows: list[dict[str, object]] = []
    for dataset in selected_datasets:
        input_path = args.input_dir / DATASET_FILES[dataset]
        samples = read_fusion_csv(input_path)
        if samples[0].dataset != dataset:
            raise ValueError(
                f"{input_path} contains dataset {samples[0].dataset!r}, expected {dataset!r}"
            )

        result = solve_parallel_fusion(
            samples,
            epsilon=args.epsilon,
            time_limit=args.time_limit,
            mip_rel_gap=args.mip_rel_gap,
            display_solver_log=args.solver_log,
        )
        prediction_path = args.output_dir / f"{dataset}_milp_predictions.csv"
        atomic_write_csv(
            prediction_path,
            PREDICTION_COLUMNS,
            prediction_rows(samples, result),
        )
        summary_rows.append(summary_row(result))
        print_result(result)

    summary_path = args.output_dir / "milp_summary.csv"
    atomic_write_csv(summary_path, SUMMARY_COLUMNS, summary_rows)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
