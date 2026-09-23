"""Analyze whether (FAR - alpha) * (FRR - beta) is constant."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = (
    PROJECT_ROOT / "exhaustive-search" / "threshold-sweep" / "results" / "1000"
)
DEFAULT_PARAMETERS = (
    PROJECT_ROOT
    / "far-frr-relation"
    / "results"
    / "equations"
    / "1000"
    / "fit_parameters_summary.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "far-frr-relation" / "results" / "product"

SUMMARY_FIELDS = [
    "Detector",
    "Dataset",
    "b_FAR+b_FRR",
    "K_mean",
    "K_std",
    "K_CV",
    "K_min",
    "K_max",
]
K_VALUE_FIELDS = [
    "Detector",
    "Dataset",
    "Threshold",
    "FAR",
    "FRR",
    "alpha",
    "beta",
    "FAR-alpha",
    "FRR-beta",
    "K_value",
]
DATASET_LABELS = {
    "celebdf": "Celeb-DF",
    "combined": "Combined",
    "dfdc": "DFDC",
    "faceforensics": "FF++",
}


def read_parameters(parameters_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    with parameters_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Source", "Relation", "b", "c"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"{parameters_path} is missing columns: {columns}")

        for row_number, row in enumerate(reader, start=2):
            source = row["Source"].strip().replace("\\", "/")
            relation = row["Relation"].strip().upper()
            if relation not in {"FAR", "FRR"}:
                raise ValueError(
                    f"Unexpected relation {relation!r} at row {row_number}"
                )
            if relation in grouped[source]:
                raise ValueError(f"Duplicate {relation} parameters for {source}")

            try:
                grouped[source][relation] = {
                    "b": float(row["b"]),
                    "c": float(row["c"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid b or c in {parameters_path} at row {row_number}"
                ) from error

    for source, relations in grouped.items():
        missing_relations = {"FAR", "FRR"} - set(relations)
        if missing_relations:
            missing = ", ".join(sorted(missing_relations))
            raise ValueError(f"{source} is missing parameter rows: {missing}")

    if not grouped:
        raise ValueError(f"No parameter rows found in {parameters_path}")
    return dict(grouped)


def read_sweep(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds: list[float] = []
    far_values: list[float] = []
    frr_values: list[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Threshold", "FAR", "FRR"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            columns = ", ".join(sorted(missing))
            raise ValueError(f"{csv_path} is missing columns: {columns}")

        for row_number, row in enumerate(reader, start=2):
            try:
                thresholds.append(float(row["Threshold"]))
                far_values.append(float(row["FAR"]))
                frr_values.append(float(row["FRR"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {csv_path} at row {row_number}"
                ) from error

    if not thresholds:
        raise ValueError(f"No data rows found in {csv_path}")
    return (
        np.asarray(thresholds, dtype=float),
        np.asarray(far_values, dtype=float),
        np.asarray(frr_values, dtype=float),
    )


def detector_and_dataset(source: str) -> tuple[str, str]:
    stem = Path(source).stem.removeprefix("threshold_sweep_")
    parts = stem.split("_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Cannot determine detector and dataset from {source}")

    detector, dataset = parts
    return detector.upper(), DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def analyze_source(
    source: str,
    relations: dict[str, dict[str, float]],
    sweep_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    sweep_path = sweep_dir / Path(source)
    if not sweep_path.is_file():
        raise FileNotFoundError(f"Threshold-sweep CSV not found: {sweep_path}")

    thresholds, far_values, frr_values = read_sweep(sweep_path)
    alpha = relations["FAR"]["c"]
    beta = relations["FRR"]["c"]
    far_shifted = far_values - alpha
    frr_shifted = frr_values - beta
    k_values = far_shifted * frr_shifted

    k_mean = float(np.mean(k_values))
    k_std = float(np.std(k_values))
    k_min = float(np.min(k_values))
    k_max = float(np.max(k_values))
    k_cv = k_std / abs(k_mean) if k_mean != 0 else math.inf
    detector, dataset = detector_and_dataset(source)

    summary = {
        "Detector": detector,
        "Dataset": dataset,
        "b_FAR+b_FRR": relations["FAR"]["b"] + relations["FRR"]["b"],
        "K_mean": k_mean,
        "K_std": k_std,
        "K_CV": k_cv,
        "K_min": k_min,
        "K_max": k_max,
    }
    values = [
        {
            "Detector": detector,
            "Dataset": dataset,
            "Threshold": threshold,
            "FAR": far,
            "FRR": frr,
            "alpha": alpha,
            "beta": beta,
            "FAR-alpha": shifted_far,
            "FRR-beta": shifted_frr,
            "K_value": k_value,
        }
        for threshold, far, frr, shifted_far, shifted_frr, k_value in zip(
            thresholds,
            far_values,
            frr_values,
            far_shifted,
            frr_shifted,
            k_values,
        )
    ]
    return summary, values


def write_csv(
    output_path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: object) -> str:
    return f"{float(value):.6g}"


def write_markdown(output_path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "Detector",
        "Dataset",
        "(b_FAR+b_FRR)",
        "K mean",
        "K std",
        "K CV",
        "K min",
        "K max",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: |",
    ]
    for row in rows:
        values = [
            str(row["Detector"]),
            str(row["Dataset"]),
            format_number(row["b_FAR+b_FRR"]),
            format_number(row["K_mean"]),
            format_number(row["K_std"]),
            f"{float(row['K_CV']):.2%}",
            format_number(row["K_min"]),
            format_number(row["K_max"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate K = (FAR - alpha) * (FRR - beta) for every sweep."
    )
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = read_parameters(args.parameters)
    summaries: list[dict[str, object]] = []
    all_k_values: list[dict[str, object]] = []

    for source in sorted(parameters):
        summary, k_values = analyze_source(source, parameters[source], args.sweep_dir)
        summaries.append(summary)
        all_k_values.extend(k_values)

    summaries.sort(key=lambda row: (str(row["Detector"]), str(row["Dataset"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "k_summary.csv"
    summary_markdown = args.output_dir / "k_summary.md"
    values_csv = args.output_dir / "k_values.csv"

    write_csv(summary_csv, SUMMARY_FIELDS, summaries)
    write_markdown(summary_markdown, summaries)
    write_csv(values_csv, K_VALUE_FIELDS, all_k_values)

    print(f"Processed {len(summaries)} detector/dataset combinations")
    print(f"Wrote {len(all_k_values)} threshold-level K values")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_markdown}")
    print(f"Wrote {values_csv}")


if __name__ == "__main__":
    main()
