#!/usr/bin/env python3
"""Merge Xception, UCF, and SBI fake scores into one CSV per dataset."""

from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_ROOT = PROJECT_ROOT / "exhaustive-search" / "threshold-sweep" / "input-csvs"

FIELDNAMES = [
    "dataset",
    "sample_id",
    "label(ground_truth)",
    "xception_fake_score",
    "ucf_fake_score",
    "sbi_fake_score",
]

DATASETS = [
    {
        "dataset": "Celeb-DF-v2",
        "xception": INPUT_ROOT / "xception" / "xception_Celeb-DF-v2_predictions.csv",
        "ucf": INPUT_ROOT / "ucf" / "ucf_Celeb-DF-v2_predictions.csv",
        "sbi": INPUT_ROOT / "sbi" / "scores_visual_sbi_celebdf_v0.csv",
        "output": SCRIPT_DIR / "Celeb-DF-v2_fusion.csv",
    },
    {
        "dataset": "DFDC",
        "xception": INPUT_ROOT / "xception" / "xception_DFDC_predictions.csv",
        "ucf": INPUT_ROOT / "ucf" / "ucf_DFDC_predictions.csv",
        "sbi": INPUT_ROOT / "sbi" / "scores_visual_sbi_dfdc_v0.csv",
        "output": SCRIPT_DIR / "DFDC_fusion.csv",
    },
    {
        "dataset": "FaceForensics++",
        "xception": INPUT_ROOT / "xception" / "xception_FaceForensics++_predictions.csv",
        "ucf": INPUT_ROOT / "ucf" / "ucf_FaceForensics++_predictions.csv",
        "sbi": INPUT_ROOT / "sbi" / "scores_visual_sbi_v0.csv",
        "output": SCRIPT_DIR / "FaceForensics++_fusion.csv",
    },
]

REQUIRED_COLUMNS = {"sample_id", "dataset", "label", "fake_score"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")
    return rows


def validate_score(value: str, detector: str, sample_id: str, row_number: int) -> None:
    if value == "":
        raise ValueError(
            f"Blank {detector} fake_score at row {row_number} for sample {sample_id}"
        )
    try:
        score = float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid {detector} fake_score at row {row_number}: {value!r}"
        ) from error
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Out-of-range {detector} fake_score at row {row_number}: {score}"
        )


def index_by_sample_occurrence(
    rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, int], dict[str, str]], list[tuple[str, int]]]:
    """Index rows by sample ID and its zero-based occurrence within that file."""
    occurrence_counts: dict[str, int] = {}
    indexed_rows: dict[tuple[str, int], dict[str, str]] = {}
    ordered_keys: list[tuple[str, int]] = []

    for row in rows:
        sample_id = row["sample_id"]
        occurrence = occurrence_counts.get(sample_id, 0)
        key = (sample_id, occurrence)
        occurrence_counts[sample_id] = occurrence + 1
        indexed_rows[key] = row
        ordered_keys.append(key)

    return indexed_rows, ordered_keys


def merge_dataset(config: dict[str, object]) -> int:
    dataset = str(config["dataset"])
    detector_rows = {
        detector: read_csv(Path(config[detector]))
        for detector in ("xception", "ucf", "sbi")
    }

    row_counts = {detector: len(rows) for detector, rows in detector_rows.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"{dataset} row counts do not match: {row_counts}")

    indexed = {}
    xception_order: list[tuple[str, int]] = []
    for detector, rows in detector_rows.items():
        indexed[detector], order = index_by_sample_occurrence(rows)
        if detector == "xception":
            xception_order = order

    reference_keys = set(indexed["xception"])
    for detector in ("ucf", "sbi"):
        detector_keys = set(indexed[detector])
        if detector_keys != reference_keys:
            missing = sorted(reference_keys - detector_keys)[:5]
            extra = sorted(detector_keys - reference_keys)[:5]
            raise ValueError(
                f"{dataset} sample occurrences differ for {detector}; "
                f"missing={missing}, extra={extra}"
            )

    merged_rows: list[dict[str, str]] = []
    for index, key in enumerate(xception_order, start=2):
        xception = indexed["xception"][key]
        ucf = indexed["ucf"][key]
        sbi = indexed["sbi"][key]
        labels = {xception["label"], ucf["label"], sbi["label"]}
        if len(labels) != 1:
            raise ValueError(
                f"{dataset} label mismatch at source row {index}, "
                f"sample {xception['sample_id']}"
            )

        for detector, row in (("xception", xception), ("ucf", ucf), ("sbi", sbi)):
            if "status" in row and row["status"] != "ok":
                raise ValueError(
                    f"{dataset} has non-ok {detector} result at source row {index}: "
                    f"{row['status']!r}"
                )
            validate_score(row["fake_score"], detector, row["sample_id"], index)

        merged_rows.append(
            {
                "dataset": dataset,
                "sample_id": xception["sample_id"],
                "label(ground_truth)": xception["label"],
                "xception_fake_score": xception["fake_score"],
                "ucf_fake_score": ucf["fake_score"],
                "sbi_fake_score": sbi["fake_score"],
            }
        )

    output = Path(config["output"])
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged_rows)
    temporary_output.replace(output)
    return len(merged_rows)


def main() -> None:
    for config in DATASETS:
        row_count = merge_dataset(config)
        print(f"{config['dataset']}: wrote {row_count} rows to {config['output']}")


if __name__ == "__main__":
    main()
