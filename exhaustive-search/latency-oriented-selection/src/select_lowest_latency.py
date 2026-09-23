"""Select the lowest-latency permutation under every FAR limit."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[3]
EXHAUSTIVE_SEARCH_DIR = PROJECT_DIR / "exhaustive-search"
DEFAULT_SOURCE_DIR = EXHAUSTIVE_SEARCH_DIR / "far-frr-tradeoff" / "results" / "20"
DEFAULT_OUTPUT_DIR = (
    EXHAUSTIVE_SEARCH_DIR / "latency-oriented-selection" / "results" / "20"
)
DETECTORS = ("sbi", "ucf", "xception")
DATASETS = ("celebdf", "dfdc", "faceforensics", "combined")
DATASET_TIME_LABELS = {
    "celebdf": "CelebDF",
    "dfdc": "DFDC",
    "faceforensics": "FaceForensics",
}
REQUIRED_COLUMNS = {"FAR Limit", "FAR", "FRR"}
LATENCY_COLUMN = "Inference Time (ms)"

Candidate = tuple[Decimal, Decimal, Decimal, str, int, dict[str, str]]


def parse_decimal(
    value: str,
    description: str,
    source: Path,
    row_number: int,
) -> Decimal:
    """Parse a finite decimal and report its source on failure."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"{source}: row {row_number} has an invalid {description}: {value!r}"
        ) from error

    if not parsed.is_finite():
        raise ValueError(
            f"{source}: row {row_number} has a non-finite {description}: {value!r}"
        )

    return parsed


def read_detector_times(time_dir: Path, combined_average: str) -> dict[str, dict[str, Decimal]]:
    """Read the dataset-specific average inference time for every detector."""
    combined_label = f"Combined({combined_average.capitalize()})"
    labels = {**DATASET_TIME_LABELS, "combined": combined_label}
    times: dict[str, dict[str, Decimal]] = {dataset: {} for dataset in DATASETS}

    for detector in DETECTORS:
        path = time_dir / f"{detector}_average_inference_time.csv"
        if not path.exists():
            raise FileNotFoundError(f"Inference-time CSV not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = {"Dataset", "Time(ms)"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {', '.join(sorted(missing))}"
                )

            by_label: dict[str, Decimal] = {}
            for row_number, row in enumerate(reader, start=2):
                label = row["Dataset"].strip()
                inference_time = parse_decimal(
                    row["Time(ms)"], "Time(ms)", path, row_number
                )
                if inference_time < 0:
                    raise ValueError(
                        f"{path}: row {row_number} has negative inference time"
                    )
                by_label[label] = inference_time

        for dataset, label in labels.items():
            if label not in by_label:
                raise ValueError(f"{path} does not contain the {label!r} time")
            times[dataset][detector] = by_label[label]

    return times


def detectors_in_row(row: dict[str, str], source: Path, row_number: int) -> tuple[str, ...]:
    """Identify used detectors from their stage columns."""
    used: list[tuple[int, str]] = []

    for detector in DETECTORS:
        column = f"{detector}_stage"
        try:
            stage = int(row[column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{source}: row {row_number} has an invalid {column} value"
            ) from error

        if stage == -1:
            continue
        if stage < 1:
            raise ValueError(
                f"{source}: row {row_number} has invalid stage {stage} for {detector}"
            )
        used.append((stage, detector))

    if not used:
        raise ValueError(f"{source}: row {row_number} does not use any detector")

    stages = [stage for stage, _ in used]
    if len(stages) != len(set(stages)):
        raise ValueError(f"{source}: row {row_number} contains duplicate stage numbers")

    return tuple(detector for _, detector in sorted(used))


def permutation_latency(
    row: dict[str, str],
    detector_times: dict[str, Decimal],
    source: Path,
    row_number: int,
) -> Decimal:
    """Sum the inference times of all detectors used by the row."""
    detectors = detectors_in_row(row, source, row_number)
    return sum((detector_times[detector] for detector in detectors), Decimal("0"))


def parse_rate(row: dict[str, str], column: str, source: Path, row_number: int) -> Decimal:
    value = parse_decimal(row[column], column, source, row_number)
    if value < 0 or value > 1:
        raise ValueError(
            f"{source}: row {row_number} has {column} outside [0, 1]: {value}"
        )
    return value


def candidate_is_better(candidate: Candidate, current: Candidate | None) -> bool:
    """Rank by latency, FRR, FAR, source path, and source row."""
    return current is None or candidate[:5] < current[:5]


def select_lowest_latency_rows(
    sources: Iterable[tuple[Path, Iterable[dict[str, str]]]],
    detector_times: dict[str, Decimal],
) -> dict[Decimal, tuple[dict[str, str], Decimal]]:
    """Return the lowest-latency candidate for every FAR Limit."""
    best_by_limit: dict[Decimal, Candidate] = {}

    for source, rows in sources:
        source_key = source.as_posix()
        for row_number, row in enumerate(rows, start=2):
            far_limit = parse_rate(row, "FAR Limit", source, row_number)
            far = parse_rate(row, "FAR", source, row_number)
            frr = parse_rate(row, "FRR", source, row_number)
            if far > far_limit:
                raise ValueError(
                    f"{source}: row {row_number} violates FAR <= FAR Limit "
                    f"({far} > {far_limit})"
                )

            latency = permutation_latency(row, detector_times, source, row_number)
            candidate: Candidate = (
                latency,
                frr,
                far,
                source_key,
                row_number,
                row.copy(),
            )
            current = best_by_limit.get(far_limit)
            if candidate_is_better(candidate, current):
                best_by_limit[far_limit] = candidate

    return {
        limit: (candidate[5], candidate[0])
        for limit, candidate in best_by_limit.items()
    }


def read_dataset_sources(
    paths: list[Path],
) -> tuple[list[str], list[tuple[Path, list[dict[str, str]]]]]:
    """Read one dataset's per-permutation tables and validate matching headers."""
    expected_fieldnames: list[str] | None = None
    sources: list[tuple[Path, list[dict[str, str]]]] = []

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"{path} is missing a header row")

            missing = REQUIRED_COLUMNS - set(fieldnames)
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {', '.join(sorted(missing))}"
                )
            if LATENCY_COLUMN in fieldnames:
                raise ValueError(f"{path} already contains {LATENCY_COLUMN}")

            if expected_fieldnames is None:
                expected_fieldnames = fieldnames
            elif fieldnames != expected_fieldnames:
                raise ValueError(f"{path} does not use the expected table header")

            sources.append((path, list(reader)))

    if expected_fieldnames is None:
        raise ValueError("No source tables were provided")

    return expected_fieldnames, sources


def output_fieldnames(source_fieldnames: list[str]) -> list[str]:
    """Insert latency immediately before FAR Limit."""
    far_limit_index = source_fieldnames.index("FAR Limit")
    return (
        source_fieldnames[:far_limit_index]
        + [LATENCY_COLUMN]
        + source_fieldnames[far_limit_index:]
    )


def find_dataset_paths(source_dir: Path, dataset: str) -> list[Path]:
    return sorted(source_dir.glob(f"*/far_frr_tradeoff_*_{dataset}.csv"))


def write_dataset_result(
    dataset: str,
    source_dir: Path,
    output_dir: Path,
    detector_times: dict[str, Decimal],
) -> tuple[Path, int, int]:
    paths = find_dataset_paths(source_dir, dataset)
    if not paths:
        raise FileNotFoundError(
            f"No FAR/FRR trade-off files found for {dataset} under {source_dir}"
        )

    source_fieldnames, sources = read_dataset_sources(paths)
    best_by_limit = select_lowest_latency_rows(sources, detector_times)
    fieldnames = output_fieldnames(source_fieldnames)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"latency_oriented_selection_{dataset}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for far_limit in sorted(best_by_limit):
            source_row, latency = best_by_limit[far_limit]
            output_row = source_row.copy()
            output_row[LATENCY_COLUMN] = f"{latency:.2f}"
            writer.writerow(output_row)

    return output_path, len(best_by_limit), len(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the lowest-total-inference-time permutation under each FAR limit."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Per-permutation trade-off root (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--time-dir",
        type=Path,
        default=PROJECT_DIR,
        help=f"Directory containing detector time CSVs (default: {PROJECT_DIR})",
    )
    parser.add_argument(
        "--combined-average",
        choices=("macro", "micro"),
        default="macro",
        help="Inference-time row used for the combined dataset (default: macro)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    detector_times = read_detector_times(
        args.time_dir.resolve(), args.combined_average
    )

    for dataset in DATASETS:
        output_path, row_count, source_count = write_dataset_result(
            dataset,
            source_dir,
            output_dir,
            detector_times[dataset],
        )
        print(
            f"Wrote {output_path} ({row_count} rows; "
            f"compared {source_count} permutations)"
        )


if __name__ == "__main__":
    main()
