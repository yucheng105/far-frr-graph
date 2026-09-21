"""Select the globally lowest-FRR configuration under every FAR limit."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


EXHAUSTIVE_SEARCH_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = EXHAUSTIVE_SEARCH_DIR / "far-frr-tradeoff" / "results" / "20"
DEFAULT_OUTPUT_DIR = (
    EXHAUSTIVE_SEARCH_DIR / "accuracy-oriented-selection" / "results" / "20"
)
DATASETS = ("celebdf", "dfdc", "faceforensics", "combined")
REQUIRED_COLUMNS = {"FAR Limit", "FAR", "FRR"}

Candidate = tuple[Decimal, Decimal, str, int, dict[str, str]]


def parse_rate(row: dict[str, str], column: str, source: Path, row_number: int) -> Decimal:
    """Parse and validate one rate field."""
    try:
        value = Decimal(row[column])
    except (KeyError, InvalidOperation) as error:
        raise ValueError(
            f"{source}: row {row_number} has an invalid {column} value"
        ) from error

    if not value.is_finite() or value < 0 or value > 1:
        raise ValueError(
            f"{source}: row {row_number} has {column} outside [0, 1]: {value}"
        )

    return value


def candidate_is_better(candidate: Candidate, current: Candidate | None) -> bool:
    """Rank by FRR, FAR, source path, and source row for deterministic output."""
    return current is None or candidate[:4] < current[:4]


def select_global_rows(
    sources: Iterable[tuple[Path, Iterable[dict[str, str]]]],
) -> dict[Decimal, dict[str, str]]:
    """Return the globally best row for each FAR Limit."""
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

            candidate: Candidate = (frr, far, source_key, row_number, row.copy())
            current = best_by_limit.get(far_limit)
            if candidate_is_better(candidate, current):
                best_by_limit[far_limit] = candidate

    return {limit: candidate[4] for limit, candidate in best_by_limit.items()}


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

            if expected_fieldnames is None:
                expected_fieldnames = fieldnames
            elif fieldnames != expected_fieldnames:
                raise ValueError(f"{path} does not use the expected table header")

            sources.append((path, list(reader)))

    if expected_fieldnames is None:
        raise ValueError("No source tables were provided")

    return expected_fieldnames, sources


def find_dataset_paths(source_dir: Path, dataset: str) -> list[Path]:
    """Find every per-permutation trade-off table for one dataset."""
    return sorted(source_dir.glob(f"*/far_frr_tradeoff_*_{dataset}.csv"))


def write_dataset_result(
    dataset: str,
    source_dir: Path,
    output_dir: Path,
) -> tuple[Path, int, int]:
    """Build and write one dataset's global accuracy-oriented selection table."""
    paths = find_dataset_paths(source_dir, dataset)
    if not paths:
        raise FileNotFoundError(
            f"No FAR/FRR trade-off files found for {dataset} under {source_dir}"
        )

    fieldnames, sources = read_dataset_sources(paths)
    best_by_limit = select_global_rows(sources)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"accuracy_oriented_selection_{dataset}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for far_limit in sorted(best_by_limit):
            writer.writerow(best_by_limit[far_limit])

    return output_path, len(best_by_limit), len(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the globally lowest-FRR detector and threshold combination "
            "under each FAR limit."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    for dataset in DATASETS:
        output_path, row_count, source_count = write_dataset_result(
            dataset,
            source_dir,
            output_dir,
        )
        print(
            f"Wrote {output_path} ({row_count} rows; "
            f"compared {source_count} permutations)"
        )


if __name__ == "__main__":
    main()
