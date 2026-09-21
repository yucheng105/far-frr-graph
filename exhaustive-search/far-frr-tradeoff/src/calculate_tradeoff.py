"""Select the lowest-FRR threshold configuration under each FAR limit."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_DIR / "threshold-sweep" / "results" / "20"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "far-frr-tradeoff" / "results" / "20"
DEFAULT_FAR_STEP = Decimal("0.005")
MIN_FAR_LIMIT = Decimal("0")
MAX_FAR_LIMIT = Decimal("1")


def parse_positive_decimal(value: str) -> Decimal:
    """Parse a positive decimal CLI value."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"Invalid decimal value: {value}") from error

    if not parsed.is_finite() or parsed <= 0:
        raise argparse.ArgumentTypeError("FAR step must be a finite positive number")

    return parsed


def build_far_limits(step: Decimal) -> list[Decimal]:
    """Return limits from zero through one, requiring the step to divide one."""
    quotient = MAX_FAR_LIMIT / step
    if quotient != quotient.to_integral_value():
        raise ValueError(f"FAR step {step} must divide 1 exactly")

    return [step * index for index in range(int(quotient) + 1)]


def format_limit(value: Decimal, step: Decimal) -> str:
    """Format limits with enough fixed decimal places to show the configured step."""
    decimal_places = max(0, -step.normalize().as_tuple().exponent)
    return f"{value:.{decimal_places}f}"


def candidate_is_better(
    candidate: tuple[Decimal, Decimal, int, dict[str, str]],
    current: tuple[Decimal, Decimal, int, dict[str, str]] | None,
) -> bool:
    """Rank by FRR, then FAR, then original input order."""
    return current is None or candidate[:3] < current[:3]


def select_tradeoff_rows(
    rows: Iterable[dict[str, str]],
    far_limits: list[Decimal],
    step: Decimal,
) -> list[tuple[Decimal, dict[str, str]]]:
    """Select the best eligible source row for every FAR limit in one pass."""
    best_by_first_eligible_limit: list[
        tuple[Decimal, Decimal, int, dict[str, str]] | None
    ] = [None] * len(far_limits)

    for row_number, row in enumerate(rows, start=2):
        try:
            far = Decimal(row["FAR"])
            frr = Decimal(row["FRR"])
        except (KeyError, InvalidOperation) as error:
            raise ValueError(
                f"Row {row_number} has an invalid FAR or FRR value"
            ) from error

        if not far.is_finite() or not frr.is_finite():
            raise ValueError(f"Row {row_number} has a non-finite FAR or FRR value")
        if far < MIN_FAR_LIMIT or far > MAX_FAR_LIMIT:
            raise ValueError(f"Row {row_number} has FAR outside [0, 1]: {far}")

        first_limit_index = int((far / step).to_integral_value(rounding=ROUND_CEILING))
        candidate = (frr, far, row_number, row.copy())
        current = best_by_first_eligible_limit[first_limit_index]
        if candidate_is_better(candidate, current):
            best_by_first_eligible_limit[first_limit_index] = candidate

    selected_rows: list[tuple[Decimal, dict[str, str]]] = []
    best_so_far: tuple[Decimal, Decimal, int, dict[str, str]] | None = None

    for limit, new_candidate in zip(far_limits, best_by_first_eligible_limit):
        if new_candidate is not None and candidate_is_better(
            new_candidate, best_so_far
        ):
            best_so_far = new_candidate

        if best_so_far is not None:
            selected_rows.append((limit, best_so_far[3]))

    return selected_rows


def output_fieldnames(source_fieldnames: list[str] | None) -> list[str]:
    """Insert FAR Limit immediately before FAR while preserving source columns."""
    if not source_fieldnames:
        raise ValueError("CSV is missing a header row")

    missing = {"FAR", "FRR"} - set(source_fieldnames)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    if "FAR Limit" in source_fieldnames:
        raise ValueError("Source CSV already contains a FAR Limit column")

    far_index = source_fieldnames.index("FAR")
    return source_fieldnames[:far_index] + ["FAR Limit"] + source_fieldnames[far_index:]


def write_tradeoff_csv(
    source_path: Path,
    output_path: Path,
    far_limits: list[Decimal],
    step: Decimal,
) -> int:
    """Read one threshold sweep and write its FAR/FRR trade-off rows."""
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = output_fieldnames(reader.fieldnames)
        selected_rows = select_tradeoff_rows(reader, far_limits, step)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for limit, source_row in selected_rows:
            output_row = source_row.copy()
            output_row["FAR Limit"] = format_limit(limit, step)
            writer.writerow(output_row)

    return len(selected_rows)


def find_source_csvs(source_dir: Path) -> list[Path]:
    """Find only canonical cascade sweep files, one level below source_dir."""
    return sorted(source_dir.glob("*/cascade_threshold_sweep_*.csv"))


def make_output_path(source_path: Path, source_dir: Path, output_dir: Path) -> Path:
    """Mirror the permutation directory and use a trade-off-specific filename."""
    relative_path = source_path.relative_to(source_dir)
    filename = relative_path.name.replace(
        "cascade_threshold_sweep_", "far_frr_tradeoff_", 1
    )
    return output_dir / relative_path.parent / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the lowest FRR available under each FAR limit."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Threshold sweep root (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Trade-off output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--far-step",
        type=parse_positive_decimal,
        default=DEFAULT_FAR_STEP,
        help="FAR Limit interval; must divide 1 exactly (default: 0.005)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    far_limits = build_far_limits(args.far_step)

    source_paths = find_source_csvs(source_dir)
    if not source_paths:
        raise FileNotFoundError(
            f"No cascade threshold sweep CSV files found under {source_dir}"
        )

    for source_path in source_paths:
        output_path = make_output_path(source_path, source_dir, output_dir)
        row_count = write_tradeoff_csv(
            source_path,
            output_path,
            far_limits,
            args.far_step,
        )
        print(f"Wrote {output_path} ({row_count} rows)")


if __name__ == "__main__":
    main()
