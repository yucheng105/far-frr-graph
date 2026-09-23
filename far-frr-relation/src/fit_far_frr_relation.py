"""Batch-fit exponential FAR/FRR relations for threshold-sweep CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit

from plot_fit_diagnostics import write_fit_diagnostic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "exhaustive-search" / "threshold-sweep" / "results" / "1000"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "far-frr-relation" / "results" / "1000"
PARAMETER_FIELDS = [
    "Source",
    "Relation",
    "a",
    "b",
    "c",
    "a_standard_error",
    "b_standard_error",
    "c_standard_error",
    "RMSE",
    "MAE",
    "Max_absolute_error",
    "R_squared",
    "Equation",
]
# curve_fit bounds are closed. The smallest positive float gives strict signs
# without imposing a meaningful lower scale on a or |b|.
MIN_MONOTONIC_PARAMETER = np.nextafter(0.0, 1.0)


class FitResult(NamedTuple):
    params: np.ndarray
    covariance: np.ndarray
    fitted_values: np.ndarray
    rmse: float
    mae: float
    max_absolute_error: float
    r_squared: float


def exp_func(threshold: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential relation: rate = a * exp(b * threshold) + c."""
    return a * np.exp(b * threshold) + c


def read_sweep(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds: list[float] = []
    far_values: list[float] = []
    frr_values: list[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"Threshold", "FAR", "FRR"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                threshold = float(row["Threshold"])
                far = float(row["FAR"])
                frr = float(row["FRR"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {csv_path} at row {row_number}"
                ) from error

            thresholds.append(threshold)
            far_values.append(far)
            frr_values.append(frr)

    if len(thresholds) < 3:
        raise ValueError(f"{csv_path} needs at least three threshold rows")

    return (
        np.asarray(thresholds, dtype=float),
        np.asarray(far_values, dtype=float),
        np.asarray(frr_values, dtype=float),
    )


def fit_relation(
    thresholds: np.ndarray, values: np.ndarray, increasing: bool
) -> FitResult:
    span = max(float(np.ptp(values)), 1e-6)
    initial_params = (span, 1.0 if increasing else -5.0, float(np.min(values)))
    if increasing:
        # derivative = a * b * exp(b * t), so a > 0 and b > 0 guarantee FAR rises.
        bounds = (
            (MIN_MONOTONIC_PARAMETER, MIN_MONOTONIC_PARAMETER, -np.inf),
            (np.inf, np.inf, np.inf),
        )
    else:
        # a > 0 and b < 0 guarantee the FRR derivative is negative.
        bounds = (
            (MIN_MONOTONIC_PARAMETER, -np.inf, -np.inf),
            (np.inf, -MIN_MONOTONIC_PARAMETER, np.inf),
        )

    params, covariance = curve_fit(
        exp_func,
        thresholds,
        values,
        p0=initial_params,
        bounds=bounds,
        max_nfev=100_000,
    )
    fitted_values = exp_func(thresholds, *params)
    residuals = values - fitted_values
    absolute_errors = np.abs(residuals)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(absolute_errors))
    max_absolute_error = float(np.max(absolute_errors))
    total_variation = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total_variation
    return FitResult(
        params,
        covariance,
        fitted_values,
        rmse,
        mae,
        max_absolute_error,
        r_squared,
    )


def equation(name: str, params: np.ndarray) -> str:
    a, b, c = params
    operator = "+" if c >= 0 else "-"
    return (
        f"{name}(t) = {a:.12g} * exp({b:.12g} * t) "
        f"{operator} {abs(c):.12g}"
    )


def parameter_rows(source: str, fits: dict[str, FitResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, fit in fits.items():
        standard_errors = np.sqrt(np.diag(fit.covariance))
        rows.append(
            {
                "Source": source,
                "Relation": name,
                "a": fit.params[0],
                "b": fit.params[1],
                "c": fit.params[2],
                "a_standard_error": standard_errors[0],
                "b_standard_error": standard_errors[1],
                "c_standard_error": standard_errors[2],
                "RMSE": fit.rmse,
                "MAE": fit.mae,
                "Max_absolute_error": fit.max_absolute_error,
                "R_squared": fit.r_squared,
                "Equation": equation(name, fit.params),
            }
        )
    return rows


def write_parameters(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PARAMETER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_fitted_values(
    output_path: Path,
    thresholds: np.ndarray,
    far_values: np.ndarray,
    frr_values: np.ndarray,
    far_fitted: np.ndarray,
    frr_fitted: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Threshold", "FAR", "FAR_fitted", "FRR", "FRR_fitted"])
        writer.writerows(
            zip(thresholds, far_values, far_fitted, frr_values, frr_fitted)
        )


def make_title(csv_path: Path) -> str:
    name = csv_path.stem.removeprefix("threshold_sweep_")
    parts = name.split("_", maxsplit=1)
    detector = parts[0].upper()
    dataset = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
    return f"{detector} {dataset}: FAR/FRR exponential relations".strip()


def process_csv(
    input_path: Path, input_dir: Path, output_dir: Path
) -> list[dict[str, object]]:
    relative_path = input_path.relative_to(input_dir)
    source = relative_path.as_posix()
    destination = output_dir / relative_path.parent
    destination.mkdir(parents=True, exist_ok=True)

    thresholds, far_values, frr_values = read_sweep(input_path)
    fits = {
        "FAR": fit_relation(thresholds, far_values, increasing=True),
        "FRR": fit_relation(thresholds, frr_values, increasing=False),
    }
    rows = parameter_rows(source, fits)
    stem = input_path.stem

    write_parameters(destination / f"{stem}_fit_parameters.csv", rows)
    write_fitted_values(
        destination / f"{stem}_fitted_values.csv",
        thresholds,
        far_values,
        frr_values,
        fits["FAR"].fitted_values,
        fits["FRR"].fitted_values,
    )
    write_fit_diagnostic(
        destination / f"{stem}_exponential_fit.png",
        make_title(input_path),
        thresholds,
        far_values,
        frr_values,
        fits["FAR"].fitted_values,
        fits["FRR"].fitted_values,
    )

    print(source)
    for row in rows:
        print(
            f"  {row['Equation']} "
            f"(RMSE={row['RMSE']:.8f}, MAE={row['MAE']:.8f}, "
            f"MaxAE={row['Max_absolute_error']:.8f}, "
            f"R^2={row['R_squared']:.8f})"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively fit every threshold-sweep CSV with "
            "a * exp(b * threshold) + c."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.rglob("*.csv"))
    if not input_paths:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")

    all_parameter_rows: list[dict[str, object]] = []
    for input_path in input_paths:
        all_parameter_rows.extend(process_csv(input_path, args.input_dir, args.output_dir))

    summary_path = args.output_dir / "fit_parameters_summary.csv"
    write_parameters(summary_path, all_parameter_rows)
    print(f"Processed {len(input_paths)} CSV files")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
