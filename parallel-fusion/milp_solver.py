"""MILP model for parallel fusion of Xception, UCF, and SBI scores."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


REQUIRED_COLUMNS = {
    "dataset",
    "sample_id",
    "label(ground_truth)",
    "xception_fake_score",
    "ucf_fake_score",
    "sbi_fake_score",
}


class MilpSolveError(RuntimeError):
    """Raised when the MILP solver does not return a usable solution."""


@dataclass(frozen=True, slots=True)
class FusionSample:
    dataset: str
    sample_id: str
    label: int
    xception_score: float
    ucf_score: float
    sbi_score: float
    source_row: int


@dataclass(slots=True)
class FusionResult:
    dataset: str
    alpha_xception: float
    alpha_ucf: float
    alpha_sbi: float
    beta: float
    predictions: np.ndarray
    fusion_scores: np.ndarray
    false_accepts: int
    false_rejections: int
    n_real: int
    n_fake: int
    far: float
    frr: float
    objective: float
    optimal: bool
    solver_status: int
    solver_message: str
    mip_gap: float | None
    mip_node_count: int | None


def _parse_score(value: str, column: str, path: Path, row_number: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: row {row_number} has invalid {column}: {value!r}"
        ) from error

    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{path}: row {row_number} has out-of-range {column}: {score}"
        )
    return score


def read_fusion_csv(path: Path) -> list[FusionSample]:
    """Read and validate a merged score CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Fusion CSV not found: {path}")

    samples: list[FusionSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            dataset = row["dataset"].strip()
            sample_id = row["sample_id"].strip()
            if not dataset:
                raise ValueError(f"{path}: row {row_number} has a blank dataset")
            if not sample_id:
                raise ValueError(f"{path}: row {row_number} has a blank sample_id")

            try:
                label = int(row["label(ground_truth)"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: row {row_number} has an invalid ground-truth label"
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f"{path}: row {row_number} label must be 0 or 1, got {label}"
                )

            samples.append(
                FusionSample(
                    dataset=dataset,
                    sample_id=sample_id,
                    label=label,
                    xception_score=_parse_score(
                        row["xception_fake_score"],
                        "xception_fake_score",
                        path,
                        row_number,
                    ),
                    ucf_score=_parse_score(
                        row["ucf_fake_score"],
                        "ucf_fake_score",
                        path,
                        row_number,
                    ),
                    sbi_score=_parse_score(
                        row["sbi_fake_score"],
                        "sbi_fake_score",
                        path,
                        row_number,
                    ),
                    source_row=row_number,
                )
            )

    if not samples:
        raise ValueError(f"{path} contains no samples")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(datasets)}")
    return samples


def _stable_threshold(
    fusion_scores: np.ndarray,
    predictions: np.ndarray,
    epsilon: float,
) -> float:
    """Choose a numerically stable beta from the feasible threshold interval."""
    predicted_real = fusion_scores[predictions == 0]
    predicted_fake = fusion_scores[predictions == 1]

    lower = 0.0 if predicted_real.size == 0 else float(predicted_real.max() + epsilon)
    upper = 1.0 if predicted_fake.size == 0 else float(predicted_fake.min())
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    tolerance = 1e-7
    if lower > upper + tolerance:
        raise MilpSolveError(
            "The solver returned predictions that do not admit a valid threshold: "
            f"lower={lower}, upper={upper}"
        )
    return (lower + upper) / 2.0


def solve_parallel_fusion(
    samples: list[FusionSample],
    *,
    epsilon: float = 1e-6,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    display_solver_log: bool = False,
) -> FusionResult:
    """Solve the parallel-fusion MILP for one dataset."""
    if not samples:
        raise ValueError("At least one sample is required")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between 0 and 1")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if mip_rel_gap is not None and not 0.0 <= mip_rel_gap <= 1.0:
        raise ValueError("mip_rel_gap must be between 0 and 1")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset, got: {sorted(datasets)}")
    dataset = next(iter(datasets))

    labels = np.fromiter((sample.label for sample in samples), dtype=np.int8)
    scores = np.array(
        [
            (sample.xception_score, sample.ucf_score, sample.sbi_score)
            for sample in samples
        ],
        dtype=float,
    )
    sample_count = len(samples)
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"{dataset} needs both classes; n_real={n_real}, n_fake={n_fake}"
        )

    # Variable order: alpha_X, alpha_U, alpha_S, beta, z_1, ..., z_N, where
    # z_j is 1 exactly when sample j is misclassified.  This is an equivalent
    # but substantially tighter formulation than assigning two Big-M rows to
    # every prediction e_j.
    variable_count = 4 + sample_count
    objective_coefficients = np.zeros(variable_count, dtype=float)
    objective_coefficients[4:][labels == 0] = 1.0 / n_real
    objective_coefficients[4:][labels == 1] = 1.0 / n_fake

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[4:] = 1

    # Constraint 0 is alpha_X + alpha_U + alpha_S = 1. Each sample contributes
    # one class-specific Big-M row. q_j is substituted directly as
    # alpha_X*S_Xj + alpha_U*S_Uj + alpha_S*S_Sj.
    constraint_count = 1 + sample_count
    row_indices: list[int] = [0, 0, 0]
    column_indices: list[int] = [0, 1, 2]
    values: list[float] = [1.0, 1.0, 1.0]
    constraint_lower = np.full(constraint_count, -np.inf, dtype=float)
    constraint_upper = np.full(constraint_count, np.inf, dtype=float)
    constraint_lower[0] = 1.0
    constraint_upper[0] = 1.0

    big_m = 1.0
    for sample_index, detector_scores in enumerate(scores):
        error_column = 4 + sample_index
        constraint_row = 1 + sample_index

        for detector_column, score in enumerate(detector_scores):
            row_indices.append(constraint_row)
            column_indices.append(detector_column)
            values.append(float(score))

        row_indices.extend((constraint_row, constraint_row))
        column_indices.extend((3, error_column))
        if labels[sample_index] == 0:
            # A real sample is correct when q_j <= beta - epsilon.
            # q_j - beta - (M + epsilon)*z_j <= -epsilon
            values.extend((-1.0, -(big_m + epsilon)))
            constraint_upper[constraint_row] = -epsilon
        else:
            # A fake sample is correct when q_j >= beta.
            # q_j - beta + M*z_j >= 0
            values.extend((-1.0, big_m))
            constraint_lower[constraint_row] = 0.0

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(constraint_count, variable_count),
    ).tocsr()
    constraints = LinearConstraint(matrix, constraint_lower, constraint_upper)

    options: dict[str, Any] = {"disp": display_solver_log}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    solution = milp(
        c=objective_coefficients,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options=options,
    )
    if solution.x is None:
        raise MilpSolveError(
            f"MILP failed for {dataset}: status={solution.status}, "
            f"message={solution.message}"
        )

    weights = np.asarray(solution.x[:3], dtype=float)
    raw_errors = np.asarray(solution.x[4:], dtype=float)
    error_variables = np.rint(raw_errors).astype(np.int8)
    if np.max(np.abs(raw_errors - error_variables)) > 1e-5:
        raise MilpSolveError("The MILP returned non-integral error variables")
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise MilpSolveError(f"Detector weights do not sum to 1: {weights}")

    fusion_scores = scores @ weights
    raw_beta = float(solution.x[3])
    predictions = (fusion_scores >= raw_beta).astype(np.int8)
    # Select the midpoint of the equivalent threshold interval so exported
    # predictions remain stable when the solver places beta on a score.
    beta = _stable_threshold(fusion_scores, predictions, 0.0)
    threshold_predictions = (fusion_scores >= beta).astype(np.int8)
    if not np.array_equal(predictions, threshold_predictions):
        raise MilpSolveError("Could not stabilize the solver threshold")
    realized_errors = (predictions != labels).astype(np.int8)
    if np.any(realized_errors > error_variables):
        raise MilpSolveError("A realized error is not covered by its MILP error variable")

    false_accepts = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_rejections = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    far = false_accepts / n_real
    frr = false_rejections / n_fake

    raw_gap = getattr(solution, "mip_gap", None)
    raw_node_count = getattr(solution, "mip_node_count", None)
    return FusionResult(
        dataset=dataset,
        alpha_xception=float(weights[0]),
        alpha_ucf=float(weights[1]),
        alpha_sbi=float(weights[2]),
        beta=beta,
        predictions=predictions,
        fusion_scores=fusion_scores,
        false_accepts=false_accepts,
        false_rejections=false_rejections,
        n_real=n_real,
        n_fake=n_fake,
        far=far,
        frr=frr,
        objective=far + frr,
        # If a time-limited incumbent contains conservative error flags, it is
        # valid but has not proved the actual threshold metrics optimal.
        optimal=bool(solution.success and np.array_equal(error_variables, realized_errors)),
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        mip_gap=None if raw_gap is None else float(raw_gap),
        mip_node_count=None if raw_node_count is None else int(raw_node_count),
    )
