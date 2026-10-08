"""MILP model for parallel fusion of Xception, UCF, and SBI scores."""

from __future__ import annotations

import csv
import warnings
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
class FusionSeed:
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
    best_bound: float | None
    mip_gap: float | None
    mip_node_count: int | None
    termination_reason: str
    solution_source: str
    grid_start_objective: float | None


def _termination_reason(status: int, message: str) -> str:
    message_lower = message.lower()
    if status == 0:
        return "optimal"
    if status == 1:
        if "time limit" in message_lower:
            return "time_limit"
        if "iteration" in message_lower or "node limit" in message_lower:
            return "iteration_or_node_limit"
        return "limit_reached"
    if status == 2:
        return "infeasible"
    if status == 3:
        return "unbounded"
    return "solver_error_or_other"


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

    numerical_tolerance = 1e-12
    if lower > upper + numerical_tolerance:
        raise MilpSolveError(
            "The solver returned predictions that do not admit a valid threshold: "
            f"lower={lower}, upper={upper}"
        )
    if lower > upper:
        # A sub-picounit overlap can arise from floating-point addition of
        # epsilon. Choosing the Fake boundary preserves q == beta -> e_j = 1.
        return upper
    return (lower + upper) / 2.0


def _build_indicator_constraints(
    scores: np.ndarray,
    epsilon: float,
) -> tuple[coo_matrix, np.ndarray, np.ndarray]:
    """Build strict Big-M rows for e_j = 1 iff q_j >= beta.

    Variable order is alpha_X, alpha_U, alpha_S, beta, e_1, ..., e_N,
    followed by one fixed objective-offset variable.
    """
    sample_count = scores.shape[0]
    variable_count = 5 + sample_count
    constraint_count = 1 + 2 * sample_count
    row_indices: list[int] = [0, 0, 0]
    column_indices: list[int] = [0, 1, 2]
    values: list[float] = [1.0, 1.0, 1.0]
    constraint_lower = np.full(constraint_count, -np.inf, dtype=float)
    constraint_upper = np.full(constraint_count, np.inf, dtype=float)
    constraint_lower[0] = 1.0
    constraint_upper[0] = 1.0

    big_m = 1.0
    for sample_index, detector_scores in enumerate(scores):
        prediction_column = 4 + sample_index
        lower_row = 1 + 2 * sample_index
        upper_row = lower_row + 1

        for detector_column, score in enumerate(detector_scores):
            row_indices.extend((lower_row, upper_row))
            column_indices.extend((detector_column, detector_column))
            values.extend((float(score), float(score)))

        # q_j - beta >= -M(1 - e_j)
        # q_j - beta - M*e_j >= -M
        row_indices.extend((lower_row, lower_row))
        column_indices.extend((3, prediction_column))
        values.extend((-1.0, -big_m))
        constraint_lower[lower_row] = -big_m

        # q_j - beta <= -epsilon + (M + epsilon)*e_j
        # q_j - beta - (M + epsilon)*e_j <= -epsilon
        row_indices.extend((upper_row, upper_row))
        column_indices.extend((3, prediction_column))
        values.extend((-1.0, -(big_m + epsilon)))
        constraint_upper[upper_row] = -epsilon

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(constraint_count, variable_count),
    ).tocsr()
    return matrix, constraint_lower, constraint_upper


def build_fusion_seed(
    samples: list[FusionSample],
    *,
    alpha_xception: float,
    alpha_ucf: float,
    alpha_sbi: float,
    beta: float,
    epsilon: float = 1e-6,
) -> FusionSeed:
    """Validate a grid-search point and convert it to a strict MILP incumbent."""
    if not samples:
        raise ValueError("At least one sample is required")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between 0 and 1")

    datasets = {sample.dataset for sample in samples}
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset, got: {sorted(datasets)}")
    dataset = next(iter(datasets))

    weights = np.array(
        [alpha_xception, alpha_ucf, alpha_sbi], dtype=float
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError(f"Grid-start weights must be in [0, 1], got {weights}")
    if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"Grid-start weights must sum to 1, got {weights}")
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"Grid-start beta must be in [0, 1], got {beta}")

    labels = np.fromiter((sample.label for sample in samples), dtype=np.int8)
    scores = np.array(
        [
            (sample.xception_score, sample.ucf_score, sample.sbi_score)
            for sample in samples
        ],
        dtype=float,
    )
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"{dataset} needs both classes; n_real={n_real}, n_fake={n_fake}"
        )

    fusion_scores = scores @ weights
    predictions = (fusion_scores >= beta).astype(np.int8)
    try:
        stable_beta = _stable_threshold(fusion_scores, predictions, epsilon)
    except MilpSolveError as error:
        raise ValueError(
            f"Grid-start point for {dataset} is not feasible under epsilon={epsilon}: "
            f"{error}"
        ) from error
    stable_predictions = (fusion_scores >= stable_beta).astype(np.int8)
    if not np.array_equal(predictions, stable_predictions):
        raise ValueError(
            f"Grid-start point for {dataset} changes labels under the strict epsilon rule"
        )

    false_accepts = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_rejections = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    far = false_accepts / n_real
    frr = false_rejections / n_fake
    return FusionSeed(
        dataset=dataset,
        alpha_xception=float(weights[0]),
        alpha_ucf=float(weights[1]),
        alpha_sbi=float(weights[2]),
        beta=stable_beta,
        predictions=predictions,
        fusion_scores=fusion_scores,
        false_accepts=false_accepts,
        false_rejections=false_rejections,
        n_real=n_real,
        n_fake=n_fake,
        far=far,
        frr=frr,
        objective=far + frr,
    )


def solve_parallel_fusion(
    samples: list[FusionSample],
    *,
    epsilon: float = 1e-6,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    display_solver_log: bool = False,
    grid_seed: FusionSeed | None = None,
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
    if grid_seed is not None:
        if grid_seed.dataset != dataset:
            raise ValueError(
                f"Grid seed is for {grid_seed.dataset!r}, expected {dataset!r}"
            )
        if len(grid_seed.predictions) != len(samples):
            raise ValueError("Grid seed sample count does not match MILP input")

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

    # Variable order: alpha_X, alpha_U, alpha_S, beta, e_1, ..., e_N, followed
    # by a fixed-one variable that represents the constant term in FRR. The
    # fixed variable keeps the solver objective and reported MIP gap equal to
    # FAR + FRR instead of a constant-shifted objective.
    prediction_start = 4
    objective_offset_index = prediction_start + sample_count
    variable_count = objective_offset_index + 1
    objective_coefficients = np.zeros(variable_count, dtype=float)
    prediction_objective = objective_coefficients[
        prediction_start:objective_offset_index
    ]
    prediction_objective[labels == 0] = 1.0 / n_real
    prediction_objective[labels == 1] = -1.0 / n_fake
    objective_coefficients[objective_offset_index] = 1.0

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    lower_bounds[objective_offset_index] = 1.0
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[prediction_start:objective_offset_index] = 1

    matrix, constraint_lower, constraint_upper = _build_indicator_constraints(
        scores, epsilon
    )
    constraints: list[LinearConstraint] = [
        LinearConstraint(matrix, constraint_lower, constraint_upper)
    ]
    if grid_seed is not None:
        # scipy.optimize.milp has no x0/MIP-start API. This objective cutoff is
        # the strongest safe use of the known incumbent through this backend:
        # the solver searches only solutions at least as good as the grid point.
        cutoff_tolerance = 1e-12
        constraints.append(
            LinearConstraint(
                objective_coefficients.reshape(1, -1),
                -np.inf,
                grid_seed.objective + cutoff_tolerance,
            )
        )

    # HiGHS' default MIP feasibility tolerance is too close to epsilon=1e-6
    # and can numerically reintroduce the boundary ambiguity. Keep it at least
    # two orders of magnitude smaller than epsilon.
    feasibility_tolerance = max(1e-10, min(1e-8, epsilon / 100.0))
    options: dict[str, Any] = {
        "disp": display_solver_log,
        "mip_feasibility_tolerance": feasibility_tolerance,
        "primal_feasibility_tolerance": feasibility_tolerance,
    }
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    with warnings.catch_warnings():
        # SciPy forwards these supported HiGHS options but warns because they
        # are not part of scipy.optimize.milp's short public option list.
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        solution = milp(
            c=objective_coefficients,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=constraints,
            options=options,
        )
    if solution.x is None and grid_seed is None:
        raise MilpSolveError(
            f"MILP failed for {dataset}: status={solution.status}, "
            f"message={solution.message}"
        )

    raw_gap = getattr(solution, "mip_gap", None)
    raw_node_count = getattr(solution, "mip_node_count", None)
    raw_best_bound = getattr(solution, "mip_dual_bound", None)
    best_bound = (
        None
        if raw_best_bound is None or not np.isfinite(raw_best_bound)
        else float(raw_best_bound)
    )
    termination_reason = _termination_reason(
        int(solution.status), str(solution.message)
    )
    if solution.x is None:
        fallback_gap = None
        if best_bound is not None:
            fallback_gap = max(
                0.0,
                (grid_seed.objective - best_bound)
                / max(abs(grid_seed.objective), 1e-12),
            )
        return FusionResult(
            dataset=dataset,
            alpha_xception=grid_seed.alpha_xception,
            alpha_ucf=grid_seed.alpha_ucf,
            alpha_sbi=grid_seed.alpha_sbi,
            beta=grid_seed.beta,
            predictions=grid_seed.predictions,
            fusion_scores=grid_seed.fusion_scores,
            false_accepts=grid_seed.false_accepts,
            false_rejections=grid_seed.false_rejections,
            n_real=grid_seed.n_real,
            n_fake=grid_seed.n_fake,
            far=grid_seed.far,
            frr=grid_seed.frr,
            objective=grid_seed.objective,
            optimal=False,
            solver_status=int(solution.status),
            solver_message=f"{solution.message} Using validated grid seed.",
            best_bound=best_bound,
            mip_gap=fallback_gap,
            mip_node_count=None if raw_node_count is None else int(raw_node_count),
            termination_reason=termination_reason,
            solution_source="grid_seed",
            grid_start_objective=grid_seed.objective,
        )

    weights = np.asarray(solution.x[:3], dtype=float)
    raw_predictions = np.asarray(
        solution.x[prediction_start:objective_offset_index], dtype=float
    )
    predictions = np.rint(raw_predictions).astype(np.int8)
    if np.max(np.abs(raw_predictions - predictions)) > 1e-5:
        raise MilpSolveError("The MILP returned non-integral prediction variables")
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise MilpSolveError(f"Detector weights do not sum to 1: {weights}")

    fusion_scores = scores @ weights
    # Choose the midpoint of the feasible threshold interval. This preserves
    # every e_j while keeping Real samples at least epsilon below beta.
    beta = _stable_threshold(fusion_scores, predictions, epsilon)
    threshold_predictions = (fusion_scores >= beta).astype(np.int8)
    if not np.array_equal(predictions, threshold_predictions):
        mismatch_count = int(np.count_nonzero(predictions != threshold_predictions))
        raise MilpSolveError(
            f"MILP predictions disagree with q >= beta for {mismatch_count} samples"
        )

    tolerance = feasibility_tolerance * 2.0
    if np.any(fusion_scores[predictions == 1] < beta - tolerance):
        raise MilpSolveError("A Fake prediction violates q_j >= beta")
    if np.any(fusion_scores[predictions == 0] > beta - epsilon + tolerance):
        raise MilpSolveError("A Real prediction violates q_j <= beta - epsilon")

    false_accepts = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_rejections = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    far = false_accepts / n_real
    frr = false_rejections / n_fake

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
        optimal=bool(solution.success),
        solver_status=int(solution.status),
        solver_message=str(solution.message),
        best_bound=best_bound,
        mip_gap=None if raw_gap is None else float(raw_gap),
        mip_node_count=None if raw_node_count is None else int(raw_node_count),
        termination_reason=termination_reason,
        solution_source="milp",
        grid_start_objective=(
            None if grid_seed is None else grid_seed.objective
        ),
    )
