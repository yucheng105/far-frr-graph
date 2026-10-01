"""Grid-based exhaustive search for parallel detector fusion."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "dataset",
    "sample_id",
    "label(ground_truth)",
    "xception_fake_score",
    "ucf_fake_score",
    "sbi_fake_score",
}


@dataclass(slots=True)
class DatasetScores:
    dataset: str
    sample_ids: list[str]
    labels: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class GridPoint:
    alpha_xception: float
    alpha_ucf: float
    alpha_sbi: float
    beta: float
    false_accepts: int
    false_rejections: int
    far: float
    frr: float
    objective: float


@dataclass(slots=True)
class ExhaustiveResult:
    dataset: str
    step: float
    interval_count: int
    alpha_combination_count: int
    points: list[GridPoint]
    best: GridPoint
    best_tie_count: int


def _parse_score(value: str, column: str, path: Path, row_number: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: row {row_number} has invalid {column}: {value!r}"
        ) from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{path}: row {row_number} has out-of-range {column}: {score}"
        )
    return score


def read_fusion_csv(path: Path) -> DatasetScores:
    if not path.is_file():
        raise FileNotFoundError(f"Fusion CSV not found: {path}")

    datasets: set[str] = set()
    sample_ids: list[str] = []
    labels: list[int] = []
    scores: list[tuple[float, float, float]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            dataset = row["dataset"].strip()
            sample_id = row["sample_id"].strip()
            if not dataset or not sample_id:
                raise ValueError(
                    f"{path}: row {row_number} has a blank dataset or sample_id"
                )
            try:
                label = int(row["label(ground_truth)"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}: row {row_number} has an invalid label"
                ) from error
            if label not in (0, 1):
                raise ValueError(
                    f"{path}: row {row_number} label must be 0 or 1, got {label}"
                )

            datasets.add(dataset)
            sample_ids.append(sample_id)
            labels.append(label)
            scores.append(
                (
                    _parse_score(
                        row["xception_fake_score"],
                        "xception_fake_score",
                        path,
                        row_number,
                    ),
                    _parse_score(
                        row["ucf_fake_score"],
                        "ucf_fake_score",
                        path,
                        row_number,
                    ),
                    _parse_score(
                        row["sbi_fake_score"],
                        "sbi_fake_score",
                        path,
                        row_number,
                    ),
                )
            )

    if not sample_ids:
        raise ValueError(f"{path} contains no samples")
    if len(datasets) != 1:
        raise ValueError(f"{path} contains multiple datasets: {sorted(datasets)}")

    return DatasetScores(
        dataset=next(iter(datasets)),
        sample_ids=sample_ids,
        labels=np.asarray(labels, dtype=np.int8),
        scores=np.asarray(scores, dtype=float),
    )


def step_to_interval_count(step: float) -> int:
    if not math.isfinite(step) or not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    interval_count = round(1.0 / step)
    if not math.isclose(interval_count * step, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("step must divide 1 exactly, for example 0.1 or 0.05")
    return interval_count


def alpha_unit_combinations(interval_count: int):
    """Yield integer alpha units whose sum is interval_count."""
    for xception_units in range(interval_count + 1):
        for ucf_units in range(interval_count - xception_units + 1):
            sbi_units = interval_count - xception_units - ucf_units
            yield xception_units, ucf_units, sbi_units


def evaluate_grid_point(
    data: DatasetScores,
    weights: np.ndarray,
    beta: float,
) -> tuple[int, int, float, float, float, np.ndarray, np.ndarray]:
    labels = data.labels
    n_real = int(np.count_nonzero(labels == 0))
    n_fake = int(np.count_nonzero(labels == 1))
    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"{data.dataset} needs both classes; n_real={n_real}, n_fake={n_fake}"
        )

    fusion_scores = data.scores @ weights
    predictions = (fusion_scores >= beta).astype(np.int8)
    false_accepts = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    false_rejections = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    far = false_accepts / n_real
    frr = false_rejections / n_fake
    return (
        false_accepts,
        false_rejections,
        far,
        frr,
        far + frr,
        fusion_scores,
        predictions,
    )


def exhaustive_search(data: DatasetScores, step: float = 0.05) -> ExhaustiveResult:
    interval_count = step_to_interval_count(step)
    labels = data.labels
    real_mask = labels == 0
    fake_mask = labels == 1
    n_real = int(np.count_nonzero(real_mask))
    n_fake = int(np.count_nonzero(fake_mask))
    if n_real == 0 or n_fake == 0:
        raise ValueError(
            f"{data.dataset} needs both classes; n_real={n_real}, n_fake={n_fake}"
        )

    betas = np.arange(interval_count + 1, dtype=float) / interval_count
    points: list[GridPoint] = []
    alpha_count = 0

    for xception_units, ucf_units, sbi_units in alpha_unit_combinations(interval_count):
        alpha_count += 1
        weights = np.array(
            [xception_units, ucf_units, sbi_units], dtype=float
        ) / interval_count
        fusion_scores = data.scores @ weights

        # Shape: samples x beta values. Equality is classified as Fake.
        predictions = fusion_scores[:, np.newaxis] >= betas[np.newaxis, :]
        false_accepts = predictions[real_mask].sum(axis=0, dtype=np.int64)
        false_rejections = (~predictions[fake_mask]).sum(axis=0, dtype=np.int64)
        fars = false_accepts / n_real
        frrs = false_rejections / n_fake
        objectives = fars + frrs

        for beta_index, beta in enumerate(betas):
            points.append(
                GridPoint(
                    alpha_xception=float(weights[0]),
                    alpha_ucf=float(weights[1]),
                    alpha_sbi=float(weights[2]),
                    beta=float(beta),
                    false_accepts=int(false_accepts[beta_index]),
                    false_rejections=int(false_rejections[beta_index]),
                    far=float(fars[beta_index]),
                    frr=float(frrs[beta_index]),
                    objective=float(objectives[beta_index]),
                )
            )

    # FAR+FRR is primary. Among exact ties, prefer the more balanced FAR/FRR,
    # then use parameter order for deterministic output.
    best = min(
        points,
        key=lambda point: (
            point.objective,
            abs(point.far - point.frr),
            point.alpha_xception,
            point.alpha_ucf,
            point.beta,
        ),
    )
    best_tie_count = sum(
        math.isclose(point.objective, best.objective, rel_tol=0.0, abs_tol=1e-15)
        for point in points
    )
    return ExhaustiveResult(
        dataset=data.dataset,
        step=step,
        interval_count=interval_count,
        alpha_combination_count=alpha_count,
        points=points,
        best=best,
        best_tie_count=best_tie_count,
    )
