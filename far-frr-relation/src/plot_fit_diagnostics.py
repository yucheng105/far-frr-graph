"""Plotting utilities for FAR/FRR fit diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_fit_diagnostic(
    output_path: Path,
    title: str,
    thresholds: np.ndarray,
    far_values: np.ndarray,
    frr_values: np.ndarray,
    far_fitted: np.ndarray,
    frr_fitted: np.ndarray,
) -> None:
    """Write a diagnostic plot without clipping negative fitted values."""
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(thresholds, far_values, s=8, alpha=0.35, label="FAR observed")
    axis.plot(thresholds, far_fitted, linewidth=2, label="FAR exponential fit")
    axis.scatter(thresholds, frr_values, s=8, alpha=0.35, label="FRR observed")
    axis.plot(thresholds, frr_fitted, linewidth=2, label="FRR exponential fit")
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set(
        xlabel="Threshold",
        ylabel="Error rate",
        title=title,
        xlim=(float(np.min(thresholds)), float(np.max(thresholds))),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
