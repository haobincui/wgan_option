"""Plot histogram outputs for analyze-error runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_mse_histogram(
    error_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    bins: int,
) -> Path:
    """Plot the sample-level MSE distribution from error rows."""

    if int(bins) <= 0:
        raise ValueError("bins must be > 0.")

    mse_values = np.asarray([float(row["mse"]) for row in error_rows], dtype=np.float64)
    if mse_values.size == 0:
        raise ValueError("error_rows must contain at least one MSE value.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    mean_value = float(mse_values.mean())
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    ax.hist(mse_values, bins=int(bins), color="#4c78a8", edgecolor="white", alpha=0.9)
    ax.axvline(mean_value, color="#d62728", linewidth=2.0, linestyle="--", label=f"mean={mean_value:.6f}")
    ax.set_title("Sample-level MSE Distribution")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_bootstrap_mean_histogram(
    distribution: Sequence[float],
    summary: Mapping[str, Any],
    output_path: str | Path,
    bins: int,
) -> Path:
    """Plot the bootstrap mean-MSE distribution with observed mean and CI markers."""

    if int(bins) <= 0:
        raise ValueError("bins must be > 0.")

    values = np.asarray(distribution, dtype=np.float64)
    if values.size == 0:
        raise ValueError("distribution must contain at least one value.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    observed_mean = float(summary["observed_mean_mse"])
    ci_lower = float(summary["ci_lower"])
    ci_upper = float(summary["ci_upper"])

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    ax.hist(values, bins=int(bins), color="#72b7b2", edgecolor="white", alpha=0.9)
    ax.axvline(
        observed_mean,
        color="#d62728",
        linewidth=2.0,
        linestyle="-",
        label=f"observed_mean={observed_mean:.6f}",
    )
    ax.axvline(ci_lower, color="#2ca02c", linewidth=2.0, linestyle="--", label=f"ci_lower={ci_lower:.6f}")
    ax.axvline(ci_upper, color="#9467bd", linewidth=2.0, linestyle="--", label=f"ci_upper={ci_upper:.6f}")
    ax.set_title("Bootstrap Mean MSE Distribution")
    ax.set_xlabel("Bootstrap Mean MSE")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
