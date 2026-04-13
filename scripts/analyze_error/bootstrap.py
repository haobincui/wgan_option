"""Bootstrap helpers for per-sample MSE analysis."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def bootstrap_mean_mse(
    mse_values: Sequence[float],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    chunk_size: int = 1024,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Bootstrap the mean sample-level MSE with percentile confidence intervals."""

    values = np.asarray(mse_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mse_values must contain at least one sample-level value.")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be > 0.")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")

    sample_count = int(values.size)
    replicate_count = int(bootstrap_samples)
    batch_size = max(1, min(int(chunk_size), replicate_count))
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(replicate_count, dtype=np.float64)

    for start in range(0, replicate_count, batch_size):
        batch = min(batch_size, replicate_count - start)
        indices = rng.integers(0, sample_count, size=(batch, sample_count))
        distribution[start : start + batch] = values[indices].mean(axis=1)

    alpha = 1.0 - float(confidence_level)
    ci_lower, ci_upper = np.quantile(distribution, [alpha / 2.0, 1.0 - alpha / 2.0])
    summary: dict[str, float | int] = {
        "sample_count": sample_count,
        "bootstrap_samples": replicate_count,
        "confidence_level": float(confidence_level),
        "observed_mean_mse": float(values.mean()),
        "bootstrap_mean": float(distribution.mean()),
        "bootstrap_std": float(distribution.std(ddof=0)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "seed": int(seed),
    }
    return summary, distribution


def bootstrap_from_error_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Run bootstrap directly from error CSV-style rows."""

    mse_values = [float(row["mse"]) for row in rows]
    return bootstrap_mean_mse(
        mse_values,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
