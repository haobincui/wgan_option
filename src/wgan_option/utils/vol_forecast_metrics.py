"""Shared metrics for baseline-aware vol-surface forecasting experiments."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def mean_abs_error(predicted_surface, target_surface, support_mask=None) -> float:
    """Return mean absolute error, optionally restricted to supported cells."""

    predicted = np.asarray(predicted_surface, dtype=np.float32)
    target = np.asarray(target_surface, dtype=np.float32)
    absolute_error = np.abs(predicted - target)
    if support_mask is None:
        return float(np.mean(absolute_error))
    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim == absolute_error.ndim + 1 and mask.shape[0] == 1:
        mask = mask[0]
    try:
        mask = np.broadcast_to(mask, absolute_error.shape)
    except ValueError as exc:
        raise ValueError(
            f"support_mask shape {mask.shape} cannot cover error shape "
            f"{absolute_error.shape}"
        ) from exc
    if not bool(mask.any()):
        raise ValueError("support_mask must contain at least one supported cell.")
    return float(np.mean(absolute_error[mask]))


def summarize_baseline_aware_metrics(
    recon_values: Iterable[float],
    current_recon_values: Iterable[float],
    *,
    baseline_penalty_weight: float,
) -> dict[str, float]:
    """Aggregate val reconstruction metrics and compare them to the persistence baseline."""

    recon_array = np.asarray(list(recon_values), dtype=np.float32)
    current_array = np.asarray(list(current_recon_values), dtype=np.float32)
    val_recon = float(recon_array.mean()) if recon_array.size else 0.0
    val_current_recon = float(current_array.mean()) if current_array.size else 0.0
    val_baseline_gap = float(val_recon - val_current_recon)
    val_hybrid_score = float(
        val_recon + float(baseline_penalty_weight) * max(0.0, val_baseline_gap)
    )
    return {
        "val_recon": val_recon,
        "val_current_recon": val_current_recon,
        "val_baseline_gap": val_baseline_gap,
        "val_hybrid_score": val_hybrid_score,
    }


def resolve_monitor_metric(metrics: Mapping[str, float], monitor_metric: str) -> float:
    """Return the configured monitor metric or fail fast if it is unavailable."""

    metric_name = str(monitor_metric).strip()
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise ValueError(
            f"Requested best_checkpoint_metric='{metric_name}' is unavailable. "
            f"Available metrics: {available}"
        )
    return float(metrics[metric_name])
