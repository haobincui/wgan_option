"""Training-curve plotting helpers for standalone FiLM WGAN."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MetricRow = Mapping[str, float | int]


def plot_training_curves(
    metrics_rows: Sequence[MetricRow],
    *,
    output_path: str | Path,
    title: str = "FiLM WGAN Training Curves",
) -> Path:
    """Render a multi-panel loss-curve figure from standalone FiLM WGAN metrics."""

    if not metrics_rows:
        raise ValueError("metrics_rows must not be empty.")

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(metrics_rows)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_groups: list[tuple[str, tuple[str, ...], str]] = [
        ("Adversarial", ("g_total", "g_adv", "d_total", "gp"), "loss"),
        ("Constraints", ("g_calendar", "g_butterfly", "g_smooth", "g_recon"), "penalty"),
        ("Validation vs Current", ("val_mae", "val_current_mae", "val_rmse", "val_current_rmse"), "metric"),
        (
            "Gap and Distances",
            ("val_mae_gap_vs_current", "val_win_rate_vs_current", "val_generated_current_mae", "val_real_current_mae"),
            "metric",
        ),
        ("Arbitrage and Weighting", ("val_calendar", "val_butterfly", "val_penalty_mean", "val_penalty_std", "val_weight_entropy"), "metric"),
    ]

    fig, axes = plt.subplots(len(metric_groups), 1, figsize=(12, 12), sharex=True)
    if len(metric_groups) == 1:
        axes = [axes]

    for ax, (group_title, metric_names, ylabel) in zip(axes, metric_groups):
        plotted = False
        for metric_name in metric_names:
            if not all(metric_name in row for row in metrics_rows):
                continue
            values = [float(row[metric_name]) for row in metrics_rows]
            ax.plot(epochs, values, linewidth=2.0, label=metric_name)
            plotted = True
        ax.set_title(group_title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", transform=ax.transAxes)

    axes[-1].set_xlabel("epoch")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
