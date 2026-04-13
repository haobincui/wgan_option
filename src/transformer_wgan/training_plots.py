"""Training-curve plotting helpers for standalone Transformer WGAN."""

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
    title: str = "Transformer WGAN Training Curves",
) -> Path:
    """Render a multi-panel loss-curve figure from standalone Transformer WGAN metrics."""

    if not metrics_rows:
        raise ValueError("metrics_rows must not be empty.")

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(metrics_rows)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metric_groups: list[tuple[str, tuple[str, ...], str]] = [
        ("Adversarial", ("g_total", "g_adv", "d_total", "gp"), "loss"),
        ("Reconstruction", ("g_recon", "g_delta_shrink", "val_recon", "val_current_recon"), "metric"),
        ("Constraints", ("g_calendar", "g_butterfly", "g_smooth", "val_calendar", "val_butterfly", "val_delta_shrink"), "penalty"),
        ("Baseline Aware", ("val_baseline_gap", "val_hybrid_score"), "metric"),
    ]

    fig, axes = plt.subplots(len(metric_groups), 1, figsize=(12, 10), sharex=True)
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
