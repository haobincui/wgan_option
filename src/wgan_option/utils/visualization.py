from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MetricRow = Mapping[str, float | int]

def save_images(gen_imgs, epoch, n_row=3, output_dir='outputs/samples'):
    """Saves a grid of generated digits ranging from 0 to n_row**2"""
    fig, axs = plt.subplots(n_row, n_row)
    cnt = 0
    for i in range(n_row):
        for j in range(n_row):
            axs[i, j].imshow(gen_imgs[cnt, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()


def plot_training_curves(
    metrics_rows: Sequence[MetricRow],
    *,
    title: str,
    metric_groups: Sequence[tuple[str, Sequence[str]]],
    output_path: str | Path,
) -> Path:
    """Save a multi-panel training-curve figure from epoch metric rows.

    Missing metric names are skipped so the helper works for train-only runs
    and for training paths with different metric schemas.
    """
    if not metrics_rows:
        raise ValueError("metrics_rows must not be empty.")

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(metrics_rows)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(metric_groups), 1, figsize=(12, 8), sharex=True)
    if len(metric_groups) == 1:
        axes = [axes]

    for ax, (group_title, metric_names) in zip(axes, metric_groups):
        plotted = False
        for metric_name in metric_names:
            if not all(metric_name in row for row in metrics_rows):
                continue
            values = [float(row[metric_name]) for row in metrics_rows]
            ax.plot(epochs, values, linewidth=2.0, label=metric_name)
            plotted = True

        ax.set_title(group_title)
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No metrics available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    axes[-1].set_xlabel("epoch")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
