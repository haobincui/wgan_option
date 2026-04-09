"""Plot generated vs real vol-surface comparisons from JSON payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _surface_array(raw_value: Any) -> Optional[np.ndarray]:
    if raw_value is None:
        return None
    array = np.asarray(raw_value, dtype=np.float32)
    if array.size == 0:
        return None
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D surface array, got shape {array.shape}")
    return array


def _heatmap(
    ax,
    surface: np.ndarray,
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
    title: str,
    cmap: str,
):
    image = ax.imshow(
        surface,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(
            float(strike_grid[0]),
            float(strike_grid[-1]),
            float(maturity_days_grid[0]),
            float(maturity_days_grid[-1]),
        ),
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Strike / Forward")
    ax.set_ylabel("Maturity (business days)")
    return image


def _nearest_atm_index(strike_grid: Sequence[float]) -> int:
    """Return the index of the strike closest to ATM=1.0."""

    strikes = np.asarray(strike_grid, dtype=np.float32)
    if strikes.size == 0:
        raise ValueError("strike_grid must not be empty.")
    return int(np.argmin(np.abs(strikes - 1.0)))


def _short_maturity_index(maturity_days_grid: Sequence[float]) -> int:
    """Return the shortest maturity slice index used for smile plots."""

    maturities = np.asarray(maturity_days_grid, dtype=np.float32)
    if maturities.size == 0:
        raise ValueError("maturity_days_grid must not be empty.")
    return 0


def _sidecar_output_path(output_path: str | Path, suffix: str) -> Path:
    """Build a sidecar PNG path that shares the base stem with the main output."""

    output = Path(output_path)
    return output.with_name(f"{output.stem}{suffix}{output.suffix}")


def _surface_series(
    *,
    current_surface: Optional[np.ndarray],
    generated_surface: Optional[np.ndarray],
    real_surface: Optional[np.ndarray],
) -> list[tuple[str, np.ndarray, str]]:
    series: list[tuple[str, np.ndarray, str]] = []
    if current_surface is not None:
        series.append(("Current", current_surface, "#1f77b4"))
    if generated_surface is not None:
        series.append(("Generated", generated_surface, "#d62728"))
    if real_surface is not None:
        series.append(("Real", real_surface, "#2ca02c"))
    return series


def _style_line_axis(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def _plot_line_sidecars(
    *,
    sample_id: str,
    output_path: str | Path,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
    current_surface: Optional[np.ndarray],
    generated_surface: Optional[np.ndarray],
    real_surface: Optional[np.ndarray],
) -> None:
    line_series = _surface_series(
        current_surface=current_surface,
        generated_surface=generated_surface,
        real_surface=real_surface,
    )
    if not line_series:
        return

    atm_idx = _nearest_atm_index(strike_grid)
    short_idx = _short_maturity_index(maturity_days_grid)
    atm_strike = float(strike_grid[atm_idx])
    smile_maturity = float(maturity_days_grid[short_idx])

    lines_output = _sidecar_output_path(output_path, "_lines")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    atm_ax, smile_ax = axes[0]

    for label, surface, color in line_series:
        atm_ax.plot(maturity_days_grid, surface[:, atm_idx], linewidth=2.0, label=label, color=color)
        smile_ax.plot(strike_grid, surface[short_idx, :], linewidth=2.0, label=label, color=color)

    _style_line_axis(
        atm_ax,
        xlabel="Maturity (business days)",
        ylabel="Implied Volatility",
        title=f"ATM Term Structure (strike={atm_strike:.4f})",
    )
    _style_line_axis(
        smile_ax,
        xlabel="Strike / Forward",
        ylabel="Implied Volatility",
        title=f"Short-Maturity Smile ({smile_maturity:.1f}d)",
    )
    fig.suptitle(f"{sample_id} Line Views")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(lines_output, dpi=150)
    plt.close(fig)

    atm_output = _sidecar_output_path(output_path, "_atm")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    for label, surface, color in line_series:
        ax.plot(maturity_days_grid, surface[:, atm_idx], linewidth=2.0, label=label, color=color)
    _style_line_axis(
        ax,
        xlabel="Maturity (business days)",
        ylabel="Implied Volatility",
        title=f"ATM Term Structure Only (strike={atm_strike:.4f})",
    )
    fig.suptitle(f"{sample_id} ATM View")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(atm_output, dpi=150)
    plt.close(fig)


def plot_surface_payload(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render the standard generate_result payload into a PNG comparison figure."""

    strike_grid = [float(value) for value in payload["strike_grid"]]
    maturity_days_grid = [float(value) for value in payload["maturity_days_grid"]]
    current_surface = _surface_array(payload.get("current_surface"))
    generated_surface = _surface_array(payload.get("generated_surface"))
    real_surface = _surface_array(payload.get("real_surface"))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if generated_surface is None:
        raise ValueError("payload must include generated_surface")

    panels: list[tuple[str, np.ndarray, str]] = []
    if current_surface is not None:
        panels.append(("Current", current_surface, "viridis"))
    panels.append(("Generated", generated_surface, "viridis"))
    if real_surface is not None:
        panels.append(("Real", real_surface, "viridis"))
        panels.append(("Generated - Real", generated_surface - real_surface, "coolwarm"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    axes_row = axes[0]
    for ax, (title, surface, cmap) in zip(axes_row, panels):
        image = _heatmap(
            ax,
            surface,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_days_grid,
            title=title,
            cmap=cmap,
        )
        fig.colorbar(image, ax=ax, shrink=0.8)

    metrics = payload.get("metrics", {})
    metric_parts = []
    for key in ("mae", "rmse", "max_abs"):
        if key in metrics:
            metric_parts.append(f"{key}={float(metrics[key]):.6f}")
    sample_id = str(payload.get("sample_id", "sample"))
    mode = str(payload.get("mode", ""))
    subtitle = " | ".join(part for part in [mode, *metric_parts] if part)
    fig.suptitle(sample_id if not subtitle else f"{sample_id}\n{subtitle}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output, dpi=150)
    plt.close(fig)

    _plot_line_sidecars(
        sample_id=sample_id,
        output_path=output,
        strike_grid=strike_grid,
        maturity_days_grid=maturity_days_grid,
        current_surface=current_surface,
        generated_surface=generated_surface,
        real_surface=real_surface,
    )
    return output


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a surface comparison payload JSON into a PNG.")
    parser.add_argument("--input-json", required=True, help="Path to a payload JSON generated by scripts/generate_result.")
    parser.add_argument("--output", default=None, help="Optional PNG output path. Defaults next to the input JSON.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv=None) -> Path:
    args = _parse_args(argv)
    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")
    return plot_surface_payload(payload, output_path)


if __name__ == "__main__":
    main()
