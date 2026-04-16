"""Plot generated vs real vol-surface comparisons from JSON payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    return int(np.argmin(maturities))


def extract_short_end_atm_band_value(
    surface: np.ndarray,
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
    atm_range: float,
    short_end_max_days: float,
) -> dict[str, Any]:
    """Extract one short-end near-ATM volatility scalar from one surface."""

    surface_array = _surface_array(surface)
    if surface_array is None:
        raise ValueError("surface must not be empty.")

    strikes = np.asarray(strike_grid, dtype=np.float32)
    maturities = np.asarray(maturity_days_grid, dtype=np.float32)
    if surface_array.shape != (maturities.size, strikes.size):
        raise ValueError(
            "surface shape must match maturity_days_grid x strike_grid, "
            f"got {surface_array.shape} vs {(maturities.size, strikes.size)}"
        )

    strike_mask = np.abs(strikes - 1.0) <= float(atm_range) + 1e-6
    maturity_mask = maturities <= float(short_end_max_days) + 1e-6
    band_mask = np.outer(maturity_mask, strike_mask)
    if np.any(band_mask):
        band_values = surface_array[band_mask]
        return {
            "value": float(band_values.mean()),
            "point_count": int(band_values.size),
            "selection": "band_mean",
        }

    short_idx = _short_maturity_index(maturities)
    atm_idx = _nearest_atm_index(strikes)
    return {
        "value": float(surface_array[short_idx, atm_idx]),
        "point_count": 1,
        "selection": "nearest_cell_fallback",
    }


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


def _parse_timestamp_utc(value: Any) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("news_timestamp_utc must not be empty.")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def plot_short_end_atm_band_timeseries(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    atm_range: float,
    short_end_max_days: float,
) -> Path:
    """Render one run-level short-end near-ATM volatility time-series plot."""

    if not rows:
        raise ValueError("rows must not be empty.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        rows,
        key=lambda row: (str(row.get("news_timestamp_utc", "")), int(row.get("global_index", -1))),
    )
    timestamps = [_parse_timestamp_utc(row["news_timestamp_utc"]) for row in ordered_rows]

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.8))
    series_specs = [
        ("Current", "short_atm_band_current_vol", "#1f77b4"),
        ("Generated Future", "short_atm_band_generated_future_vol", "#d62728"),
        ("Real Future", "short_atm_band_real_future_vol", "#2ca02c"),
    ]
    for label, key, color in series_specs:
        ax.plot(
            timestamps,
            [float(row[key]) for row in ordered_rows],
            linewidth=2.0,
            marker="o",
            markersize=4.0,
            label=label,
            color=color,
        )

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
    _style_line_axis(
        ax,
        xlabel="News Timestamp (UTC)",
        ylabel="Implied Volatility",
        title="Short-End ATM Band",
    )
    fig.suptitle(
        "Short-End ATM Vol Time Series\n"
        f"|K/F - 1| <= {float(atm_range):.2f}, maturity <= {float(short_end_max_days):.0f}d | "
        "x-axis = news_timestamp_utc | Generated/Real = future surfaces"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


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
