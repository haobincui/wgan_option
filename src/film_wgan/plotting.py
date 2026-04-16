"""Standalone plotting helpers for FiLM WGAN generate-result outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _to_surface(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.size <= 0:
        return None
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D surface array, got shape {array.shape}")
    return array


def _heatmap(ax, surface: np.ndarray, *, strike_grid: Sequence[float], maturity_days_grid: Sequence[float], title: str, cmap: str):
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
    ax.set_ylabel("Maturity (days)")
    return image


def _atm_index(strike_grid: Sequence[float]) -> int:
    strikes = np.asarray(strike_grid, dtype=np.float32)
    return int(np.argmin(np.abs(strikes - 1.0)))


def _short_idx(maturity_days_grid: Sequence[float]) -> int:
    maturities = np.asarray(maturity_days_grid, dtype=np.float32)
    return int(np.argmin(maturities))


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


def extract_atm_short_value(
    surface: Any,
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> dict[str, float | int]:
    """Extract the nearest-ATM value at the shortest maturity row."""

    surface_array = _to_surface(surface)
    if surface_array is None:
        raise ValueError("surface must not be empty.")

    strikes = np.asarray(strike_grid, dtype=np.float32)
    maturities = np.asarray(maturity_days_grid, dtype=np.float32)
    expected_shape = (int(maturities.size), int(strikes.size))
    if tuple(surface_array.shape) != expected_shape:
        raise ValueError(f"Expected surface shape {expected_shape}, got {surface_array.shape}")

    atm_idx = _atm_index(strikes)
    short_idx = _short_idx(maturities)
    return {
        "value": float(surface_array[short_idx, atm_idx]),
        "atm_index": int(atm_idx),
        "short_index": int(short_idx),
        "atm_strike": float(strikes[atm_idx]),
        "short_maturity_days": float(maturities[short_idx]),
    }


def plot_atm_vol_timeseries(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Plot one run-level nearest-ATM shortest-maturity volatility time series."""

    if not rows:
        raise ValueError("rows must not be empty.")

    ordered_rows = sorted(
        rows,
        key=lambda row: (str(row.get("news_timestamp_utc", "")), int(row.get("global_index", -1))),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [_parse_timestamp_utc(row["news_timestamp_utc"]) for row in ordered_rows]

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.8))
    for label, key, color in [
        ("Current", "current_atm_vol", "#1f77b4"),
        ("Generated", "generated_atm_vol", "#d62728"),
        ("Target", "target_atm_vol", "#2ca02c"),
    ]:
        ax.plot(
            timestamps,
            [float(row[key]) for row in ordered_rows],
            label=label,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.0,
        )

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
    ax.set_xlabel("News Timestamp (UTC)")
    ax.set_ylabel("Implied Volatility")
    ax.grid(alpha=0.3)
    ax.legend()

    first_row = ordered_rows[0]
    fig.suptitle(
        "Nearest-ATM / Shortest-Maturity Vol Time Series\n"
        f"nearest strike to 1.0 = {float(first_row['atm_strike']):.4f} | "
        f"shortest maturity = {float(first_row['short_maturity_days']):.1f}d"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_film_wgan_payload(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render one standalone FiLM WGAN generate-result payload into PNG files."""

    strike_grid = [float(value) for value in payload["strike_grid"]]
    maturity_days_grid = [float(value) for value in payload["maturity_days_grid"]]
    current_surface = _to_surface(payload.get("current_surface"))
    generated_surface = _to_surface(payload.get("generated_surface"))
    target_surface = _to_surface(payload.get("target_surface"))

    if generated_surface is None:
        raise ValueError("payload must include generated_surface")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray, str]] = []
    if current_surface is not None:
        panels.append(("Current", current_surface, "viridis"))
    panels.append(("Generated", generated_surface, "viridis"))
    if target_surface is not None:
        panels.append(("Target", target_surface, "viridis"))
        panels.append(("Generated - Target", generated_surface - target_surface, "coolwarm"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.25 * len(panels), 4.5), squeeze=False)
    for ax, (title, surface, cmap) in zip(axes[0], panels):
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
    sample_id = str(payload.get("sample_id", "sample"))
    summary_bits = [
        f"mae={float(metrics['mae']):.6f}" if "mae" in metrics else "",
        f"rmse={float(metrics['rmse']):.6f}" if "rmse" in metrics else "",
        f"max_abs={float(metrics['max_abs']):.6f}" if "max_abs" in metrics else "",
    ]
    summary = " | ".join(bit for bit in summary_bits if bit)
    fig.suptitle(sample_id if not summary else f"{sample_id}\n{summary}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output, dpi=150)
    plt.close(fig)

    if current_surface is None and target_surface is None:
        return output

    atm_idx = _atm_index(strike_grid)
    short_idx = _short_idx(maturity_days_grid)
    line_output = output.with_name(f"{output.stem}_lines{output.suffix}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    term_ax, smile_ax = axes[0]
    for label, surface, color in [
        ("Current", current_surface, "#1f77b4"),
        ("Generated", generated_surface, "#d62728"),
        ("Target", target_surface, "#2ca02c"),
    ]:
        if surface is None:
            continue
        term_ax.plot(maturity_days_grid, surface[:, atm_idx], label=label, color=color, linewidth=2.0)
        smile_ax.plot(strike_grid, surface[short_idx, :], label=label, color=color, linewidth=2.0)
    term_ax.set_title(f"ATM Term Structure (strike={strike_grid[atm_idx]:.4f})")
    term_ax.set_xlabel("Maturity (days)")
    term_ax.set_ylabel("Implied Volatility")
    term_ax.grid(alpha=0.3)
    term_ax.legend()
    smile_ax.set_title(f"Short-Maturity Smile ({maturity_days_grid[short_idx]:.1f}d)")
    smile_ax.set_xlabel("Strike / Forward")
    smile_ax.set_ylabel("Implied Volatility")
    smile_ax.grid(alpha=0.3)
    smile_ax.legend()
    fig.tight_layout()
    fig.savefig(line_output, dpi=150)
    plt.close(fig)
    return output
