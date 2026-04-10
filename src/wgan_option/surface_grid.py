"""Shared fixed-grid defaults for reconstructed volatility surfaces."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

DEFAULT_STRIKE_BINS = 16
DEFAULT_MATURITY_BINS = 16
DEFAULT_MONEYNESS_MIN = 0.7
DEFAULT_MONEYNESS_MAX = 1.3
DEFAULT_MATURITY_MIN_DAYS = 7
DEFAULT_MATURITY_MAX_DAYS = 365


def build_surface_grids(
    *,
    strike_bins: int = DEFAULT_STRIKE_BINS,
    maturity_bins: int = DEFAULT_MATURITY_BINS,
    moneyness_min: float = DEFAULT_MONEYNESS_MIN,
    moneyness_max: float = DEFAULT_MONEYNESS_MAX,
    maturity_min_days: int = DEFAULT_MATURITY_MIN_DAYS,
    maturity_max_days: int = DEFAULT_MATURITY_MAX_DAYS,
    dtype: Any = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the shared fixed grid used by merge/result/analysis workflows."""

    strike_grid = np.linspace(
        float(moneyness_min),
        float(moneyness_max),
        int(strike_bins),
        dtype=dtype,
    )
    maturity_days_grid = np.linspace(
        int(maturity_min_days),
        int(maturity_max_days),
        int(maturity_bins),
        dtype=dtype,
    )
    return strike_grid, maturity_days_grid


def build_surface_grids_from_config(config: Any, *, dtype: Any = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Build the shared fixed grid from a config object with matching field names."""

    return build_surface_grids(
        strike_bins=int(config.strike_bins),
        maturity_bins=int(config.maturity_bins),
        moneyness_min=float(config.moneyness_min),
        moneyness_max=float(config.moneyness_max),
        maturity_min_days=int(config.maturity_min_days),
        maturity_max_days=int(config.maturity_max_days),
        dtype=dtype,
    )


def default_surface_shape() -> list[int]:
    """Return the canonical `[rows, cols]` shape for serialized grid payloads."""

    return surface_shape()


def surface_shape(*, strike_bins: int = DEFAULT_STRIKE_BINS, maturity_bins: int = DEFAULT_MATURITY_BINS) -> list[int]:
    """Return the serialized `[rows, cols]` shape for a given grid definition."""

    return [int(maturity_bins), int(strike_bins)]
