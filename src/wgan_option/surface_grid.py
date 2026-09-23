"""Shared fixed-grid defaults for reconstructed volatility surfaces."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

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
    integer_maturity_days: bool = False,
    maturity_days_nodes: Sequence[float] | None = None,
    dtype: Any = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the shared fixed grid used by merge/result/analysis workflows.

    ``integer_maturity_days`` is opt-in so existing persisted datasets retain
    their historical floating-point linspace.  When enabled, the maturity
    linspace is rounded once and the exact integer nodes are returned for both
    surface evaluation and serialization.  ``maturity_days_nodes`` provides an
    explicit, strictly increasing axis while leaving the legacy linspace path
    unchanged when omitted.
    """

    strike_grid = np.linspace(
        float(moneyness_min),
        float(moneyness_max),
        int(strike_bins),
        dtype=dtype,
    )
    explicit_maturity_nodes = maturity_days_nodes is not None
    if not explicit_maturity_nodes:
        maturity_days_grid = np.linspace(
            int(maturity_min_days),
            int(maturity_max_days),
            int(maturity_bins),
            dtype=np.float64 if integer_maturity_days else dtype,
        )
    else:
        if isinstance(maturity_days_nodes, (str, bytes)) or any(
            isinstance(value, (bool, np.bool_)) for value in maturity_days_nodes
        ):
            raise ValueError("maturity_days_nodes must be a numeric sequence")
        maturity_days_grid = np.asarray(list(maturity_days_nodes), dtype=np.float64)
        if maturity_days_grid.ndim != 1 or maturity_days_grid.size != int(
            maturity_bins
        ):
            raise ValueError(
                "maturity_days_nodes must contain exactly maturity_bins values"
            )
        if not np.all(np.isfinite(maturity_days_grid)) or np.any(
            maturity_days_grid <= 0.0
        ):
            raise ValueError("maturity_days_nodes must contain finite positive values")
        if np.any(np.diff(maturity_days_grid) <= 0.0):
            raise ValueError("maturity_days_nodes must be strictly increasing")
    if integer_maturity_days:
        if explicit_maturity_nodes and not np.allclose(
            maturity_days_grid,
            np.rint(maturity_days_grid),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "integer maturity_days_nodes must contain whole business days"
            )
        maturity_days_grid = np.rint(maturity_days_grid)
        if np.unique(maturity_days_grid).size != int(maturity_bins):
            raise ValueError(
                "Rounded integer maturity grid contains duplicate business-day "
                "nodes; reduce maturity_bins or widen the maturity range."
            )
        maturity_days_grid = maturity_days_grid.astype(dtype, copy=False)
    return strike_grid, maturity_days_grid


def build_surface_grids_from_config(
    config: Any, *, dtype: Any = np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the shared fixed grid from a config object with matching field names."""

    return build_surface_grids(
        strike_bins=int(config.strike_bins),
        maturity_bins=int(config.maturity_bins),
        moneyness_min=float(config.moneyness_min),
        moneyness_max=float(config.moneyness_max),
        maturity_min_days=int(config.maturity_min_days),
        maturity_max_days=int(config.maturity_max_days),
        integer_maturity_days=bool(getattr(config, "integer_maturity_days", False)),
        maturity_days_nodes=getattr(config, "maturity_days_nodes", None),
        dtype=dtype,
    )


def default_surface_shape() -> list[int]:
    """Return the canonical `[rows, cols]` shape for serialized grid payloads."""

    return surface_shape()


def surface_shape(
    *,
    strike_bins: int = DEFAULT_STRIKE_BINS,
    maturity_bins: int = DEFAULT_MATURITY_BINS,
) -> list[int]:
    """Return the serialized `[rows, cols]` shape for a given grid definition."""

    return [int(maturity_bins), int(strike_bins)]
