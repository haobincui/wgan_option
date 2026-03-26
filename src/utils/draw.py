"""Plotting helpers for volatility-surface related data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


Number = Union[int, float]


class draw:
    """Draw 2D volatility curves from processed SVI parameter files."""

    def __init__(self, processed_dir: Union[str, Path] = "data/processed", days_in_year: int = 250):
        self.processed_dir = Path(processed_dir)
        if days_in_year <= 0:
            raise ValueError("days_in_year must be positive.")
        self.days_in_year = int(days_in_year)

    def _resolve_data_path(self, data_file: Union[str, Path]) -> Path:
        path = Path(data_file)
        if not path.is_absolute():
            path = self.processed_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Cannot find data file: {path}")
        return path

    @staticmethod
    def _to_request_list(time_to_maturity: Optional[Union[Number, Sequence[Number]]]) -> Optional[list[float]]:
        if time_to_maturity is None:
            return None
        if isinstance(time_to_maturity, (int, float)):
            return [float(time_to_maturity)]
        requests = [float(x) for x in time_to_maturity]
        if not requests:
            raise ValueError("time_to_maturity is provided but empty.")
        return requests

    @staticmethod
    def _select_maturity_indices(available_days: np.ndarray, requests: Optional[list[float]]) -> list[int]:
        if requests is None:
            return list(range(available_days.size))

        selected: list[int] = []
        seen = set()
        for requested_day in requests:
            idx = int(np.argmin(np.abs(available_days - requested_day)))
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
        return selected

    def plot_vol_surface_2d(
        self,
        data_file: Union[str, Path] = "minute_svi_params_smoke_1000.json",
        snapshot_time: Optional[str] = None,
        time_to_maturity: Optional[Union[Number, Sequence[Number]]] = None,
        moneyness_min: float = 0.7,
        moneyness_max: float = 1.3,
        num_points: int = 100,
        figsize: Tuple[float, float] = (10.0, 6.0),
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Tuple[Figure, Axes]:
        """Plot vol curves by TTM on a 2D chart (x=moneyness, y=vol).

        Args:
            data_file: JSON file in `data/processed`, keyed by minute timestamp.
            snapshot_time: Which minute to plot. If None, uses the latest minute key.
            time_to_maturity: Optional maturity day(s) filter; nearest available term is used.
            moneyness_min: Left bound of moneyness grid.
            moneyness_max: Right bound of moneyness grid.
            num_points: Number of moneyness points in each curve.
            figsize: Matplotlib figure size.
            save_path: Optional image output path.
            show: Whether to call `plt.show()`.
        """
        if moneyness_min <= 0:
            raise ValueError("moneyness_min must be positive.")
        if moneyness_max <= moneyness_min:
            raise ValueError("moneyness_max must be greater than moneyness_min.")
        if num_points < 2:
            raise ValueError("num_points must be >= 2.")

        data_path = self._resolve_data_path(data_file)
        with data_path.open("r", encoding="utf-8") as f:
            minute_to_params = json.load(f)

        if not isinstance(minute_to_params, dict) or not minute_to_params:
            raise ValueError(f"No SVI records found in: {data_path}")

        if snapshot_time is None:
            snapshot_time = sorted(minute_to_params.keys())[-1]
        if snapshot_time not in minute_to_params:
            raise KeyError(
                f"snapshot_time={snapshot_time} not in data. "
                f"Available range: {min(minute_to_params.keys())} -> {max(minute_to_params.keys())}"
            )

        params = minute_to_params[snapshot_time]
        required = ("business_days", "a", "b", "rho", "m", "sigma")
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(f"Missing required SVI fields in snapshot {snapshot_time}: {missing}")

        business_days = np.asarray(params["business_days"], dtype=np.float64)
        a = np.asarray(params["a"], dtype=np.float64)
        b = np.asarray(params["b"], dtype=np.float64)
        rho = np.asarray(params["rho"], dtype=np.float64)
        m = np.asarray(params["m"], dtype=np.float64)
        sigma = np.asarray(params["sigma"], dtype=np.float64)

        n = business_days.size
        if n == 0:
            raise ValueError(f"Snapshot {snapshot_time} has no term points.")
        for name, arr in (("a", a), ("b", b), ("rho", rho), ("m", m), ("sigma", sigma)):
            if arr.size != n:
                raise ValueError(
                    f"Length mismatch in snapshot {snapshot_time}: business_days={n}, {name}={arr.size}"
                )

        requests = self._to_request_list(time_to_maturity)
        selected_indices = self._select_maturity_indices(business_days, requests)
        if not selected_indices:
            raise ValueError("No maturities selected for plotting.")

        moneyness_grid = np.linspace(moneyness_min, moneyness_max, num_points, dtype=np.float64)
        log_moneyness = np.log(moneyness_grid)

        fig, ax = plt.subplots(figsize=figsize)
        for idx in selected_indices:
            day = max(float(business_days[idx]), 1e-8)
            sigma_i = max(float(sigma[idx]), 1e-8)
            total_var = a[idx] + b[idx] * (
                rho[idx] * (log_moneyness - m[idx]) + np.sqrt((log_moneyness - m[idx]) ** 2 + sigma_i * sigma_i)
            )
            total_var = np.maximum(total_var, 0.0)
            vol = np.sqrt(total_var * float(self.days_in_year) / day)
            ax.plot(moneyness_grid, vol, linewidth=2.0, label=f"TTM={int(round(day))}d")

        ax.set_xlabel("moneyness")
        ax.set_ylabel("vol")
        ax.set_title(f"Vol Surface Slices @ {snapshot_time}")
        ax.grid(alpha=0.3)
        ax.legend(title="time to maturity")
        fig.tight_layout()

        if save_path is not None:
            output = Path(save_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)

        if show:
            plt.show()
        return fig, ax
