"""Build daily strike-maturity surface tensors from raw option trades."""

import calendar
import glob
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_RIC_PATTERN = re.compile(
    r"^(?P<root>[A-Z0-9#\+]+?)(?P<strike>\d+(?:\.\d+)?)(?P<month_code>[A-X])(?P<year_digit>\d)$"
)


@dataclass
class OptionSurfaceBuilderConfig:
    """Configuration for raw-data filtering and output grid resolution."""

    option_data_glob: str = "data/raw/option_data/*.csv.gz"
    strike_bins: int = 16
    maturity_bins: int = 16
    moneyness_min: float = 0.5
    moneyness_max: float = 1.5
    maturity_min_days: int = 1
    maturity_max_days: int = 365


def _decode_strike(raw_strike: str) -> float:
    """Decode strike token from RIC-like symbol."""
    if "." in raw_strike:
        return float(raw_strike)
    digits = raw_strike.strip()
    if len(digits) <= 3:
        return float(int(digits))
    scale = 10 ** (len(digits) - 3)
    return float(int(digits)) / float(scale)


def _option_month_and_type(month_code: str) -> Tuple[int, str]:
    """Map month code to expiry month and call/put marker."""
    if "A" <= month_code <= "L":
        return ord(month_code) - ord("A") + 1, "C"
    if "M" <= month_code <= "X":
        return ord(month_code) - ord("M") + 1, "P"
    raise ValueError(f"Unsupported month code: {month_code}")


def _resolve_year_digit(trade_year: int, year_digit: int) -> int:
    """Resolve one-digit year token to nearest plausible full year."""
    decade = (trade_year // 10) * 10
    candidate = decade + year_digit
    if candidate < trade_year - 5:
        candidate += 10
    if candidate > trade_year + 5:
        candidate -= 10
    return candidate


def _fill_missing_surface(surface: np.ndarray) -> np.ndarray:
    """Fill sparse grids via linear interpolation in both axes."""
    filled = surface.copy()
    h, w = filled.shape

    if np.isnan(filled).all():
        return np.zeros_like(filled, dtype=np.float32)

    for i in range(h):
        row = filled[i]
        mask = ~np.isnan(row)
        if mask.any():
            x = np.arange(w)
            filled[i] = np.interp(x, x[mask], row[mask])

    for j in range(w):
        col = filled[:, j]
        mask = ~np.isnan(col)
        if mask.any():
            x = np.arange(h)
            filled[:, j] = np.interp(x, x[mask], col[mask])

    if np.isnan(filled).any():
        mean_value = np.nanmean(filled)
        filled[np.isnan(filled)] = mean_value if np.isfinite(mean_value) else 0.0
    return filled.astype(np.float32)


class OptionSurfaceBuilder:
    """Construct and persist daily 2D surfaces from option trade history."""

    def __init__(self, config: OptionSurfaceBuilderConfig):
        self.config = config

    def _load_option_trades(self) -> pd.DataFrame:
        """Load and normalize raw option records from configured glob."""
        frames = []
        for path in sorted(glob.glob(self.config.option_data_glob)):
            frame = pd.read_csv(path, usecols=["#RIC", "Date-Time", "Price", "Volume"])
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No option files found for pattern: {self.config.option_data_glob}")

        df = pd.concat(frames, ignore_index=True)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(1.0)
        df = df[df["Price"].notna() & (df["Price"] > 0)].copy()
        df["trade_dt"] = pd.to_datetime(df["Date-Time"], errors="coerce", utc=True)
        df = df[df["trade_dt"].notna()].copy()
        df["trade_date"] = df["trade_dt"].dt.date

        parsed = df["#RIC"].astype(str).str.extract(_RIC_PATTERN)
        df = pd.concat([df, parsed], axis=1)
        df = df[df["strike"].notna()].copy()
        if df.empty:
            raise ValueError("No parsable option symbols found in #RIC.")

        df["strike_val"] = df["strike"].map(_decode_strike)
        df["month_code"] = df["month_code"].astype(str)
        df["year_digit"] = df["year_digit"].astype(int)

        expiries = []
        option_types = []
        for trade_date, month_code, year_digit in zip(df["trade_date"], df["month_code"], df["year_digit"]):
            expiry_month, option_type = _option_month_and_type(month_code)
            expiry_year = _resolve_year_digit(trade_date.year, int(year_digit))
            expiry_day = calendar.monthrange(expiry_year, expiry_month)[1]
            expiry_date = date(expiry_year, expiry_month, expiry_day)
            if expiry_date <= trade_date:
                adjusted_year = expiry_year + 1
                adjusted_day = calendar.monthrange(adjusted_year, expiry_month)[1]
                expiry_date = date(adjusted_year, expiry_month, adjusted_day)
            expiries.append(expiry_date)
            option_types.append(option_type)

        df["expiry_date"] = expiries
        df["option_type"] = option_types
        df["maturity_days"] = (
            pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["trade_date"])
        ).dt.days
        df = df[
            (df["maturity_days"] >= self.config.maturity_min_days)
            & (df["maturity_days"] <= self.config.maturity_max_days)
        ].copy()
        if df.empty:
            raise ValueError("No rows left after maturity filtering.")
        return df

    def build_surface_stack(
        self,
    ) -> Tuple[np.ndarray, List[date], np.ndarray, np.ndarray]:
        """Return `(surface_stack, dates, strike_grid, maturity_grid)` arrays."""
        option_df = self._load_option_trades()

        all_moneyness = []
        all_maturity = []
        grouped = list(option_df.groupby("trade_date"))
        for _, frame in grouped:
            strike_ref = float(np.median(frame["strike_val"].values))
            if strike_ref <= 0:
                continue
            local = frame.copy()
            local["moneyness"] = local["strike_val"] / strike_ref
            local = local[
                (local["moneyness"] >= self.config.moneyness_min)
                & (local["moneyness"] <= self.config.moneyness_max)
            ]
            if local.empty:
                continue
            all_moneyness.append(local["moneyness"].values)
            all_maturity.append(local["maturity_days"].values)

        if not all_moneyness:
            raise ValueError("Unable to build surface: no valid moneyness points.")

        m_concat = np.concatenate(all_moneyness)
        t_concat = np.concatenate(all_maturity)
        m_low, m_high = np.quantile(m_concat, [0.05, 0.95])
        t_low, t_high = np.quantile(t_concat, [0.05, 0.95])

        strike_grid = np.linspace(
            max(self.config.moneyness_min, float(m_low)),
            min(self.config.moneyness_max, float(m_high)),
            self.config.strike_bins,
            dtype=np.float32,
        )
        maturity_grid = np.linspace(
            max(self.config.maturity_min_days, int(t_low)),
            min(self.config.maturity_max_days, int(t_high)),
            self.config.maturity_bins,
            dtype=np.float32,
        )

        surfaces: Dict[date, np.ndarray] = {}
        for trade_date, frame in grouped:
            strike_ref = float(np.median(frame["strike_val"].values))
            if strike_ref <= 0:
                continue
            local = frame.copy()
            local["moneyness"] = local["strike_val"] / strike_ref
            local = local[
                (local["moneyness"] >= strike_grid.min())
                & (local["moneyness"] <= strike_grid.max())
                & (local["maturity_days"] >= maturity_grid.min())
                & (local["maturity_days"] <= maturity_grid.max())
            ]
            if local.empty:
                continue

            h, w = maturity_grid.size, strike_grid.size
            value_sum = np.zeros((h, w), dtype=np.float64)
            weight_sum = np.zeros((h, w), dtype=np.float64)

            m_idx = np.abs(local["moneyness"].to_numpy()[:, None] - strike_grid[None, :]).argmin(axis=1)
            t_idx = np.abs(local["maturity_days"].to_numpy()[:, None] - maturity_grid[None, :]).argmin(axis=1)
            prices = local["Price"].to_numpy(dtype=np.float64)
            volumes = np.clip(local["Volume"].to_numpy(dtype=np.float64), 1.0, None)

            for i in range(prices.size):
                ii, jj = t_idx[i], m_idx[i]
                value_sum[ii, jj] += prices[i] * volumes[i]
                weight_sum[ii, jj] += volumes[i]

            with np.errstate(divide="ignore", invalid="ignore"):
                surface = value_sum / weight_sum
            surface[weight_sum <= 0] = np.nan
            surfaces[trade_date] = _fill_missing_surface(surface)

        if not surfaces:
            raise ValueError("No surfaces were generated.")

        dates = sorted(surfaces.keys())
        surface_stack = np.stack([surfaces[d] for d in dates], axis=0).astype(np.float32)
        return surface_stack, dates, strike_grid, maturity_grid

    def save(
        self,
        output_dir: str,
        save_daily_csv: bool = True,
        npz_name: str = "vol_surface_stack.npz",
    ) -> Path:
        """Persist surface stack NPZ and optionally per-day CSV snapshots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        surface_stack, dates, strike_grid, maturity_grid = self.build_surface_stack()
        np.savez_compressed(
            output_path / npz_name,
            surface_stack=surface_stack,
            dates=np.array([d.isoformat() for d in dates]),
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )

        if save_daily_csv:
            daily_dir = output_path / "daily_csv"
            daily_dir.mkdir(parents=True, exist_ok=True)
            for i, day in enumerate(dates):
                df = pd.DataFrame(
                    surface_stack[i],
                    index=[f"tau_{int(x)}d" for x in maturity_grid],
                    columns=[f"k_{x:.4f}" for x in strike_grid],
                )
                df.to_csv(daily_dir / f"surface_{day.isoformat()}.csv")
        return output_path / npz_name
