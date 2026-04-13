"""Data preparation pipeline for conditional WGAN option-surface training."""

import calendar
import datetime as dt
import glob
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from wgan_option.config import Config

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RIC_PATTERN = re.compile(
    r"^(?P<root>[A-Z0-9#\+]+?)(?P<strike>\d+(?:\.\d+)?)(?P<month_code>[A-X])(?P<year_digit>\d)$"
)


@dataclass
class ForecastDataBundle:
    """Container for training/validation dataloaders and related metadata."""

    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    strike_grid: np.ndarray
    maturity_grid_days: np.ndarray
    embedding_dim: int
    train_samples: int
    val_samples: int
    dates: List[dt.date]


def _parse_cell_text(cell: ET.Element) -> str:
    """Extract display text from a worksheet cell node."""
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text_nodes = cell.findall(f".//{_NS}t")
        return "".join((node.text or "") for node in text_nodes).strip()

    value_node = cell.find(f"{_NS}v")
    if value_node is None or value_node.text is None:
        return ""
    return value_node.text.strip()


def _column_from_ref(ref: str) -> str:
    """Return Excel column token (e.g. 'AB') from cell reference."""
    match = re.match(r"([A-Z]+)", ref or "")
    return match.group(1) if match else ""


def _decode_strike(raw_strike: str) -> float:
    """Decode strike token from symbol to numeric strike value."""
    if "." in raw_strike:
        return float(raw_strike)
    digits = raw_strike.strip()
    if len(digits) <= 3:
        return float(int(digits))
    scale = 10 ** (len(digits) - 3)
    return float(int(digits)) / float(scale)


def _option_month_and_type(month_code: str) -> Tuple[int, str]:
    """Map RIC month code to expiry month and option type."""
    if "A" <= month_code <= "L":
        return ord(month_code) - ord("A") + 1, "C"
    if "M" <= month_code <= "X":
        return ord(month_code) - ord("M") + 1, "P"
    raise ValueError(f"Unsupported month code: {month_code}")


def _resolve_year_digit(trade_year: int, year_digit: int) -> int:
    """Resolve one-digit year code to nearest full year around trade year."""
    decade = (trade_year // 10) * 10
    candidate = decade + year_digit
    if candidate < trade_year - 5:
        candidate += 10
    if candidate > trade_year + 5:
        candidate -= 10
    return candidate


def _parse_news_date(value: str) -> Optional[dt.date]:
    """Parse date text from embedding file using common formats."""
    if not value:
        return None
    for fmt in ("%d %B %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return dt.datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_embedding_vector(raw_text: str) -> Optional[np.ndarray]:
    """Parse serialized embedding array text into float32 vector."""
    text = (raw_text or "").strip()
    if not text:
        return None
    stripped = text.strip("[]")
    if not stripped:
        return None
    values = np.fromstring(stripped, sep=",", dtype=np.float32)
    if values.size == 0:
        return None
    return values


def _load_daily_news_embeddings(
    xlsx_path: str,
    date_column: str,
    embedding_column: str,
    allowed_dates: Optional[set] = None,
) -> Tuple[Dict[dt.date, np.ndarray], int]:
    """Load and aggregate per-day news embeddings from xlsx source."""
    daily_vectors: Dict[dt.date, List[np.ndarray]] = {}
    embedding_dim = 0
    path = Path(xlsx_path)
    if not path.exists():
        warnings.warn(f"News embedding file not found: {xlsx_path}. Text embedding will be zero vectors.")
        return {}, embedding_dim

    with zipfile.ZipFile(path) as workbook:
        with workbook.open("xl/worksheets/sheet1.xml") as sheet_file:
            headers: Dict[str, str] = {}
            for _, row in ET.iterparse(sheet_file, events=("end",)):
                if row.tag != f"{_NS}row":
                    continue

                row_idx = int(row.attrib.get("r", "0"))
                cells = row.findall(f"{_NS}c")

                if row_idx == 1:
                    for cell in cells:
                        col = _column_from_ref(cell.attrib.get("r", ""))
                        headers[col] = _parse_cell_text(cell)
                    row.clear()
                    continue

                date_value: Optional[dt.date] = None
                embedding_text = ""
                for cell in cells:
                    col = _column_from_ref(cell.attrib.get("r", ""))
                    header = headers.get(col, "")
                    if header == date_column:
                        date_value = _parse_news_date(_parse_cell_text(cell))
                    elif header == embedding_column:
                        embedding_text = _parse_cell_text(cell)

                if date_value is None:
                    row.clear()
                    continue
                if allowed_dates is not None and date_value not in allowed_dates:
                    row.clear()
                    continue

                vector = _parse_embedding_vector(embedding_text)
                if vector is None:
                    row.clear()
                    continue

                embedding_dim = max(embedding_dim, int(vector.size))
                daily_vectors.setdefault(date_value, []).append(vector)
                row.clear()

    aggregated: Dict[dt.date, np.ndarray] = {}
    for day, vectors in daily_vectors.items():
        if not vectors:
            continue
        target_dim = max(vec.size for vec in vectors)
        aligned = []
        for vec in vectors:
            if vec.size < target_dim:
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[: vec.size] = vec
                aligned.append(padded)
            else:
                aligned.append(vec[:target_dim])
        aggregated[day] = np.mean(np.stack(aligned, axis=0), axis=0).astype(np.float32)
    return aggregated, embedding_dim


def _load_option_trades(config: Config) -> pd.DataFrame:
    """Load and normalize raw option trade files referenced by config."""
    frames = []
    for path in sorted(glob.glob(config.option_data_glob)):
        frame = pd.read_csv(path, usecols=["#RIC", "Date-Time", "Price", "Volume"])
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No option files found for pattern: {config.option_data_glob}")

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

    df["strike_val"] = df["strike"].map(_decode_strike)
    df["month_code"] = df["month_code"].astype(str)
    df["year_digit"] = df["year_digit"].astype(int)

    expiries = []
    option_types = []
    for trade_date, month_code, year_digit in zip(df["trade_date"], df["month_code"], df["year_digit"]):
        expiry_month, option_type = _option_month_and_type(month_code)
        expiry_year = _resolve_year_digit(trade_date.year, int(year_digit))
        expiry_day = calendar.monthrange(expiry_year, expiry_month)[1]
        expiry_date = dt.date(expiry_year, expiry_month, expiry_day)
        if expiry_date <= trade_date:
            # Conservative fallback for symbols near month boundaries.
            adjusted_year = expiry_year + 1
            adjusted_day = calendar.monthrange(adjusted_year, expiry_month)[1]
            expiry_date = dt.date(adjusted_year, expiry_month, adjusted_day)
        expiries.append(expiry_date)
        option_types.append(option_type)

    df["expiry_date"] = expiries
    df["option_type"] = option_types
    df["maturity_days"] = (pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["trade_date"])).dt.days
    df = df[df["maturity_days"] > 0].copy()
    df = df[df["maturity_days"] <= config.maturity_max_days].copy()
    return df


def _fill_missing_surface(surface: np.ndarray) -> np.ndarray:
    """Interpolate missing cells first row-wise then column-wise."""
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


def _build_daily_surface_tensors(config: Config, option_df: pd.DataFrame):
    """Build normalized daily proxy-vol surfaces on fixed strike/maturity grids."""
    if option_df.empty:
        raise ValueError("No option observations available after parsing and filtering.")

    date_to_surface: Dict[dt.date, np.ndarray] = {}
    all_moneyness = []
    all_maturity = []

    for trade_date, frame in option_df.groupby("trade_date"):
        strike_ref = float(np.median(frame["strike_val"].values))
        if strike_ref <= 0:
            continue
        local = frame.copy()
        local["moneyness"] = local["strike_val"] / strike_ref
        local = local[
            (local["moneyness"] >= config.moneyness_min)
            & (local["moneyness"] <= config.moneyness_max)
            & (local["maturity_days"] >= config.maturity_min_days)
            & (local["maturity_days"] <= config.maturity_max_days)
        ]
        if local.empty:
            continue
        all_moneyness.append(local["moneyness"].values)
        all_maturity.append(local["maturity_days"].values)

    if not all_moneyness:
        raise ValueError("Unable to build any surface after moneyness/maturity filtering.")

    moneyness_concat = np.concatenate(all_moneyness)
    maturity_concat = np.concatenate(all_maturity)
    m_low, m_high = np.quantile(moneyness_concat, [0.05, 0.95])
    t_low, t_high = np.quantile(maturity_concat, [0.05, 0.95])

    strike_grid = np.linspace(
        max(config.moneyness_min, float(m_low)),
        min(config.moneyness_max, float(m_high)),
        config.strike_bins,
        dtype=np.float32,
    )
    maturity_grid = np.linspace(
        max(config.maturity_min_days, int(t_low)),
        min(config.maturity_max_days, int(t_high)),
        config.maturity_bins,
        dtype=np.float32,
    )

    for trade_date, frame in option_df.groupby("trade_date"):
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
        date_to_surface[trade_date] = _fill_missing_surface(surface)

    if not date_to_surface:
        raise ValueError("No daily surfaces were produced from the option dataset.")

    dates = sorted(date_to_surface.keys())
    surfaces = np.stack([date_to_surface[d] for d in dates], axis=0)
    low, high = np.quantile(surfaces, [0.01, 0.99])
    if not np.isfinite(high - low) or (high - low) < 1e-8:
        scaled = np.zeros_like(surfaces, dtype=np.float32)
    else:
        scaled = np.clip((surfaces - low) / (high - low), 0.0, 1.0).astype(np.float32)
    vol_like_surfaces = config.vol_floor + scaled * (config.vol_cap - config.vol_floor)
    return dates, vol_like_surfaces.astype(np.float32), strike_grid, maturity_grid


def _build_supervised_samples(
    config: Config,
    dates: List[dt.date],
    surfaces: np.ndarray,
    embedding_by_day: Dict[dt.date, np.ndarray],
    embedding_dim: int,
):
    """Create (today_surface, today_text, future_surface) supervised tuples."""
    x_surface, x_text, y_surface = [], [], []
    horizon = max(1, int(config.prediction_horizon))

    if embedding_dim <= 0:
        embedding_dim = config.embedding_dim

    for idx in range(len(dates) - horizon):
        today = dates[idx]
        future_idx = idx + horizon
        target_day = dates[future_idx]

        current_surface = surfaces[idx]
        target_surface = surfaces[future_idx]
        text_vec = embedding_by_day.get(today, None)
        if text_vec is None:
            text_vec = np.zeros(embedding_dim, dtype=np.float32)
        elif text_vec.size < embedding_dim:
            padded = np.zeros(embedding_dim, dtype=np.float32)
            padded[: text_vec.size] = text_vec
            text_vec = padded
        elif text_vec.size > embedding_dim:
            text_vec = text_vec[:embedding_dim]

        x_surface.append(current_surface.astype(np.float32))
        x_text.append(text_vec.astype(np.float32))
        y_surface.append(target_surface.astype(np.float32))

        _ = target_day  # kept for readability in case of future extension

    if len(x_surface) < config.min_samples_for_training:
        warnings.warn(
            f"Only {len(x_surface)} supervised samples available. "
            f"Consider adding more option dates and overlapping news dates."
        )

    return (
        np.asarray(x_surface, dtype=np.float32),
        np.asarray(x_text, dtype=np.float32),
        np.asarray(y_surface, dtype=np.float32),
    )


def create_bond_option_forecast_dataloaders(config: Config) -> ForecastDataBundle:
    """Create train/validation dataloaders, optionally backed by on-disk cache."""
    cache_path = Path(config.processed_cache_path)
    if config.use_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        dates = [dt.datetime.strptime(d, "%Y-%m-%d").date() for d in cached["dates"]]
        x_surface = cached["x_surface"]
        x_text = cached["x_text"]
        y_surface = cached["y_surface"]
        strike_grid = cached["strike_grid"]
        maturity_grid = cached["maturity_grid"]
        embedding_dim = int(cached["embedding_dim"])
    else:
        option_df = _load_option_trades(config)
        dates, surfaces, strike_grid, maturity_grid = _build_daily_surface_tensors(config, option_df)
        embedding_by_day, inferred_embedding_dim = _load_daily_news_embeddings(
            xlsx_path=config.news_embedding_path,
            date_column=config.news_date_column,
            embedding_column=config.embedding_column,
            allowed_dates=set(dates),
        )
        embedding_dim = inferred_embedding_dim if inferred_embedding_dim > 0 else config.embedding_dim
        x_surface, x_text, y_surface = _build_supervised_samples(
            config=config,
            dates=dates,
            surfaces=surfaces,
            embedding_by_day=embedding_by_day,
            embedding_dim=embedding_dim,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "x_surface": x_surface,
                "x_text": x_text,
                "y_surface": y_surface,
                "strike_grid": strike_grid,
                "maturity_grid": maturity_grid,
                "embedding_dim": embedding_dim,
            },
            cache_path,
        )

    if x_surface.shape[0] == 0:
        raise ValueError("No supervised samples were created from the provided data.")

    x_surface_tensor = torch.tensor(x_surface, dtype=torch.float32).unsqueeze(1)
    x_text_tensor = torch.tensor(x_text, dtype=torch.float32)
    y_surface_tensor = torch.tensor(y_surface, dtype=torch.float32).unsqueeze(1)

    split_idx = max(1, int(len(x_surface_tensor) * config.train_ratio))
    split_idx = min(split_idx, len(x_surface_tensor))

    train_ds = TensorDataset(
        x_surface_tensor[:split_idx],
        x_text_tensor[:split_idx],
        y_surface_tensor[:split_idx],
    )
    val_ds = None
    if split_idx < len(x_surface_tensor):
        val_ds = TensorDataset(
            x_surface_tensor[split_idx:],
            x_text_tensor[split_idx:],
            y_surface_tensor[split_idx:],
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(config.batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=min(config.batch_size, len(val_ds)),
            shuffle=False,
            num_workers=config.num_workers,
        )

    return ForecastDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        strike_grid=np.asarray(strike_grid, dtype=np.float32),
        maturity_grid_days=np.asarray(maturity_grid, dtype=np.float32),
        embedding_dim=int(embedding_dim),
        train_samples=len(train_ds),
        val_samples=0 if val_ds is None else len(val_ds),
        dates=dates,
    )


def create_dataloader(raw_data: np.ndarray, batch_size=64, shuffle=True):
    """Compatibility helper: wrap a 3D numpy array into DataLoader."""
    data_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_random_dataloader(number_of_data: int, x_length: int, y_length: int, batch_size=64, shuffle=True):
    """Utility helper for synthetic shape checks and smoke tests."""
    np_data = np.random.rand(number_of_data, x_length, y_length)
    data_tensor = torch.tensor(np_data, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
