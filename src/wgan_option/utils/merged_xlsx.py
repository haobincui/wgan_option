"""Load merged xlsx datasets for vol-surface and SVI training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from wgan_option.config import Config

SVI_FEATURE_ORDER = ("business_days", "a", "b", "rho", "m", "sigma")


@dataclass
class VolSurfaceXlsxBundle:
    """Container for vol-surface train/validation loaders."""

    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    strike_grid: np.ndarray
    maturity_grid_days: np.ndarray
    embedding_dim: int
    train_samples: int
    val_samples: int
    timestamps: List[str]
    train_timestamps: List[str]
    val_timestamps: List[str]


@dataclass
class SviXlsxBundle:
    """Container for SVI train/validation loaders and normalization stats."""

    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    embedding_dim: int
    train_samples: int
    val_samples: int
    current_input_dim: int
    regression_dim: int
    max_slices: int
    normalization_stats: Dict[str, Any]
    train_timestamps: List[str]
    val_timestamps: List[str]


def _read_sheet(data_path: str, sheet_name: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook does not exist: {data_path}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def _parse_timestamp_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required timestamp column: {column}")
    return pd.to_datetime(frame[column], errors="coerce", utc=True)


def _split_index(total: int, train_ratio: float) -> int:
    if total <= 0:
        raise ValueError("At least one sample is required.")
    split_idx = max(1, int(total * float(train_ratio)))
    return min(split_idx, total)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value)) or pd.isna(value)


def _parse_serialized_list(raw_value: Any) -> List[float]:
    if isinstance(raw_value, list):
        return [float(value) for value in raw_value]
    if _is_missing(raw_value):
        return []
    if isinstance(raw_value, np.ndarray):
        return [float(value) for value in raw_value.tolist()]

    text = str(raw_value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        stripped = text.strip("[]")
        if not stripped:
            return []
        values = np.fromstring(stripped, sep=",", dtype=np.float32)
        return [float(value) for value in values.tolist()]
    if isinstance(parsed, list):
        return [float(value) for value in parsed]
    raise ValueError(f"Expected serialized list, got: {raw_value}")


def _parse_serialized_vector(raw_value: Any) -> np.ndarray:
    values = _parse_serialized_list(raw_value)
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _parse_surface_shape(raw_value: Any) -> Tuple[int, int]:
    shape_values = _parse_serialized_list(raw_value)
    if len(shape_values) != 2:
        raise ValueError(f"surface_shape must contain exactly 2 values, got: {raw_value}")
    return int(shape_values[0]), int(shape_values[1])


def _resolve_text_embedding(
    hd_raw: Any,
    lp_raw: Any,
    mode: str,
) -> np.ndarray:
    mode_text = str(mode).strip().lower()
    hd_vector = _parse_serialized_vector(hd_raw)
    lp_vector = _parse_serialized_vector(lp_raw)
    if mode_text == "hd":
        return hd_vector
    if mode_text == "lp":
        return lp_vector
    if mode_text == "concat":
        return np.concatenate([hd_vector, lp_vector], axis=0)
    raise ValueError(f"text_embedding_mode must be one of hd/lp/concat, got: {mode}")


def _align_embeddings(vectors: Sequence[np.ndarray], fallback_dim: int) -> Tuple[np.ndarray, int]:
    inferred_dim = max((int(vector.size) for vector in vectors), default=0)
    embedding_dim = inferred_dim if inferred_dim > 0 else int(fallback_dim)
    if embedding_dim <= 0:
        raise ValueError("Failed to infer a positive embedding dimension.")
    aligned = np.zeros((len(vectors), embedding_dim), dtype=np.float32)
    for idx, vector in enumerate(vectors):
        if vector.size == 0:
            continue
        width = min(int(vector.size), embedding_dim)
        aligned[idx, :width] = vector[:width]
    return aligned, embedding_dim


def _build_data_loaders(
    inputs: np.ndarray,
    embeddings: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    target_counts: Optional[np.ndarray] = None,
    target_masks: Optional[np.ndarray] = None,
    timestamps: Optional[Sequence[str]] = None,
) -> Tuple[DataLoader, Optional[DataLoader], int, int, List[str], List[str]]:
    total = int(inputs.shape[0])
    split_idx = _split_index(total, train_ratio)

    train_timestamps = list(timestamps[:split_idx]) if timestamps is not None else []
    val_timestamps = list(timestamps[split_idx:]) if timestamps is not None else []

    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    tensor_embeddings = torch.tensor(embeddings, dtype=torch.float32)
    tensor_targets = torch.tensor(targets, dtype=torch.float32)

    if target_counts is None:
        train_ds = TensorDataset(
            tensor_inputs[:split_idx],
            tensor_embeddings[:split_idx],
            tensor_targets[:split_idx],
        )
        val_ds = None
        if split_idx < total:
            val_ds = TensorDataset(
                tensor_inputs[split_idx:],
                tensor_embeddings[split_idx:],
                tensor_targets[split_idx:],
            )
    else:
        tensor_target_counts = torch.tensor(target_counts, dtype=torch.long)
        tensor_target_masks = torch.tensor(target_masks, dtype=torch.float32)
        train_ds = TensorDataset(
            tensor_inputs[:split_idx],
            tensor_embeddings[:split_idx],
            tensor_targets[:split_idx],
            tensor_target_masks[:split_idx],
            tensor_target_counts[:split_idx],
        )
        val_ds = None
        if split_idx < total:
            val_ds = TensorDataset(
                tensor_inputs[split_idx:],
                tensor_embeddings[split_idx:],
                tensor_targets[split_idx:],
                tensor_target_masks[split_idx:],
                tensor_target_counts[split_idx:],
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(batch_size), max(1, len(train_ds))),
        shuffle=True,
        num_workers=int(num_workers),
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=min(int(batch_size), len(val_ds)),
            shuffle=False,
            num_workers=int(num_workers),
        )

    return train_loader, val_loader, len(train_ds), 0 if val_ds is None else len(val_ds), train_timestamps, val_timestamps


def create_vol_surface_xlsx_dataloaders(config: Config) -> VolSurfaceXlsxBundle:
    """Create train/validation dataloaders from merged vol-surface xlsx rows."""

    df = _read_sheet(config.data_path, config.sheet_name)
    required_columns = {
        "news_timestamp_utc",
        "hd_embedding",
        "lp_embedding",
        "current_surface_flat",
        "target_surface_flat",
        "surface_shape",
        "strike_grid",
        "maturity_days_grid",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Vol workbook is missing required columns: {missing}")

    if "training_candidate_flag" in df.columns:
        df = df[df["training_candidate_flag"] == 1].copy()
    if df.empty:
        raise ValueError("No training-ready vol rows remain after filtering.")

    df = df.copy()
    df["_timestamp"] = _parse_timestamp_column(df, "news_timestamp_utc")
    df = df.sort_values(["_timestamp", "sample_id"], kind="stable").reset_index(drop=True)

    surface_shape = _parse_surface_shape(df.iloc[0]["surface_shape"])
    strike_grid = np.asarray(_parse_serialized_list(df.iloc[0]["strike_grid"]), dtype=np.float32)
    maturity_grid_days = np.asarray(_parse_serialized_list(df.iloc[0]["maturity_days_grid"]), dtype=np.float32)
    expected_cells = int(surface_shape[0] * surface_shape[1])

    current_surfaces: List[np.ndarray] = []
    target_surfaces: List[np.ndarray] = []
    embeddings: List[np.ndarray] = []
    timestamps: List[str] = []

    for row in df.itertuples(index=False):
        current_surface = np.asarray(_parse_serialized_list(row.current_surface_flat), dtype=np.float32)
        target_surface = np.asarray(_parse_serialized_list(row.target_surface_flat), dtype=np.float32)
        if int(current_surface.size) != expected_cells or int(target_surface.size) != expected_cells:
            raise ValueError(
                f"Surface length mismatch for sample {getattr(row, 'sample_id', '')}: "
                f"expected {expected_cells}, got {current_surface.size} and {target_surface.size}"
            )
        current_surfaces.append(current_surface.reshape(1, surface_shape[0], surface_shape[1]))
        target_surfaces.append(target_surface.reshape(1, surface_shape[0], surface_shape[1]))
        embeddings.append(_resolve_text_embedding(row.hd_embedding, row.lp_embedding, config.text_embedding_mode))
        timestamps.append(str(getattr(row, "news_timestamp_utc")))

    aligned_embeddings, embedding_dim = _align_embeddings(embeddings, fallback_dim=config.embedding_dim)
    current_array = np.stack(current_surfaces, axis=0).astype(np.float32)
    target_array = np.stack(target_surfaces, axis=0).astype(np.float32)
    train_loader, val_loader, train_samples, val_samples, train_timestamps, val_timestamps = _build_data_loaders(
        current_array,
        aligned_embeddings,
        target_array,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        timestamps=timestamps,
    )

    return VolSurfaceXlsxBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        strike_grid=strike_grid,
        maturity_grid_days=maturity_grid_days,
        embedding_dim=embedding_dim,
        train_samples=train_samples,
        val_samples=val_samples,
        timestamps=timestamps,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
    )


def _build_svi_matrix(row: pd.Series, max_slices: int) -> Tuple[np.ndarray, np.ndarray, int]:
    feature_lists = [np.asarray(_parse_serialized_list(row[f"svi_{feature}_list"]), dtype=np.float32) for feature in SVI_FEATURE_ORDER]
    lengths = {int(values.size) for values in feature_lists}
    if len(lengths) != 1:
        raise ValueError(f"SVI feature list lengths must match for news_row_id={row['news_row_id']}")
    slice_count = int(lengths.pop()) if feature_lists else 0
    if slice_count <= 0 or slice_count > int(max_slices):
        raise ValueError(
            f"SVI slice count must be between 1 and {max_slices} for news_row_id={row['news_row_id']}, got {slice_count}"
        )

    matrix = np.zeros((int(max_slices), len(SVI_FEATURE_ORDER)), dtype=np.float32)
    mask = np.zeros(int(max_slices), dtype=np.float32)
    for feature_idx, values in enumerate(feature_lists):
        matrix[:slice_count, feature_idx] = values
    mask[:slice_count] = 1.0
    return matrix, mask, slice_count


def _fit_svi_normalization(train_samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows: List[np.ndarray] = []
    for sample in train_samples:
        for key in ("current_matrix", "future_matrix"):
            matrix = np.asarray(sample[key], dtype=np.float32)
            mask = np.asarray(sample[f"{key.split('_')[0]}_mask"], dtype=np.float32).astype(bool)
            if mask.any():
                valid_rows.append(matrix[mask])
    if not valid_rows:
        raise ValueError("Cannot fit SVI normalization stats without valid train slices.")
    stacked = np.concatenate(valid_rows, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "feature_order": list(SVI_FEATURE_ORDER),
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }


def _normalize_svi_matrix(matrix: np.ndarray, mask: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    normalized = np.zeros_like(matrix, dtype=np.float32)
    if mask.any():
        normalized[mask.astype(bool)] = (matrix[mask.astype(bool)] - mean) / std
    return normalized


def create_svi_xlsx_dataloaders(config: Config) -> SviXlsxBundle:
    """Create train/validation dataloaders from merged SVI audit rows."""

    df = _read_sheet(config.data_path, config.sheet_name)
    required_columns = {
        "news_row_id",
        "direction",
        "training_candidate_flag",
        "news_timestamp_utc",
        "hd_embedding",
        "lp_embedding",
        "svi_business_days_list",
        "svi_a_list",
        "svi_b_list",
        "svi_rho_list",
        "svi_m_list",
        "svi_sigma_list",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"SVI workbook is missing required columns: {missing}")

    usable = df[df["training_candidate_flag"] == 1].copy()
    if usable.empty:
        raise ValueError("No usable SVI rows remain after filtering.")

    usable["_timestamp"] = _parse_timestamp_column(usable, "news_timestamp_utc")
    usable = usable.sort_values(["news_row_id", "_timestamp", "direction"], kind="stable")

    paired_samples: List[Dict[str, Any]] = []
    for news_row_id, group in usable.groupby("news_row_id", sort=False):
        directions = {str(direction): row for direction, row in group.set_index("direction").iterrows()}
        if "backward" not in directions or "forward" not in directions:
            continue

        current_row = directions["backward"]
        future_row = directions["forward"]
        current_matrix, current_mask, current_count = _build_svi_matrix(current_row, config.max_slices)
        future_matrix, future_mask, future_count = _build_svi_matrix(future_row, config.max_slices)
        paired_samples.append(
            {
                "sample_id": f"news_{int(news_row_id)}",
                "timestamp": str(current_row["news_timestamp_utc"]),
                "embedding": _resolve_text_embedding(
                    current_row.get("hd_embedding", ""),
                    current_row.get("lp_embedding", ""),
                    config.text_embedding_mode,
                ),
                "current_matrix": current_matrix,
                "current_mask": current_mask,
                "current_count": current_count,
                "future_matrix": future_matrix,
                "future_mask": future_mask,
                "future_count": future_count,
            }
        )

    if not paired_samples:
        raise ValueError("No paired backward/forward SVI samples are available for training.")

    paired_samples = sorted(paired_samples, key=lambda sample: pd.Timestamp(sample["timestamp"]))
    split_idx = _split_index(len(paired_samples), config.train_ratio)
    normalization_stats = _fit_svi_normalization(paired_samples[:split_idx])

    current_vectors: List[np.ndarray] = []
    text_vectors: List[np.ndarray] = []
    future_vectors: List[np.ndarray] = []
    future_masks: List[np.ndarray] = []
    future_counts: List[int] = []
    timestamps: List[str] = []

    for sample in paired_samples:
        normalized_current = _normalize_svi_matrix(sample["current_matrix"], sample["current_mask"], normalization_stats)
        normalized_future = _normalize_svi_matrix(sample["future_matrix"], sample["future_mask"], normalization_stats)
        current_vector = np.concatenate(
            [
                normalized_current.reshape(-1),
                sample["current_mask"].astype(np.float32),
                np.asarray([float(sample["current_count"]) / float(config.max_slices)], dtype=np.float32),
            ],
            axis=0,
        )
        current_vectors.append(current_vector.astype(np.float32))
        text_vectors.append(sample["embedding"])
        future_vectors.append(normalized_future.reshape(-1).astype(np.float32))
        future_masks.append(sample["future_mask"].astype(np.float32))
        future_counts.append(int(sample["future_count"]) - 1)
        timestamps.append(sample["timestamp"])

    aligned_embeddings, embedding_dim = _align_embeddings(text_vectors, fallback_dim=config.embedding_dim)
    current_array = np.stack(current_vectors, axis=0).astype(np.float32)
    future_array = np.stack(future_vectors, axis=0).astype(np.float32)
    future_mask_array = np.stack(future_masks, axis=0).astype(np.float32)
    future_count_array = np.asarray(future_counts, dtype=np.int64)

    train_loader, val_loader, train_samples, val_samples, train_timestamps, val_timestamps = _build_data_loaders(
        current_array,
        aligned_embeddings,
        future_array,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        target_counts=future_count_array,
        target_masks=future_mask_array,
        timestamps=timestamps,
    )

    return SviXlsxBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        embedding_dim=embedding_dim,
        train_samples=train_samples,
        val_samples=val_samples,
        current_input_dim=int(current_array.shape[1]),
        regression_dim=int(future_array.shape[1]),
        max_slices=int(config.max_slices),
        normalization_stats=normalization_stats,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
    )
