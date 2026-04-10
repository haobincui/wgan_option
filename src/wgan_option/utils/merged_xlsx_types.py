"""Types shared across merged-xlsx loaders, splits, and dataloader builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

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


@dataclass
class VolSurfaceSample:
    """One merged vol-surface workbook row prepared for inference or training."""

    sample_id: str
    timestamp: str
    current_snapshot_time_utc: str
    target_snapshot_time_utc: str
    current_surface: np.ndarray
    target_surface: np.ndarray
    text_embedding: np.ndarray
    strike_grid: np.ndarray
    maturity_grid_days: np.ndarray
    surface_shape: Tuple[int, int]
    global_index: int
    metadata: Dict[str, Any]


@dataclass
class SviPairedSample:
    """One backward/current -> forward/future SVI pair prepared for inference or training."""

    sample_id: str
    news_row_id: int
    timestamp: str
    current_timestamp_utc: str
    future_timestamp_utc: str
    text_embedding: np.ndarray
    current_svi: Dict[str, List[float]]
    future_svi: Dict[str, List[float]]
    global_index: int
    current_metadata: Dict[str, Any]
    future_metadata: Dict[str, Any]


@dataclass
class OrderedSplitSelection:
    """Chronological split metadata for ordered samples."""

    all_items: List[Any]
    train_items: List[Any]
    val_items: List[Any]
    selected_items: List[Any]
    split_idx: int
    requested_split: str
    effective_split: str
    fallback_to_all: bool
