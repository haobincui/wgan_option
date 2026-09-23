"""Types shared across merged-xlsx loaders, splits, and dataloader builders."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    all_items: List["VolSurfaceSample"]
    train_items: List["VolSurfaceSample"]
    val_items: List["VolSurfaceSample"]
    test_loader: Optional[DataLoader] = None
    test_samples: int = 0
    test_timestamps: List[str] = field(default_factory=list)
    test_items: List["VolSurfaceSample"] = field(default_factory=list)
    uses_sample_weights: bool = False
    uses_label_reliability_weights: bool = False
    uses_support_masks: bool = False
    uses_current_support_masks: bool = False
    split_metadata: Dict[str, Any] = field(default_factory=dict)


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
    news_row_id: Optional[int] = None
    pair_id: str = ""
    session_id: str = ""
    effective_origin_utc: str = ""
    sample_weight: float = 1.0
    label_reliability_weight: float = 1.0
    stable_sample_key: str = ""
    # ``support_mask`` is the future-aware current∩target mask used only for
    # losses/evaluation.  ``current_support_mask`` is derived exclusively from
    # the current raw-surface parameters and is safe to use for conditioning.
    support_mask: Optional[np.ndarray] = None
    current_support_mask: Optional[np.ndarray] = None
    support_grid_fingerprint: str = ""
    support_mask_fingerprint: str = ""
    current_support_mask_fingerprint: str = ""


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
