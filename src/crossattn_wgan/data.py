"""Standalone merged-vol workbook parsing for the Cross-Attention WGAN module."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CrossAttnWGANSampleConfig, CrossAttnWGANTrainConfig

_VOL_FLOOR = 1e-4


def _parse_serialized_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [float(item) for item in parsed]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
    raise ValueError(f"Unsupported serialized list value: {value!r}")


def _parse_embedding(row: Any, mode: str) -> np.ndarray:
    normalized_mode = str(mode).strip().lower()
    hd = np.asarray(_parse_serialized_list(getattr(row, "hd_embedding", [])), dtype=np.float32)
    lp = np.asarray(_parse_serialized_list(getattr(row, "lp_embedding", [])), dtype=np.float32)
    if normalized_mode == "hd":
        return hd
    if normalized_mode == "lp":
        return lp
    if normalized_mode == "concat":
        return np.concatenate([hd, lp], axis=0).astype(np.float32)
    raise ValueError(f"text_embedding_mode must be one of ['hd', 'lp', 'concat'], got: {mode}")


@dataclass(frozen=True)
class CrossAttnWGANSample:
    """One chronological backward/current -> forward/future sample from merged_vol.xlsx."""

    sample_id: str
    global_index: int
    timestamp: str
    current_snapshot_time_utc: str
    target_snapshot_time_utc: str
    current_surface: np.ndarray
    target_surface: np.ndarray
    strike_grid: np.ndarray
    maturity_days_grid: np.ndarray
    text_embedding: np.ndarray
    metadata: dict[str, Any]

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.current_surface.shape


@dataclass(frozen=True)
class CrossAttnWGANNormalizationStats:
    """Feature statistics used to normalize Cross-Attention WGAN inputs and outputs."""

    current_log_mean: np.ndarray
    current_log_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray
    text_mean: np.ndarray
    text_std: np.ndarray


@dataclass(frozen=True)
class CrossAttnWGANDataBundle:
    """Train/validation loaders and metadata for the standalone Cross-Attention WGAN module."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    train_samples: int
    val_samples: int
    surface_shape: tuple[int, int]
    strike_grid: np.ndarray
    maturity_days_grid: np.ndarray
    embedding_dim: int
    normalization_stats: CrossAttnWGANNormalizationStats
    train_items: list[CrossAttnWGANSample]
    val_items: list[CrossAttnWGANSample]
    all_items: list[CrossAttnWGANSample]


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.std(values, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return std


def _compute_normalization_stats(samples: Sequence[CrossAttnWGANSample]) -> CrossAttnWGANNormalizationStats:
    current_logs: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    for sample in samples:
        current_flat = sample.current_surface.astype(np.float32).reshape(-1)
        target_flat = sample.target_surface.astype(np.float32).reshape(-1)
        current_log = np.log(np.clip(current_flat, _VOL_FLOOR, None))
        target_log = np.log(np.clip(target_flat, _VOL_FLOOR, None))
        current_logs.append(current_log.astype(np.float32))
        deltas.append((target_log - current_log).astype(np.float32))
        text_embeddings.append(sample.text_embedding.astype(np.float32))
    current_logs_arr = np.stack(current_logs, axis=0)
    deltas_arr = np.stack(deltas, axis=0)
    text_arr = np.stack(text_embeddings, axis=0)
    return CrossAttnWGANNormalizationStats(
        current_log_mean=np.mean(current_logs_arr, axis=0).astype(np.float32),
        current_log_std=_safe_std(current_logs_arr),
        delta_mean=np.mean(deltas_arr, axis=0).astype(np.float32),
        delta_std=_safe_std(deltas_arr),
        text_mean=np.mean(text_arr, axis=0).astype(np.float32),
        text_std=_safe_std(text_arr),
    )


def normalize_tensor(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values - mean) / torch.clamp(std, min=1e-6)


def denormalize_tensor(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return values * torch.clamp(std, min=1e-6) + mean


def normalize_surface_tensor(
    surface_flat: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    surface_log = torch.log(torch.clamp(surface_flat, min=_VOL_FLOOR))
    return normalize_tensor(surface_log, mean, std)


class CrossAttnWGANDataset(Dataset):
    """Torch dataset yielding the tensors needed by the standalone Cross-Attention WGAN trainer."""

    def __init__(
        self,
        samples: Sequence[CrossAttnWGANSample],
        normalization_stats: CrossAttnWGANNormalizationStats,
        *,
        normalize_current_surface: bool,
        normalize_target_delta: bool,
        normalize_text_embedding: bool,
    ):
        self.samples = list(samples)
        self.normalization_stats = normalization_stats
        self.normalize_current_surface = bool(normalize_current_surface)
        self.normalize_target_delta = bool(normalize_target_delta)
        self.normalize_text_embedding = bool(normalize_text_embedding)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        current_flat = sample.current_surface.astype(np.float32).reshape(-1)
        target_flat = sample.target_surface.astype(np.float32).reshape(-1)
        current_log = np.log(np.clip(current_flat, _VOL_FLOOR, None))
        target_log = np.log(np.clip(target_flat, _VOL_FLOOR, None))
        target_delta = (target_log - current_log).astype(np.float32)
        current_features = current_log.astype(np.float32)
        text_features = sample.text_embedding.astype(np.float32)
        target_delta_features = target_delta.astype(np.float32)
        stats = self.normalization_stats
        if self.normalize_current_surface:
            current_features = ((current_features - stats.current_log_mean) / stats.current_log_std).astype(np.float32)
        if self.normalize_text_embedding:
            text_features = ((text_features - stats.text_mean) / stats.text_std).astype(np.float32)
        if self.normalize_target_delta:
            target_delta_features = ((target_delta_features - stats.delta_mean) / stats.delta_std).astype(np.float32)
        height, width = sample.surface_shape
        return (
            torch.from_numpy(current_features.reshape(1, height, width)),
            torch.from_numpy(text_features),
            torch.from_numpy(target_delta_features),
            torch.from_numpy(current_flat),
            torch.from_numpy(target_flat),
        )


def _validate_shape(sample_id: str, surface: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if tuple(surface.shape) != tuple(expected_shape):
        raise ValueError(f"Sample {sample_id} surface shape mismatch: expected {expected_shape}, got {surface.shape}")


def load_crossattn_wgan_samples(config: CrossAttnWGANTrainConfig | CrossAttnWGANSampleConfig) -> list[CrossAttnWGANSample]:
    workbook_path = Path(config.data_path)
    dataframe = pd.read_excel(workbook_path, sheet_name=config.sheet_name)
    if "training_candidate_flag" in dataframe.columns:
        filtered = dataframe[dataframe["training_candidate_flag"].fillna(0).astype(int) == 1].copy()
        if not filtered.empty:
            dataframe = filtered
    if dataframe.empty:
        raise ValueError(f"No usable rows found in workbook {workbook_path} sheet {config.sheet_name}.")

    rows = list(dataframe.itertuples(index=False))
    parsed: list[CrossAttnWGANSample] = []
    for global_index, row in enumerate(rows):
        strike_grid = np.asarray(_parse_serialized_list(getattr(row, "strike_grid")), dtype=np.float32)
        maturity_days_grid = np.asarray(_parse_serialized_list(getattr(row, "maturity_days_grid")), dtype=np.float32)
        if strike_grid.size <= 0 or maturity_days_grid.size <= 0:
            raise ValueError("strike_grid and maturity_days_grid must be non-empty for standalone Cross-Attention WGAN.")
        surface_shape = (int(maturity_days_grid.size), int(strike_grid.size))
        current_flat = np.asarray(_parse_serialized_list(getattr(row, "current_surface_flat")), dtype=np.float32)
        target_flat = np.asarray(_parse_serialized_list(getattr(row, "target_surface_flat")), dtype=np.float32)
        current_surface = current_flat.reshape(surface_shape)
        target_surface = target_flat.reshape(surface_shape)
        _validate_shape(str(getattr(row, "sample_id")), current_surface, surface_shape)
        _validate_shape(str(getattr(row, "sample_id")), target_surface, surface_shape)
        parsed.append(
            CrossAttnWGANSample(
                sample_id=str(getattr(row, "sample_id")),
                global_index=global_index,
                timestamp=str(getattr(row, "news_timestamp_utc")),
                current_snapshot_time_utc=str(getattr(row, "current_snapshot_time_utc", "")),
                target_snapshot_time_utc=str(getattr(row, "target_snapshot_time_utc", "")),
                current_surface=current_surface,
                target_surface=target_surface,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
                text_embedding=_parse_embedding(row, config.text_embedding_mode),
                metadata={
                    "pair_quality_label": str(getattr(row, "pair_quality_label", "")),
                    "current_weighted_iv_rmse": getattr(row, "current_weighted_iv_rmse", None),
                    "target_weighted_iv_rmse": getattr(row, "target_weighted_iv_rmse", None),
                },
            )
        )

    parsed = sorted(parsed, key=lambda sample: pd.Timestamp(sample.timestamp))
    for index, sample in enumerate(parsed):
        parsed[index] = CrossAttnWGANSample(
            sample_id=sample.sample_id,
            global_index=index,
            timestamp=sample.timestamp,
            current_snapshot_time_utc=sample.current_snapshot_time_utc,
            target_snapshot_time_utc=sample.target_snapshot_time_utc,
            current_surface=sample.current_surface,
            target_surface=sample.target_surface,
            strike_grid=sample.strike_grid,
            maturity_days_grid=sample.maturity_days_grid,
            text_embedding=sample.text_embedding,
            metadata=sample.metadata,
        )
    return parsed


def _split_index(total_items: int, train_ratio: float) -> int:
    if total_items < 2:
        return total_items
    split_idx = int(total_items * float(train_ratio))
    split_idx = max(1, min(total_items - 1, split_idx))
    return split_idx


def _check_consistent_grids(samples: Sequence[CrossAttnWGANSample]) -> tuple[np.ndarray, np.ndarray]:
    strike_grid = samples[0].strike_grid
    maturity_days_grid = samples[0].maturity_days_grid
    for sample in samples[1:]:
        if sample.strike_grid.shape != strike_grid.shape or not np.allclose(sample.strike_grid, strike_grid):
            raise ValueError("All standalone Cross-Attention WGAN samples must share the same strike grid.")
        if sample.maturity_days_grid.shape != maturity_days_grid.shape or not np.allclose(
            sample.maturity_days_grid,
            maturity_days_grid,
        ):
            raise ValueError("All standalone Cross-Attention WGAN samples must share the same maturity grid.")
    return strike_grid.copy(), maturity_days_grid.copy()


def create_train_val_bundle(config: CrossAttnWGANTrainConfig) -> CrossAttnWGANDataBundle:
    samples = load_crossattn_wgan_samples(config)
    if len(samples) < int(config.min_samples_for_training):
        raise ValueError(f"Standalone Cross-Attention WGAN requires at least {config.min_samples_for_training} samples.")
    strike_grid, maturity_days_grid = _check_consistent_grids(samples)
    split_idx = _split_index(len(samples), config.train_ratio)
    train_items = list(samples[:split_idx])
    val_items = list(samples[split_idx:])
    normalization_stats = _compute_normalization_stats(train_items)
    train_loader = DataLoader(
        CrossAttnWGANDataset(
            train_items,
            normalization_stats,
            normalize_current_surface=config.normalize_current_surface,
            normalize_target_delta=config.normalize_target_delta,
            normalize_text_embedding=config.normalize_text_embedding,
        ),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
    )
    val_loader = None
    if val_items:
        val_loader = DataLoader(
            CrossAttnWGANDataset(
                val_items,
                normalization_stats,
                normalize_current_surface=config.normalize_current_surface,
                normalize_target_delta=config.normalize_target_delta,
                normalize_text_embedding=config.normalize_text_embedding,
            ),
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
        )
    return CrossAttnWGANDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_samples=len(train_items),
        val_samples=len(val_items),
        surface_shape=train_items[0].surface_shape,
        strike_grid=strike_grid,
        maturity_days_grid=maturity_days_grid,
        embedding_dim=int(train_items[0].text_embedding.size),
        normalization_stats=normalization_stats,
        train_items=train_items,
        val_items=val_items,
        all_items=list(samples),
    )


def select_samples(samples: Sequence[CrossAttnWGANSample], *, selection_mode: str, selection_count: int) -> list[CrossAttnWGANSample]:
    normalized_mode = str(selection_mode).strip().lower()
    count = int(selection_count)
    items = list(samples)
    if normalized_mode == "all" or count <= 0:
        return items
    if normalized_mode == "head":
        return items[:count]
    if normalized_mode == "tail":
        return items[-count:]
    raise ValueError(f"selection_mode must be one of ['all', 'head', 'tail'], got: {selection_mode}")


def split_samples(config: CrossAttnWGANSampleConfig, samples: Sequence[CrossAttnWGANSample]) -> list[CrossAttnWGANSample]:
    split_idx = _split_index(len(samples), config.train_ratio)
    normalized_split = str(config.split).strip().lower()
    if normalized_split == "train":
        selected = list(samples[:split_idx])
    elif normalized_split == "val":
        selected = list(samples[split_idx:])
    elif normalized_split == "all":
        selected = list(samples)
    else:
        raise ValueError(f"split must be one of ['train', 'val', 'all'], got: {config.split}")
    return select_samples(selected, selection_mode=config.selection_mode, selection_count=config.selection_count)
