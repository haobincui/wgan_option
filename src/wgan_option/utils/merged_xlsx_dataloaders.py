"""Dataloader builders and SVI normalization helpers for merged-xlsx workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from wgan_option.config import Config

from .merged_xlsx_parsing import _align_embeddings, _split_index
from .merged_xlsx_samples import load_svi_paired_samples, load_vol_surface_samples
from .merged_xlsx_types import SVI_FEATURE_ORDER, SviPairedSample, SviXlsxBundle, VolSurfaceXlsxBundle


def _build_svi_matrix_from_params(
    svi_params: Dict[str, List[float]],
    max_slices: int,
    sample_label: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    feature_lists = [np.asarray(svi_params[feature], dtype=np.float32) for feature in SVI_FEATURE_ORDER]
    lengths = {int(values.size) for values in feature_lists}
    if len(lengths) != 1:
        raise ValueError(f"SVI feature list lengths must match for {sample_label}")
    slice_count = int(lengths.pop()) if feature_lists else 0
    if slice_count <= 0 or slice_count > int(max_slices):
        raise ValueError(
            f"SVI slice count must be between 1 and {max_slices} for {sample_label}, got {slice_count}"
        )

    matrix = np.zeros((int(max_slices), len(SVI_FEATURE_ORDER)), dtype=np.float32)
    mask = np.zeros(int(max_slices), dtype=np.float32)
    for feature_idx, values in enumerate(feature_lists):
        matrix[:slice_count, feature_idx] = values
    mask[:slice_count] = 1.0
    return matrix, mask, slice_count


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

    samples = load_vol_surface_samples(config)
    strike_grid = samples[0].strike_grid.copy()
    maturity_grid_days = samples[0].maturity_grid_days.copy()
    current_surfaces = [sample.current_surface for sample in samples]
    target_surfaces = [sample.target_surface for sample in samples]
    embeddings = [sample.text_embedding for sample in samples]
    timestamps = [sample.timestamp for sample in samples]

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
        all_items=list(samples),
        train_items=list(samples[:train_samples]),
        val_items=list(samples[train_samples:]),
    )


def _fit_svi_normalization(train_samples: Sequence[SviPairedSample], max_slices: int) -> Dict[str, Any]:
    valid_rows: List[np.ndarray] = []
    for sample in train_samples:
        current_matrix, current_mask, _ = _build_svi_matrix_from_params(
            sample.current_svi,
            max_slices,
            sample_label=f"{sample.sample_id}:current",
        )
        future_matrix, future_mask, _ = _build_svi_matrix_from_params(
            sample.future_svi,
            max_slices,
            sample_label=f"{sample.sample_id}:future",
        )
        for matrix, mask in ((current_matrix, current_mask), (future_matrix, future_mask)):
            mask_bool = mask.astype(bool)
            if mask_bool.any():
                valid_rows.append(matrix[mask_bool])
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

    paired_samples = load_svi_paired_samples(config)
    split_idx = _split_index(len(paired_samples), config.train_ratio)
    normalization_stats = _fit_svi_normalization(paired_samples[:split_idx], int(config.max_slices))

    current_vectors: List[np.ndarray] = []
    text_vectors: List[np.ndarray] = []
    future_vectors: List[np.ndarray] = []
    future_masks: List[np.ndarray] = []
    future_counts: List[int] = []
    timestamps: List[str] = []

    for sample in paired_samples:
        current_matrix, current_mask, current_count = _build_svi_matrix_from_params(
            sample.current_svi,
            int(config.max_slices),
            sample_label=f"{sample.sample_id}:current",
        )
        future_matrix, future_mask, future_count = _build_svi_matrix_from_params(
            sample.future_svi,
            int(config.max_slices),
            sample_label=f"{sample.sample_id}:future",
        )
        normalized_current = _normalize_svi_matrix(current_matrix, current_mask, normalization_stats)
        normalized_future = _normalize_svi_matrix(future_matrix, future_mask, normalization_stats)
        current_vector = np.concatenate(
            [
                normalized_current.reshape(-1),
                current_mask.astype(np.float32),
                np.asarray([float(current_count) / float(config.max_slices)], dtype=np.float32),
            ],
            axis=0,
        )
        current_vectors.append(current_vector.astype(np.float32))
        text_vectors.append(sample.text_embedding)
        future_vectors.append(normalized_future.reshape(-1).astype(np.float32))
        future_masks.append(future_mask.astype(np.float32))
        future_counts.append(int(future_count) - 1)
        timestamps.append(sample.timestamp)

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
