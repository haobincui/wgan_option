"""Rolling-history NPZ data interface for intraday raw-vol FiLM-WGAN runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import FilmWGANSampleConfig, FilmWGANTrainConfig
from .text_transform import sha256_file


_VOL_FLOOR = 1e-4
_REQUIRED_ARRAYS = {
    "history_surfaces",
    "history_support_masks",
    "history_step_masks",
    "target_surfaces",
    "target_support_masks",
    "matched_text_features",
    "count_text_features",
    "shuffled_text_features",
    "has_news",
    "strike_grid",
    "maturity_days_grid",
}
_REQUIRED_MANIFEST_COLUMNS = {
    "global_index",
    "sample_id",
    "surface_pair_id",
    "split",
    "current_snapshot_time_utc",
    "target_snapshot_time_utc",
    "history_anchor_times_json",
    "history_valid_steps",
    "supported_cell_count",
    "short_atm_supported_cell_count",
    "short_atm_eligible",
    "trading_day",
}


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _text_array_name(variant: str) -> str | None:
    normalized = str(variant).strip().lower()
    return {
        "no_text": None,
        "news_count": "count_text_features",
        "matched_lp": "matched_text_features",
        "shuffled_lp": "shuffled_text_features",
    }[normalized]


def _manifest_for_config(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
) -> pd.DataFrame:
    path = _resolved_path(config.split_manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sequence split manifest does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(_REQUIRED_MANIFEST_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Sequence split manifest is missing columns {missing}: {path}")
    frame = frame.copy()
    frame["global_index"] = pd.to_numeric(
        frame["global_index"], errors="raise"
    ).astype(int)
    frame["history_valid_steps"] = pd.to_numeric(
        frame["history_valid_steps"], errors="raise"
    ).astype(int)
    for column in (
        "supported_cell_count",
        "short_atm_supported_cell_count",
        "short_atm_eligible",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    if frame["global_index"].tolist() != list(range(len(frame))):
        raise ValueError(
            "Sequence split manifest global_index must be contiguous and ordered."
        )
    invalid_splits = sorted(set(frame["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Sequence split manifest contains invalid splits: {invalid_splits}")
    return frame


def load_sequence_samples(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
):
    """Load one fold-specific sequence bundle without fitting any transforms."""

    from .data import FilmWGANSample

    data_path = _resolved_path(config.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Sequence NPZ does not exist: {data_path}")
    manifest = _manifest_for_config(config)
    variant = str(config.sequence_text_variant).strip().lower()
    text_array_name = _text_array_name(variant)
    with np.load(data_path, allow_pickle=False) as payload:
        missing = sorted(_REQUIRED_ARRAYS - set(payload.files))
        if missing:
            raise ValueError(f"Sequence NPZ is missing arrays {missing}: {data_path}")
        arrays = {
            name: np.asarray(payload[name])
            for name in _REQUIRED_ARRAYS
        }

    history = np.asarray(arrays["history_surfaces"], dtype=np.float32)
    history_support = np.asarray(arrays["history_support_masks"], dtype=bool)
    history_steps = np.asarray(arrays["history_step_masks"], dtype=bool)
    targets = np.asarray(arrays["target_surfaces"], dtype=np.float32)
    target_support = np.asarray(arrays["target_support_masks"], dtype=bool)
    has_news = np.asarray(arrays["has_news"], dtype=np.float32).reshape(-1)
    strike_grid = np.asarray(arrays["strike_grid"], dtype=np.float32).reshape(-1)
    maturity_grid = np.asarray(arrays["maturity_days_grid"], dtype=np.float32).reshape(-1)
    sample_count = len(manifest)
    expected_history_shape = (
        sample_count,
        int(config.history_steps),
        maturity_grid.size,
        strike_grid.size,
    )
    if history.shape != expected_history_shape:
        raise ValueError(
            f"Sequence history shape mismatch: expected {expected_history_shape}, "
            f"found {history.shape}."
        )
    if history_support.shape != history.shape:
        raise ValueError("Sequence history support shape does not match history surfaces.")
    if history_steps.shape != (sample_count, int(config.history_steps)):
        raise ValueError("Sequence history step-mask shape is invalid.")
    expected_target_shape = (sample_count, maturity_grid.size, strike_grid.size)
    if targets.shape != expected_target_shape or target_support.shape != targets.shape:
        raise ValueError("Sequence target arrays have invalid shapes.")
    if has_news.size != sample_count:
        raise ValueError("Sequence has_news length mismatch.")
    if not np.all(np.isin(has_news, (0.0, 1.0))):
        raise ValueError("Sequence has_news must be binary.")
    if not np.all(history_steps[:, -1]):
        raise ValueError("Every rolling-history sample must contain the latest surface.")
    valid_counts = history_steps.sum(axis=1)
    if np.any(valid_counts < int(config.history_min_valid_steps)):
        raise ValueError("Sequence NPZ contains samples below history_min_valid_steps.")
    recorded_counts = manifest["history_valid_steps"].to_numpy(dtype=int)
    if not np.array_equal(valid_counts.astype(int), recorded_counts):
        raise ValueError("Sequence manifest history counts do not match the NPZ masks.")
    evaluation_support = history_support[:, -1] & target_support
    supported_counts = evaluation_support.sum(axis=(1, 2)).astype(int)
    if np.any(supported_counts < 1):
        raise ValueError("Sequence NPZ contains an empty current-target support intersection.")
    if not np.array_equal(
        supported_counts,
        manifest["supported_cell_count"].to_numpy(dtype=int),
    ):
        raise ValueError("Sequence manifest support counts do not match the NPZ masks.")
    short_atm_grid = (
        (maturity_grid[:, None] <= float(config.atm_short_max_days) + 1e-6)
        & (
            np.abs(strike_grid[None, :] - 1.0)
            <= float(config.atm_short_range) + 1e-6
        )
    )
    short_atm_counts = (evaluation_support & short_atm_grid).sum(
        axis=(1, 2)
    ).astype(int)
    if not np.array_equal(
        short_atm_counts,
        manifest["short_atm_supported_cell_count"].to_numpy(dtype=int),
    ):
        raise ValueError(
            "Sequence manifest short-ATM support counts do not match the NPZ masks."
        )
    if not np.array_equal(
        (short_atm_counts > 0).astype(int),
        manifest["short_atm_eligible"].to_numpy(dtype=int),
    ):
        raise ValueError("Sequence manifest short-ATM eligibility is inconsistent.")

    matched_features = np.asarray(arrays["matched_text_features"], dtype=np.float32)
    text_dim = int(matched_features.shape[1])
    if matched_features.shape != (sample_count, text_dim) or text_dim != 389:
        raise ValueError(
            f"Sequence matched text features must have shape ({sample_count}, 389)."
        )
    for key in ("count_text_features", "shuffled_text_features"):
        if np.asarray(arrays[key]).shape != matched_features.shape:
            raise ValueError(f"Sequence {key} shape does not match matched features.")
    for key in (
        "matched_text_features",
        "count_text_features",
        "shuffled_text_features",
    ):
        if not np.all(np.isfinite(np.asarray(arrays[key]))):
            raise ValueError(f"Sequence {key} contains non-finite values.")
        if np.any(np.asarray(arrays[key])[has_news <= 0.0] != 0.0):
            raise ValueError(f"Sequence {key} must be zero for no-news samples.")
    if text_array_name is None:
        text_features = np.zeros_like(matched_features, dtype=np.float32)
        has_text = np.zeros(sample_count, dtype=np.float32)
    else:
        text_features = np.asarray(arrays[text_array_name], dtype=np.float32)
        has_text = has_news.astype(np.float32)
        text_features = np.where(
            has_text[:, None] > 0.0,
            text_features,
            0.0,
        ).astype(np.float32)

    if bool(config.normalize_text_embedding):
        raise ValueError(
            "Rolling-history features are already train-fold transformed; "
            "normalize_text_embedding must be false."
        )

    samples: list[FilmWGANSample] = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        row_history = history[index].copy()
        row_support = history_support[index].copy()
        row_steps = history_steps[index].copy()
        timestamp_values = tuple(json.loads(str(row.history_anchor_times_json)))
        if len(timestamp_values) != int(config.history_steps):
            raise ValueError(
                f"Sample {row.sample_id} history timestamp count mismatch."
            )
        metadata = {
            "surface_model": "raw",
            "split": str(row.split),
            "trading_day": str(row.trading_day),
            "history_valid_steps": int(row_steps.sum()),
            "history_observed_valid_steps": int(row_steps.sum()),
            "history_effective_valid_steps": (
                1
                if str(config.history_input_policy).strip().lower()
                == "current_only"
                else int(row_steps.sum())
            ),
            "history_input_policy": str(config.history_input_policy),
            "sequence_text_variant": variant,
            "has_text": float(has_text[index]),
            "has_news": float(has_news[index]),
            "report_atm7_metric": bool(config.report_atm7_metric),
        }
        for name in (
            "news_count_total",
            "news_count_0_15",
            "news_count_15_30",
            "news_count_30_60",
            "latest_news_recency_minutes",
            "source_article_ids_json",
            "session_id",
        ):
            if hasattr(row, name):
                metadata[name] = getattr(row, name)
        sample = FilmWGANSample(
            sample_id=str(row.sample_id),
            global_index=int(row.global_index),
            timestamp=str(row.current_snapshot_time_utc),
            current_snapshot_time_utc=str(row.current_snapshot_time_utc),
            target_snapshot_time_utc=str(row.target_snapshot_time_utc),
            current_surface=row_history[-1].copy(),
            target_surface=targets[index].copy(),
            strike_grid=strike_grid.copy(),
            maturity_days_grid=maturity_grid.copy(),
            text_embedding=text_features[index].copy(),
            metadata=metadata,
            raw_text_embedding=text_features[index].copy(),
            current_support_mask=row_support[-1].copy(),
            target_support_mask=target_support[index].copy(),
            surface_history=row_history,
            history_support_mask=row_support,
            history_step_mask=row_steps,
            history_timestamps_utc=timestamp_values,
        )
        if sample.surface_pair_id != str(row.surface_pair_id):
            raise ValueError(
                f"Sequence surface_pair_id mismatch for {sample.sample_id}: "
                f"{sample.surface_pair_id} != {row.surface_pair_id}."
            )
        samples.append(sample)
    return samples


class FilmWGANSequenceDataset(Dataset):
    """Dataset that emits a masked `[T, 3, H, W]` history condition."""

    def __init__(
        self,
        samples: Sequence,
        normalization_stats,
        *,
        normalize_current_surface: bool,
        normalize_target_delta: bool,
    ) -> None:
        self.samples = list(samples)
        self.normalization_stats = normalization_stats
        self.normalize_current_surface = bool(normalize_current_surface)
        self.normalize_target_delta = bool(normalize_target_delta)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        history = np.asarray(sample.surface_history, dtype=np.float32)
        support = np.asarray(sample.history_support_mask, dtype=np.float32)
        step_mask = np.asarray(sample.history_step_mask, dtype=np.float32)
        if str(sample.metadata.get("history_input_policy", "full_history")) == "current_only":
            history = history.copy()
            support = support.copy()
            step_mask = step_mask.copy()
            history[:-1] = 0.0
            support[:-1] = 0.0
            step_mask[:-1] = 0.0
        current_flat = sample.current_surface.astype(np.float32).reshape(-1)
        target_flat = sample.target_surface.astype(np.float32).reshape(-1)
        history_log = np.log(np.clip(history, _VOL_FLOOR, None))
        current_log = np.log(np.clip(current_flat, _VOL_FLOOR, None))
        target_log = np.log(np.clip(target_flat, _VOL_FLOOR, None))
        target_delta = target_log - current_log
        stats = self.normalization_stats
        if self.normalize_current_surface:
            history_log = (
                history_log - stats.current_log_mean.reshape(1, *sample.surface_shape)
            ) / stats.current_log_std.reshape(1, *sample.surface_shape)
        evaluation_support = sample.evaluation_support_mask.astype(np.float32)
        if self.normalize_target_delta:
            target_delta = (
                target_delta - stats.delta_mean
            ) / stats.delta_std
        valid = step_mask[:, None, None]
        normalized_history = np.where(
            (support > 0.0) & (valid > 0.0),
            history_log,
            0.0,
        ).astype(np.float32)
        support = np.where(valid > 0.0, support, 0.0).astype(np.float32)
        step_channel = np.broadcast_to(valid, history.shape).astype(np.float32)
        history_tensor = np.stack(
            [normalized_history, support, step_channel],
            axis=1,
        )
        target_delta = np.where(
            evaluation_support.reshape(-1) > 0.0,
            target_delta,
            0.0,
        ).astype(np.float32)
        return (
            torch.from_numpy(history_tensor),
            torch.from_numpy(sample.text_embedding.astype(np.float32)),
            torch.from_numpy(target_delta),
            torch.from_numpy(current_flat),
            torch.from_numpy(target_flat),
            torch.from_numpy(evaluation_support.reshape(-1)),
            torch.tensor(float(sample.metadata.get("has_text", 0.0)), dtype=torch.float32),
        )


def _partitions(samples: Sequence) -> dict[str, list]:
    result = {"train": [], "val": [], "test": []}
    for sample in samples:
        split = str(sample.metadata["split"])
        result[split].append(sample)
    return result


def create_sequence_data_bundle(config: FilmWGANTrainConfig):
    from .data import (
        FilmWGANDataBundle,
        _check_consistent_grids,
        _compute_normalization_stats,
    )

    samples = load_sequence_samples(config)
    partitions = _partitions(samples)
    if len(partitions["train"]) < int(config.min_samples_for_training):
        raise ValueError("Sequence fold contains too few training samples.")
    all_items = [
        *partitions["train"],
        *partitions["val"],
        *partitions["test"],
    ]
    strike_grid, maturity_grid = _check_consistent_grids(all_items)
    normalization = _compute_normalization_stats(partitions["train"])

    def loader(split: str, *, shuffle: bool) -> DataLoader | None:
        items = partitions[split]
        if not items:
            return None
        return DataLoader(
            FilmWGANSequenceDataset(
                items,
                normalization,
                normalize_current_surface=config.normalize_current_surface,
                normalize_target_delta=config.normalize_target_delta,
            ),
            batch_size=int(config.batch_size),
            shuffle=shuffle,
            num_workers=int(config.num_workers),
        )

    manifest = _manifest_for_config(config)
    support_path = _resolved_path(config.surface_support_path)
    transform_path = _resolved_path(config.text_transform_path)
    return FilmWGANDataBundle(
        train_loader=loader("train", shuffle=True),
        val_loader=loader("val", shuffle=False),
        test_loader=loader("test", shuffle=False),
        train_samples=len(partitions["train"]),
        val_samples=len(partitions["val"]),
        test_samples=len(partitions["test"]),
        surface_shape=partitions["train"][0].surface_shape,
        strike_grid=strike_grid,
        maturity_days_grid=maturity_grid,
        embedding_dim=int(partitions["train"][0].text_embedding.size),
        normalization_stats=normalization,
        train_items=partitions["train"],
        val_items=partitions["val"],
        test_items=partitions["test"],
        all_items=all_items,
        split_manifest=manifest,
        text_transform_path=str(transform_path),
        text_transform_sha256=(
            sha256_file(transform_path) if transform_path.is_file() else ""
        ),
        surface_support_path=str(support_path),
        surface_support_sha256=(
            sha256_file(support_path) if support_path.is_file() else ""
        ),
    )


def select_sequence_samples(
    config: FilmWGANSampleConfig,
    samples: Sequence,
) -> list:
    from .data import select_samples

    requested = str(config.split).strip().lower()
    selected = [
        sample
        for sample in samples
        if requested == "all" or str(sample.metadata["split"]) == requested
    ]
    return select_samples(
        selected,
        selection_mode=config.selection_mode,
        selection_count=config.selection_count,
    )
