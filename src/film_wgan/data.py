"""Standalone merged-vol workbook parsing for the FiLM WGAN module."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import FilmWGANSampleConfig, FilmWGANTrainConfig
from .matching import (
    TextAlignmentPlan,
    build_text_alignment_plan,
)
from .support import (
    RawSurfaceSupportArtifact,
    fit_raw_support_artifact,
    parse_raw_surface_params,
    raw_support_mask,
    reconstruct_raw_surface,
    validate_support_artifact_lineage,
)
from .text_transform import (
    FilmWGANTextTransform,
    fit_text_transform,
    l2_normalize_rows,
    ordered_ids_sha256,
    sha256_file,
)

_VOL_FLOOR = 1e-4
_FORECAST_HORIZON_MINUTES = 5
_INTRADAY_ALIGNMENT_MAX_MINUTES = 15
_FORWARD_ALIGNMENT_MAX_MINUTES = 72 * 60
_SESSION_ORIGIN_TOLERANCE_MAX_MINUTES = 5


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
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode == "none":
        return np.zeros(1, dtype=np.float32)
    hd = np.asarray(_parse_serialized_list(getattr(row, "hd_embedding", [])), dtype=np.float32)
    lp = np.asarray(_parse_serialized_list(getattr(row, "lp_embedding", [])), dtype=np.float32)
    if normalized_mode == "zero_lp":
        if lp.size <= 0:
            raise ValueError("text_embedding_mode=zero_lp requires a non-empty 'lp_embedding' column.")
        return np.zeros_like(lp, dtype=np.float32)
    if normalized_mode == "hd":
        return hd
    if normalized_mode == "lp":
        return lp
    if normalized_mode == "concat":
        return np.concatenate([hd, lp], axis=0).astype(np.float32)
    if normalized_mode == "bow":
        embedding = np.asarray(_parse_serialized_list(getattr(row, "bow_embedding", [])), dtype=np.float32)
        if embedding.size <= 0:
            raise ValueError("text_embedding_mode=bow requires a non-empty 'bow_embedding' column.")
        return embedding
    if normalized_mode in {"sentiment", "llm_sentiment"}:
        embedding = np.asarray(_parse_serialized_list(getattr(row, "sentiment_embedding", [])), dtype=np.float32)
        if embedding.size <= 0:
            raise ValueError("text_embedding_mode=sentiment requires a non-empty 'sentiment_embedding' column.")
        return embedding
    raise ValueError(
        "text_embedding_mode must be one of "
        "['none', 'zero_lp', 'hd', 'lp', 'concat', 'bow', 'sentiment', 'llm_sentiment'], "
        f"got: {mode}"
    )


def _text_mode_has_conditioning(mode: str) -> bool:
    normalized_mode = str(mode).strip().lower().replace("-", "_")
    return normalized_mode not in {"none", "zero_lp"}


@dataclass(frozen=True)
class FilmWGANSample:
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
    raw_text_embedding: np.ndarray | None = None
    current_support_mask: np.ndarray | None = None
    target_support_mask: np.ndarray | None = None

    @property
    def surface_shape(self) -> tuple[int, int]:
        return self.current_surface.shape

    @property
    def surface_pair_id(self) -> str:
        current = _canonical_timestamp(self.current_snapshot_time_utc)
        target = _canonical_timestamp(self.target_snapshot_time_utc)
        return hashlib.sha256(f"{current}|{target}".encode("utf-8")).hexdigest()[:20]

    @property
    def evaluation_support_mask(self) -> np.ndarray:
        if self.current_support_mask is None or self.target_support_mask is None:
            return np.ones(self.surface_shape, dtype=bool)
        return np.asarray(self.current_support_mask, dtype=bool) & np.asarray(
            self.target_support_mask,
            dtype=bool,
        )


@dataclass(frozen=True)
class FilmWGANNormalizationStats:
    """Feature statistics used to normalize FiLM WGAN inputs and outputs."""

    current_log_mean: np.ndarray
    current_log_std: np.ndarray
    delta_mean: np.ndarray
    delta_std: np.ndarray
    text_mean: np.ndarray
    text_std: np.ndarray


@dataclass(frozen=True)
class FilmWGANDataBundle:
    """Train/validation loaders and metadata for the standalone FiLM WGAN module."""

    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader | None
    train_samples: int
    val_samples: int
    test_samples: int
    surface_shape: tuple[int, int]
    strike_grid: np.ndarray
    maturity_days_grid: np.ndarray
    embedding_dim: int
    normalization_stats: FilmWGANNormalizationStats
    train_items: list[FilmWGANSample]
    val_items: list[FilmWGANSample]
    test_items: list[FilmWGANSample]
    all_items: list[FilmWGANSample]
    split_manifest: pd.DataFrame
    text_transform_path: str = ""
    text_transform_sha256: str = ""
    surface_support_path: str = ""
    surface_support_sha256: str = ""
    native_train_items: list[FilmWGANSample] = field(default_factory=list)
    native_val_items: list[FilmWGANSample] = field(default_factory=list)
    native_test_items: list[FilmWGANSample] = field(default_factory=list)
    text_alignment_plans: dict[str, TextAlignmentPlan] = field(default_factory=dict)


def _canonical_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _validate_sample_timing(sample: FilmWGANSample) -> None:
    """Validate strict or explicitly forward-aligned news/surface timing."""

    news_timestamp = pd.Timestamp(_canonical_timestamp(sample.timestamp))
    current_timestamp = pd.Timestamp(
        _canonical_timestamp(sample.current_snapshot_time_utc)
    )
    target_timestamp = pd.Timestamp(
        _canonical_timestamp(sample.target_snapshot_time_utc)
    )
    expected_horizon = pd.Timedelta(minutes=_FORECAST_HORIZON_MINUTES)
    if target_timestamp - current_timestamp != expected_horizon:
        raise ValueError(
            f"Sample {sample.sample_id} must forecast exactly "
            f"{_FORECAST_HORIZON_MINUTES} minutes: "
            f"{_canonical_timestamp(sample.current_snapshot_time_utc)} -> "
            f"{_canonical_timestamp(sample.target_snapshot_time_utc)}."
        )

    alignment_mode = str(
        sample.metadata.get("news_alignment_mode", "exact")
    ).strip().lower()
    if alignment_mode in {"", "nan", "none"}:
        alignment_mode = "exact"
    if alignment_mode not in {"forward_valid_pair", "exchange_session"}:
        if news_timestamp != current_timestamp:
            raise ValueError(
                f"Sample {sample.sample_id} violates news=current timestamp: "
                f"{_canonical_timestamp(sample.timestamp)} != "
                f"{_canonical_timestamp(sample.current_snapshot_time_utc)}."
            )
        return

    available_timestamp = pd.Timestamp(
        _canonical_timestamp(
            sample.metadata.get("news_available_time_utc", sample.timestamp)
        )
    )
    effective_origin = pd.Timestamp(
        _canonical_timestamp(
            sample.metadata.get(
                "effective_origin_utc",
                sample.current_snapshot_time_utc,
            )
        )
    )
    if news_timestamp != available_timestamp:
        raise ValueError(
            f"Sample {sample.sample_id} news timestamp does not match "
            "news_available_time_utc."
        )
    if effective_origin != current_timestamp:
        raise ValueError(
            f"Sample {sample.sample_id} effective origin does not match "
            "current_snapshot_time_utc."
        )

    shift_minutes = (
        current_timestamp - available_timestamp
    ).total_seconds() / 60.0
    if shift_minutes < 0:
        raise ValueError(
            f"Sample {sample.sample_id} has an origin before news availability."
        )

    recorded_shift = pd.to_numeric(
        pd.Series([sample.metadata.get("origin_shift_minutes")]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(recorded_shift) or not np.isclose(
        float(recorded_shift),
        shift_minutes,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            f"Sample {sample.sample_id} origin_shift_minutes does not match "
            "the timestamp-derived shift."
        )

    recorded_alignment_type = str(
        sample.metadata.get("alignment_type", "")
    ).strip().lower()
    if alignment_mode == "exchange_session":
        scheduled_origin = pd.Timestamp(
            _canonical_timestamp(
                sample.metadata.get("scheduled_origin_utc", "")
            )
        )
        if scheduled_origin < available_timestamp:
            raise ValueError(
                f"Sample {sample.sample_id} scheduled origin precedes "
                "news availability."
            )
        tolerance_minutes = (
            current_timestamp - scheduled_origin
        ).total_seconds() / 60.0
        if tolerance_minutes < 0 or (
            tolerance_minutes
            > _SESSION_ORIGIN_TOLERANCE_MAX_MINUTES
        ):
            raise ValueError(
                f"Sample {sample.sample_id} exceeds the "
                f"{_SESSION_ORIGIN_TOLERANCE_MAX_MINUTES}-minute "
                "session-origin tolerance."
            )
        recorded_tolerance = pd.to_numeric(
            pd.Series(
                [
                    sample.metadata.get(
                        "origin_tolerance_minutes_used"
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]
        if pd.isna(recorded_tolerance) or not np.isclose(
            float(recorded_tolerance),
            tolerance_minutes,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"Sample {sample.sample_id} origin tolerance does not "
                "match the timestamp-derived tolerance."
            )

        market_state = str(
            sample.metadata.get("publication_market_state", "")
        ).strip().lower()
        if market_state == "open":
            if scheduled_origin != available_timestamp:
                raise ValueError(
                    f"Sample {sample.sample_id} open-session news must "
                    "retain its publication minute."
                )
            expected_alignment_type = (
                "exact"
                if tolerance_minutes == 0
                else "market_open_tolerance_shift"
            )
        elif market_state == "closed":
            if scheduled_origin <= available_timestamp:
                raise ValueError(
                    f"Sample {sample.sample_id} closed-session news must "
                    "move to a later session open."
                )
            expected_alignment_type = "closed_to_next_open"
        else:
            raise ValueError(
                f"Sample {sample.sample_id} has invalid "
                f"publication_market_state={market_state!r}."
            )

        session_open = pd.Timestamp(
            _canonical_timestamp(
                sample.metadata.get("session_open_utc", "")
            )
        )
        session_close = pd.Timestamp(
            _canonical_timestamp(
                sample.metadata.get("session_close_utc", "")
            )
        )
        current_window_start = pd.Timestamp(
            _canonical_timestamp(
                sample.metadata.get("current_window_start_utc", "")
            )
        )
        target_window_end = pd.Timestamp(
            _canonical_timestamp(
                sample.metadata.get("target_window_end_utc", "")
            )
        )
        if (
            current_window_start
            != current_timestamp
            - pd.Timedelta(minutes=_FORECAST_HORIZON_MINUTES)
            or target_window_end != target_timestamp
            or current_window_start < session_open
            or target_window_end > session_close
        ):
            raise ValueError(
                f"Sample {sample.sample_id} crosses its recorded CME "
                "continuous-session boundary."
            )
        if recorded_alignment_type != expected_alignment_type:
            raise ValueError(
                f"Sample {sample.sample_id} alignment_type="
                f"{recorded_alignment_type!r} does not match "
                f"{expected_alignment_type!r}."
            )
        return

    if shift_minutes > _FORWARD_ALIGNMENT_MAX_MINUTES:
        raise ValueError(
            f"Sample {sample.sample_id} exceeds the "
            f"{_FORWARD_ALIGNMENT_MAX_MINUTES}-minute forward alignment limit."
        )
    if shift_minutes == 0:
        expected_alignment_type = "exact"
    elif shift_minutes <= _INTRADAY_ALIGNMENT_MAX_MINUTES:
        expected_alignment_type = "intraday_shift"
    else:
        expected_alignment_type = "session_shift"
    if recorded_alignment_type != expected_alignment_type:
        raise ValueError(
            f"Sample {sample.sample_id} alignment_type="
            f"{recorded_alignment_type!r} does not match "
            f"{expected_alignment_type!r}."
        )


def workbook_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.std(values, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return std


def _masked_mean_std(values: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = masks.sum(axis=0)
    sums = np.where(masks, values, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    squared = np.where(masks, np.square(values - means), 0.0).sum(axis=0)
    variances = np.divide(squared, counts, out=np.ones_like(squared), where=counts > 0)
    std = np.sqrt(np.maximum(variances, 0.0)).astype(np.float32)
    std[(std < 1e-6) | (counts <= 0)] = 1.0
    return means.astype(np.float32), std


def _compute_normalization_stats(samples: Sequence[FilmWGANSample]) -> FilmWGANNormalizationStats:
    current_logs: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    current_masks: list[np.ndarray] = []
    delta_masks: list[np.ndarray] = []
    text_embeddings: list[np.ndarray] = []
    for sample in samples:
        current_flat = sample.current_surface.astype(np.float32).reshape(-1)
        target_flat = sample.target_surface.astype(np.float32).reshape(-1)
        current_log = np.log(np.clip(current_flat, _VOL_FLOOR, None))
        target_log = np.log(np.clip(target_flat, _VOL_FLOOR, None))
        current_logs.append(current_log.astype(np.float32))
        deltas.append((target_log - current_log).astype(np.float32))
        current_masks.append(
            (
                np.asarray(sample.current_support_mask, dtype=bool).reshape(-1)
                if sample.current_support_mask is not None
                else np.ones_like(current_log, dtype=bool)
            )
        )
        delta_masks.append(sample.evaluation_support_mask.reshape(-1))
        text_embeddings.append(sample.text_embedding.astype(np.float32))
    current_logs_arr = np.stack(current_logs, axis=0)
    deltas_arr = np.stack(deltas, axis=0)
    current_masks_arr = np.stack(current_masks, axis=0)
    delta_masks_arr = np.stack(delta_masks, axis=0)
    text_arr = np.stack(text_embeddings, axis=0)
    current_mean, current_std = _masked_mean_std(current_logs_arr, current_masks_arr)
    delta_mean, delta_std = _masked_mean_std(deltas_arr, delta_masks_arr)
    return FilmWGANNormalizationStats(
        current_log_mean=current_mean,
        current_log_std=current_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
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


class FilmWGANDataset(Dataset):
    """Torch dataset yielding the tensors needed by the standalone FiLM WGAN trainer."""

    def __init__(
        self,
        samples: Sequence[FilmWGANSample],
        normalization_stats: FilmWGANNormalizationStats,
        *,
        normalize_current_surface: bool,
        normalize_target_delta: bool,
        normalize_text_embedding: bool,
        include_has_text: bool = False,
        include_support_mask: bool = False,
        include_sample_index: bool = False,
    ):
        self.samples = list(samples)
        self.normalization_stats = normalization_stats
        self.normalize_current_surface = bool(normalize_current_surface)
        self.normalize_target_delta = bool(normalize_target_delta)
        self.normalize_text_embedding = bool(normalize_text_embedding)
        self.include_has_text = bool(include_has_text)
        self.include_support_mask = bool(include_support_mask)
        self.include_sample_index = bool(include_sample_index)

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
        current_support = (
            np.asarray(sample.current_support_mask, dtype=np.float32).reshape(-1)
            if sample.current_support_mask is not None
            else np.ones_like(current_features, dtype=np.float32)
        )
        evaluation_support = sample.evaluation_support_mask.astype(np.float32).reshape(-1)
        stats = self.normalization_stats
        if self.normalize_current_surface:
            current_features = ((current_features - stats.current_log_mean) / stats.current_log_std).astype(np.float32)
        if self.normalize_text_embedding:
            text_features = ((text_features - stats.text_mean) / stats.text_std).astype(np.float32)
        if self.normalize_target_delta:
            target_delta_features = ((target_delta_features - stats.delta_mean) / stats.delta_std).astype(np.float32)
        current_features = np.where(current_support > 0.0, current_features, 0.0).astype(np.float32)
        target_delta_features = np.where(
            evaluation_support > 0.0,
            target_delta_features,
            0.0,
        ).astype(np.float32)
        height, width = sample.surface_shape
        current_tensor = torch.from_numpy(current_features.reshape(1, height, width))
        if self.include_support_mask:
            current_tensor = torch.cat(
                [
                    current_tensor,
                    torch.from_numpy(current_support.reshape(1, height, width)),
                ],
                dim=0,
            )
        tensors = (
            current_tensor,
            torch.from_numpy(text_features),
            torch.from_numpy(target_delta_features),
            torch.from_numpy(current_flat),
            torch.from_numpy(target_flat),
        )
        extras: tuple[torch.Tensor, ...] = ()
        if self.include_support_mask:
            extras = (*extras, torch.from_numpy(evaluation_support))
        if self.include_has_text:
            extras = (
                *extras,
                torch.tensor(float(sample.metadata.get("has_text", 1.0)), dtype=torch.float32),
            )
        if self.include_sample_index:
            extras = (*extras, torch.tensor(index, dtype=torch.long))
        return (*tensors, *extras)


def _validate_shape(sample_id: str, surface: np.ndarray, expected_shape: tuple[int, int]) -> None:
    if tuple(surface.shape) != tuple(expected_shape):
        raise ValueError(f"Sample {sample_id} surface shape mismatch: expected {expected_shape}, got {surface.shape}")


def load_film_wgan_samples(config: FilmWGANTrainConfig | FilmWGANSampleConfig) -> list[FilmWGANSample]:
    workbook_path = Path(config.data_path)
    dataframe = pd.read_excel(workbook_path, sheet_name=config.sheet_name)
    if "training_candidate_flag" in dataframe.columns:
        filtered = dataframe[dataframe["training_candidate_flag"].fillna(0).astype(int) == 1].copy()
        if not filtered.empty:
            dataframe = filtered
    if dataframe.empty:
        raise ValueError(f"No usable rows found in workbook {workbook_path} sheet {config.sheet_name}.")

    rows = list(dataframe.itertuples(index=False))
    parsed: list[FilmWGANSample] = []
    has_text = 1.0 if _text_mode_has_conditioning(config.text_embedding_mode) else 0.0
    for global_index, row in enumerate(rows):
        strike_grid = np.asarray(_parse_serialized_list(getattr(row, "strike_grid")), dtype=np.float32)
        maturity_days_grid = np.asarray(_parse_serialized_list(getattr(row, "maturity_days_grid")), dtype=np.float32)
        if strike_grid.size <= 0 or maturity_days_grid.size <= 0:
            raise ValueError("strike_grid and maturity_days_grid must be non-empty for standalone FiLM WGAN.")
        surface_shape = (int(maturity_days_grid.size), int(strike_grid.size))
        current_flat = np.asarray(_parse_serialized_list(getattr(row, "current_surface_flat")), dtype=np.float32)
        target_flat = np.asarray(_parse_serialized_list(getattr(row, "target_surface_flat")), dtype=np.float32)
        current_surface = current_flat.reshape(surface_shape)
        target_surface = target_flat.reshape(surface_shape)
        _validate_shape(str(getattr(row, "sample_id")), current_surface, surface_shape)
        _validate_shape(str(getattr(row, "sample_id")), target_surface, surface_shape)
        raw_lp_embedding = np.asarray(
            _parse_serialized_list(getattr(row, "lp_embedding", [])),
            dtype=np.float32,
        )
        parsed.append(
            FilmWGANSample(
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
                    "surface_model": str(getattr(row, "surface_model", "")),
                    "publication_timestamp_utc": str(
                        getattr(row, "publication_timestamp_utc", "")
                    ),
                    "publication_availability_lag_minutes": int(
                        getattr(row, "publication_availability_lag_minutes", 0)
                        if not pd.isna(
                            getattr(row, "publication_availability_lag_minutes", 0)
                        )
                        else 0
                    ),
                    "current_surface_param_json": getattr(
                        row,
                        "current_surface_param_json",
                        "",
                    ),
                    "target_surface_param_json": getattr(
                        row,
                        "target_surface_param_json",
                        "",
                    ),
                    "pair_quality_label": str(getattr(row, "pair_quality_label", "")),
                    "current_weighted_iv_rmse": getattr(row, "current_weighted_iv_rmse", None),
                    "target_weighted_iv_rmse": getattr(row, "target_weighted_iv_rmse", None),
                    "event_group": getattr(row, "event_group", ""),
                    "has_news": getattr(row, "has_news", ""),
                    "news_alignment_mode": getattr(
                        row,
                        "news_alignment_mode",
                        "exact",
                    ),
                    "news_available_time_utc": getattr(
                        row,
                        "news_available_time_utc",
                        getattr(row, "news_timestamp_utc", ""),
                    ),
                    "effective_origin_utc": getattr(
                        row,
                        "effective_origin_utc",
                        getattr(row, "current_snapshot_time_utc", ""),
                    ),
                    "origin_shift_minutes": getattr(
                        row,
                        "origin_shift_minutes",
                        0,
                    ),
                    "alignment_type": getattr(
                        row,
                        "alignment_type",
                        "exact",
                    ),
                    "matching_rank": getattr(row, "matching_rank", ""),
                    "collision_count": getattr(row, "collision_count", 1),
                    "news_cluster_id": getattr(row, "news_cluster_id", ""),
                    "current_window_start_utc": getattr(
                        row,
                        "current_window_start_utc",
                        "",
                    ),
                    "current_window_end_utc": getattr(
                        row,
                        "current_window_end_utc",
                        "",
                    ),
                    "target_window_start_utc": getattr(
                        row,
                        "target_window_start_utc",
                        "",
                    ),
                    "target_window_end_utc": getattr(
                        row,
                        "target_window_end_utc",
                        "",
                    ),
                    "original_news_quarter": getattr(
                        row,
                        "original_news_quarter",
                        "",
                    ),
                    "effective_origin_quarter": getattr(
                        row,
                        "effective_origin_quarter",
                        "",
                    ),
                    "publication_market_state": getattr(
                        row,
                        "publication_market_state",
                        "",
                    ),
                    "scheduled_origin_utc": getattr(
                        row,
                        "scheduled_origin_utc",
                        "",
                    ),
                    "origin_tolerance_minutes_used": getattr(
                        row,
                        "origin_tolerance_minutes_used",
                        "",
                    ),
                    "session_shift_minutes": getattr(
                        row,
                        "session_shift_minutes",
                        "",
                    ),
                    "session_shift_reason": getattr(
                        row,
                        "session_shift_reason",
                        "",
                    ),
                    "session_id": getattr(row, "session_id", ""),
                    "session_open_utc": getattr(
                        row,
                        "session_open_utc",
                        "",
                    ),
                    "session_close_utc": getattr(
                        row,
                        "session_close_utc",
                        "",
                    ),
                    "quiet_buffer_minutes": getattr(row, "quiet_buffer_minutes", ""),
                    "quiet_grid_minutes": getattr(row, "quiet_grid_minutes", ""),
                    "has_text": has_text,
                },
                raw_text_embedding=raw_lp_embedding if raw_lp_embedding.size else None,
            )
        )

    split_strategy = str(getattr(config, "split_strategy", "legacy_row")).strip().lower()
    if split_strategy == "grouped_chronological":
        parsed = sorted(
            parsed,
            key=lambda sample: (
                pd.Timestamp(sample.current_snapshot_time_utc),
                pd.Timestamp(sample.target_snapshot_time_utc),
                pd.Timestamp(sample.timestamp),
                sample.sample_id,
            ),
        )
    else:
        parsed = sorted(parsed, key=lambda sample: (pd.Timestamp(sample.timestamp), sample.sample_id))
    for index, sample in enumerate(parsed):
        parsed[index] = FilmWGANSample(
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
            raw_text_embedding=sample.raw_text_embedding,
            current_support_mask=sample.current_support_mask,
            target_support_mask=sample.target_support_mask,
        )
    return parsed


_NEWS_SAMPLE_ID_PATTERN = re.compile(r"^news_(\d+)$")


def _optional_source_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _resolved_config_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _parse_json_string_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pair feature field {field!r} is not valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Pair feature field {field!r} must contain a JSON list.")
    return [str(item) for item in parsed]


@lru_cache(maxsize=16)
def _load_pair_text_features(path_value: str) -> dict[str, dict[str, Any]]:
    path = _resolved_config_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"pair_text_feature_path does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("pair_text_feature_path currently supports CSV artifacts only.")
    frame = pd.read_csv(path)
    required = {
        "surface_pair_id",
        "text_embedding",
        "representation",
        "pooling_mode",
        "source_sample_ids",
        "article_ids",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair text feature artifact {path} is missing columns: {missing}")
    if frame["surface_pair_id"].astype(str).duplicated().any():
        raise ValueError(f"Pair text feature artifact has duplicate surface_pair_id values: {path}")
    return {
        str(row.surface_pair_id): {
            "text_embedding": np.asarray(
                _parse_serialized_list(row.text_embedding),
                dtype=np.float32,
            ),
            "representation": str(row.representation),
            "pooling_mode": str(row.pooling_mode),
            "source_sample_ids": _parse_json_string_list(
                row.source_sample_ids,
                field="source_sample_ids",
            ),
            "article_ids": _parse_json_string_list(row.article_ids, field="article_ids"),
            "source_files": (
                _parse_json_string_list(row.source_files, field="source_files")
                if hasattr(row, "source_files") and not pd.isna(row.source_files)
                else []
            ),
            "feature_sha256": str(getattr(row, "feature_sha256", "")),
        }
        for row in frame.itertuples(index=False)
    }


def _assert_pair_member_consistency(reference: FilmWGANSample, candidate: FilmWGANSample) -> None:
    fields = (
        ("current_surface", reference.current_surface, candidate.current_surface),
        ("target_surface", reference.target_surface, candidate.target_surface),
        ("strike_grid", reference.strike_grid, candidate.strike_grid),
        ("maturity_days_grid", reference.maturity_days_grid, candidate.maturity_days_grid),
    )
    for field_name, left, right in fields:
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(
                f"Surface pair {reference.surface_pair_id} has inconsistent {field_name}: "
                f"{reference.sample_id} vs {candidate.sample_id}."
            )
    if _canonical_timestamp(reference.current_snapshot_time_utc) != _canonical_timestamp(
        candidate.current_snapshot_time_utc
    ):
        raise ValueError(f"Surface pair {reference.surface_pair_id} has inconsistent current timestamps.")
    if _canonical_timestamp(reference.target_snapshot_time_utc) != _canonical_timestamp(
        candidate.target_snapshot_time_utc
    ):
        raise ValueError(f"Surface pair {reference.surface_pair_id} has inconsistent target timestamps.")
    for field_name in ("current_surface_param_json", "target_surface_param_json"):
        reference_value = reference.metadata.get(field_name, "")
        candidate_value = candidate.metadata.get(field_name, "")
        if bool(str(reference_value).strip()) != bool(str(candidate_value).strip()):
            raise ValueError(
                f"Surface pair {reference.surface_pair_id} has inconsistent {field_name} presence."
            )
        if str(reference_value).strip():
            reference_params = parse_raw_surface_params(reference_value)
            candidate_params = parse_raw_surface_params(candidate_value)
            if reference_params != candidate_params:
                raise ValueError(
                    f"Surface pair {reference.surface_pair_id} has inconsistent {field_name}."
                )


def _news_lineage_rows(
    samples: Sequence[FilmWGANSample],
    *,
    news_workbook: pd.DataFrame,
    strict: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        matched = _NEWS_SAMPLE_ID_PATTERN.fullmatch(sample.sample_id)
        if matched is None:
            raise ValueError(
                "sample_unit=surface_pair requires sample_id=news_<1-based-row-id>; "
                f"got {sample.sample_id!r}."
            )
        news_row_id = int(matched.group(1))
        if news_row_id <= 0 or news_row_id > len(news_workbook):
            raise ValueError(
                f"Sample {sample.sample_id} references news row {news_row_id}, "
                f"but the news workbook has {len(news_workbook)} rows."
            )
        source_row = news_workbook.iloc[news_row_id - 1]
        raw_embedding = np.asarray(
            _parse_serialized_list(source_row.get("LP_embedding", [])),
            dtype=np.float32,
        )
        lp_text = _optional_source_text(source_row.get("LP", ""))
        if strict and not lp_text:
            lineage_status = (
                "empty_lp_nonzero_embedding"
                if raw_embedding.size and bool(np.any(np.abs(raw_embedding) > 0.0))
                else "empty_lp"
            )
        elif strict and raw_embedding.size <= 0:
            lineage_status = "nonempty_lp_missing_embedding"
        else:
            lineage_status = "ok"
        merged_embedding = sample.raw_text_embedding
        if merged_embedding is None or raw_embedding.shape != merged_embedding.shape or not np.array_equal(
            raw_embedding,
            merged_embedding,
        ):
            raise ValueError(
                f"LP embedding lineage mismatch for {sample.sample_id} against 1-based news row {news_row_id}."
            )
        _validate_sample_timing(sample)
        rows.append(
            {
                "sample": sample,
                "news_row_id": news_row_id,
                "article_id": _optional_source_text(source_row.get("ArticleID", "")),
                "source_file": _optional_source_text(source_row.get("SourceFile", "")),
                "lp_text": lp_text,
                "embedding": raw_embedding,
                "lineage_status": lineage_status,
            }
        )
    return rows


def _pool_pair_text(lineage_rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    article_unique: list[dict[str, Any]] = []
    seen_articles: set[str] = set()
    for row in lineage_rows:
        article_key = _optional_source_text(row["article_id"]) or f"news_row_{int(row['news_row_id'])}"
        if article_key in seen_articles:
            continue
        seen_articles.add(article_key)
        article_unique.append(row)

    content_unique: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in article_unique:
        embedding = np.asarray(row["embedding"], dtype=np.float32)
        embedding_hash = hashlib.sha256(embedding.tobytes(order="C")).hexdigest()
        if embedding_hash in seen_hashes:
            continue
        seen_hashes.add(embedding_hash)
        content_unique.append({**row, "embedding_sha256": embedding_hash})
    if not content_unique:
        raise ValueError("A surface pair has no unique LP embeddings after article/content deduplication.")

    normalized = l2_normalize_rows(
        np.stack([np.asarray(row["embedding"], dtype=np.float32) for row in content_unique], axis=0)
    )
    pooled = l2_normalize_rows(np.mean(normalized, axis=0, keepdims=True))[0]
    return pooled.astype(np.float32), content_unique


def aggregate_surface_pair_samples(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    samples: Sequence[FilmWGANSample],
    *,
    news_workbook: pd.DataFrame | None = None,
) -> list[FilmWGANSample]:
    """Collapse article rows to one rigorously audited sample per surface pair."""

    if not samples:
        return []
    mode = str(config.text_embedding_mode).strip().lower().replace("-", "_")
    pair_feature_path = str(config.pair_text_feature_path).strip()
    external_features = (
        _load_pair_text_features(str(_resolved_config_path(pair_feature_path)))
        if pair_feature_path
        else None
    )
    if external_features is None and mode not in {"lp", "zero_lp"}:
        raise ValueError(
            "sample_unit=surface_pair requires pair_text_feature_path for non-LP representations; "
            f"got text_embedding_mode={config.text_embedding_mode!r}."
        )
    strict_lineage = str(config.text_lineage_mode).strip().lower() == "strict"
    if (external_features is None or strict_lineage) and news_workbook is None:
        news_path = Path(config.news_workbook_path)
        if not news_path.is_file():
            raise FileNotFoundError(f"news_workbook_path does not exist: {news_path}")
        news_workbook = pd.read_excel(news_path)
    if external_features is None or strict_lineage:
        assert news_workbook is not None
        required_columns = {"ArticleID", "SourceFile", "LP_embedding"}
        missing_columns = sorted(required_columns - set(news_workbook.columns))
        if missing_columns:
            raise ValueError(
                f"News workbook {config.news_workbook_path} is missing pair-lineage columns: "
                f"{missing_columns}."
            )

    grouped: dict[str, list[FilmWGANSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.surface_pair_id, []).append(sample)
    pooled_samples: list[FilmWGANSample] = []
    has_text = 1.0 if _text_mode_has_conditioning(config.text_embedding_mode) else 0.0
    for pair_id, members in sorted(
        grouped.items(),
        key=lambda item: (
            pd.Timestamp(item[1][0].current_snapshot_time_utc),
            pd.Timestamp(item[1][0].target_snapshot_time_utc),
            item[0],
        ),
    ):
        reference = members[0]
        excluded_rows: list[dict[str, Any]] = []
        for candidate in members[1:]:
            _assert_pair_member_consistency(reference, candidate)
        if external_features is None:
            assert news_workbook is not None
            lineage_rows = _news_lineage_rows(
                members,
                news_workbook=news_workbook,
                strict=strict_lineage,
            )
            excluded_rows = [
                row for row in lineage_rows if str(row["lineage_status"]) != "ok"
            ]
            usable_lineage_rows = (
                [row for row in lineage_rows if str(row["lineage_status"]) == "ok"]
                if strict_lineage
                else lineage_rows
            )
            if not usable_lineage_rows:
                continue
            pooled_text, unique_rows = _pool_pair_text(usable_lineage_rows)
            source_sample_ids = [
                str(row["sample"].sample_id) for row in usable_lineage_rows
            ]
            article_ids = [str(row["article_id"]) for row in unique_rows]
            source_files = [str(row["source_file"]) for row in unique_rows]
            pooling_mode = "mean_l2"
            feature_sha256 = hashlib.sha256(
                pooled_text.tobytes(order="C")
            ).hexdigest()
        else:
            if pair_id not in external_features:
                raise ValueError(
                    f"Pair feature artifact {pair_feature_path} has no row for {pair_id}."
                )
            feature = external_features[pair_id]
            configured_pooling = str(config.text_pooling_mode).strip().lower()
            if str(feature["pooling_mode"]).strip().lower() != configured_pooling:
                raise ValueError(
                    f"Pair {pair_id} pooling mode mismatch: artifact={feature['pooling_mode']!r}, "
                    f"config={configured_pooling!r}."
                )
            artifact_representation = (
                str(feature["representation"]).strip().lower().replace("-", "_")
            )
            if artifact_representation != mode:
                raise ValueError(
                    f"Pair {pair_id} representation mismatch: "
                    f"artifact={feature['representation']!r}, "
                    f"config={config.text_embedding_mode!r}."
                )
            expected_samples = sorted(sample.sample_id for sample in members)
            if strict_lineage:
                assert news_workbook is not None
                external_lineage_rows = _news_lineage_rows(
                    members,
                    news_workbook=news_workbook,
                    strict=True,
                )
                excluded_rows = [
                    row
                    for row in external_lineage_rows
                    if str(row["lineage_status"]) != "ok"
                ]
                expected_samples = sorted(
                    str(row["sample"].sample_id)
                    for row in external_lineage_rows
                    if str(row["lineage_status"]) == "ok"
                )
                if not expected_samples:
                    continue
            artifact_samples = sorted(str(item) for item in feature["source_sample_ids"])
            if expected_samples != artifact_samples:
                raise ValueError(
                    f"Pair {pair_id} source_sample_ids do not match the split workbook rows."
                )
            pooled_text = np.asarray(feature["text_embedding"], dtype=np.float32)
            if pooled_text.size <= 0 or not np.all(np.isfinite(pooled_text)):
                raise ValueError(f"Pair {pair_id} has an invalid external text vector.")
            source_sample_ids = list(feature["source_sample_ids"])
            article_ids = list(feature["article_ids"])
            source_files = list(feature["source_files"])
            pooling_mode = configured_pooling
            feature_sha256 = str(feature["feature_sha256"]) or hashlib.sha256(
                pooled_text.tobytes(order="C")
            ).hexdigest()
        metadata = dict(reference.metadata)
        source_alignment_types = [
            str(sample.metadata.get("alignment_type", "exact") or "exact")
            for sample in members
        ]
        source_shift_minutes = [
            float(value)
            for value in (
                pd.to_numeric(
                    pd.Series(
                        [
                            sample.metadata.get("origin_shift_minutes", 0)
                            for sample in members
                        ]
                    ),
                    errors="coerce",
                )
                .dropna()
                .tolist()
            )
        ]
        minimum_shift = min(source_shift_minutes) if source_shift_minutes else 0.0
        if minimum_shift <= 0.0:
            pair_alignment_type = "exact"
        elif minimum_shift <= 15.0:
            pair_alignment_type = "intraday_shift"
        else:
            pair_alignment_type = "session_shift"
        alignment_type_counts = {
            alignment_type: source_alignment_types.count(alignment_type)
            for alignment_type in sorted(set(source_alignment_types))
        }
        metadata.update(
            {
                "source_sample_ids": source_sample_ids,
                "article_ids": article_ids,
                "source_files": source_files,
                "news_count": int(len(source_sample_ids)),
                "unique_embedding_count": int(len(article_ids)),
                "pooling_mode": pooling_mode,
                "pair_text_feature_path": pair_feature_path,
                "pair_text_feature_sha256": feature_sha256,
                "has_text": has_text,
                "alignment_type": pair_alignment_type,
                "source_alignment_types": source_alignment_types,
                "alignment_type_counts": alignment_type_counts,
                "origin_shift_minutes": minimum_shift,
                "origin_shift_minutes_min": minimum_shift,
                "origin_shift_minutes_max": (
                    max(source_shift_minutes)
                    if source_shift_minutes
                    else 0.0
                ),
                "origin_shift_minutes_mean": (
                    float(np.mean(source_shift_minutes))
                    if source_shift_minutes
                    else 0.0
                ),
                "text_lineage_mode": str(config.text_lineage_mode).strip().lower(),
                "excluded_source_sample_ids": (
                    [str(row["sample"].sample_id) for row in excluded_rows]
                    if strict_lineage
                    else []
                ),
                "excluded_text_lineage_reasons": (
                    [str(row["lineage_status"]) for row in excluded_rows]
                    if strict_lineage
                    else []
                ),
            }
        )
        pooled_samples.append(
            FilmWGANSample(
                sample_id=f"pair_{pair_id}",
                global_index=min(int(sample.global_index) for sample in members),
                timestamp=_canonical_timestamp(reference.current_snapshot_time_utc),
                current_snapshot_time_utc=reference.current_snapshot_time_utc,
                target_snapshot_time_utc=reference.target_snapshot_time_utc,
                current_surface=reference.current_surface.copy(),
                target_surface=reference.target_surface.copy(),
                strike_grid=reference.strike_grid.copy(),
                maturity_days_grid=reference.maturity_days_grid.copy(),
                text_embedding=pooled_text.copy(),
                metadata=metadata,
                raw_text_embedding=pooled_text.copy(),
                current_support_mask=(
                    reference.current_support_mask.copy()
                    if reference.current_support_mask is not None
                    else None
                ),
                target_support_mask=(
                    reference.target_support_mask.copy()
                    if reference.target_support_mask is not None
                    else None
                ),
            )
        )
    return pooled_samples


def _fit_or_load_transform(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    train_samples: Sequence[FilmWGANSample],
) -> FilmWGANTextTransform | None:
    mode = str(config.text_preprocessing_mode).strip().lower()
    if mode == "coordinate_zscore":
        return None
    transform_path_value = str(config.text_transform_path).strip()
    if mode in {"pca", "zscore_pad"} and not transform_path_value:
        raise ValueError(f"text_preprocessing_mode={mode} requires text_transform_path.")

    source_vectors = [
        sample.raw_text_embedding if sample.raw_text_embedding is not None else sample.text_embedding
        for sample in train_samples
    ]
    matrix = np.stack(source_vectors, axis=0).astype(np.float32)
    pair_ids = [sample.surface_pair_id for sample in train_samples]
    if transform_path_value and Path(transform_path_value).is_file():
        transform = FilmWGANTextTransform.load(transform_path_value)
        metadata = transform.metadata
        expected_pair_hash = ordered_ids_sha256(pair_ids)
        expected_workbook_hash = workbook_sha256(config.data_path)
        if str(metadata.get("train_pair_ids_sha256", "")) != expected_pair_hash:
            raise ValueError(
                f"Text transform train-pair SHA mismatch for {transform_path_value}; "
                "the artifact was not fitted on this fold's training pairs."
            )
        if str(metadata.get("input_workbook_sha256", "")) != expected_workbook_hash:
            raise ValueError(
                f"Text transform workbook SHA mismatch for {transform_path_value}."
            )
        pair_feature_path = str(config.pair_text_feature_path).strip()
        if pair_feature_path:
            expected_feature_hash = sha256_file(_resolved_config_path(pair_feature_path))
            if str(metadata.get("input_feature_sha256", "")) != expected_feature_hash:
                raise ValueError(
                    f"Text transform pair-feature SHA mismatch for {transform_path_value}."
                )
        expected_output_dim = (
            int(config.text_pca_components)
            if mode == "pca"
            else int(config.text_output_dim)
            if mode == "zscore_pad"
            else int(matrix.shape[1])
        )
        if (
            transform.mode != mode
            or int(transform.input_dim) != int(matrix.shape[1])
            or int(transform.output_dim) != expected_output_dim
        ):
            raise ValueError(
                "Text transform schema mismatch: "
                f"expected mode={mode}, input_dim={matrix.shape[1]}, "
                f"output_dim={expected_output_dim}; found mode={transform.mode}, "
                f"input_dim={transform.input_dim}, output_dim={transform.output_dim}."
            )
        return transform

    if isinstance(config, FilmWGANSampleConfig) and mode in {"pca", "zscore_pad"}:
        raise FileNotFoundError(
            f"Generate-result cannot fit a missing text transform: {transform_path_value!r}."
        )
    transform = fit_text_transform(
        matrix,
        mode=mode,
        components=int(config.text_pca_components),
        whiten=bool(config.text_pca_whiten),
        train_pair_ids=pair_ids,
        input_workbook_path=config.data_path,
        output_dim=int(config.text_output_dim),
        input_feature_path=(
            _resolved_config_path(str(config.pair_text_feature_path))
            if str(config.pair_text_feature_path).strip()
            else None
        ),
    )
    if transform_path_value:
        transform.save(transform_path_value)
    return transform


def _apply_text_transform(
    samples: Sequence[FilmWGANSample],
    transform: FilmWGANTextTransform | None,
    *,
    preprocessing_mode: str,
) -> list[FilmWGANSample]:
    transformed: list[FilmWGANSample] = []
    mode = str(preprocessing_mode).strip().lower()
    for sample in samples:
        source = (
            sample.raw_text_embedding
            if mode in {"pca", "raw_l2", "zscore_pad"} and sample.raw_text_embedding is not None
            else sample.text_embedding
        )
        features = (
            transform.transform(source)
            if transform is not None
            else np.asarray(source, dtype=np.float32)
        )
        if float(sample.metadata.get("has_text", 1.0)) <= 0.0:
            features = np.zeros_like(features, dtype=np.float32)
        metadata = dict(sample.metadata)
        metadata["text_preprocessing_mode"] = mode
        transformed.append(
            replace(
                sample,
                text_embedding=np.asarray(features, dtype=np.float32),
                metadata=metadata,
            )
        )
    return transformed


def _split_index(total_items: int, train_ratio: float) -> int:
    if total_items < 2:
        return total_items
    split_idx = int(total_items * float(train_ratio))
    split_idx = max(1, min(total_items - 1, split_idx))
    return split_idx


def _grouped_split_boundaries(total_groups: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if total_groups < 3:
        raise ValueError("grouped_chronological splitting requires at least three unique surface pairs.")
    train_end = int(total_groups * float(train_ratio))
    val_end = int(total_groups * (float(train_ratio) + float(val_ratio)))
    train_end = max(1, min(total_groups - 2, train_end))
    val_end = max(train_end + 1, min(total_groups - 1, val_end))
    return train_end, val_end


def build_split_manifest_frame(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    samples: Sequence[FilmWGANSample],
) -> pd.DataFrame:
    strategy = str(getattr(config, "split_strategy", "legacy_row")).strip().lower()
    workbook_hash = workbook_sha256(config.data_path)
    if strategy == "legacy_row":
        split_idx = _split_index(len(samples), config.train_ratio)
        assignments = ["train" if index < split_idx else "val" for index in range(len(samples))]
    elif strategy == "grouped_chronological":
        groups = sorted(
            {
                (
                    sample.surface_pair_id,
                    _canonical_timestamp(sample.current_snapshot_time_utc),
                    _canonical_timestamp(sample.target_snapshot_time_utc),
                )
                for sample in samples
            },
            key=lambda item: (item[1], item[2], item[0]),
        )
        train_end, val_end = _grouped_split_boundaries(len(groups), config.train_ratio, config.val_ratio)
        split_by_pair: dict[str, str] = {}
        for index, (pair_id, _current, _target) in enumerate(groups):
            split_by_pair[pair_id] = "train" if index < train_end else "val" if index < val_end else "test"
        assignments = [split_by_pair[sample.surface_pair_id] for sample in samples]
    else:
        raise ValueError(f"Unsupported split_strategy: {config.split_strategy}")

    rows = []
    for sample, split in zip(samples, assignments):
        rows.append(
            {
                "global_index": int(sample.global_index),
                "sample_id": sample.sample_id,
                "surface_pair_id": sample.surface_pair_id,
                "split": split,
                "news_timestamp_utc": _canonical_timestamp(sample.timestamp),
                "current_snapshot_time_utc": _canonical_timestamp(sample.current_snapshot_time_utc),
                "target_snapshot_time_utc": _canonical_timestamp(sample.target_snapshot_time_utc),
                "source_path": str(Path(config.data_path)),
                "source_sha256": workbook_hash,
                "sheet_name": str(config.sheet_name),
                "split_strategy": strategy,
            }
        )
    return pd.DataFrame(rows)


def write_split_manifest(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    output_path: str | Path,
) -> Path:
    clean_config = replace(config, split_manifest_path="")
    samples = load_film_wgan_samples(clean_config)
    frame = build_split_manifest_frame(clean_config, samples)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _load_and_validate_split_manifest(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    samples: Sequence[FilmWGANSample],
) -> pd.DataFrame:
    manifest_path = str(getattr(config, "split_manifest_path", "")).strip()
    if not manifest_path:
        return build_split_manifest_frame(config, samples)
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"split_manifest_path does not exist: {path}")
    frame = pd.read_csv(path)
    required = {
        "global_index",
        "sample_id",
        "surface_pair_id",
        "split",
        "source_sha256",
        "split_strategy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Split manifest {path} is missing required columns: {missing}")
    expected_hash = workbook_sha256(config.data_path)
    hashes = set(frame["source_sha256"].astype(str))
    if hashes != {expected_hash}:
        raise ValueError(
            f"Split manifest source SHA256 mismatch for {config.data_path}: "
            f"expected {expected_hash}, found {sorted(hashes)}"
        )
    if len(frame) != len(samples):
        raise ValueError(f"Split manifest row count mismatch: expected {len(samples)}, found {len(frame)}")
    frame = frame.sort_values("global_index").reset_index(drop=True)
    for sample, row in zip(samples, frame.itertuples(index=False)):
        if int(row.global_index) != int(sample.global_index):
            raise ValueError("Split manifest global_index order does not match the workbook samples.")
        if str(row.sample_id) != sample.sample_id or str(row.surface_pair_id) != sample.surface_pair_id:
            raise ValueError(f"Split manifest sample mismatch at global_index={sample.global_index}.")
    strategy = str(config.split_strategy).strip().lower()
    required_splits = {"train", "val"} if strategy == "legacy_row" else {"train", "val", "test"}
    allowed_splits = set(required_splits)
    if strategy == "grouped_chronological":
        allowed_splits.add("excluded")
    actual_splits = set(frame["split"].astype(str))
    if not actual_splits.issubset(allowed_splits) or not required_splits.issubset(actual_splits):
        raise ValueError(
            f"Split manifest must contain {sorted(required_splits)} and may additionally contain "
            f"{sorted(allowed_splits - required_splits)}, found {sorted(actual_splits)}"
        )
    return frame


def _partition_samples(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    samples: Sequence[FilmWGANSample],
) -> tuple[dict[str, list[FilmWGANSample]], pd.DataFrame]:
    manifest = _load_and_validate_split_manifest(config, samples)
    partitions: dict[str, list[FilmWGANSample]] = {"train": [], "val": [], "test": []}
    for sample, split in zip(samples, manifest["split"].astype(str)):
        if split == "excluded":
            continue
        partitions[split].append(sample)
    return partitions, manifest


def _fit_or_load_surface_support(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    train_samples: Sequence[FilmWGANSample],
) -> RawSurfaceSupportArtifact | None:
    mode = str(config.surface_support_mode).strip().lower()
    if mode == "full_grid":
        return None
    support_path = _resolved_config_path(str(config.surface_support_path))
    train_pair_ids = [sample.surface_pair_id for sample in train_samples]
    input_hash = workbook_sha256(config.data_path)
    if support_path.is_file():
        artifact = RawSurfaceSupportArtifact.load(support_path)
        validate_support_artifact_lineage(
            artifact,
            train_pair_ids=train_pair_ids,
            input_workbook_sha256=input_hash,
        )
    else:
        if isinstance(config, FilmWGANSampleConfig):
            raise FileNotFoundError(
                f"Generate-result cannot fit a missing raw support artifact: {support_path}"
            )
        records = []
        for sample in train_samples:
            if str(sample.metadata.get("surface_model", "")).strip().lower() != "raw":
                raise ValueError(
                    "surface_support_mode=raw_observed requires surface_model=raw for every sample."
                )
            records.append(
                {
                    "surface_pair_id": sample.surface_pair_id,
                    "current_surface_params": sample.metadata.get(
                        "current_surface_param_json",
                        "",
                    ),
                    "target_surface_params": sample.metadata.get(
                        "target_surface_param_json",
                        "",
                    ),
                }
            )
        artifact = fit_raw_support_artifact(
            records,
            input_workbook_sha256=input_hash,
            strike_bins=int(config.support_strike_bins),
            maturity_bins=int(config.support_maturity_bins),
            quantile_low=float(config.support_grid_quantile_low),
            quantile_high=float(config.support_grid_quantile_high),
        )
        artifact.save(support_path)
    return artifact


def _apply_surface_support(
    samples: Sequence[FilmWGANSample],
    artifact: RawSurfaceSupportArtifact | None,
    *,
    report_atm7_metric: bool,
) -> list[FilmWGANSample]:
    if artifact is None:
        return list(samples)
    strike_grid = np.asarray(artifact.strike_grid, dtype=np.float32)
    maturity_grid = np.asarray(artifact.maturity_days_grid, dtype=np.float32)
    transformed: list[FilmWGANSample] = []
    for sample in samples:
        current_params = parse_raw_surface_params(
            sample.metadata.get("current_surface_param_json", "")
        )
        target_params = parse_raw_surface_params(
            sample.metadata.get("target_surface_param_json", "")
        )
        current_mask = raw_support_mask(
            current_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        target_mask = raw_support_mask(
            target_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        current_surface = reconstruct_raw_surface(
            current_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        target_surface = reconstruct_raw_surface(
            target_params,
            strike_grid=strike_grid,
            maturity_days_grid=maturity_grid,
        )
        pair_mask = current_mask & target_mask
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "surface_support_mode": "raw_observed",
                "surface_support_method": artifact.method,
                "current_supported_cell_count": int(current_mask.sum()),
                "target_supported_cell_count": int(target_mask.sum()),
                "evaluation_supported_cell_count": int(pair_mask.sum()),
                "evaluation_supported_fraction": float(pair_mask.mean()),
                "report_atm7_metric": bool(report_atm7_metric),
            }
        )
        transformed.append(
            replace(
                sample,
                current_surface=current_surface,
                target_surface=target_surface,
                strike_grid=strike_grid.copy(),
                maturity_days_grid=maturity_grid.copy(),
                current_support_mask=current_mask,
                target_support_mask=target_mask,
                metadata=metadata,
            )
        )
    return transformed


def _prepare_partitioned_samples(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    samples: Sequence[FilmWGANSample],
) -> tuple[
    dict[str, list[FilmWGANSample]],
    pd.DataFrame,
    FilmWGANTextTransform | None,
    dict[str, list[FilmWGANSample]],
    dict[str, TextAlignmentPlan],
]:
    partitions, manifest = _partition_samples(config, samples)
    if str(config.sample_unit).strip().lower() == "surface_pair":
        news_workbook = None
        if (
            not str(config.pair_text_feature_path).strip()
            or str(config.text_lineage_mode).strip().lower() == "strict"
        ):
            news_path = Path(config.news_workbook_path)
            if not news_path.is_file():
                raise FileNotFoundError(f"news_workbook_path does not exist: {news_path}")
            news_workbook = pd.read_excel(news_path)
        partitions = {
            split: aggregate_surface_pair_samples(
                config,
                split_samples_raw,
                news_workbook=news_workbook,
            )
            for split, split_samples_raw in partitions.items()
        }
    if not partitions["train"]:
        raise ValueError("The resolved split contains no training samples.")
    support_artifact = _fit_or_load_surface_support(config, partitions["train"])
    partitions = {
        split: _apply_surface_support(
            split_samples_raw,
            support_artifact,
            report_atm7_metric=bool(config.report_atm7_metric),
        )
        for split, split_samples_raw in partitions.items()
    }
    if support_artifact is not None:
        minimum_cells = int(config.support_min_train_pair_cells)
        partitions = {
            split: [
                sample
                for sample in split_samples_raw
                if int(sample.evaluation_support_mask.sum()) >= minimum_cells
            ]
            for split, split_samples_raw in partitions.items()
        }
        if not partitions["train"]:
            raise ValueError(
                "No training pairs remain after applying the train-derived raw support grid "
                f"with support_min_train_pair_cells={minimum_cells}."
            )
    transform = _fit_or_load_transform(config, partitions["train"])
    transformed = {
        split: _apply_text_transform(
            split_samples_raw,
            transform,
            preprocessing_mode=config.text_preprocessing_mode,
        )
        for split, split_samples_raw in partitions.items()
    }
    alignment_mode = str(
        getattr(config, "text_alignment_mode", "matched")
    ).strip().lower()
    # Legacy matched configurations do not need a synthetic placebo plan.
    # Explicit v3 paths are loaded below for both arms; implicit plans are
    # needed only to reproduce the historical permuted-text behavior.
    needs_frozen_alignment = alignment_mode != "matched"
    alignment_plans: dict[str, TextAlignmentPlan] = {}
    alignment_plan_path_value = str(
        getattr(config, "text_alignment_plan_path", "")
    ).strip()
    frozen_plan_frame: pd.DataFrame | None = None
    required_frozen_splits: set[str] = set()
    if alignment_plan_path_value:
        alignment_plan_path = _resolved_config_path(alignment_plan_path_value)
        if not alignment_plan_path.is_file():
            raise FileNotFoundError(
                f"text_alignment_plan_path does not exist: {alignment_plan_path}"
            )
        frozen_plan_frame = pd.read_csv(alignment_plan_path)
        if isinstance(config, FilmWGANTrainConfig):
            # The v3 pilot is validation-only. Deliberately do not prepare or
            # consume an outer-test alignment plan in the training bundle.
            required_frozen_splits = {
                name for name in ("train", "val") if transformed[name]
            }
        else:
            requested = str(getattr(config, "split", "val")).strip().lower()
            required_frozen_splits = (
                {"train", "val", "test"} if requested == "all" else {requested}
            )
            if requested not in {"train", "val", "test", "all"}:
                raise ValueError(
                    "split must be one of ['train', 'val', 'test', 'all'], "
                    f"got: {requested}"
                )
            required_frozen_splits = {
                name for name in required_frozen_splits if transformed[name]
            }
        for required_split in sorted(required_frozen_splits):
            plan = TextAlignmentPlan.from_frame(
                frozen_plan_frame,
                split=required_split,
            )
            expected_sample_ids = tuple(
                sample.sample_id for sample in transformed[required_split]
            )
            expected_pair_ids = tuple(
                sample.surface_pair_id for sample in transformed[required_split]
            )
            if (
                plan.target_sample_ids != expected_sample_ids
                or plan.target_surface_pair_ids != expected_pair_ids
            ):
                raise ValueError(
                    "Frozen text alignment plan canonical item order/identities "
                    f"do not match split={required_split!r}."
                )
            expected_seed = int(getattr(config, "text_permutation_seed", 20260722))
            expected_effective_seed = expected_seed + {
                "train": 11,
                "val": 23,
                "test": 37,
            }.get(required_split, 0)
            if (
                int(plan.base_seed) != expected_seed
                or int(plan.effective_seed) != expected_effective_seed
            ):
                raise ValueError(
                    "Frozen text alignment plan seed mismatch for "
                    f"split={required_split!r}: expected base/effective "
                    f"{expected_seed}/{expected_effective_seed}, found "
                    f"{plan.base_seed}/{plan.effective_seed}."
                )
            alignment_plans[required_split] = plan
    aligned: dict[str, list[FilmWGANSample]] = {}
    for split, split_samples_transformed in transformed.items():
        if not split_samples_transformed:
            aligned[split] = []
            continue
        plan = alignment_plans.get(split)
        if frozen_plan_frame is not None and split not in required_frozen_splits:
            # In particular, a train bundle keeps the outer-test partition
            # native and untouched under the validation-pilot protocol.
            aligned[split] = list(split_samples_transformed)
            continue
        if plan is None and needs_frozen_alignment:
            # Empty path is the explicit legacy mode. It remains deterministic
            # for old callers, while v3 always loads the frozen CSV above.
            base_seed = int(
                getattr(config, "text_permutation_seed", 20260722)
            )
            placebo_sources = None
            if alignment_mode != "matched" and not alignment_plan_path_value:
                placebo_sources = _permutation_for_pair_ids(
                    [sample.surface_pair_id for sample in split_samples_transformed],
                    seed=base_seed
                    + {"train": 11, "val": 23, "test": 37}.get(split, 0),
                )
            plan = build_text_alignment_plan(
                split_samples_transformed,
                split=split,
                seed=base_seed,
                placebo_source_indices=placebo_sources,
            )
            alignment_plans[split] = plan
        aligned[split] = apply_text_alignment(
            config,
            split_samples_transformed,
            split=split,
            alignment_plan=plan,
        )
    return aligned, manifest, transform, transformed, alignment_plans


def _permutation_for_pair_ids(pair_ids: Sequence[str], *, seed: int) -> np.ndarray:
    """Reproduce the unpublished v1/v2 seeded permutation exactly.

    V3 training never relies on this implicit mapping: it loads a frozen
    :class:`TextAlignmentPlan`.  Keeping the historical NumPy procedure here
    preserves diagnostic inference for schema-5 shuffled checkpoints whose
    resolved configs predate the frozen-plan fields.
    """

    count = len(pair_ids)
    if count < 2 or len(set(pair_ids)) < 2:
        raise ValueError(
            "Permuted text requires at least two distinct surface pairs in each split."
        )
    rng = np.random.default_rng(int(seed))
    base = np.arange(count, dtype=np.int64)
    for _attempt in range(4096):
        donors = rng.permutation(base)
        if all(
            pair_ids[index] != pair_ids[int(donor)]
            for index, donor in enumerate(donors)
        ):
            return donors
    raise ValueError(
        "Could not build a same-pair-free text permutation for this split."
    )


def build_text_permutation_mapping(split_manifest: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Build the exact split-local shuffled-text donor mapping used by the loader."""

    required = {"global_index", "sample_id", "surface_pair_id", "split"}
    missing = sorted(required - set(split_manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing permutation fields: {missing}")
    offsets = {"train": 11, "val": 23, "test": 37}
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        partition = (
            split_manifest[split_manifest["split"].astype(str) == split_name]
            .sort_values("global_index")
            .reset_index(drop=True)
        )
        if partition.empty:
            continue
        donors = _permutation_for_pair_ids(
            partition["surface_pair_id"].astype(str).tolist(),
            seed=int(seed) + offsets[split_name],
        )
        for target_index, donor_index in enumerate(donors):
            target = partition.iloc[target_index]
            donor = partition.iloc[int(donor_index)]
            rows.append(
                {
                    "split": split_name,
                    "target_global_index": int(target["global_index"]),
                    "target_sample_id": str(target["sample_id"]),
                    "target_surface_pair_id": str(target["surface_pair_id"]),
                    "donor_global_index": int(donor["global_index"]),
                    "donor_sample_id": str(donor["sample_id"]),
                    "donor_surface_pair_id": str(donor["surface_pair_id"]),
                    "permutation_seed": int(seed) + offsets[split_name],
                }
            )
    return pd.DataFrame(rows)


def apply_text_alignment(
    config: FilmWGANTrainConfig | FilmWGANSampleConfig,
    samples: Sequence[FilmWGANSample],
    *,
    split: str,
    alignment_plan: TextAlignmentPlan | None = None,
) -> list[FilmWGANSample]:
    items = list(samples)
    mode = str(getattr(config, "text_alignment_mode", "matched")).strip().lower()
    if alignment_plan is None:
        if mode == "matched":
            return items
        base_seed = int(getattr(config, "text_permutation_seed", 20260722))
        placebo_sources = _permutation_for_pair_ids(
            [sample.surface_pair_id for sample in items],
            seed=base_seed + {"train": 11, "val": 23, "test": 37}.get(split, 0),
        )
        alignment_plan = build_text_alignment_plan(
            items,
            split=split,
            seed=base_seed,
            placebo_source_indices=placebo_sources,
        )
    if alignment_plan.target_sample_ids != tuple(sample.sample_id for sample in items):
        raise ValueError("Text alignment plan sample order does not match the supplied split.")
    donors = alignment_plan.positive_source_indices(mode)
    aligned: list[FilmWGANSample] = []
    for index, donor_index in enumerate(donors):
        sample = items[index]
        donor = items[int(donor_index)]
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "text_alignment_mode": "matched" if mode == "matched" else "permuted",
                "text_alignment_plan_version": alignment_plan.mapping_version,
                "text_alignment_plan_sha256": alignment_plan.sha256,
                "text_native_source_index": index,
                "text_positive_source_index": int(donor_index),
                "text_placebo_source_index": int(
                    alignment_plan.placebo_source_indices[index]
                ),
                "text_source_sample_id": donor.sample_id,
                "text_source_surface_pair_id": donor.surface_pair_id,
                "text_source_article_ids": list(
                    donor.metadata.get("article_ids", [])
                ),
                "text_source_files": list(
                    donor.metadata.get("source_files", [])
                ),
                "text_source_event_group": donor.metadata.get("event_group", ""),
                "text_source_news_cluster_id": donor.metadata.get(
                    "news_cluster_id",
                    "",
                ),
                "text_source_feature_sha256": donor.metadata.get(
                    "pair_text_feature_sha256",
                    "",
                ),
            }
        )
        aligned.append(replace(sample, text_embedding=donor.text_embedding.copy(), metadata=metadata))
    return aligned


def _check_consistent_grids(samples: Sequence[FilmWGANSample]) -> tuple[np.ndarray, np.ndarray]:
    strike_grid = samples[0].strike_grid
    maturity_days_grid = samples[0].maturity_days_grid
    for sample in samples[1:]:
        if sample.strike_grid.shape != strike_grid.shape or not np.allclose(sample.strike_grid, strike_grid):
            raise ValueError("All standalone FiLM WGAN samples must share the same strike grid.")
        if sample.maturity_days_grid.shape != maturity_days_grid.shape or not np.allclose(
            sample.maturity_days_grid,
            maturity_days_grid,
        ):
            raise ValueError("All standalone FiLM WGAN samples must share the same maturity grid.")
    return strike_grid.copy(), maturity_days_grid.copy()


def create_train_val_bundle(config: FilmWGANTrainConfig) -> FilmWGANDataBundle:
    samples = load_film_wgan_samples(config)
    if len(samples) < int(config.min_samples_for_training):
        raise ValueError(f"Standalone FiLM WGAN requires at least {config.min_samples_for_training} samples.")
    (
        partitions,
        split_manifest,
        transform,
        native_partitions,
        alignment_plans,
    ) = _prepare_partitioned_samples(config, samples)
    train_items = partitions["train"]
    val_items = partitions["val"]
    test_items = partitions["test"]
    strike_grid, maturity_days_grid = _check_consistent_grids(
        [*train_items, *val_items, *test_items]
    )
    normalization_stats = _compute_normalization_stats(train_items)
    support_enabled = str(config.surface_support_mode).strip().lower() == "raw_observed"
    matching_enabled = (
        str(config.critic_conditioning_mode).strip().lower()
        == "transition_matching"
    )
    train_loader = DataLoader(
        FilmWGANDataset(
            train_items,
            normalization_stats,
            normalize_current_surface=config.normalize_current_surface,
            normalize_target_delta=config.normalize_target_delta,
            normalize_text_embedding=config.normalize_text_embedding,
            include_has_text=str(config.conditioning_mode).strip().lower() == "residual_film",
            include_support_mask=support_enabled,
            include_sample_index=matching_enabled,
        ),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
    )
    val_loader = None
    if val_items:
        val_loader = DataLoader(
            FilmWGANDataset(
                val_items,
                normalization_stats,
                normalize_current_surface=config.normalize_current_surface,
                normalize_target_delta=config.normalize_target_delta,
                normalize_text_embedding=config.normalize_text_embedding,
                include_has_text=str(config.conditioning_mode).strip().lower() == "residual_film",
                include_support_mask=support_enabled,
                include_sample_index=matching_enabled,
            ),
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
        )
    test_loader = None
    if test_items:
        test_loader = DataLoader(
            FilmWGANDataset(
                test_items,
                normalization_stats,
                normalize_current_surface=config.normalize_current_surface,
                normalize_target_delta=config.normalize_target_delta,
                normalize_text_embedding=config.normalize_text_embedding,
                include_has_text=str(config.conditioning_mode).strip().lower() == "residual_film",
                include_support_mask=support_enabled,
                include_sample_index=matching_enabled,
            ),
            batch_size=int(config.batch_size),
            shuffle=False,
            num_workers=int(config.num_workers),
        )
    return FilmWGANDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_samples=len(train_items),
        val_samples=len(val_items),
        test_samples=len(test_items),
        surface_shape=train_items[0].surface_shape,
        strike_grid=strike_grid,
        maturity_days_grid=maturity_days_grid,
        embedding_dim=int(train_items[0].text_embedding.size),
        normalization_stats=normalization_stats,
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        all_items=[*train_items, *val_items, *test_items],
        split_manifest=split_manifest,
        text_transform_path=str(config.text_transform_path),
        text_transform_sha256=(
            sha256_file(config.text_transform_path)
            if str(config.text_transform_path).strip() and Path(config.text_transform_path).is_file()
            else ""
        ),
        surface_support_path=str(config.surface_support_path),
        surface_support_sha256=(
            sha256_file(_resolved_config_path(str(config.surface_support_path)))
            if str(config.surface_support_path).strip()
            and _resolved_config_path(str(config.surface_support_path)).is_file()
            else ""
        ),
        native_train_items=native_partitions["train"],
        native_val_items=native_partitions["val"],
        native_test_items=native_partitions["test"],
        text_alignment_plans=alignment_plans,
    )


def select_samples(samples: Sequence[FilmWGANSample], *, selection_mode: str, selection_count: int) -> list[FilmWGANSample]:
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


def split_samples(config: FilmWGANSampleConfig, samples: Sequence[FilmWGANSample]) -> list[FilmWGANSample]:
    partitions, _manifest, _transform, _native, _plans = _prepare_partitioned_samples(config, samples)
    normalized_split = str(config.split).strip().lower()
    if normalized_split == "train":
        selected = partitions["train"]
    elif normalized_split == "val":
        selected = partitions["val"]
    elif normalized_split == "test":
        selected = partitions["test"]
    elif normalized_split == "all":
        selected = []
        for split_name in ("train", "val", "test"):
            if partitions[split_name]:
                selected.extend(partitions[split_name])
    else:
        raise ValueError(f"split must be one of ['train', 'val', 'test', 'all'], got: {config.split}")
    return select_samples(selected, selection_mode=config.selection_mode, selection_count=config.selection_count)
