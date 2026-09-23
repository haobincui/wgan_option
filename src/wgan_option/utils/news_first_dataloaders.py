"""Explicit, pair-balanced splits for news-first volatility experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from wgan_option.config import Config, normalize_label_reliability_mode

from .merged_xlsx_parsing import _align_embeddings
from .merged_xlsx_samples import (
    load_vol_surface_samples,
    load_vol_surface_samples_with_diagnostics,
)
from .merged_xlsx_types import VolSurfaceSample, VolSurfaceXlsxBundle
from .news_first_experiment_core import (
    NO_PAIR_TEXT_OVERLAY,
    apply_pair_text_overlay,
    load_pair_text_overlay_manifest,
    normalize_pair_text_overlay_mode,
    pair_universe_sha256,
)
from .reproducibility import seeded_torch_generator
from .text_ablation import (
    normalize_text_ablation_mode,
    text_information_path,
    transform_embedding_matrix,
)
from .weighted_training import stable_key_to_int64

LABEL_RELIABILITY_MANIFEST_COLUMNS = (
    "tolerance_minutes",
    "fold_id",
    "pair_id",
    "included",
    "normalized_label_weight",
    "reliability_score",
)
LABEL_RELIABILITY_TRAIN_UNIVERSE_COLUMN = "train_pair_universe_sha256"
_LABEL_RELIABILITY_FILTER_MODES = frozenset(
    {"support_filter", "support_filter_soft_weight"}
)
_LABEL_RELIABILITY_SOFT_WEIGHT_MODES = frozenset(
    {"soft_weight", "support_filter_soft_weight"}
)


@dataclass(frozen=True)
class NewsFirstSplitSpec:
    """Fixed out-of-time boundaries shared by all wait-tolerance datasets."""

    train_end_utc: str = "2023-07-01T00:00:00Z"
    validation_end_utc: str = "2023-10-01T00:00:00Z"

    def parsed_boundaries(
        self,
        *,
        allow_empty_validation_window: bool = False,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        train_end = _utc_timestamp(self.train_end_utc, label="train_end_utc")
        validation_end = _utc_timestamp(
            self.validation_end_utc,
            label="validation_end_utc",
        )
        if validation_end < train_end or (
            validation_end == train_end and not allow_empty_validation_window
        ):
            qualifier = (
                "at or later than" if allow_empty_validation_window else "later than"
            )
            raise ValueError(f"validation_end_utc must be {qualifier} train_end_utc.")
        return train_end, validation_end


@dataclass(frozen=True)
class NewsFirstVolSplitSelection:
    """Materialized train/common-validation/common-test sample selections."""

    train_items: list[VolSurfaceSample]
    val_items: list[VolSurfaceSample]
    test_items: list[VolSurfaceSample]
    train_end_utc: str
    validation_end_utc: str


def _utc_timestamp(value: str, *, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a valid UTC timestamp, got: {value}")
    return pd.Timestamp(parsed)


def _sample_origin(sample: VolSurfaceSample) -> pd.Timestamp:
    raw_value = str(sample.effective_origin_utc).strip()
    if not raw_value:
        raw_value = str(sample.current_snapshot_time_utc).strip()
    parsed = pd.to_datetime(raw_value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(
            f"Sample {sample.sample_id} has no valid effective_origin_utc/current snapshot."
        )
    return pd.Timestamp(parsed)


def _load_surface_items(
    config: Config,
) -> tuple[list[VolSurfaceSample], dict[str, object]]:
    """Keep the historical unmasked loader patch point while adding diagnostics."""

    support_mode = str(getattr(config, "support_mask_mode", "none") or "none").lower()
    if support_mode == "raw_joint":
        return load_vol_surface_samples_with_diagnostics(config)
    items = load_vol_surface_samples(config)
    train_end, validation_end = NewsFirstSplitSpec(
        train_end_utc=str(config.news_first_train_end_utc),
        validation_end_utc=str(config.news_first_validation_end_utc),
    ).parsed_boundaries(
        allow_empty_validation_window=not bool(
            getattr(config, "news_first_materialize_validation_loader", True)
        )
    )

    def partition_summary(name: str) -> dict[str, object]:
        if name == "train":
            selected = [item for item in items if _sample_origin(item) < train_end]
        elif name == "validation":
            selected = [
                item
                for item in items
                if train_end <= _sample_origin(item) < validation_end
            ]
        else:
            selected = [
                item for item in items if _sample_origin(item) >= validation_end
            ]
        return {
            "input_rows": len(selected),
            "input_pairs": len(_pair_sets(selected)),
            "input_sessions": len(
                {item.session_id for item in selected if item.session_id}
            ),
            "excluded_zero_joint_support_rows": 0,
            "excluded_zero_joint_support_pairs": 0,
            "excluded_zero_joint_support_sessions": 0,
            "kept_rows": len(selected),
            "kept_pairs": len(_pair_sets(selected)),
            "kept_sessions": len(
                {item.session_id for item in selected if item.session_id}
            ),
            "joint_support_cell_count": {},
        }

    first = items[0]
    return items, {
        "support_mask_mode": "none",
        "support_mask_applied": False,
        "input_rows": len(items),
        "loaded_rows": len(items),
        "excluded_zero_joint_support_rows": 0,
        "grid_cell_count": int(np.prod(first.surface_shape)),
        "grid_fingerprint": first.support_grid_fingerprint,
        "support_method": "",
        "time_partitions": {
            name: partition_summary(name) for name in ("train", "validation", "test")
        },
    }


def _pair_key(sample: VolSurfaceSample) -> str:
    pair_id = str(sample.pair_id).strip()
    return (
        pair_id
        if pair_id
        else f"__sample__:{sample.stable_sample_key or sample.sample_id}"
    )


def _pair_sets(items: Sequence[VolSurfaceSample]) -> set[str]:
    return {_pair_key(sample) for sample in items}


def _session_sets(
    items: Sequence[VolSurfaceSample],
    *,
    split_name: str,
) -> set[str]:
    missing = [
        sample.sample_id for sample in items if not str(sample.session_id).strip()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"session_id is required for the explicit news-first {split_name} split; "
            f"missing for {len(missing)} sample(s): {preview}{suffix}"
        )
    return {str(sample.session_id).strip() for sample in items}


def _with_pair_balanced_weights(
    items: Sequence[VolSurfaceSample],
    *,
    scale_for_training: bool,
) -> list[VolSurfaceSample]:
    counts: dict[str, int] = {}
    for sample in items:
        key = _pair_key(sample)
        counts[key] = counts.get(key, 0) + 1
    row_count = len(items)
    pair_count = len(counts)
    population_scale = float(row_count) / float(pair_count) if pair_count else 1.0

    weighted: list[VolSurfaceSample] = []
    for sample in items:
        raw_weight = 1.0 / float(counts[_pair_key(sample)])
        scaled_weight = (
            raw_weight * population_scale if scale_for_training else raw_weight
        )
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "source_sample_weight": float(
                    metadata.get("source_sample_weight", sample.sample_weight)
                ),
                "raw_pair_sample_weight": raw_weight,
                "scaled_training_sample_weight": scaled_weight,
                "pair_weight_population_scale": population_scale,
            }
        )
        weighted.append(
            replace(
                sample,
                sample_weight=raw_weight,
                metadata=metadata,
            )
        )
    return weighted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_manifest_bool(value: object, *, pair_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(
        "Label-reliability manifest included must be true/false/1/0; "
        f"pair_id={pair_id!r}, value={value!r}"
    )


def _canonical_label_reliability_records(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Normalize the six persisted profile columns before hashing or use."""

    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    missing = sorted(set(LABEL_RELIABILITY_MANIFEST_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Label-reliability manifest is missing required columns: {missing}"
        )
    canonical: list[dict[str, object]] = []
    for row_index, row in frame.iterrows():
        pair_id = str(row["pair_id"]).strip()
        fold_id = str(row["fold_id"]).strip()
        if not pair_id or not fold_id:
            raise ValueError(
                "Label-reliability manifest pair_id and fold_id must be non-empty; "
                f"row={row_index}"
            )
        try:
            tolerance_value = float(row["tolerance_minutes"])
            normalized_weight = float(row["normalized_label_weight"])
            reliability_score = float(row["reliability_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Label-reliability manifest numeric fields must be parseable; "
                f"pair_id={pair_id!r}"
            ) from exc
        if not np.isfinite(tolerance_value) or not tolerance_value.is_integer():
            raise ValueError(
                "Label-reliability tolerance_minutes must be a finite integer; "
                f"pair_id={pair_id!r}"
            )
        if not np.isfinite(normalized_weight) or not np.isfinite(reliability_score):
            raise ValueError(
                "Label-reliability weights and scores must be finite; "
                f"pair_id={pair_id!r}"
            )
        if reliability_score < 0.0 or reliability_score > 1.0:
            raise ValueError(
                "reliability_score must be within [0, 1]; "
                f"pair_id={pair_id!r}, value={reliability_score}"
            )
        included = _parse_manifest_bool(row["included"], pair_id=pair_id)
        if included and (normalized_weight < 0.5 or normalized_weight > 2.0):
            raise ValueError(
                "Included normalized_label_weight must be within [0.5, 2.0]; "
                f"pair_id={pair_id!r}, value={normalized_weight}"
            )
        if not included and normalized_weight != 0.0:
            raise ValueError(
                "Excluded manifest pairs must have normalized_label_weight=0; "
                f"pair_id={pair_id!r}, value={normalized_weight}"
            )
        canonical.append(
            {
                "tolerance_minutes": int(tolerance_value),
                "fold_id": fold_id,
                "pair_id": pair_id,
                "included": included,
                "normalized_label_weight": normalized_weight,
                "reliability_score": reliability_score,
            }
        )
    canonical.sort(key=lambda row: str(row["pair_id"]))
    return canonical


def label_reliability_profile_sha256(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
) -> str:
    """Hash canonical pair rows for one arm/tolerance/fold profile."""

    records = _canonical_label_reliability_records(rows)
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def label_reliability_train_pair_universe_sha256(
    pair_ids: Sequence[object],
    *,
    fold_id: str,
    tolerance_minutes: int,
) -> str:
    """Hash the exact pre-filter train-pair universe for one fold/tolerance."""

    normalized_fold = str(fold_id).strip()
    normalized_pairs = sorted({str(pair_id).strip() for pair_id in pair_ids})
    if (
        not normalized_fold
        or not normalized_pairs
        or any(not pair_id for pair_id in normalized_pairs)
    ):
        raise ValueError("Train-pair universe fold_id/pair_ids must be non-empty.")
    payload = {
        "schema_version": 1,
        "fold_id": normalized_fold,
        "tolerance_minutes": int(tolerance_minutes),
        "pair_ids": normalized_pairs,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_window_timestamp(value: str, *, label: str) -> str:
    return _utc_timestamp(value, label=label).isoformat()


def label_reliability_data_window_contract_sha256(
    *,
    fold_id: str,
    train_data_sha256: str,
    validation_data_sha256: str,
    train_end_utc: str,
    validation_end_utc: str,
) -> str:
    """Hash the two immutable fold workbooks and their half-open origin windows."""

    normalized_fold = str(fold_id).strip()
    if not normalized_fold:
        raise ValueError("Label-reliability data-window fold_id must be non-empty.")
    train_sha = str(train_data_sha256).strip().lower()
    validation_sha = str(validation_data_sha256).strip().lower()
    for label, digest in (
        ("train_data_sha256", train_sha),
        ("validation_data_sha256", validation_sha),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{label} must be a lowercase 64-character SHA256 digest.")
    train_end = _canonical_window_timestamp(train_end_utc, label="train_end_utc")
    validation_end = _canonical_window_timestamp(
        validation_end_utc,
        label="validation_end_utc",
    )
    if _utc_timestamp(validation_end, label="validation_end_utc") <= _utc_timestamp(
        train_end,
        label="train_end_utc",
    ):
        raise ValueError("validation_end_utc must be later than train_end_utc.")
    payload = {
        "schema_version": 1,
        "fold_id": normalized_fold,
        "train": {
            "origin_start_utc_inclusive": None,
            "origin_end_utc_exclusive": train_end,
            "workbook_sha256": train_sha,
        },
        "validation": {
            "origin_start_utc_inclusive": train_end,
            "origin_end_utc_exclusive": validation_end,
            "workbook_sha256": validation_sha,
        },
        "post_validation": {
            "forbidden_from_utc_inclusive": validation_end,
            "materialized": False,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_label_reliability_data_window_contract(
    config: Config,
    common_eval_data_path: str,
    *,
    split_spec: NewsFirstSplitSpec,
) -> dict[str, str]:
    """Fail closed unless formal jobs use the declared immutable fold inputs."""

    if not str(config.news_first_label_reliability_manifest_path).strip():
        return {}
    declared_common_path = str(config.news_first_common_eval_data_path).strip()
    if Path(declared_common_path).resolve() != Path(common_eval_data_path).resolve():
        raise ValueError(
            "Formal label-reliability common_eval_data_path differs from the "
            "hash-bound config path."
        )
    configured_train_end = _canonical_window_timestamp(
        config.news_first_train_end_utc,
        label="news_first_train_end_utc",
    )
    configured_validation_end = _canonical_window_timestamp(
        config.news_first_validation_end_utc,
        label="news_first_validation_end_utc",
    )
    if configured_train_end != _canonical_window_timestamp(
        split_spec.train_end_utc,
        label="train_end_utc",
    ) or configured_validation_end != _canonical_window_timestamp(
        split_spec.validation_end_utc,
        label="validation_end_utc",
    ):
        raise ValueError(
            "Formal label-reliability split bounds differ between config and loader."
        )
    train_path = Path(str(config.data_path))
    validation_path = Path(str(common_eval_data_path))
    if train_path.resolve() == validation_path.resolve():
        raise ValueError(
            "Formal label-reliability train and validation inputs must be separate "
            "fold-scoped workbooks."
        )
    if not train_path.is_file():
        raise FileNotFoundError(
            f"Fold-scoped training workbook does not exist: {train_path}"
        )
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Fold-scoped validation workbook does not exist: {validation_path}"
        )
    actual_train_sha = _sha256_file(train_path)
    actual_validation_sha = _sha256_file(validation_path)
    if actual_train_sha != config.news_first_label_reliability_train_data_sha256:
        raise ValueError(
            "Fold-scoped training workbook SHA256 mismatch: "
            f"expected {config.news_first_label_reliability_train_data_sha256}, "
            f"got {actual_train_sha}"
        )
    if (
        actual_validation_sha
        != config.news_first_label_reliability_validation_data_sha256
    ):
        raise ValueError(
            "Fold-scoped validation workbook SHA256 mismatch: "
            f"expected {config.news_first_label_reliability_validation_data_sha256}, "
            f"got {actual_validation_sha}"
        )
    actual_contract_sha = label_reliability_data_window_contract_sha256(
        fold_id=config.news_first_label_reliability_fold_id,
        train_data_sha256=actual_train_sha,
        validation_data_sha256=actual_validation_sha,
        train_end_utc=split_spec.train_end_utc,
        validation_end_utc=split_spec.validation_end_utc,
    )
    if (
        actual_contract_sha
        != config.news_first_label_reliability_data_window_contract_sha256
    ):
        raise ValueError(
            "Label-reliability data-window contract SHA256 mismatch: "
            f"expected {config.news_first_label_reliability_data_window_contract_sha256}, "
            f"got {actual_contract_sha}"
        )
    return {
        "train_data_sha256": actual_train_sha,
        "validation_data_sha256": actual_validation_sha,
        "data_window_contract_sha256": actual_contract_sha,
    }


def _load_label_reliability_profile(
    config: Config,
    train_items: Sequence[VolSurfaceSample],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Load one declared train-fold profile and fail closed on any drift."""

    manifest_path = Path(config.news_first_label_reliability_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Label-reliability manifest does not exist: {manifest_path}"
        )
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != config.news_first_label_reliability_manifest_sha256:
        raise ValueError(
            "Label-reliability manifest SHA256 mismatch: "
            f"expected {config.news_first_label_reliability_manifest_sha256}, "
            f"got {actual_manifest_sha}"
        )
    try:
        raw_frame = pd.read_csv(
            manifest_path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to read label-reliability manifest: {manifest_path}"
        ) from exc
    records = _canonical_label_reliability_records(raw_frame)
    if LABEL_RELIABILITY_TRAIN_UNIVERSE_COLUMN not in raw_frame.columns:
        raise ValueError(
            "Label-reliability manifest is missing required column: "
            f"{LABEL_RELIABILITY_TRAIN_UNIVERSE_COLUMN}"
        )
    key_counts: dict[tuple[int, str, str], int] = {}
    for row in records:
        key = (
            int(row["tolerance_minutes"]),
            str(row["fold_id"]),
            str(row["pair_id"]),
        )
        key_counts[key] = key_counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in key_counts.items() if count != 1)
    if duplicates:
        raise ValueError(
            "Label-reliability manifest keys must be unique; duplicate keys: "
            f"{duplicates[:5]}"
        )

    tolerance = int(config.news_first_dataset_tolerance_minutes)
    fold_id = config.news_first_label_reliability_fold_id
    selected = [
        row
        for row in records
        if int(row["tolerance_minutes"]) == tolerance and str(row["fold_id"]) == fold_id
    ]
    if not selected:
        raise ValueError(
            "Label-reliability manifest contains no rows for "
            f"tolerance={tolerance}, fold_id={fold_id!r}."
        )
    actual_profile_sha = label_reliability_profile_sha256(selected)
    if actual_profile_sha != config.news_first_label_reliability_profile_sha256:
        raise ValueError(
            "Label-reliability profile SHA256 mismatch: "
            f"expected {config.news_first_label_reliability_profile_sha256}, "
            f"got {actual_profile_sha}"
        )

    missing_pair_ids = sorted(
        {sample.sample_id for sample in train_items if not str(sample.pair_id).strip()}
    )
    if missing_pair_ids:
        raise ValueError(
            "Label-reliability training requires a non-empty pair_id for every row; "
            f"missing on: {missing_pair_ids[:5]}"
        )
    train_pairs = _pair_sets(train_items)
    manifest_pairs = {str(row["pair_id"]) for row in selected}
    if train_pairs != manifest_pairs:
        raise ValueError(
            "Label-reliability profile pair universe differs from the raw training "
            f"split; missing={sorted(train_pairs - manifest_pairs)[:5]}, "
            f"extra={sorted(manifest_pairs - train_pairs)[:5]}"
        )
    selected_mask = (
        pd.to_numeric(raw_frame["tolerance_minutes"], errors="coerce") == tolerance
    ) & (raw_frame["fold_id"].astype(str).str.strip() == fold_id)
    declared_universe_hashes = {
        str(value).strip().lower()
        for value in raw_frame.loc[
            selected_mask,
            LABEL_RELIABILITY_TRAIN_UNIVERSE_COLUMN,
        ]
        if str(value).strip()
    }
    if len(declared_universe_hashes) != 1:
        raise ValueError(
            "Selected label-reliability manifest rows must declare exactly one "
            "train_pair_universe_sha256."
        )
    declared_universe_sha = next(iter(declared_universe_hashes))
    config_universe_sha = config.news_first_label_reliability_train_pair_universe_sha256
    actual_universe_sha = label_reliability_train_pair_universe_sha256(
        sorted(train_pairs),
        fold_id=fold_id,
        tolerance_minutes=tolerance,
    )
    if declared_universe_sha != config_universe_sha:
        raise ValueError(
            "Label-reliability train-pair universe SHA256 differs between manifest "
            f"and config: {declared_universe_sha} != {config_universe_sha}"
        )
    if actual_universe_sha != config_universe_sha:
        raise ValueError(
            "Label-reliability train-pair universe SHA256 mismatch: "
            f"expected {config_universe_sha}, got {actual_universe_sha}"
        )
    by_pair = {str(row["pair_id"]): row for row in selected}
    return by_pair, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "profile_sha256": actual_profile_sha,
        "fold_id": fold_id,
        "tolerance_minutes": tolerance,
        "manifest_pair_count": len(selected),
        "train_pair_universe_sha256": actual_universe_sha,
    }


def _apply_label_reliability_contract(
    selection: NewsFirstVolSplitSelection,
    config: Config,
    *,
    materialize_evaluation_items: bool = True,
) -> tuple[NewsFirstVolSplitSelection, dict[str, object]]:
    """Apply train-pair filtering/weighting while keeping evaluation neutral."""

    mode = normalize_label_reliability_mode(config.news_first_label_reliability_mode)
    lineage_declared = bool(config.news_first_label_reliability_manifest_path)
    original_train_items = list(selection.train_items)
    original_train_pairs = _pair_sets(original_train_items)
    manifest_audit: dict[str, object] = {
        "manifest_path": "",
        "manifest_sha256": "",
        "profile_sha256": "",
        "fold_id": "",
        "tolerance_minutes": int(config.news_first_dataset_tolerance_minutes),
        "manifest_pair_count": 0,
        "train_pair_universe_sha256": "",
    }
    profile_by_pair: dict[str, dict[str, object]] = {}
    if lineage_declared:
        profile_by_pair, manifest_audit = _load_label_reliability_profile(
            config,
            original_train_items,
        )

    filtered_items: list[VolSurfaceSample] = []
    effective_pair_weights: dict[str, float] = {}
    for sample in original_train_items:
        pair_id = _pair_key(sample)
        profile_row = profile_by_pair.get(pair_id)
        included = True if profile_row is None else bool(profile_row["included"])
        if mode in _LABEL_RELIABILITY_FILTER_MODES and not included:
            continue
        label_weight = (
            float(profile_row["normalized_label_weight"])
            if mode in _LABEL_RELIABILITY_SOFT_WEIGHT_MODES and profile_row is not None
            else 1.0
        )
        effective_pair_weights[pair_id] = label_weight
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "label_reliability_mode": mode,
                "label_reliability_lineage_validated": lineage_declared,
                "label_reliability_included": included,
                "label_reliability_score": (
                    float(profile_row["reliability_score"])
                    if profile_row is not None
                    else 1.0
                ),
                "manifest_normalized_label_weight": (
                    float(profile_row["normalized_label_weight"])
                    if profile_row is not None
                    else 1.0
                ),
                "label_reliability_weight": label_weight,
            }
        )
        filtered_items.append(
            replace(
                sample,
                label_reliability_weight=label_weight,
                metadata=metadata,
            )
        )
    if not filtered_items:
        raise ValueError("Label-reliability filtering removed every training pair.")
    if mode in _LABEL_RELIABILITY_SOFT_WEIGHT_MODES:
        pair_weight_mean = float(np.mean(list(effective_pair_weights.values())))
        if not np.isclose(pair_weight_mean, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(
                "normalized_label_weight must have pair-level mean 1 over the "
                f"effective training population; got {pair_weight_mean:.12g}"
            )
    else:
        pair_weight_mean = 1.0

    # A hard filter changes pair cardinalities and therefore requires a fresh
    # population-scale normalization of the base pair-balanced weights.
    train_items = _with_pair_balanced_weights(
        filtered_items,
        scale_for_training=True,
    )

    def neutral_evaluation_items(
        items: Sequence[VolSurfaceSample],
    ) -> list[VolSurfaceSample]:
        neutral: list[VolSurfaceSample] = []
        for sample in items:
            metadata = dict(sample.metadata)
            metadata.update(
                {
                    "label_reliability_mode": mode,
                    "label_reliability_lineage_validated": lineage_declared,
                    "label_reliability_weight": 1.0,
                    "label_reliability_evaluation_neutral": True,
                }
            )
            neutral.append(
                replace(
                    sample,
                    label_reliability_weight=1.0,
                    metadata=metadata,
                )
            )
        return neutral

    kept_pairs = _pair_sets(train_items)
    audit = {
        "mode": mode,
        **manifest_audit,
        "lineage_validated": lineage_declared,
        "raw_train_rows": len(original_train_items),
        "raw_train_pairs": len(original_train_pairs),
        "kept_train_rows": len(train_items),
        "kept_train_pairs": len(kept_pairs),
        "filtered_train_rows": len(original_train_items) - len(train_items),
        "filtered_train_pairs": len(original_train_pairs) - len(kept_pairs),
        "effective_pair_weight_mean": pair_weight_mean,
        "effective_pair_weight_min": float(min(effective_pair_weights.values())),
        "effective_pair_weight_max": float(max(effective_pair_weights.values())),
        "evaluation_label_weight": 1.0,
    }
    if materialize_evaluation_items:
        val_items = neutral_evaluation_items(selection.val_items)
        test_items = neutral_evaluation_items(selection.test_items)
    else:
        if selection.val_items or selection.test_items:
            raise ValueError(
                "Evaluation items must be empty when evaluation materialization is disabled."
            )
        val_items = []
        test_items = []
    return (
        replace(
            selection,
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
        ),
        audit,
    )


def select_news_first_vol_splits(
    training_items: Sequence[VolSurfaceSample],
    common_evaluation_items: Sequence[VolSurfaceSample],
    *,
    split_spec: NewsFirstSplitSpec = NewsFirstSplitSpec(),
    materialize_validation_items: bool = True,
    materialize_test_items: bool = True,
) -> NewsFirstVolSplitSelection:
    """Select tolerance-specific training and common 5m validation/test rows."""

    train_end, validation_end = split_spec.parsed_boundaries(
        allow_empty_validation_window=not materialize_validation_items
    )
    train_items = [
        sample for sample in training_items if _sample_origin(sample) < train_end
    ]
    if materialize_validation_items:
        val_items = [
            sample
            for sample in common_evaluation_items
            if train_end <= _sample_origin(sample) < validation_end
        ]
    else:
        if materialize_test_items:
            raise ValueError(
                "test items cannot be materialized when validation items are disabled"
            )
        val_items = []
    if materialize_test_items:
        test_items = [
            sample
            for sample in common_evaluation_items
            if _sample_origin(sample) >= validation_end
        ]
    else:
        if any(_sample_origin(sample) >= train_end for sample in training_items):
            raise ValueError(
                "Fold-scoped training input contains a row outside its declared "
                "origin<train_end window."
            )
        if any(
            not (train_end <= _sample_origin(sample) < validation_end)
            for sample in common_evaluation_items
        ):
            raise ValueError(
                "Fold-scoped validation input contains a row outside its declared "
                "half-open inner-validation window."
            )
        test_items = []
    if not train_items:
        raise ValueError("The explicit news-first training split is empty.")
    if materialize_validation_items and not val_items:
        raise ValueError("The common news-first validation split is empty.")
    if materialize_test_items and not test_items:
        raise ValueError("The common news-first test split is empty.")

    split_pair_sets = {"train": _pair_sets(train_items)}
    if materialize_validation_items:
        split_pair_sets["val"] = _pair_sets(val_items)
    if materialize_test_items:
        split_pair_sets["test"] = _pair_sets(test_items)
    overlaps = {}
    if materialize_validation_items:
        overlaps["train_val"] = split_pair_sets["train"] & split_pair_sets["val"]
    if materialize_test_items:
        overlaps.update(
            {
                "train_test": split_pair_sets["train"] & split_pair_sets["test"],
                "val_test": (
                    split_pair_sets["val"] & split_pair_sets["test"]
                    if materialize_validation_items
                    else set()
                ),
            }
        )
    leaked = {name: sorted(values) for name, values in overlaps.items() if values}
    if leaked:
        raise ValueError(f"pair_id leakage across explicit news-first splits: {leaked}")

    split_session_sets = {
        "train": _session_sets(train_items, split_name="train"),
    }
    if materialize_validation_items:
        split_session_sets["val"] = _session_sets(val_items, split_name="validation")
    if materialize_test_items:
        split_session_sets["test"] = _session_sets(test_items, split_name="test")
    session_overlaps = {}
    if materialize_validation_items:
        session_overlaps["train_val"] = (
            split_session_sets["train"] & split_session_sets["val"]
        )
    if materialize_test_items:
        session_overlaps.update(
            {
                "train_test": (
                    split_session_sets["train"] & split_session_sets["test"]
                ),
                "val_test": (
                    split_session_sets["val"] & split_session_sets["test"]
                    if materialize_validation_items
                    else set()
                ),
            }
        )
    leaked_sessions = {
        name: sorted(values) for name, values in session_overlaps.items() if values
    }
    if leaked_sessions:
        raise ValueError(
            f"session_id leakage across explicit news-first splits: {leaked_sessions}"
        )

    return NewsFirstVolSplitSelection(
        train_items=_with_pair_balanced_weights(train_items, scale_for_training=True),
        val_items=_with_pair_balanced_weights(val_items, scale_for_training=False),
        test_items=_with_pair_balanced_weights(test_items, scale_for_training=False),
        train_end_utc=train_end.isoformat(),
        validation_end_utc=validation_end.isoformat(),
    )


def _validate_shared_grid(
    train_items: Sequence[VolSurfaceSample],
    other_items: Sequence[VolSurfaceSample],
    *,
    split_name: str,
) -> None:
    reference = train_items[0]
    for sample in other_items:
        if tuple(sample.surface_shape) != tuple(reference.surface_shape):
            raise ValueError(
                f"Surface shape differs in {split_name}: {sample.sample_id}"
            )
        if not np.array_equal(sample.strike_grid, reference.strike_grid):
            raise ValueError(f"Strike grid differs in {split_name}: {sample.sample_id}")
        if not np.array_equal(sample.maturity_grid_days, reference.maturity_grid_days):
            raise ValueError(
                f"Maturity grid differs in {split_name}: {sample.sample_id}"
            )
        if (sample.support_mask is None) != (reference.support_mask is None):
            raise ValueError(
                f"Support-mask mode differs in {split_name}: {sample.sample_id}"
            )
        if (sample.current_support_mask is None) != (
            reference.current_support_mask is None
        ):
            raise ValueError(
                f"Current-support-mask mode differs in {split_name}: {sample.sample_id}"
            )
        if sample.support_grid_fingerprint != reference.support_grid_fingerprint:
            raise ValueError(
                f"Support-grid fingerprint differs in {split_name}: {sample.sample_id}"
            )


def _apply_text_ablation(
    items: Sequence[VolSurfaceSample],
    *,
    mode: str,
    seed: int,
    namespace: str,
) -> tuple[list[VolSurfaceSample], dict[str, object]]:
    """Apply one split-internal text treatment without changing sample order."""

    if not items:
        raise ValueError(f"Cannot apply text ablation to empty split {namespace!r}.")
    stable_keys = [sample.stable_sample_key or sample.sample_id for sample in items]
    embeddings, _ = _align_embeddings(
        [sample.text_embedding for sample in items],
        fallback_dim=1,
    )
    transformed, audit = transform_embedding_matrix(
        embeddings,
        stable_keys,
        mode=mode,
        seed=int(seed),
        namespace=namespace,
        pair_ids=[_pair_key(sample) for sample in items],
        session_ids=[str(sample.session_id).strip() for sample in items],
    )
    donors = dict(audit.get("text_shuffle_donor_by_receiver") or {})
    output: list[VolSurfaceSample] = []
    for index, sample in enumerate(items):
        stable_key = stable_keys[index]
        metadata = dict(sample.metadata)
        metadata.update(
            {
                "text_ablation_mode": normalize_text_ablation_mode(mode),
                "text_information_path": text_information_path(mode),
                "text_shuffle_namespace": namespace,
                "text_shuffle_seed": int(seed),
                "text_shuffle_mapping_sha256": str(
                    audit.get("text_shuffle_mapping_sha256", "")
                ),
                "text_shuffle_donor_sample_id": donors.get(stable_key, ""),
            }
        )
        output.append(
            replace(
                sample,
                text_embedding=np.asarray(transformed[index], dtype=np.float32),
                metadata=metadata,
            )
        )
    return output, {
        key: value
        for key, value in audit.items()
        if key != "text_shuffle_donor_by_receiver"
    }


def _weighted_loader(
    items: Sequence[VolSurfaceSample],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    training_weights: bool,
    expected_embedding_dim: int | None = None,
) -> tuple[DataLoader, int]:
    current = np.stack([sample.current_surface for sample in items], axis=0).astype(
        np.float32
    )
    target = np.stack([sample.target_surface for sample in items], axis=0).astype(
        np.float32
    )
    embeddings, embedding_dim = _align_embeddings(
        [sample.text_embedding for sample in items],
        fallback_dim=int(expected_embedding_dim or 1),
    )
    if expected_embedding_dim is not None and embedding_dim != int(
        expected_embedding_dim
    ):
        raise ValueError(
            f"Embedding dimension differs across news-first splits: "
            f"{embedding_dim} != {expected_embedding_dim}"
        )
    weights = np.asarray(
        [
            float(sample.metadata["scaled_training_sample_weight"])
            if training_weights
            else float(sample.metadata["raw_pair_sample_weight"])
            for sample in items
        ],
        dtype=np.float32,
    )
    stable_keys = np.asarray(
        [
            stable_key_to_int64(sample.stable_sample_key or sample.sample_id)
            for sample in items
        ],
        dtype=np.int64,
    )
    tensors = [
        torch.tensor(current, dtype=torch.float32),
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(stable_keys, dtype=torch.int64),
    ]
    mask_presence = [sample.support_mask is not None for sample in items]
    if any(mask_presence) and not all(mask_presence):
        raise ValueError(
            "Support masks must be present for either every row or no rows."
        )
    if all(mask_presence):
        masks = np.stack([sample.support_mask for sample in items], axis=0).astype(
            np.float32
        )
        tensors.append(torch.tensor(masks, dtype=torch.float32))
        current_mask_presence = [
            sample.current_support_mask is not None for sample in items
        ]
        if not all(current_mask_presence):
            raise ValueError(
                "Current support masks must accompany every joint support mask."
            )
        current_masks = np.stack(
            [sample.current_support_mask for sample in items], axis=0
        ).astype(np.float32)
        tensors.append(torch.tensor(current_masks, dtype=torch.float32))
        emit_label_reliability = [
            bool(sample.metadata.get("label_reliability_lineage_validated", False))
            for sample in items
        ]
        if any(emit_label_reliability) and not all(emit_label_reliability):
            raise ValueError(
                "Label-reliability lineage must be present for every row or no rows."
            )
        if all(emit_label_reliability):
            label_reliability_weights = np.asarray(
                [float(sample.label_reliability_weight) for sample in items],
                dtype=np.float32,
            )
            if not training_weights and not np.array_equal(
                label_reliability_weights,
                np.ones_like(label_reliability_weights),
            ):
                raise ValueError(
                    "Validation/evaluation label reliability weights must be exactly 1."
                )
            if (
                not np.all(np.isfinite(label_reliability_weights))
                or np.any(label_reliability_weights < 0.5)
                or np.any(label_reliability_weights > 2.0)
            ):
                raise ValueError(
                    "label_reliability_weight values must be finite and within "
                    "[0.5, 2.0]."
                )
            tensors.append(torch.tensor(label_reliability_weights, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=seeded_torch_generator(seed) if shuffle else None,
    )
    return loader, embedding_dim


def _support_diagnostics_for_splits(
    diagnostics: Mapping[str, object],
    *,
    split_names: Sequence[str],
) -> dict[str, object]:
    """Copy diagnostics while omitting every non-materialized time partition."""

    result = dict(diagnostics)
    raw_partitions = diagnostics.get("time_partitions", {})
    if isinstance(raw_partitions, Mapping):
        result["time_partitions"] = {
            split_name: dict(raw_partitions[split_name])
            for split_name in split_names
            if split_name in raw_partitions
            and isinstance(raw_partitions[split_name], Mapping)
        }
    return result


def create_news_first_vol_surface_dataloaders(
    config: Config,
    common_eval_data_path: str,
    *,
    split_spec: NewsFirstSplitSpec = NewsFirstSplitSpec(),
) -> VolSurfaceXlsxBundle:
    """Build pair-balanced train and common-5m validation/test dataloaders."""

    materialize_validation_loader = bool(
        config.news_first_materialize_validation_loader
    )
    materialize_test_loader = bool(config.news_first_materialize_test_loader)
    if materialize_test_loader and not materialize_validation_loader:
        raise ValueError(
            "news_first_materialize_test_loader=true requires "
            "news_first_materialize_validation_loader=true"
        )
    if materialize_validation_loader and not str(common_eval_data_path).strip():
        raise ValueError(
            "common_eval_data_path must point to the canonical 5m workbook."
        )
    train_end, validation_end = split_spec.parsed_boundaries(
        allow_empty_validation_window=not materialize_validation_loader
    )
    window_lineage: dict[str, str] = {}
    if materialize_validation_loader:
        window_lineage = _validate_label_reliability_data_window_contract(
            config,
            common_eval_data_path,
            split_spec=split_spec,
        )
    if not materialize_validation_loader:
        training_config = replace(
            config,
            news_first_data_window_start_utc_inclusive="",
            news_first_data_window_end_utc_exclusive=train_end.isoformat(),
        )
        common_config = None
    elif materialize_test_loader:
        training_config = config
        common_config = replace(config, data_path=str(common_eval_data_path))
    else:
        training_config = replace(
            config,
            news_first_data_window_start_utc_inclusive="",
            news_first_data_window_end_utc_exclusive=train_end.isoformat(),
        )
        common_config = replace(
            config,
            data_path=str(common_eval_data_path),
            news_first_data_window_start_utc_inclusive=train_end.isoformat(),
            news_first_data_window_end_utc_exclusive=validation_end.isoformat(),
        )
    training_items, training_support = _load_surface_items(training_config)
    if common_config is None:
        common_evaluation_items: list[VolSurfaceSample] = []
        common_support: dict[str, object] = {}
    else:
        common_evaluation_items, common_support = _load_surface_items(common_config)
    selection = select_news_first_vol_splits(
        training_items,
        common_evaluation_items,
        split_spec=split_spec,
        materialize_validation_items=materialize_validation_loader,
        materialize_test_items=materialize_test_loader,
    )
    selection, label_reliability_audit = _apply_label_reliability_contract(
        selection,
        config,
        materialize_evaluation_items=materialize_validation_loader,
    )

    pair_text_mode = normalize_pair_text_overlay_mode(
        getattr(config, "news_first_pair_text_overlay_mode", NO_PAIR_TEXT_OVERLAY)
    )
    shuffle_seed = int(config.news_first_text_shuffle_seed)
    tolerance = int(config.news_first_dataset_tolerance_minutes)
    pair_text_audit: dict[str, object] = {
        "mode": NO_PAIR_TEXT_OVERLAY,
        "enabled": False,
    }
    if pair_text_mode != NO_PAIR_TEXT_OVERLAY:
        manifest = load_pair_text_overlay_manifest(
            config.news_first_pair_text_manifest_path,
            config.news_first_pair_text_manifest_sha256,
            config.news_first_pair_text_profile_sha256,
            expected_mode=pair_text_mode,
        )
        materialized_items = [
            *selection.train_items,
            *selection.val_items,
            *selection.test_items,
        ]
        selected_pair_ids = sorted({_pair_key(sample) for sample in materialized_items})
        selected_universe_sha = pair_universe_sha256(selected_pair_ids)
        if materialize_validation_loader:
            if selected_universe_sha != manifest.pair_universe_sha256:
                raise ValueError(
                    "Pair-text manifest must cover exactly the materialized split "
                    f"universe: selected={selected_universe_sha}, "
                    f"manifest={manifest.pair_universe_sha256}"
                )
        else:
            declared_train_sha = str(
                manifest.transform.get("train_pair_universe_sha256", "")
            )
            if selected_universe_sha != declared_train_sha or not set(
                selected_pair_ids
            ).issubset(manifest.embeddings):
                raise ValueError(
                    "Training-only pair-text overlay must match the manifest's "
                    f"frozen train universe: selected={selected_universe_sha}, "
                    f"declared_train={declared_train_sha}"
                )
        train_items, train_text_audit = apply_pair_text_overlay(
            selection.train_items,
            manifest,
            require_exact_universe=False,
        )
        train_items = _with_pair_balanced_weights(train_items, scale_for_training=True)
        if materialize_validation_loader:
            val_items, validation_text_audit = apply_pair_text_overlay(
                selection.val_items,
                manifest,
                require_exact_universe=False,
            )
            val_items = _with_pair_balanced_weights(val_items, scale_for_training=False)
        else:
            val_items = []
        if materialize_test_loader:
            test_items, test_text_audit = apply_pair_text_overlay(
                selection.test_items,
                manifest,
                require_exact_universe=False,
            )
            test_items = _with_pair_balanced_weights(
                test_items, scale_for_training=False
            )
        else:
            test_items = []
        text_mode = pair_text_mode
        expected_text_path = f"pair_text_overlay:{pair_text_mode}"
        pair_text_audit = {
            "enabled": True,
            "mode": pair_text_mode,
            "manifest_sha256": manifest.file_sha256,
            "profile_sha256": manifest.profile_sha256,
            "pair_universe_sha256": manifest.pair_universe_sha256,
            "namespace": manifest.namespace,
            "embedding_dim": manifest.embedding_dim,
            "pair_count": len(selected_pair_ids),
        }
    else:
        text_mode = normalize_text_ablation_mode(config.news_first_text_ablation_mode)
        declared_text_path = str(config.news_first_text_information_path).strip()
        expected_text_path = text_information_path(text_mode)
        if declared_text_path and declared_text_path != expected_text_path:
            raise ValueError(
                "news_first_text_information_path conflicts with ablation mode: "
                f"{declared_text_path!r} != {expected_text_path!r}"
            )
        if text_mode == "text_shuffle" and tolerance not in {5, 10, 15, 30}:
            raise ValueError(
                "news_first_dataset_tolerance_minutes must be 5/10/15/30 for text_shuffle"
            )
        train_items, train_text_audit = _apply_text_ablation(
            selection.train_items,
            mode=text_mode,
            seed=shuffle_seed,
            namespace=f"train_{tolerance:02d}m",
        )
        if materialize_validation_loader:
            val_items, validation_text_audit = _apply_text_ablation(
                selection.val_items,
                mode=text_mode,
                seed=shuffle_seed,
                namespace="common_validation_05m",
            )
        else:
            val_items = []
        if materialize_test_loader:
            test_items, test_text_audit = _apply_text_ablation(
                selection.test_items,
                mode=text_mode,
                seed=shuffle_seed,
                namespace="common_test_core_05m",
            )
        else:
            test_items = []
    selection = replace(
        selection,
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
    )

    if materialize_validation_loader:
        _validate_shared_grid(
            selection.train_items, selection.val_items, split_name="validation"
        )
    if materialize_test_loader:
        _validate_shared_grid(
            selection.train_items, selection.test_items, split_name="test"
        )
    train_loader, embedding_dim = _weighted_loader(
        selection.train_items,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        shuffle=True,
        training_weights=True,
    )
    val_loader = None
    if materialize_validation_loader:
        val_loader, _ = _weighted_loader(
            selection.val_items,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            shuffle=False,
            training_weights=False,
            expected_embedding_dim=embedding_dim,
        )
    test_loader = None
    if materialize_test_loader:
        test_loader, _ = _weighted_loader(
            selection.test_items,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            shuffle=False,
            training_weights=False,
            expected_embedding_dim=embedding_dim,
        )
    first = selection.train_items[0]
    materialized_test_items = list(selection.test_items)
    all_items = [
        *selection.train_items,
        *selection.val_items,
        *materialized_test_items,
    ]
    if materialize_test_loader:
        support_split_names = ("train", "validation", "test")
    elif materialize_validation_loader:
        support_split_names = ("train", "validation")
    else:
        support_split_names = ("train",)
    training_support_metadata = _support_diagnostics_for_splits(
        training_support,
        split_names=support_split_names,
    )
    common_support_metadata = (
        _support_diagnostics_for_splits(
            common_support,
            split_names=support_split_names,
        )
        if materialize_validation_loader
        else {}
    )
    support_filter_splits = {
        "train": training_support["time_partitions"]["train"],
    }
    text_ablation_splits = {"train": train_text_audit}
    if materialize_validation_loader:
        support_filter_splits["validation"] = common_support["time_partitions"][
            "validation"
        ]
        text_ablation_splits["validation"] = validation_text_audit
    split_metadata: dict[str, object] = {
        "mode": (
            "news_first_explicit_common_5m"
            if materialize_test_loader
            else (
                "news_first_inner_validation_only"
                if materialize_validation_loader
                else "news_first_training_only"
            )
        ),
        "training_data_path": str(config.data_path),
        "common_eval_data_path": str(common_eval_data_path),
        "train_end_utc": selection.train_end_utc,
        "validation_end_utc": selection.validation_end_utc,
        "train_rows": len(selection.train_items),
        "validation_rows": len(selection.val_items),
        "train_pairs": len(_pair_sets(selection.train_items)),
        "validation_pairs": len(_pair_sets(selection.val_items)),
        "train_sessions": len(_session_sets(selection.train_items, split_name="train")),
        "validation_sessions": len(
            _session_sets(selection.val_items, split_name="validation")
        ),
        "validation_materialized": materialize_validation_loader,
        "news_first_materialize_validation_loader": (materialize_validation_loader),
        "news_first_materialize_test_loader": materialize_test_loader,
        "training_surface_support": training_support_metadata,
        "common_evaluation_surface_support": common_support_metadata,
        "support_grid_fingerprint": first.support_grid_fingerprint,
        "support_filter_splits": support_filter_splits,
        "text_ablation_mode": text_mode,
        "text_information_path": expected_text_path,
        "text_shuffle_seed": shuffle_seed,
        "text_ablation_splits": text_ablation_splits,
        "pair_text_overlay": pair_text_audit,
        "label_reliability": label_reliability_audit,
    }
    if not materialize_validation_loader:
        split_metadata["validation_data_policy"] = {
            "forbidden_from_utc_inclusive": selection.train_end_utc,
            "sample_objects_materialized": False,
            "loader_materialized": False,
            "predictions_permitted": False,
        }
        split_metadata["validation_forbidden_from_utc_inclusive"] = (
            selection.train_end_utc
        )
    elif materialize_test_loader:
        split_metadata.update(
            {
                "test_rows": len(materialized_test_items),
                "test_pairs": len(_pair_sets(materialized_test_items)),
                "test_sessions": len(
                    _session_sets(materialized_test_items, split_name="test")
                ),
                "materialized_test_rows": len(materialized_test_items),
            }
        )
        support_filter_splits["test"] = common_support["time_partitions"]["test"]
        text_ablation_splits["test"] = test_text_audit
    else:
        split_metadata["post_validation_data_policy"] = {
            "forbidden_from_utc_inclusive": selection.validation_end_utc,
            "sample_objects_materialized": False,
            "predictions_permitted": False,
        }
        split_metadata["label_reliability_data_window"] = {
            **window_lineage,
            "train_origin_end_utc_exclusive": selection.train_end_utc,
            "validation_origin_start_utc_inclusive": selection.train_end_utc,
            "validation_origin_end_utc_exclusive": selection.validation_end_utc,
        }

    return VolSurfaceXlsxBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        strike_grid=first.strike_grid.copy(),
        maturity_grid_days=first.maturity_grid_days.copy(),
        embedding_dim=embedding_dim,
        train_samples=len(selection.train_items),
        val_samples=len(selection.val_items),
        timestamps=[sample.timestamp for sample in all_items],
        train_timestamps=[sample.timestamp for sample in selection.train_items],
        val_timestamps=[sample.timestamp for sample in selection.val_items],
        all_items=all_items,
        train_items=list(selection.train_items),
        val_items=list(selection.val_items),
        test_loader=test_loader,
        test_samples=len(materialized_test_items),
        test_timestamps=[sample.timestamp for sample in materialized_test_items],
        test_items=materialized_test_items,
        uses_sample_weights=True,
        uses_label_reliability_weights=(
            label_reliability_audit["mode"] in _LABEL_RELIABILITY_SOFT_WEIGHT_MODES
        ),
        uses_support_masks=bool(training_support["support_mask_applied"]),
        uses_current_support_masks=bool(training_support["support_mask_applied"]),
        split_metadata=split_metadata,
    )


def create_configured_vol_surface_dataloaders(config: Config) -> VolSurfaceXlsxBundle:
    """Select the explicit news-first loader when its common workbook is set."""

    common_path = str(getattr(config, "news_first_common_eval_data_path", "")).strip()
    if not common_path and bool(config.news_first_materialize_validation_loader):
        from .merged_xlsx_dataloaders import create_vol_surface_xlsx_dataloaders

        return create_vol_surface_xlsx_dataloaders(config)
    return create_news_first_vol_surface_dataloaders(
        config,
        common_path,
        split_spec=NewsFirstSplitSpec(
            train_end_utc=str(config.news_first_train_end_utc),
            validation_end_utc=str(config.news_first_validation_end_utc),
        ),
    )
