"""Sample loaders and chronological split helpers for merged-xlsx workflows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from film_wgan.support import SUPPORT_METHOD, parse_raw_surface_params, raw_support_mask

from .merged_xlsx_parsing import (
    _parse_optional_int,
    _parse_serialized_list,
    _parse_surface_shape,
    _parse_timestamp_column,
    _read_sheet,
    _resolve_text_embedding,
    _split_index,
)
from .merged_xlsx_types import (
    OrderedSplitSelection,
    SVI_FEATURE_ORDER,
    SviPairedSample,
    VolSurfaceSample,
)


SUPPORT_MASK_MODES = frozenset({"none", "raw_joint"})


def _support_mask_mode(config: Any) -> str:
    mode = str(getattr(config, "support_mask_mode", "none") or "none").strip().lower()
    if mode not in SUPPORT_MASK_MODES:
        raise ValueError(
            f"support_mask_mode must be one of {sorted(SUPPORT_MASK_MODES)}, got: {mode}"
        )
    return mode


def _support_grid_fingerprint(
    strike_grid: np.ndarray,
    maturity_grid_days: np.ndarray,
) -> str:
    payload = {
        "schema_version": 1,
        "support_method": SUPPORT_METHOD,
        "strike_grid": [float(value) for value in strike_grid],
        "maturity_days_grid": [
            int(round(float(value))) for value in maturity_grid_days
        ],
        "surface_shape": [int(maturity_grid_days.size), int(strike_grid.size)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _support_mask_fingerprint(
    support_mask: np.ndarray,
    *,
    role: str,
    grid_fingerprint: str,
) -> str:
    """Fingerprint one mask without conflating current-only and joint support."""

    mask = np.asarray(support_mask, dtype=bool)
    payload = {
        "schema_version": 1,
        "support_method": SUPPORT_METHOD,
        "role": str(role),
        "support_grid_fingerprint": str(grid_fingerprint),
        "surface_shape": [int(value) for value in mask.shape],
        "supported_cells": [int(value) for value in mask.reshape(-1)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _time_partition(value: str, config: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"Cannot assign support diagnostic time partition: {value}")
    train_end = pd.to_datetime(
        getattr(config, "news_first_train_end_utc", "2023-07-01T00:00:00Z"),
        errors="raise",
        utc=True,
    )
    validation_end = pd.to_datetime(
        getattr(
            config,
            "news_first_validation_end_utc",
            "2023-10-01T00:00:00Z",
        ),
        errors="raise",
        utc=True,
    )
    if timestamp < train_end:
        return "train"
    if timestamp < validation_end:
        return "validation"
    return "test"


def _support_partition_summaries(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for partition in ("train", "validation", "test"):
        selected = [record for record in records if record["partition"] == partition]
        kept = [record for record in selected if record["kept"]]
        excluded = [record for record in selected if not record["kept"]]
        counts = [
            int(record["joint_support_cell_count"])
            for record in kept
            if record["joint_support_cell_count"] is not None
        ]

        def unique_count(rows: Sequence[Dict[str, Any]], column: str) -> int:
            return len({str(row[column]) for row in rows if str(row[column]).strip()})

        summary: Dict[str, Any] = {
            "input_rows": len(selected),
            "input_pairs": unique_count(selected, "pair_id"),
            "input_sessions": unique_count(selected, "session_id"),
            "excluded_zero_joint_support_rows": len(excluded),
            "excluded_zero_joint_support_pairs": unique_count(excluded, "pair_id"),
            "excluded_zero_joint_support_sessions": unique_count(
                excluded, "session_id"
            ),
            "kept_rows": len(kept),
            "kept_pairs": unique_count(kept, "pair_id"),
            "kept_sessions": unique_count(kept, "session_id"),
        }
        if counts:
            summary["joint_support_cell_count"] = {
                "min": int(min(counts)),
                "p25": float(np.percentile(counts, 25)),
                "median": float(np.median(counts)),
                "mean": float(np.mean(counts)),
                "p75": float(np.percentile(counts, 75)),
                "max": int(max(counts)),
            }
        else:
            summary["joint_support_cell_count"] = {}
        result[partition] = summary
    return result


def _optional_text(value: Any) -> str:
    """Return a stable text representation for optional workbook metadata."""

    if (
        value is None
        or (isinstance(value, float) and not np.isfinite(value))
        or pd.isna(value)
    ):
        return ""
    return str(value)


def _positive_sample_weight(value: Any) -> float:
    """Parse one positive finite sample weight, falling back to equal weight."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return parsed if np.isfinite(parsed) and parsed > 0.0 else 1.0


def select_ordered_split(
    items: Sequence[Any],
    *,
    train_ratio: float,
    split: str,
) -> OrderedSplitSelection:
    """Split ordered samples chronologically and select the requested partition."""

    all_items = list(items)
    split_idx = _split_index(len(all_items), train_ratio)
    train_items = list(all_items[:split_idx])
    val_items = list(all_items[split_idx:])

    requested_split = str(split).strip().lower()
    fallback_to_all = False
    effective_split = requested_split
    if requested_split == "train":
        selected_items = train_items
    elif requested_split == "val":
        if val_items:
            selected_items = val_items
        else:
            selected_items = all_items
            effective_split = "all"
            fallback_to_all = True
    elif requested_split == "all":
        selected_items = all_items
    else:
        raise ValueError(f"split must be one of train/val/all, got: {split}")

    return OrderedSplitSelection(
        all_items=all_items,
        train_items=train_items,
        val_items=val_items,
        selected_items=list(selected_items),
        split_idx=split_idx,
        requested_split=requested_split,
        effective_split=effective_split,
        fallback_to_all=fallback_to_all,
    )


def _prepare_vol_surface_frame(
    config: Any,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Tuple[int, int]]:
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
    if _support_mask_mode(config) == "raw_joint":
        required_columns.update(
            {
                "surface_model",
                "current_surface_param_json",
                "target_surface_param_json",
            }
        )
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Vol workbook is missing required columns: {missing}")

    if "training_candidate_flag" in df.columns:
        df = df[df["training_candidate_flag"] == 1].copy()
    if df.empty:
        raise ValueError("No training-ready vol rows remain after filtering.")

    df = df.copy()
    window_start = str(
        getattr(config, "news_first_data_window_start_utc_inclusive", "") or ""
    ).strip()
    window_end = str(
        getattr(config, "news_first_data_window_end_utc_exclusive", "") or ""
    ).strip()
    if window_start or window_end:
        origin_column = next(
            (
                column
                for column in ("effective_origin_utc", "current_snapshot_time_utc")
                if column in df.columns
            ),
            None,
        )
        if origin_column is None:
            raise ValueError(
                "A news-first input window requires effective_origin_utc or "
                "current_snapshot_time_utc."
            )
        origins = pd.to_datetime(df[origin_column], errors="coerce", utc=True)
        if origins.isna().any():
            raise ValueError(
                f"Cannot apply the news-first input window because {origin_column} "
                "contains invalid timestamps."
            )
        keep = pd.Series(True, index=df.index)
        if window_start:
            parsed_start = pd.to_datetime(window_start, errors="coerce", utc=True)
            if pd.isna(parsed_start):
                raise ValueError(
                    "news_first_data_window_start_utc_inclusive must be a valid "
                    f"UTC timestamp, got: {window_start}"
                )
            keep &= origins >= parsed_start
        if window_end:
            parsed_end = pd.to_datetime(window_end, errors="coerce", utc=True)
            if pd.isna(parsed_end):
                raise ValueError(
                    "news_first_data_window_end_utc_exclusive must be a valid UTC "
                    f"timestamp, got: {window_end}"
                )
            if window_start and parsed_end <= parsed_start:
                raise ValueError(
                    "news-first input window end must be later than its start."
                )
            keep &= origins < parsed_end
        # This happens before any serialized surface, raw-param, or text field is
        # parsed. Formal reliability jobs additionally point at independently
        # materialized, byte-hash-bound fold workbooks, so excluded Q3/Q4 rows are
        # not present in the file read by this loader.
        df = df.loc[keep].copy()
        if df.empty:
            raise ValueError("No training-ready vol rows fall inside the input window.")

    df["_timestamp"] = _parse_timestamp_column(df, "news_timestamp_utc")
    sort_columns = ["_timestamp"]
    if "sample_id" in df.columns:
        sort_columns.append("sample_id")
    df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    surface_shape = _parse_surface_shape(df.iloc[0]["surface_shape"])
    strike_grid = np.asarray(
        _parse_serialized_list(df.iloc[0]["strike_grid"]), dtype=np.float32
    )
    maturity_grid_days = np.asarray(
        _parse_serialized_list(df.iloc[0]["maturity_days_grid"]), dtype=np.float32
    )
    return df, strike_grid, maturity_grid_days, surface_shape


def load_vol_surface_samples_with_diagnostics(
    config: Any,
) -> tuple[List[VolSurfaceSample], Dict[str, Any]]:
    """Load ordered vol samples and derive optional raw joint-support masks."""

    df, strike_grid, maturity_grid_days, surface_shape = _prepare_vol_surface_frame(
        config
    )
    expected_cells = int(surface_shape[0] * surface_shape[1])
    support_mode = _support_mask_mode(config)
    # Preserve workbook precision for support-boundary comparisons and for the
    # fingerprint shared with the generation audit. Model tensors remain float32.
    support_strike_grid = np.asarray(
        _parse_serialized_list(df.iloc[0]["strike_grid"]), dtype=np.float64
    )
    support_maturity_grid = np.asarray(
        _parse_serialized_list(df.iloc[0]["maturity_days_grid"]), dtype=np.float64
    )
    if (int(support_maturity_grid.size), int(support_strike_grid.size)) != tuple(
        surface_shape
    ):
        raise ValueError(
            "Surface grid lengths do not match surface_shape: "
            f"{(support_maturity_grid.size, support_strike_grid.size)} != "
            f"{surface_shape}"
        )
    grid_fingerprint = _support_grid_fingerprint(
        support_strike_grid,
        support_maturity_grid,
    )
    current_counts: List[int] = []
    joint_counts: List[int] = []
    excluded_zero_support_ids: List[str] = []
    support_records: List[Dict[str, Any]] = []

    samples: List[VolSurfaceSample] = []
    for source_global_index, row in enumerate(df.itertuples(index=False)):
        current_surface = np.asarray(
            _parse_serialized_list(row.current_surface_flat), dtype=np.float32
        )
        target_surface = np.asarray(
            _parse_serialized_list(row.target_surface_flat), dtype=np.float32
        )
        if (
            int(current_surface.size) != expected_cells
            or int(target_surface.size) != expected_cells
        ):
            raise ValueError(
                f"Surface length mismatch for sample {getattr(row, 'sample_id', '')}: "
                f"expected {expected_cells}, got {current_surface.size} and {target_surface.size}"
            )

        sample_id = str(getattr(row, "sample_id", f"row_{source_global_index}"))
        news_row_id = _parse_optional_int(getattr(row, "news_row_id", None))
        pair_id = _optional_text(getattr(row, "pair_id", ""))
        session_id = _optional_text(getattr(row, "session_id", ""))
        effective_origin_utc = _optional_text(
            getattr(
                row,
                "effective_origin_utc",
                getattr(row, "current_snapshot_time_utc", ""),
            )
        )
        diagnostic_origin = effective_origin_utc or str(row.news_timestamp_utc)
        support_mask: np.ndarray | None = None
        current_support_mask: np.ndarray | None = None
        support_mask_fingerprint = ""
        current_support_mask_fingerprint = ""
        current_support_count: int | None = None
        target_support_count: int | None = None
        joint_support_count: int | None = None
        if support_mode == "raw_joint":
            surface_model = str(getattr(row, "surface_model", "")).strip().lower()
            if surface_model != "raw":
                raise ValueError(
                    f"support_mask_mode=raw_joint requires surface_model=raw for "
                    f"sample {sample_id}, got: {surface_model or '<empty>'}"
                )
            row_surface_shape = _parse_surface_shape(row.surface_shape)
            row_support_strike_grid = np.asarray(
                _parse_serialized_list(row.strike_grid), dtype=np.float64
            )
            row_support_maturity_grid = np.asarray(
                _parse_serialized_list(row.maturity_days_grid), dtype=np.float64
            )
            if tuple(row_surface_shape) != tuple(surface_shape):
                raise ValueError(
                    f"Surface shape differs within workbook for sample {sample_id}: "
                    f"{row_surface_shape} != {surface_shape}"
                )
            row_grid_fingerprint = _support_grid_fingerprint(
                row_support_strike_grid,
                row_support_maturity_grid,
            )
            if (
                not np.array_equal(row_support_strike_grid, support_strike_grid)
                or not np.array_equal(row_support_maturity_grid, support_maturity_grid)
                or row_grid_fingerprint != grid_fingerprint
            ):
                raise ValueError(
                    f"Support grid differs within workbook for sample {sample_id}: "
                    f"{row_grid_fingerprint} != {grid_fingerprint}"
                )
            try:
                current_params = parse_raw_surface_params(
                    row.current_surface_param_json
                )
                target_params = parse_raw_surface_params(row.target_surface_param_json)
                current_mask = raw_support_mask(
                    current_params,
                    strike_grid=row_support_strike_grid,
                    maturity_days_grid=row_support_maturity_grid,
                )
                target_mask = raw_support_mask(
                    target_params,
                    strike_grid=row_support_strike_grid,
                    maturity_days_grid=row_support_maturity_grid,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot derive raw joint support for sample {sample_id}: {exc}"
                ) from exc
            if tuple(current_mask.shape) != tuple(surface_shape):
                raise ValueError(
                    f"Raw current support shape mismatch for sample {sample_id}: "
                    f"{current_mask.shape} != {surface_shape}"
                )
            if tuple(target_mask.shape) != tuple(surface_shape):
                raise ValueError(
                    f"Raw target support shape mismatch for sample {sample_id}: "
                    f"{target_mask.shape} != {surface_shape}"
                )
            support_mask = np.logical_and(current_mask, target_mask)
            if tuple(support_mask.shape) != tuple(surface_shape):
                raise ValueError(
                    f"Raw joint support shape mismatch for sample {sample_id}: "
                    f"{support_mask.shape} != {surface_shape}"
                )
            current_support_count = int(current_mask.sum())
            target_support_count = int(target_mask.sum())
            joint_support_count = int(support_mask.sum())
            current_counts.append(current_support_count)
            joint_counts.append(joint_support_count)
            if joint_support_count == 0:
                excluded_zero_support_ids.append(sample_id)
                support_records.append(
                    {
                        "partition": _time_partition(diagnostic_origin, config),
                        "pair_id": pair_id,
                        "session_id": session_id,
                        "joint_support_cell_count": joint_support_count,
                        "kept": False,
                    }
                )
                continue
            current_support_mask_fingerprint = _support_mask_fingerprint(
                current_mask,
                role="current",
                grid_fingerprint=grid_fingerprint,
            )
            support_mask_fingerprint = _support_mask_fingerprint(
                support_mask,
                role="joint",
                grid_fingerprint=grid_fingerprint,
            )
            current_support_mask = current_mask.reshape(
                1, surface_shape[0], surface_shape[1]
            ).astype(np.float32)
            support_mask = support_mask.reshape(
                1, surface_shape[0], surface_shape[1]
            ).astype(np.float32)

        support_records.append(
            {
                "partition": _time_partition(diagnostic_origin, config),
                "pair_id": pair_id,
                "session_id": session_id,
                "joint_support_cell_count": (
                    joint_support_count if support_mode == "raw_joint" else None
                ),
                "kept": True,
            }
        )
        sample_weight = _positive_sample_weight(getattr(row, "sample_weight", 1.0))
        stable_sample_key = sample_id or (
            f"news={news_row_id}|pair={pair_id}|origin={effective_origin_utc}"
        )

        samples.append(
            VolSurfaceSample(
                sample_id=sample_id,
                timestamp=str(getattr(row, "news_timestamp_utc")),
                current_snapshot_time_utc=str(
                    getattr(row, "current_snapshot_time_utc", "")
                ),
                target_snapshot_time_utc=str(
                    getattr(row, "target_snapshot_time_utc", "")
                ),
                current_surface=current_surface.reshape(
                    1, surface_shape[0], surface_shape[1]
                ).astype(np.float32),
                target_surface=target_surface.reshape(
                    1, surface_shape[0], surface_shape[1]
                ).astype(np.float32),
                text_embedding=_resolve_text_embedding(
                    getattr(row, "hd_embedding", ""),
                    getattr(row, "lp_embedding", ""),
                    config.text_embedding_mode,
                ),
                strike_grid=strike_grid.copy(),
                maturity_grid_days=maturity_grid_days.copy(),
                surface_shape=surface_shape,
                global_index=len(samples),
                metadata={
                    "news_row_id": news_row_id,
                    "pair_id": pair_id,
                    "session_id": session_id,
                    "effective_origin_utc": effective_origin_utc,
                    "sample_weight": sample_weight,
                    "source_sample_weight": sample_weight,
                    "stable_sample_key": stable_sample_key,
                    "pair_quality_label": str(getattr(row, "pair_quality_label", "")),
                    "current_weighted_iv_rmse": getattr(
                        row, "current_weighted_iv_rmse", None
                    ),
                    "target_weighted_iv_rmse": getattr(
                        row, "target_weighted_iv_rmse", None
                    ),
                    "training_candidate_flag": _parse_optional_int(
                        getattr(row, "training_candidate_flag", None)
                    ),
                    "support_mask_mode": support_mode,
                    "support_mask_applied": support_mask is not None,
                    "support_method": SUPPORT_METHOD
                    if support_mask is not None
                    else "",
                    "support_grid_fingerprint": grid_fingerprint,
                    "current_support_mask_fingerprint": (
                        current_support_mask_fingerprint
                    ),
                    "joint_support_mask_fingerprint": support_mask_fingerprint,
                    "current_raw_support_cell_count": current_support_count,
                    "current_raw_support_fraction": (
                        float(current_support_count) / float(expected_cells)
                        if current_support_count is not None
                        else None
                    ),
                    "target_raw_support_cell_count": target_support_count,
                    "joint_raw_support_cell_count": joint_support_count,
                    "joint_raw_support_fraction": (
                        float(joint_support_count) / float(expected_cells)
                        if joint_support_count is not None
                        else None
                    ),
                    "source_global_index": source_global_index,
                },
                news_row_id=news_row_id,
                pair_id=pair_id,
                session_id=session_id,
                effective_origin_utc=effective_origin_utc,
                sample_weight=sample_weight,
                stable_sample_key=stable_sample_key,
                support_mask=support_mask,
                current_support_mask=current_support_mask,
                support_grid_fingerprint=grid_fingerprint,
                support_mask_fingerprint=support_mask_fingerprint,
                current_support_mask_fingerprint=(current_support_mask_fingerprint),
            )
        )
    if not samples:
        if support_mode == "raw_joint" and excluded_zero_support_ids:
            raise ValueError(
                "No training-ready vol rows retain positive raw joint support."
            )
        raise ValueError("No training-ready vol samples are available.")
    diagnostics: Dict[str, Any] = {
        "support_mask_mode": support_mode,
        "support_mask_applied": support_mode == "raw_joint",
        "input_rows": int(len(df)),
        "loaded_rows": int(len(samples)),
        "excluded_zero_joint_support_rows": int(len(excluded_zero_support_ids)),
        "excluded_zero_joint_support_sample_ids_sha256": hashlib.sha256(
            "\n".join(excluded_zero_support_ids).encode("utf-8")
        ).hexdigest(),
        "grid_cell_count": expected_cells,
        "grid_fingerprint": grid_fingerprint,
        "support_method": SUPPORT_METHOD if support_mode == "raw_joint" else "",
        "time_partitions": _support_partition_summaries(support_records),
    }
    if joint_counts:
        diagnostics.update(
            {
                "current_support_cell_count_min": int(min(current_counts)),
                "current_support_cell_count_median": float(np.median(current_counts)),
                "current_support_cell_count_max": int(max(current_counts)),
                "joint_support_cell_count_min": int(min(joint_counts)),
                "joint_support_cell_count_median": float(np.median(joint_counts)),
                "joint_support_cell_count_max": int(max(joint_counts)),
            }
        )
    return samples, diagnostics


def load_vol_surface_samples(config: Any) -> List[VolSurfaceSample]:
    """Load ordered merged vol samples; legacy callers receive the sample list."""

    samples, _ = load_vol_surface_samples_with_diagnostics(config)
    return samples


def _build_svi_param_dict_from_row(row: pd.Series) -> Dict[str, List[float]]:
    return {
        feature: _parse_serialized_list(row[f"svi_{feature}_list"])
        for feature in SVI_FEATURE_ORDER
    }


def _prepare_svi_usable_frame(config: Any) -> pd.DataFrame:
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
    usable = usable.sort_values(
        ["news_row_id", "_timestamp", "direction"], kind="stable"
    )
    return usable


def load_svi_paired_samples(config: Any) -> List[SviPairedSample]:
    """Load ordered backward/current -> forward/future SVI pairs."""

    usable = _prepare_svi_usable_frame(config)

    paired_samples: List[SviPairedSample] = []
    for news_row_id, group in usable.groupby("news_row_id", sort=False):
        directions = {
            str(direction): row
            for direction, row in group.set_index("direction").iterrows()
        }
        if "backward" not in directions or "forward" not in directions:
            continue

        current_row = directions["backward"]
        future_row = directions["forward"]
        global_index = len(paired_samples)
        paired_samples.append(
            SviPairedSample(
                sample_id=f"news_{int(news_row_id)}",
                news_row_id=int(news_row_id),
                timestamp=str(current_row["news_timestamp_utc"]),
                current_timestamp_utc=str(current_row["news_timestamp_utc"]),
                future_timestamp_utc=str(future_row["news_timestamp_utc"]),
                text_embedding=_resolve_text_embedding(
                    current_row.get("hd_embedding", ""),
                    current_row.get("lp_embedding", ""),
                    config.text_embedding_mode,
                ),
                current_svi=_build_svi_param_dict_from_row(current_row),
                future_svi=_build_svi_param_dict_from_row(future_row),
                global_index=global_index,
                current_metadata=current_row.to_dict(),
                future_metadata=future_row.to_dict(),
            )
        )

    if not paired_samples:
        raise ValueError(
            "No paired backward/forward SVI samples are available for training."
        )

    paired_samples = sorted(
        paired_samples, key=lambda sample: pd.Timestamp(sample.timestamp)
    )
    for global_index, sample in enumerate(paired_samples):
        sample.global_index = global_index
    return paired_samples
