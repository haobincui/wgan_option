"""Sample loaders and chronological split helpers for merged-xlsx workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .merged_xlsx_parsing import (
    _parse_optional_int,
    _parse_serialized_list,
    _parse_surface_shape,
    _parse_timestamp_column,
    _read_sheet,
    _resolve_text_embedding,
    _split_index,
)
from .merged_xlsx_types import OrderedSplitSelection, SVI_FEATURE_ORDER, SviPairedSample, VolSurfaceSample


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


def _prepare_vol_surface_frame(config: Any) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Tuple[int, int]]:
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
    sort_columns = ["_timestamp"]
    if "sample_id" in df.columns:
        sort_columns.append("sample_id")
    df = df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    surface_shape = _parse_surface_shape(df.iloc[0]["surface_shape"])
    strike_grid = np.asarray(_parse_serialized_list(df.iloc[0]["strike_grid"]), dtype=np.float32)
    maturity_grid_days = np.asarray(_parse_serialized_list(df.iloc[0]["maturity_days_grid"]), dtype=np.float32)
    return df, strike_grid, maturity_grid_days, surface_shape


def load_vol_surface_samples(config: Any) -> List[VolSurfaceSample]:
    """Load ordered merged vol samples with metadata for inference or training."""

    df, strike_grid, maturity_grid_days, surface_shape = _prepare_vol_surface_frame(config)
    expected_cells = int(surface_shape[0] * surface_shape[1])

    samples: List[VolSurfaceSample] = []
    for global_index, row in enumerate(df.itertuples(index=False)):
        current_surface = np.asarray(_parse_serialized_list(row.current_surface_flat), dtype=np.float32)
        target_surface = np.asarray(_parse_serialized_list(row.target_surface_flat), dtype=np.float32)
        if int(current_surface.size) != expected_cells or int(target_surface.size) != expected_cells:
            raise ValueError(
                f"Surface length mismatch for sample {getattr(row, 'sample_id', '')}: "
                f"expected {expected_cells}, got {current_surface.size} and {target_surface.size}"
            )

        samples.append(
            VolSurfaceSample(
                sample_id=str(getattr(row, "sample_id", f"row_{global_index}")),
                timestamp=str(getattr(row, "news_timestamp_utc")),
                current_snapshot_time_utc=str(getattr(row, "current_snapshot_time_utc", "")),
                target_snapshot_time_utc=str(getattr(row, "target_snapshot_time_utc", "")),
                current_surface=current_surface.reshape(1, surface_shape[0], surface_shape[1]).astype(np.float32),
                target_surface=target_surface.reshape(1, surface_shape[0], surface_shape[1]).astype(np.float32),
                text_embedding=_resolve_text_embedding(
                    getattr(row, "hd_embedding", ""),
                    getattr(row, "lp_embedding", ""),
                    config.text_embedding_mode,
                ),
                strike_grid=strike_grid.copy(),
                maturity_grid_days=maturity_grid_days.copy(),
                surface_shape=surface_shape,
                global_index=global_index,
                metadata={
                    "pair_quality_label": str(getattr(row, "pair_quality_label", "")),
                    "current_weighted_iv_rmse": getattr(row, "current_weighted_iv_rmse", None),
                    "target_weighted_iv_rmse": getattr(row, "target_weighted_iv_rmse", None),
                    "training_candidate_flag": _parse_optional_int(getattr(row, "training_candidate_flag", None)),
                },
            )
        )
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
    usable = usable.sort_values(["news_row_id", "_timestamp", "direction"], kind="stable")
    return usable


def load_svi_paired_samples(config: Any) -> List[SviPairedSample]:
    """Load ordered backward/current -> forward/future SVI pairs."""

    usable = _prepare_svi_usable_frame(config)

    paired_samples: List[SviPairedSample] = []
    for news_row_id, group in usable.groupby("news_row_id", sort=False):
        directions = {str(direction): row for direction, row in group.set_index("direction").iterrows()}
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
        raise ValueError("No paired backward/forward SVI samples are available for training.")

    paired_samples = sorted(paired_samples, key=lambda sample: pd.Timestamp(sample.timestamp))
    for global_index, sample in enumerate(paired_samples):
        sample.global_index = global_index
    return paired_samples
