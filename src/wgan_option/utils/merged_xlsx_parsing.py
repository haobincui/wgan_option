"""Parsing helpers shared by merged-xlsx loaders and dataloader builders."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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


def _resolve_text_embedding(hd_raw: Any, lp_raw: Any, mode: str) -> np.ndarray:
    mode_text = str(mode).strip().lower()
    if mode_text == "none":
        return np.zeros(0, dtype=np.float32)
    hd_vector = _parse_serialized_vector(hd_raw)
    lp_vector = _parse_serialized_vector(lp_raw)
    if mode_text == "hd":
        return hd_vector
    if mode_text == "lp":
        return lp_vector
    if mode_text == "concat":
        return np.concatenate([hd_vector, lp_vector], axis=0)
    raise ValueError(f"text_embedding_mode must be one of none/hd/lp/concat, got: {mode}")


def _parse_optional_int(raw_value: Any) -> Optional[int]:
    if _is_missing(raw_value):
        return None
    return int(round(float(raw_value)))


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
