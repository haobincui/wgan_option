"""Leakage-safe text features and negative-text plans for the FiLM probe.

This module is deliberately independent of model and orchestration code.  It
keeps the historical pair-level LP baseline intact while providing opt-in,
hash-bound utilities for the single-seed text-signal experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .news_first_experiment_core import (
    _atomic_json_write,
    _canonical_json_bytes,
    _fixed_partition_derangement,
    _l2,
    _normalized_article_key,
    _parsed_vector,
    pair_universe_sha256,
    sha256_file,
)


IMPROVED_LP_TRANSFORM_KIND = "text_signal_improved_lp_transform_v1"
IMPROVED_LP_TRANSFORM_SCHEMA_VERSION = 1
WRONG_TEXT_DONOR_KIND = "text_signal_wrong_text_donor_manifest_v1"
WRONG_TEXT_DONOR_SCHEMA_VERSION = 1

_IMPROVED_METHOD = (
    "article_l2_recency_half_life_train_center_top_pc1_remove_final_l2_v1"
)
_WRONG_TEXT_METHOD = "partition_min_cost_time_current_surface_different_session_v1"
_SURFACE_SUMMARY_LABELS = (
    "mean",
    "std",
    "minimum",
    "q25",
    "median",
    "q75",
    "maximum",
)


@dataclass(frozen=True)
class PairArticleEmbedding:
    """One deterministically selected article embedding within a market pair."""

    pair_id: str
    article_key: str
    news_available_time_utc: pd.Timestamp
    pair_timestamp_utc: pd.Timestamp
    embedding: np.ndarray


@dataclass(frozen=True)
class ImprovedLPTransform:
    """Train-only centering and top-PC removal for recency-weighted LP means."""

    embedding_dim: int
    half_life_minutes: float
    train_pair_ids: tuple[str, ...]
    train_pair_universe_sha256: str
    train_source_sha256: str
    train_mean: np.ndarray
    top_pc1: np.ndarray
    train_mean_array_sha256: str
    top_pc1_array_sha256: str
    transform_sha256: str


@dataclass(frozen=True)
class WrongTextDonorPlan:
    """A partition-local, session-exclusive minimum-cost donor permutation."""

    namespace: str
    master_seed: int
    time_cost_scale_minutes: float
    mapping: Mapping[str, str]
    partition_by_pair: Mapping[str, str]
    session_by_pair: Mapping[str, str]
    cost_by_pair: Mapping[str, float]
    descriptor_source_sha256: str
    train_pair_universe_sha256: str
    partition_universe_sha256: Mapping[str, str]
    surface_summary_mean: np.ndarray
    surface_summary_std: np.ndarray
    profile_sha256: str


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Input frame is missing columns: {missing}")


def _normalized_pair_ids(values: Sequence[object], *, label: str) -> tuple[str, ...]:
    pair_ids = tuple(sorted(str(value).strip() for value in values))
    if not pair_ids or any(not pair_id for pair_id in pair_ids):
        raise ValueError(f"{label} must contain non-empty pair IDs")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"{label} contains duplicate pair IDs")
    return pair_ids


def _utc_timestamp(value: object, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{label} is missing")
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _timestamp_text(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _array_sha256(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    header = _canonical_json_bytes(
        {"dtype": "float32-le", "shape": list(normalized.shape)}
    )
    return hashlib.sha256(header + b"\n" + normalized.tobytes(order="C")).hexdigest()


def _parse_surface_vector(value: object, *, dimension: int, label: str) -> np.ndarray:
    return _parsed_vector(value, dimension=dimension, label=label)


def deduplicate_lp_articles(
    frame: pd.DataFrame,
    *,
    required_pair_ids: Sequence[object] | None = None,
    embedding_dim: int = 1024,
    pair_id_column: str = "pair_id",
    embedding_column: str = "lp_embedding",
    available_time_column: str = "news_available_time_utc",
    pair_timestamp_column: str = "current_snapshot_time_utc",
) -> dict[str, tuple[PairArticleEmbedding, ...]]:
    """Parse and deduplicate pair articles using the historical overlay rule.

    Within a pair, rows are sorted by numeric ``news_row_id`` then
    ``sample_id``.  The first row for the existing article-key fallback
    (article_id, sample_id, news_row_id) is retained, matching the old overlay.
    """

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    _require_columns(
        frame,
        (
            pair_id_column,
            embedding_column,
            available_time_column,
            pair_timestamp_column,
            "news_row_id",
            "sample_id",
        ),
    )
    selected = frame.copy()
    selected[pair_id_column] = selected[pair_id_column].map(
        lambda value: str(value).strip()
    )
    if (selected[pair_id_column] == "").any():
        raise ValueError("Article rows contain an empty pair_id")

    expected: tuple[str, ...] | None = None
    if required_pair_ids is not None:
        expected = _normalized_pair_ids(required_pair_ids, label="required_pair_ids")
        selected = selected.loc[selected[pair_id_column].isin(expected)].copy()
        observed = set(selected[pair_id_column])
        if observed != set(expected):
            missing = sorted(set(expected) - observed)
            raise ValueError(
                f"Article coverage is missing required pairs: {missing[:5]}"
            )
    if selected.empty:
        raise ValueError("No article rows were selected")

    result: dict[str, tuple[PairArticleEmbedding, ...]] = {}
    for pair_id, pair_rows in selected.groupby(pair_id_column, sort=False):
        ordered = pair_rows.assign(
            _probe_news_row_sort=pd.to_numeric(pair_rows["news_row_id"], errors="raise")
        ).sort_values(["_probe_news_row_sort", "sample_id"], kind="stable")
        articles: dict[str, PairArticleEmbedding] = {}
        pair_timestamp: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            article_key = _normalized_article_key(row)
            if article_key in articles:
                continue
            row_mapping = row._asdict()
            available = _utc_timestamp(
                row_mapping[available_time_column],
                label=f"news availability for {pair_id}/{article_key}",
            )
            current_pair_timestamp = _utc_timestamp(
                row_mapping[pair_timestamp_column],
                label=f"pair timestamp for {pair_id}/{article_key}",
            )
            if pair_timestamp is None:
                pair_timestamp = current_pair_timestamp
            elif current_pair_timestamp != pair_timestamp:
                raise ValueError(f"Pair timestamp drift within pair {pair_id}")
            if available > current_pair_timestamp:
                raise ValueError(
                    "News availability occurs after the pair timestamp: "
                    f"{pair_id}/{article_key}"
                )
            embedding = _parsed_vector(
                row_mapping[embedding_column],
                dimension=embedding_dim,
                label=f"LP embedding for {pair_id}/{article_key}",
            )
            articles[article_key] = PairArticleEmbedding(
                pair_id=str(pair_id),
                article_key=article_key,
                news_available_time_utc=available,
                pair_timestamp_utc=current_pair_timestamp,
                embedding=embedding.copy(),
            )
        if not articles:
            raise ValueError(f"Pair has no usable articles: {pair_id}")
        result[str(pair_id)] = tuple(articles[key] for key in sorted(articles))

    if expected is not None and set(result) != set(expected):
        raise ValueError("Deduplicated article universe does not match required pairs")
    return dict(sorted(result.items()))


def baseline_unique_article_mean_l2(
    pair_articles: Mapping[str, Sequence[PairArticleEmbedding]],
) -> dict[str, np.ndarray]:
    """Return the exact historical unique-article raw-mean then L2 baseline."""

    vectors: dict[str, np.ndarray] = {}
    for pair_id, articles in sorted(pair_articles.items()):
        if not articles:
            raise ValueError(f"Pair has no articles: {pair_id}")
        raw = np.stack(
            [np.asarray(article.embedding, dtype=np.float32) for article in articles],
            axis=0,
        )
        if raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError(f"Pair contains invalid embeddings: {pair_id}")
        vectors[str(pair_id)] = _l2(raw.mean(axis=0))
    if not vectors:
        raise ValueError("pair_articles must be non-empty")
    return vectors


def recency_weighted_pair_means(
    pair_articles: Mapping[str, Sequence[PairArticleEmbedding]],
    *,
    half_life_minutes: float = 5.0,
) -> dict[str, np.ndarray]:
    """L2 each article, apply causal recency weights, then take a weighted mean."""

    if not math.isfinite(half_life_minutes) or half_life_minutes <= 0:
        raise ValueError("half_life_minutes must be finite and positive")
    vectors: dict[str, np.ndarray] = {}
    for pair_id, articles in sorted(pair_articles.items()):
        if not articles:
            raise ValueError(f"Pair has no articles: {pair_id}")
        normalized: list[np.ndarray] = []
        weights: list[float] = []
        dimension: int | None = None
        for article in articles:
            vector = np.asarray(article.embedding, dtype=np.float32)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"Invalid article embedding in pair {pair_id}")
            if dimension is None:
                dimension = int(vector.size)
            elif vector.size != dimension:
                raise ValueError(f"Embedding dimension drift in pair {pair_id}")
            age_minutes = (
                article.pair_timestamp_utc - article.news_available_time_utc
            ).total_seconds() / 60.0
            if age_minutes < 0:
                raise ValueError(f"Future article detected in pair {pair_id}")
            normalized.append(_l2(vector))
            weights.append(math.exp(-math.log(2.0) * age_minutes / half_life_minutes))
        weight_array = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weight_array).all() or float(weight_array.sum()) <= 0:
            raise ValueError(f"Invalid recency weights for pair {pair_id}")
        matrix = np.stack(normalized, axis=0).astype(np.float64)
        vectors[str(pair_id)] = np.average(matrix, axis=0, weights=weight_array).astype(
            np.float32
        )
    if not vectors:
        raise ValueError("pair_articles must be non-empty")
    return vectors


def _train_source_sha256(
    pair_articles: Mapping[str, Sequence[PairArticleEmbedding]],
    train_pair_ids: Sequence[str],
) -> str:
    records: list[dict[str, object]] = []
    for pair_id in sorted(train_pair_ids):
        for article in pair_articles[pair_id]:
            records.append(
                {
                    "pair_id": pair_id,
                    "article_key": article.article_key,
                    "news_available_time_utc": _timestamp_text(
                        article.news_available_time_utc
                    ),
                    "pair_timestamp_utc": _timestamp_text(article.pair_timestamp_utc),
                    "embedding_sha256": _array_sha256(article.embedding),
                }
            )
    return hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def _transform_profile(transform: ImprovedLPTransform) -> dict[str, object]:
    return {
        "schema_version": IMPROVED_LP_TRANSFORM_SCHEMA_VERSION,
        "kind": IMPROVED_LP_TRANSFORM_KIND,
        "method": _IMPROVED_METHOD,
        "fit_scope": "train_pairs_only",
        "embedding_dim": int(transform.embedding_dim),
        "half_life_minutes": float(transform.half_life_minutes),
        "train_pair_ids": list(transform.train_pair_ids),
        "train_pair_universe_sha256": transform.train_pair_universe_sha256,
        "train_source_sha256": transform.train_source_sha256,
        "train_mean_array_sha256": transform.train_mean_array_sha256,
        "top_pc1_array_sha256": transform.top_pc1_array_sha256,
        "pc_sign_rule": "largest_absolute_loading_positive_v1",
    }


def _computed_transform_sha256(transform: ImprovedLPTransform) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_transform_profile(transform))
    ).hexdigest()


def fit_improved_lp_transform(
    pair_articles: Mapping[str, Sequence[PairArticleEmbedding]],
    train_pair_ids: Sequence[object],
    *,
    half_life_minutes: float = 5.0,
) -> ImprovedLPTransform:
    """Fit centering and PC1 exclusively on the declared train-pair universe."""

    train_ids = _normalized_pair_ids(train_pair_ids, label="train_pair_ids")
    missing = sorted(set(train_ids) - set(pair_articles))
    if missing:
        raise ValueError(f"Training article coverage is missing pairs: {missing[:5]}")
    if len(train_ids) < 2:
        raise ValueError("At least two train pairs are required to fit PC1")
    train_articles = {pair_id: pair_articles[pair_id] for pair_id in train_ids}
    raw_by_pair = recency_weighted_pair_means(
        train_articles, half_life_minutes=half_life_minutes
    )
    train_matrix = np.stack([raw_by_pair[pair_id] for pair_id in train_ids], axis=0)
    if train_matrix.ndim != 2 or train_matrix.shape[1] == 0:
        raise ValueError("Train pair embeddings must form a non-empty matrix")
    mean = train_matrix.astype(np.float64).mean(axis=0)
    centered = train_matrix.astype(np.float64) - mean
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or not math.isfinite(float(singular_values[0])):
        raise ValueError("Unable to fit top PC1")
    tolerance = (
        np.finfo(np.float64).eps
        * max(centered.shape)
        * max(1.0, float(singular_values[0]))
    )
    if float(singular_values[0]) <= tolerance:
        raise ValueError("Train pair embeddings have zero centered variance")
    top_pc1 = np.asarray(vh[0], dtype=np.float64)
    pivot = int(np.argmax(np.abs(top_pc1)))
    if top_pc1[pivot] < 0:
        top_pc1 = -top_pc1
    top_pc1 = top_pc1 / np.linalg.norm(top_pc1)
    mean32 = np.asarray(mean, dtype=np.float32)
    pc32 = _l2(np.asarray(top_pc1, dtype=np.float32))
    mean_sha = _array_sha256(mean32)
    pc_sha = _array_sha256(pc32)
    provisional = ImprovedLPTransform(
        embedding_dim=int(train_matrix.shape[1]),
        half_life_minutes=float(half_life_minutes),
        train_pair_ids=train_ids,
        train_pair_universe_sha256=pair_universe_sha256(train_ids),
        train_source_sha256=_train_source_sha256(pair_articles, train_ids),
        train_mean=mean32,
        top_pc1=pc32,
        train_mean_array_sha256=mean_sha,
        top_pc1_array_sha256=pc_sha,
        transform_sha256="",
    )
    return ImprovedLPTransform(
        **{
            **provisional.__dict__,
            "transform_sha256": _computed_transform_sha256(provisional),
        }
    )


def transform_improved_lp(
    pair_articles: Mapping[str, Sequence[PairArticleEmbedding]],
    transform: ImprovedLPTransform,
) -> dict[str, np.ndarray]:
    """Apply a frozen train-only LP transform to any declared partition."""

    _validate_improved_transform(transform)
    raw = recency_weighted_pair_means(
        pair_articles, half_life_minutes=transform.half_life_minutes
    )
    vectors: dict[str, np.ndarray] = {}
    mean = np.asarray(transform.train_mean, dtype=np.float32)
    pc1 = np.asarray(transform.top_pc1, dtype=np.float32)
    for pair_id, vector in raw.items():
        if vector.shape != (transform.embedding_dim,):
            raise ValueError(f"Embedding dimension drift for pair {pair_id}")
        centered = vector - mean
        cleaned = centered - float(np.dot(centered, pc1)) * pc1
        vectors[pair_id] = _l2(cleaned)
    return vectors


def _validate_improved_transform(transform: ImprovedLPTransform) -> None:
    if transform.embedding_dim <= 0 or transform.half_life_minutes <= 0:
        raise ValueError("Improved LP transform dimensions/half-life are invalid")
    if transform.train_mean.shape != (
        transform.embedding_dim,
    ) or transform.top_pc1.shape != (transform.embedding_dim,):
        raise ValueError("Improved LP transform array shape mismatch")
    if (
        not np.isfinite(transform.train_mean).all()
        or not np.isfinite(transform.top_pc1).all()
    ):
        raise ValueError("Improved LP transform arrays must be finite")
    if not np.isclose(np.linalg.norm(transform.top_pc1), 1.0, atol=1e-6):
        raise ValueError("Improved LP top_pc1 must be unit norm")
    if _array_sha256(transform.train_mean) != transform.train_mean_array_sha256:
        raise ValueError("Improved LP train_mean array SHA mismatch")
    if _array_sha256(transform.top_pc1) != transform.top_pc1_array_sha256:
        raise ValueError("Improved LP top_pc1 array SHA mismatch")
    if (
        pair_universe_sha256(transform.train_pair_ids)
        != transform.train_pair_universe_sha256
    ):
        raise ValueError("Improved LP train-pair universe SHA mismatch")
    if _computed_transform_sha256(transform) != transform.transform_sha256:
        raise ValueError("Improved LP transform SHA mismatch")


def _atomic_npy_write(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def save_improved_lp_transform(
    transform: ImprovedLPTransform,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the train-only transform, its arrays, and all file/array hashes."""

    _validate_improved_transform(transform)
    output = Path(output_dir).expanduser()
    mean_path = output / "train_mean.npy"
    pc_path = output / "top_pc1.npy"
    mean_file_sha = _atomic_npy_write(mean_path, transform.train_mean)
    pc_file_sha = _atomic_npy_write(pc_path, transform.top_pc1)
    profile = {
        **_transform_profile(transform),
        "transform_sha256": transform.transform_sha256,
        "files": {
            "train_mean": {
                "path": mean_path.name,
                "file_sha256": mean_file_sha,
                "array_sha256": transform.train_mean_array_sha256,
            },
            "top_pc1": {
                "path": pc_path.name,
                "file_sha256": pc_file_sha,
                "array_sha256": transform.top_pc1_array_sha256,
            },
        },
    }
    profile_sha = hashlib.sha256(_canonical_json_bytes(profile)).hexdigest()
    manifest_path = output / "transform.json"
    manifest_sha = _atomic_json_write(
        manifest_path, {**profile, "profile_sha256": profile_sha}
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "profile_sha256": profile_sha,
        "transform_sha256": transform.transform_sha256,
        "train_mean_path": str(mean_path),
        "train_mean_file_sha256": mean_file_sha,
        "top_pc1_path": str(pc_path),
        "top_pc1_file_sha256": pc_file_sha,
    }


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def load_improved_lp_transform(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_transform_sha256: str | None = None,
) -> ImprovedLPTransform:
    """Load and fully revalidate a serialized improved-LP transform."""

    path = Path(manifest_path).expanduser()
    if expected_manifest_sha256 is not None and sha256_file(path) != str(
        expected_manifest_sha256
    ):
        raise ValueError("Improved LP manifest SHA mismatch")
    payload = _read_json_mapping(path)
    if (
        payload.get("kind") != IMPROVED_LP_TRANSFORM_KIND
        or int(payload.get("schema_version", -1))
        != IMPROVED_LP_TRANSFORM_SCHEMA_VERSION
    ):
        raise ValueError("Improved LP transform schema/kind mismatch")
    if (
        payload.get("method") != _IMPROVED_METHOD
        or payload.get("fit_scope") != "train_pairs_only"
        or payload.get("pc_sign_rule") != "largest_absolute_loading_positive_v1"
    ):
        raise ValueError("Improved LP transform method contract mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Improved LP transform files are missing")

    arrays: dict[str, np.ndarray] = {}
    for name in ("train_mean", "top_pc1"):
        record = files.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"Improved LP transform file record is missing: {name}")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(
                "Improved LP transform array path must be a local filename"
            )
        array_path = path.parent / relative
        if sha256_file(array_path) != str(record.get("file_sha256", "")):
            raise ValueError(f"Improved LP {name} file SHA mismatch")
        try:
            array = np.load(array_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to load improved LP array: {name}") from exc
        arrays[name] = np.asarray(array, dtype=np.float32)
        if _array_sha256(arrays[name]) != str(record.get("array_sha256", "")):
            raise ValueError(f"Improved LP {name} array SHA mismatch")

    train_ids = _normalized_pair_ids(
        list(payload.get("train_pair_ids") or []), label="train_pair_ids"
    )
    transform = ImprovedLPTransform(
        embedding_dim=int(payload.get("embedding_dim", 0)),
        half_life_minutes=float(payload.get("half_life_minutes", 0.0)),
        train_pair_ids=train_ids,
        train_pair_universe_sha256=str(payload.get("train_pair_universe_sha256", "")),
        train_source_sha256=str(payload.get("train_source_sha256", "")),
        train_mean=arrays["train_mean"],
        top_pc1=arrays["top_pc1"],
        train_mean_array_sha256=str(payload.get("train_mean_array_sha256", "")),
        top_pc1_array_sha256=str(payload.get("top_pc1_array_sha256", "")),
        transform_sha256=str(payload.get("transform_sha256", "")),
    )
    _validate_improved_transform(transform)
    if expected_transform_sha256 is not None and transform.transform_sha256 != str(
        expected_transform_sha256
    ):
        raise ValueError("Unexpected improved LP transform SHA")
    profile_without_hash = dict(payload)
    profile_sha = str(profile_without_hash.pop("profile_sha256", ""))
    if (
        hashlib.sha256(_canonical_json_bytes(profile_without_hash)).hexdigest()
        != profile_sha
    ):
        raise ValueError("Improved LP manifest profile SHA mismatch")
    return transform


def _surface_summary(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    return np.asarray(
        [
            values.mean(),
            values.std(ddof=0),
            values.min(),
            np.quantile(values, 0.25),
            np.quantile(values, 0.50),
            np.quantile(values, 0.75),
            values.max(),
        ],
        dtype=np.float64,
    )


def _pair_descriptors(
    frame: pd.DataFrame,
    *,
    surface_dim: int,
    pair_id_column: str,
    partition_column: str,
    session_id_column: str,
    session_open_column: str,
    pair_timestamp_column: str,
    current_surface_column: str,
) -> dict[str, dict[str, object]]:
    _require_columns(
        frame,
        (
            pair_id_column,
            partition_column,
            session_id_column,
            session_open_column,
            pair_timestamp_column,
            current_surface_column,
        ),
    )
    descriptors: dict[str, dict[str, object]] = {}
    for raw_pair_id, rows in frame.groupby(pair_id_column, sort=False):
        pair_id = str(raw_pair_id).strip()
        if not pair_id:
            raise ValueError("Pair descriptor contains an empty pair_id")
        parsed: list[dict[str, object]] = []
        for row in rows.to_dict(orient="records"):
            partition = str(row[partition_column]).strip()
            session = str(row[session_id_column]).strip()
            if not partition or not session:
                raise ValueError(f"Pair partition/session is empty: {pair_id}")
            session_open = _utc_timestamp(
                row[session_open_column], label=f"session open for {pair_id}"
            )
            pair_timestamp = _utc_timestamp(
                row[pair_timestamp_column], label=f"pair timestamp for {pair_id}"
            )
            time_of_session = (pair_timestamp - session_open).total_seconds() / 60.0
            if not math.isfinite(time_of_session) or time_of_session < 0:
                raise ValueError(f"Invalid time-of-session for pair {pair_id}")
            surface = _parse_surface_vector(
                row[current_surface_column],
                dimension=surface_dim,
                label=f"current surface for {pair_id}",
            )
            parsed.append(
                {
                    "partition": partition,
                    "session": session,
                    "time_of_session": float(time_of_session),
                    "summary": _surface_summary(surface),
                }
            )
        first = parsed[0]
        for candidate in parsed[1:]:
            if (
                candidate["partition"] != first["partition"]
                or candidate["session"] != first["session"]
                or candidate["time_of_session"] != first["time_of_session"]
                or not np.array_equal(candidate["summary"], first["summary"])
            ):
                raise ValueError(
                    f"Pair descriptor drift across article rows: {pair_id}"
                )
        descriptors[pair_id] = first
    if not descriptors:
        raise ValueError("Pair descriptor frame is empty")
    return dict(sorted(descriptors.items()))


def validate_wrong_text_donor_mapping(
    mapping: Mapping[str, str],
    partition_by_pair: Mapping[str, str],
    session_by_pair: Mapping[str, str],
) -> None:
    """Reject any non-bijection, cross-partition, fixed, or same-session donor."""

    pair_ids = set(partition_by_pair)
    if set(mapping) != pair_ids or set(session_by_pair) != pair_ids:
        raise ValueError("Wrong-text mapping universe mismatch")
    for partition in sorted(set(partition_by_pair.values())):
        receivers = {
            pair_id
            for pair_id, value in partition_by_pair.items()
            if value == partition
        }
        donors = {mapping[pair_id] for pair_id in receivers}
        if donors != receivers:
            raise ValueError(f"Wrong-text donors are not a bijection in {partition}")
        for receiver in receivers:
            donor = mapping[receiver]
            if receiver == donor:
                raise ValueError(f"Wrong-text mapping has a fixed point: {receiver}")
            if partition_by_pair[donor] != partition:
                raise ValueError("Wrong-text mapping crosses partitions")
            if session_by_pair[receiver] == session_by_pair[donor]:
                raise ValueError(
                    f"Wrong-text mapping uses the receiver session: {receiver}"
                )


def _wrong_text_profile(plan: WrongTextDonorPlan) -> dict[str, object]:
    records = [
        {
            "receiver_pair_id": pair_id,
            "donor_pair_id": plan.mapping[pair_id],
            "partition": plan.partition_by_pair[pair_id],
            "receiver_session_id": plan.session_by_pair[pair_id],
            "donor_session_id": plan.session_by_pair[plan.mapping[pair_id]],
            "assignment_cost": float(plan.cost_by_pair[pair_id]),
        }
        for pair_id in sorted(plan.mapping)
    ]
    return {
        "schema_version": WRONG_TEXT_DONOR_SCHEMA_VERSION,
        "kind": WRONG_TEXT_DONOR_KIND,
        "method": _WRONG_TEXT_METHOD,
        "namespace": plan.namespace,
        "master_seed": int(plan.master_seed),
        "cost_inputs": [
            "time_of_session_minutes",
            "train_scaled_current_surface_summary",
        ],
        "prohibited_cost_inputs": ["target", "error", "text"],
        "time_cost_scale_minutes": float(plan.time_cost_scale_minutes),
        "surface_summary_labels": list(_SURFACE_SUMMARY_LABELS),
        "surface_summary_mean": [
            float(value) for value in plan.surface_summary_mean.tolist()
        ],
        "surface_summary_std": [
            float(value) for value in plan.surface_summary_std.tolist()
        ],
        "surface_summary_mean_array_sha256": _array_sha256(plan.surface_summary_mean),
        "surface_summary_std_array_sha256": _array_sha256(plan.surface_summary_std),
        "descriptor_source_sha256": plan.descriptor_source_sha256,
        "train_pair_universe_sha256": plan.train_pair_universe_sha256,
        "partition_universe_sha256": dict(
            sorted(plan.partition_universe_sha256.items())
        ),
        "records": records,
    }


def build_wrong_text_donor_plan(
    pair_frame: pd.DataFrame,
    *,
    master_seed: int,
    namespace: str,
    surface_dim: int = 256,
    time_cost_scale_minutes: float = 60.0,
    pair_id_column: str = "pair_id",
    partition_column: str = "partition",
    session_id_column: str = "session_id",
    session_open_column: str = "session_open_utc",
    pair_timestamp_column: str = "current_snapshot_time_utc",
    current_surface_column: str = "current_surface_flat",
    allowed_partitions: Sequence[str] = ("train", "validation"),
) -> WrongTextDonorPlan:
    """Build deterministic wrong-text donors from an explicit input allowlist."""

    normalized_namespace = str(namespace).strip()
    if not normalized_namespace:
        raise ValueError("namespace must be non-empty")
    if surface_dim <= 0:
        raise ValueError("surface_dim must be positive")
    if not math.isfinite(time_cost_scale_minutes) or time_cost_scale_minutes <= 0:
        raise ValueError("time_cost_scale_minutes must be finite and positive")
    descriptors = _pair_descriptors(
        pair_frame,
        surface_dim=surface_dim,
        pair_id_column=pair_id_column,
        partition_column=partition_column,
        session_id_column=session_id_column,
        session_open_column=session_open_column,
        pair_timestamp_column=pair_timestamp_column,
        current_surface_column=current_surface_column,
    )
    partition_by_pair = {
        pair_id: str(row["partition"]) for pair_id, row in descriptors.items()
    }
    allowed = {str(partition).strip() for partition in allowed_partitions}
    observed_partitions = set(partition_by_pair.values())
    if not allowed or "train" not in allowed:
        raise ValueError("allowed_partitions must include train")
    if not observed_partitions.issubset(allowed):
        raise ValueError(
            "Wrong-text input contains forbidden partitions: "
            f"{sorted(observed_partitions - allowed)}"
        )
    session_by_pair = {
        pair_id: str(row["session"]) for pair_id, row in descriptors.items()
    }
    train_ids = sorted(
        pair_id
        for pair_id, partition in partition_by_pair.items()
        if partition == "train"
    )
    if len(train_ids) < 2:
        raise ValueError("Wrong-text donor scaling requires at least two train pairs")
    train_summaries = np.stack(
        [
            np.asarray(descriptors[pair_id]["summary"], dtype=np.float64)
            for pair_id in train_ids
        ]
    )
    summary_mean = train_summaries.mean(axis=0)
    summary_std = train_summaries.std(axis=0, ddof=0)
    summary_std = np.where(summary_std == 0.0, 1.0, summary_std)
    summary_mean32 = np.asarray(summary_mean, dtype=np.float32)
    summary_std32 = np.asarray(summary_std, dtype=np.float32)
    descriptor_source_sha = hashlib.sha256(
        _canonical_json_bytes(
            [
                {
                    "pair_id": pair_id,
                    "partition": partition_by_pair[pair_id],
                    "session_id": session_by_pair[pair_id],
                    "time_of_session_minutes": float(
                        descriptors[pair_id]["time_of_session"]
                    ),
                    "current_surface_summary_sha256": _array_sha256(
                        np.asarray(descriptors[pair_id]["summary"], dtype=np.float32)
                    ),
                }
                for pair_id in sorted(descriptors)
            ]
        )
    ).hexdigest()

    mapping: dict[str, str] = {}
    costs: dict[str, float] = {}
    partition_hashes: dict[str, str] = {}
    for partition in sorted(set(partition_by_pair.values())):
        pair_ids = sorted(
            pair_id
            for pair_id, value in partition_by_pair.items()
            if value == partition
        )
        if len(pair_ids) < 2:
            raise ValueError(
                f"Wrong-text partition has fewer than two pairs: {partition}"
            )
        partition_hashes[partition] = pair_universe_sha256(pair_ids)
        largest_session = max(
            sum(session_by_pair[pair_id] == session for pair_id in pair_ids)
            for session in {session_by_pair[pair_id] for pair_id in pair_ids}
        )
        if largest_session * 2 > len(pair_ids):
            raise ValueError(
                f"No different-session donor bijection exists for partition {partition}"
            )
        times = np.asarray(
            [float(descriptors[pair_id]["time_of_session"]) for pair_id in pair_ids],
            dtype=np.float64,
        )
        summaries = np.stack(
            [
                np.asarray(descriptors[pair_id]["summary"], dtype=np.float64)
                for pair_id in pair_ids
            ]
        )
        # Use the serialized float32 scaler itself so the recorded artifact is
        # sufficient to reproduce each assignment cost exactly.
        scaled = (summaries - summary_mean32.astype(np.float64)) / summary_std32.astype(
            np.float64
        )
        time_cost = np.abs(times[:, None] - times[None, :]) / time_cost_scale_minutes
        surface_cost = np.sqrt(
            np.mean((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2)
        )
        base_cost = time_cost + surface_cost
        derived_seed = int.from_bytes(
            hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "method": _WRONG_TEXT_METHOD,
                        "master_seed": int(master_seed),
                        "namespace": normalized_namespace,
                        "partition": partition,
                        "pair_universe_sha256": partition_hashes[partition],
                    }
                )
            ).digest()[:8],
            byteorder="big",
            signed=False,
        )
        rng = np.random.default_rng(derived_seed)
        assignment_cost = base_cost + rng.uniform(0.0, 1.0e-9, base_cost.shape)
        forbidden = np.zeros(base_cost.shape, dtype=bool)
        for receiver_index, receiver in enumerate(pair_ids):
            for donor_index, donor in enumerate(pair_ids):
                if (
                    receiver == donor
                    or session_by_pair[receiver] == session_by_pair[donor]
                ):
                    forbidden[receiver_index, donor_index] = True
        assignment_cost[forbidden] = 1.0e12
        receiver_indices, donor_indices = linear_sum_assignment(assignment_cost)
        if (
            len(receiver_indices) != len(pair_ids)
            or forbidden[receiver_indices, donor_indices].any()
        ):
            raise ValueError(
                f"Unable to construct different-session donors for {partition}"
            )
        for receiver_index, donor_index in zip(receiver_indices, donor_indices):
            receiver = pair_ids[int(receiver_index)]
            donor = pair_ids[int(donor_index)]
            mapping[receiver] = donor
            costs[receiver] = float(base_cost[receiver_index, donor_index])

    validate_wrong_text_donor_mapping(mapping, partition_by_pair, session_by_pair)
    provisional = WrongTextDonorPlan(
        namespace=normalized_namespace,
        master_seed=int(master_seed),
        time_cost_scale_minutes=float(time_cost_scale_minutes),
        mapping=dict(sorted(mapping.items())),
        partition_by_pair=dict(sorted(partition_by_pair.items())),
        session_by_pair=dict(sorted(session_by_pair.items())),
        cost_by_pair=dict(sorted(costs.items())),
        descriptor_source_sha256=descriptor_source_sha,
        train_pair_universe_sha256=pair_universe_sha256(train_ids),
        partition_universe_sha256=dict(sorted(partition_hashes.items())),
        surface_summary_mean=summary_mean32,
        surface_summary_std=summary_std32,
        profile_sha256="",
    )
    profile_sha = hashlib.sha256(
        _canonical_json_bytes(_wrong_text_profile(provisional))
    ).hexdigest()
    return WrongTextDonorPlan(**{**provisional.__dict__, "profile_sha256": profile_sha})


def write_wrong_text_donor_manifest(
    plan: WrongTextDonorPlan,
    path: str | Path,
) -> dict[str, str]:
    """Validate and atomically serialize a wrong-text donor plan."""

    validate_wrong_text_donor_mapping(
        plan.mapping, plan.partition_by_pair, plan.session_by_pair
    )
    profile = _wrong_text_profile(plan)
    actual_profile_sha = hashlib.sha256(_canonical_json_bytes(profile)).hexdigest()
    if actual_profile_sha != plan.profile_sha256:
        raise ValueError("Wrong-text donor profile SHA mismatch")
    manifest_path = Path(path).expanduser()
    manifest_sha = _atomic_json_write(
        manifest_path, {**profile, "profile_sha256": actual_profile_sha}
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "profile_sha256": actual_profile_sha,
    }


def load_wrong_text_donor_manifest(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_profile_sha256: str | None = None,
) -> WrongTextDonorPlan:
    """Load, hash-check, and validate a serialized wrong-text donor plan."""

    manifest_path = Path(path).expanduser()
    if expected_manifest_sha256 is not None and sha256_file(manifest_path) != str(
        expected_manifest_sha256
    ):
        raise ValueError("Wrong-text donor manifest SHA mismatch")
    payload = _read_json_mapping(manifest_path)
    if (
        payload.get("kind") != WRONG_TEXT_DONOR_KIND
        or int(payload.get("schema_version", -1)) != WRONG_TEXT_DONOR_SCHEMA_VERSION
    ):
        raise ValueError("Wrong-text donor manifest schema/kind mismatch")
    if (
        payload.get("method") != _WRONG_TEXT_METHOD
        or payload.get("cost_inputs")
        != [
            "time_of_session_minutes",
            "train_scaled_current_surface_summary",
        ]
        or payload.get("prohibited_cost_inputs") != ["target", "error", "text"]
    ):
        raise ValueError("Wrong-text donor method contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Wrong-text donor manifest records are missing")
    mapping: dict[str, str] = {}
    partitions: dict[str, str] = {}
    sessions: dict[str, str] = {}
    costs: dict[str, float] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Wrong-text donor record must be an object")
        receiver = str(record.get("receiver_pair_id", "")).strip()
        donor = str(record.get("donor_pair_id", "")).strip()
        partition = str(record.get("partition", "")).strip()
        receiver_session = str(record.get("receiver_session_id", "")).strip()
        donor_session = str(record.get("donor_session_id", "")).strip()
        if not all((receiver, donor, partition, receiver_session, donor_session)):
            raise ValueError("Wrong-text donor record contains empty identifiers")
        if receiver in mapping:
            raise ValueError("Wrong-text donor manifest has duplicate receivers")
        mapping[receiver] = donor
        partitions[receiver] = partition
        sessions[receiver] = receiver_session
        costs[receiver] = float(record.get("assignment_cost", float("nan")))
        if not math.isfinite(costs[receiver]):
            raise ValueError("Wrong-text donor assignment cost must be finite")
    for record in records:
        donor = str(record["donor_pair_id"])
        if donor not in sessions or sessions[donor] != str(record["donor_session_id"]):
            raise ValueError("Wrong-text donor session lineage mismatch")
    mean = np.asarray(payload.get("surface_summary_mean"), dtype=np.float32)
    std = np.asarray(payload.get("surface_summary_std"), dtype=np.float32)
    if mean.shape != (len(_SURFACE_SUMMARY_LABELS),) or std.shape != mean.shape:
        raise ValueError("Wrong-text donor scaler shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Wrong-text donor scaler is invalid")
    if _array_sha256(mean) != str(payload.get("surface_summary_mean_array_sha256", "")):
        raise ValueError("Wrong-text donor mean array SHA mismatch")
    if _array_sha256(std) != str(payload.get("surface_summary_std_array_sha256", "")):
        raise ValueError("Wrong-text donor std array SHA mismatch")
    partition_hashes_raw = payload.get("partition_universe_sha256")
    if not isinstance(partition_hashes_raw, Mapping):
        raise ValueError("Wrong-text donor partition hashes are missing")
    partition_hashes = {
        str(key): str(value) for key, value in partition_hashes_raw.items()
    }
    for partition in set(partitions.values()):
        ids = sorted(
            pair_id for pair_id, value in partitions.items() if value == partition
        )
        if pair_universe_sha256(ids) != partition_hashes.get(partition):
            raise ValueError("Wrong-text donor partition universe SHA mismatch")
    train_ids = sorted(
        pair_id for pair_id, partition in partitions.items() if partition == "train"
    )
    if pair_universe_sha256(train_ids) != str(
        payload.get("train_pair_universe_sha256", "")
    ):
        raise ValueError("Wrong-text donor train universe SHA mismatch")
    validate_wrong_text_donor_mapping(mapping, partitions, sessions)
    profile_without_hash = dict(payload)
    profile_sha = str(profile_without_hash.pop("profile_sha256", ""))
    if (
        hashlib.sha256(_canonical_json_bytes(profile_without_hash)).hexdigest()
        != profile_sha
    ):
        raise ValueError("Wrong-text donor profile SHA mismatch")
    if expected_profile_sha256 is not None and profile_sha != str(
        expected_profile_sha256
    ):
        raise ValueError("Unexpected wrong-text donor profile SHA")
    plan = WrongTextDonorPlan(
        namespace=str(payload.get("namespace", "")),
        master_seed=int(payload.get("master_seed", 0)),
        time_cost_scale_minutes=float(payload.get("time_cost_scale_minutes", 0.0)),
        mapping=dict(sorted(mapping.items())),
        partition_by_pair=dict(sorted(partitions.items())),
        session_by_pair=dict(sorted(sessions.items())),
        cost_by_pair=dict(sorted(costs.items())),
        descriptor_source_sha256=str(payload.get("descriptor_source_sha256", "")),
        train_pair_universe_sha256=str(payload.get("train_pair_universe_sha256", "")),
        partition_universe_sha256=dict(sorted(partition_hashes.items())),
        surface_summary_mean=mean,
        surface_summary_std=std,
        profile_sha256=profile_sha,
    )
    if (
        hashlib.sha256(_canonical_json_bytes(_wrong_text_profile(plan))).hexdigest()
        != profile_sha
    ):
        raise ValueError("Wrong-text donor reconstructed profile SHA mismatch")
    return plan


def _embedding_multiset_sha256(vectors: Sequence[np.ndarray]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(sorted(_array_sha256(vector) for vector in vectors))
    ).hexdigest()


def transform_then_derange(
    transformed_embeddings: Mapping[str, np.ndarray],
    partition_by_pair: Mapping[str, str],
    *,
    master_seed: int,
    namespace: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Derange already-transformed vectors with the historical split semantics.

    The caller must use a master seed independent of the wrong-text donor plan.
    No vector is recomputed after permutation, so each partition's exact bytewise
    embedding multiset is preserved.
    """

    pair_ids = set(str(pair_id) for pair_id in transformed_embeddings)
    if pair_ids != set(partition_by_pair):
        raise ValueError("Shuffle embedding/partition universes do not match")
    normalized: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for pair_id, raw in transformed_embeddings.items():
        vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"Shuffle embedding is invalid: {pair_id}")
        if shape is None:
            shape = vector.shape
        elif vector.shape != shape:
            raise ValueError("Shuffle embedding dimensions are inconsistent")
        normalized[str(pair_id)] = vector.copy()
    donors: dict[str, str] = {}
    shuffled: dict[str, np.ndarray] = {}
    for partition in sorted(set(partition_by_pair.values())):
        partition_ids = sorted(
            pair_id
            for pair_id, value in partition_by_pair.items()
            if value == partition
        )
        partition_mapping = _fixed_partition_derangement(
            partition_ids,
            master_seed=int(master_seed),
            namespace=f"{namespace}/{partition}",
        )
        donors.update(partition_mapping)
        for receiver, donor in partition_mapping.items():
            shuffled[receiver] = normalized[donor].copy()
        before_sha = _embedding_multiset_sha256(
            [normalized[pair_id] for pair_id in partition_ids]
        )
        after_sha = _embedding_multiset_sha256(
            [shuffled[pair_id] for pair_id in partition_ids]
        )
        if before_sha != after_sha:
            raise RuntimeError(f"Shuffle changed the embedding multiset in {partition}")
    return dict(sorted(shuffled.items())), dict(sorted(donors.items()))


__all__ = [
    "IMPROVED_LP_TRANSFORM_KIND",
    "IMPROVED_LP_TRANSFORM_SCHEMA_VERSION",
    "ImprovedLPTransform",
    "PairArticleEmbedding",
    "WRONG_TEXT_DONOR_KIND",
    "WRONG_TEXT_DONOR_SCHEMA_VERSION",
    "WrongTextDonorPlan",
    "baseline_unique_article_mean_l2",
    "build_wrong_text_donor_plan",
    "deduplicate_lp_articles",
    "fit_improved_lp_transform",
    "load_improved_lp_transform",
    "load_wrong_text_donor_manifest",
    "recency_weighted_pair_means",
    "save_improved_lp_transform",
    "transform_improved_lp",
    "transform_then_derange",
    "validate_wrong_text_donor_mapping",
    "write_wrong_text_donor_manifest",
]
