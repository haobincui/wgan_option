#!/usr/bin/env python3
"""Build all-origin and recent-news RQ1 datasets with 30-minute histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from film_wgan.support import (
    fit_raw_support_artifact,
    parse_raw_surface_params,
    raw_support_mask,
    reconstruct_raw_surface,
)
from film_wgan.text_transform import (
    fit_text_transform,
    l2_normalize_rows,
    ordered_ids_sha256,
    sha256_file,
)
from wgan_option.market.treasury_sessions import TreasuryGlobexSessionCalendar
from wgan_option.merge_support import load_news_base_frame


HISTORY_STEPS = 6
HISTORY_INTERVAL_MINUTES = 5
HISTORY_WINDOW_MINUTES = 30
HISTORY_MIN_VALID_STEPS = 3
TEXT_PCA_DIM = 128
TEXT_FEATURE_DIM = 389
FOLDS = {
    "2023Q1": {"train_end": "2022Q3", "val": "2022Q4", "test": "2023Q1"},
    "2023Q2": {"train_end": "2022Q4", "val": "2023Q1", "test": "2023Q2"},
    "2023Q3": {"train_end": "2023Q1", "val": "2023Q2", "test": "2023Q3"},
    "2023Q4": {"train_end": "2023Q2", "val": "2023Q3", "test": "2023Q4"},
}


def _canonical_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("min").strftime("%Y-%m-%dT%H:%M:00Z")


def _quarter(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.year}Q{timestamp.quarter}"


def _quarter_key(value: str) -> tuple[int, int]:
    return int(value[:4]), int(value[-1])


def _pair_id(origin: pd.Timestamp, target: pd.Timestamp) -> str:
    payload = f"{_canonical_timestamp(origin)}|{_canonical_timestamp(target)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value.astype(np.float32).reshape(-1)
    elif isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("LP embedding is empty.")
        try:
            result = np.asarray(json.loads(text), dtype=np.float32).reshape(-1)
        except json.JSONDecodeError:
            import ast

            result = np.asarray(ast.literal_eval(text), dtype=np.float32).reshape(-1)
    if result.size <= 0 or not np.all(np.isfinite(result)):
        raise ValueError("LP embedding is empty or non-finite.")
    return result


def _pool_embeddings(values: Sequence[np.ndarray], dimension: int) -> np.ndarray:
    if not values:
        return np.zeros(dimension, dtype=np.float32)
    normalized = l2_normalize_rows(np.stack(values).astype(np.float32))
    pooled = normalized.mean(axis=0)
    return (pooled / max(float(np.linalg.norm(pooled)), 1e-12)).astype(np.float32)


@dataclass(frozen=True)
class Candidate:
    origin: pd.Timestamp
    target: pd.Timestamp
    history_anchors: tuple[pd.Timestamp, ...]
    history_valid: tuple[bool, ...]
    session_id: str

    @property
    def pair_id(self) -> str:
        return _pair_id(self.origin, self.target)

    @property
    def quarter(self) -> str:
        return _quarter(self.origin)

    @property
    def sample_id(self) -> str:
        return f"event30_{self.origin.strftime('%Y%m%dT%H%MZ')}"


@dataclass(frozen=True)
class NewsSet:
    article_ids: tuple[str, ...]
    embedding_hashes: tuple[str, ...]
    embeddings: tuple[np.ndarray, ...]
    publication_times_utc: tuple[str, ...]
    pooled_embedding: np.ndarray
    latest_recency_minutes: float

    @property
    def has_news(self) -> bool:
        return bool(self.article_ids)


def _split(candidate: Candidate, fold: str) -> str:
    design = FOLDS[fold]
    if _quarter_key(candidate.quarter) <= _quarter_key(str(design["train_end"])):
        return "train"
    if candidate.quarter == design["val"]:
        return "val"
    if candidate.quarter == design["test"]:
        return "test"
    return "excluded"


def _read_index(
    database: Path,
    calendar: TreasuryGlobexSessionCalendar,
) -> tuple[dict[str, dict[str, Any]], list[Candidate], dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{database.resolve()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        model_counts = {
            str(row["surface_model"]): int(row["n"])
            for row in connection.execute(
                "SELECT surface_model, COUNT(*) AS n FROM surface_anchor GROUP BY surface_model"
            )
        }
        if set(model_counts) != {"raw"}:
            raise ValueError(f"Event30 requires a raw-only index; found {model_counts}.")
        anchors = {
            str(row["anchor_time_utc"]): {
                "surface_params": parse_raw_surface_params(row["params_json"]),
                "params_sha256": str(row["params_sha256"]),
            }
            for row in connection.execute(
                "SELECT anchor_time_utc, params_json, params_sha256 FROM surface_anchor"
            )
        }
        pairs = connection.execute(
            "SELECT origin_time_utc, target_time_utc FROM valid_pair ORDER BY origin_time_utc"
        ).fetchall()
        metadata = {
            str(row["key"]): json.loads(str(row["value"]))
            for row in connection.execute("SELECT key, value FROM metadata")
        }
    finally:
        connection.close()

    candidates: list[Candidate] = []
    for row in pairs:
        origin = pd.Timestamp(row["origin_time_utc"])
        target = pd.Timestamp(row["target_time_utc"])
        if target - origin != pd.Timedelta(minutes=5):
            raise ValueError(f"Unexpected forecast horizon: {origin} -> {target}")
        session = calendar.session_for_interval(
            origin - pd.Timedelta(minutes=HISTORY_WINDOW_MINUTES),
            target,
        )
        if session is None:
            continue
        history = tuple(
            origin - pd.Timedelta(minutes=HISTORY_INTERVAL_MINUTES * offset)
            for offset in range(HISTORY_STEPS - 1, -1, -1)
        )
        valid = tuple(_canonical_timestamp(value) in anchors for value in history)
        if not valid[-1]:
            raise ValueError(f"Valid pair is missing current anchor: {origin}")
        if sum(valid) < HISTORY_MIN_VALID_STEPS:
            continue
        candidates.append(Candidate(origin, target, history, valid, session.session_id))
    return anchors, candidates, {
        "surface_model_counts": model_counts,
        "surface_anchor_count": len(anchors),
        "valid_pair_count": len(pairs),
        "candidate_count": len(candidates),
        "metadata": metadata,
    }


def _load_news_by_minute(
    workbook: Path,
) -> tuple[dict[str, NewsSet], int, pd.DataFrame]:
    frame = load_news_base_frame(
        workbook,
        source_timezone="Europe/London",
        offset_minutes=0,
        publication_availability_lag_minutes=0,
    )
    frame = frame[frame["timestamp_parse_status"].eq("ok")].copy()
    frame["available_time"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    ).dt.floor("min")
    frame["article_key"] = frame["ArticleID"].astype(str).str.strip()
    missing = frame["article_key"].isin({"", "nan", "None"})
    frame.loc[missing, "article_key"] = frame.loc[missing, "news_row_id"].map(
        lambda value: f"news_row_{int(value)}"
    )

    valid_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    for row in frame.itertuples(index=False):
        try:
            embedding = _parse_embedding(row.LP_embedding)
        except (ValueError, SyntaxError) as exc:
            skipped_rows.append(
                {"news_row_id": int(row.news_row_id), "reason": str(exc)}
            )
            continue
        dimensions.add(int(embedding.size))
        valid_rows.append(
            {
                "available_time": row.available_time,
                "news_row_id": int(row.news_row_id),
                "article_key": str(row.article_key),
                "embedding": embedding,
                "embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
            }
        )
    if len(dimensions) != 1:
        raise ValueError(f"LP embedding dimensions are inconsistent: {sorted(dimensions)}")
    dimension = dimensions.pop()
    valid = pd.DataFrame(valid_rows).sort_values(
        ["available_time", "news_row_id"]
    )
    news_by_minute: dict[str, NewsSet] = {}
    for timestamp, group in valid.groupby("available_time", sort=True):
        unique = group.drop_duplicates("article_key", keep="first")
        unique = unique.drop_duplicates("embedding_sha256", keep="first")
        ids = tuple(unique["article_key"].astype(str))
        hashes = tuple(unique["embedding_sha256"].astype(str))
        embeddings = tuple(
            np.asarray(value, dtype=np.float32) for value in unique["embedding"]
        )
        canonical_time = _canonical_timestamp(timestamp)
        news_by_minute[canonical_time] = NewsSet(
            article_ids=ids,
            embedding_hashes=hashes,
            embeddings=embeddings,
            publication_times_utc=(canonical_time,) * len(ids),
            pooled_embedding=_pool_embeddings(embeddings, dimension),
            latest_recency_minutes=0.0,
        )
    return news_by_minute, dimension, pd.DataFrame(skipped_rows)


def _empty_news(dimension: int) -> NewsSet:
    return NewsSet(
        (),
        (),
        (),
        (),
        np.zeros(dimension, dtype=np.float32),
        0.0,
    )


def _news_for_origin(
    origin: pd.Timestamp,
    *,
    news_by_minute: Mapping[str, NewsSet],
    embedding_dim: int,
    lookback_minutes: int,
    exclude_origin_minute: bool,
) -> NewsSet:
    """Pool unique articles available in the configured minute window."""

    lookback = int(lookback_minutes)
    if lookback < 0:
        raise ValueError("news lookback minutes must be non-negative")
    if lookback == 0 and exclude_origin_minute:
        raise ValueError(
            "A zero-minute news lookback cannot exclude the origin minute."
        )
    first_offset = 1 if exclude_origin_minute else 0
    records: list[tuple[str, str, np.ndarray, str]] = []
    for offset in range(lookback, first_offset - 1, -1):
        publication_time = origin - pd.Timedelta(minutes=offset)
        minute_news = news_by_minute.get(_canonical_timestamp(publication_time))
        if minute_news is None:
            continue
        records.extend(
            zip(
                minute_news.article_ids,
                minute_news.embedding_hashes,
                minute_news.embeddings,
                minute_news.publication_times_utc,
            )
        )
    if not records:
        return _empty_news(embedding_dim)

    article_ids: list[str] = []
    embedding_hashes: list[str] = []
    embeddings: list[np.ndarray] = []
    publication_times: list[str] = []
    seen_articles: set[str] = set()
    seen_embeddings: set[str] = set()
    for article_id, embedding_hash, embedding, publication_time in records:
        if article_id in seen_articles or embedding_hash in seen_embeddings:
            continue
        seen_articles.add(article_id)
        seen_embeddings.add(embedding_hash)
        article_ids.append(article_id)
        embedding_hashes.append(embedding_hash)
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        publication_times.append(publication_time)
    recencies = [
        (origin - pd.Timestamp(value)).total_seconds() / 60.0
        for value in publication_times
    ]
    if any(value < 0.0 for value in recencies):
        raise ValueError("News window contains an article published after the origin.")
    if exclude_origin_minute and any(value < 1.0 for value in recencies):
        raise ValueError("Exclusive news window contains an origin-minute article.")
    return NewsSet(
        article_ids=tuple(article_ids),
        embedding_hashes=tuple(embedding_hashes),
        embeddings=tuple(embeddings),
        publication_times_utc=tuple(publication_times),
        pooled_embedding=_pool_embeddings(embeddings, embedding_dim),
        latest_recency_minutes=float(min(recencies)),
    )


def _fit_event_text_features(
    *,
    fold: str,
    retained: Sequence[Candidate],
    splits: Mapping[str, str],
    news_by_pair: Mapping[str, NewsSet],
    workbook: Path,
    fold_dir: Path,
    permutation_seed: int,
    dataset_tag: str,
    feature_mode: str,
    news_lookback_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
    train_events = [
        candidate
        for candidate in retained
        if splits[candidate.pair_id] == "train" and news_by_pair[candidate.pair_id].has_news
    ]
    unique_pools: dict[str, tuple[np.ndarray, str]] = {}
    for candidate in train_events:
        key = "|".join(news_by_pair[candidate.pair_id].embedding_hashes)
        unique_pools.setdefault(
            key,
            (news_by_pair[candidate.pair_id].pooled_embedding, candidate.pair_id),
        )
    if len(unique_pools) < 2:
        raise ValueError(
            f"Fold {fold} has fewer than two news-conditioned train pools."
        )
    input_dim = int(next(iter(unique_pools.values()))[0].size)
    fitted_components = min(TEXT_PCA_DIM, len(unique_pools), input_dim)
    pool_keys = sorted(unique_pools)
    transform = fit_text_transform(
        np.stack([unique_pools[key][0] for key in pool_keys]),
        mode="pca",
        components=fitted_components,
        whiten=False,
        train_pair_ids=[unique_pools[key][1] for key in pool_keys],
        input_workbook_path=workbook,
    )
    transform.metadata.update(
        {
            "feature_mode": feature_mode,
            "requested_output_dim": TEXT_PCA_DIM,
            "fitted_components": fitted_components,
            "zero_padding": TEXT_PCA_DIM - fitted_components,
            "fit_unique_pool_count": len(pool_keys),
            "fit_pool_keys_sha256": ordered_ids_sha256(pool_keys),
        }
    )
    transform_path = fold_dir / f"text_transform_{dataset_tag}.npz"
    transform.save(transform_path)

    train_counts = np.asarray(
        [len(news_by_pair[item.pair_id].article_ids) for item in train_events],
        dtype=np.float32,
    )
    count_mean = float(np.log1p(train_counts).mean())
    count_std = float(np.log1p(train_counts).std())
    if count_std < 1e-6:
        count_std = 1.0
    (fold_dir / "news_count_transform.json").write_text(
        json.dumps(
            {
                "fit_split": f"train_{dataset_tag}_only",
                "mean_log1p_count": count_mean,
                "std_log1p_count": count_std,
                "event_count": len(train_events),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    donor_candidates = sorted(train_events, key=lambda item: item.pair_id)
    donor_mapping: list[dict[str, Any]] = []

    def transformed(pool: np.ndarray) -> np.ndarray:
        value = transform.transform(pool).astype(np.float32)
        return np.pad(value, (0, TEXT_PCA_DIM - value.size)).astype(np.float32)

    donor_features = {
        item.pair_id: transformed(news_by_pair[item.pair_id].pooled_embedding)
        for item in donor_candidates
    }

    def donor_for(candidate: Candidate, event: NewsSet) -> Candidate:
        digest = hashlib.sha256(
            f"{permutation_seed}|{fold}|{candidate.pair_id}".encode("utf-8")
        ).digest()
        start = int.from_bytes(digest[:8], "big") % len(donor_candidates)
        target_ids = set(event.article_ids)
        for offset in range(len(donor_candidates)):
            donor = donor_candidates[(start + offset) % len(donor_candidates)]
            donor_ids = set(news_by_pair[donor.pair_id].article_ids)
            if donor.pair_id != candidate.pair_id and target_ids.isdisjoint(donor_ids):
                return donor
        raise ValueError(f"Fold {fold} has no disjoint shuffled donor for {candidate.pair_id}.")

    matched_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []
    shuffled_rows: list[np.ndarray] = []
    has_news: list[float] = []
    for candidate in retained:
        event = news_by_pair[candidate.pair_id]
        if not event.has_news:
            matched_rows.append(np.zeros(TEXT_FEATURE_DIM, dtype=np.float32))
            count_rows.append(np.zeros(TEXT_FEATURE_DIM, dtype=np.float32))
            shuffled_rows.append(np.zeros(TEXT_FEATURE_DIM, dtype=np.float32))
            has_news.append(0.0)
            continue
        semantic = transformed(event.pooled_embedding)
        count_z = (np.log1p(float(len(event.article_ids))) - count_mean) / count_std
        recency_scale = max(float(news_lookback_minutes), 1.0)
        normalized_recency = float(event.latest_recency_minutes) / recency_scale
        auxiliary = np.asarray(
            [count_z, 0.0, 0.0, 1.0, normalized_recency],
            dtype=np.float32,
        )
        matched_rows.append(
            np.concatenate([semantic, np.zeros(TEXT_PCA_DIM * 2), auxiliary]).astype(np.float32)
        )
        count_rows.append(
            np.concatenate([np.zeros(TEXT_PCA_DIM * 3), auxiliary]).astype(np.float32)
        )
        donor = donor_for(candidate, event)
        shuffled_rows.append(
            np.concatenate(
                [donor_features[donor.pair_id], np.zeros(TEXT_PCA_DIM * 2), auxiliary]
            ).astype(np.float32)
        )
        donor_mapping.append(
            {
                "fold": fold,
                "target_surface_pair_id": candidate.pair_id,
                "donor_surface_pair_id": donor.pair_id,
                "target_article_ids_json": json.dumps(event.article_ids),
                "donor_article_ids_json": json.dumps(news_by_pair[donor.pair_id].article_ids),
                "donor_split": "train",
                "permutation_seed": permutation_seed,
            }
        )
        has_news.append(1.0)
    return (
        np.stack(matched_rows),
        np.stack(count_rows),
        np.stack(shuffled_rows),
        np.asarray(has_news, dtype=np.float32),
        pd.DataFrame(donor_mapping),
        fitted_components,
    )


def _subset_npz(arrays: Mapping[str, np.ndarray], selector: np.ndarray) -> dict[str, np.ndarray]:
    sample_count = int(selector.size)
    result: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        result[key] = value[selector] if value.ndim > 0 and value.shape[0] == sample_count else value
    return result


def _materialize_fold(
    *,
    fold: str,
    anchors: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Candidate],
    news_by_pair: Mapping[str, NewsSet],
    workbook: Path,
    database_sha256: str,
    output_root: Path,
    permutation_seed: int,
    dataset_tag: str,
    news_alignment_mode: str,
    news_lookback_minutes: int,
    exclude_origin_minute: bool,
) -> dict[str, Any]:
    fold_dir = output_root / "folds" / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    included = [candidate for candidate in candidates if _split(candidate, fold) != "excluded"]
    splits = {candidate.pair_id: _split(candidate, fold) for candidate in included}
    train_records = [
        {
            "surface_pair_id": candidate.pair_id,
            "current_surface_params": anchors[_canonical_timestamp(candidate.origin)]["surface_params"],
            "target_surface_params": anchors[_canonical_timestamp(candidate.target)]["surface_params"],
        }
        for candidate in included
        if splits[candidate.pair_id] == "train"
    ]
    support = fit_raw_support_artifact(
        train_records,
        input_workbook_sha256=database_sha256,
        strike_bins=16,
        maturity_bins=16,
        quantile_low=0.05,
        quantile_high=0.95,
    )
    support_path = fold_dir / "raw_support_artifact.json"
    support.save(support_path)
    strike_grid = np.asarray(support.strike_grid, dtype=np.float32)
    maturity_grid = np.asarray(support.maturity_days_grid, dtype=np.float32)
    short_atm = (
        (maturity_grid[:, None] <= 21.0 + 1e-6)
        & (np.abs(strike_grid[None, :] - 1.0) <= 0.02 + 1e-6)
    )
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def reconstruct(timestamp: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        key = _canonical_timestamp(timestamp)
        if key not in cache:
            params = anchors[key]["surface_params"]
            cache[key] = (
                reconstruct_raw_surface(params, strike_grid=strike_grid, maturity_days_grid=maturity_grid),
                raw_support_mask(params, strike_grid=strike_grid, maturity_days_grid=maturity_grid),
            )
        return cache[key]

    history_surfaces: list[np.ndarray] = []
    history_support: list[np.ndarray] = []
    history_steps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_support: list[np.ndarray] = []
    retained: list[Candidate] = []
    manifest_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in included:
        values = np.zeros((HISTORY_STEPS, len(maturity_grid), len(strike_grid)), dtype=np.float32)
        masks = np.zeros_like(values, dtype=bool)
        steps = np.asarray(candidate.history_valid, dtype=bool)
        for index, (timestamp, valid) in enumerate(zip(candidate.history_anchors, steps)):
            if valid:
                values[index], masks[index] = reconstruct(timestamp)
        target, target_mask = reconstruct(candidate.target)
        evaluation = masks[-1] & target_mask
        if not np.any(evaluation):
            dropped.append(
                {
                    "surface_pair_id": candidate.pair_id,
                    "split": splits[candidate.pair_id],
                    "reason": "zero_current_target_support_intersection",
                }
            )
            continue
        event = news_by_pair[candidate.pair_id]
        retained.append(candidate)
        history_surfaces.append(values)
        history_support.append(masks)
        history_steps.append(steps)
        targets.append(target)
        target_support.append(target_mask)
        short_count = int((evaluation & short_atm).sum())
        manifest_rows.append(
            {
                "global_index": len(manifest_rows),
                "sample_id": candidate.sample_id,
                "surface_pair_id": candidate.pair_id,
                "split": splits[candidate.pair_id],
                "current_snapshot_time_utc": _canonical_timestamp(candidate.origin),
                "target_snapshot_time_utc": _canonical_timestamp(candidate.target),
                "history_anchor_times_json": json.dumps(
                    [_canonical_timestamp(value) for value in candidate.history_anchors]
                ),
                "history_valid_steps": int(steps.sum()),
                "supported_cell_count": int(evaluation.sum()),
                "short_atm_supported_cell_count": short_count,
                "short_atm_eligible": int(short_count > 0),
                "trading_day": candidate.origin.strftime("%Y-%m-%d"),
                "session_id": candidate.session_id,
                "event_group": "news_event" if event.has_news else "quiet",
                "has_news": int(event.has_news),
                "news_count_total": len(event.article_ids),
                "source_article_ids_json": json.dumps(event.article_ids),
                "source_news_times_utc_json": json.dumps(
                    event.publication_times_utc
                ),
                "latest_news_recency_minutes": (
                    event.latest_recency_minutes if event.has_news else np.nan
                ),
                "news_lookback_minutes": int(news_lookback_minutes),
                "origin_minute_news_excluded": int(exclude_origin_minute),
                "news_alignment_mode": news_alignment_mode,
            }
        )

    matched, counts, shuffled, has_news, donors, fitted_components = _fit_event_text_features(
        fold=fold,
        retained=retained,
        splits=splits,
        news_by_pair=news_by_pair,
        workbook=workbook,
        fold_dir=fold_dir,
        permutation_seed=permutation_seed,
        dataset_tag=dataset_tag,
        feature_mode=news_alignment_mode,
        news_lookback_minutes=news_lookback_minutes,
    )
    donors.to_csv(fold_dir / "shuffled_text_donor_mapping.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = fold_dir / "split_manifest_all_origins.csv"
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(dropped).to_csv(fold_dir / "support_grid_dropped_samples.csv", index=False)
    arrays = {
        "history_surfaces": np.stack(history_surfaces).astype(np.float32),
        "history_support_masks": np.stack(history_support).astype(np.uint8),
        "history_step_masks": np.stack(history_steps).astype(np.uint8),
        "target_surfaces": np.stack(targets).astype(np.float32),
        "target_support_masks": np.stack(target_support).astype(np.uint8),
        "matched_text_features": matched.astype(np.float32),
        "count_text_features": counts.astype(np.float32),
        "shuffled_text_features": shuffled.astype(np.float32),
        "has_news": has_news.astype(np.float32),
        "strike_grid": strike_grid,
        "maturity_days_grid": maturity_grid,
    }
    all_npz = fold_dir / "sequence_dataset_all_origins.npz"
    np.savez_compressed(all_npz, **arrays)

    event_selector = has_news > 0.0
    event_manifest = manifest.loc[event_selector].copy()
    event_manifest.insert(0, "all_origin_global_index", event_manifest["global_index"].astype(int))
    event_manifest["global_index"] = np.arange(len(event_manifest), dtype=int)
    event_manifest_path = fold_dir / f"split_manifest_{dataset_tag}.csv"
    event_manifest.to_csv(event_manifest_path, index=False)
    event_npz = fold_dir / f"sequence_dataset_{dataset_tag}.npz"
    np.savez_compressed(event_npz, **_subset_npz(arrays, event_selector))

    split_counts = {
        split: int(manifest["split"].eq(split).sum()) for split in ("train", "val", "test")
    }
    event_split_counts = {
        split: int(event_manifest["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    validation = {
        "status": "ok",
        "fold": fold,
        "included_before_support": len(included),
        "retained_after_support": len(manifest),
        "support_drop_count": len(dropped),
        "all_origin_split_counts": split_counts,
        "news_conditioned_dataset_tag": dataset_tag,
        "news_alignment_mode": news_alignment_mode,
        "news_lookback_minutes": int(news_lookback_minutes),
        "origin_minute_news_excluded": bool(exclude_origin_minute),
        "news_conditioned_split_counts": event_split_counts,
        "news_conditioned_total": len(event_manifest),
        "news_conditioned_article_assignments": int(
            event_manifest["news_count_total"].sum()
        ),
        "text_pca_requested_components": TEXT_PCA_DIM,
        "text_pca_fitted_components": fitted_components,
        "all_origin_npz_sha256": sha256_file(all_npz),
        "news_conditioned_npz_sha256": sha256_file(event_npz),
        "all_origin_manifest_sha256": sha256_file(manifest_path),
        "news_conditioned_manifest_sha256": sha256_file(event_manifest_path),
        "raw_support_sha256": sha256_file(support_path),
        "text_transform_sha256": sha256_file(
            fold_dir / f"text_transform_{dataset_tag}.npz"
        ),
    }
    (fold_dir / "dataset_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def _write_fold_configs(
    output_root: Path,
    base_config: Path,
    *,
    dataset_tag: str,
) -> None:
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    base_training = dict(payload.get("training") or payload)
    generate = dict(payload.get("generate_result") or {})
    for fold in FOLDS:
        fold_dir = output_root / "folds" / fold
        for stage, data_name, manifest_name in (
            ("stage_a_all_origins", "sequence_dataset_all_origins.npz", "split_manifest_all_origins.csv"),
            (
                f"stage_b_{dataset_tag}",
                f"sequence_dataset_{dataset_tag}.npz",
                f"split_manifest_{dataset_tag}.csv",
            ),
        ):
            training = dict(base_training)
            training.update(
                {
                    "data_path": str(fold_dir / data_name),
                    "split_manifest_path": str(fold_dir / manifest_name),
                    "surface_support_path": str(fold_dir / "raw_support_artifact.json"),
                    "text_transform_path": str(
                        fold_dir / f"text_transform_{dataset_tag}.npz"
                    ),
                    "output_root": "",
                    "checkpoints_path": "",
                    "metrics_path": "",
                }
            )
            if stage == "stage_a_all_origins":
                training.update(
                    {
                        "sequence_text_variant": "no_text",
                        "lambda_film": 0.0,
                        "lambda_mismatch": 0.0,
                    }
                )
            path = fold_dir / f"{stage}_training_config.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"training": training, "generate_result": generate},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )


def _news_mode(
    lookback_minutes: int,
    exclude_origin_minute: bool,
) -> tuple[str, str]:
    lookback = int(lookback_minutes)
    if lookback < 0:
        raise ValueError("--news-lookback-minutes must be non-negative")
    if lookback == 0:
        if exclude_origin_minute:
            raise ValueError(
                "--news-lookback-minutes=0 requires "
                "--no-exclude-origin-minute-news"
            )
        return "exact_news", "exact_origin_minute"
    boundary = "exclusive" if exclude_origin_minute else "inclusive"
    return f"prior_{lookback}m_news", f"prior_{lookback}m_{boundary}"


def build(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    database = Path(args.database).expanduser().resolve()
    workbook = Path(args.news_workbook).expanduser().resolve()
    calendar_path = Path(args.session_calendar).expanduser().resolve()
    base_config = Path(args.base_config).expanduser().resolve()
    for path in (database, workbook, calendar_path, base_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    calendar = TreasuryGlobexSessionCalendar.from_csv(calendar_path)
    anchors, candidates, index_audit = _read_index(database, calendar)
    news_by_minute, embedding_dim, skipped_news = _load_news_by_minute(workbook)
    skipped_news.to_csv(output_root / "skipped_news_rows.csv", index=False)
    dataset_tag, news_alignment_mode = _news_mode(
        int(args.news_lookback_minutes),
        bool(args.exclude_origin_minute_news),
    )
    news_by_pair = {
        candidate.pair_id: _news_for_origin(
            candidate.origin,
            news_by_minute=news_by_minute,
            embedding_dim=embedding_dim,
            lookback_minutes=int(args.news_lookback_minutes),
            exclude_origin_minute=bool(args.exclude_origin_minute_news),
        )
        for candidate in candidates
    }
    candidate_rows = []
    for candidate in candidates:
        event = news_by_pair[candidate.pair_id]
        candidate_rows.append(
            {
                "surface_pair_id": candidate.pair_id,
                "sample_id": candidate.sample_id,
                "current_snapshot_time_utc": _canonical_timestamp(candidate.origin),
                "target_snapshot_time_utc": _canonical_timestamp(candidate.target),
                "quarter": candidate.quarter,
                "history_valid_steps": int(sum(candidate.history_valid)),
                "history_anchor_times_json": json.dumps(
                    [_canonical_timestamp(value) for value in candidate.history_anchors]
                ),
                "session_id": candidate.session_id,
                "has_news": int(event.has_news),
                "news_count_total": len(event.article_ids),
                "source_article_ids_json": json.dumps(event.article_ids),
                "source_news_times_utc_json": json.dumps(
                    event.publication_times_utc
                ),
                "latest_news_recency_minutes": (
                    event.latest_recency_minutes if event.has_news else np.nan
                ),
                "news_alignment_mode": news_alignment_mode,
            }
        )
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output_root / "event30_candidates.csv", index=False)
    database_sha = sha256_file(database)
    folds = {
        fold: _materialize_fold(
            fold=fold,
            anchors=anchors,
            candidates=candidates,
            news_by_pair=news_by_pair,
            workbook=workbook,
            database_sha256=database_sha,
            output_root=output_root,
            permutation_seed=int(args.permutation_seed),
            dataset_tag=dataset_tag,
            news_alignment_mode=news_alignment_mode,
            news_lookback_minutes=int(args.news_lookback_minutes),
            exclude_origin_minute=bool(args.exclude_origin_minute_news),
        )
        for fold in FOLDS
    }
    _write_fold_configs(output_root, base_config, dataset_tag=dataset_tag)
    summary = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "surface_source": "raw",
        "source_timezone": "Europe/London",
        "news_conditioned_dataset_tag": dataset_tag,
        "news_alignment_mode": news_alignment_mode,
        "news_lookback_minutes": int(args.news_lookback_minutes),
        "origin_minute_news_excluded": bool(args.exclude_origin_minute_news),
        "history_steps": HISTORY_STEPS,
        "history_interval_minutes": HISTORY_INTERVAL_MINUTES,
        "history_window_minutes": HISTORY_WINDOW_MINUTES,
        "history_min_valid_steps": HISTORY_MIN_VALID_STEPS,
        "forecast_horizon_minutes": 5,
        "all_origin_candidate_count": len(candidates),
        "news_conditioned_candidate_count": int(candidate_frame["has_news"].sum()),
        "news_conditioned_article_assignments": int(
            candidate_frame["news_count_total"].sum()
        ),
        "database_path": str(database),
        "database_sha256": database_sha,
        "news_workbook_path": str(workbook),
        "news_workbook_sha256": sha256_file(workbook),
        "session_calendar_path": str(calendar_path),
        "session_calendar_sha256": sha256_file(calendar_path),
        "base_config_path": str(base_config),
        "base_config_sha256": sha256_file(base_config),
        "index_audit": index_audit,
        "folds": folds,
    }
    (output_root / "event30_dataset_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--news-workbook", required=True)
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--permutation-seed", type=int, default=20260722)
    parser.add_argument("--news-lookback-minutes", type=int, default=5)
    parser.add_argument(
        "--exclude-origin-minute-news",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    return build(build_parser().parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main()
