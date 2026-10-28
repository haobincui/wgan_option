"""Build fold-specific raw-vol rolling-history bundles for RQ1."""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

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


HISTORY_STEPS = 12
HISTORY_INTERVAL_MINUTES = 5
HISTORY_MIN_VALID_STEPS = 6
TEXT_BUCKETS = ((0, 15), (15, 30), (30, 60))
TEXT_PCA_COMPONENTS = 128
TEXT_OUTPUT_DIM = TEXT_PCA_COMPONENTS * len(TEXT_BUCKETS) + 5
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


def _surface_pair_id(origin: Any, target: Any) -> str:
    payload = f"{pd.Timestamp(origin).isoformat().replace('+00:00', 'Z')}|"
    payload += pd.Timestamp(target).isoformat().replace("+00:00", "Z")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _quarter(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    return f"{timestamp.year}Q{timestamp.quarter}"


def _quarter_key(value: str) -> tuple[int, int]:
    return int(value[:4]), int(value[-1])


def _parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value.astype(np.float32).reshape(-1)
    elif isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    else:
        text = str(value or "").strip()
        parsed: Any = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            break
        result = np.asarray(parsed if parsed is not None else [], dtype=np.float32).reshape(-1)
    if result.size <= 0 or not np.all(np.isfinite(result)):
        raise ValueError("LP embedding is empty or non-finite.")
    return result


def _pool_embeddings(values: Sequence[np.ndarray], *, dimension: int) -> np.ndarray:
    if not values:
        return np.zeros(dimension, dtype=np.float32)
    matrix = np.stack(values).astype(np.float32)
    normalized = l2_normalize_rows(matrix)
    pooled = normalized.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    return (
        pooled / max(norm, 1e-12)
    ).astype(np.float32)


@dataclass(frozen=True)
class SequenceCandidate:
    origin: pd.Timestamp
    target: pd.Timestamp
    history_anchors: tuple[pd.Timestamp, ...]
    history_valid: tuple[bool, ...]
    session_id: str

    @property
    def quarter(self) -> str:
        return _quarter(self.origin)

    @property
    def pair_id(self) -> str:
        return _surface_pair_id(self.origin, self.target)

    @property
    def sample_id(self) -> str:
        return f"seq_{self.origin.strftime('%Y%m%dT%H%MZ')}"


@dataclass(frozen=True)
class NewsWindow:
    bucket_article_ids: tuple[tuple[str, ...], ...]
    bucket_pools: np.ndarray
    article_ids: tuple[str, ...]
    latest_recency_minutes: float

    @property
    def counts(self) -> np.ndarray:
        return np.asarray(
            [len(values) for values in self.bucket_article_ids],
            dtype=np.float32,
        )

    @property
    def has_news(self) -> bool:
        return bool(self.article_ids)


def _read_market_index(
    database_path: Path,
    calendar: TreasuryGlobexSessionCalendar,
) -> tuple[dict[str, dict[str, Any]], list[SequenceCandidate], dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        model_counts = {
            str(row["surface_model"]): int(row["n"])
            for row in connection.execute(
                "SELECT surface_model, COUNT(*) AS n FROM surface_anchor "
                "GROUP BY surface_model"
            )
        }
        if set(model_counts) != {"raw"}:
            raise ValueError(
                f"Sequence input must be raw-only; found surface models {model_counts}."
            )
        anchors = {
            str(row["anchor_time_utc"]): {
                "surface_model": str(row["surface_model"]),
                "surface_params": parse_raw_surface_params(row["params_json"]),
                "params_sha256": str(row["params_sha256"]),
            }
            for row in connection.execute(
                "SELECT anchor_time_utc, surface_model, params_json, params_sha256 "
                "FROM surface_anchor"
            )
        }
        pair_rows = connection.execute(
            "SELECT origin_time_utc, target_time_utc FROM valid_pair "
            "ORDER BY origin_time_utc"
        ).fetchall()
        metadata = {
            str(row["key"]): json.loads(str(row["value"]))
            for row in connection.execute("SELECT key, value FROM metadata")
        }
    finally:
        connection.close()

    immutable = dict(metadata.get("immutable_settings") or {})
    if int(metadata.get("schema_version", -1)) != 2:
        raise ValueError(
            f"Unsupported market-index schema: {metadata.get('schema_version')}"
        )
    if int(immutable.get("window_minutes", -1)) != HISTORY_INTERVAL_MINUTES:
        raise ValueError(
            "Sequence input requires five-minute backward raw-vol windows; "
            f"found window_minutes={immutable.get('window_minutes')}."
        )
    if immutable.get("candidate_anchor_source") != "positive_price_volume_option_minutes":
        raise ValueError(
            "Unexpected market-index anchor source: "
            f"{immutable.get('candidate_anchor_source')}."
        )

    candidates: list[SequenceCandidate] = []
    for pair in pair_rows:
        origin = pd.Timestamp(pair["origin_time_utc"])
        target = pd.Timestamp(pair["target_time_utc"])
        if int(origin.minute) % HISTORY_INTERVAL_MINUTES != 0:
            continue
        if target - origin != pd.Timedelta(minutes=HISTORY_INTERVAL_MINUTES):
            raise ValueError(f"Invalid market-index pair horizon: {origin} -> {target}")
        session = calendar.session_for_interval(
            origin - pd.Timedelta(minutes=60),
            target,
        )
        if session is None:
            continue
        history_anchors = tuple(
            origin - pd.Timedelta(minutes=HISTORY_INTERVAL_MINUTES * offset)
            for offset in range(HISTORY_STEPS - 1, -1, -1)
        )
        history_valid = tuple(
            _canonical_timestamp(anchor) in anchors
            for anchor in history_anchors
        )
        if not history_valid[-1]:
            raise ValueError(f"Valid pair is missing its current anchor: {origin}")
        if sum(history_valid) < HISTORY_MIN_VALID_STEPS:
            continue
        candidates.append(
            SequenceCandidate(
                origin=origin,
                target=target,
                history_anchors=history_anchors,
                history_valid=history_valid,
                session_id=session.session_id,
            )
        )
    return anchors, candidates, {
        "surface_model_counts": model_counts,
        "market_index_metadata": metadata,
        "surface_anchor_count": len(anchors),
        "candidate_count": len(candidates),
    }


def _load_news_catalog(
    news_workbook: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], int, pd.DataFrame]:
    frame = load_news_base_frame(
        news_workbook,
        source_timezone="Europe/London",
        offset_minutes=0,
        publication_availability_lag_minutes=0,
    )
    frame = frame[frame["timestamp_parse_status"].eq("ok")].copy()
    frame["available_time"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="raise"
    )
    frame["article_key"] = frame["ArticleID"].astype(str).str.strip()
    missing_id = frame["article_key"].isin({"", "nan", "None"})
    frame.loc[missing_id, "article_key"] = frame.loc[missing_id, "news_row_id"].map(
        lambda value: f"news_row_{int(value)}"
    )
    embeddings = [_parse_embedding(value) for value in frame["LP_embedding"]]
    dimensions = {int(value.size) for value in embeddings}
    if len(dimensions) != 1:
        raise ValueError(f"LP embeddings have inconsistent dimensions: {sorted(dimensions)}")
    dimension = dimensions.pop()
    frame["embedding"] = embeddings
    frame["embedding_sha256"] = [
        hashlib.sha256(value.tobytes()).hexdigest()
        for value in embeddings
    ]
    frame = frame.sort_values(["available_time", "news_row_id"]).reset_index(drop=True)
    catalog: dict[str, np.ndarray] = {}
    conflict_rows: list[dict[str, Any]] = []
    for article_key, group in frame.groupby("article_key", sort=True):
        ordered = group.sort_values(["available_time", "news_row_id"])
        selected = ordered.iloc[0]
        catalog[str(article_key)] = np.asarray(selected["embedding"], dtype=np.float32)
        distinct_hashes = tuple(dict.fromkeys(ordered["embedding_sha256"].astype(str)))
        if len(distinct_hashes) > 1:
            conflict_rows.append(
                {
                    "article_id": str(article_key),
                    "row_count": int(len(ordered)),
                    "distinct_embedding_count": len(distinct_hashes),
                    "selected_news_row_id": int(selected["news_row_id"]),
                    "selected_embedding_sha256": str(selected["embedding_sha256"]),
                    "news_row_ids_json": json.dumps(
                        ordered["news_row_id"].astype(int).tolist()
                    ),
                    "embedding_sha256_json": json.dumps(distinct_hashes),
                    "source_files_json": json.dumps(
                        ordered["SourceFile"].astype(str).tolist()
                    ),
                    "resolution": "keep_earliest_available_then_news_row_id",
                }
            )
    return frame, catalog, dimension, pd.DataFrame(conflict_rows)


def _build_news_windows(
    candidates: Sequence[SequenceCandidate],
    news: pd.DataFrame,
    catalog: Mapping[str, np.ndarray],
    dimension: int,
) -> tuple[dict[str, NewsWindow], pd.DataFrame]:
    news_times = news["available_time"].astype("int64").to_numpy()
    windows: dict[str, NewsWindow] = {}
    audit_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        origin_ns = int(candidate.origin.value)
        lower_ns = int((candidate.origin - pd.Timedelta(minutes=60)).value)
        start = int(np.searchsorted(news_times, lower_ns, side="right"))
        stop = int(np.searchsorted(news_times, origin_ns, side="right"))
        selected = news.iloc[start:stop]
        bucket_ids: list[tuple[str, ...]] = []
        bucket_pools: list[np.ndarray] = []
        all_ids: list[str] = []
        for lower, upper in TEXT_BUCKETS:
            lag = (
                candidate.origin - selected["available_time"]
            ).dt.total_seconds().div(60.0)
            bucket = selected[(lag >= lower) & (lag < upper)].copy()
            bucket = bucket.drop_duplicates("article_key", keep="first")
            bucket = bucket.drop_duplicates("embedding_sha256", keep="first")
            ids = tuple(bucket["article_key"].astype(str).tolist())
            pool = _pool_embeddings(
                [catalog[value] for value in ids],
                dimension=dimension,
            )
            bucket_ids.append(ids)
            bucket_pools.append(pool)
            all_ids.extend(ids)
            audit_rows.append(
                {
                    "surface_pair_id": candidate.pair_id,
                    "current_snapshot_time_utc": _canonical_timestamp(candidate.origin),
                    "bucket_start_minutes": lower,
                    "bucket_end_minutes": upper,
                    "unique_article_count": len(ids),
                    "article_ids_json": json.dumps(ids),
                }
            )
        unique_ids = tuple(dict.fromkeys(all_ids))
        if selected.empty:
            recency = 60.0
        else:
            recency = float(
                (candidate.origin - selected["available_time"].max()).total_seconds()
                / 60.0
            )
            recency = min(max(recency, 0.0), 60.0)
        windows[candidate.pair_id] = NewsWindow(
            bucket_article_ids=tuple(bucket_ids),
            bucket_pools=np.stack(bucket_pools).astype(np.float32),
            article_ids=unique_ids,
            latest_recency_minutes=recency,
        )
    return windows, pd.DataFrame(audit_rows)


def _split_for_fold(candidate: SequenceCandidate, fold: str) -> str:
    design = FOLDS[fold]
    quarter = candidate.quarter
    if _quarter_key(quarter) <= _quarter_key(str(design["train_end"])):
        return "train"
    if quarter == design["val"]:
        return "val"
    if quarter == design["test"]:
        return "test"
    return "excluded"


def _fit_fold_text_features(
    *,
    fold: str,
    candidates: Sequence[SequenceCandidate],
    splits: Mapping[str, str],
    windows: Mapping[str, NewsWindow],
    catalog: Mapping[str, np.ndarray],
    news_workbook: Path,
    fold_dir: Path,
    permutation_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    unique_pool_vectors: dict[str, np.ndarray] = {}
    train_article_ids: set[str] = set()
    for candidate in candidates:
        if splits[candidate.pair_id] != "train":
            continue
        window = windows[candidate.pair_id]
        train_article_ids.update(window.article_ids)
        for ids, pool in zip(window.bucket_article_ids, window.bucket_pools):
            if ids:
                unique_pool_vectors.setdefault("|".join(ids), pool)
    if len(unique_pool_vectors) < TEXT_PCA_COMPONENTS:
        raise ValueError(
            f"Fold {fold} has only {len(unique_pool_vectors)} unique train text pools; "
            f"cannot fit PCA-{TEXT_PCA_COMPONENTS}."
        )
    pool_keys = sorted(unique_pool_vectors)
    train_candidates = [
        candidate
        for candidate in candidates
        if splits[candidate.pair_id] == "train"
    ]
    train_pair_ids = [candidate.pair_id for candidate in train_candidates]
    transform = fit_text_transform(
        np.stack([unique_pool_vectors[key] for key in pool_keys]),
        mode="pca",
        components=TEXT_PCA_COMPONENTS,
        whiten=False,
        train_pair_ids=train_pair_ids,
        input_workbook_path=news_workbook,
    )
    transform.metadata.update(
        {
            "fit_unique_pool_count": len(pool_keys),
            "fit_pool_keys_sha256": ordered_ids_sha256(pool_keys),
            "pool_weighting": "unique_bucket_article_set",
        }
    )
    transform_path = fold_dir / "text_transform.npz"
    transform.save(transform_path)

    scalar_train = np.stack(
        [
            np.concatenate(
                [
                    np.log1p(windows[item.pair_id].counts),
                    np.asarray(
                        [windows[item.pair_id].latest_recency_minutes / 60.0],
                        dtype=np.float32,
                    ),
                ]
            )
            for item in train_candidates
        ]
    ).astype(np.float32)
    scalar_mean = scalar_train.mean(axis=0).astype(np.float32)
    scalar_std = scalar_train.std(axis=0).astype(np.float32)
    scalar_std[scalar_std < 1e-6] = 1.0
    (fold_dir / "news_scalar_transform.json").write_text(
        json.dumps(
            {
                "features": [
                    "log1p_count_0_15",
                    "log1p_count_15_30",
                    "log1p_count_30_60",
                    "latest_news_recency_div_60",
                ],
                "mean": scalar_mean.astype(float).tolist(),
                "std": scalar_std.astype(float).tolist(),
                "fit_split": "train",
                "fold": fold,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    donor_ids = sorted(train_article_ids)
    if len(donor_ids) < 2:
        raise ValueError(f"Fold {fold} has too few train articles for shuffled LP.")
    donor_rows: dict[str, str] = {}

    def donor_for(article_id: str) -> str:
        if article_id not in donor_rows:
            digest = hashlib.sha256(
                f"{permutation_seed}|{fold}|{article_id}".encode("utf-8")
            ).digest()
            index = int.from_bytes(digest[:8], "big") % len(donor_ids)
            if donor_ids[index] == article_id:
                index = (index + 1) % len(donor_ids)
            donor_rows[article_id] = donor_ids[index]
        return donor_rows[article_id]

    matched_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []
    shuffled_rows: list[np.ndarray] = []
    has_news_rows: list[float] = []
    for candidate in candidates:
        window = windows[candidate.pair_id]
        semantic: list[np.ndarray] = []
        shuffled_semantic: list[np.ndarray] = []
        for ids, pool in zip(window.bucket_article_ids, window.bucket_pools):
            if ids:
                semantic.append(transform.transform(pool))
                shuffled_pool = _pool_embeddings(
                    [catalog[donor_for(article_id)] for article_id in ids],
                    dimension=transform.input_dim,
                )
                shuffled_semantic.append(transform.transform(shuffled_pool))
            else:
                semantic.append(np.zeros(TEXT_PCA_COMPONENTS, dtype=np.float32))
                shuffled_semantic.append(
                    np.zeros(TEXT_PCA_COMPONENTS, dtype=np.float32)
                )
        scalar_raw = np.concatenate(
            [
                np.log1p(window.counts),
                np.asarray(
                    [window.latest_recency_minutes / 60.0],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        scalar_z = ((scalar_raw - scalar_mean) / scalar_std).astype(np.float32)
        has_news = float(window.has_news)
        auxiliary = np.concatenate(
            [scalar_z[:3], np.asarray([has_news]), scalar_z[3:]]
        ).astype(np.float32)
        matched = np.concatenate([*semantic, auxiliary]).astype(np.float32)
        count = np.concatenate(
            [np.zeros(TEXT_PCA_COMPONENTS * 3, dtype=np.float32), auxiliary]
        )
        shuffled = np.concatenate([*shuffled_semantic, auxiliary]).astype(np.float32)
        if not window.has_news:
            matched.fill(0.0)
            count.fill(0.0)
            shuffled.fill(0.0)
        matched_rows.append(matched)
        count_rows.append(count)
        shuffled_rows.append(shuffled)
        has_news_rows.append(has_news)
    mapping = pd.DataFrame(
        [
            {
                "target_article_id": target,
                "donor_article_id": donor,
                "donor_source_split": "train",
                "fold": fold,
                "permutation_seed": int(permutation_seed),
            }
            for target, donor in sorted(donor_rows.items())
        ]
    )
    return (
        np.stack(matched_rows),
        np.stack(count_rows),
        np.stack(shuffled_rows),
        np.asarray(has_news_rows, dtype=np.float32),
        mapping,
    )


def _materialize_fold(
    *,
    fold: str,
    anchors: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[SequenceCandidate],
    windows: Mapping[str, NewsWindow],
    catalog: Mapping[str, np.ndarray],
    news_workbook: Path,
    database_sha256: str,
    output_root: Path,
    permutation_seed: int,
) -> dict[str, Any]:
    fold_dir = output_root / "folds" / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    included = [
        candidate
        for candidate in candidates
        if _split_for_fold(candidate, fold) != "excluded"
    ]
    splits = {
        candidate.pair_id: _split_for_fold(candidate, fold)
        for candidate in included
    }
    train_records = [
        {
            "surface_pair_id": candidate.pair_id,
            "current_surface_params": anchors[
                _canonical_timestamp(candidate.origin)
            ]["surface_params"],
            "target_surface_params": anchors[
                _canonical_timestamp(candidate.target)
            ]["surface_params"],
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
    short_atm_mask = (
        (maturity_grid[:, None] <= 21.0 + 1e-6)
        & (np.abs(strike_grid[None, :] - 1.0) <= 0.02 + 1e-6)
    )

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def reconstruct(anchor: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        key = _canonical_timestamp(anchor)
        if key not in cache:
            params = anchors[key]["surface_params"]
            cache[key] = (
                reconstruct_raw_surface(
                    params,
                    strike_grid=strike_grid,
                    maturity_days_grid=maturity_grid,
                ),
                raw_support_mask(
                    params,
                    strike_grid=strike_grid,
                    maturity_days_grid=maturity_grid,
                ),
            )
        return cache[key]

    history_surfaces: list[np.ndarray] = []
    history_support: list[np.ndarray] = []
    history_steps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_support: list[np.ndarray] = []
    manifest_rows: list[dict[str, Any]] = []
    retained_candidates: list[SequenceCandidate] = []
    dropped_rows: list[dict[str, Any]] = []
    for candidate in included:
        history_values = np.zeros(
            (HISTORY_STEPS, maturity_grid.size, strike_grid.size),
            dtype=np.float32,
        )
        support_values = np.zeros_like(history_values, dtype=bool)
        step_values = np.asarray(candidate.history_valid, dtype=bool)
        for step_index, (anchor, valid) in enumerate(
            zip(candidate.history_anchors, candidate.history_valid)
        ):
            if valid:
                history_values[step_index], support_values[step_index] = reconstruct(anchor)
        target_value, target_mask = reconstruct(candidate.target)
        evaluation_mask = support_values[-1] & target_mask
        supported_cell_count = int(evaluation_mask.sum())
        if supported_cell_count < 1:
            dropped_rows.append(
                {
                    "surface_pair_id": candidate.pair_id,
                    "split": splits[candidate.pair_id],
                    "reason": "zero_current_target_support_intersection",
                }
            )
            continue
        retained_candidates.append(candidate)
        global_index = len(manifest_rows)
        window = windows[candidate.pair_id]
        history_surfaces.append(history_values)
        history_support.append(support_values)
        history_steps.append(step_values)
        targets.append(target_value)
        target_support.append(target_mask)
        short_atm_supported_cell_count = int(
            (evaluation_mask & short_atm_mask).sum()
        )
        manifest_rows.append(
            {
                "global_index": global_index,
                "sample_id": candidate.sample_id,
                "surface_pair_id": candidate.pair_id,
                "split": splits[candidate.pair_id],
                "current_snapshot_time_utc": _canonical_timestamp(candidate.origin),
                "target_snapshot_time_utc": _canonical_timestamp(candidate.target),
                "history_anchor_times_json": json.dumps(
                    [_canonical_timestamp(value) for value in candidate.history_anchors]
                ),
                "history_valid_steps": int(step_values.sum()),
                "supported_cell_count": supported_cell_count,
                "short_atm_supported_cell_count": short_atm_supported_cell_count,
                "short_atm_eligible": int(short_atm_supported_cell_count > 0),
                "trading_day": candidate.origin.strftime("%Y-%m-%d"),
                "session_id": candidate.session_id,
                "news_count_total": int(len(window.article_ids)),
                "news_count_0_15": int(window.counts[0]),
                "news_count_15_30": int(window.counts[1]),
                "news_count_30_60": int(window.counts[2]),
                "latest_news_recency_minutes": float(window.latest_recency_minutes),
                "source_article_ids_json": json.dumps(window.article_ids),
            }
        )
    matched, counts, shuffled, has_news, donor_mapping = _fit_fold_text_features(
        fold=fold,
        candidates=retained_candidates,
        splits=splits,
        windows=windows,
        catalog=catalog,
        news_workbook=news_workbook,
        fold_dir=fold_dir,
        permutation_seed=permutation_seed,
    )
    donor_mapping.to_csv(fold_dir / "shuffled_text_donor_mapping.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(fold_dir / "split_manifest.csv", index=False)
    pd.DataFrame(dropped_rows).to_csv(
        fold_dir / "support_grid_dropped_samples.csv",
        index=False,
    )
    np.savez_compressed(
        fold_dir / "sequence_dataset.npz",
        history_surfaces=np.stack(history_surfaces).astype(np.float32),
        history_support_masks=np.stack(history_support).astype(np.uint8),
        history_step_masks=np.stack(history_steps).astype(np.uint8),
        target_surfaces=np.stack(targets).astype(np.float32),
        target_support_masks=np.stack(target_support).astype(np.uint8),
        matched_text_features=matched.astype(np.float32),
        count_text_features=counts.astype(np.float32),
        shuffled_text_features=shuffled.astype(np.float32),
        has_news=has_news.astype(np.float32),
        strike_grid=strike_grid,
        maturity_days_grid=maturity_grid,
    )
    split_counts = {
        str(key): int(value)
        for key, value in manifest["split"].value_counts().sort_index().items()
    }
    short_atm_eligible_counts = {
        split: {
            "eligible": int(
                manifest.loc[
                    manifest["split"].eq(split), "short_atm_eligible"
                ].sum()
            ),
            "total": int(manifest["split"].eq(split).sum()),
        }
        for split in ("train", "val", "test")
    }
    validation = {
        "status": "ok",
        "fold": fold,
        "source_surface_model": "raw",
        "included_before_support": len(included),
        "retained_after_support": len(manifest),
        "support_drop_count": len(dropped_rows),
        "split_counts": split_counts,
        "short_atm_definition": {
            "absolute_moneyness_distance_max": 0.02,
            "maturity_days_max": 21.0,
        },
        "short_atm_eligible_counts": short_atm_eligible_counts,
        "history_valid_steps_counts": {
            str(key): int(value)
            for key, value in manifest["history_valid_steps"].value_counts().sort_index().items()
        },
        "with_news": int((manifest["news_count_total"] > 0).sum()),
        "without_news": int((manifest["news_count_total"] == 0).sum()),
        "sequence_dataset_sha256": sha256_file(fold_dir / "sequence_dataset.npz"),
        "split_manifest_sha256": sha256_file(fold_dir / "split_manifest.csv"),
        "raw_support_sha256": sha256_file(support_path),
        "text_transform_sha256": sha256_file(fold_dir / "text_transform.npz"),
    }
    (fold_dir / "dataset_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def build_sequence_datasets(
    *,
    database_path: Path,
    news_workbook: Path,
    session_calendar_path: Path,
    output_root: Path,
    permutation_seed: int = 20260722,
) -> dict[str, Any]:
    """Build candidate audit and all four fold-specific NPZ bundles."""

    calendar = TreasuryGlobexSessionCalendar.from_csv(session_calendar_path)
    anchors, candidates, index_audit = _read_market_index(database_path, calendar)
    news, catalog, embedding_dim, article_conflicts = _load_news_catalog(
        news_workbook
    )
    windows, news_audit = _build_news_windows(
        candidates,
        news,
        catalog,
        embedding_dim,
    )
    candidate_frame = pd.DataFrame(
        [
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
                "has_news": int(windows[candidate.pair_id].has_news),
                "news_count_total": len(windows[candidate.pair_id].article_ids),
            }
            for candidate in candidates
        ]
    )
    candidate_frame.to_csv(output_root / "sequence_candidates.csv", index=False)
    news_audit.to_csv(output_root / "news_bucket_audit.csv", index=False)
    article_conflicts.to_csv(
        output_root / "article_id_embedding_conflicts.csv",
        index=False,
    )
    database_hash = sha256_file(database_path)
    fold_validation = {
        fold: _materialize_fold(
            fold=fold,
            anchors=anchors,
            candidates=candidates,
            windows=windows,
            catalog=catalog,
            news_workbook=news_workbook,
            database_sha256=database_hash,
            output_root=output_root,
            permutation_seed=permutation_seed,
        )
        for fold in FOLDS
    }
    summary = {
        "status": "ok",
        "surface_source": "raw",
        "history_steps": HISTORY_STEPS,
        "history_interval_minutes": HISTORY_INTERVAL_MINUTES,
        "history_min_valid_steps": HISTORY_MIN_VALID_STEPS,
        "forecast_horizon_minutes": HISTORY_INTERVAL_MINUTES,
        "candidate_count": len(candidates),
        "with_news": int(candidate_frame["has_news"].sum()),
        "without_news": int((candidate_frame["has_news"] == 0).sum()),
        "lp_embedding_dim": int(embedding_dim),
        "article_id_embedding_conflict_count": int(len(article_conflicts)),
        "sequence_text_output_dim": TEXT_OUTPUT_DIM,
        "database_path": str(database_path),
        "database_sha256": database_hash,
        "news_workbook_path": str(news_workbook),
        "news_workbook_sha256": sha256_file(news_workbook),
        "session_calendar_path": str(session_calendar_path),
        "session_calendar_sha256": sha256_file(session_calendar_path),
        "index_audit": index_audit,
        "folds": fold_validation,
    }
    (output_root / "sequence_dataset_validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
