"""Leakage-controlled pair-level BoW and ChatGPT sentiment feature construction."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from bow import fit_bow_vocabulary, transform_bow_counts
from film_wgan.text_transform import fit_text_transform, sha256_file


SENTIMENT_DIMENSIONS = (
    "macroeconomic_uncertainty",
    "institutional_action",
    "risk_off_intensity",
)
SENTIMENT_MODEL_ID = "gpt-5.4-mini"
SENTIMENT_PROMPT_VERSION = "sun2026_zero_shot_chatgpt_v1"
PAIR_FEATURE_SCHEMA_VERSION = "rq2_pair_feature_dual_universe_v1"


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    rendered = str(value).strip()
    if not rendered:
        return np.asarray([], dtype=np.float32)
    try:
        parsed = json.loads(rendered)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(rendered)
    return np.asarray(parsed, dtype=np.float32)


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return [str(item) for item in parsed]


def _sample_news_row_id(sample_id: str) -> int:
    rendered = str(sample_id)
    if not rendered.startswith("news_") or not rendered[5:].isdigit():
        raise ValueError(f"Expected sample_id=news_<row-id>, got {sample_id!r}.")
    value = int(rendered[5:])
    if value <= 0:
        raise ValueError(f"news row id must be positive, got {value}.")
    return value


def _article_key(row: pd.Series, news_row_id: int) -> str:
    value = row.get("ArticleID", "")
    article_id = "" if pd.isna(value) else str(value).strip()
    return article_id or f"news_row_{news_row_id}"


def _clean_string(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _embedding_sha(row: pd.Series) -> str:
    embedding = _parse_vector(row.get("LP_embedding", ""))
    if embedding.size == 0:
        return hashlib.sha256(
            _clean_string(row.get("LP", "")).encode("utf-8")
        ).hexdigest()
    return hashlib.sha256(embedding.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class PairArticle:
    news_row_id: int
    sample_id: str
    article_id: str
    source_file: str
    text: str
    embedding_sha256: str
    sentiment: np.ndarray
    sentiment_parse_status: str


@dataclass(frozen=True)
class PairFeatureArtifacts:
    bow_feature_path: Path
    bow_vocabulary_path: Path
    bow_transform_path: Path
    sentiment_feature_path: Path
    sentiment_transform_path: Path
    audit_path: Path


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def _strict_text_sample_ids(
    sample_ids: Sequence[str],
    news: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    usable: list[str] = []
    excluded: list[str] = []
    reasons: list[str] = []
    for sample_id in sample_ids:
        news_row_id = _sample_news_row_id(sample_id)
        if news_row_id > len(news):
            raise ValueError(
                f"{sample_id} references row {news_row_id}, but news workbook "
                f"has {len(news)} rows."
            )
        source = news.iloc[news_row_id - 1]
        text = _clean_string(source.get("LP", ""))
        embedding = _parse_vector(source.get("LP_embedding", ""))
        if not text:
            reason = "empty_lp"
        elif embedding.size <= 0:
            reason = "nonempty_lp_missing_embedding"
        elif not np.all(np.isfinite(embedding)):
            reason = "nonfinite_lp_embedding"
        else:
            usable.append(str(sample_id))
            continue
        excluded.append(str(sample_id))
        reasons.append(reason)
    return usable, excluded, reasons


def build_pair_feature_coverage_lineage(
    *,
    fold: str,
    split_manifest_path: str | Path,
    news_workbook_path: str | Path,
    fit_lineage_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Freeze the strict-text-valid pair universe before support filtering.

    External text features are resolved before raw-surface support is applied by
    the loader.  Consequently this coverage lineage intentionally contains
    support-zero pairs that are absent from ``fit_lineage_path``.
    """

    split_manifest = pd.read_csv(split_manifest_path).sort_values(
        "global_index", kind="stable"
    )
    required = {
        "global_index",
        "sample_id",
        "surface_pair_id",
        "split",
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
    }
    missing = sorted(required - set(split_manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing coverage columns: {missing}")
    news = _read_excel_cached(str(Path(news_workbook_path).resolve()))

    rows: list[dict[str, Any]] = []
    included = split_manifest[
        split_manifest["split"].astype(str).isin({"train", "val", "test"})
    ]
    for pair_id, members in included.groupby(
        "surface_pair_id", sort=False, dropna=False
    ):
        splits = list(dict.fromkeys(members["split"].astype(str)))
        if len(splits) != 1:
            raise ValueError(
                f"Pair {pair_id} appears in multiple splits: {sorted(splits)}"
            )
        sample_ids = members["sample_id"].astype(str).tolist()
        usable, excluded, reasons = _strict_text_sample_ids(sample_ids, news)
        if not usable:
            continue
        current_times = list(
            dict.fromkeys(members["current_snapshot_time_utc"].astype(str))
        )
        target_times = list(
            dict.fromkeys(members["target_snapshot_time_utc"].astype(str))
        )
        if len(current_times) != 1 or len(target_times) != 1:
            raise ValueError(f"Pair {pair_id} has inconsistent snapshot timestamps.")
        rows.append(
            {
                "fold": str(fold),
                "split": splits[0],
                "surface_pair_id": str(pair_id),
                "sample_id": f"pair_{pair_id}",
                "current_snapshot_time_utc": current_times[0],
                "target_snapshot_time_utc": target_times[0],
                "source_sample_count_before_strict_text": int(len(sample_ids)),
                "source_sample_count": int(len(usable)),
                "excluded_source_sample_count": int(len(excluded)),
                "source_sample_ids": _serialize_list(usable),
                "excluded_source_sample_ids": _serialize_list(excluded),
                "excluded_text_lineage_reasons": _serialize_list(reasons),
                "coverage_stage": "strict_text_valid_pre_surface_support",
            }
        )
    coverage = pd.DataFrame(rows)
    if coverage.empty:
        raise ValueError(f"Fold {fold} has no strict-text-valid coverage pairs.")
    if coverage["surface_pair_id"].astype(str).duplicated().any():
        raise ValueError("Coverage lineage must contain one row per surface pair.")

    fit = pd.read_csv(fit_lineage_path)
    _validate_coverage_and_fit_lineages(coverage, fit)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    coverage.to_csv(temporary, index=False)
    try:
        if target.exists():
            if sha256_file(target) != sha256_file(temporary):
                raise ValueError(
                    "Existing pair feature coverage lineage differs from the "
                    f"deterministic rebuild: {target}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _validate_coverage_and_fit_lineages(
    coverage: pd.DataFrame,
    fit: pd.DataFrame,
) -> None:
    required = {"surface_pair_id", "split", "source_sample_ids"}
    for name, frame in (("coverage", coverage), ("fit", fit)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name.title()} lineage is missing columns: {missing}")
        if frame["surface_pair_id"].astype(str).duplicated().any():
            raise ValueError(f"{name.title()} lineage must contain unique pair IDs.")

    coverage_by_pair = coverage.assign(
        surface_pair_id=coverage["surface_pair_id"].astype(str)
    ).set_index("surface_pair_id")
    fit_pair_ids = fit["surface_pair_id"].astype(str).tolist()
    missing_fit = [
        pair_id for pair_id in fit_pair_ids if pair_id not in coverage_by_pair.index
    ]
    if missing_fit:
        raise ValueError(
            "Post-support fit lineage is not a subset of feature coverage lineage: "
            f"{missing_fit[:10]}"
        )
    for row in fit.itertuples(index=False):
        pair_id = str(row.surface_pair_id)
        coverage_row = coverage_by_pair.loc[pair_id]
        if str(coverage_row["split"]) != str(row.split):
            raise ValueError(
                f"Pair {pair_id} split differs across coverage and fit lineages."
            )
        coverage_samples = _parse_json_list(coverage_row["source_sample_ids"])
        fit_samples = _parse_json_list(row.source_sample_ids)
        if coverage_samples != fit_samples:
            raise ValueError(
                f"Pair {pair_id} source_sample_ids differ across coverage and fit lineages."
            )


@lru_cache(maxsize=8)
def _read_excel_cached(path_value: str) -> pd.DataFrame:
    return pd.read_excel(path_value)


@lru_cache(maxsize=8)
def _load_sentiment_lookup_cached(path_value: str) -> dict[int, dict[str, Any]]:
    path = Path(path_value)
    frame = _read_excel_cached(str(path))
    required = {
        "news_row_id",
        "article_id",
        "source_file",
        "sentiment_embedding",
        "sentiment_model_id",
        "sentiment_prompt_version",
        "sentiment_parse_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Sentiment artifact {path} is missing columns: {missing}")
    if frame["news_row_id"].duplicated().any():
        raise ValueError(f"Sentiment artifact has duplicate news_row_id values: {path}")
    model_ids = {
        _clean_string(value)
        for value in frame["sentiment_model_id"]
        if _clean_string(value)
    }
    prompt_versions = {
        _clean_string(value)
        for value in frame["sentiment_prompt_version"]
        if _clean_string(value)
    }
    if model_ids != {SENTIMENT_MODEL_ID}:
        raise ValueError(
            f"Unexpected sentiment model IDs in {path}: {sorted(model_ids)}"
        )
    if prompt_versions != {SENTIMENT_PROMPT_VERSION}:
        raise ValueError(
            f"Unexpected sentiment prompt versions in {path}: {sorted(prompt_versions)}"
        )
    lookup: dict[int, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        vector = _parse_vector(row.sentiment_embedding)
        if vector.size < len(SENTIMENT_DIMENSIONS):
            raise ValueError(
                f"Sentiment row {row.news_row_id} has {vector.size} values; "
                f"expected at least {len(SENTIMENT_DIMENSIONS)}."
            )
        lookup[int(row.news_row_id)] = {
            "vector": vector[: len(SENTIMENT_DIMENSIONS)].astype(np.float32),
            "article_id": _clean_string(row.article_id),
            "source_file": _clean_string(row.source_file),
            "model_id": _clean_string(row.sentiment_model_id),
            "prompt_version": _clean_string(row.sentiment_prompt_version),
            "parse_status": _clean_string(row.sentiment_parse_status),
        }
    return lookup


def load_sentiment_lookup(path: str | Path) -> dict[int, dict[str, Any]]:
    return _load_sentiment_lookup_cached(str(Path(path).resolve()))


def build_pair_articles(
    lineage: pd.DataFrame,
    news: pd.DataFrame,
    sentiment_lookup: dict[int, dict[str, Any]],
) -> dict[str, list[PairArticle]]:
    """Resolve and deduplicate articles exactly once for every surface pair."""

    required = {"surface_pair_id", "split", "source_sample_ids"}
    missing = sorted(required - set(lineage.columns))
    if missing:
        raise ValueError(f"Pair lineage is missing columns: {missing}")
    if lineage["surface_pair_id"].astype(str).duplicated().any():
        raise ValueError("Pair lineage must contain one row per surface_pair_id.")

    output: dict[str, list[PairArticle]] = {}
    for record in lineage.itertuples(index=False):
        pair_id = str(record.surface_pair_id)
        sample_ids = _parse_json_list(record.source_sample_ids)
        by_article: list[PairArticle] = []
        seen_articles: set[str] = set()
        for sample_id in sample_ids:
            news_row_id = _sample_news_row_id(sample_id)
            if news_row_id > len(news):
                raise ValueError(
                    f"{sample_id} references row {news_row_id}, but news workbook has {len(news)} rows."
                )
            source = news.iloc[news_row_id - 1]
            article_id = _article_key(source, news_row_id)
            if article_id in seen_articles:
                continue
            seen_articles.add(article_id)
            sentiment = sentiment_lookup.get(news_row_id)
            if sentiment is None:
                raise ValueError(f"No ChatGPT sentiment score for {sample_id}.")
            source_file = _clean_string(source.get("SourceFile", ""))
            article_text = _clean_string(source.get("LP", ""))
            if not article_text:
                raise ValueError(
                    f"Strict RQ2 lineage cannot use empty LP text for {sample_id}."
                )
            source_embedding = _parse_vector(source.get("LP_embedding", ""))
            if source_embedding.size <= 0 or not np.all(np.isfinite(source_embedding)):
                raise ValueError(
                    f"Strict RQ2 lineage requires a finite LP embedding for {sample_id}."
                )
            if str(sentiment["article_id"]).strip() not in {"", article_id}:
                raise ValueError(
                    f"Sentiment ArticleID lineage mismatch for {sample_id}."
                )
            if str(sentiment["source_file"]).strip() not in {"", source_file}:
                raise ValueError(
                    f"Sentiment SourceFile lineage mismatch for {sample_id}."
                )
            if str(sentiment["parse_status"]).strip().lower() not in {
                "cache_json",
                "json",
            }:
                raise ValueError(
                    f"Sentiment score for {sample_id} is not reproducible/usable: "
                    f"parse_status={sentiment['parse_status']!r}."
                )
            by_article.append(
                PairArticle(
                    news_row_id=news_row_id,
                    sample_id=sample_id,
                    article_id=article_id,
                    source_file=source_file,
                    text=article_text,
                    embedding_sha256=_embedding_sha(source),
                    sentiment=np.asarray(sentiment["vector"], dtype=np.float32),
                    sentiment_parse_status=str(sentiment["parse_status"]),
                )
            )

        unique: list[PairArticle] = []
        seen_embeddings: set[str] = set()
        for article in by_article:
            if article.embedding_sha256 in seen_embeddings:
                continue
            seen_embeddings.add(article.embedding_sha256)
            unique.append(article)
        if not unique:
            raise ValueError(
                f"Pair {pair_id} has no unique articles after deduplication."
            )
        output[pair_id] = unique
    return output


def _serialize_vector(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values.tolist()], ensure_ascii=True)


def _serialize_list(values: Iterable[str]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=True)


def _l2_or_zero(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array if norm <= 1e-12 else (array / norm).astype(np.float32)


def _pair_feature_row(
    *,
    pair_id: str,
    split: str,
    articles: Sequence[PairArticle],
    source_sample_ids: Sequence[str],
    vector: np.ndarray,
    representation: str,
    pooling_mode: str,
) -> dict[str, Any]:
    feature_hash = hashlib.sha256(
        np.asarray(vector, dtype=np.float32).tobytes(order="C")
    ).hexdigest()
    return {
        "surface_pair_id": pair_id,
        "split": split,
        "text_embedding": _serialize_vector(np.asarray(vector, dtype=np.float32)),
        "representation": representation,
        "pooling_mode": pooling_mode,
        "source_sample_ids": _serialize_list(source_sample_ids),
        "article_ids": _serialize_list(article.article_id for article in articles),
        "source_files": _serialize_list(article.source_file for article in articles),
        "news_row_ids": _serialize_list(
            str(article.news_row_id) for article in articles
        ),
        "unique_article_count": int(len(articles)),
        "feature_sha256": feature_hash,
    }


def _build_fold_pair_features_in_dir(
    *,
    fold: str,
    feature_coverage_lineage_path: str | Path,
    fit_lineage_path: str | Path,
    news_workbook_path: str | Path,
    sentiment_feature_path: str | Path,
    output_dir: str | Path,
    input_workbook_path: str | Path,
    vocabulary_size: int = 1024,
    output_dim: int = 128,
) -> PairFeatureArtifacts:
    """Build fold-frozen representation artifacts and their train-only transforms."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    coverage_lineage = pd.read_csv(feature_coverage_lineage_path)
    fit_lineage = pd.read_csv(fit_lineage_path)
    _validate_coverage_and_fit_lineages(coverage_lineage, fit_lineage)
    news = _read_excel_cached(str(Path(news_workbook_path).resolve()))
    sentiment_lookup = load_sentiment_lookup(sentiment_feature_path)
    pair_articles = build_pair_articles(coverage_lineage, news, sentiment_lookup)
    split_by_pair = {
        str(row.surface_pair_id): str(row.split)
        for row in coverage_lineage.itertuples(index=False)
    }
    fit_pair_ids = fit_lineage["surface_pair_id"].astype(str).tolist()
    fit_pair_id_set = set(fit_pair_ids)

    train_article_by_id: dict[str, PairArticle] = {}
    for pair_id, articles in pair_articles.items():
        if pair_id not in fit_pair_id_set or split_by_pair[pair_id] != "train":
            continue
        for article in articles:
            train_article_by_id.setdefault(article.article_id, article)
    train_articles = list(train_article_by_id.values())
    vocabulary = fit_bow_vocabulary(
        [article.text for article in train_articles],
        target_dim=int(vocabulary_size),
        ngram_range=(1, 2),
    )
    if len(vocabulary) != int(vocabulary_size):
        raise ValueError(
            f"Fold {fold} produced {len(vocabulary)} BoW terms; expected {vocabulary_size}."
        )

    bow_rows: list[dict[str, Any]] = []
    sentiment_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for pair_id in coverage_lineage["surface_pair_id"].astype(str):
        articles = pair_articles[pair_id]
        source_sample_ids = _parse_json_list(
            coverage_lineage.loc[
                coverage_lineage["surface_pair_id"].astype(str) == pair_id,
                "source_sample_ids",
            ].iloc[0]
        )
        counts = transform_bow_counts(
            [article.text for article in articles],
            vocabulary,
            ngram_range=(1, 2),
        )
        bow_vector = _l2_or_zero(np.log1p(counts.sum(axis=0)).astype(np.float32))
        sentiment_vector = np.mean(
            np.stack([article.sentiment for article in articles], axis=0),
            axis=0,
        ).astype(np.float32)
        split = split_by_pair[pair_id]
        bow_rows.append(
            _pair_feature_row(
                pair_id=pair_id,
                split=split,
                articles=articles,
                source_sample_ids=source_sample_ids,
                vector=bow_vector,
                representation="bow",
                pooling_mode="bow_log_count_l2",
            )
        )
        sentiment_rows.append(
            _pair_feature_row(
                pair_id=pair_id,
                split=split,
                articles=articles,
                source_sample_ids=source_sample_ids,
                vector=sentiment_vector,
                representation="llm_sentiment",
                pooling_mode="mean_scores",
            )
        )
        audit_rows.append(
            {
                "fold": fold,
                "surface_pair_id": pair_id,
                "split": split,
                "source_sample_count": len(source_sample_ids),
                "unique_article_count": len(articles),
                "empty_text_count": int(
                    sum(not article.text.strip() for article in articles)
                ),
                "sentiment_non_ok_count": int(
                    sum(
                        article.sentiment_parse_status not in {"ok", "cache_json"}
                        for article in articles
                    )
                ),
                "bow_nonzero_terms": int(np.count_nonzero(bow_vector)),
                "sentiment_all_zero": bool(np.allclose(sentiment_vector, 0.0)),
            }
        )

    bow_path = output / "bow_pair_features.csv"
    sentiment_path = output / "llm_sentiment_pair_features.csv"
    pd.DataFrame(bow_rows).to_csv(bow_path, index=False)
    pd.DataFrame(sentiment_rows).to_csv(sentiment_path, index=False)
    audit_path = output / "pair_feature_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    vocabulary_path = output / "bow_vocabulary.json"
    vocabulary_path.write_text(
        json.dumps(
            {
                "fold": fold,
                "fit_scope": "fold_train_unique_articles_only",
                "vocabulary_size": len(vocabulary),
                "ngram_range": [1, 2],
                "terms": vocabulary,
                "train_article_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(train_article_by_id)).encode("utf-8")
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    train_pair_ids = [
        pair_id
        for pair_id, split in zip(
            fit_lineage["surface_pair_id"].astype(str),
            fit_lineage["split"].astype(str),
        )
        if split == "train"
    ]
    if not train_pair_ids:
        raise ValueError(f"Fold {fold} has no post-support training pairs.")
    bow_frame = pd.DataFrame(bow_rows).set_index("surface_pair_id")
    sentiment_frame = pd.DataFrame(sentiment_rows).set_index("surface_pair_id")
    bow_matrix = np.stack(
        [
            _parse_vector(bow_frame.loc[pair_id, "text_embedding"])
            for pair_id in train_pair_ids
        ],
        axis=0,
    )
    sentiment_matrix = np.stack(
        [
            _parse_vector(sentiment_frame.loc[pair_id, "text_embedding"])
            for pair_id in train_pair_ids
        ],
        axis=0,
    )
    bow_transform = fit_text_transform(
        bow_matrix,
        mode="pca",
        components=int(output_dim),
        whiten=False,
        train_pair_ids=train_pair_ids,
        input_workbook_path=input_workbook_path,
        output_dim=int(output_dim),
        input_feature_path=bow_path,
    )
    bow_transform_path = output / "bow_text_transform.npz"
    bow_transform.save(bow_transform_path)
    sentiment_transform = fit_text_transform(
        sentiment_matrix,
        mode="zscore_pad",
        components=int(output_dim),
        whiten=False,
        train_pair_ids=train_pair_ids,
        input_workbook_path=input_workbook_path,
        output_dim=int(output_dim),
        input_feature_path=sentiment_path,
    )
    sentiment_transform_path = output / "llm_sentiment_text_transform.npz"
    sentiment_transform.save(sentiment_transform_path)

    for metadata_path in (
        bow_transform_path.with_name(f"{bow_transform_path.stem}_metadata.json"),
        sentiment_transform_path.with_name(
            f"{sentiment_transform_path.stem}_metadata.json"
        ),
    ):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload = {
            key: (
                str(value).replace(str(output), "__FINAL_OUTPUT_DIR__")
                if isinstance(value, str)
                else value
            )
            for key, value in payload.items()
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": PAIR_FEATURE_SCHEMA_VERSION,
        "fold": fold,
        "feature_coverage_lineage_path": str(Path(feature_coverage_lineage_path)),
        "feature_coverage_lineage_sha256": sha256_file(feature_coverage_lineage_path),
        "fit_lineage_path": str(Path(fit_lineage_path)),
        "fit_lineage_sha256": sha256_file(fit_lineage_path),
        "news_workbook_path": str(Path(news_workbook_path)),
        "news_workbook_sha256": sha256_file(news_workbook_path),
        "sentiment_feature_path": str(Path(sentiment_feature_path)),
        "sentiment_source_sha256": sha256_file(sentiment_feature_path),
        "feature_coverage_pair_count": int(len(coverage_lineage)),
        "fit_pair_count": int(len(fit_lineage)),
        "train_pair_count": int(len(train_pair_ids)),
        "feature_coverage_ordered_pair_ids_sha256": _ordered_ids_sha256(
            coverage_lineage["surface_pair_id"].astype(str).tolist()
        ),
        "fit_ordered_pair_ids_sha256": _ordered_ids_sha256(fit_pair_ids),
        "fit_train_ordered_pair_ids_sha256": _ordered_ids_sha256(train_pair_ids),
        "vocabulary_fit_article_count": int(len(train_articles)),
        "vocabulary_size": int(len(vocabulary)),
        "bow_pair_formula": "L2(log1p(sum unique-article ngram counts))",
        "sentiment_pair_formula": "mean unique-article scores by dimension",
        "sentiment_dimensions": list(SENTIMENT_DIMENSIONS),
        "sentiment_model_id": SENTIMENT_MODEL_ID,
        "sentiment_prompt_version": SENTIMENT_PROMPT_VERSION,
        "output_dim": int(output_dim),
        "bow_feature_sha256": sha256_file(bow_path),
        "bow_vocabulary_sha256": sha256_file(vocabulary_path),
        "sentiment_feature_sha256": sha256_file(sentiment_path),
        "bow_transform_sha256": sha256_file(bow_transform_path),
        "bow_transform_metadata_sha256": sha256_file(
            bow_transform_path.with_name(f"{bow_transform_path.stem}_metadata.json")
        ),
        "sentiment_transform_sha256": sha256_file(sentiment_transform_path),
        "sentiment_transform_metadata_sha256": sha256_file(
            sentiment_transform_path.with_name(
                f"{sentiment_transform_path.stem}_metadata.json"
            )
        ),
        "audit_sha256": sha256_file(audit_path),
    }
    (output / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PairFeatureArtifacts(
        bow_feature_path=bow_path,
        bow_vocabulary_path=vocabulary_path,
        bow_transform_path=bow_transform_path,
        sentiment_feature_path=sentiment_path,
        sentiment_transform_path=sentiment_transform_path,
        audit_path=audit_path,
    )


def build_fold_pair_features(
    *,
    fold: str,
    feature_coverage_lineage_path: str | Path,
    fit_lineage_path: str | Path,
    news_workbook_path: str | Path,
    sentiment_feature_path: str | Path,
    output_dir: str | Path,
    input_workbook_path: str | Path,
    vocabulary_size: int = 1024,
    output_dim: int = 128,
) -> PairFeatureArtifacts:
    """Atomically publish dual-universe pair features for one fold."""

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty pair feature directory: {output}"
            )
        output.rmdir()
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        _build_fold_pair_features_in_dir(
            fold=fold,
            feature_coverage_lineage_path=feature_coverage_lineage_path,
            fit_lineage_path=fit_lineage_path,
            news_workbook_path=news_workbook_path,
            sentiment_feature_path=sentiment_feature_path,
            output_dir=staging,
            input_workbook_path=input_workbook_path,
            vocabulary_size=vocabulary_size,
            output_dim=output_dim,
        )
        for metadata_path in staging.glob("*_text_transform_metadata.json"):
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = {
                key: (
                    str(value).replace("__FINAL_OUTPUT_DIR__", str(output))
                    if isinstance(value, str)
                    else value
                )
                for key, value in payload.items()
            }
            metadata_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifest_path = staging / "feature_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bow_transform_metadata_sha256"] = sha256_file(
            staging / "bow_text_transform_metadata.json"
        )
        manifest["sentiment_transform_metadata_sha256"] = sha256_file(
            staging / "llm_sentiment_text_transform_metadata.json"
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PairFeatureArtifacts(
        bow_feature_path=output / "bow_pair_features.csv",
        bow_vocabulary_path=output / "bow_vocabulary.json",
        bow_transform_path=output / "bow_text_transform.npz",
        sentiment_feature_path=output / "llm_sentiment_pair_features.csv",
        sentiment_transform_path=output / "llm_sentiment_text_transform.npz",
        audit_path=output / "pair_feature_audit.csv",
    )
