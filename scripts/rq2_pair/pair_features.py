"""Leakage-controlled pair-level BoW and ChatGPT sentiment feature construction."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from bow import fit_bow_vocabulary, transform_bow_counts
from film_wgan.text_transform import fit_text_transform, l2_normalize_rows, sha256_file


SENTIMENT_DIMENSIONS = (
    "macroeconomic_uncertainty",
    "institutional_action",
    "risk_off_intensity",
)
SENTIMENT_MODEL_ID = "gpt-5.4-mini"
SENTIMENT_PROMPT_VERSION = "sun2026_zero_shot_chatgpt_v1"


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
        return hashlib.sha256(_clean_string(row.get("LP", "")).encode("utf-8")).hexdigest()
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
            f"Unexpected sentiment prompt versions in {path}: "
            f"{sorted(prompt_versions)}"
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
            if str(sentiment["article_id"]).strip() not in {"", article_id}:
                raise ValueError(f"Sentiment ArticleID lineage mismatch for {sample_id}.")
            if str(sentiment["source_file"]).strip() not in {"", source_file}:
                raise ValueError(f"Sentiment SourceFile lineage mismatch for {sample_id}.")
            by_article.append(
                PairArticle(
                    news_row_id=news_row_id,
                    sample_id=sample_id,
                    article_id=article_id,
                    source_file=source_file,
                    text=_clean_string(source.get("LP", "")),
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
            raise ValueError(f"Pair {pair_id} has no unique articles after deduplication.")
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
        "news_row_ids": _serialize_list(str(article.news_row_id) for article in articles),
        "unique_article_count": int(len(articles)),
        "feature_sha256": feature_hash,
    }


def build_fold_pair_features(
    *,
    fold: str,
    lineage_path: str | Path,
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
    lineage = pd.read_csv(lineage_path)
    news = _read_excel_cached(str(Path(news_workbook_path).resolve()))
    sentiment_lookup = load_sentiment_lookup(sentiment_feature_path)
    pair_articles = build_pair_articles(lineage, news, sentiment_lookup)
    split_by_pair = {
        str(row.surface_pair_id): str(row.split)
        for row in lineage.itertuples(index=False)
    }

    train_article_by_id: dict[str, PairArticle] = {}
    for pair_id, articles in pair_articles.items():
        if split_by_pair[pair_id] != "train":
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
    for pair_id in lineage["surface_pair_id"].astype(str):
        articles = pair_articles[pair_id]
        source_sample_ids = _parse_json_list(
            lineage.loc[
                lineage["surface_pair_id"].astype(str) == pair_id,
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
                "empty_text_count": int(sum(not article.text.strip() for article in articles)),
                "sentiment_non_ok_count": int(
                    sum(article.sentiment_parse_status not in {"ok", "cache_json"} for article in articles)
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
        for pair_id in lineage["surface_pair_id"].astype(str)
        if split_by_pair[pair_id] == "train"
    ]
    bow_frame = pd.DataFrame(bow_rows).set_index("surface_pair_id")
    sentiment_frame = pd.DataFrame(sentiment_rows).set_index("surface_pair_id")
    bow_matrix = np.stack(
        [_parse_vector(bow_frame.loc[pair_id, "text_embedding"]) for pair_id in train_pair_ids],
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

    manifest = {
        "fold": fold,
        "lineage_path": str(Path(lineage_path)),
        "lineage_sha256": sha256_file(lineage_path),
        "news_workbook_path": str(Path(news_workbook_path)),
        "news_workbook_sha256": sha256_file(news_workbook_path),
        "sentiment_feature_path": str(Path(sentiment_feature_path)),
        "sentiment_feature_sha256": sha256_file(sentiment_feature_path),
        "pair_count": int(len(lineage)),
        "train_pair_count": int(len(train_pair_ids)),
        "vocabulary_fit_article_count": int(len(train_articles)),
        "vocabulary_size": int(len(vocabulary)),
        "bow_pair_formula": "L2(log1p(sum unique-article ngram counts))",
        "sentiment_pair_formula": "mean unique-article scores by dimension",
        "sentiment_dimensions": list(SENTIMENT_DIMENSIONS),
        "sentiment_model_id": SENTIMENT_MODEL_ID,
        "sentiment_prompt_version": SENTIMENT_PROMPT_VERSION,
        "output_dim": int(output_dim),
        "bow_feature_sha256": sha256_file(bow_path),
        "sentiment_feature_sha256": sha256_file(sentiment_path),
        "bow_transform_sha256": sha256_file(bow_transform_path),
        "sentiment_transform_sha256": sha256_file(sentiment_transform_path),
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
