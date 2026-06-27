"""Build n-gram frequency BoW features for RQ2 traditional text baselines."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")
_REFERENCE_METHOD = "Manela_Moreira_2017_style_ngram_frequency"


@dataclass(frozen=True)
class BowFeatureResult:
    """Feature frame plus fitted vocabulary needed for reproducibility."""

    frame: pd.DataFrame
    vocabulary: list[str]
    manifest: dict[str, Any]


def _serialize_vector(values: np.ndarray) -> str:
    return json.dumps([float(value) for value in values.tolist()], ensure_ascii=True)


def _ensure_news_row_id(frame: pd.DataFrame) -> pd.Series:
    if "news_row_id" in frame.columns:
        return pd.to_numeric(frame["news_row_id"], errors="coerce").fillna(0).astype(int)
    return pd.Series(range(1, len(frame) + 1), index=frame.index, dtype="int64")


def _metadata_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _text_series(frame: pd.DataFrame, text_column: str) -> pd.Series:
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column '{text_column}'. Available columns: {list(frame.columns)}")
    return frame[text_column].fillna("").astype(str)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(str(text))]


def _normalize_ngram_range(ngram_range: tuple[int, int]) -> tuple[int, int]:
    ngram_min, ngram_max = tuple(int(value) for value in ngram_range)
    if ngram_min <= 0 or ngram_max <= 0:
        raise ValueError(f"ngram_range values must be positive, got: {ngram_range}")
    if ngram_min > ngram_max:
        raise ValueError(f"ngram_range minimum cannot exceed maximum, got: {ngram_range}")
    return ngram_min, ngram_max


def _ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    ngram_min, ngram_max = _normalize_ngram_range(ngram_range)
    output: list[str] = []
    for ngram_size in range(ngram_min, ngram_max + 1):
        if len(tokens) < ngram_size:
            continue
        output.extend(" ".join(tokens[idx : idx + ngram_size]) for idx in range(0, len(tokens) - ngram_size + 1))
    return output


def _build_vocabulary(documents: list[list[str]], *, target_dim: int) -> list[str]:
    corpus_counts: Counter[str] = Counter()
    for terms in documents:
        corpus_counts.update(terms)
    return [
        term
        for term, _count in sorted(corpus_counts.items(), key=lambda item: (-item[1], item[0]))[: int(target_dim)]
    ]


def _frequency_features(documents: list[list[str]], vocabulary: list[str], *, target_dim: int) -> np.ndarray:
    features = np.zeros((len(documents), int(target_dim)), dtype=np.float32)
    if not vocabulary:
        return features
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    for row_idx, terms in enumerate(documents):
        row_counts = Counter(terms)
        for term, count in row_counts.items():
            column_idx = term_to_idx.get(term)
            if column_idx is not None:
                features[row_idx, column_idx] = math.log1p(float(count))
    return features


def _feature_frame(news_df: pd.DataFrame, *, features: np.ndarray, target_dim: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "news_row_id": _ensure_news_row_id(news_df),
            "article_id": _metadata_column(news_df, "ArticleID", "article_id"),
            "source_file": _metadata_column(news_df, "SourceFile", "source_file"),
            "bow_embedding": [_serialize_vector(row) for row in features],
            "bow_dim": [int(target_dim)] * len(news_df),
        }
    )


def fit_bow_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 42,
) -> BowFeatureResult:
    """Fit a deterministic n-gram frequency BoW representation.

    ``max_features`` and ``random_state`` are retained only for CLI/API
    compatibility with earlier generated commands.
    """

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    ngram_range = _normalize_ngram_range(ngram_range)

    texts = _text_series(news_df, text_column)
    documents = [_ngrams(_tokens(text), ngram_range) for text in texts.tolist()]
    vocabulary = _build_vocabulary(documents, target_dim=target_dim)
    features = _frequency_features(documents, vocabulary, target_dim=target_dim)
    frame = _feature_frame(news_df, features=features, target_dim=target_dim)

    fitted = bool(vocabulary)
    manifest = {
        "text_column": text_column,
        "target_dim": target_dim,
        "max_features_deprecated_ignored": int(max_features),
        "random_state_deprecated_ignored": int(random_state),
        "ngram_range": list(ngram_range),
        "row_count": int(len(news_df)),
        "fitted": fitted,
        "reason": "" if fitted else "all_text_empty",
        "backend": "python_counter",
        "representation": "ngram_frequency",
        "weighting": "log1p_count",
        "reference_method": _REFERENCE_METHOD,
        "vocabulary_size": int(len(vocabulary)),
    }
    return BowFeatureResult(frame=frame, vocabulary=vocabulary, manifest=manifest)


def build_bow_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Return only the BoW feature frame for callers that do not need metadata."""

    return fit_bow_features(
        news_df,
        text_column=text_column,
        target_dim=target_dim,
        max_features=max_features,
        ngram_range=ngram_range,
    ).frame
