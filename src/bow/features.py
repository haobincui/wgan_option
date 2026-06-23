"""Build BoW/TF-IDF features for RQ2 traditional text baselines."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass(frozen=True)
class BowFeatureResult:
    """Feature frame plus fitted sklearn objects needed for reproducibility."""

    frame: pd.DataFrame
    vectorizer: Any | None
    svd: Any | None
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


def _zero_feature_frame(news_df: pd.DataFrame, *, target_dim: int) -> pd.DataFrame:
    vector = _serialize_vector(np.zeros(int(target_dim), dtype=np.float32))
    return pd.DataFrame(
        {
            "news_row_id": _ensure_news_row_id(news_df),
            "article_id": _metadata_column(news_df, "ArticleID", "article_id"),
            "source_file": _metadata_column(news_df, "SourceFile", "source_file"),
            "bow_embedding": [vector] * len(news_df),
            "bow_dim": [int(target_dim)] * len(news_df),
        }
    )


def _dense_aligned_features(matrix: Any, *, target_dim: int, random_state: int) -> tuple[np.ndarray, Any | None]:
    from sklearn.decomposition import TruncatedSVD

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    n_features = int(matrix.shape[1])
    if n_features <= 0:
        return np.zeros((matrix.shape[0], target_dim), dtype=np.float32), None

    n_components = min(target_dim, n_features)
    if n_components == n_features:
        dense = matrix.toarray().astype(np.float32)
        svd = None
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=int(random_state))
        dense = svd.fit_transform(matrix).astype(np.float32)

    aligned = np.zeros((matrix.shape[0], target_dim), dtype=np.float32)
    aligned[:, : dense.shape[1]] = dense[:, :target_dim]
    return aligned, svd


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(str(text))]


def _ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    ngram_min, ngram_max = tuple(int(value) for value in ngram_range)
    output: list[str] = []
    for ngram_size in range(ngram_min, ngram_max + 1):
        if ngram_size <= 0 or len(tokens) < ngram_size:
            continue
        output.extend(" ".join(tokens[idx : idx + ngram_size]) for idx in range(0, len(tokens) - ngram_size + 1))
    return output


def _fit_tfidf_numpy(
    texts: list[str],
    *,
    target_dim: int,
    max_features: int,
    ngram_range: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    documents = [_ngrams(_tokens(text), ngram_range) for text in texts]
    term_counts: dict[str, int] = {}
    doc_freq: dict[str, int] = {}
    for terms in documents:
        seen = set()
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
            if term not in seen:
                doc_freq[term] = doc_freq.get(term, 0) + 1
                seen.add(term)

    vocabulary = [
        term
        for term, _count in sorted(term_counts.items(), key=lambda item: (-item[1], item[0]))[: int(max_features)]
    ]
    if not vocabulary:
        return np.zeros((len(texts), int(target_dim)), dtype=np.float32), {
            "backend": "numpy_fallback",
            "vocabulary_size": 0,
            "svd_components": 0,
        }

    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    counts = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)
    for row_idx, terms in enumerate(documents):
        for term in terms:
            column_idx = term_to_idx.get(term)
            if column_idx is not None:
                counts[row_idx, column_idx] += 1.0
    row_sums = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    tf = counts / row_sums
    idf = np.asarray(
        [math.log((1.0 + len(documents)) / (1.0 + doc_freq.get(term, 0))) + 1.0 for term in vocabulary],
        dtype=np.float32,
    )
    tfidf = tf * idf.reshape(1, -1)

    target_dim = int(target_dim)
    if tfidf.shape[1] <= target_dim:
        aligned = np.zeros((tfidf.shape[0], target_dim), dtype=np.float32)
        aligned[:, : tfidf.shape[1]] = tfidf
        svd_components = 0
    else:
        centered = tfidf - tfidf.mean(axis=0, keepdims=True)
        u_matrix, singular_values, _v_transpose = np.linalg.svd(centered, full_matrices=False)
        svd_components = min(target_dim, u_matrix.shape[1])
        reduced = (u_matrix[:, :svd_components] * singular_values[:svd_components]).astype(np.float32)
        aligned = np.zeros((tfidf.shape[0], target_dim), dtype=np.float32)
        aligned[:, :svd_components] = reduced
    return aligned, {
        "backend": "numpy_fallback",
        "vocabulary_size": int(len(vocabulary)),
        "svd_components": int(svd_components),
    }


def fit_bow_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    random_state: int = 42,
) -> BowFeatureResult:
    """Fit a TF-IDF/SVD BoW representation and return fixed-width features."""

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    texts = _text_series(news_df, text_column)
    non_empty_texts = [text for text in texts.tolist() if text.strip()]
    if not non_empty_texts:
        frame = _zero_feature_frame(news_df, target_dim=target_dim)
        return BowFeatureResult(
            frame=frame,
            vectorizer=None,
            svd=None,
            manifest={
                "text_column": text_column,
                "target_dim": target_dim,
                "max_features": int(max_features),
                "ngram_range": list(ngram_range),
                "random_state": int(random_state),
                "row_count": int(len(news_df)),
                "fitted": False,
                "reason": "all_text_empty",
            },
        )

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=int(max_features),
            ngram_range=tuple(int(value) for value in ngram_range),
            lowercase=True,
        )
        tfidf = vectorizer.fit_transform(texts.tolist())
        features, svd = _dense_aligned_features(tfidf, target_dim=target_dim, random_state=int(random_state))
        backend_metadata = {
            "backend": "sklearn",
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "svd_components": int(0 if svd is None else svd.n_components),
        }
    except ModuleNotFoundError:
        features, backend_metadata = _fit_tfidf_numpy(
            texts.tolist(),
            target_dim=target_dim,
            max_features=int(max_features),
            ngram_range=tuple(int(value) for value in ngram_range),
        )
        vectorizer = None
        svd = None
    frame = pd.DataFrame(
        {
            "news_row_id": _ensure_news_row_id(news_df),
            "article_id": _metadata_column(news_df, "ArticleID", "article_id"),
            "source_file": _metadata_column(news_df, "SourceFile", "source_file"),
            "bow_embedding": [_serialize_vector(row) for row in features],
            "bow_dim": [target_dim] * len(news_df),
        }
    )
    return BowFeatureResult(
        frame=frame,
        vectorizer=vectorizer,
        svd=svd,
        manifest={
            "text_column": text_column,
            "target_dim": target_dim,
            "max_features": int(max_features),
            "ngram_range": list(ngram_range),
            "random_state": int(random_state),
            "row_count": int(len(news_df)),
            "fitted": True,
            **backend_metadata,
        },
    )


def build_bow_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Return only the BoW feature frame for callers that do not need fitted models."""

    return fit_bow_features(
        news_df,
        text_column=text_column,
        target_dim=target_dim,
        max_features=max_features,
        ngram_range=ngram_range,
    ).frame
