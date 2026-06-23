"""Build dictionary-based sentiment features for RQ2 traditional text baselines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")
_CATEGORIES = (
    "positive",
    "negative",
    "uncertainty",
    "litigious",
    "strong_modal",
    "weak_modal",
    "constraining",
)
_LM_COLUMN_CANDIDATES = {
    "positive": ("positive",),
    "negative": ("negative",),
    "uncertainty": ("uncertainty",),
    "litigious": ("litigious",),
    "strong_modal": ("strong_modal", "strong modal", "strongmodal"),
    "weak_modal": ("weak_modal", "weak modal", "weakmodal"),
    "constraining": ("constraining",),
}
_BUILTIN_LEXICON: dict[str, set[str]] = {
    "positive": {
        "benefit",
        "beneficial",
        "gain",
        "gains",
        "growth",
        "improve",
        "improved",
        "improvement",
        "positive",
        "profit",
        "strong",
        "upside",
    },
    "negative": {
        "adverse",
        "decline",
        "declined",
        "downside",
        "loss",
        "losses",
        "negative",
        "risk",
        "risks",
        "weak",
        "weakness",
        "worse",
    },
    "uncertainty": {
        "ambiguous",
        "contingent",
        "could",
        "may",
        "might",
        "uncertain",
        "uncertainty",
        "unknown",
        "volatile",
        "volatility",
    },
    "litigious": {
        "claim",
        "claims",
        "court",
        "legal",
        "litigation",
        "regulation",
        "regulatory",
        "sue",
        "sued",
    },
    "strong_modal": {"always", "definitely", "must", "undoubtedly", "will"},
    "weak_modal": {"could", "may", "might", "perhaps", "possibly"},
    "constraining": {"constraint", "constraints", "restrict", "restricted", "restriction"},
}


@dataclass(frozen=True)
class SentimentFeatureResult:
    """Feature frame plus metadata describing the dictionary source."""

    frame: pd.DataFrame
    lexicon: dict[str, set[str]]
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


def _is_active_dictionary_value(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return float(text) > 0.0
    except ValueError:
        return text.lower() not in {"false", "no", "none", "nan", "0"}


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _find_word_column(frame: pd.DataFrame) -> str:
    normalized = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in ("word", "words", "term"):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"Dictionary is missing a word column. Available columns: {list(frame.columns)}")


def _find_category_column(frame: pd.DataFrame, category: str) -> str | None:
    normalized = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in _LM_COLUMN_CANDIDATES[category]:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def load_sentiment_lexicon(
    dictionary_path: str | Path | None,
    *,
    allow_builtin_fallback: bool = True,
) -> tuple[dict[str, set[str]], str]:
    """Load a Loughran-McDonald-style dictionary or an explicitly marked fallback."""

    if dictionary_path is None or not str(dictionary_path).strip():
        if not allow_builtin_fallback:
            raise FileNotFoundError("No sentiment dictionary path was provided.")
        return {category: set(words) for category, words in _BUILTIN_LEXICON.items()}, "fallback_builtin"

    path = Path(dictionary_path)
    if not path.exists():
        if not allow_builtin_fallback:
            raise FileNotFoundError(f"Sentiment dictionary does not exist: {path}")
        return {category: set(words) for category, words in _BUILTIN_LEXICON.items()}, "fallback_builtin"

    dictionary = pd.read_csv(path)
    word_column = _find_word_column(dictionary)
    lexicon = {category: set() for category in _CATEGORIES}
    for category in _CATEGORIES:
        category_column = _find_category_column(dictionary, category)
        if category_column is None:
            continue
        active = dictionary[dictionary[category_column].map(_is_active_dictionary_value)]
        lexicon[category] = {str(word).strip().lower() for word in active[word_column] if str(word).strip()}
    return lexicon, str(path)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(str(text))]


def _sentiment_vector(tokens: list[str], lexicon: Mapping[str, set[str]]) -> np.ndarray:
    token_count = max(1, len(tokens))
    token_set_counts = {category: 0.0 for category in _CATEGORIES}
    for token in tokens:
        for category in _CATEGORIES:
            if token in lexicon.get(category, set()):
                token_set_counts[category] += 1.0
    counts = np.asarray([token_set_counts[category] for category in _CATEGORIES], dtype=np.float32)
    shares = counts / float(token_count)
    positive = float(token_set_counts["positive"])
    negative = float(token_set_counts["negative"])
    net_positive = positive - negative
    polarity = net_positive / max(1.0, positive + negative)
    return np.concatenate(
        [
            counts,
            shares,
            np.asarray([net_positive, polarity, float(len(tokens))], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _align_vector(values: np.ndarray, *, target_dim: int) -> np.ndarray:
    aligned = np.zeros(int(target_dim), dtype=np.float32)
    width = min(int(values.size), int(target_dim))
    if width > 0:
        aligned[:width] = values[:width]
    return aligned


def fit_sentiment_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    dictionary_path: str | Path | None = None,
    allow_builtin_fallback: bool = True,
) -> SentimentFeatureResult:
    """Build fixed-width dictionary sentiment vectors from news text."""

    target_dim = int(target_dim)
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    lexicon, dictionary_source = load_sentiment_lexicon(
        dictionary_path,
        allow_builtin_fallback=bool(allow_builtin_fallback),
    )
    texts = _text_series(news_df, text_column)
    vectors = [
        _align_vector(_sentiment_vector(_tokenize(text), lexicon), target_dim=target_dim)
        for text in texts.tolist()
    ]
    frame = pd.DataFrame(
        {
            "news_row_id": _ensure_news_row_id(news_df),
            "article_id": _metadata_column(news_df, "ArticleID", "article_id"),
            "source_file": _metadata_column(news_df, "SourceFile", "source_file"),
            "sentiment_embedding": [_serialize_vector(vector) for vector in vectors],
            "sentiment_dim": [target_dim] * len(news_df),
            "sentiment_dictionary_source": [dictionary_source] * len(news_df),
        }
    )
    return SentimentFeatureResult(
        frame=frame,
        lexicon={category: set(words) for category, words in lexicon.items()},
        manifest={
            "text_column": text_column,
            "target_dim": target_dim,
            "row_count": int(len(news_df)),
            "dictionary_source": dictionary_source,
            "categories": list(_CATEGORIES),
            "base_feature_dim": int(len(_CATEGORIES) * 2 + 3),
        },
    )


def build_sentiment_features(
    news_df: pd.DataFrame,
    *,
    text_column: str = "LP",
    target_dim: int = 1024,
    dictionary_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return only the sentiment feature frame for simple callers."""

    return fit_sentiment_features(
        news_df,
        text_column=text_column,
        target_dim=target_dim,
        dictionary_path=dictionary_path,
    ).frame
