"""Compute auditable Loughran--McDonald dictionary sentiment scores.

The primary signed score in this module is a derived net-tone measure,

    (adjusted positive count - negative count) / LM-valid word count.

Loughran and McDonald's (2011) original headline measure is the negative-word
share.  Both quantities are retained so callers cannot accidentally describe
the derived signed score as the paper's only or original specification.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LM_CATEGORY_COLUMNS: Mapping[str, str] = {
    "negative": "Negative",
    "positive": "Positive",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
    "strong_modal": "Strong_Modal",
    "weak_modal": "Weak_Modal",
    "constraining": "Constraining",
}
LM_NEGATORS = frozenset({"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY"})

_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
_SCHEMA_VERSION = "lm_dictionary_sentiment_v1"
_PRIMARY_SCORE = "lm_net_tone"
_LM_2011_DOI = "https://doi.org/10.1111/j.1540-6261.2010.01625.x"
_LM_DICTIONARY_URL = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
_NET_TONE_APPLICATION_DOI = "https://doi.org/10.1007/s11156-022-01098-0"


@dataclass(frozen=True)
class LMDictionary:
    """Validated active word lists from one LM Master Dictionary file."""

    words: frozenset[str]
    categories: Mapping[str, frozenset[str]]
    path: Path
    sha256: str
    row_count: int


@dataclass(frozen=True)
class LMFeatureResult:
    """Article and optional pair scores plus reproducibility metadata."""

    article_scores: pd.DataFrame
    pair_scores: pd.DataFrame | None
    manifest: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {
        str(column).strip().lower().replace("-", "_").replace(" ", "_"): str(column)
        for column in frame.columns
    }


def _require_column(frame: pd.DataFrame, requested: str) -> str:
    lookup = _normalized_column_lookup(frame)
    key = requested.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in lookup:
        raise ValueError(
            f"LM dictionary is missing required column {requested!r}; "
            f"available columns: {list(frame.columns)}"
        )
    return lookup[key]


def load_lm_dictionary(path: str | Path) -> LMDictionary:
    """Load the LM dictionary, treating only positive category flags as active.

    The current Master Dictionary uses negative year values to mark words that
    were removed from a sentiment category.  Testing ``!= 0`` would therefore
    silently reintroduce retired terms; this loader deliberately requires
    ``flag > 0``.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"LM Master Dictionary does not exist: {resolved}")

    frame = pd.read_csv(resolved)
    word_column = _require_column(frame, "Word")
    raw_words = frame[word_column]
    usable = raw_words.notna() & raw_words.astype(str).str.strip().ne("")
    normalized_words = raw_words.loc[usable].astype(str).str.strip().str.upper()
    if normalized_words.duplicated().any():
        duplicates = sorted(
            normalized_words.loc[normalized_words.duplicated()].unique()
        )
        raise ValueError(f"LM dictionary contains duplicate words: {duplicates[:5]}")

    categories: dict[str, frozenset[str]] = {}
    for category, requested_column in LM_CATEGORY_COLUMNS.items():
        column = _require_column(frame, requested_column)
        flags = pd.to_numeric(frame.loc[usable, column], errors="raise")
        categories[category] = frozenset(normalized_words.loc[flags.gt(0)].tolist())

    return LMDictionary(
        words=frozenset(normalized_words.tolist()),
        categories=categories,
        path=resolved,
        sha256=_sha256_file(resolved),
        row_count=int(usable.sum()),
    )


def _normalize_text(value: object) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    return str(value)


def _tokens(text: object) -> list[str]:
    """Apply the current Notre Dame generic-parser token convention."""

    candidates = _TOKEN_PATTERN.findall(_normalize_text(text))
    return [
        token.upper()
        for token in candidates
        if token not in {"May", "MAY"} and len(token) >= 2 and not token.isdigit()
    ]


def _safe_share(numerator: int | float, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def score_lm_text(
    text: object,
    dictionary: LMDictionary,
    *,
    negation_window: int = 4,
) -> dict[str, int | float | bool]:
    """Score one document using LM category counts and valid-word shares."""

    if int(negation_window) < 0:
        raise ValueError("negation_window must be non-negative")

    tokens = _tokens(text)
    valid_tokens = [token for token in tokens if token in dictionary.words]
    counts = {
        category: sum(token in words for token in valid_tokens)
        for category, words in dictionary.categories.items()
    }

    positive_words = dictionary.categories["positive"]
    negated_positive_count = 0
    for index, token in enumerate(tokens):
        if token not in positive_words:
            continue
        prior_tokens = tokens[max(0, index - int(negation_window)) : index]
        if any(prior in LM_NEGATORS for prior in prior_tokens):
            negated_positive_count += 1

    positive_count = int(counts["positive"])
    adjusted_positive_count = max(0, positive_count - negated_positive_count)
    negative_count = int(counts["negative"])
    valid_word_count = len(valid_tokens)
    raw_word_count = len(tokens)

    result: dict[str, int | float | bool] = {
        "lm_raw_word_count": raw_word_count,
        "lm_valid_word_count": valid_word_count,
        "lm_dictionary_coverage": _safe_share(valid_word_count, raw_word_count),
        "lm_negated_positive_count": int(negated_positive_count),
        "lm_adjusted_positive_count": adjusted_positive_count,
        "lm_has_valid_words": bool(valid_word_count),
        "lm_has_sentiment_words": bool(adjusted_positive_count + negative_count),
    }
    for category in LM_CATEGORY_COLUMNS:
        count = int(counts[category])
        result[f"lm_{category}_count"] = count
        result[f"lm_{category}_share"] = _safe_share(count, valid_word_count)

    adjusted_positive_share = _safe_share(adjusted_positive_count, valid_word_count)
    negative_share = _safe_share(negative_count, valid_word_count)
    net_tone = adjusted_positive_share - negative_share
    result.update(
        {
            "lm_adjusted_positive_share": adjusted_positive_share,
            "lm_negative_share": negative_share,
            "lm_net_tone": net_tone,
            "lm_sentiment_score": net_tone,
        }
    )
    return result


def _news_row_ids(frame: pd.DataFrame) -> pd.Series:
    if "news_row_id" in frame.columns:
        values = pd.to_numeric(frame["news_row_id"], errors="raise").astype("int64")
    else:
        values = pd.Series(
            np.arange(1, len(frame) + 1), index=frame.index, dtype="int64"
        )
    if values.duplicated().any():
        duplicates = sorted(values.loc[values.duplicated()].unique())
        raise ValueError(f"news_row_id is not unique: {duplicates[:5]}")
    return values


def _optional_text_column(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def score_lm_articles(
    news_frame: pd.DataFrame,
    dictionary: LMDictionary,
    *,
    text_column: str = "LP",
    negation_window: int = 4,
) -> pd.DataFrame:
    """Return one LM score row for every article/news row."""

    if text_column not in news_frame.columns:
        raise ValueError(
            f"Missing article text column {text_column!r}; "
            f"available columns: {list(news_frame.columns)}"
        )
    texts = news_frame[text_column].tolist()
    rows = [
        score_lm_text(text, dictionary, negation_window=negation_window)
        for text in texts
    ]
    scores = pd.DataFrame(rows)
    scores.insert(
        0,
        "lm_text_sha256",
        [
            hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()
            for text in texts
        ],
    )
    metadata = pd.DataFrame(
        {
            "news_row_id": _news_row_ids(news_frame),
            "article_id": _optional_text_column(news_frame, "ArticleID", "article_id"),
            "source_file": _optional_text_column(
                news_frame, "SourceFile", "source_file"
            ),
        }
    )
    return pd.concat([metadata.reset_index(drop=True), scores], axis=1)


def _article_key(row: object) -> str:
    for column in ("article_id", "sample_id", "news_row_id"):
        value = getattr(row, column, None)
        if value is not None and not bool(pd.isna(value)) and str(value).strip():
            return f"{column}:{str(value).strip()}"
    raise ValueError("Pair row lacks article_id, sample_id, and news_row_id")


def _aggregate_lm_score_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    integer_columns = [
        "lm_raw_word_count",
        "lm_valid_word_count",
        "lm_negated_positive_count",
        "lm_adjusted_positive_count",
        *(f"lm_{category}_count" for category in LM_CATEGORY_COLUMNS),
    ]
    totals = {
        column: int(sum(int(row[column]) for row in rows)) for column in integer_columns
    }
    valid_word_count = totals["lm_valid_word_count"]
    raw_word_count = totals["lm_raw_word_count"]
    result: dict[str, object] = {
        **totals,
        "lm_dictionary_coverage": _safe_share(valid_word_count, raw_word_count),
        "lm_has_valid_words": bool(valid_word_count),
    }
    for category in LM_CATEGORY_COLUMNS:
        result[f"lm_{category}_share"] = _safe_share(
            totals[f"lm_{category}_count"], valid_word_count
        )
    adjusted_positive_share = _safe_share(
        totals["lm_adjusted_positive_count"], valid_word_count
    )
    negative_share = _safe_share(totals["lm_negative_count"], valid_word_count)
    net_tone = adjusted_positive_share - negative_share
    result.update(
        {
            "lm_adjusted_positive_share": adjusted_positive_share,
            "lm_negative_share": negative_share,
            "lm_net_tone": net_tone,
            "lm_sentiment_score": net_tone,
            "lm_has_sentiment_words": bool(
                totals["lm_adjusted_positive_count"] + totals["lm_negative_count"]
            ),
        }
    )
    return result


def score_lm_pairs(
    pair_frame: pd.DataFrame,
    article_scores: pd.DataFrame,
    *,
    pair_text_column: str = "lp_text",
) -> pd.DataFrame:
    """Aggregate article LM counts to pair scores after within-pair deduplication."""

    required_pair_columns = {"pair_id", "news_row_id", pair_text_column}
    missing = sorted(required_pair_columns - set(pair_frame.columns))
    if missing:
        raise ValueError(f"Pair workbook is missing required columns: {missing}")

    pairs = pair_frame.copy()
    pairs["news_row_id"] = pd.to_numeric(pairs["news_row_id"], errors="raise").astype(
        "int64"
    )
    scores = article_scores.copy()
    scores["news_row_id"] = pd.to_numeric(scores["news_row_id"], errors="raise").astype(
        "int64"
    )
    if scores["news_row_id"].duplicated().any():
        raise ValueError("Article scores contain duplicate news_row_id values")

    score_columns = [column for column in scores.columns if column.startswith("lm_")]
    joined = pairs.merge(
        scores[["news_row_id", "article_id", *score_columns]],
        on="news_row_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_source"),
        indicator=True,
    )
    missing_ids = sorted(
        joined.loc[joined["_merge"] != "both", "news_row_id"].unique().tolist()
    )
    if missing_ids:
        raise ValueError(f"Pair rows lack article LM scores: {missing_ids[:5]}")
    joined = joined.drop(columns="_merge")

    pair_text_sha = joined[pair_text_column].map(
        lambda value: hashlib.sha256(_normalize_text(value).encode("utf-8")).hexdigest()
    )
    text_mismatch = pair_text_sha.ne(joined["lm_text_sha256"])
    if text_mismatch.any():
        sample = joined.loc[
            text_mismatch, ["news_row_id", pair_text_column, "lm_text_sha256"]
        ]
        raise ValueError(
            "Pair/article text lineage mismatch for news rows:\n"
            f"{sample.head().to_string(index=False)}"
        )

    if "article_id" in joined and "article_id_source" in joined:
        pair_ids = joined["article_id"].fillna("").astype(str).str.strip()
        source_ids = joined["article_id_source"].fillna("").astype(str).str.strip()
        mismatch = pair_ids.ne("") & source_ids.ne("") & pair_ids.ne(source_ids)
        if mismatch.any():
            sample = joined.loc[
                mismatch, ["news_row_id", "article_id", "article_id_source"]
            ]
            raise ValueError(
                f"Article lineage mismatch:\n{sample.head().to_string(index=False)}"
            )

    if "sample_id" not in joined:
        joined["sample_id"] = ""
    sort_news = pd.to_numeric(joined["news_row_id"], errors="raise")
    joined = joined.assign(_news_sort=sort_news).sort_values(
        ["pair_id", "_news_sort", "sample_id"], kind="stable"
    )

    output_rows: list[dict[str, object]] = []
    for pair_id, group in joined.groupby("pair_id", sort=True):
        deduplicated: dict[str, object] = {}
        duplicate_rows = 0
        for row in group.itertuples(index=False):
            key = _article_key(row)
            previous = deduplicated.get(key)
            if previous is not None:
                if previous.lm_text_sha256 != row.lm_text_sha256:
                    raise ValueError(
                        "Duplicate article key maps to different LP text within pair: "
                        f"{pair_id}/{key}"
                    )
                duplicate_rows += 1
                continue
            deduplicated[key] = row
        unique_rows = list(deduplicated.values())
        aggregate = _aggregate_lm_score_rows(
            [
                {column: getattr(row, column) for column in score_columns}
                for row in unique_rows
            ]
        )
        record: dict[str, object] = {
            "pair_id": str(pair_id),
            "lm_unique_article_count": len(unique_rows),
            "lm_duplicate_article_row_count": duplicate_rows,
            **aggregate,
        }
        for column in (
            "session_id",
            "effective_origin_utc",
            "target_anchor_utc",
            "tolerance_minutes",
        ):
            if column in group.columns:
                values = group[column].dropna().astype(str).unique().tolist()
                if len(values) > 1:
                    raise ValueError(
                        f"Pair {pair_id} has conflicting {column}: {values}"
                    )
                record[column] = values[0] if values else ""
        output_rows.append(record)

    return pd.DataFrame(output_rows)


def build_lm_feature_result(
    news_frame: pd.DataFrame,
    dictionary: LMDictionary,
    *,
    text_column: str = "LP",
    negation_window: int = 4,
    pair_frame: pd.DataFrame | None = None,
    pair_text_column: str = "lp_text",
) -> LMFeatureResult:
    """Build article/pair scores and a compact method manifest."""

    article_scores = score_lm_articles(
        news_frame,
        dictionary,
        text_column=text_column,
        negation_window=negation_window,
    )
    pair_scores = (
        score_lm_pairs(
            pair_frame,
            article_scores,
            pair_text_column=pair_text_column,
        )
        if pair_frame is not None
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "representation": "loughran_mcdonald_dictionary_sentiment",
        "primary_score": _PRIMARY_SCORE,
        "primary_score_formula": (
            "(adjusted_positive_count-negative_count)/lm_valid_word_count"
        ),
        "lm_2011_primary_measure": "negative_count/lm_valid_word_count",
        "denominator_definition": (
            "tokens_of_length_at_least_2_that_appear_in_lm_master_dictionary"
        ),
        "tokenizer": {
            "pattern": r"(?u)\b\w+\b",
            "case": "uppercase_after_tokenization",
            "exclusions": ["numeric_tokens", "single_character_tokens", "May", "MAY"],
            "stemming": False,
        },
        "negation_rule": {
            "applies_to": "positive_words_only",
            "preceding_token_window": int(negation_window),
            "negators": sorted(LM_NEGATORS),
        },
        "dictionary": {
            "path": str(dictionary.path),
            "sha256": dictionary.sha256,
            "word_count": dictionary.row_count,
            "active_category_word_counts": {
                category: len(words)
                for category, words in dictionary.categories.items()
            },
            "active_flag_rule": "strictly_greater_than_zero",
        },
        "references": {
            "loughran_mcdonald_2011": _LM_2011_DOI,
            "official_dictionary": _LM_DICTIONARY_URL,
            "net_tone_positive_minus_negative_over_words_application": (
                _NET_TONE_APPLICATION_DOI
            ),
        },
        "article_row_count": len(article_scores),
        "pair_row_count": 0 if pair_scores is None else len(pair_scores),
    }
    return LMFeatureResult(
        article_scores=article_scores,
        pair_scores=pair_scores,
        manifest=manifest,
    )
