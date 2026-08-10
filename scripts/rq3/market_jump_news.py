"""Deterministic news and scheduled-event bridges for market-jump episodes.

The anomaly detector deliberately discovers episodes from the full market
index first.  This module is the separate, post-hoc evidence layer: it retains
every Factiva article or official release in the configured half-open impact
and context windows and never promotes one candidate to a causal explanation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import scripts._path_setup  # noqa: F401

from wgan_option.news_time import (
    DEFAULT_NEWS_SOURCE_TIMEZONE,
    parse_news_timestamps,
)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_OLD_HELPER_MARKER = "with_ty_plus_trade_counts"
_EPISODE_START_ALIASES = (
    "episode_start_utc",
    "start_utc",
    "current_snapshot_time_utc",
)
_EPISODE_END_ALIASES = (
    "episode_end_utc",
    "end_utc",
    "target_snapshot_time_utc",
)
_WINDOW_ORDER = {"pre_context": 0, "impact": 1, "post_reporting": 2}


NEWS_BRIDGE_COLUMNS = [
    "episode_news_bridge_id",
    "episode_id",
    "episode_start_utc",
    "episode_end_utc",
    "window_relation",
    "news_row_id",
    "sample_id",
    "news_available_time_utc",
    "publication_timestamp_utc",
    "minutes_from_episode_start",
    "minutes_from_episode_end",
    "workbook_sha256",
    "lp_text_sha256",
    "source_file",
    "article_id",
    "headline",
    "lead_paragraph",
    "publication_group_size",
    "publication_collision_count",
    "article_id_group_size",
    "lp_text_group_size",
    "episode_window_candidate_count",
    "episode_candidate_count",
    "alignment_row_count",
    "alignment_match_method",
    "alignment_closed_to_next_open",
    "alignment_effective_origin_utc",
    "alignment_target_anchor_utc",
    "alignment_session_regime",
    "alignment_quality_status",
    "alignment_has_match",
    "alignment_collision_count",
]

OFFICIAL_BRIDGE_COLUMNS = [
    "episode_official_bridge_id",
    "episode_id",
    "episode_start_utc",
    "episode_end_utc",
    "window_relation",
    "calendar_row_id",
    "event_id",
    "release_time_utc",
    "minutes_from_episode_start",
    "minutes_from_episode_end",
    "episode_window_candidate_count",
    "episode_candidate_count",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _lp_sha256(value: Any) -> str:
    normalized = " ".join(_TOKEN_PATTERN.findall(_optional_text(value).lower()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _utc_string(value: Any) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _int_or_zero(value: Any) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0 if pd.isna(numeric) else int(numeric)


def _group_size_for_nonempty(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].fillna("").astype(str).str.strip()
    sizes = values.groupby(values, sort=False).transform("size").astype(int)
    return sizes.where(values.ne(""), 0).astype(int)


def parse_factiva_news(
    news_xlsx: str | Path,
    source_timezone: str = DEFAULT_NEWS_SOURCE_TIMEZONE,
) -> pd.DataFrame:
    """Read the canonical Factiva workbook and add auditable UTC identities.

    The older ``*_with_ty_plus_trade_counts.xlsx`` helper encoded Factiva
    timestamps with the wrong source timezone.  It is rejected by name so that
    it cannot silently re-enter the RQ3 evidence pipeline.
    """

    path = Path(news_xlsx).expanduser()
    if _OLD_HELPER_MARKER in path.name.lower():
        raise ValueError(
            "Refusing legacy with_ty_plus_trade_counts workbook: its Factiva "
            "timestamps were interpreted with the wrong source timezone."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Factiva workbook does not exist: {path}")

    # Embedding vectors are deliberately excluded: this evidence bridge needs
    # article identity/text and timestamps, not model inputs.
    source_columns = {"SourceFile", "ArticleID", "AN", "HD", "PD", "ET", "SN", "LP"}
    frame = pd.read_excel(
        path,
        engine="openpyxl",
        dtype=object,
        usecols=lambda column: str(column) in source_columns,
    )
    missing = sorted({"SourceFile", "ArticleID", "HD", "PD", "ET", "LP"} - set(frame.columns))
    if missing:
        raise ValueError(f"Factiva workbook is missing required columns {missing}: {path}")

    frame = frame.reset_index(drop=True).copy()
    frame.insert(0, "news_row_id", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame.insert(1, "sample_id", frame["news_row_id"].map(lambda value: f"news_{value}"))
    parsed = parse_news_timestamps(
        frame["PD"],
        frame["ET"],
        source_timezone=source_timezone,
    )
    frame["source_local_timestamp"] = parsed.source_local_timestamp
    frame["source_timezone"] = str(source_timezone)
    frame["source_utc_offset_minutes"] = parsed.utc_offset_minutes
    frame["timestamp_parse_status"] = parsed.parse_status
    frame["publication_timestamp_utc"] = parsed.timestamp_utc
    # No availability lag is asserted by the frozen source workbook.  Keeping
    # this as a separate field makes a later, explicit lag policy possible.
    frame["news_available_time_utc"] = parsed.timestamp_utc
    frame["publication_availability_lag_minutes"] = 0
    frame["workbook_sha256"] = _sha256_file(path)

    frame["lp_text_sha256"] = frame["LP"].map(_lp_sha256)
    frame["publication_group_size"] = _group_size_for_nonempty(
        frame, "publication_timestamp_utc"
    )
    frame["publication_collision_count"] = (
        frame["publication_group_size"] - 1
    ).clip(lower=0)
    frame["article_id_group_size"] = _group_size_for_nonempty(frame, "ArticleID")
    frame["lp_text_group_size"] = _group_size_for_nonempty(frame, "lp_text_sha256")
    return frame


def _frame_from_csv_or_frame(value: pd.DataFrame | str | Path, *, label: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return pd.read_csv(path, low_memory=False)


def _first_column(frame: pd.DataFrame, aliases: Iterable[str], *, label: str) -> str:
    for column in aliases:
        if column in frame.columns:
            return column
    raise ValueError(f"{label} is missing a supported timestamp column: {list(aliases)}")


def _prepare_episodes(episodes: pd.DataFrame | str | Path) -> pd.DataFrame:
    frame = _frame_from_csv_or_frame(episodes, label="Episodes table").reset_index(drop=True)
    if "episode_id" not in frame.columns:
        raise ValueError("Episodes table is missing required column: episode_id")
    if frame["episode_id"].astype(str).duplicated().any():
        raise ValueError("Episodes table contains duplicate episode_id values")
    start_column = _first_column(frame, _EPISODE_START_ALIASES, label="Episodes table")
    end_column = _first_column(frame, _EPISODE_END_ALIASES, label="Episodes table")
    frame["_start"] = pd.to_datetime(frame[start_column], utc=True, errors="coerce")
    frame["_end"] = pd.to_datetime(frame[end_column], utc=True, errors="coerce")
    invalid = frame["_start"].isna() | frame["_end"].isna()
    if invalid.any():
        ids = frame.loc[invalid, "episode_id"].astype(str).tolist()[:5]
        raise ValueError(f"Episodes table contains invalid UTC timestamps: {ids}")
    reversed_window = frame["_end"] < frame["_start"]
    if reversed_window.any():
        ids = frame.loc[reversed_window, "episode_id"].astype(str).tolist()[:5]
        raise ValueError(f"Episode end precedes start: {ids}")
    frame["episode_id"] = frame["episode_id"].astype(str)
    frame["episode_start_utc"] = frame["_start"].map(_utc_string)
    frame["episode_end_utc"] = frame["_end"].map(_utc_string)
    return frame.sort_values(["_start", "_end", "episode_id"], kind="mergesort").reset_index(drop=True)


def _window_relation(
    timestamp: pd.Timestamp,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pre: pd.Timedelta,
    post: pd.Timedelta,
) -> str | None:
    # All windows are half-open so five/15-minute settings represent exactly
    # five/15 minutes at minute-granularity and adjacent windows never overlap.
    if start <= timestamp < end:
        return "impact"
    if start - pre <= timestamp < start:
        return "pre_context"
    if end <= timestamp < end + post:
        return "post_reporting"
    return None


def _alignment_value(row: pd.Series, *columns: str) -> Any:
    for column in columns:
        if column in row.index:
            value = row[column]
            if _optional_text(value):
                return value
    return ""


def _prepare_alignment(alignment: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    columns = [
        "news_row_id",
        "alignment_row_count",
        "alignment_match_method",
        "alignment_closed_to_next_open",
        "alignment_effective_origin_utc",
        "alignment_target_anchor_utc",
        "alignment_session_regime",
        "alignment_quality_status",
        "alignment_has_match",
        "alignment_collision_count",
    ]
    if alignment is None:
        return pd.DataFrame(columns=columns)
    frame = _frame_from_csv_or_frame(alignment, label="News alignment table").reset_index(drop=True)
    if "news_row_id" not in frame.columns:
        raise ValueError("News alignment table is missing required column: news_row_id")
    frame["news_row_id"] = pd.to_numeric(frame["news_row_id"], errors="raise").astype(int)
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    frame["_has_match"] = pd.to_numeric(
        frame.get("has_match", pd.Series(0, index=frame.index)), errors="coerce"
    ).fillna(0).astype(int)
    frame["_matching_rank"] = pd.to_numeric(
        frame.get("matching_rank", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).fillna(np.inf)
    row_counts = frame.groupby("news_row_id", sort=False).size().to_dict()
    frame = frame.sort_values(
        ["news_row_id", "_has_match", "_matching_rank", "_source_order"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).drop_duplicates("news_row_id", keep="first")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        method = _optional_text(_alignment_value(row, "match_method", "alignment_type"))
        session_regime = _optional_text(
            _alignment_value(row, "session_regime", "publication_market_state")
        )
        closed = any(
            _optional_text(_alignment_value(row, column)).lower() == "closed_to_next_open"
            for column in ("match_method", "alignment_type", "session_regime")
        )
        rows.append(
            {
                "news_row_id": int(row["news_row_id"]),
                "alignment_row_count": int(row_counts[int(row["news_row_id"])]),
                "alignment_match_method": method,
                "alignment_closed_to_next_open": int(closed),
                "alignment_effective_origin_utc": _utc_string(
                    _alignment_value(row, "effective_origin_utc")
                ),
                "alignment_target_anchor_utc": _utc_string(
                    _alignment_value(row, "target_anchor_utc")
                ),
                "alignment_session_regime": session_regime,
                "alignment_quality_status": _optional_text(
                    _alignment_value(row, "quality_status")
                ),
                "alignment_has_match": int(row["_has_match"]),
                "alignment_collision_count": int(
                    pd.to_numeric(
                        pd.Series([_alignment_value(row, "collision_count")]),
                        errors="coerce",
                    ).fillna(0).iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_episode_news_bridge(
    episodes: pd.DataFrame | str | Path,
    news: pd.DataFrame,
    alignment: pd.DataFrame | str | Path | None = None,
    pre_minutes: int = 5,
    post_minutes: int = 15,
) -> pd.DataFrame:
    """Return every Factiva candidate in each episode's three evidence windows."""

    if int(pre_minutes) < 0 or int(post_minutes) < 0:
        raise ValueError("pre_minutes and post_minutes must be non-negative")
    episode_frame = _prepare_episodes(episodes)
    news_frame = news.copy().reset_index(drop=True)
    required = {"news_row_id", "news_available_time_utc"}
    missing = sorted(required - set(news_frame.columns))
    if missing:
        raise ValueError(f"Parsed news frame is missing required columns: {missing}")
    if pd.to_numeric(news_frame["news_row_id"], errors="coerce").isna().any():
        raise ValueError("Parsed news frame contains invalid news_row_id values")
    news_frame["news_row_id"] = pd.to_numeric(news_frame["news_row_id"]).astype(int)
    if news_frame["news_row_id"].duplicated().any():
        raise ValueError("Parsed news frame contains duplicate news_row_id values")
    news_frame["_available"] = pd.to_datetime(
        news_frame["news_available_time_utc"], utc=True, errors="coerce"
    )
    alignment_frame = _prepare_alignment(alignment)
    if not alignment_frame.empty:
        news_frame = news_frame.merge(
            alignment_frame, on="news_row_id", how="left", validate="one_to_one"
        )

    pre = pd.Timedelta(minutes=int(pre_minutes))
    post = pd.Timedelta(minutes=int(post_minutes))
    rows: list[dict[str, Any]] = []
    for _, episode in episode_frame.iterrows():
        start = pd.Timestamp(episode["_start"])
        end = pd.Timestamp(episode["_end"])
        lower = start - pre
        upper = end + post
        candidates = news_frame[
            news_frame["_available"].notna()
            & (news_frame["_available"] >= lower)
            & (news_frame["_available"] < upper)
        ]
        for _, candidate in candidates.iterrows():
            available = pd.Timestamp(candidate["_available"])
            relation = _window_relation(
                available, start=start, end=end, pre=pre, post=post
            )
            if relation is None:
                continue
            news_row_id = int(candidate["news_row_id"])
            row = {
                "episode_news_bridge_id": f"{episode['episode_id']}::news_{news_row_id}::{relation}",
                "episode_id": str(episode["episode_id"]),
                "episode_start_utc": _utc_string(start),
                "episode_end_utc": _utc_string(end),
                "window_relation": relation,
                "news_row_id": news_row_id,
                "sample_id": _optional_text(candidate.get("sample_id")) or f"news_{news_row_id}",
                "news_available_time_utc": _utc_string(available),
                "publication_timestamp_utc": _utc_string(
                    candidate.get("publication_timestamp_utc")
                ),
                "minutes_from_episode_start": (available - start).total_seconds() / 60.0,
                "minutes_from_episode_end": (available - end).total_seconds() / 60.0,
                "workbook_sha256": _optional_text(candidate.get("workbook_sha256")),
                "lp_text_sha256": _optional_text(candidate.get("lp_text_sha256")),
                "source_file": _optional_text(candidate.get("SourceFile", candidate.get("source_file"))),
                "article_id": _optional_text(candidate.get("ArticleID", candidate.get("article_id"))),
                "headline": _optional_text(candidate.get("HD", candidate.get("headline"))),
                "lead_paragraph": _optional_text(candidate.get("LP", candidate.get("lead_paragraph"))),
                "publication_group_size": _int_or_zero(candidate.get("publication_group_size", 0)),
                "publication_collision_count": _int_or_zero(candidate.get("publication_collision_count", 0)),
                "article_id_group_size": _int_or_zero(candidate.get("article_id_group_size", 0)),
                "lp_text_group_size": _int_or_zero(candidate.get("lp_text_group_size", 0)),
                "alignment_row_count": _int_or_zero(candidate.get("alignment_row_count", 0)),
                "alignment_match_method": _optional_text(candidate.get("alignment_match_method")),
                "alignment_closed_to_next_open": _int_or_zero(candidate.get("alignment_closed_to_next_open", 0)),
                "alignment_effective_origin_utc": _optional_text(candidate.get("alignment_effective_origin_utc")),
                "alignment_target_anchor_utc": _optional_text(candidate.get("alignment_target_anchor_utc")),
                "alignment_session_regime": _optional_text(candidate.get("alignment_session_regime")),
                "alignment_quality_status": _optional_text(candidate.get("alignment_quality_status")),
                "alignment_has_match": _int_or_zero(candidate.get("alignment_has_match", 0)),
                "alignment_collision_count": _int_or_zero(candidate.get("alignment_collision_count", 0)),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=NEWS_BRIDGE_COLUMNS)
    result = pd.DataFrame(rows)
    result["episode_window_candidate_count"] = result.groupby(
        ["episode_id", "window_relation"], sort=False
    )["news_row_id"].transform("size")
    result["episode_candidate_count"] = result.groupby("episode_id", sort=False)[
        "news_row_id"
    ].transform("size")
    result["_window_order"] = result["window_relation"].map(_WINDOW_ORDER)
    result = result.sort_values(
        ["episode_start_utc", "episode_id", "_window_order", "news_available_time_utc", "news_row_id"],
        kind="mergesort",
    ).drop(columns="_window_order")
    return result[NEWS_BRIDGE_COLUMNS].reset_index(drop=True)


def build_episode_official_bridge(
    episodes: pd.DataFrame | str | Path,
    calendar: pd.DataFrame | str | Path,
    pre_minutes: int = 5,
    post_minutes: int = 15,
) -> pd.DataFrame:
    """Return every official release in each episode's evidence windows."""

    if int(pre_minutes) < 0 or int(post_minutes) < 0:
        raise ValueError("pre_minutes and post_minutes must be non-negative")
    episode_frame = _prepare_episodes(episodes)
    calendar_frame = _frame_from_csv_or_frame(calendar, label="Official calendar").reset_index(drop=True)
    time_column = _first_column(
        calendar_frame, ("release_time_utc", "event_time_utc"), label="Official calendar"
    )
    calendar_frame.insert(
        0, "calendar_row_id", np.arange(1, len(calendar_frame) + 1, dtype=np.int64)
    )
    calendar_frame["_release"] = pd.to_datetime(
        calendar_frame[time_column], utc=True, errors="coerce"
    )
    invalid = calendar_frame["_release"].isna()
    if invalid.any():
        raise ValueError(
            f"Official calendar contains {int(invalid.sum())} invalid UTC release timestamps"
        )
    if "event_id" not in calendar_frame.columns:
        calendar_frame["event_id"] = calendar_frame["calendar_row_id"].map(
            lambda value: f"official_{value}"
        )

    pre = pd.Timedelta(minutes=int(pre_minutes))
    post = pd.Timedelta(minutes=int(post_minutes))
    rows: list[dict[str, Any]] = []
    for _, episode in episode_frame.iterrows():
        start = pd.Timestamp(episode["_start"])
        end = pd.Timestamp(episode["_end"])
        candidates = calendar_frame[
            (calendar_frame["_release"] >= start - pre)
            & (calendar_frame["_release"] < end + post)
        ]
        for _, event in candidates.iterrows():
            release = pd.Timestamp(event["_release"])
            relation = _window_relation(
                release, start=start, end=end, pre=pre, post=post
            )
            if relation is None:
                continue
            calendar_row_id = int(event["calendar_row_id"])
            event_id = _optional_text(event.get("event_id")) or f"official_{calendar_row_id}"
            row = {
                "episode_official_bridge_id": (
                    f"{episode['episode_id']}::{event_id}::{calendar_row_id}::{relation}"
                ),
                "episode_id": str(episode["episode_id"]),
                "episode_start_utc": _utc_string(start),
                "episode_end_utc": _utc_string(end),
                "window_relation": relation,
                "calendar_row_id": calendar_row_id,
                "event_id": event_id,
                "release_time_utc": _utc_string(release),
                "minutes_from_episode_start": (release - start).total_seconds() / 60.0,
                "minutes_from_episode_end": (release - end).total_seconds() / 60.0,
            }
            for column in calendar_frame.columns:
                if column in row or column.startswith("_") or column == time_column:
                    continue
                row[column] = event[column]
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=OFFICIAL_BRIDGE_COLUMNS)
    result = pd.DataFrame(rows)
    result["episode_window_candidate_count"] = result.groupby(
        ["episode_id", "window_relation"], sort=False
    )["calendar_row_id"].transform("size")
    result["episode_candidate_count"] = result.groupby("episode_id", sort=False)[
        "calendar_row_id"
    ].transform("size")
    result["_window_order"] = result["window_relation"].map(_WINDOW_ORDER)
    result = result.sort_values(
        ["episode_start_utc", "episode_id", "_window_order", "release_time_utc", "calendar_row_id"],
        kind="mergesort",
    ).drop(columns="_window_order")
    leading = OFFICIAL_BRIDGE_COLUMNS
    trailing = [column for column in result.columns if column not in leading]
    return result[leading + trailing].reset_index(drop=True)
