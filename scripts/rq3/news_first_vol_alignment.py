"""News-first sensitivity alignment against the complete market-pair universe.

This module deliberately does not use :class:`SessionAlignmentPolicy`.  That
policy freezes the chapter's primary five-minute alignment, while this module
builds cumulative 5/10/15/20/30-minute sensitivity samples.  Pair selection is
based only on time and CME-session continuity; downstream surface or ATM
quality must never influence which pair is selected.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from wgan_option.market.treasury_sessions import TreasuryGlobexSessionCalendar
from wgan_option.surface_generation.market_index import (
    to_utc_minute,
    utc_minute_string,
)


DEFAULT_TOLERANCES_MINUTES: tuple[int, ...] = (5, 10, 15, 20, 30)

PAIR_UNIVERSE_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "origin_time_utc",
    "target_time_utc",
    "session_id",
    "session_open_utc",
    "session_close_utc",
)

ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "news_row_id",
    "sample_id",
    "news_alignment_mode",
    "publication_timestamp_utc",
    "news_available_time_utc",
    "timestamp_parse_status",
    "tolerance_minutes",
    "has_candidate",
    "has_match",
    "training_eligible",
    "training_exclusion_reason",
    "publication_market_state",
    "scheduled_origin_utc",
    "pair_id",
    "effective_origin_utc",
    "target_anchor_utc",
    "origin_shift_minutes",
    "origin_tolerance_minutes_used",
    "first_included_tolerance_minutes",
    "alignment_type",
    "matching_rank",
    "collision_count",
    "training_pair_article_count",
    "session_shift_minutes",
    "session_shift_reason",
    "session_id",
    "session_open_utc",
    "session_close_utc",
    "current_window_start_utc",
    "current_window_end_utc",
    "target_window_start_utc",
    "target_window_end_utc",
    "window_relation",
    "post_news_current_overlap_minutes",
    "unmatched_reason",
)


def _normalise_tolerances(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("At least one alignment tolerance is required")
    tolerances: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            raise ValueError("Alignment tolerances must be positive integers")
        value = int(raw_value)
        if value != raw_value or value <= 0:
            raise ValueError("Alignment tolerances must be positive integers")
        tolerances.append(value)
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("Alignment tolerances must be unique")
    return tuple(sorted(tolerances))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
    if pd.isna(value):
        return False
    return bool(value)


def _window_relation(
    available: pd.Timestamp,
    current_start: pd.Timestamp,
    current_end: pd.Timestamp,
) -> tuple[str, int]:
    """Describe leakage of the backward-looking current window past news."""

    overlap = int(
        max(
            0.0,
            min(
                (current_end - available).total_seconds() / 60.0,
                (current_end - current_start).total_seconds() / 60.0,
            ),
        )
    )
    if current_end <= available:
        relation = "current_pre_news_target_post_news"
    elif current_start < available:
        relation = "current_partially_post_news"
    else:
        relation = "current_fully_post_news"
    return relation, overlap


def prepare_pair_universe(
    pair_slice_metrics: pd.DataFrame,
    *,
    session_calendar: TreasuryGlobexSessionCalendar,
    horizon_minutes: int = 5,
    current_window_minutes: int = 5,
) -> pd.DataFrame:
    """Collapse maturity-level metrics to one validated row per market pair.

    No metric-quality field is inspected.  All same-session pairs remain
    eligible, including pairs whose maturity rows later fail ATM or surface
    quality checks.
    """

    if int(horizon_minutes) <= 0:
        raise ValueError("horizon_minutes must be positive")
    if int(current_window_minutes) <= 0:
        raise ValueError("current_window_minutes must be positive")
    required = {
        "pair_id",
        "origin_time_utc",
        "target_time_utc",
        "session_id",
        "same_cme_continuous_session",
    }
    missing = sorted(required - set(pair_slice_metrics.columns))
    if missing:
        raise ValueError(f"Pair metrics are missing required columns: {missing}")

    work = pair_slice_metrics.loc[:, sorted(required)].copy()
    if work.empty:
        return pd.DataFrame(columns=PAIR_UNIVERSE_COLUMNS)
    work["pair_id"] = work["pair_id"].fillna("").astype(str).str.strip()
    work["session_id"] = work["session_id"].fillna("").astype(str).str.strip()
    if work["pair_id"].eq("").any():
        raise ValueError("Pair metrics contain a missing pair_id")
    work["same_cme_continuous_session"] = work["same_cme_continuous_session"].map(
        _as_bool
    )
    work["origin_time_utc"] = pd.to_datetime(
        work["origin_time_utc"], utc=True, errors="raise"
    ).map(to_utc_minute)
    work["target_time_utc"] = pd.to_datetime(
        work["target_time_utc"], utc=True, errors="raise"
    ).map(to_utc_minute)

    consistency_columns = (
        "origin_time_utc",
        "target_time_utc",
        "session_id",
        "same_cme_continuous_session",
    )
    grouped = work.groupby("pair_id", sort=False, dropna=False)
    inconsistent: list[str] = []
    for column in consistency_columns:
        counts = grouped[column].nunique(dropna=False)
        inconsistent.extend(counts.index[counts.ne(1)].astype(str).tolist())
    if inconsistent:
        examples = sorted(set(inconsistent))[:10]
        raise ValueError(
            "Pair metrics map a pair_id to inconsistent time/session values: "
            f"{examples}"
        )

    universe = work.drop_duplicates("pair_id", keep="first").copy()
    universe = universe.loc[universe["same_cme_continuous_session"]].copy()
    if universe["origin_time_utc"].duplicated().any():
        duplicated = universe.loc[
            universe["origin_time_utc"].duplicated(keep=False), "origin_time_utc"
        ].astype(str)
        raise ValueError(
            "Pair universe contains multiple pair_id values for one origin: "
            f"{duplicated.head(10).tolist()}"
        )

    rows: list[dict[str, Any]] = []
    expected_horizon = pd.Timedelta(minutes=int(horizon_minutes))
    current_width = pd.Timedelta(minutes=int(current_window_minutes))
    for pair in universe.itertuples(index=False):
        origin = to_utc_minute(pair.origin_time_utc)
        target = to_utc_minute(pair.target_time_utc)
        if target - origin != expected_horizon:
            raise ValueError(
                f"Pair {pair.pair_id} does not have the required "
                f"{horizon_minutes}-minute horizon"
            )
        session = session_calendar.session_for_interval(origin - current_width, target)
        if session is None:
            raise ValueError(
                f"Pair {pair.pair_id} is marked same-session but its full window "
                "crosses a CME session boundary"
            )
        if str(pair.session_id) != session.session_id:
            raise ValueError(
                f"Pair {pair.pair_id} has inconsistent session_id "
                f"{pair.session_id!r}; expected {session.session_id!r}"
            )
        rows.append(
            {
                "pair_id": str(pair.pair_id),
                "origin_time_utc": origin,
                "target_time_utc": target,
                "session_id": session.session_id,
                "session_open_utc": session.open_utc,
                "session_close_utc": session.close_utc,
            }
        )
    return pd.DataFrame(rows, columns=PAIR_UNIVERSE_COLUMNS).sort_values(
        "origin_time_utc", kind="stable", ignore_index=True
    )


def _empty_alignment_base(
    news_row: Mapping[str, Any],
    *,
    available: pd.Timestamp | None,
    tolerance_minutes: int,
) -> dict[str, Any]:
    news_row_id = int(news_row["news_row_id"])
    sample_id = str(news_row.get("sample_id", "") or f"news_{news_row_id}")
    publication = str(news_row.get("publication_timestamp_utc", "") or "")
    parse_status = str(news_row.get("timestamp_parse_status", "") or "")
    return {
        "news_row_id": news_row_id,
        "sample_id": sample_id,
        "news_alignment_mode": "news_first_forward_sensitivity",
        "publication_timestamp_utc": publication,
        "news_available_time_utc": (
            utc_minute_string(available) if available is not None else ""
        ),
        "timestamp_parse_status": parse_status,
        "tolerance_minutes": int(tolerance_minutes),
        "has_candidate": 0,
        "has_match": 0,
        "training_eligible": 0,
        "training_exclusion_reason": "unmatched",
        "publication_market_state": "",
        "scheduled_origin_utc": "",
        "pair_id": "",
        "effective_origin_utc": "",
        "target_anchor_utc": "",
        "origin_shift_minutes": pd.NA,
        "origin_tolerance_minutes_used": pd.NA,
        "first_included_tolerance_minutes": pd.NA,
        "alignment_type": "unmatched",
        "matching_rank": pd.NA,
        "collision_count": 0,
        "training_pair_article_count": 0,
        "session_shift_minutes": pd.NA,
        "session_shift_reason": "",
        "session_id": "",
        "session_open_utc": "",
        "session_close_utc": "",
        "current_window_start_utc": "",
        "current_window_end_utc": "",
        "target_window_start_utc": "",
        "target_window_end_utc": "",
        "window_relation": "",
        "post_news_current_overlap_minutes": pd.NA,
        "unmatched_reason": "",
    }


def build_news_first_alignments(
    news_frame: pd.DataFrame,
    pair_slice_metrics: pd.DataFrame,
    *,
    session_calendar: TreasuryGlobexSessionCalendar,
    tolerances: Sequence[int] = DEFAULT_TOLERANCES_MINUTES,
    available_time_column: str = "news_available_time_utc",
    horizon_minutes: int = 5,
    current_window_minutes: int = 5,
) -> dict[int, pd.DataFrame]:
    """Build nested news-first alignments at cumulative tolerance thresholds.

    Open-market news is training eligible when a candidate exists.  Closed
    news is still aligned from its next CME open for audit purposes, but is
    always marked ``training_eligible=0``.
    """

    normalized_tolerances = _normalise_tolerances(tolerances)
    required_news = {"news_row_id", available_time_column}
    missing_news = sorted(required_news - set(news_frame.columns))
    if missing_news:
        raise ValueError(f"News frame is missing required columns: {missing_news}")
    if news_frame["news_row_id"].isna().any():
        raise ValueError("News frame contains a missing news_row_id")
    if news_frame["news_row_id"].duplicated().any():
        duplicates = news_frame.loc[
            news_frame["news_row_id"].duplicated(keep=False), "news_row_id"
        ].tolist()
        raise ValueError(
            f"News frame contains duplicate news_row_id values: {duplicates[:10]}"
        )
    if int(horizon_minutes) <= 0:
        raise ValueError("horizon_minutes must be positive")
    if int(current_window_minutes) <= 0:
        raise ValueError("current_window_minutes must be positive")

    universe = prepare_pair_universe(
        pair_slice_metrics,
        session_calendar=session_calendar,
        horizon_minutes=horizon_minutes,
        current_window_minutes=current_window_minutes,
    )
    pair_records = universe.to_dict(orient="records")
    origins = universe["origin_time_utc"].tolist()
    origin_ns = [int(timestamp.value) for timestamp in origins]
    max_tolerance = normalized_tolerances[-1]
    selected_rows: list[dict[str, Any]] = []

    for news_row in news_frame.to_dict(orient="records"):
        raw_available = news_row.get(available_time_column)
        parsed = pd.to_datetime(raw_available, utc=True, errors="coerce")
        if pd.isna(parsed):
            selected_rows.append(
                {
                    "news_row": news_row,
                    "available": None,
                    "unmatched_reason": "invalid_news_timestamp",
                }
            )
            continue
        available = to_utc_minute(parsed)
        if session_calendar.is_open(available):
            market_state = "open"
            scheduled_origin = available
            session_shift_reason = "none"
        else:
            market_state = "closed"
            scheduled_origin = session_calendar.next_open(available)
            session_shift_reason = session_calendar.closed_reason(available)
        expected_session = session_calendar.containing_session(scheduled_origin)
        if expected_session is None:
            raise AssertionError("Scheduled origin must lie inside a CME session")

        position = bisect.bisect_left(origin_ns, int(scheduled_origin.value))
        deadline = scheduled_origin + pd.Timedelta(minutes=max_tolerance)
        selected_pair: dict[str, Any] | None = None
        matching_rank = 0
        saw_other_session = False
        scan_position = position
        while scan_position < len(pair_records):
            candidate = pair_records[scan_position]
            candidate_origin = candidate["origin_time_utc"]
            if candidate_origin > deadline:
                break
            matching_rank += 1
            if candidate["session_id"] == expected_session.session_id:
                selected_pair = candidate
                break
            saw_other_session = True
            scan_position += 1

        selected_rows.append(
            {
                "news_row": news_row,
                "available": available,
                "publication_market_state": market_state,
                "scheduled_origin": scheduled_origin,
                "session_shift_reason": session_shift_reason,
                "expected_session": expected_session,
                "selected_pair": selected_pair,
                "matching_rank": matching_rank,
                "unmatched_reason": (
                    "no_same_session_pair_within_origin_tolerance"
                    if selected_pair is None and saw_other_session
                    else (
                        "no_valid_pair_within_origin_tolerance"
                        if selected_pair is None
                        else ""
                    )
                ),
            }
        )

    results: dict[int, pd.DataFrame] = {}
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    current_width = pd.Timedelta(minutes=int(current_window_minutes))
    for tolerance in normalized_tolerances:
        rows: list[dict[str, Any]] = []
        for selection in selected_rows:
            news_row = selection["news_row"]
            available = selection["available"]
            base = _empty_alignment_base(
                news_row,
                available=available,
                tolerance_minutes=tolerance,
            )
            if available is None:
                base["unmatched_reason"] = "invalid_news_timestamp"
                rows.append(base)
                continue

            market_state = str(selection["publication_market_state"])
            scheduled_origin = selection["scheduled_origin"]
            base.update(
                {
                    "publication_market_state": market_state,
                    "scheduled_origin_utc": utc_minute_string(scheduled_origin),
                    "session_shift_minutes": int(
                        (scheduled_origin - available).total_seconds() // 60
                    ),
                    "session_shift_reason": selection["session_shift_reason"],
                }
            )
            pair = selection["selected_pair"]
            if pair is None:
                base["unmatched_reason"] = selection["unmatched_reason"]
                rows.append(base)
                continue

            origin = pair["origin_time_utc"]
            target = pair["target_time_utc"]
            tolerance_used = int((origin - scheduled_origin).total_seconds() // 60)
            first_tolerance = next(
                item for item in normalized_tolerances if tolerance_used <= item
            )
            # Keep the cohort-entry threshold visible even in a narrower
            # workbook where this news is currently unmatched.  Pair fields
            # remain blank until the threshold is actually reached.
            base["first_included_tolerance_minutes"] = first_tolerance
            if tolerance_used > tolerance:
                base["unmatched_reason"] = "no_valid_pair_within_origin_tolerance"
                rows.append(base)
                continue

            current_start = origin - current_width
            relation, overlap = _window_relation(available, current_start, origin)
            if market_state == "closed":
                alignment_type = "closed_to_next_open"
                training_eligible = 0
                training_exclusion = "closed_market_publication"
            elif tolerance_used == 0:
                alignment_type = "exact"
                training_eligible = 1
                training_exclusion = ""
            else:
                alignment_type = "market_open_tolerance_shift"
                training_eligible = 1
                training_exclusion = ""
            base.update(
                {
                    "has_candidate": 1,
                    "has_match": 1,
                    "training_eligible": training_eligible,
                    "training_exclusion_reason": training_exclusion,
                    "pair_id": pair["pair_id"],
                    "effective_origin_utc": utc_minute_string(origin),
                    "target_anchor_utc": utc_minute_string(target),
                    "origin_shift_minutes": int(
                        (origin - available).total_seconds() // 60
                    ),
                    "origin_tolerance_minutes_used": tolerance_used,
                    "first_included_tolerance_minutes": first_tolerance,
                    "alignment_type": alignment_type,
                    "matching_rank": int(selection["matching_rank"]),
                    "session_id": pair["session_id"],
                    "session_open_utc": utc_minute_string(pair["session_open_utc"]),
                    "session_close_utc": utc_minute_string(pair["session_close_utc"]),
                    "current_window_start_utc": utc_minute_string(current_start),
                    "current_window_end_utc": utc_minute_string(origin),
                    "target_window_start_utc": utc_minute_string(origin),
                    "target_window_end_utc": utc_minute_string(origin + horizon),
                    "window_relation": relation,
                    "post_news_current_overlap_minutes": overlap,
                    "unmatched_reason": "",
                }
            )
            rows.append(base)

        aligned = pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)
        candidate_mask = aligned["has_candidate"].eq(1)
        if candidate_mask.any():
            collisions = aligned.loc[candidate_mask, "pair_id"].value_counts()
            aligned.loc[candidate_mask, "collision_count"] = (
                aligned.loc[candidate_mask, "pair_id"].map(collisions).astype(int)
            )
        eligible_mask = aligned["training_eligible"].eq(1)
        if eligible_mask.any():
            article_counts = aligned.loc[eligible_mask, "pair_id"].value_counts()
            aligned.loc[eligible_mask, "training_pair_article_count"] = (
                aligned.loc[eligible_mask, "pair_id"].map(article_counts).astype(int)
            )
        results[int(tolerance)] = aligned

    validate_news_first_alignments(
        results,
        session_calendar=session_calendar,
        tolerances=normalized_tolerances,
        horizon_minutes=horizon_minutes,
        current_window_minutes=current_window_minutes,
    )
    return results


def validate_news_first_alignments(
    alignments: Mapping[int, pd.DataFrame],
    *,
    session_calendar: TreasuryGlobexSessionCalendar,
    tolerances: Sequence[int] | None = None,
    horizon_minutes: int = 5,
    current_window_minutes: int = 5,
) -> None:
    """Validate row grain, temporal invariants, session safety, and nesting."""

    expected_tolerances = _normalise_tolerances(
        tuple(alignments) if tolerances is None else tolerances
    )
    if set(alignments) != set(expected_tolerances):
        raise ValueError(
            "Alignment mapping keys do not match configured tolerances: "
            f"expected {expected_tolerances}, got {sorted(alignments)}"
        )
    reference_ids: list[Any] | None = None
    prior_matched: set[Any] = set()
    prior_selection: dict[Any, tuple[str, str, str]] = {}
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    current_width = pd.Timedelta(minutes=int(current_window_minutes))

    for tolerance in expected_tolerances:
        frame = alignments[tolerance]
        missing = sorted(set(ALIGNMENT_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{tolerance}-minute alignment is missing columns: {missing}"
            )
        if frame["news_row_id"].duplicated().any():
            raise ValueError(
                f"{tolerance}-minute alignment contains duplicate news_row_id values"
            )
        ids = frame["news_row_id"].tolist()
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("Alignment row grain/order differs across tolerances")
        if not frame["tolerance_minutes"].eq(tolerance).all():
            raise ValueError("Alignment rows contain the wrong tolerance label")

        matched = frame.loc[frame["has_candidate"].eq(1)].copy()
        if not matched["has_match"].eq(1).all():
            raise ValueError("has_candidate and has_match are inconsistent")
        unmatched = frame.loc[frame["has_candidate"].eq(0)]
        if not unmatched["has_match"].eq(0).all():
            raise ValueError("has_candidate and has_match are inconsistent")
        current_ids = set(matched["news_row_id"].tolist())
        if not prior_matched.issubset(current_ids):
            raise ValueError("News-first alignment is not nested across tolerances")

        if not matched.empty:
            available = pd.to_datetime(
                matched["news_available_time_utc"], utc=True, errors="raise"
            )
            scheduled = pd.to_datetime(
                matched["scheduled_origin_utc"], utc=True, errors="raise"
            )
            origin = pd.to_datetime(
                matched["effective_origin_utc"], utc=True, errors="raise"
            )
            target = pd.to_datetime(
                matched["target_anchor_utc"], utc=True, errors="raise"
            )
            if scheduled.lt(available).any() or origin.lt(scheduled).any():
                raise ValueError("News-first alignment contains a pre-news origin")
            if not target.sub(origin).eq(horizon).all():
                raise ValueError("News-first alignment target horizon is invalid")
            tolerance_used = origin.sub(scheduled).dt.total_seconds().div(60)
            if tolerance_used.lt(0).any() or tolerance_used.gt(tolerance).any():
                raise ValueError("News-first alignment exceeds its inclusive tolerance")
            if (
                not matched["origin_tolerance_minutes_used"]
                .astype(int)
                .eq(tolerance_used.astype(int))
                .all()
            ):
                raise ValueError("Stored origin tolerance is inconsistent")

            current_start = pd.to_datetime(
                matched["current_window_start_utc"], utc=True, errors="raise"
            )
            current_end = pd.to_datetime(
                matched["current_window_end_utc"], utc=True, errors="raise"
            )
            target_start = pd.to_datetime(
                matched["target_window_start_utc"], utc=True, errors="raise"
            )
            target_end = pd.to_datetime(
                matched["target_window_end_utc"], utc=True, errors="raise"
            )
            if not (
                current_end.eq(origin).all()
                and target_start.eq(origin).all()
                and target_end.eq(target).all()
                and current_end.sub(current_start).eq(current_width).all()
            ):
                raise ValueError("News-first alignment window boundaries are invalid")

            for row in matched.itertuples(index=False):
                interval_session = session_calendar.session_for_interval(
                    row.current_window_start_utc, row.target_window_end_utc
                )
                if (
                    interval_session is None
                    or interval_session.session_id != row.session_id
                ):
                    raise ValueError("Matched pair is not in one stored CME session")
                state = str(row.publication_market_state)
                is_open = session_calendar.is_open(row.news_available_time_utc)
                if state == "open":
                    if not is_open or int(row.training_eligible) != 1:
                        raise ValueError(
                            "Open matched news has invalid training eligibility"
                        )
                    publication_session = session_calendar.containing_session(
                        row.news_available_time_utc
                    )
                    if publication_session is None or (
                        publication_session.session_id != row.session_id
                    ):
                        raise ValueError(
                            "Open news and selected pair are not in one session"
                        )
                elif state == "closed":
                    if is_open or int(row.training_eligible) != 0:
                        raise ValueError("Closed news must remain audit-only")
                    if to_utc_minute(row.scheduled_origin_utc) != (
                        session_calendar.next_open(row.news_available_time_utc)
                    ):
                        raise ValueError(
                            "Closed news did not schedule from next CME open"
                        )
                else:
                    raise ValueError(f"Unknown publication market state: {state!r}")

                expected_relation, expected_overlap = _window_relation(
                    to_utc_minute(row.news_available_time_utc),
                    to_utc_minute(row.current_window_start_utc),
                    to_utc_minute(row.current_window_end_utc),
                )
                if (
                    row.window_relation != expected_relation
                    or int(row.post_news_current_overlap_minutes) != expected_overlap
                ):
                    raise ValueError("Stored news/window relation is inconsistent")
                expected_first_tolerance = next(
                    item
                    for item in expected_tolerances
                    if int(row.origin_tolerance_minutes_used) <= item
                )
                if (
                    int(row.first_included_tolerance_minutes)
                    != expected_first_tolerance
                ):
                    raise ValueError("Stored first-included tolerance is inconsistent")

                previous = prior_selection.get(row.news_row_id)
                selection = (
                    str(row.pair_id),
                    str(row.effective_origin_utc),
                    str(row.target_anchor_utc),
                )
                if previous is not None and previous != selection:
                    raise ValueError(
                        "A lower-tolerance match changed pair at a higher tolerance"
                    )
                prior_selection[row.news_row_id] = selection

        prior_matched = current_ids


__all__ = [
    "ALIGNMENT_COLUMNS",
    "DEFAULT_TOLERANCES_MINUTES",
    "PAIR_UNIVERSE_COLUMNS",
    "build_news_first_alignments",
    "prepare_pair_universe",
    "validate_news_first_alignments",
]
