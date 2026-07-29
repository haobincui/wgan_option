"""Persistent market-surface index and forward news alignment.

The index stores one surface per five-minute trailing-window anchor. A valid
forecast pair at origin ``o`` requires surfaces at ``o`` and ``o + 5min``.
News is matched only forward so no pre-publication surface can become a model
origin.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ISO_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
SCHEMA_VERSION = "2"


def to_utc_minute(value: Any) -> pd.Timestamp:
    """Normalize a datetime-like value to an aware UTC minute."""

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Cannot normalize missing timestamp: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("min")


def utc_minute_string(value: Any) -> str:
    return to_utc_minute(value).strftime(ISO_UTC_FORMAT)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def connect_market_index(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    initialize_market_index(connection)
    return connection


def initialize_market_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS input_file_state (
            source_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            scan_status TEXT NOT NULL,
            observed_minute_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS candidate_anchor (
            anchor_time_utc TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS calibration_batch (
            batch_id TEXT PRIMARY KEY,
            first_anchor_utc TEXT NOT NULL,
            last_anchor_utc TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            surface_count INTEGER NOT NULL DEFAULT 0,
            output_json TEXT NOT NULL DEFAULT '',
            precalib_csv TEXT NOT NULL DEFAULT '',
            message TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS surface_anchor (
            anchor_time_utc TEXT PRIMARY KEY,
            surface_model TEXT NOT NULL,
            params_json TEXT NOT NULL,
            params_sha256 TEXT NOT NULL,
            source_target_utc TEXT NOT NULL,
            source_direction TEXT NOT NULL,
            audit_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS precalib_point (
            anchor_time_utc TEXT NOT NULL,
            row_sha256 TEXT NOT NULL,
            row_json TEXT NOT NULL,
            PRIMARY KEY (anchor_time_utc, row_sha256),
            FOREIGN KEY (anchor_time_utc)
                REFERENCES surface_anchor(anchor_time_utc)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_precalib_anchor
            ON precalib_point(anchor_time_utc);

        CREATE TABLE IF NOT EXISTS valid_pair (
            origin_time_utc TEXT PRIMARY KEY,
            target_time_utc TEXT NOT NULL,
            current_params_sha256 TEXT NOT NULL,
            target_params_sha256 TEXT NOT NULL,
            FOREIGN KEY (origin_time_utc)
                REFERENCES surface_anchor(anchor_time_utc),
            FOREIGN KEY (target_time_utc)
                REFERENCES surface_anchor(anchor_time_utc)
        );
        """
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if existing is not None and str(existing["value"]) != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported market-index schema version: "
            f"{existing['value']} (expected {SCHEMA_VERSION})"
        )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    connection.commit()


def set_metadata(
    connection: sqlite3.Connection,
    key: str,
    value: Any,
    *,
    require_same: bool = False,
) -> None:
    serialized = canonical_json(value)
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (str(key),),
    ).fetchone()
    if require_same and existing is not None and str(existing["value"]) != serialized:
        raise ValueError(
            f"Market-index metadata mismatch for {key}: "
            f"existing={existing['value']}, requested={serialized}"
        )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
        (str(key), serialized),
    )
    connection.commit()


def candidate_anchors_from_minutes(
    observed_minutes: Iterable[Any],
    *,
    window_minutes: int = 5,
) -> list[str]:
    """Return every possible trailing-window anchor containing an observation."""

    if int(window_minutes) <= 0:
        raise ValueError("window_minutes must be positive")
    anchors: set[str] = set()
    for value in observed_minutes:
        minute = to_utc_minute(value)
        for offset in range(1, int(window_minutes) + 1):
            anchors.add(
                utc_minute_string(minute + pd.Timedelta(minutes=offset))
            )
    return sorted(anchors)


def insert_candidate_anchors(
    connection: sqlite3.Connection,
    anchors: Iterable[Any],
) -> int:
    normalized = [(utc_minute_string(anchor),) for anchor in anchors]
    before = int(
        connection.execute("SELECT COUNT(*) FROM candidate_anchor").fetchone()[0]
    )
    connection.executemany(
        "INSERT OR IGNORE INTO candidate_anchor(anchor_time_utc) VALUES(?)",
        normalized,
    )
    connection.commit()
    after = int(
        connection.execute("SELECT COUNT(*) FROM candidate_anchor").fetchone()[0]
    )
    return after - before


def insert_surface_anchor(
    connection: sqlite3.Connection,
    *,
    anchor_time_utc: Any,
    surface_model: str,
    surface_params: Mapping[str, Any],
    source_target_utc: Any,
    source_direction: str,
    surface_audit: Mapping[str, Any] | None = None,
) -> bool:
    anchor = utc_minute_string(anchor_time_utc)
    params_json = canonical_json(surface_params)
    params_sha256 = sha256_text(params_json)
    existing = connection.execute(
        "SELECT params_sha256 FROM surface_anchor WHERE anchor_time_utc = ?",
        (anchor,),
    ).fetchone()
    if existing is not None:
        if str(existing["params_sha256"]) != params_sha256:
            raise ValueError(
                "Conflicting surface parameters for anchor "
                f"{anchor}: {existing['params_sha256']} != {params_sha256}"
            )
        return False
    connection.execute(
        """
        INSERT INTO surface_anchor(
            anchor_time_utc,
            surface_model,
            params_json,
            params_sha256,
            source_target_utc,
            source_direction,
            audit_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            anchor,
            str(surface_model),
            params_json,
            params_sha256,
            utc_minute_string(source_target_utc),
            str(source_direction),
            canonical_json(surface_audit or {}),
        ),
    )
    return True


def insert_precalib_rows(
    connection: sqlite3.Connection,
    *,
    anchor_time_utc: Any,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    anchor = utc_minute_string(anchor_time_utc)
    payloads: list[tuple[str, str, str]] = []
    for row in rows:
        row_json = canonical_json(dict(row))
        payloads.append((anchor, sha256_text(row_json), row_json))
    before = int(
        connection.execute(
            "SELECT COUNT(*) FROM precalib_point WHERE anchor_time_utc = ?",
            (anchor,),
        ).fetchone()[0]
    )
    connection.executemany(
        """
        INSERT OR IGNORE INTO precalib_point(
            anchor_time_utc,
            row_sha256,
            row_json
        ) VALUES (?, ?, ?)
        """,
        payloads,
    )
    after = int(
        connection.execute(
            "SELECT COUNT(*) FROM precalib_point WHERE anchor_time_utc = ?",
            (anchor,),
        ).fetchone()[0]
    )
    return after - before


def refresh_valid_pairs(
    connection: sqlite3.Connection,
    *,
    horizon_minutes: int = 5,
) -> int:
    if int(horizon_minutes) <= 0:
        raise ValueError("horizon_minutes must be positive")
    anchors = connection.execute(
        """
        SELECT anchor_time_utc, params_sha256
        FROM surface_anchor
        ORDER BY anchor_time_utc
        """
    ).fetchall()
    sha_by_anchor = {
        str(row["anchor_time_utc"]): str(row["params_sha256"])
        for row in anchors
    }
    connection.execute("DELETE FROM valid_pair")
    pair_rows: list[tuple[str, str, str, str]] = []
    for origin, current_sha in sha_by_anchor.items():
        target = utc_minute_string(
            to_utc_minute(origin)
            + pd.Timedelta(minutes=int(horizon_minutes))
        )
        target_sha = sha_by_anchor.get(target)
        if target_sha is None:
            continue
        pair_rows.append((origin, target, current_sha, target_sha))
    connection.executemany(
        """
        INSERT INTO valid_pair(
            origin_time_utc,
            target_time_utc,
            current_params_sha256,
            target_params_sha256
        ) VALUES (?, ?, ?, ?)
        """,
        pair_rows,
    )
    connection.commit()
    return len(pair_rows)


def fetch_valid_pair_origins(connection: sqlite3.Connection) -> list[pd.Timestamp]:
    rows = connection.execute(
        "SELECT origin_time_utc FROM valid_pair ORDER BY origin_time_utc"
    ).fetchall()
    return [to_utc_minute(row["origin_time_utc"]) for row in rows]


@dataclass(frozen=True)
class ForwardAlignmentPolicy:
    intraday_tolerance_minutes: int = 15
    max_session_shift_minutes: int = 4320
    horizon_minutes: int = 5
    include_session_shifted: bool = True

    def __post_init__(self) -> None:
        if self.intraday_tolerance_minutes < 0:
            raise ValueError("intraday_tolerance_minutes must be non-negative")
        if self.max_session_shift_minutes < self.intraday_tolerance_minutes:
            raise ValueError(
                "max_session_shift_minutes must be >= intraday_tolerance_minutes"
            )
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")


ALIGNMENT_COLUMNS = [
    "news_row_id",
    "sample_id",
    "publication_timestamp_utc",
    "news_available_time_utc",
    "timestamp_parse_status",
    "has_match",
    "effective_origin_utc",
    "target_anchor_utc",
    "origin_shift_minutes",
    "alignment_type",
    "matching_rank",
    "collision_count",
    "news_cluster_id",
    "quiet_buffer_minutes",
    "quiet_grid_minutes",
    "current_window_start_utc",
    "current_window_end_utc",
    "target_window_start_utc",
    "target_window_end_utc",
    "original_news_quarter",
    "effective_origin_quarter",
    "unmatched_reason",
]


def _quarter(value: pd.Timestamp) -> str:
    return str(value.tz_localize(None).to_period("Q"))


def align_news_to_valid_pairs(
    news_frame: pd.DataFrame,
    valid_pair_origins: Sequence[Any],
    *,
    policy: ForwardAlignmentPolicy | None = None,
    available_time_column: str = "timestamp_utc",
) -> pd.DataFrame:
    """Match each news row to the earliest valid origin at or after availability."""

    policy = policy or ForwardAlignmentPolicy()
    required = {"news_row_id", available_time_column}
    missing = sorted(required - set(news_frame.columns))
    if missing:
        raise ValueError(f"News frame is missing required columns: {missing}")

    origins = sorted({to_utc_minute(value) for value in valid_pair_origins})
    origin_ns = [int(value.value) for value in origins]
    rows: list[dict[str, Any]] = []

    for news_row in news_frame.to_dict(orient="records"):
        news_row_id = int(news_row["news_row_id"])
        raw_available = news_row.get(available_time_column)
        available = pd.to_datetime(raw_available, errors="coerce", utc=True)
        parse_status = str(news_row.get("timestamp_parse_status", "") or "")
        base: dict[str, Any] = {
            "news_row_id": news_row_id,
            "sample_id": f"news_{news_row_id}",
            "publication_timestamp_utc": str(
                news_row.get("publication_timestamp_utc", "") or ""
            ),
            "news_available_time_utc": (
                utc_minute_string(available) if not pd.isna(available) else ""
            ),
            "timestamp_parse_status": parse_status,
            "has_match": 0,
            "effective_origin_utc": "",
            "target_anchor_utc": "",
            "origin_shift_minutes": "",
            "alignment_type": "unmatched",
            "matching_rank": "",
            "collision_count": 0,
            "news_cluster_id": "",
            "quiet_buffer_minutes": "",
            "quiet_grid_minutes": "",
            "current_window_start_utc": "",
            "current_window_end_utc": "",
            "target_window_start_utc": "",
            "target_window_end_utc": "",
            "original_news_quarter": "",
            "effective_origin_quarter": "",
            "unmatched_reason": "",
        }
        if pd.isna(available):
            base["unmatched_reason"] = "invalid_news_timestamp"
            rows.append(base)
            continue

        available = to_utc_minute(available)
        base["original_news_quarter"] = _quarter(available)
        position = bisect.bisect_left(origin_ns, int(available.value))
        if position >= len(origins):
            base["unmatched_reason"] = "no_later_valid_pair"
            rows.append(base)
            continue

        origin = origins[position]
        shift_minutes = int((origin - available).total_seconds() // 60)
        if shift_minutes < 0:
            raise AssertionError("Forward alignment selected a pre-news origin")
        if shift_minutes > policy.max_session_shift_minutes:
            base["unmatched_reason"] = "no_valid_pair_within_max_shift"
            rows.append(base)
            continue
        if shift_minutes == 0:
            alignment_type = "exact"
        elif shift_minutes <= policy.intraday_tolerance_minutes:
            alignment_type = "intraday_shift"
        else:
            alignment_type = "session_shift"
        if alignment_type == "session_shift" and not policy.include_session_shifted:
            base["unmatched_reason"] = "session_shift_disabled"
            rows.append(base)
            continue

        target = origin + pd.Timedelta(minutes=policy.horizon_minutes)
        current_start = origin - pd.Timedelta(minutes=policy.horizon_minutes)
        base.update(
            {
                "has_match": 1,
                "effective_origin_utc": utc_minute_string(origin),
                "target_anchor_utc": utc_minute_string(target),
                "origin_shift_minutes": shift_minutes,
                "alignment_type": alignment_type,
                "matching_rank": 1,
                "news_cluster_id": (
                    f"surface_pair_{origin.strftime('%Y%m%dT%H%MZ')}"
                ),
                "current_window_start_utc": utc_minute_string(current_start),
                "current_window_end_utc": utc_minute_string(origin),
                "target_window_start_utc": utc_minute_string(origin),
                "target_window_end_utc": utc_minute_string(target),
                "effective_origin_quarter": _quarter(origin),
            }
        )
        rows.append(base)

    aligned = pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)
    matched = aligned["has_match"].eq(1)
    if matched.any():
        collision_counts = (
            aligned.loc[matched, "effective_origin_utc"]
            .value_counts()
            .to_dict()
        )
        aligned.loc[matched, "collision_count"] = aligned.loc[
            matched, "effective_origin_utc"
        ].map(collision_counts)
    validate_forward_alignment(aligned, policy=policy)
    return aligned


def validate_forward_alignment(
    alignment: pd.DataFrame,
    *,
    policy: ForwardAlignmentPolicy | None = None,
) -> None:
    policy = policy or ForwardAlignmentPolicy()
    missing = sorted(set(ALIGNMENT_COLUMNS) - set(alignment.columns))
    if missing:
        raise ValueError(f"Alignment frame is missing columns: {missing}")
    if alignment["news_row_id"].duplicated().any():
        duplicates = alignment.loc[
            alignment["news_row_id"].duplicated(keep=False),
            "news_row_id",
        ].tolist()
        raise ValueError(f"Duplicate news_row_id values in alignment: {duplicates[:10]}")

    matched = alignment[alignment["has_match"].eq(1)].copy()
    if matched.empty:
        return
    available = pd.to_datetime(
        matched["news_available_time_utc"],
        utc=True,
        errors="raise",
    )
    origin = pd.to_datetime(
        matched["effective_origin_utc"],
        utc=True,
        errors="raise",
    )
    target = pd.to_datetime(
        matched["target_anchor_utc"],
        utc=True,
        errors="raise",
    )
    shift = (origin - available).dt.total_seconds().div(60.0)
    if shift.lt(0).any():
        raise ValueError("Alignment contains an origin before news availability")
    if shift.gt(policy.max_session_shift_minutes).any():
        raise ValueError("Alignment exceeds max_session_shift_minutes")
    if not target.sub(origin).eq(
        pd.Timedelta(minutes=policy.horizon_minutes)
    ).all():
        raise ValueError("Alignment target horizon is not fixed at five minutes")

    current_start = pd.to_datetime(
        matched["current_window_start_utc"],
        utc=True,
        errors="raise",
    )
    current_end = pd.to_datetime(
        matched["current_window_end_utc"],
        utc=True,
        errors="raise",
    )
    target_start = pd.to_datetime(
        matched["target_window_start_utc"],
        utc=True,
        errors="raise",
    )
    target_end = pd.to_datetime(
        matched["target_window_end_utc"],
        utc=True,
        errors="raise",
    )
    horizon = pd.Timedelta(minutes=policy.horizon_minutes)
    if not (
        current_end.eq(origin).all()
        and target_start.eq(origin).all()
        and current_end.eq(target_start).all()
        and current_end.sub(current_start).eq(horizon).all()
        and target_end.sub(target_start).eq(horizon).all()
    ):
        raise ValueError("Alignment contains invalid or overlapping window boundaries")


def alignment_summary(alignment: pd.DataFrame) -> dict[str, Any]:
    matched = alignment[alignment["has_match"].eq(1)].copy()
    shifts = pd.to_numeric(
        matched.get("origin_shift_minutes", pd.Series(dtype=float)),
        errors="coerce",
    )
    return {
        "news_rows": int(len(alignment)),
        "matched_article_rows": int(len(matched)),
        "unmatched_article_rows": int(len(alignment) - len(matched)),
        "unique_surface_pairs": int(
            matched["effective_origin_utc"].nunique()
            if not matched.empty
            else 0
        ),
        "alignment_type_counts": {
            str(key): int(value)
            for key, value in alignment["alignment_type"]
            .value_counts(dropna=False)
            .items()
        },
        "unmatched_reason_counts": {
            str(key): int(value)
            for key, value in alignment.loc[
                alignment["has_match"].ne(1),
                "unmatched_reason",
            ]
            .value_counts(dropna=False)
            .items()
        },
        "shift_minutes_min": (
            int(shifts.min()) if not shifts.empty else None
        ),
        "shift_minutes_median": (
            float(shifts.median()) if not shifts.empty else None
        ),
        "shift_minutes_max": (
            int(shifts.max()) if not shifts.empty else None
        ),
        "collision_pair_count": int(
            matched.loc[matched["collision_count"].gt(1), "effective_origin_utc"]
            .nunique()
            if not matched.empty
            else 0
        ),
    }


def find_unmatched_with_eligible_pair(
    alignment: pd.DataFrame,
    valid_pair_origins: Sequence[Any],
    *,
    max_shift_minutes: int,
) -> list[int]:
    """Return unmatched news IDs that actually had an eligible forward pair."""

    origins = sorted({to_utc_minute(value) for value in valid_pair_origins})
    origin_ns = [int(value.value) for value in origins]
    violations: list[int] = []
    for row in alignment.loc[alignment["has_match"].ne(1)].itertuples(
        index=False
    ):
        available = pd.to_datetime(
            row.news_available_time_utc,
            utc=True,
            errors="coerce",
        )
        if pd.isna(available):
            continue
        available = to_utc_minute(available)
        position = bisect.bisect_left(origin_ns, int(available.value))
        if position >= len(origins):
            continue
        shift = int((origins[position] - available).total_seconds() // 60)
        if 0 <= shift <= int(max_shift_minutes):
            violations.append(int(row.news_row_id))
    return violations
