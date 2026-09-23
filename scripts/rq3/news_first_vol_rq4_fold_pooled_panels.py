"""Freeze the RQ4 special-time, fold-pooled evaluation panels.

The panels produced here are inference inputs, not new training splits.  For
each rolling checkpoint fold, the existing ``train``, ``validation`` and
``test`` pair universes are pooled while ``source_split`` is retained.  A pair
is admitted when its effective origin is either in the inclusive scheduled
release window ``[0, +30]`` minutes or exactly matches a frozen market-jump
origin.

This module deliberately owns no model or prediction logic.  It freezes one
canonical workbook row per pair, carries every original ``merged_vol`` column,
adds event/support lineage, and writes a signed, fail-closed artifact bundle.
Existing bundles are validated rather than overwritten.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class FoldPooledPanelError(RuntimeError):
    """Raised when an RQ4 fold-pooled panel contract is violated."""


SCHEMA_VERSION = 1
MANIFEST_KIND = "rq4_fold_pooled_panel_manifest_v1"
TOLERANCE_MINUTES = 30
SOURCE_5M_TOLERANCE_MINUTES = 5
FORECAST_HORIZON_MINUTES = 5
SCHEDULED_LOWER_MINUTES = 0
SCHEDULED_UPPER_MINUTES = 30
CANONICAL_FOLDS = (
    "f1_2023q1",
    "f2_2023q2",
    "f3_2023q3",
    "f4_2023q4",
)
SOURCE_SPLITS = ("train", "validation", "test")
EXPECTED_MARKET_JUMP_TIERS = ("broad", "primary", "high")

# These counts are part of the frozen scientific contract.  The split session
# counts are included even though only the pooled values are reported in the
# main table; they catch boundary and de-duplication drift earlier.
EXPECTED_FOLD_COUNTS: Mapping[str, Mapping[str, Any]] = {
    "f1_2023q1": {
        "splits": {
            "train": {"pairs": 53, "sessions": 25},
            "validation": {"pairs": 37, "sessions": 19},
            "test": {"pairs": 12, "sessions": 9},
        },
        "pairs": 102,
        "sessions": 53,
        "scheduled": 72,
        "jump": 43,
        "both": 13,
    },
    "f2_2023q2": {
        "splits": {
            "train": {"pairs": 90, "sessions": 44},
            "validation": {"pairs": 12, "sessions": 9},
            "test": {"pairs": 26, "sessions": 11},
        },
        "pairs": 128,
        "sessions": 64,
        "scheduled": 93,
        "jump": 50,
        "both": 15,
    },
    "f3_2023q3": {
        "splits": {
            "train": {"pairs": 102, "sessions": 53},
            "validation": {"pairs": 26, "sessions": 11},
            "test": {"pairs": 26, "sessions": 9},
        },
        "pairs": 154,
        "sessions": 73,
        "scheduled": 111,
        "jump": 60,
        "both": 17,
    },
    "f4_2023q4": {
        "splits": {
            "train": {"pairs": 128, "sessions": 64},
            "validation": {"pairs": 26, "sessions": 9},
            "test": {"pairs": 31, "sessions": 15},
        },
        "pairs": 185,
        "sessions": 88,
        "scheduled": 136,
        "jump": 68,
        "both": 19,
    },
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PAIR_UNIVERSE_MANIFEST_KIND = "rq123_pair_universe_manifest_v1"
_EVENT_SOURCE_MANIFEST_KIND = "rq123_frozen_event_sources_v1"
_REQUIRED_MERGED_COLUMNS = {
    "pair_id",
    "session_id",
    "effective_origin_utc",
    "current_snapshot_time_utc",
    "target_snapshot_time_utc",
    "current_surface_flat",
    "target_surface_flat",
    "current_surface_param_json",
    "target_surface_param_json",
    "strike_grid",
    "maturity_days_grid",
    "surface_shape",
    "sample_id",
    "news_row_id",
    "lp_embedding",
}
_REQUIRED_UNIVERSE_COLUMNS = {
    "tolerance_minutes",
    "fold",
    "partition",
    "pair_id",
    "session_id",
    "effective_origin_utc",
    "current_snapshot_time_utc",
    "target_snapshot_time_utc",
    "joint_support_cells",
}
_REQUIRED_SUPPORT_COLUMNS = {
    "tolerance_minutes",
    "pair_id",
    "effective_origin_utc",
    "target_snapshot_time_utc",
    "surface_training_eligible",
    "grid_cell_count",
    "joint_strict_support_cell_count",
    "joint_zero_support",
    "support_method",
    "grid_fingerprint",
}
_REQUIRED_SCHEDULED_COLUMNS = {"event_id", "release_time_utc"}
_REQUIRED_JUMP_COLUMNS = {
    "pair_id",
    "origin_time_utc",
    "target_time_utc",
    "session_id",
    "horizon_minutes",
    "anomaly_tier",
}


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FoldPooledPanelError("Manifest payload is not canonical JSON") from exc


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _pair_universe_sha256(pair_ids: Sequence[object]) -> str:
    normalized = [str(value).strip() for value in pair_ids]
    if not normalized or any(not value for value in normalized):
        raise FoldPooledPanelError("Pair universe contains an empty pair_id")
    if len(normalized) != len(set(normalized)):
        raise FoldPooledPanelError("Pair universe contains duplicate pair_id values")
    return _payload_sha256(sorted(normalized))


def _require_file(path_value: object, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise FoldPooledPanelError(f"{label} is missing: {path}")
    return path


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if frame.empty or missing:
        raise FoldPooledPanelError(f"{label} is empty or missing columns: {missing}")


def _canonical_utc(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.isna().any():
        bad = values.loc[parsed.isna()].head(3).tolist()
        raise FoldPooledPanelError(f"{label} contains invalid UTC timestamps: {bad}")
    return parsed


def _as_bool(values: pd.Series, label: str) -> pd.Series:
    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        raise FoldPooledPanelError(f"{label} contains a non-boolean value: {value!r}")

    return values.map(parse).astype(bool)


def _parse_vector(value: object, *, length: int, label: str) -> np.ndarray:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise FoldPooledPanelError(f"{label} is not a valid vector") from exc
    try:
        vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise FoldPooledPanelError(f"{label} is not numeric") from exc
    if vector.size != int(length) or not np.isfinite(vector).all():
        raise FoldPooledPanelError(
            f"{label} must have {int(length)} finite values, got {vector.size}"
        )
    return vector


def _article_key(row: object) -> str:
    # This is intentionally identical to RQ1 core._normalized_article_key.
    for field_name in ("article_id", "sample_id", "news_row_id"):
        value = getattr(row, field_name, "")
        if pd.notna(value) and str(value).strip():
            return f"{field_name}:{str(value).strip()}"
    raise FoldPooledPanelError("News row has no article_id, sample_id, or news_row_id")


def _article_text(value: object) -> str:
    # This is intentionally identical to RQ1 core._normalized_article_text.
    if value is None or bool(pd.isna(value)):
        return ""
    return str(value).strip()


def _l2(vector: np.ndarray) -> np.ndarray:
    # This is intentionally identical to RQ1 core._l2.
    output = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(output))
    return output.copy() if norm == 0.0 else (output / norm).astype(np.float32)


def _read_table(path: Path, *, sheet_name: str) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffixes.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, low_memory=False)
    raise FoldPooledPanelError(f"Unsupported tabular input: {path}")


def _atomic_write_csv(path: Path, frame: pd.DataFrame, *, compressed: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if compressed:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                    with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                        frame.to_csv(text, index=False)
        else:
            frame.to_csv(temporary, index=False)
        if path.exists():
            if not path.is_file() or sha256_file(path) != sha256_file(temporary):
                raise FoldPooledPanelError(f"Existing panel artifact drift: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FoldPooledPanelError(f"Existing manifest drift: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _source_record(role: str, path: Path) -> dict[str, Any]:
    return {
        "artifact_role": str(role),
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _normalize_expected_counts(
    raw: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    source = EXPECTED_FOLD_COUNTS if raw is None else raw
    if not isinstance(source, Mapping) or not source:
        raise FoldPooledPanelError("expected_counts must be a non-empty mapping")
    result: dict[str, dict[str, Any]] = {}
    for fold, value in source.items():
        if not isinstance(value, Mapping):
            raise FoldPooledPanelError(f"Expected counts for {fold} must be a mapping")
        splits = value.get("splits")
        if not isinstance(splits, Mapping) or set(map(str, splits)) != set(
            SOURCE_SPLITS
        ):
            raise FoldPooledPanelError(
                f"Expected counts for {fold} require train/validation/test"
            )
        normalized_splits: dict[str, dict[str, int]] = {}
        for split in SOURCE_SPLITS:
            split_value = splits[split]
            if not isinstance(split_value, Mapping):
                raise FoldPooledPanelError(f"Expected {fold}/{split} must be a mapping")
            normalized_splits[split] = {
                "pairs": int(split_value["pairs"]),
                "sessions": int(split_value["sessions"]),
            }
        item = {
            "splits": normalized_splits,
            "pairs": int(value["pairs"]),
            "sessions": int(value["sessions"]),
        }
        for optional in ("scheduled", "jump", "both"):
            if optional in value:
                item[optional] = int(value[optional])
        result[str(fold)] = item
    return result


def _validate_pair_universes(
    frame: pd.DataFrame,
    *,
    folds: Sequence[str],
) -> tuple[pd.DataFrame, set[tuple[str, str, str]]]:
    _require_columns(frame, _REQUIRED_UNIVERSE_COLUMNS, "pair universes")
    universe = frame.copy()
    universe["tolerance_minutes"] = pd.to_numeric(
        universe["tolerance_minutes"], errors="raise"
    ).astype(int)
    universe["fold"] = universe["fold"].astype(str)
    universe["partition"] = universe["partition"].astype(str)
    universe["pair_id"] = universe["pair_id"].astype(str).str.strip()
    universe["session_id"] = universe["session_id"].astype(str).str.strip()
    if universe["pair_id"].eq("").any() or universe["session_id"].eq("").any():
        raise FoldPooledPanelError("Pair universes contain empty pair/session IDs")
    if not {TOLERANCE_MINUTES, SOURCE_5M_TOLERANCE_MINUTES}.issubset(
        set(universe["tolerance_minutes"])
    ):
        raise FoldPooledPanelError("Pair universes must contain both 5m and 30m rows")
    selected = universe[universe["tolerance_minutes"].isin((5, 30))].copy()
    selected["effective_origin_utc"] = _canonical_utc(
        selected["effective_origin_utc"], "pair-universe effective_origin_utc"
    )
    selected["current_snapshot_time_utc"] = _canonical_utc(
        selected["current_snapshot_time_utc"],
        "pair-universe current_snapshot_time_utc",
    )
    selected["target_snapshot_time_utc"] = _canonical_utc(
        selected["target_snapshot_time_utc"], "pair-universe target_snapshot_time_utc"
    )
    if (
        not selected["effective_origin_utc"]
        .eq(selected["current_snapshot_time_utc"])
        .all()
    ):
        raise FoldPooledPanelError("Effective origin differs from current snapshot")
    horizons = (
        selected["target_snapshot_time_utc"] - selected["current_snapshot_time_utc"]
    ).dt.total_seconds()
    if not horizons.eq(FORECAST_HORIZON_MINUTES * 60).all():
        raise FoldPooledPanelError("Pair universes do not retain a strict +5m target")
    if selected.duplicated(["tolerance_minutes", "fold", "partition", "pair_id"]).any():
        raise FoldPooledPanelError("Pair universes contain duplicate routed pair IDs")
    expected_fold_set = set(map(str, folds))
    for tolerance in (5, 30):
        observed_folds = set(
            selected.loc[selected["tolerance_minutes"].eq(tolerance), "fold"]
        )
        if observed_folds != expected_fold_set:
            raise FoldPooledPanelError(
                f"{tolerance}m pair-universe fold drift: {sorted(observed_folds)}"
            )
        for fold in folds:
            fold_rows = selected[
                selected["tolerance_minutes"].eq(tolerance)
                & selected["fold"].eq(str(fold))
            ]
            if set(fold_rows["partition"]) != set(SOURCE_SPLITS):
                raise FoldPooledPanelError(
                    f"{tolerance}m {fold} lacks train/validation/test partitions"
                )
            if fold_rows.groupby("pair_id")["partition"].nunique().gt(1).any():
                raise FoldPooledPanelError(
                    f"{tolerance}m {fold} pair appears in multiple source splits"
                )
            split_times = {
                split: fold_rows.loc[
                    fold_rows["partition"].eq(split), "effective_origin_utc"
                ]
                for split in SOURCE_SPLITS
            }
            if not (
                split_times["train"].max() < split_times["validation"].min()
                and split_times["validation"].max() < split_times["test"].min()
            ):
                raise FoldPooledPanelError(
                    f"{tolerance}m {fold} source-split time ranges overlap"
                )
            if "pair_universe_sha256" in fold_rows:
                for split in SOURCE_SPLITS:
                    split_rows = fold_rows[fold_rows["partition"].eq(split)]
                    declared = set(split_rows["pair_universe_sha256"].astype(str))
                    actual = _pair_universe_sha256(split_rows["pair_id"].astype(str))
                    if declared != {actual}:
                        raise FoldPooledPanelError(
                            f"{tolerance}m {fold}/{split} pair-universe SHA drift"
                        )
    five = selected[selected["tolerance_minutes"].eq(5)]
    thirty = selected[selected["tolerance_minutes"].eq(30)]
    five_keys = set(zip(five["fold"], five["partition"], five["pair_id"], strict=True))
    thirty_keys = set(
        zip(thirty["fold"], thirty["partition"], thirty["pair_id"], strict=True)
    )
    if not five_keys.issubset(thirty_keys):
        raise FoldPooledPanelError(
            "5m pair universe is not nested in 30m by fold/split"
        )
    return thirty.reset_index(drop=True), five_keys


def _validate_support(
    frame: pd.DataFrame,
    universe_30m: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    _require_columns(frame, _REQUIRED_SUPPORT_COLUMNS, "30m support audit")
    support = frame.copy()
    tolerance = pd.to_numeric(support["tolerance_minutes"], errors="raise").astype(int)
    if not tolerance.eq(TOLERANCE_MINUTES).all():
        raise FoldPooledPanelError("Support audit must contain only 30m rows")
    support["pair_id"] = support["pair_id"].astype(str).str.strip()
    if support["pair_id"].eq("").any() or support["pair_id"].duplicated().any():
        raise FoldPooledPanelError("Support audit pair_id values must be unique")
    required_pairs = set(universe_30m["pair_id"].astype(str))
    selected = support[support["pair_id"].isin(required_pairs)].copy()
    if set(selected["pair_id"]) != required_pairs:
        raise FoldPooledPanelError("Support audit lacks 30m pair-universe coverage")
    eligible = _as_bool(selected["surface_training_eligible"], "support eligibility")
    zero = _as_bool(selected["joint_zero_support"], "joint_zero_support")
    counts = pd.to_numeric(
        selected["joint_strict_support_cell_count"], errors="raise"
    ).astype(int)
    grid_counts = pd.to_numeric(selected["grid_cell_count"], errors="raise").astype(int)
    if not eligible.all() or zero.any() or not counts.gt(0).all():
        raise FoldPooledPanelError(
            "30m universe contains ineligible or zero-support pairs"
        )
    if not grid_counts.eq(256).all():
        raise FoldPooledPanelError("RQ4 support grid must contain exactly 256 cells")
    fingerprints = set(selected["grid_fingerprint"].astype(str))
    if len(fingerprints) != 1 or not next(iter(fingerprints)).strip():
        raise FoldPooledPanelError("Support grid fingerprint must be unique")
    declared = (
        universe_30m[["pair_id", "joint_support_cells"]]
        .drop_duplicates("pair_id")
        .set_index("pair_id")["joint_support_cells"]
    )
    observed = selected.set_index("pair_id")["joint_strict_support_cell_count"]
    if (
        not pd.to_numeric(declared, errors="raise")
        .astype(int)
        .sort_index()
        .equals(pd.to_numeric(observed, errors="raise").astype(int).sort_index())
    ):
        raise FoldPooledPanelError("Pair-universe/support joint-cell counts differ")
    selected["effective_origin_utc"] = _canonical_utc(
        selected["effective_origin_utc"], "support effective_origin_utc"
    )
    selected["target_snapshot_time_utc"] = _canonical_utc(
        selected["target_snapshot_time_utc"], "support target_snapshot_time_utc"
    )
    return selected.reset_index(drop=True), next(iter(fingerprints))


def _scheduled_membership(
    universe: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(events, _REQUIRED_SCHEDULED_COLUMNS, "frozen scheduled events")
    scheduled = events.copy()
    scheduled["event_id"] = scheduled["event_id"].astype(str).str.strip()
    if scheduled["event_id"].eq("").any() or scheduled["event_id"].duplicated().any():
        raise FoldPooledPanelError("Scheduled event IDs must be non-empty and unique")
    scheduled["release_time_utc"] = _canonical_utc(
        scheduled["release_time_utc"], "scheduled release_time_utc"
    )
    if "scheduled_or_unscheduled" in scheduled:
        flags = (
            scheduled["scheduled_or_unscheduled"].astype(str).str.lower().str.strip()
        )
        if not flags.eq("scheduled").all():
            raise FoldPooledPanelError(
                "Scheduled-event source contains unscheduled rows"
            )
    scheduled = scheduled.sort_values(
        ["release_time_utc", "event_id"], kind="stable"
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for pair in universe.itertuples(index=False):
        origin = pd.Timestamp(pair.effective_origin_utc)
        deltas = (origin - scheduled["release_time_utc"]).dt.total_seconds().div(60.0)
        eligible = scheduled.loc[
            deltas.ge(SCHEDULED_LOWER_MINUTES) & deltas.le(SCHEDULED_UPPER_MINUTES)
        ].copy()
        if eligible.empty:
            continue
        eligible["_delta"] = deltas.loc[eligible.index].astype(float)
        chosen = eligible.sort_values(["_delta", "event_id"], kind="stable").iloc[0]
        rows.append(
            {
                "fold": str(pair.fold),
                "pair_id": str(pair.pair_id),
                "scheduled_event_id": str(chosen["event_id"]),
                "scheduled_event_ids": ";".join(eligible["event_id"].astype(str)),
                "scheduled_event_count": int(eligible["event_id"].nunique()),
                "scheduled_release_time_utc": pd.Timestamp(chosen["release_time_utc"]),
                "scheduled_delta_minutes": float(chosen["_delta"]),
            }
        )
    if not rows:
        raise FoldPooledPanelError("No pairs fall in scheduled [0,+30] window")
    result = pd.DataFrame(rows)
    if result.duplicated(["fold", "pair_id"]).any():
        raise FoldPooledPanelError("Scheduled membership contains duplicate fold/pair")
    return result


def _jump_membership(
    universe: pd.DataFrame,
    market_jumps: pd.DataFrame,
    *,
    expected_tiers: Sequence[str],
) -> pd.DataFrame:
    _require_columns(market_jumps, _REQUIRED_JUMP_COLUMNS, "market-jump candidates")
    jumps = market_jumps.copy()
    jumps["pair_id"] = jumps["pair_id"].astype(str).str.strip()
    jumps["session_id"] = jumps["session_id"].astype(str).str.strip()
    jumps["origin_time_utc"] = _canonical_utc(
        jumps["origin_time_utc"], "jump origin_time_utc"
    )
    jumps["target_time_utc"] = _canonical_utc(
        jumps["target_time_utc"], "jump target_time_utc"
    )
    jumps["anomaly_tier"] = jumps["anomaly_tier"].astype(str).str.lower().str.strip()
    tiers = tuple(map(str, expected_tiers))
    jumps = jumps[jumps["anomaly_tier"].isin(tiers)].copy()
    if jumps.empty or set(jumps["anomaly_tier"]) != set(tiers):
        raise FoldPooledPanelError(
            "Market-jump candidates lack the frozen tier universe"
        )
    if jumps["pair_id"].eq("").any() or jumps["session_id"].eq("").any():
        raise FoldPooledPanelError("Market jumps contain empty pair/session IDs")
    if (
        jumps["pair_id"].duplicated().any()
        or jumps["origin_time_utc"].duplicated().any()
    ):
        raise FoldPooledPanelError("Market-jump pair IDs and origins must be unique")
    horizons = (jumps["target_time_utc"] - jumps["origin_time_utc"]).dt.total_seconds()
    declared_horizons = pd.to_numeric(jumps["horizon_minutes"], errors="raise")
    if (
        not horizons.eq(FORECAST_HORIZON_MINUTES * 60).all()
        or not declared_horizons.eq(FORECAST_HORIZON_MINUTES).all()
    ):
        raise FoldPooledPanelError("Market jumps do not retain the exact +5m horizon")
    keep = [
        "pair_id",
        "session_id",
        "origin_time_utc",
        "target_time_utc",
        "anomaly_tier",
    ]
    for optional in (
        "anomaly_tier_order",
        "pair_rank",
        "peak_metric_name",
        "peak_metric_change",
        "max_abs_robust_z",
        "max_abs_empirical_percentile",
    ):
        if optional in jumps:
            keep.append(optional)
    source = jumps[keep].rename(
        columns={
            "pair_id": "jump_source_pair_id",
            "session_id": "jump_source_session_id",
            "origin_time_utc": "jump_origin_time_utc",
            "target_time_utc": "jump_target_time_utc",
            "anomaly_tier": "jump_anomaly_tier",
        }
    )
    joined = universe.merge(
        source,
        left_on="effective_origin_utc",
        right_on="jump_origin_time_utc",
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        raise FoldPooledPanelError("No exact market-jump origin joins the 30m universe")
    if (
        not joined["pair_id"]
        .astype(str)
        .eq(joined["jump_source_pair_id"].astype(str))
        .all()
    ):
        raise FoldPooledPanelError("Exact jump join disagrees on pair_id lineage")
    if (
        not joined["session_id"]
        .astype(str)
        .eq(joined["jump_source_session_id"].astype(str))
        .all()
    ):
        raise FoldPooledPanelError("Exact jump join disagrees on session lineage")
    result = joined[["fold", "pair_id", *source.columns]].drop(
        columns=["jump_source_pair_id", "jump_source_session_id"], errors="ignore"
    )
    # Restore the source identifiers explicitly after validation; keeping them
    # makes the exact timestamp bridge auditable downstream.
    result["jump_source_pair_id"] = joined["jump_source_pair_id"].astype(str).values
    result["jump_source_session_id"] = (
        joined["jump_source_session_id"].astype(str).values
    )
    if result.duplicated(["fold", "pair_id"]).any():
        raise FoldPooledPanelError("Jump membership contains duplicate fold/pair")
    return result.reset_index(drop=True)


def _canonical_workbook_rows(
    merged: pd.DataFrame,
    required_pair_ids: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    _require_columns(merged, _REQUIRED_MERGED_COLUMNS, "merged_vol panel")
    original_columns = list(map(str, merged.columns))
    if len(original_columns) != len(set(original_columns)):
        raise FoldPooledPanelError("merged_vol contains duplicate column names")
    frame = merged.copy()
    frame["pair_id"] = frame["pair_id"].astype(str).str.strip()
    frame = frame[frame["pair_id"].isin(required_pair_ids)].copy()
    if set(frame["pair_id"]) != required_pair_ids:
        missing = sorted(required_pair_ids - set(frame["pair_id"]))
        raise FoldPooledPanelError(f"merged_vol lacks required pairs: {missing[:5]}")
    for column in (
        "session_id",
        "effective_origin_utc",
        "current_snapshot_time_utc",
        "target_snapshot_time_utc",
        "current_surface_flat",
        "target_surface_flat",
        "current_surface_param_json",
        "target_surface_param_json",
        "strike_grid",
        "maturity_days_grid",
        "surface_shape",
    ):
        variants = frame.groupby("pair_id", sort=False)[column].nunique(dropna=False)
        if variants.gt(1).any():
            bad = variants[variants.gt(1)].index.astype(str).tolist()[:5]
            raise FoldPooledPanelError(
                f"merged_vol pair-invariant column {column} drifts: {bad}"
            )
    if "dataset_tolerance_minutes" in frame:
        values = pd.to_numeric(frame["dataset_tolerance_minutes"], errors="raise")
        if not values.eq(TOLERANCE_MINUTES).all():
            raise FoldPooledPanelError("merged_vol includes non-30m rows")
    canonical_records: list[dict[str, Any]] = []
    for pair_id, pair_rows in frame.groupby("pair_id", sort=False):
        ordered = pair_rows.assign(
            _canonical_news_row=pd.to_numeric(pair_rows["news_row_id"], errors="raise"),
            _canonical_sample_id=pair_rows["sample_id"].astype(str),
        ).sort_values(["_canonical_news_row", "_canonical_sample_id"], kind="stable")
        canonical = (
            ordered.iloc[0]
            .drop(labels=["_canonical_news_row", "_canonical_sample_id"])
            .to_dict()
        )
        articles: dict[str, dict[str, Any]] = {}
        for row in ordered.itertuples(index=False):
            article_key = _article_key(row)
            article = {
                "lp_text": _article_text(getattr(row, "lp_text", "")),
                "lp_embedding": _parse_vector(
                    row.lp_embedding,
                    length=1024,
                    label=f"RQ4 pooled LP pair={pair_id}/{article_key}",
                ),
            }
            previous = articles.get(article_key)
            if previous is not None:
                if previous["lp_text"] != article["lp_text"]:
                    raise FoldPooledPanelError(
                        f"Article text drift within pair: {pair_id}/{article_key}"
                    )
                # Match the frozen RQ1 transform: a repeated article contributes
                # once, using its first stable row.
                continue
            articles[article_key] = article
        if not articles:
            raise FoldPooledPanelError(f"Pair has no usable LP articles: {pair_id}")
        pair_lp = _l2(
            np.mean(
                np.stack(
                    [articles[key]["lp_embedding"] for key in sorted(articles)],
                    axis=0,
                ),
                axis=0,
            )
        )
        canonical_source_sample_id = str(canonical["sample_id"])
        canonical_source_news_row_id = str(canonical["news_row_id"])
        canonical["lp_embedding"] = json.dumps(
            pair_lp.astype(float).tolist(), separators=(",", ":")
        )
        if "lp_dim" in canonical:
            canonical["lp_dim"] = 1024
        if "pair_article_count" in canonical:
            canonical["pair_article_count"] = len(articles)
        canonical["sample_id"] = f"pair::{pair_id}"
        canonical["sample_weight"] = 1.0
        canonical["canonical_source_sample_id"] = canonical_source_sample_id
        canonical["canonical_source_news_row_id"] = canonical_source_news_row_id
        canonical["pair_unique_article_count"] = len(articles)
        canonical["pair_lp_transform"] = "unique_article_lp_mean_l2_v1"
        canonical_records.append(canonical)
    canonical_frame = pd.DataFrame(canonical_records)
    if (
        len(canonical_frame) != len(required_pair_ids)
        or canonical_frame["pair_id"].duplicated().any()
    ):
        raise FoldPooledPanelError(
            "Canonical workbook collapse is not one row per pair"
        )
    canonical_frame["effective_origin_utc"] = _canonical_utc(
        canonical_frame["effective_origin_utc"], "merged effective_origin_utc"
    )
    canonical_frame["current_snapshot_time_utc"] = _canonical_utc(
        canonical_frame["current_snapshot_time_utc"], "merged current_snapshot_time_utc"
    )
    canonical_frame["target_snapshot_time_utc"] = _canonical_utc(
        canonical_frame["target_snapshot_time_utc"], "merged target_snapshot_time_utc"
    )
    if (
        not canonical_frame["effective_origin_utc"]
        .eq(canonical_frame["current_snapshot_time_utc"])
        .all()
    ):
        raise FoldPooledPanelError(
            "Merged effective origin differs from current snapshot"
        )
    horizon = (
        canonical_frame["target_snapshot_time_utc"]
        - canonical_frame["current_snapshot_time_utc"]
    ).dt.total_seconds()
    if not horizon.eq(FORECAST_HORIZON_MINUTES * 60).all():
        raise FoldPooledPanelError("Merged rows do not retain a strict +5m target")
    for row in canonical_frame.itertuples(index=False):
        pair_id = str(row.pair_id)
        _parse_vector(
            row.current_surface_flat, length=256, label=f"{pair_id} current surface"
        )
        _parse_vector(
            row.target_surface_flat, length=256, label=f"{pair_id} target surface"
        )
    aggregate_columns = [
        "canonical_source_sample_id",
        "canonical_source_news_row_id",
        "pair_unique_article_count",
        "pair_lp_transform",
    ]
    return (
        canonical_frame[[*original_columns, *aggregate_columns]].reset_index(drop=True),
        original_columns,
    )


def build_fold_pooled_panels(
    *,
    merged_vol: pd.DataFrame,
    support_audit: pd.DataFrame,
    pair_universes: pd.DataFrame,
    frozen_scheduled_events: pd.DataFrame,
    market_jump_pairs: pd.DataFrame,
    expected_counts: Mapping[str, Mapping[str, Any]] | None = None,
    expected_jump_tiers: Sequence[str] = EXPECTED_MARKET_JUMP_TIERS,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Build and validate in-memory fold-pooled panels.

    This lower-level API is useful for unit tests and callers that have already
    hash-bound their inputs.  Production callers should normally use
    :func:`materialize_fold_pooled_panels`.
    """

    expected = _normalize_expected_counts(expected_counts)
    folds = tuple(expected)
    universe, five_keys = _validate_pair_universes(pair_universes, folds=folds)
    support, grid_fingerprint = _validate_support(support_audit, universe)
    scheduled = _scheduled_membership(universe, frozen_scheduled_events)
    jumps = _jump_membership(
        universe, market_jump_pairs, expected_tiers=expected_jump_tiers
    )
    event_keys = set(zip(scheduled["fold"], scheduled["pair_id"], strict=True)) | set(
        zip(jumps["fold"], jumps["pair_id"], strict=True)
    )
    routed = universe.loc[
        [
            (str(fold), str(pair_id)) in event_keys
            for fold, pair_id in zip(universe["fold"], universe["pair_id"], strict=True)
        ]
    ].copy()
    if routed.empty or routed.duplicated(["fold", "pair_id"]).any():
        raise FoldPooledPanelError(
            "Special-time routed universe is empty or duplicated"
        )
    required_pair_ids = set(routed["pair_id"].astype(str))
    canonical, original_columns = _canonical_workbook_rows(
        merged_vol, required_pair_ids
    )
    canonical_by_pair = canonical.set_index("pair_id", drop=False)
    support_by_pair = support.set_index("pair_id", drop=False)
    scheduled_by_key = scheduled.set_index(["fold", "pair_id"], drop=False)
    jumps_by_key = jumps.set_index(["fold", "pair_id"], drop=False)

    support_export_columns = [
        column
        for column in support.columns
        if column not in {"pair_id", "effective_origin_utc", "target_snapshot_time_utc"}
    ]
    panels: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    for fold in folds:
        fold_routes = routed[routed["fold"].eq(fold)].sort_values(
            ["effective_origin_utc", "pair_id"], kind="stable"
        )
        rows: list[dict[str, Any]] = []
        for route in fold_routes.itertuples(index=False):
            pair_id = str(route.pair_id)
            key = (fold, pair_id)
            scheduled_flag = key in scheduled_by_key.index
            jump_flag = key in jumps_by_key.index
            if not (scheduled_flag or jump_flag):
                raise FoldPooledPanelError(f"Unlabeled routed pair: {fold}/{pair_id}")
            workbook_row = canonical_by_pair.loc[pair_id]
            if isinstance(workbook_row, pd.DataFrame):
                raise FoldPooledPanelError(f"Canonical pair is duplicated: {pair_id}")
            if str(workbook_row["session_id"]) != str(route.session_id):
                raise FoldPooledPanelError(f"Workbook/session drift: {fold}/{pair_id}")
            if pd.Timestamp(workbook_row["effective_origin_utc"]) != pd.Timestamp(
                route.effective_origin_utc
            ):
                raise FoldPooledPanelError(f"Workbook/origin drift: {fold}/{pair_id}")
            support_row = support_by_pair.loc[pair_id]
            if pd.Timestamp(support_row["effective_origin_utc"]) != pd.Timestamp(
                route.effective_origin_utc
            ) or pd.Timestamp(support_row["target_snapshot_time_utc"]) != pd.Timestamp(
                route.target_snapshot_time_utc
            ):
                raise FoldPooledPanelError(f"Support timestamp drift: {fold}/{pair_id}")
            row = workbook_row.to_dict()
            row.update(
                {
                    "checkpoint_fold": fold,
                    "source_split": str(route.partition),
                    "scheduled_event": bool(scheduled_flag),
                    "market_jump": bool(jump_flag),
                    "is_scheduled_event": bool(scheduled_flag),
                    "is_market_jump": bool(jump_flag),
                    "event_regime": (
                        "both"
                        if scheduled_flag and jump_flag
                        else ("scheduled_only" if scheduled_flag else "jump_only")
                    ),
                    "source_5m_membership": (
                        fold,
                        str(route.partition),
                        pair_id,
                    )
                    in five_keys,
                    "panel_pair_id": pair_id,
                    "panel_sample_id": f"pair::{pair_id}",
                    "panel_canonical_row_method": (
                        "minimum_numeric_news_row_then_sample_id_v1"
                    ),
                }
            )
            if scheduled_flag:
                event = scheduled_by_key.loc[key]
                for column in (
                    "scheduled_event_id",
                    "scheduled_event_ids",
                    "scheduled_event_count",
                    "scheduled_release_time_utc",
                    "scheduled_delta_minutes",
                ):
                    row[column] = event[column]
            else:
                row.update(
                    scheduled_event_id="",
                    scheduled_event_ids="",
                    scheduled_event_count=0,
                    scheduled_release_time_utc=pd.NaT,
                    scheduled_delta_minutes=math.nan,
                )
            if jump_flag:
                jump = jumps_by_key.loc[key]
                for column in jumps.columns:
                    if column not in {"fold", "pair_id"}:
                        row[column] = jump[column]
            else:
                for column in jumps.columns:
                    if column not in {"fold", "pair_id"}:
                        row[column] = ""
            for column in support_export_columns:
                row[f"support_{column}"] = support_row[column]
            rows.append(row)
        panel = pd.DataFrame(rows)
        expected_item = expected[fold]
        if panel["pair_id"].astype(str).duplicated().any():
            raise FoldPooledPanelError(f"Pooled panel has duplicate pair IDs: {fold}")
        observed_split_pairs = panel.groupby("source_split")["pair_id"].nunique()
        observed_split_sessions = panel.groupby("source_split")["session_id"].nunique()
        if set(observed_split_pairs.index) != set(SOURCE_SPLITS):
            raise FoldPooledPanelError(f"Pooled panel lacks source splits: {fold}")
        for split in SOURCE_SPLITS:
            split_expected = expected_item["splits"][split]
            if int(observed_split_pairs[split]) != int(split_expected["pairs"]) or int(
                observed_split_sessions[split]
            ) != int(split_expected["sessions"]):
                raise FoldPooledPanelError(
                    f"Count drift for {fold}/{split}: "
                    f"pairs={int(observed_split_pairs[split])}, "
                    f"sessions={int(observed_split_sessions[split])}"
                )
        pair_count = int(panel["pair_id"].nunique())
        session_count = int(panel["session_id"].astype(str).nunique())
        scheduled_count = int(
            _as_bool(panel["is_scheduled_event"], "scheduled flag").sum()
        )
        jump_count = int(_as_bool(panel["is_market_jump"], "jump flag").sum())
        both_count = int(panel["event_regime"].eq("both").sum())
        for key_name, observed in (
            ("pairs", pair_count),
            ("sessions", session_count),
            ("scheduled", scheduled_count),
            ("jump", jump_count),
            ("both", both_count),
        ):
            if key_name in expected_item and observed != int(expected_item[key_name]):
                raise FoldPooledPanelError(
                    f"Count drift for {fold}/{key_name}: {observed} != "
                    f"{int(expected_item[key_name])}"
                )
        extras = [column for column in panel.columns if column not in original_columns]
        panel = panel[[*original_columns, *extras]].reset_index(drop=True)
        panels[fold] = panel
        summaries.append(
            {
                "checkpoint_fold": fold,
                "pair_count": pair_count,
                "session_count": session_count,
                **{
                    f"{split}_pair_count": int(observed_split_pairs[split])
                    for split in SOURCE_SPLITS
                },
                **{
                    f"{split}_session_count": int(observed_split_sessions[split])
                    for split in SOURCE_SPLITS
                },
                "scheduled_pair_count": scheduled_count,
                "jump_pair_count": jump_count,
                "both_pair_count": both_count,
                "scheduled_only_pair_count": int(
                    panel["event_regime"].eq("scheduled_only").sum()
                ),
                "jump_only_pair_count": int(
                    panel["event_regime"].eq("jump_only").sum()
                ),
                "source_5m_pair_count": int(
                    _as_bool(panel["source_5m_membership"], "5m membership").sum()
                ),
                "source_30m_only_pair_count": int(
                    (~_as_bool(panel["source_5m_membership"], "5m membership")).sum()
                ),
                "pair_universe_sha256": _pair_universe_sha256(panel["pair_id"]),
            }
        )
    summary = pd.DataFrame(summaries)
    audit = {
        "grid_fingerprint": grid_fingerprint,
        "original_merged_columns": original_columns,
        "routed_row_count": int(sum(len(panel) for panel in panels.values())),
        "unique_pair_count": int(
            len({pair_id for panel in panels.values() for pair_id in panel["pair_id"]})
        ),
        "scheduled_routed_count": int(summary["scheduled_pair_count"].sum()),
        "jump_routed_count": int(summary["jump_pair_count"].sum()),
        "both_routed_count": int(summary["both_pair_count"].sum()),
    }
    return panels, summary, audit


def _config_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "rq4_fold_pooled",
        "rq4_fold_pooled_evaluation",
        "fold_pooled_panels",
    ):
        value = config.get(key)
        if isinstance(value, Mapping):
            return value
    return config


def _data_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = _config_section(config)
    value = section.get("data")
    return value if isinstance(value, Mapping) else section


def _config_path(data: Mapping[str, Any], key: str) -> Path:
    if key not in data:
        raise FoldPooledPanelError(f"Panel config is missing data.{key}")
    return _require_file(data[key], f"data.{key}")


def _verify_optional_expected_hash(
    config: Mapping[str, Any], *, role: str, path: Path
) -> None:
    section = _config_section(config)
    raw_hashes = section.get("source_sha256", {})
    expected = raw_hashes.get(role) if isinstance(raw_hashes, Mapping) else None
    if expected in (None, ""):
        expected = _data_section(config).get(f"{role}_sha256")
    if expected in (None, ""):
        return
    digest = str(expected).strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise FoldPooledPanelError(f"Configured {role} SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != digest:
        raise FoldPooledPanelError(
            f"Configured {role} SHA-256 drift: expected={digest}, actual={actual}"
        )


def _validate_upstream_manifests(
    config: Mapping[str, Any],
    *,
    pair_universes_path: Path,
    scheduled_events_path: Path,
    market_jump_pairs_path: Path,
    merged_vol_path: Path,
) -> dict[str, Path]:
    data = _data_section(config)
    pair_manifest = (
        Path(
            str(
                data.get(
                    "pair_universe_manifest_path",
                    pair_universes_path.with_name("pair_universe_manifest.json"),
                )
            )
        )
        .expanduser()
        .resolve()
    )
    event_manifest = (
        Path(
            str(
                data.get(
                    "frozen_event_sources_manifest_path",
                    scheduled_events_path.with_name("frozen_event_sources.json"),
                )
            )
        )
        .expanduser()
        .resolve()
    )
    dataset_validation = (
        Path(
            str(
                data.get(
                    "dataset_validation_path",
                    merged_vol_path.with_name("validation_summary.json"),
                )
            )
        )
        .expanduser()
        .resolve()
    )
    for path, label in (
        (pair_manifest, "pair-universe manifest"),
        (event_manifest, "event-source manifest"),
        (dataset_validation, "30m dataset validation"),
    ):
        if not path.is_file():
            raise FoldPooledPanelError(f"{label} is missing: {path}")

    pair_payload = json.loads(pair_manifest.read_text(encoding="utf-8"))
    pair_unsigned = {
        key: value for key, value in pair_payload.items() if key != "manifest_sha256"
    }
    pair_record = pair_payload.get("pair_universes")
    if (
        pair_payload.get("kind") != _PAIR_UNIVERSE_MANIFEST_KIND
        or _payload_sha256(pair_unsigned) != pair_payload.get("manifest_sha256")
        or not isinstance(pair_record, Mapping)
        or Path(str(pair_record.get("path", ""))).resolve()
        != pair_universes_path.resolve()
        or pair_record.get("sha256") != sha256_file(pair_universes_path)
        or int(pair_record.get("size_bytes", -1)) != pair_universes_path.stat().st_size
    ):
        raise FoldPooledPanelError("Pair-universe upstream manifest drift")

    event_payload = json.loads(event_manifest.read_text(encoding="utf-8"))
    event_unsigned = {
        key: value for key, value in event_payload.items() if key != "payload_sha256"
    }
    if (
        event_payload.get("kind") != _EVENT_SOURCE_MANIFEST_KIND
        or _payload_sha256(event_unsigned) != event_payload.get("payload_sha256")
        or Path(str(event_payload.get("scheduled_events_path", ""))).resolve()
        != scheduled_events_path.resolve()
        or event_payload.get("scheduled_events_sha256")
        != sha256_file(scheduled_events_path)
        or Path(str(event_payload.get("market_jump_pairs_path", ""))).resolve()
        != market_jump_pairs_path.resolve()
        or event_payload.get("market_jump_pairs_sha256")
        != sha256_file(market_jump_pairs_path)
    ):
        raise FoldPooledPanelError("Frozen event-source upstream manifest drift")

    validation = json.loads(dataset_validation.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "pass"
        or int(validation.get("tolerance_minutes", -1)) != TOLERANCE_MINUTES
        or Path(str(validation.get("workbook_path", ""))).resolve()
        != merged_vol_path.resolve()
        or int(validation.get("workbook_size_bytes", -1))
        != merged_vol_path.stat().st_size
        or int(validation.get("joint_strict_support_pairs", -1)) <= 0
    ):
        raise FoldPooledPanelError("30m dataset validation summary drift")
    return {
        "pair_universe_manifest": pair_manifest,
        "frozen_event_sources_manifest": event_manifest,
        "dataset_validation": dataset_validation,
    }


def _bundle_paths(output_dir: Path, folds: Sequence[str]) -> dict[str, Path]:
    return {
        "manifest": output_dir / "fold_pooled_panel_manifest.json",
        "summary": output_dir / "fold_pooled_panel_summary.csv",
        **{fold: output_dir / "panels" / f"{fold}.csv.gz" for fold in folds},
    }


def validate_fold_pooled_bundle(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Validate every source and artifact referenced by an existing bundle."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "fold_pooled_panel_manifest.json"
    if not manifest_path.is_file():
        raise FoldPooledPanelError(f"Fold-pooled manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoldPooledPanelError("Fold-pooled manifest is invalid JSON") from exc
    unsigned = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("kind") != MANIFEST_KIND
        or _payload_sha256(unsigned) != manifest.get("payload_sha256")
    ):
        raise FoldPooledPanelError("Fold-pooled manifest signature drift")
    source_records = manifest.get("sources")
    panel_records = manifest.get("panels")
    if not isinstance(source_records, Mapping) or not isinstance(
        panel_records, Mapping
    ):
        raise FoldPooledPanelError(
            "Fold-pooled manifest source/panel records are invalid"
        )
    for role, record in source_records.items():
        if not isinstance(record, Mapping):
            raise FoldPooledPanelError(f"Invalid source manifest row: {role}")
        path = _require_file(record.get("path"), f"source {role}")
        if path.stat().st_size != int(record.get("size_bytes", -1)) or sha256_file(
            path
        ) != record.get("sha256"):
            raise FoldPooledPanelError(f"Source artifact drift: {role}")
    if config is not None:
        data = _data_section(config)
        role_to_key = {
            "merged_vol": "merged_vol_path",
            "support_audit": "support_audit_path",
            "pair_universes": "pair_universes_path",
            "frozen_scheduled_events": "frozen_scheduled_events_path",
            "market_jump_pairs": "market_jump_pairs_path",
        }
        for role, key in role_to_key.items():
            configured = _config_path(data, key)
            recorded = Path(str(source_records[role]["path"])).resolve()
            if configured != recorded:
                raise FoldPooledPanelError(f"Configured source path drift: {role}")
            _verify_optional_expected_hash(config, role=role, path=configured)
    original_columns = manifest.get("original_merged_columns")
    if not isinstance(original_columns, list) or not original_columns:
        raise FoldPooledPanelError("Manifest lacks original merged_vol columns")
    paths: dict[str, Path] = {"manifest": manifest_path}
    for fold, record in panel_records.items():
        if not isinstance(record, Mapping):
            raise FoldPooledPanelError(f"Invalid panel record: {fold}")
        path = _require_file(record.get("path"), f"panel {fold}")
        if path.stat().st_size != int(record.get("size_bytes", -1)) or sha256_file(
            path
        ) != record.get("sha256"):
            raise FoldPooledPanelError(f"Panel artifact drift: {fold}")
        panel = pd.read_csv(path, low_memory=False)
        required = {
            *map(str, original_columns),
            "checkpoint_fold",
            "source_split",
            "scheduled_event",
            "market_jump",
            "is_scheduled_event",
            "is_market_jump",
            "event_regime",
            "source_5m_membership",
        }
        _require_columns(panel, required, f"panel {fold}")
        scheduled = _as_bool(panel["is_scheduled_event"], f"{fold} scheduled")
        jumps = _as_bool(panel["is_market_jump"], f"{fold} jumps")
        if not scheduled.equals(
            _as_bool(panel["scheduled_event"], f"{fold} scheduled alias")
        ) or not jumps.equals(_as_bool(panel["market_jump"], f"{fold} jump alias")):
            raise FoldPooledPanelError(f"Panel event-flag alias drift: {fold}")
        expected_regime = np.where(
            scheduled & jumps,
            "both",
            np.where(scheduled, "scheduled_only", "jump_only"),
        )
        if (
            len(panel) != int(record.get("row_count", -1))
            or panel["pair_id"].astype(str).nunique()
            != int(record.get("pair_count", -1))
            or panel["session_id"].astype(str).nunique()
            != int(record.get("session_count", -1))
            or panel["pair_id"].astype(str).duplicated().any()
            or set(panel["checkpoint_fold"].astype(str)) != {str(fold)}
            or set(panel["source_split"].astype(str)) != set(SOURCE_SPLITS)
            or not (scheduled | jumps).all()
            or not np.array_equal(panel["event_regime"].astype(str), expected_regime)
            or _pair_universe_sha256(panel["pair_id"])
            != record.get("pair_universe_sha256")
        ):
            raise FoldPooledPanelError(f"Panel semantic drift: {fold}")
        current = _canonical_utc(
            panel["current_snapshot_time_utc"], f"{fold} current timestamp"
        )
        target = _canonical_utc(
            panel["target_snapshot_time_utc"], f"{fold} target timestamp"
        )
        if not (target - current).dt.total_seconds().eq(300.0).all():
            raise FoldPooledPanelError(f"Panel target horizon drift: {fold}")
        paths[str(fold)] = path
    summary_record = manifest.get("summary")
    if not isinstance(summary_record, Mapping):
        raise FoldPooledPanelError("Manifest summary record is invalid")
    summary_path = _require_file(summary_record.get("path"), "panel summary")
    if summary_path.stat().st_size != int(
        summary_record.get("size_bytes", -1)
    ) or sha256_file(summary_path) != summary_record.get("sha256"):
        raise FoldPooledPanelError("Panel summary artifact drift")
    summary = pd.read_csv(summary_path)
    if len(summary) != len(panel_records) or set(
        summary["checkpoint_fold"].astype(str)
    ) != set(panel_records):
        raise FoldPooledPanelError("Panel summary rows drift")
    paths["summary"] = summary_path
    return paths


def materialize_fold_pooled_panels(
    config: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create or validate the signed four-panel RQ4 input bundle.

    Required config keys may live at the root, below ``rq4_fold_pooled`` (or
    ``rq4_fold_pooled_evaluation``), and optionally below its ``data`` mapping:

    ``merged_vol_path``, ``support_audit_path``, ``pair_universes_path``,
    ``frozen_scheduled_events_path`` and ``market_jump_pairs_path``.
    """

    if not isinstance(config, Mapping):
        raise FoldPooledPanelError("Panel config must be a mapping")
    root = Path(output_dir).expanduser().resolve()
    data = _data_section(config)
    source_paths = {
        "merged_vol": _config_path(data, "merged_vol_path"),
        "support_audit": _config_path(data, "support_audit_path"),
        "pair_universes": _config_path(data, "pair_universes_path"),
        "frozen_scheduled_events": _config_path(data, "frozen_scheduled_events_path"),
        "market_jump_pairs": _config_path(data, "market_jump_pairs_path"),
    }
    for role, path in source_paths.items():
        _verify_optional_expected_hash(config, role=role, path=path)
    section = _config_section(config)
    expected_raw = section.get("expected_fold_counts")
    expected = _normalize_expected_counts(
        expected_raw if isinstance(expected_raw, Mapping) else None
    )
    paths = _bundle_paths(root, tuple(expected))
    if paths["manifest"].is_file():
        return validate_fold_pooled_bundle(root, config=config)
    partial = [
        path for role, path in paths.items() if role != "manifest" and path.exists()
    ]
    if partial:
        raise FoldPooledPanelError(
            "Partial fold-pooled bundle exists without a signed manifest: "
            f"{partial[:3]}"
        )
    upstream = _validate_upstream_manifests(
        config,
        pair_universes_path=source_paths["pair_universes"],
        scheduled_events_path=source_paths["frozen_scheduled_events"],
        market_jump_pairs_path=source_paths["market_jump_pairs"],
        merged_vol_path=source_paths["merged_vol"],
    )
    sheet_name = str(
        section.get("sheet_name", data.get("sheet_name", "gan_input_ready"))
    )
    merged = _read_table(source_paths["merged_vol"], sheet_name=sheet_name)
    support = _read_table(source_paths["support_audit"], sheet_name=sheet_name)
    universes = _read_table(source_paths["pair_universes"], sheet_name=sheet_name)
    events = _read_table(source_paths["frozen_scheduled_events"], sheet_name=sheet_name)
    jumps = _read_table(source_paths["market_jump_pairs"], sheet_name=sheet_name)
    panels, summary, audit = build_fold_pooled_panels(
        merged_vol=merged,
        support_audit=support,
        pair_universes=universes,
        frozen_scheduled_events=events,
        market_jump_pairs=jumps,
        expected_counts=expected,
    )
    root.mkdir(parents=True, exist_ok=True)
    panel_records: dict[str, dict[str, Any]] = {}
    for fold, panel in panels.items():
        panel_path = _atomic_write_csv(paths[fold], panel, compressed=True)
        summary_row = summary.loc[summary["checkpoint_fold"].eq(fold)].iloc[0]
        panel_records[fold] = {
            **_source_record(f"fold_pooled_panel:{fold}", panel_path),
            "row_count": int(len(panel)),
            "pair_count": int(panel["pair_id"].nunique()),
            "session_count": int(panel["session_id"].nunique()),
            "source_split_pair_counts": {
                split: int(summary_row[f"{split}_pair_count"])
                for split in SOURCE_SPLITS
            },
            "source_split_session_counts": {
                split: int(summary_row[f"{split}_session_count"])
                for split in SOURCE_SPLITS
            },
            "scheduled_pair_count": int(summary_row["scheduled_pair_count"]),
            "jump_pair_count": int(summary_row["jump_pair_count"]),
            "both_pair_count": int(summary_row["both_pair_count"]),
            "source_5m_pair_count": int(summary_row["source_5m_pair_count"]),
            "pair_universe_sha256": str(summary_row["pair_universe_sha256"]),
        }
    summary_path = _atomic_write_csv(paths["summary"], summary, compressed=False)
    source_records = {
        role: _source_record(role, path) for role, path in source_paths.items()
    }
    source_records.update(
        {role: _source_record(role, path) for role, path in upstream.items()}
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "interpretation": "retrospective_rolling_development",
        "tolerance_minutes": TOLERANCE_MINUTES,
        "forecast_horizon_minutes": FORECAST_HORIZON_MINUTES,
        "scheduled_window": {
            "lower_minutes": SCHEDULED_LOWER_MINUTES,
            "upper_minutes": SCHEDULED_UPPER_MINUTES,
            "inclusive": True,
        },
        "market_jump_join": "exact_effective_origin_utc_v1",
        "fold_pooling": "train_plus_validation_plus_test_with_source_split_v1",
        "canonical_row_method": "minimum_numeric_news_row_then_sample_id_v1",
        "sources": source_records,
        "grid_fingerprint": audit["grid_fingerprint"],
        "original_merged_columns": audit["original_merged_columns"],
        "routed_row_count": audit["routed_row_count"],
        "unique_pair_count": audit["unique_pair_count"],
        "scheduled_routed_count": audit["scheduled_routed_count"],
        "jump_routed_count": audit["jump_routed_count"],
        "both_routed_count": audit["both_routed_count"],
        "panels": panel_records,
        "summary": _source_record("fold_pooled_panel_summary", summary_path),
    }
    manifest["payload_sha256"] = _payload_sha256(manifest)
    _atomic_write_json(paths["manifest"], manifest)
    return validate_fold_pooled_bundle(root, config=config)


__all__ = [
    "CANONICAL_FOLDS",
    "EXPECTED_FOLD_COUNTS",
    "FoldPooledPanelError",
    "build_fold_pooled_panels",
    "materialize_fold_pooled_panels",
    "sha256_file",
    "validate_fold_pooled_bundle",
]
