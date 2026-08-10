"""Market-first approximate-ATM and volatility-skew jump detection.

The discovery population is the complete raw-IV market index.  News and
scheduled releases are joined only after market anomalies have been ranked so
that the event sample is not conditioned on the availability of an article.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import scripts._path_setup  # noqa: F401

from wgan_option.market.treasury_sessions import TreasuryGlobexSessionCalendar


ROOT = Path(__file__).resolve().parents[2]
ISO_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
DEFAULT_MATURITY_BUCKETS = (
    (1, 5, "01_05bd"),
    (6, 21, "06_21bd"),
    (22, 63, "22_63bd"),
    (64, 107, "64_107bd"),
)
TIER_ORDER = {"none": 0, "broad": 1, "primary": 2, "high": 3}


@dataclass(frozen=True)
class ExtractionAudit:
    accepted_precalib_rows: int
    reconstructed_slices: int
    reconstructed_points: int
    surface_parameter_mismatch_slices: int
    surface_q_mismatch_slices: int
    surface_iv_mismatch_slices: int
    max_abs_surface_q_difference: float
    max_abs_surface_iv_difference: float
    ambiguous_maturity_slices: int
    ambiguous_underlying_slices: int


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Missing timestamp: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _utc_text(value: Any) -> str:
    return _utc(value).strftime(ISO_UTC_FORMAT)


def _london_text(value: Any) -> str:
    return _utc(value).tz_convert("Europe/London").isoformat()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    """Match the surface generator's left-continuous weighted median."""

    pairs = sorted(
        (
            (float(value), max(float(weight), 0.0))
            for value, weight in zip(values, weights)
            if math.isfinite(float(value)) and math.isfinite(float(weight))
        ),
        key=lambda item: item[0],
    )
    if not pairs:
        return math.nan
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return float(np.median([value for value, _ in pairs]))
    threshold = total_weight / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def _joined_unique(values: Iterable[Any]) -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if str(value).strip() and str(value).strip().lower() not in {"nan", "none"}
        }
    )
    return ";".join(cleaned)


def _aggregate_anchor_records(
    anchor_time_utc: str,
    records: Sequence[Mapping[str, Any]],
    surface_params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = [
        dict(record)
        for record in records
        if str(record.get("passes_precalib_filter", "")).strip().lower() == "true"
    ]
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        grouped[(int(record["business_days"]), float(record["strike"]))].append(record)

    by_business_days: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (business_days, strike), rows in grouped.items():
        weights = np.asarray([float(row["weight"]) for row in rows], dtype=float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            continue
        implied_vols = np.asarray([float(row["implied_vol"]) for row in rows], dtype=float)
        moneyness = np.asarray([float(row["percent_strike"]) for row in rows], dtype=float)
        spots = np.asarray([float(row["spot"]) for row in rows], dtype=float)
        prices = np.asarray([float(row["price"]) for row in rows], dtype=float)
        implied_vol = weighted_median(implied_vols, weights)
        strike_over_forward = float(np.average(moneyness, weights=weights))
        if not (
            math.isfinite(implied_vol)
            and implied_vol > 0
            and math.isfinite(strike_over_forward)
            and strike_over_forward > 0
        ):
            continue
        maturities = sorted({str(row.get("maturity_date", "")) for row in rows})
        underlyings = sorted({str(row.get("underlying_contract_id", "")) for row in rows})
        by_business_days[business_days].append(
            {
                "anchor_time_utc": str(anchor_time_utc),
                "anchor_time_london": _london_text(anchor_time_utc),
                "business_days": int(business_days),
                "maturity_date": _joined_unique(maturities),
                "maturity_date_count": len([value for value in maturities if value]),
                "underlying_contract_id": _joined_unique(underlyings),
                "underlying_contract_count": len([value for value in underlyings if value]),
                "strike": float(strike),
                "strike_over_forward": strike_over_forward,
                "percent_strike": strike_over_forward,
                "log_moneyness": float(math.log(strike_over_forward)),
                "implied_vol": float(implied_vol),
                "implied_vol_pct": float(implied_vol * 100.0),
                "weight_sum": float(weights.sum()),
                "observation_count": int(len(rows)),
                "weighted_spot": float(np.average(spots, weights=weights)),
                "weighted_option_price": float(np.average(prices, weights=weights)),
                "option_types": _joined_unique(row.get("option_type", "") for row in rows),
                "surface_input_roles": _joined_unique(
                    row.get("surface_input_role", "") for row in rows
                ),
                "option_contract_ids": _joined_unique(row.get("contract_id", "") for row in rows),
            }
        )

    expected_business_days = [int(value) for value in surface_params.get("business_days", [])]
    expected_q = surface_params.get("percent_strikes", [])
    expected_iv = surface_params.get("implied_vols", [])
    expected_by_day = {
        business_days: (list(q_values), list(iv_values))
        for business_days, q_values, iv_values in zip(
            expected_business_days, expected_q, expected_iv
        )
    }
    output: list[dict[str, Any]] = []
    mismatches = 0
    q_mismatches = 0
    iv_mismatches = 0
    max_abs_q_difference = 0.0
    max_abs_iv_difference = 0.0
    ambiguous_maturity = 0
    ambiguous_underlying = 0
    for business_days in expected_business_days:
        points = sorted(
            by_business_days.get(business_days, []),
            key=lambda row: float(row["strike_over_forward"]),
        )
        expected_q_values, expected_iv_values = expected_by_day[business_days]
        matches = len(points) == len(expected_q_values)
        q_matches = matches
        iv_matches = matches
        if matches:
            reconstructed_q = np.asarray(
                [row["strike_over_forward"] for row in points], dtype=float
            )
            reconstructed_iv = np.asarray(
                [row["implied_vol"] for row in points], dtype=float
            )
            authoritative_q = np.asarray(expected_q_values, dtype=float)
            authoritative_iv = np.asarray(expected_iv_values, dtype=float)
            q_differences = reconstructed_q - authoritative_q
            iv_differences = reconstructed_iv - authoritative_iv
            max_abs_q_difference = max(
                max_abs_q_difference, float(np.max(np.abs(q_differences), initial=0.0))
            )
            max_abs_iv_difference = max(
                max_abs_iv_difference, float(np.max(np.abs(iv_differences), initial=0.0))
            )
            # Treat sub-1e-11 serialization noise as equal.  The persisted
            # params_json values remain authoritative; this tolerance only
            # controls whether harmless float round-trips are reported as
            # historical generator drift.
            q_matches = bool(
                np.allclose(reconstructed_q, authoritative_q, rtol=1e-11, atol=1e-11)
            )
            iv_matches = bool(
                np.allclose(reconstructed_iv, authoritative_iv, rtol=1e-11, atol=1e-11)
            )
            matches = q_matches and iv_matches
            # params_json is the frozen surface actually consumed downstream.
            # The precalibration rows are retained as lineage, but historical
            # generator drift must not silently replace the indexed surface.
            for index, point in enumerate(points):
                point["precalib_reconstructed_strike_over_forward"] = float(
                    reconstructed_q[index]
                )
                point["precalib_reconstructed_implied_vol"] = float(
                    reconstructed_iv[index]
                )
                point["surface_q_difference"] = float(q_differences[index])
                point["surface_iv_difference"] = float(iv_differences[index])
                point["strike_over_forward"] = float(authoritative_q[index])
                point["percent_strike"] = float(authoritative_q[index])
                point["log_moneyness"] = float(math.log(authoritative_q[index]))
                point["implied_vol"] = float(authoritative_iv[index])
                point["implied_vol_pct"] = float(authoritative_iv[index] * 100.0)
        if not matches:
            mismatches += 1
        if not q_matches:
            q_mismatches += 1
        if not iv_matches:
            iv_mismatches += 1
        slice_id = _stable_id("slice", anchor_time_utc, business_days)
        for index, point in enumerate(points):
            point["slice_id"] = slice_id
            point["slice_point_count"] = int(len(points))
            point["point_index"] = int(index)
            point["surface_parameter_match"] = bool(matches)
            output.append(point)
        if points and any(int(row["maturity_date_count"]) != 1 for row in points):
            ambiguous_maturity += 1
        if points and any(int(row["underlying_contract_count"]) != 1 for row in points):
            ambiguous_underlying += 1
    return output, {
        "selected_rows": len(selected),
        "slices": len(expected_business_days),
        "points": len(output),
        "mismatch_slices": mismatches,
        "q_mismatch_slices": q_mismatches,
        "iv_mismatch_slices": iv_mismatches,
        "max_abs_q_difference": max_abs_q_difference,
        "max_abs_iv_difference": max_abs_iv_difference,
        "ambiguous_maturity_slices": ambiguous_maturity,
        "ambiguous_underlying_slices": ambiguous_underlying,
    }


def extract_surface_points(sqlite_path: str | Path) -> tuple[pd.DataFrame, ExtractionAudit]:
    """Rebuild the raw surface strike points and reconcile them to params_json."""

    path = Path(sqlite_path)
    if not path.is_file():
        raise FileNotFoundError(f"Market surface index does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        surface_params = {
            str(anchor): json.loads(params_json)
            for anchor, params_json in connection.execute(
                "SELECT anchor_time_utc, params_json FROM surface_anchor"
            )
        }
        cursor = connection.execute(
            "SELECT anchor_time_utc, row_json "
            "FROM precalib_point ORDER BY anchor_time_utc"
        )
        output: list[dict[str, Any]] = []
        totals = defaultdict(int)
        for anchor, rows in itertools.groupby(cursor, key=lambda row: str(row[0])):
            records = [json.loads(row_json) for _, row_json in rows]
            points, audit = _aggregate_anchor_records(
                anchor,
                records,
                surface_params.get(anchor, {}),
            )
            output.extend(points)
            for key, value in audit.items():
                if key.startswith("max_abs_"):
                    totals[key] = max(float(totals[key]), float(value))
                else:
                    totals[key] += int(value)
    finally:
        connection.close()
    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError(f"No accepted raw-IV surface points were reconstructed from {path}")
    frame = frame.sort_values(
        ["anchor_time_utc", "business_days", "strike_over_forward", "strike"],
        kind="stable",
    ).reset_index(drop=True)
    return frame, ExtractionAudit(
        accepted_precalib_rows=int(totals["selected_rows"]),
        reconstructed_slices=int(totals["slices"]),
        reconstructed_points=int(totals["points"]),
        surface_parameter_mismatch_slices=int(totals["mismatch_slices"]),
        surface_q_mismatch_slices=int(totals["q_mismatch_slices"]),
        surface_iv_mismatch_slices=int(totals["iv_mismatch_slices"]),
        max_abs_surface_q_difference=float(totals["max_abs_q_difference"]),
        max_abs_surface_iv_difference=float(totals["max_abs_iv_difference"]),
        ambiguous_maturity_slices=int(totals["ambiguous_maturity_slices"]),
        ambiguous_underlying_slices=int(totals["ambiguous_underlying_slices"]),
    )


def _quality_from_distance(distance: float, thresholds: Sequence[float]) -> str:
    if not math.isfinite(float(distance)):
        return "unusable"
    for label, threshold in zip(("A", "B", "C"), thresholds):
        if float(distance) <= float(threshold) + 1e-15:
            return label
    return "outside_5pct"


def select_approximate_atm(
    points: pd.DataFrame,
    *,
    distance_thresholds: Sequence[float] = (0.01, 0.02, 0.05),
    tie_tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one observed near-ATM point per real maturity slice."""

    required = {
        "slice_id",
        "anchor_time_utc",
        "business_days",
        "maturity_date",
        "underlying_contract_id",
        "strike",
        "strike_over_forward",
        "log_moneyness",
        "implied_vol",
        "weight_sum",
    }
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"Surface points are missing columns: {missing}")
    work = points.copy()
    work["abs_log_moneyness"] = work["log_moneyness"].abs()
    work["abs_moneyness_distance"] = (work["strike_over_forward"] - 1.0).abs()
    minimum = work.groupby("slice_id", sort=False)["abs_log_moneyness"].transform("min")
    work["is_nearest_atm_candidate"] = (
        (work["abs_log_moneyness"] - minimum).abs() <= float(tie_tolerance)
    )
    candidates = work[work["is_nearest_atm_candidate"]].copy()
    tie_counts = candidates.groupby("slice_id", sort=False).size().rename("atm_tie_count")
    candidates = candidates.sort_values(
        ["slice_id", "weight_sum", "strike"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = candidates.drop_duplicates("slice_id", keep="first").copy()
    selected["atm_tie_count"] = selected["slice_id"].map(tie_counts).astype(int)
    selected["atm_quality"] = selected["abs_moneyness_distance"].map(
        lambda value: _quality_from_distance(float(value), distance_thresholds)
    )
    selected["atm_iv_decimal"] = selected["implied_vol"].astype(float)
    selected["atm_iv_pct"] = selected["atm_iv_decimal"] * 100.0
    selected["atm_strike"] = selected["strike"].astype(float)
    selected["atm_strike_over_forward"] = selected["strike_over_forward"].astype(float)
    selected["atm_log_moneyness"] = selected["log_moneyness"].astype(float)
    selected["is_exact_atm"] = selected["abs_moneyness_distance"] <= 1e-12
    selected["selected_atm_method"] = "nearest_observed_abs_log_moneyness"
    selected_ids = set(selected.index.tolist())
    work["is_selected_atm"] = work.index.to_series().isin(selected_ids)
    work["atm_tie_count"] = work["slice_id"].map(tie_counts).fillna(0).astype(int)
    selected = selected.sort_values(
        ["anchor_time_utc", "business_days", "maturity_date"], kind="stable"
    ).reset_index(drop=True)
    return work, selected


def load_valid_pairs(
    sqlite_path: str | Path,
    session_calendar_path: str | Path,
    *,
    current_window_minutes: int = 5,
) -> pd.DataFrame:
    path = Path(sqlite_path)
    calendar = TreasuryGlobexSessionCalendar.from_csv(Path(session_calendar_path))
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            "SELECT origin_time_utc, target_time_utc, current_params_sha256, "
            "target_params_sha256 FROM valid_pair ORDER BY origin_time_utc",
            connection,
        )
    finally:
        connection.close()
    audit_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        origin = _utc(row.origin_time_utc)
        target = _utc(row.target_time_utc)
        analysis_start = origin - pd.Timedelta(minutes=int(current_window_minutes))
        # A pair represents two backward-looking five-minute market windows:
        # current [origin-5m, origin] and target [origin, target].  Both must
        # lie in one uninterrupted CME continuous session.
        session = calendar.session_for_interval(analysis_start, target)
        same_session = session is not None
        audit_rows.append(
            {
                "pair_id": _stable_id("pair", _utc_text(origin), _utc_text(target)),
                "origin_time_utc": _utc_text(origin),
                "target_time_utc": _utc_text(target),
                "origin_time_london": _london_text(origin),
                "target_time_london": _london_text(target),
                "analysis_window_start_utc": _utc_text(analysis_start),
                "analysis_window_end_utc": _utc_text(target),
                "current_params_sha256": str(row.current_params_sha256),
                "target_params_sha256": str(row.target_params_sha256),
                "horizon_minutes": float((target - origin).total_seconds() / 60.0),
                "same_cme_continuous_session": bool(same_session),
                "session_id": session.session_id if session is not None else "",
                "session_open_utc": _utc_text(session.open_utc) if session is not None else "",
                "session_close_utc": _utc_text(session.close_utc) if session is not None else "",
                "pair_exclusion_reason": "" if same_session else "cross_or_closed_session",
            }
        )
    return pd.DataFrame(audit_rows)


def _slice_maps(points: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for (anchor, maturity), frame in points.groupby(
        ["anchor_time_utc", "maturity_date"], sort=False
    ):
        result[str(anchor)][str(maturity)] = frame.sort_values("strike", kind="stable")
    return result


def _linear_slope(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) < 2 or float(np.ptp(x_values)) <= 0:
        return math.nan, math.nan
    design = np.column_stack([np.ones(len(x_values)), x_values])
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted = design @ coefficients
    residual_sum = float(np.sum((y_values - fitted) ** 2))
    total_sum = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r_squared = 1.0 if total_sum <= 0 and residual_sum <= 1e-15 else (
        math.nan if total_sum <= 0 else 1.0 - residual_sum / total_sum
    )
    return float(coefficients[1]), float(r_squared)


def _common_strike_rows(
    current: pd.DataFrame,
    target: pd.DataFrame,
) -> list[dict[str, Any]]:
    current_map = {round(float(row.strike), 10): row for row in current.itertuples(index=False)}
    target_map = {round(float(row.strike), 10): row for row in target.itertuples(index=False)}
    rows = []
    for strike_key in sorted(set(current_map) & set(target_map)):
        left = current_map[strike_key]
        right = target_map[strike_key]
        rows.append(
            {
                "strike": float(left.strike),
                "current_q": float(left.strike_over_forward),
                "target_q": float(right.strike_over_forward),
                "current_k": float(left.log_moneyness),
                "target_k": float(right.log_moneyness),
                "current_iv": float(left.implied_vol),
                "target_iv": float(right.implied_vol),
                "current_weight": float(left.weight_sum),
                "target_weight": float(right.weight_sum),
            }
        )
    return rows


def _select_pair_atm(common: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not common:
        return None
    return min(
        common,
        key=lambda row: (
            max(abs(float(row["current_k"])), abs(float(row["target_k"]))),
            (abs(float(row["current_k"])) + abs(float(row["target_k"]))) / 2.0,
            -(float(row["current_weight"]) + float(row["target_weight"])),
            float(row["strike"]),
        ),
    )


def _select_wing(
    rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> Mapping[str, Any] | None:
    if side == "left":
        eligible = [
            row
            for row in rows
            if float(row["current_k"]) < 0 and float(row["target_k"]) < 0
        ]
    else:
        eligible = [
            row
            for row in rows
            if float(row["current_k"]) > 0 and float(row["target_k"]) > 0
        ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            max(abs(float(row["current_k"])), abs(float(row["target_k"]))),
            (abs(float(row["current_k"])) + abs(float(row["target_k"]))) / 2.0,
            float(row["strike"]),
        ),
    )


def build_pair_slice_metrics(
    points: pd.DataFrame,
    pair_audit: pd.DataFrame,
    *,
    distance_thresholds: Sequence[float] = (0.01, 0.02, 0.05),
    skew_band: float = 0.05,
    min_skew_span: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build same-expiry, same-underlying, same-strike five-minute metrics."""

    slices = _slice_maps(points)
    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for pair in pair_audit.itertuples(index=False):
        current_slices = slices.get(str(pair.origin_time_utc), {})
        target_slices = slices.get(str(pair.target_time_utc), {})
        common_maturities = sorted(set(current_slices) & set(target_slices))
        pair_rows.append(
            {
                **pair._asdict(),
                "current_maturity_count": int(len(current_slices)),
                "target_maturity_count": int(len(target_slices)),
                "common_maturity_count": int(len(common_maturities)),
                "has_common_maturity": bool(common_maturities),
            }
        )
        # Cross-session pairs remain visible in market_pair_audit.csv with an
        # explicit exclusion reason, but they are not analytical slice pairs.
        if not bool(pair.same_cme_continuous_session):
            continue
        for maturity_date in common_maturities:
            current = current_slices[maturity_date]
            target = target_slices[maturity_date]
            current_underlyings = sorted(set(current["underlying_contract_id"].astype(str)))
            target_underlyings = sorted(set(target["underlying_contract_id"].astype(str)))
            same_underlying = (
                len(current_underlyings) == 1
                and len(target_underlyings) == 1
                and current_underlyings[0] == target_underlyings[0]
                and bool(current_underlyings[0])
            )
            underlying = current_underlyings[0] if same_underlying else ""
            base = {
                "slice_pair_id": _stable_id(
                    "slice_pair", pair.origin_time_utc, pair.target_time_utc, maturity_date, underlying
                ),
                "pair_id": str(pair.pair_id),
                "origin_time_utc": str(pair.origin_time_utc),
                "target_time_utc": str(pair.target_time_utc),
                "origin_time_london": str(pair.origin_time_london),
                "target_time_london": str(pair.target_time_london),
                "session_id": str(pair.session_id),
                "same_cme_continuous_session": bool(pair.same_cme_continuous_session),
                "maturity_date": str(maturity_date),
                "origin_business_days": int(current["business_days"].iloc[0]),
                "target_business_days": int(target["business_days"].iloc[0]),
                "underlying_contract_id": underlying,
                "same_underlying_contract": bool(same_underlying),
                "current_point_count": int(len(current)),
                "target_point_count": int(len(target)),
            }
            if not same_underlying:
                metric_rows.append({**base, "metric_status": "underlying_contract_mismatch"})
                continue
            common = _common_strike_rows(current, target)
            base["common_strike_count"] = int(len(common))
            atm = _select_pair_atm(common)
            if atm is None:
                metric_rows.append({**base, "metric_status": "no_common_nominal_strike"})
                continue
            max_atm_distance = max(
                abs(float(atm["current_q"]) - 1.0),
                abs(float(atm["target_q"]) - 1.0),
            )
            atm_quality = _quality_from_distance(max_atm_distance, distance_thresholds)
            row = {
                **base,
                "metric_status": "ok",
                "pair_atm_strike": float(atm["strike"]),
                "current_atm_q": float(atm["current_q"]),
                "target_atm_q": float(atm["target_q"]),
                "current_atm_k": float(atm["current_k"]),
                "target_atm_k": float(atm["target_k"]),
                "current_atm_distance": abs(float(atm["current_q"]) - 1.0),
                "target_atm_distance": abs(float(atm["target_q"]) - 1.0),
                "max_atm_distance": float(max_atm_distance),
                "pair_atm_quality": atm_quality,
                "current_atm_iv": float(atm["current_iv"]),
                "target_atm_iv": float(atm["target_iv"]),
                "delta_atm_iv": float(atm["target_iv"]) - float(atm["current_iv"]),
                "abs_delta_atm_iv": abs(float(atm["target_iv"]) - float(atm["current_iv"])),
                "delta_atm_iv_pct_points": (
                    float(atm["target_iv"]) - float(atm["current_iv"])
                ) * 100.0,
            }
            band_rows = [
                item
                for item in common
                if abs(float(item["current_q"]) - 1.0) <= float(skew_band)
                and abs(float(item["target_q"]) - 1.0) <= float(skew_band)
            ]
            left = _select_wing(band_rows, side="left")
            right = _select_wing(band_rows, side="right")
            row["common_near_atm_strike_count"] = int(len(band_rows))
            row["has_common_two_sided_support"] = bool(left is not None and right is not None)
            row["skew_quality"] = "unusable"
            row["skew_status"] = "missing_two_sided_support"
            if left is not None and right is not None:
                current_span = float(right["current_k"]) - float(left["current_k"])
                target_span = float(right["target_k"]) - float(left["target_k"])
                row.update(
                    {
                        "skew_left_strike": float(left["strike"]),
                        "skew_right_strike": float(right["strike"]),
                        "current_skew_k_span": current_span,
                        "target_skew_k_span": target_span,
                    }
                )
                if current_span >= float(min_skew_span) and target_span >= float(min_skew_span):
                    current_skew = (
                        float(right["current_iv"]) - float(left["current_iv"])
                    ) / current_span
                    target_skew = (
                        float(right["target_iv"]) - float(left["target_iv"])
                    ) / target_span
                    row.update(
                        {
                            "current_atm_iv_skew_secant": current_skew,
                            "target_atm_iv_skew_secant": target_skew,
                            "delta_atm_iv_skew_secant": target_skew - current_skew,
                            "abs_delta_atm_iv_skew_secant": abs(target_skew - current_skew),
                            "skew_status": "ok",
                        }
                    )
                    if len(band_rows) >= 3:
                        current_ols, current_r2 = _linear_slope(
                            [item["current_k"] for item in band_rows],
                            [item["current_iv"] for item in band_rows],
                        )
                        target_ols, target_r2 = _linear_slope(
                            [item["target_k"] for item in band_rows],
                            [item["target_iv"] for item in band_rows],
                        )
                        row.update(
                            {
                                "current_atm_iv_skew_ols": current_ols,
                                "target_atm_iv_skew_ols": target_ols,
                                "delta_atm_iv_skew_ols": target_ols - current_ols,
                                "current_skew_ols_r2": current_r2,
                                "target_skew_ols_r2": target_r2,
                            }
                        )
                    if len(band_rows) >= 3 and max_atm_distance <= distance_thresholds[0]:
                        row["skew_quality"] = "A"
                    elif len(band_rows) >= 3 and max_atm_distance <= distance_thresholds[1]:
                        row["skew_quality"] = "B"
                    elif len(band_rows) >= 2 and max_atm_distance <= distance_thresholds[2]:
                        row["skew_quality"] = "C"
                else:
                    row["skew_status"] = "log_moneyness_span_below_minimum"
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    return metrics.sort_values(
        ["origin_time_utc", "maturity_date", "underlying_contract_id"], kind="stable"
    ).reset_index(drop=True), pairs.sort_values("origin_time_utc", kind="stable").reset_index(drop=True)


def add_rolling_jump_skewness(
    metrics: pd.DataFrame,
    *,
    windows: Mapping[int, int] | None = None,
    max_gap_minutes: int = 5,
) -> pd.DataFrame:
    """Add descriptive rolling Fisher-Pearson skewness of signed ATM jumps."""

    windows = windows or {30: 15, 60: 30, 120: 60}
    result = metrics.copy()
    for window in windows:
        result[f"rolling_jump_skew_{int(window)}m"] = np.nan
        result[f"rolling_jump_skew_{int(window)}m_n"] = 0
        result[f"delta_rolling_jump_skew_{int(window)}m"] = np.nan
    required = {
        "metric_status",
        "pair_atm_quality",
        "delta_atm_iv",
        "origin_time_utc",
        "maturity_date",
        "underlying_contract_id",
        "session_id",
    }
    if not required.issubset(result.columns):
        result["rolling_skew_quality"] = "unusable"
        return result
    eligible = result[
        result["metric_status"].astype(str).eq("ok")
        & result["pair_atm_quality"].astype(str).isin(["A", "B"])
        & pd.to_numeric(result["delta_atm_iv"], errors="coerce").notna()
    ].copy()
    if eligible.empty:
        return result
    eligible["origin_timestamp"] = pd.to_datetime(eligible["origin_time_utc"], utc=True)
    group_columns = ["maturity_date", "underlying_contract_id", "session_id"]
    for _, group in eligible.groupby(group_columns, sort=False):
        group = group.sort_values("origin_timestamp", kind="stable")
        gaps = group["origin_timestamp"].diff().dt.total_seconds().div(60.0)
        run_ids = (gaps.isna() | (gaps > float(max_gap_minutes))).cumsum()
        for _, run in group.groupby(run_ids, sort=False):
            run = run.sort_values("origin_timestamp", kind="stable")
            series = pd.Series(
                run["delta_atm_iv"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(run["origin_timestamp"]),
            )
            for window, minimum in windows.items():
                rolling = series.rolling(
                    f"{int(window)}min",
                    min_periods=int(minimum),
                    closed="both",
                )
                skew_values = rolling.skew().to_numpy(dtype=float)
                counts = (
                    series.rolling(f"{int(window)}min", min_periods=1, closed="both")
                    .count()
                    .to_numpy(dtype=int)
                )
                # Compare to the previous *valid* rolling estimate.  Sparse
                # observations can temporarily drop a clock-time window below
                # min_periods even when no >5 minute reset occurs; a plain
                # np.diff would incorrectly discard the first recovered change.
                valid_skew = pd.Series(skew_values, dtype=float)
                deltas = (valid_skew - valid_skew.ffill().shift(1)).to_numpy(dtype=float)
                deltas[~np.isfinite(skew_values)] = np.nan
                result.loc[run.index, f"rolling_jump_skew_{int(window)}m"] = skew_values
                result.loc[run.index, f"rolling_jump_skew_{int(window)}m_n"] = counts
                result.loc[
                    run.index, f"delta_rolling_jump_skew_{int(window)}m"
                ] = deltas
    result["rolling_skew_quality"] = np.where(
        result["delta_rolling_jump_skew_60m"].notna(), "B", "unusable"
    )
    return result


def _maturity_bucket(value: Any) -> str:
    try:
        business_days = int(value)
    except (TypeError, ValueError):
        return "outside"
    for lower, upper, label in DEFAULT_MATURITY_BUCKETS:
        if lower <= business_days <= upper:
            return label
    return "outside"


def _robust_scale(values: np.ndarray) -> tuple[float, str]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return math.nan, "none"
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0:
        return 1.4826 * mad, "mad"
    q75, q25 = np.percentile(finite, [75, 25])
    iqr_scale = float((q75 - q25) / 1.349)
    if iqr_scale > 0:
        return iqr_scale, "iqr"
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return (std, "std") if std > 0 else (math.nan, "none")


def score_metric_changes(
    metrics: pd.DataFrame,
    *,
    min_group_size: int = 30,
    tier_thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    tier_thresholds = tier_thresholds or {
        "high": {"percentile": 0.999, "robust_z": 6.0},
        "primary": {"percentile": 0.995, "robust_z": 5.0},
        "broad": {"percentile": 0.99, "robust_z": 4.0},
    }
    specifications = (
        ("atm_iv_jump", "delta_atm_iv", "pair_atm_quality", False),
        ("smile_skew_jump", "delta_atm_iv_skew_secant", "skew_quality", False),
        ("rolling_jump_skew_change", "delta_rolling_jump_skew_60m", "rolling_skew_quality", True),
    )
    rows: list[pd.DataFrame] = []
    base_columns = [
        "slice_pair_id",
        "pair_id",
        "origin_time_utc",
        "target_time_utc",
        "session_id",
        "maturity_date",
        "origin_business_days",
        "underlying_contract_id",
    ]
    for metric_name, value_column, quality_column, rolling_only in specifications:
        if value_column not in metrics.columns or quality_column not in metrics.columns:
            continue
        frame = metrics[base_columns + [value_column, quality_column]].copy()
        frame = frame.rename(columns={value_column: "metric_change", quality_column: "metric_quality"})
        frame["metric_name"] = metric_name
        frame["rolling_only_metric"] = bool(rolling_only)
        frame["metric_change"] = pd.to_numeric(frame["metric_change"], errors="coerce")
        frame = frame[
            frame["metric_change"].notna()
            & frame["metric_quality"].astype(str).isin(["A", "B", "C"])
        ]
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    ranking = pd.concat(rows, ignore_index=True)
    ranking["abs_metric_change"] = ranking["metric_change"].abs()
    ranking["calendar_year"] = pd.to_datetime(
        ranking["origin_time_utc"], utc=True
    ).dt.year.astype(int)
    ranking["maturity_bucket"] = ranking["origin_business_days"].map(_maturity_bucket)
    ranking["robust_center"] = np.nan
    ranking["robust_scale"] = np.nan
    ranking["robust_scale_method"] = ""
    ranking["robust_z"] = np.nan
    ranking["abs_robust_z"] = np.nan
    ranking["abs_empirical_percentile"] = np.nan
    ranking["scoring_group_size"] = 0
    group_columns = ["metric_name", "calendar_year", "maturity_bucket"]
    for _, group in ranking.groupby(group_columns, sort=False):
        indices = group.index
        values = group["metric_change"].to_numpy(dtype=float)
        center = float(np.median(values))
        scale, method = _robust_scale(values)
        percentiles = group["abs_metric_change"].rank(method="max", pct=True).to_numpy()
        ranking.loc[indices, "robust_center"] = center
        ranking.loc[indices, "robust_scale"] = scale
        ranking.loc[indices, "robust_scale_method"] = method
        ranking.loc[indices, "scoring_group_size"] = int(len(group))
        ranking.loc[indices, "abs_empirical_percentile"] = percentiles
        if math.isfinite(scale) and scale > 0 and len(group) >= int(min_group_size):
            z_values = (values - center) / scale
            ranking.loc[indices, "robust_z"] = z_values
            ranking.loc[indices, "abs_robust_z"] = np.abs(z_values)
    tiers: list[str] = []
    for row in ranking.itertuples(index=False):
        quality = str(row.metric_quality)
        percentile = float(row.abs_empirical_percentile)
        z_value = float(row.abs_robust_z) if math.isfinite(float(row.abs_robust_z)) else -math.inf
        tier = "none"
        if int(row.scoring_group_size) < int(min_group_size):
            tiers.append(tier)
            continue
        if not bool(row.rolling_only_metric):
            high = tier_thresholds["high"]
            primary = tier_thresholds["primary"]
            if quality == "A" and percentile >= float(high["percentile"]) and z_value >= float(high["robust_z"]):
                tier = "high"
            elif quality in {"A", "B"} and percentile >= float(primary["percentile"]) and z_value >= float(primary["robust_z"]):
                tier = "primary"
        broad = tier_thresholds["broad"]
        if tier == "none" and quality in {"A", "B", "C"} and (
            percentile >= float(broad["percentile"]) or z_value >= float(broad["robust_z"])
        ):
            tier = "broad"
        tiers.append(tier)
    ranking["anomaly_tier"] = tiers
    ranking["anomaly_tier_order"] = ranking["anomaly_tier"].map(TIER_ORDER).astype(int)
    ranking["signed_direction"] = np.where(
        ranking["metric_change"] > 0, "up", np.where(ranking["metric_change"] < 0, "down", "flat")
    )
    ranking = ranking.sort_values(
        ["anomaly_tier_order", "abs_robust_z", "abs_empirical_percentile"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranking["metric_rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    return ranking


def aggregate_market_pairs(
    pair_audit: pd.DataFrame,
    metric_rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metric_rankings.empty:
        pair_rankings = pair_audit.copy()
        pair_rankings["anomaly_tier"] = "none"
        pair_rankings["anomaly_tier_order"] = 0
        return pair_rankings, pair_rankings.iloc[0:0].copy()
    summaries: list[dict[str, Any]] = []
    for pair_id, group in metric_rankings.groupby("pair_id", sort=False):
        best = group.sort_values(
            ["anomaly_tier_order", "abs_robust_z", "abs_empirical_percentile"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
        summaries.append(
            {
                "pair_id": str(pair_id),
                "anomaly_tier": str(best["anomaly_tier"]),
                "anomaly_tier_order": int(best["anomaly_tier_order"]),
                # Severity shown for a pair must describe the same highest-tier
                # row named by peak_metric_name.  An auxiliary broad rolling
                # signal may have a larger raw z-score, but it must not silently
                # become the displayed severity of a high ATM/smile event.
                "max_abs_robust_z": float(best["abs_robust_z"]),
                "max_abs_empirical_percentile": float(best["abs_empirical_percentile"]),
                "all_metric_max_abs_robust_z": float(group["abs_robust_z"].max(skipna=True)),
                "all_metric_max_abs_empirical_percentile": float(
                    group["abs_empirical_percentile"].max()
                ),
                "peak_metric_name": str(best["metric_name"]),
                "peak_metric_change": float(best["metric_change"]),
                "peak_metric_quality": str(best["metric_quality"]),
                "peak_maturity_date": str(best["maturity_date"]),
                "peak_signed_direction": str(best["signed_direction"]),
                "flagged_metric_names": ";".join(
                    sorted(set(group.loc[group["anomaly_tier"] != "none", "metric_name"].astype(str)))
                ),
                "flagged_maturity_dates": ";".join(
                    sorted(
                        set(
                            group.loc[
                                group["anomaly_tier"] != "none", "maturity_date"
                            ].astype(str)
                        )
                    )
                ),
                "flagged_directions": ";".join(
                    sorted(
                        set(
                            group.loc[
                                group["anomaly_tier"] != "none", "signed_direction"
                            ].astype(str)
                        )
                    )
                ),
                "flagged_slice_count": int(group.loc[group["anomaly_tier"] != "none", "slice_pair_id"].nunique()),
                "ranked_slice_metric_count": int(len(group)),
            }
        )
    summary = pd.DataFrame(summaries)
    pair_rankings = pair_audit.merge(summary, on="pair_id", how="left", validate="one_to_one")
    pair_rankings["anomaly_tier"] = pair_rankings["anomaly_tier"].fillna("none")
    pair_rankings["anomaly_tier_order"] = pair_rankings["anomaly_tier_order"].fillna(0).astype(int)
    pair_rankings = pair_rankings.sort_values(
        ["anomaly_tier_order", "max_abs_robust_z", "origin_time_utc"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    pair_rankings["pair_rank"] = np.arange(1, len(pair_rankings) + 1, dtype=int)
    candidates = pair_rankings[pair_rankings["anomaly_tier"] != "none"].copy()
    return pair_rankings, candidates


def cluster_candidate_episodes(
    candidates: pd.DataFrame,
    *,
    tolerance_minutes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = candidates.copy()
    if "pair_id" not in work.columns:
        raise ValueError("Candidate pairs are missing required column: pair_id")
    work = work.sort_values(
        ["anomaly_tier_order", "max_abs_robust_z", "max_abs_empirical_percentile"],
        ascending=[False, False, False],
        kind="stable",
    ).drop_duplicates("pair_id", keep="first")
    work["origin_timestamp"] = pd.to_datetime(work["origin_time_utc"], utc=True)
    work["target_timestamp"] = pd.to_datetime(work["target_time_utc"], utc=True)
    episodes: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for session_id, session_rows in work.groupby("session_id", sort=False):
        session_rows = session_rows.sort_values("origin_timestamp", kind="stable")
        current: list[Any] = []
        current_end: pd.Timestamp | None = None
        groups: list[list[Any]] = []
        for row in session_rows.itertuples(index=False):
            if current_end is None or row.origin_timestamp <= current_end + pd.Timedelta(minutes=int(tolerance_minutes)):
                current.append(row)
                current_end = max(current_end, row.target_timestamp) if current_end is not None else row.target_timestamp
            else:
                groups.append(current)
                current = [row]
                current_end = row.target_timestamp
        if current:
            groups.append(current)
        for episode_rows in groups:
            start = min(row.origin_timestamp for row in episode_rows)
            end = max(row.target_timestamp for row in episode_rows)
            episode_id = _stable_id(
                "episode", session_id, _utc_text(start), _utc_text(end), tolerance_minutes
            )
            peak = max(
                episode_rows,
                key=lambda row: (
                    int(row.anomaly_tier_order),
                    float(row.max_abs_robust_z) if math.isfinite(float(row.max_abs_robust_z)) else -math.inf,
                    float(row.max_abs_empirical_percentile),
                ),
            )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "cluster_tolerance_minutes": int(tolerance_minutes),
                    "episode_start_utc": _utc_text(start),
                    "episode_end_utc": _utc_text(end),
                    "episode_start_london": _london_text(start),
                    "episode_end_london": _london_text(end),
                    "session_id": str(session_id),
                    "anomaly_tier": str(peak.anomaly_tier),
                    "anomaly_tier_order": int(peak.anomaly_tier_order),
                    "peak_pair_id": str(peak.pair_id),
                    "peak_metric_name": str(peak.peak_metric_name),
                    "peak_metric_change": getattr(
                        peak, "peak_metric_change", math.nan
                    ),
                    "peak_metric_quality": str(
                        getattr(peak, "peak_metric_quality", "")
                    ),
                    "peak_maturity_date": str(
                        getattr(peak, "peak_maturity_date", "")
                    ),
                    "peak_signed_direction": str(
                        getattr(peak, "peak_signed_direction", "")
                    ),
                    "max_abs_robust_z": float(peak.max_abs_robust_z),
                    "max_abs_empirical_percentile": float(
                        peak.max_abs_empirical_percentile
                    ),
                    "all_member_max_abs_robust_z": max(
                        float(
                            getattr(
                                row,
                                "all_metric_max_abs_robust_z",
                                row.max_abs_robust_z,
                            )
                        )
                        for row in episode_rows
                    ),
                    "market_pair_count": int(len(episode_rows)),
                    "flagged_metric_names": ";".join(
                        sorted(
                            {
                                name
                                for row in episode_rows
                                for name in str(row.flagged_metric_names).split(";")
                                if name
                            }
                        )
                    ),
                    "flagged_maturity_dates": ";".join(
                        sorted(
                            {
                                value
                                for row in episode_rows
                                for value in str(
                                    getattr(row, "flagged_maturity_dates", "")
                                ).split(";")
                                if value
                            }
                        )
                    ),
                    "flagged_directions": ";".join(
                        sorted(
                            {
                                value
                                for row in episode_rows
                                for value in str(
                                    getattr(row, "flagged_directions", "")
                                ).split(";")
                                if value
                            }
                        )
                    ),
                }
            )
            for row in episode_rows:
                members.append(
                    {
                        "episode_id": episode_id,
                        "cluster_tolerance_minutes": int(tolerance_minutes),
                        "pair_id": str(row.pair_id),
                        "origin_time_utc": str(row.origin_time_utc),
                        "target_time_utc": str(row.target_time_utc),
                        "pair_anomaly_tier": str(row.anomaly_tier),
                        "peak_metric_name": str(
                            getattr(row, "peak_metric_name", "")
                        ),
                        "peak_metric_change": getattr(
                            row, "peak_metric_change", math.nan
                        ),
                        "peak_metric_quality": str(
                            getattr(row, "peak_metric_quality", "")
                        ),
                        "peak_maturity_date": str(
                            getattr(row, "peak_maturity_date", "")
                        ),
                        "peak_signed_direction": str(
                            getattr(row, "peak_signed_direction", "")
                        ),
                        "max_abs_robust_z": getattr(
                            row, "max_abs_robust_z", math.nan
                        ),
                        "flagged_metric_names": str(
                            getattr(row, "flagged_metric_names", "")
                        ),
                        "flagged_maturity_dates": str(
                            getattr(row, "flagged_maturity_dates", "")
                        ),
                        "flagged_directions": str(
                            getattr(row, "flagged_directions", "")
                        ),
                    }
                )
    episode_frame = pd.DataFrame(episodes).sort_values(
        ["anomaly_tier_order", "max_abs_robust_z", "episode_start_utc"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    episode_frame["episode_rank"] = np.arange(1, len(episode_frame) + 1, dtype=int)
    return episode_frame, pd.DataFrame(members)


def _read_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = payload.get("market_jump_detection", payload)
    if not isinstance(config, dict):
        raise ValueError("market_jump_detection config must be a mapping")
    return dict(config)


def _write_frame(frame: pd.DataFrame, path: Path, *, compressed: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".gz") if compressed and path.suffix != ".gz" else path
    frame.to_csv(target, index=False, compression="gzip" if target.suffix == ".gz" else None)
    return target


def _coverage_record(
    dataset: str,
    path: Path,
    frame: pd.DataFrame,
    *,
    primary_key: Sequence[str],
    timestamp_column: str = "",
    notes: str = "",
) -> dict[str, Any]:
    key_columns = [column for column in primary_key if column in frame.columns]
    unique_keys = (
        int(len(frame.drop_duplicates(key_columns))) if key_columns else int(len(frame))
    )
    minimum = ""
    maximum = ""
    unique_dates = 0
    invalid_timestamps = 0
    if timestamp_column and timestamp_column in frame.columns:
        timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        valid = timestamps.dropna()
        invalid_timestamps = int(timestamps.isna().sum())
        if not valid.empty:
            minimum = _utc_text(valid.min())
            maximum = _utc_text(valid.max())
            unique_dates = int(valid.dt.date.nunique())
    return {
        "dataset": dataset,
        "path": str(path),
        "row_count": int(len(frame)),
        "primary_key": ";".join(key_columns),
        "unique_primary_key_count": unique_keys,
        "duplicate_primary_key_count": int(len(frame) - unique_keys),
        "timestamp_column": timestamp_column,
        "min_timestamp_utc": minimum,
        "max_timestamp_utc": maximum,
        "unique_utc_dates": unique_dates,
        "invalid_timestamp_count": invalid_timestamps,
        "notes": notes,
    }


_FIELD_DESCRIPTIONS = {
    "anchor_time_utc": "Raw-surface snapshot anchor minute in UTC.",
    "anchor_time_london": "Snapshot anchor displayed in Europe/London local time.",
    "business_days": "Business days from snapshot to the actual option maturity.",
    "origin_business_days": "Business days to maturity at the current/origin snapshot.",
    "target_business_days": "Business days to maturity at the target snapshot.",
    "maturity_date": "Actual option expiry date recovered from precalibration lineage.",
    "underlying_contract_id": "Matched TY futures contract identifier.",
    "strike": "Observed nominal option strike.",
    "strike_over_forward": "Observed K/F ratio; legacy source name percent_strike.",
    "percent_strike": "Legacy alias of strike_over_forward (K/F, not a percentage).",
    "log_moneyness": "Natural logarithm of K/F.",
    "implied_vol": "Annualized Black-76 implied volatility in decimal units.",
    "implied_vol_pct": "Annualized Black-76 implied volatility in percentage units.",
    "weight_sum": "Sum of source aggregation weights at the nominal strike.",
    "observation_count": "Accepted precalibration rows aggregated into the strike point.",
    "slice_id": "Stable identifier for one snapshot and maturity slice.",
    "slice_point_count": "Number of observed strike-IV points in the slice.",
    "surface_parameter_match": "Whether reconstructed q/IV matches frozen params_json within tolerance.",
    "surface_q_difference": "Precalibration-reconstructed K/F minus frozen params_json K/F.",
    "surface_iv_difference": "Precalibration-reconstructed IV minus frozen params_json IV.",
    "abs_moneyness_distance": "Absolute observed ATM distance |K/F - 1|.",
    "atm_quality": "Nearest-observed ATM quality: A <=1%, B <=2%, C <=5%, otherwise outside.",
    "atm_iv_decimal": "Selected approximate ATM IV in annualized decimal units.",
    "atm_iv_pct": "Selected approximate ATM IV in annualized percentage units.",
    "atm_strike": "Nominal strike selected as the approximate ATM observation.",
    "pair_id": "Stable identifier for one origin-to-target five-minute market pair.",
    "slice_pair_id": "Stable identifier for one market pair and actual maturity.",
    "origin_time_utc": "Current snapshot anchor minute in UTC.",
    "target_time_utc": "Target snapshot anchor minute in UTC, exactly five minutes later.",
    "analysis_window_start_utc": "Start of the current backward-looking market window.",
    "analysis_window_end_utc": "End of the target market window.",
    "same_cme_continuous_session": "Whether both five-minute windows lie in one uninterrupted CME session.",
    "session_id": "CME continuous-session identifier.",
    "pair_exclusion_reason": "Reason the market pair is excluded before slice metrics.",
    "pair_atm_strike": "One common nominal strike used at both snapshots for ATM IV change.",
    "pair_atm_quality": "Common-strike ATM quality based on the worse endpoint distance.",
    "delta_atm_iv": "Target minus current annualized ATM IV at one common strike.",
    "delta_atm_iv_pct_points": "Target minus current ATM IV in percentage points.",
    "current_atm_iv_skew_secant": "Current local IV slope per unit log(K/F), using common two-sided strikes.",
    "target_atm_iv_skew_secant": "Target local IV slope per unit log(K/F), using common two-sided strikes.",
    "delta_atm_iv_skew_secant": "Target minus current local smile-skew secant slope.",
    "skew_quality": "Smile-skew support quality based on common strikes and ATM distance.",
    "skew_status": "Smile-skew calculation status or explicit rejection reason.",
    "rolling_skew_quality": "Quality flag for the 60-minute rolling jump-skew change.",
    "metric_name": "Ranked metric family: ATM IV, smile skew, or rolling jump skewness.",
    "metric_change": "Signed change in the metric's native unit.",
    "metric_quality": "Quality grade inherited from the underlying metric construction.",
    "maturity_bucket": "Business-day maturity stratum used for robust scoring.",
    "robust_z": "Signed median-centered robust z-score within year and maturity bucket.",
    "abs_empirical_percentile": "Empirical percentile of the absolute change within its scoring stratum.",
    "anomaly_tier": "High/primary/broad/none threshold result; not a causal label.",
    "peak_metric_name": "Highest-tier metric selected as the pair or episode peak.",
    "peak_metric_quality": "Quality grade of the selected peak metric.",
    "peak_signed_direction": "Up/down/flat direction of the selected peak metric.",
    "all_metric_max_abs_robust_z": "Largest z-score across every metric in the pair, including auxiliary broad signals.",
    "episode_id": "Stable identifier for a clustered set of anomalous market pairs.",
    "episode_start_utc": "First origin minute in the episode, UTC.",
    "episode_end_utc": "Last target boundary in the episode, UTC.",
    "window_relation": "Evidence position: pre_context, impact, or post_reporting.",
    "news_row_id": "One-based row number in the frozen canonical Factiva workbook.",
    "news_available_time_utc": "Factiva PD+ET parsed from Europe/London to availability UTC.",
    "release_time_utc": "Official scheduled release timestamp in UTC.",
    "workbook_sha256": "SHA-256 of the canonical source workbook.",
    "lp_text_sha256": "SHA-256 of normalized lead-paragraph text for duplicate auditing.",
    "headline": "Factiva headline retained as candidate evidence.",
    "lead_paragraph": "Factiva lead paragraph retained as candidate evidence.",
}


def _field_metadata(table: str, column: str, dtype: Any, primary_key: Sequence[str]) -> dict[str, str]:
    description = _FIELD_DESCRIPTIONS.get(column, "")
    if not description:
        if column.endswith("_utc"):
            description = "UTC timestamp serialized in ISO-8601 form."
        elif column.endswith("_london"):
            description = "Europe/London display timestamp; UTC remains the join key."
        elif column.endswith("_sha256"):
            description = "SHA-256 lineage or content digest."
        elif column.endswith("_id"):
            description = "Stable identifier retained for joins and lineage."
        elif column.endswith("_count") or column.endswith("_n"):
            description = "Integer observation or candidate count."
        elif column.endswith("_quality"):
            description = "Quality grade used for eligibility or audit."
        elif column.endswith("_status"):
            description = "Calculation, parsing, or validation status."
        elif column.endswith("_rank"):
            description = "One-based deterministic rank."
        elif column.startswith("delta_"):
            description = "Target/current or adjacent-time signed difference; see the metric name for units."
        elif column.startswith("abs_") or column.startswith("max_abs_"):
            description = "Absolute magnitude used for ranking or quality assessment."
        else:
            description = f"Exported {table} field retained for analytical audit and reproducibility."
    unit = ""
    lowered = column.lower()
    if lowered.endswith("_utc") or lowered.endswith("_london") or "timestamp" in lowered:
        unit = "timestamp"
    elif "minutes" in lowered:
        unit = "minutes"
    elif "business_days" in lowered:
        unit = "business days"
    elif "pct_points" in lowered:
        unit = "percentage points"
    elif lowered.endswith("_pct"):
        unit = "percent"
    elif "skew" in lowered:
        unit = "IV per log(K/F) or dimensionless rolling skew; see description"
    elif "implied_vol" in lowered or "atm_iv" in lowered:
        unit = "annualized decimal IV"
    elif lowered in {"strike_over_forward", "percent_strike", "log_moneyness"}:
        unit = "ratio/log-ratio"
    elif lowered.endswith("_count") or lowered.endswith("_n"):
        unit = "count"
    role = "primary_key" if column in primary_key else "field"
    if column.endswith("_id") and role == "field":
        role = "join_key"
    elif column.endswith("_utc") or column.endswith("_london"):
        role = "time"
    elif "quality" in column or "status" in column or "reason" in column:
        role = "quality_audit"
    elif "sha256" in column or column in {"source_file", "option_contract_ids"}:
        role = "lineage"
    return {
        "table": table,
        "column": column,
        "dtype": str(dtype),
        "description": description,
        "unit": unit,
        "role": role,
    }


def _index_counts(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("surface_anchor", "precalib_point", "valid_pair")
        }
    finally:
        connection.close()


def _git_state() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        return {"commit": commit, "status": status}
    except Exception as exc:  # pragma: no cover - defensive provenance fallback
        return {"commit": "", "status": f"unavailable:{exc}"}


def run_market_jump_detection(
    config_path: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Run the complete market-first analysis and write a frozen output archive."""

    config = _read_config(config_path)
    inputs = dict(config.get("inputs", {}))
    analysis = dict(config.get("analysis", {}))
    reporting = dict(config.get("report", {}))
    market_index = _resolve_path(inputs["market_index_sqlite"])
    session_calendar = _resolve_path(inputs["session_calendar_csv"])
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved = {
        "market_jump_detection": {
            **config,
            "inputs": {
                **inputs,
                "market_index_sqlite": str(market_index),
                "session_calendar_csv": str(session_calendar),
            },
        }
    }
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    points, extraction_audit = extract_surface_points(market_index)
    distance_thresholds = tuple(analysis.get("atm_distance_thresholds", [0.01, 0.02, 0.05]))
    points, atm = select_approximate_atm(points, distance_thresholds=distance_thresholds)
    pair_audit = load_valid_pairs(market_index, session_calendar)
    metrics, pair_audit = build_pair_slice_metrics(
        points,
        pair_audit,
        distance_thresholds=distance_thresholds,
        skew_band=float(analysis.get("skew_band", 0.05)),
        min_skew_span=float(analysis.get("min_skew_span", 0.005)),
    )
    rolling_windows = {
        int(window): int(minimum)
        for window, minimum in dict(
            analysis.get("rolling_skew_windows", {30: 15, 60: 30, 120: 60})
        ).items()
    }
    metrics = add_rolling_jump_skewness(
        metrics,
        windows=rolling_windows,
        max_gap_minutes=int(analysis.get("rolling_max_gap_minutes", 5)),
    )
    rankings = score_metric_changes(
        metrics,
        min_group_size=int(analysis.get("min_scoring_group_size", 30)),
        tier_thresholds=analysis.get("tier_thresholds"),
    )
    pair_rankings, candidates = aggregate_market_pairs(pair_audit, rankings)
    episodes, episode_members = cluster_candidate_episodes(
        candidates, tolerance_minutes=int(analysis.get("episode_tolerance_minutes", 5))
    )
    sensitivity_episodes = []
    sensitivity_members = []
    for tolerance in analysis.get("episode_sensitivity_minutes", [0, 10]):
        frame, members = cluster_candidate_episodes(candidates, tolerance_minutes=int(tolerance))
        sensitivity_episodes.append(frame)
        sensitivity_members.append(members)

    paths = {
        "surface_points": _write_frame(points, output_root / "all_surface_points.csv", compressed=True),
        "atm_observations": _write_frame(atm, output_root / "atm_observations.csv", compressed=True),
        "pair_slice_metrics": _write_frame(metrics, output_root / "pair_slice_metrics.csv", compressed=True),
        "metric_rankings": _write_frame(rankings, output_root / "metric_rankings.csv", compressed=True),
        "pair_audit": _write_frame(pair_audit, output_root / "market_pair_audit.csv"),
        "pair_rankings": _write_frame(pair_rankings, output_root / "market_pair_rankings.csv", compressed=True),
        "candidate_pairs": _write_frame(candidates, output_root / "candidate_pairs.csv"),
        "episodes": _write_frame(episodes, output_root / "candidate_episodes.csv"),
        "episode_members": _write_frame(episode_members, output_root / "episode_pair_members.csv"),
    }
    sensitivity_episode_frame = (
        pd.concat(sensitivity_episodes, ignore_index=True) if sensitivity_episodes else pd.DataFrame()
    )
    sensitivity_member_frame = (
        pd.concat(sensitivity_members, ignore_index=True) if sensitivity_members else pd.DataFrame()
    )
    paths["episode_sensitivity"] = _write_frame(
        sensitivity_episode_frame, output_root / "episode_sensitivity.csv"
    )
    paths["episode_sensitivity_members"] = _write_frame(
        sensitivity_member_frame, output_root / "episode_sensitivity_members.csv"
    )

    official_bridge = pd.DataFrame()
    news_bridge = pd.DataFrame()
    official_events = pd.DataFrame()
    news = pd.DataFrame()
    official_calendars: list[Path] = []
    news_path: Path | None = None
    alignment_path: Path | None = None
    if inputs.get("official_event_calendars"):
        from scripts.rq3.market_jump_news import (
            OFFICIAL_BRIDGE_COLUMNS,
            build_episode_official_bridge,
        )

        calendars = []
        for value in inputs["official_event_calendars"]:
            calendar_path = _resolve_path(value)
            official_calendars.append(calendar_path)
            calendars.append(pd.read_csv(calendar_path, dtype=str, keep_default_na=False))
        official_events = pd.concat(calendars, ignore_index=True)
        if official_events["event_id"].duplicated().any():
            raise ValueError("Official event calendars contain duplicate event_id values")
        official_bridge = (
            build_episode_official_bridge(
                episodes,
                official_events,
                pre_minutes=int(analysis.get("news_pre_minutes", 5)),
                post_minutes=int(analysis.get("news_post_minutes", 15)),
            )
            if not episodes.empty
            else pd.DataFrame(columns=OFFICIAL_BRIDGE_COLUMNS)
        )
        paths["official_events"] = _write_frame(
            official_events, output_root / "official_macro_events_2022_2023.csv"
        )
        paths["official_bridge"] = _write_frame(
            official_bridge, output_root / "episode_official_event_bridge.csv"
        )
    if inputs.get("news_xlsx"):
        from scripts.rq3.market_jump_news import (
            NEWS_BRIDGE_COLUMNS,
            build_episode_news_bridge,
            parse_factiva_news,
        )

        news_path = _resolve_path(inputs["news_xlsx"])
        news = parse_factiva_news(
            news_path,
            source_timezone=str(inputs.get("news_source_timezone", "Europe/London")),
        )
        alignment = None
        if inputs.get("news_alignment_csv"):
            alignment_path = _resolve_path(inputs["news_alignment_csv"])
            alignment = pd.read_csv(alignment_path, dtype=str, keep_default_na=False)
        news_bridge = (
            build_episode_news_bridge(
                episodes,
                news,
                alignment=alignment,
                pre_minutes=int(analysis.get("news_pre_minutes", 5)),
                post_minutes=int(analysis.get("news_post_minutes", 15)),
            )
            if not episodes.empty
            else pd.DataFrame(columns=NEWS_BRIDGE_COLUMNS)
        )
        paths["factiva_news"] = _write_frame(
            news, output_root / "factiva_news_time_audit.csv", compressed=True
        )
        paths["news_bridge"] = _write_frame(
            news_bridge, output_root / "episode_news_bridge.csv"
        )

    index_counts = _index_counts(market_index)
    quality_rows = [
        ("surface_anchor_count", index_counts["surface_anchor"], 69227),
        ("precalib_point_count", index_counts["precalib_point"], 1322526),
        ("accepted_precalib_row_count", extraction_audit.accepted_precalib_rows, 1100437),
        ("reconstructed_slice_count", extraction_audit.reconstructed_slices, 151216),
        ("reconstructed_surface_point_count", extraction_audit.reconstructed_points, 613460),
        ("atm_observation_count", len(atm), 151216),
        ("exact_atm_observation_count", int(atm["is_exact_atm"].sum()), 1934),
        ("valid_pair_count", index_counts["valid_pair"], 34480),
        ("same_session_pair_count", int(pair_audit["same_cme_continuous_session"].sum()), 34472),
        ("cross_session_pair_count", int((~pair_audit["same_cme_continuous_session"]).sum()), 8),
        ("pair_without_common_maturity_count", int((~pair_audit["has_common_maturity"]).sum()), 22),
        (
            "pair_excluded_before_slice_metrics_count",
            int(
                (
                    (~pair_audit["same_cme_continuous_session"])
                    | (~pair_audit["has_common_maturity"])
                ).sum()
            ),
            30,
        ),
        ("pair_slice_metric_count", len(metrics), 69977),
        # The frozen index contains known historical reconstruction drift in
        # params_json versus the persisted precalibration audit rows.  The
        # indexed params are authoritative and the differences remain explicit
        # point-level lineage fields.
        ("surface_parameter_mismatch_slices", extraction_audit.surface_parameter_mismatch_slices, 3019),
        ("surface_q_mismatch_slices", extraction_audit.surface_q_mismatch_slices, 2624),
        ("surface_iv_mismatch_slices", extraction_audit.surface_iv_mismatch_slices, 1031),
        ("ambiguous_maturity_slices", extraction_audit.ambiguous_maturity_slices, 0),
    ]
    if not official_events.empty:
        quality_rows.append(("official_event_count", len(official_events), 168))
    if not news.empty:
        quality_rows.extend(
            [
                ("factiva_news_row_count", len(news), 14900),
                (
                    "factiva_invalid_timestamp_count",
                    int((~news["timestamp_parse_status"].astype(str).eq("ok")).sum()),
                    0,
                ),
            ]
        )
    quality = pd.DataFrame(
        [
            {
                "check": check,
                "actual": int(actual),
                "expected": int(expected),
                "status": "pass" if int(actual) == int(expected) else "fail",
            }
            for check, actual, expected in quality_rows
        ]
    )
    paths["data_quality"] = _write_frame(
        quality, output_root / "data_quality_summary.csv"
    )
    enforce = bool(config.get("validation", {}).get("enforce_expected_counts", True))
    if enforce and quality["status"].eq("fail").any():
        failures = quality.loc[quality["status"] == "fail", ["check", "actual", "expected"]]
        raise ValueError(f"Full-market validation failed:\n{failures.to_string(index=False)}")

    exclusion_rows = [
        {
            "stage": "approximate_atm_observation",
            "reason": "outside_5pct_observed_atm_distance",
            "count": int(atm["atm_quality"].astype(str).eq("outside_5pct").sum()),
            "denominator": int(len(atm)),
            "disposition": "retained_in_full_table_excluded_from_formal_ranking",
        },
        {
            "stage": "market_pair",
            "reason": "cross_or_closed_cme_session",
            "count": int((~pair_audit["same_cme_continuous_session"]).sum()),
            "denominator": int(len(pair_audit)),
            "disposition": "retained_in_pair_audit_excluded_before_slice_metrics",
        },
        {
            "stage": "market_pair",
            "reason": "no_common_actual_maturity",
            "count": int((~pair_audit["has_common_maturity"]).sum()),
            "denominator": int(len(pair_audit)),
            "disposition": "retained_in_pair_audit_excluded_before_slice_metrics",
        },
        {
            "stage": "slice_pair_atm",
            "reason": "no_common_nominal_strike",
            "count": int(metrics["metric_status"].astype(str).eq("no_common_nominal_strike").sum()),
            "denominator": int(len(metrics)),
            "disposition": "retained_with_status_excluded_from_atm_ranking",
        },
        {
            "stage": "slice_pair_atm",
            "reason": "underlying_contract_mismatch",
            "count": int(metrics["metric_status"].astype(str).eq("underlying_contract_mismatch").sum()),
            "denominator": int(len(metrics)),
            "disposition": "retained_with_status_excluded_from_all_rankings",
        },
        {
            "stage": "slice_pair_atm",
            "reason": "common_strike_atm_distance_outside_5pct",
            "count": int(metrics["pair_atm_quality"].astype(str).eq("outside_5pct").sum()),
            "denominator": int(len(metrics)),
            "disposition": "retained_with_quality_excluded_from_atm_ranking",
        },
        {
            "stage": "slice_pair_smile_skew",
            "reason": "missing_common_two_sided_support",
            "count": int(metrics["skew_status"].astype(str).eq("missing_two_sided_support").sum()),
            "denominator": int(len(metrics)),
            "disposition": "retained_with_status_excluded_from_skew_ranking",
        },
        {
            "stage": "slice_pair_smile_skew",
            "reason": "log_moneyness_span_below_minimum",
            "count": int(
                metrics["skew_status"].astype(str).eq("log_moneyness_span_below_minimum").sum()
            ),
            "denominator": int(len(metrics)),
            "disposition": "retained_with_status_excluded_from_skew_ranking",
        },
        {
            "stage": "factiva_timestamp",
            "reason": "timestamp_parse_failure",
            "count": int((~news["timestamp_parse_status"].astype(str).eq("ok")).sum())
            if not news.empty
            else 0,
            "denominator": int(len(news)),
            "disposition": "excluded_from_news_bridge",
        },
        {
            "stage": "surface_reconciliation",
            "reason": "frozen_params_vs_precalib_drift",
            "count": int(extraction_audit.surface_parameter_mismatch_slices),
            "denominator": int(extraction_audit.reconstructed_slices),
            "disposition": "audit_only_frozen_params_json_remains_authoritative",
            "max_abs_q_difference": extraction_audit.max_abs_surface_q_difference,
            "max_abs_iv_difference": extraction_audit.max_abs_surface_iv_difference,
        },
    ]
    exclusions = pd.DataFrame(exclusion_rows)
    paths["exclusion_reasons"] = _write_frame(
        exclusions, output_root / "exclusion_reason_summary.csv"
    )

    table_specs: list[tuple[str, Path, pd.DataFrame, Sequence[str], str, str]] = [
        ("all_surface_points", paths["surface_points"], points, ["slice_id", "point_index"], "anchor_time_utc", "Frozen q/IV with precalibration lineage retained."),
        ("atm_observations", paths["atm_observations"], atm, ["slice_id"], "anchor_time_utc", "One nearest observed strike per snapshot and actual maturity."),
        ("pair_slice_metrics", paths["pair_slice_metrics"], metrics, ["slice_pair_id"], "origin_time_utc", "Same-session, same-maturity slice pairs only."),
        ("metric_rankings", paths["metric_rankings"], rankings, ["metric_name", "slice_pair_id"], "origin_time_utc", "Market metrics ranked before any news join."),
        ("market_pair_audit", paths["pair_audit"], pair_audit, ["pair_id"], "origin_time_utc", "All valid_pair rows, including explicit exclusions."),
        ("market_pair_rankings", paths["pair_rankings"], pair_rankings, ["pair_id"], "origin_time_utc", "One row per unique market pair."),
        ("candidate_pairs", paths["candidate_pairs"], candidates, ["pair_id"], "origin_time_utc", "High, primary, and broad market pairs."),
        ("candidate_episodes", paths["episodes"], episodes, ["episode_id"], "episode_start_utc", "Five-minute tolerance is the primary clustering result."),
        ("episode_pair_members", paths["episode_members"], episode_members, ["episode_id", "pair_id"], "origin_time_utc", "One-to-many episode-to-market-pair bridge."),
        ("episode_sensitivity", paths["episode_sensitivity"], sensitivity_episode_frame, ["cluster_tolerance_minutes", "episode_id"], "episode_start_utc", "Episode clustering at 0 and 10 minute tolerances."),
        ("episode_sensitivity_members", paths["episode_sensitivity_members"], sensitivity_member_frame, ["cluster_tolerance_minutes", "episode_id", "pair_id"], "origin_time_utc", "Sensitivity episode membership bridge."),
        ("data_quality_summary", paths["data_quality"], quality, ["check"], "", "Frozen expected-count acceptance checks."),
        ("exclusion_reason_summary", paths["exclusion_reasons"], exclusions, ["stage", "reason"], "", "Explicit exclusion and audit-only reasons by stage."),
    ]
    if not official_events.empty:
        table_specs.extend(
            [
                ("official_macro_events", paths["official_events"], official_events, ["event_id"], "release_time_utc", "Independent scheduled-event evidence lane."),
                ("episode_official_event_bridge", paths["official_bridge"], official_bridge, ["episode_official_bridge_id"], "release_time_utc", "One-to-many temporal official-event candidates."),
            ]
        )
    if not news.empty:
        table_specs.extend(
            [
                ("factiva_news_time_audit", paths["factiva_news"], news, ["workbook_sha256", "news_row_id"], "news_available_time_utc", "Canonical Europe/London timestamp parsing; embeddings are not loaded."),
                ("episode_news_bridge", paths["news_bridge"], news_bridge, ["episode_news_bridge_id"], "news_available_time_utc", "One-to-many temporal Factiva candidates; no forced headline selection."),
            ]
        )

    coverage = pd.DataFrame(
        [
            _coverage_record(
                name,
                path,
                frame,
                primary_key=primary_key,
                timestamp_column=timestamp_column,
                notes=notes,
            )
            for name, path, frame, primary_key, timestamp_column, notes in table_specs
        ]
    )
    paths["dataset_coverage"] = _write_frame(
        coverage, output_root / "dataset_coverage.csv"
    )

    dictionary_rows = [
        _field_metadata(name, column, frame[column].dtype, primary_key)
        for name, _, frame, primary_key, _, _ in table_specs
        for column in frame.columns
    ]
    field_dictionary_frame = pd.DataFrame(dictionary_rows)
    paths["field_dictionary_csv"] = _write_frame(
        field_dictionary_frame, output_root / "field_dictionary.csv"
    )
    field_dictionary = {
        "schema_version": "1.0",
        "table_count": int(len(table_specs)),
        "field_definition_count": int(len(field_dictionary_frame)),
        "tables": {
            name: {
                "path": str(path),
                "row_count": int(len(frame)),
                "primary_key": list(primary_key),
                "fields": field_dictionary_frame.loc[
                    field_dictionary_frame["table"].eq(name),
                    ["column", "dtype", "description", "unit", "role"],
                ].to_dict(orient="records"),
            }
            for name, path, frame, primary_key, _, _ in table_specs
        },
    }
    field_dictionary_path = output_root / "field_dictionary.json"
    field_dictionary_path.write_text(
        json.dumps(field_dictionary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["field_dictionary_json"] = field_dictionary_path

    source_paths = [market_index, session_calendar, *official_calendars]
    if news_path is not None:
        source_paths.append(news_path)
    if alignment_path is not None:
        source_paths.append(alignment_path)
    code_paths = [
        Path(__file__).resolve(),
        (ROOT / "scripts/rq3/market_jump_news.py").resolve(),
        (ROOT / "scripts/rq3/market_jump_reporting.py").resolve(),
        (ROOT / "scripts/rq3/main.py").resolve(),
        Path(config_path).resolve(),
    ]
    source_manifest = {
        "git": _git_state(),
        "code_files": [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in code_paths
        ],
        "sources": [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for path in source_paths
        ],
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (output_root / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation_summary = {
        "status": "pass" if quality["status"].eq("pass").all() else "needs_review",
        "market_discovery_precedes_news_join": True,
        "counts": {row["check"]: int(row["actual"]) for _, row in quality.iterrows()},
        "candidate_pair_count": int(len(candidates)),
        "candidate_episode_count": int(len(episodes)),
        "official_bridge_rows": int(len(official_bridge)),
        "news_bridge_rows": int(len(news_bridge)),
        "causal_claim_supported": False,
        "causal_language": "time-associated candidate shock only",
        "surface_params_json_is_authoritative": True,
        "max_abs_precalib_vs_surface_q_difference": extraction_audit.max_abs_surface_q_difference,
        "max_abs_precalib_vs_surface_iv_difference": extraction_audit.max_abs_surface_iv_difference,
        "surface_parameter_drift_slice_rate": (
            extraction_audit.surface_parameter_mismatch_slices
            / extraction_audit.reconstructed_slices
        ),
    }
    (output_root / "validation_summary.json").write_text(
        json.dumps(validation_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if bool(reporting.get("enabled", True)):
        from scripts.rq3.market_jump_reporting import (
            build_companion_notebook,
            generate_figures,
            render_notebook_html,
        )

        generate_figures(
            output_root,
            max_case_events=int(reporting.get("max_case_events", 12)),
        )
        notebook_path = build_companion_notebook(output_root)
        report_path = render_notebook_html(notebook_path, output_root / "report.html")
        report_receipt = {
            "selected_delivery_mode": "self_contained_html",
            "renderer": "executed_jupyter_notebook_via_nbconvert",
            "notebook_path": str(notebook_path),
            "report_path": str(report_path),
            "report_size_bytes": int(report_path.stat().st_size),
            "notebook_executed_top_to_bottom": True,
            "canonical_node_artifact_builder_available": shutil.which("node") is not None,
            "canonical_node_artifact_builder_note": (
                "The requested notebook/nbconvert report is the delivered HTML. "
                "The Data Analytics portable Node builder is unavailable in this environment."
                if shutil.which("node") is None
                else "Node is available; the user-requested notebook/nbconvert report remains the selected surface."
            ),
        }
        (output_root / "report_delivery_receipt.json").write_text(
            json.dumps(report_receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return output_root


__all__ = [
    "ExtractionAudit",
    "add_rolling_jump_skewness",
    "aggregate_market_pairs",
    "build_pair_slice_metrics",
    "cluster_candidate_episodes",
    "extract_surface_points",
    "load_valid_pairs",
    "run_market_jump_detection",
    "score_metric_changes",
    "select_approximate_atm",
    "weighted_median",
]
