"""Offline raw-trade reliability bootstrap for news-first raw-vol labels.

The production surface generator intentionally collapses exact duplicate trade
rows before calibration.  This module replays that *canonical* behaviour and
keeps occurrence multiplicity as a separate diagnostic.  It is analysis-only:
bootstrap surfaces are summarized to CSV/NPZ artifacts and are never exposed as
training labels.

V1 starts from the materialized pre-calibration Black76 implied volatilities.
It does not invert option prices again.  The option occurrence, frozen
underlying lookup, and rate-curve context are nevertheless carried through the
lineage artifacts so that this limitation is explicit and testable.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wgan_option.merge_support import build_surface_from_params


SCHEMA_VERSION = 1
BOOTSTRAP_METHODS = ("support_preserving_bayesian", "strike_cluster")
PAIR_SCORE_COLUMNS = (
    "pair_id",
    "tolerance_minutes",
    "session_id",
    "effective_origin_utc",
    "c_i",
    "q_i",
    "m_i",
    "u_i",
    "u_estimable",
    "v_i",
    "h_i",
)
_RAW_COLUMNS = ("#RIC", "Date-Time", "Price", "Volume")
_PRICED_IDENTITY_COLUMNS = (
    "contract_id",
    "trade_datetime_utc",
    "price",
    "weight",
)
_PRICED_NUMERIC_COLUMNS = (
    "business_days",
    "strike",
    "price",
    "spot",
    "percent_strike",
    "implied_vol",
    "weight",
)
_PRICING_CONTEXT_COLUMNS = (
    "pricing_model",
    "rate_curve_date",
    "continuous_rate",
    "discount_factor",
    "rate_curve_sha256",
    "underlying_contract_id",
    "underlying_trade_datetime_utc",
    "underlying_staleness_seconds",
    "underlying_match_mode",
    "spot",
)


class LabelReliabilityError(ValueError):
    """Raised when a reliability run cannot preserve its frozen lineage."""


def parse_raw_surface_params(value: Any) -> dict[str, list[Any]]:
    """Parse the persisted raw-surface schema without importing model packages."""

    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        text = str(value).strip()
        payload = {}
        for loader in (json.loads, ast.literal_eval):
            try:
                candidate = loader(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(candidate, Mapping):
                payload = dict(candidate)
                break
    required = ("business_days", "percent_strikes", "implied_vols")
    if any(key not in payload or not isinstance(payload[key], list) for key in required):
        raise LabelReliabilityError("Invalid raw surface parameter schema.")
    if len({len(payload[key]) for key in required}) != 1:
        raise LabelReliabilityError("Raw surface outer-list lengths disagree.")
    cleaned: dict[int, list[tuple[float, float]]] = {}
    for raw_day, raw_strikes, raw_vols in zip(
        payload["business_days"],
        payload["percent_strikes"],
        payload["implied_vols"],
    ):
        if not isinstance(raw_strikes, list) or not isinstance(raw_vols, list):
            raise LabelReliabilityError("Raw surface slices must be lists.")
        if len(raw_strikes) != len(raw_vols):
            raise LabelReliabilityError("Raw strike/IV slice lengths disagree.")
        day = int(round(float(raw_day)))
        if day <= 0:
            continue
        points = cleaned.setdefault(day, [])
        for strike, vol in zip(raw_strikes, raw_vols):
            strike_value = float(strike)
            vol_value = float(vol)
            if (
                math.isfinite(strike_value)
                and math.isfinite(vol_value)
                and strike_value > 0
                and vol_value > 0
            ):
                points.append((strike_value, vol_value))
    business_days: list[int] = []
    percent_strikes: list[list[float]] = []
    implied_vols: list[list[float]] = []
    for day, points in sorted(cleaned.items()):
        grouped: dict[float, list[float]] = {}
        for strike, vol in points:
            grouped.setdefault(strike, []).append(vol)
        if not grouped:
            continue
        strikes = sorted(grouped)
        business_days.append(day)
        percent_strikes.append(strikes)
        implied_vols.append([float(np.mean(grouped[strike])) for strike in strikes])
    if len(business_days) < 2:
        raise LabelReliabilityError("Raw surface requires at least two maturity slices.")
    return {
        "business_days": business_days,
        "percent_strikes": percent_strikes,
        "implied_vols": implied_vols,
    }


def raw_support_mask(
    surface_params: Mapping[str, Any],
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> np.ndarray:
    """Return the production raw bracket-intersection support mask."""

    params = parse_raw_surface_params(surface_params)
    days = np.asarray(params["business_days"], dtype=np.float64)
    bounds = np.asarray(
        [(min(strikes), max(strikes)) for strikes in params["percent_strikes"]],
        dtype=np.float64,
    )
    strikes = np.asarray(strike_grid, dtype=np.float64)
    mask = np.zeros((len(maturity_days_grid), len(strikes)), dtype=bool)
    for maturity_index, raw_maturity in enumerate(maturity_days_grid):
        maturity = float(raw_maturity)
        if maturity < days[0] or maturity > days[-1]:
            continue
        exact = np.flatnonzero(np.isclose(days, maturity, atol=1e-8, rtol=0.0))
        if exact.size:
            lower, upper = bounds[int(exact[0])]
        else:
            upper_index = int(np.searchsorted(days, maturity, side="right"))
            lower_index = upper_index - 1
            lower = max(bounds[lower_index, 0], bounds[upper_index, 0])
            upper = min(bounds[lower_index, 1], bounds[upper_index, 1])
        if lower <= upper:
            mask[maturity_index] = (strikes >= lower - 1e-12) & (
                strikes <= upper + 1e-12
            )
    return mask


def reconstruct_raw_surface(
    surface_params: Mapping[str, Any],
    *,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> np.ndarray:
    """Evaluate one raw surface on the frozen grid."""

    params = parse_raw_surface_params(surface_params)
    surface = build_surface_from_params(
        surface_model="raw",
        surface_params=params,
        valuation_date=date(2000, 1, 3),
    )
    values = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid],
        business_days=[int(round(float(value))) for value in maturity_days_grid],
        forward=1.0,
    )
    result = np.asarray(values, dtype=np.float64)
    expected = (len(maturity_days_grid), len(strike_grid))
    if (
        result.shape != expected
        or not np.all(np.isfinite(result))
        or np.any(result <= 0)
    ):
        raise LabelReliabilityError(f"Invalid reconstructed raw surface: {result.shape}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return _utc_string(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise LabelReliabilityError(f"Invalid {label}: {value!r}")
    return pd.Timestamp(parsed)


def _utc_string(value: Any) -> str:
    timestamp = _utc_timestamp(value, label="UTC timestamp")
    return timestamp.isoformat().replace("+00:00", "Z")


def _canonical_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise LabelReliabilityError(f"Trade identity contains non-finite number: {value!r}")
    return f"{number:.12g}"


def _trade_identity(
    contract_id: Any,
    trade_datetime_utc: Any,
    price: Any,
    volume: Any,
) -> str:
    return "|".join(
        (
            str(contract_id).strip(),
            _utc_string(trade_datetime_utc),
            _canonical_number(price),
            _canonical_number(volume),
        )
    )


def _trade_fingerprint(
    contract_id: Any,
    trade_datetime_utc: Any,
    price: Any,
    volume: Any,
) -> str:
    return _sha256_bytes(
        _trade_identity(contract_id, trade_datetime_utc, price, volume).encode("utf-8")
    )


def _underlying_identity(contract_id: Any, timestamp: Any, spot: Any) -> str:
    return "|".join(
        (str(contract_id).strip(), _utc_string(timestamp), _canonical_number(spot))
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_table(spec: Any, *, label: str) -> tuple[pd.DataFrame, Path]:
    if isinstance(spec, Mapping):
        if "path" not in spec:
            raise LabelReliabilityError(f"{label} table spec requires `path`.")
        path = Path(str(spec["path"])).expanduser().resolve()
        sheet_name = str(spec.get("sheet_name", "gan_input_ready"))
    else:
        path = Path(str(spec)).expanduser().resolve()
        sheet_name = "gan_input_ready"
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    lower_name = path.name.lower()
    if lower_name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, low_memory=False), path
    if lower_name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl"), path
    raise LabelReliabilityError(
        f"{label} must be CSV/CSV.GZ/XLSX, received: {path}"
    )


def _parse_numeric_sequence(value: Any, *, label: str) -> list[float]:
    if isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        raw = None
        for loader in (json.loads, ast.literal_eval):
            try:
                candidate = loader(text)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(candidate, (list, tuple)):
                raw = list(candidate)
                break
        if raw is None:
            raise LabelReliabilityError(f"Invalid {label}: {value!r}")
    result = [float(item) for item in raw]
    if not result or not all(math.isfinite(item) for item in result):
        raise LabelReliabilityError(f"{label} must contain finite values.")
    return result


def _stable_unique(group: pd.DataFrame, column: str, *, label: str) -> Any:
    if column not in group:
        raise LabelReliabilityError(f"{label} is missing required column {column!r}.")
    values = group[column].dropna()
    normalized = [str(value).strip() for value in values if str(value).strip()]
    unique = list(dict.fromkeys(normalized))
    if len(unique) != 1:
        raise LabelReliabilityError(
            f"{label} requires one non-empty {column}; observed {len(unique)} values."
        )
    return unique[0]


def _load_pair_manifest_spec(spec: Any, *, tolerance: int) -> tuple[pd.DataFrame, list[Path]]:
    frame, primary_path = _read_table(spec, label=f"{tolerance}m pair manifest")
    source_paths = [primary_path]
    if isinstance(spec, Mapping) and "surface_rows" in spec:
        surface_rows, surface_path = _read_table(
            spec["surface_rows"], label=f"{tolerance}m pair surface rows"
        )
        source_paths.append(surface_path)
        if "pair_id" not in surface_rows:
            raise LabelReliabilityError("Pair surface rows require pair_id.")
        keep = [
            column
            for column in (
                "pair_id",
                "current_surface_param_json",
                "target_surface_param_json",
                "strike_grid",
                "maturity_days_grid",
            )
            if column in surface_rows
        ]
        surface_rows = surface_rows[keep].copy()
        for pair_id, group in surface_rows.groupby("pair_id", sort=False):
            for column in keep[1:]:
                _stable_unique(
                    group,
                    column,
                    label=f"surface rows pair_id={pair_id}",
                )
        surface_rows = surface_rows.drop_duplicates("pair_id", keep="first")
        overlap = (set(frame.columns) & set(surface_rows.columns)) - {"pair_id"}
        if overlap:
            surface_rows = surface_rows.drop(columns=sorted(overlap))
        frame = frame.merge(surface_rows, on="pair_id", how="left", validate="many_to_one")
    return frame, source_paths


def _load_pair_manifests(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[Path], list[float], list[float], dict[str, str]]:
    specs = config.get("pair_manifests")
    if not isinstance(specs, Mapping):
        raise LabelReliabilityError("label_reliability.pair_manifests must be a mapping.")
    normalized_specs = {int(key): value for key, value in specs.items()}
    if set(normalized_specs) != {5, 30}:
        raise LabelReliabilityError("pair_manifests must contain exactly 5 and 30 minute inputs.")
    if "pre_q3_end_utc" not in config:
        raise LabelReliabilityError("pre_q3_end_utc is required to prevent Q3 leakage.")
    if "interval_start_utc" in config or "interval_end_utc" in config:
        raise LabelReliabilityError(
            "Bounded/Q3 label audits are forbidden in this pre-selection core."
        )
    cutoff = _utc_timestamp(config["pre_q3_end_utc"], label="pre_q3_end_utc")
    scope = {
        "mode": "pre_q3",
        "interval_start_utc": "",
        "interval_end_utc": _utc_string(cutoff),
    }
    forecast_horizon_minutes = int(config.get("forecast_horizon_minutes", 5))
    if forecast_horizon_minutes <= 0:
        raise LabelReliabilityError("forecast_horizon_minutes must be positive.")

    result_rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    inferred_strikes: list[list[float]] = []
    inferred_maturities: list[list[float]] = []
    for tolerance, spec in sorted(normalized_specs.items()):
        frame, paths = _load_pair_manifest_spec(spec, tolerance=tolerance)
        source_paths.extend(paths)
        required = {
            "pair_id",
            "session_id",
            "effective_origin_utc",
            "current_surface_param_json",
            "target_surface_param_json",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise LabelReliabilityError(
                f"{tolerance}m pair manifest is missing columns: {missing}"
            )
        target_column = next(
            (
                column
                for column in ("target_anchor_utc", "target_snapshot_time_utc")
                if column in frame
            ),
            None,
        )
        if target_column is None:
            raise LabelReliabilityError(
                f"{tolerance}m pair manifest requires target anchor/snapshot time."
            )
        work = frame.copy()
        work["_origin"] = pd.to_datetime(
            work["effective_origin_utc"], errors="coerce", utc=True
        )
        work["_target"] = pd.to_datetime(work[target_column], errors="coerce", utc=True)
        if work[["_origin", "_target"]].isna().any().any():
            raise LabelReliabilityError(f"{tolerance}m pair manifest has invalid timestamps.")
        in_scope = work["_origin"] < cutoff
        work = work.loc[in_scope].copy()
        if work.empty:
            raise LabelReliabilityError(
                f"{tolerance}m pair manifest has no pairs in scope {scope}."
            )
        duration = (work["_target"] - work["_origin"]).dt.total_seconds() / 60.0
        if not np.allclose(
            duration.to_numpy(dtype=float), float(forecast_horizon_minutes), atol=1e-9
        ):
            bad = sorted(
                set(
                    float(value)
                    for value in duration[
                        ~np.isclose(duration, forecast_horizon_minutes)
                    ]
                )
            )
            raise LabelReliabilityError(
                f"{tolerance}m alignment manifest changes the frozen "
                f"{forecast_horizon_minutes}m forecast horizon; observed {bad[:5]}."
            )
        work["pair_id"] = work["pair_id"].fillna("").astype(str).str.strip()
        work["session_id"] = work["session_id"].fillna("").astype(str).str.strip()
        if work["pair_id"].eq("").any() or work["session_id"].eq("").any():
            raise LabelReliabilityError(f"{tolerance}m manifest has empty pair/session IDs.")
        if "tolerance_minutes" in work:
            persisted = pd.to_numeric(work["tolerance_minutes"], errors="coerce")
            if persisted.isna().any() or not persisted.eq(tolerance).all():
                raise LabelReliabilityError(
                    f"{tolerance}m manifest contains mismatched tolerance_minutes."
                )

        for pair_id, group in work.groupby("pair_id", sort=False):
            current_params = _stable_unique(
                group,
                "current_surface_param_json",
                label=f"{tolerance}m pair_id={pair_id}",
            )
            parse_raw_surface_params(current_params)
            target_params = _stable_unique(
                group,
                "target_surface_param_json",
                label=f"{tolerance}m pair_id={pair_id}",
            )
            parse_raw_surface_params(target_params)
            row = {
                "tolerance_minutes": int(tolerance),
                "pair_id": str(pair_id),
                "session_id": _stable_unique(
                    group, "session_id", label=f"{tolerance}m pair_id={pair_id}"
                ),
                "effective_origin_utc": _utc_string(group["_origin"].iloc[0]),
                "target_anchor_utc": _utc_string(group["_target"].iloc[0]),
                "target_window_start_utc": _utc_string(
                    group["_target"].iloc[0]
                    - pd.Timedelta(minutes=forecast_horizon_minutes)
                ),
                "current_surface_param_json": current_params,
                "target_surface_param_json": target_params,
            }
            if "joint_strict_support_cell_count" in group:
                formal = pd.to_numeric(
                    group["joint_strict_support_cell_count"], errors="coerce"
                ).dropna()
                if formal.nunique() > 1:
                    raise LabelReliabilityError(
                        f"{tolerance}m pair_id={pair_id} has inconsistent formal support."
                    )
                row["formal_joint_support_cell_count"] = (
                    int(formal.iloc[0]) if not formal.empty else np.nan
                )
            else:
                row["formal_joint_support_cell_count"] = np.nan
            result_rows.append(row)
        if "strike_grid" in work:
            inferred_strikes.extend(
                _parse_numeric_sequence(value, label="manifest strike_grid")
                for value in work["strike_grid"].dropna().unique()
            )
        if "maturity_days_grid" in work:
            inferred_maturities.extend(
                _parse_numeric_sequence(value, label="manifest maturity_days_grid")
                for value in work["maturity_days_grid"].dropna().unique()
            )

    pairs = pd.DataFrame(result_rows).sort_values(
        ["tolerance_minutes", "effective_origin_utc", "pair_id"]
    ).reset_index(drop=True)
    if pairs.duplicated(["tolerance_minutes", "pair_id"]).any():
        raise LabelReliabilityError("Pair manifests do not have unique tolerance/pair keys.")

    grid = config.get("grid", {})
    if grid and not isinstance(grid, Mapping):
        raise LabelReliabilityError("grid must be a mapping.")
    if isinstance(grid, Mapping) and "strike_grid" in grid:
        strikes = _parse_numeric_sequence(grid["strike_grid"], label="grid.strike_grid")
    elif inferred_strikes:
        strikes = inferred_strikes[0]
    else:
        raise LabelReliabilityError("A strike grid is required in config or pair manifests.")
    if isinstance(grid, Mapping) and "maturity_days_grid" in grid:
        maturities = _parse_numeric_sequence(
            grid["maturity_days_grid"], label="grid.maturity_days_grid"
        )
    elif inferred_maturities:
        maturities = inferred_maturities[0]
    else:
        raise LabelReliabilityError("A maturity grid is required in config or pair manifests.")
    for candidate in inferred_strikes:
        if not np.array_equal(np.asarray(candidate), np.asarray(strikes)):
            raise LabelReliabilityError("Pair manifests disagree on the frozen strike grid.")
    for candidate in inferred_maturities:
        if not np.array_equal(np.asarray(candidate), np.asarray(maturities)):
            raise LabelReliabilityError("Pair manifests disagree on the frozen maturity grid.")
    expected_shape = tuple(int(value) for value in config.get("expected_grid_shape", (16, 16)))
    if expected_shape != (len(maturities), len(strikes)):
        raise LabelReliabilityError(
            f"Frozen grid shape is {(len(maturities), len(strikes))}, expected {expected_shape}."
        )
    scope["forecast_horizon_minutes"] = str(forecast_horizon_minutes)
    return pairs, source_paths, strikes, maturities, scope


def _load_priced_rows(
    config: Mapping[str, Any], pairs: pd.DataFrame
) -> tuple[pd.DataFrame, Path, dict[str, str]]:
    if "priced_target_rows" not in config:
        raise LabelReliabilityError("priced_target_rows is required.")
    frame, source_path = _read_table(
        config["priced_target_rows"], label="priced target rows"
    )
    missing = sorted(set(_PRICED_IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise LabelReliabilityError(f"Priced target rows are missing columns: {missing}")
    for column in _PRICED_NUMERIC_COLUMNS:
        if column not in frame:
            raise LabelReliabilityError(f"Priced target rows require {column!r}.")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "is_otm" not in frame:
        raise LabelReliabilityError("Priced target rows require is_otm.")
    anchor_column = next(
        (
            column
            for column in ("calibration_datetime_utc", "target_anchor_utc")
            if column in frame
        ),
        None,
    )
    if anchor_column is None:
        raise LabelReliabilityError(
            "Priced target rows require calibration_datetime_utc/target_anchor_utc."
        )
    work = frame.copy()
    if "window_side" not in work:
        raise LabelReliabilityError(
            "Priced target cache requires window_side='backward'; target labels are the "
            "backward window ending at target_anchor_utc."
        )
    side = work["window_side"].fillna("").astype(str).str.strip().str.lower()
    work = work.loc[side.eq("backward")].copy()
    anchors = pd.to_datetime(work[anchor_column], errors="coerce", utc=True)
    work["target_anchor_utc"] = anchors.map(
        lambda value: _utc_string(value) if not pd.isna(value) else ""
    )
    wanted_anchors = set(pairs["target_anchor_utc"])
    work = work.loc[work["target_anchor_utc"].isin(wanted_anchors)].copy()
    if work.empty:
        raise LabelReliabilityError("No priced target rows match the pre-Q3 pair anchors.")
    work["contract_id"] = work["contract_id"].fillna("").astype(str).str.strip()
    work["trade_datetime_utc"] = work["trade_datetime_utc"].map(
        lambda value: _utc_string(value) if not pd.isna(value) else ""
    )
    identity_valid = (
        work["contract_id"].ne("")
        & work["trade_datetime_utc"].ne("")
        & work["price"].notna()
        & work["weight"].notna()
    )
    work["row_fingerprint"] = ""
    work.loc[identity_valid, "row_fingerprint"] = [
        _trade_fingerprint(contract_id, timestamp, price, weight)
        for contract_id, timestamp, price, weight in work.loc[
            identity_valid, list(_PRICED_IDENTITY_COLUMNS)
        ].itertuples(index=False, name=None)
    ]
    work["is_otm"] = work["is_otm"].map(_truthy)
    max_iv = float(config.get("max_precalib_iv", 3.0))
    max_itm_distance = float(config.get("max_itm_moneyness_distance", 0.05))
    if not math.isfinite(max_itm_distance) or max_itm_distance < 0:
        raise LabelReliabilityError(
            "max_itm_moneyness_distance must be finite and non-negative."
        )
    finite = np.ones(len(work), dtype=bool)
    for column in _PRICED_NUMERIC_COLUMNS:
        finite &= np.isfinite(work[column].to_numpy(dtype=float))
    work["base_eligible"] = (
        identity_valid.to_numpy()
        & finite
        & work["business_days"].gt(0).to_numpy()
        & work["strike"].gt(0).to_numpy()
        & work["price"].gt(0).to_numpy()
        & work["spot"].gt(0).to_numpy()
        & work["percent_strike"].gt(0).to_numpy()
        & work["implied_vol"].gt(0).to_numpy()
        & work["implied_vol"].le(max_iv).to_numpy()
        & work["weight"].gt(0).to_numpy()
        & (
            work["is_otm"].to_numpy()
            | work["percent_strike"].sub(1.0).abs().le(max_itm_distance).to_numpy()
        )
    )
    work["base_exclusion_reason"] = np.where(
        work["base_eligible"], "", "invalid_or_capped_cached_iv_candidate"
    )

    context_defaults = {
        "pricing_model": "",
        "rate_curve_date": "",
        "continuous_rate": np.nan,
        "discount_factor": np.nan,
        "rate_curve_sha256": "",
        "underlying_contract_id": "",
        "underlying_trade_datetime_utc": "",
        "underlying_staleness_seconds": np.nan,
        "underlying_match_mode": "",
    }
    for column, default in context_defaults.items():
        if column not in work:
            work[column] = default
    work["underlying_contract_id"] = (
        work["underlying_contract_id"].fillna("").astype(str).str.strip()
    )
    underlying_times = pd.to_datetime(
        work["underlying_trade_datetime_utc"], errors="coerce", utc=True
    )
    work["underlying_trade_datetime_utc"] = underlying_times.map(
        lambda value: _utc_string(value) if not pd.isna(value) else ""
    )
    has_underlying = (
        work["underlying_contract_id"].ne("")
        & work["underlying_trade_datetime_utc"].ne("")
        & work["spot"].notna()
        & work["spot"].gt(0)
    )
    work["underlying_lookup_key"] = ""
    work.loc[has_underlying, "underlying_lookup_key"] = [
        _underlying_identity(contract_id, timestamp, spot)
        for contract_id, timestamp, spot in work.loc[
            has_underlying,
            ["underlying_contract_id", "underlying_trade_datetime_utc", "spot"],
        ].itertuples(index=False, name=None)
    ]

    dedup_keys = ["target_anchor_utc", "row_fingerprint"]
    eligible_with_id = work.loc[work["base_eligible"]].copy()
    if eligible_with_id.empty:
        raise LabelReliabilityError("No valid cached-IV target candidates remain.")
    consistency_columns = [
        "business_days",
        "strike",
        "spot",
        "percent_strike",
        "implied_vol",
        "is_otm",
        *_PRICING_CONTEXT_COLUMNS,
    ]
    for key, group in eligible_with_id.groupby(dedup_keys, sort=False):
        inconsistent = [
            column
            for column in consistency_columns
            if column in group and group[column].astype(str).nunique(dropna=False) > 1
        ]
        if inconsistent:
            raise LabelReliabilityError(
                f"Canonical duplicate {key} has inconsistent cached pricing fields: {inconsistent}"
            )
    work = work.sort_values(dedup_keys + ["trade_datetime_utc", "contract_id"])
    work["priced_audit_duplicate_count"] = work.groupby(dedup_keys)[
        "row_fingerprint"
    ].transform("size")
    work["canonical_priced_row"] = ~work.duplicated(dedup_keys, keep="first")
    audit_columns = {
        column: str(work[column].dtype)
        for column in _PRICING_CONTEXT_COLUMNS
        if column in work
    }
    return work.reset_index(drop=True), source_path, audit_columns


def _resolve_raw_files(value: Any) -> list[Path]:
    raw_values = list(value) if isinstance(value, (list, tuple)) else [value]
    matches: list[Path] = []
    for raw_value in raw_values:
        text = str(raw_value)
        expanded = sorted(glob.glob(text, recursive=True))
        if not expanded and Path(text).is_file():
            expanded = [text]
        matches.extend(Path(path).expanduser().resolve() for path in expanded)
    unique = sorted({path for path in matches if path.is_file()})
    if not unique:
        raise FileNotFoundError("raw_files did not resolve to any files.")
    return unique


def _scan_raw_occurrences(
    raw_files: Sequence[Path],
    *,
    priced_rows: pd.DataFrame,
    pairs: pd.DataFrame,
    chunk_size: int,
    max_underlying_staleness_seconds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    option_fingerprints = set(
        priced_rows.loc[
            priced_rows["row_fingerprint"].ne(""), "row_fingerprint"
        ].astype(str)
    )
    underlying_keys = set(
        priced_rows.loc[
            priced_rows["underlying_lookup_key"].ne(""), "underlying_lookup_key"
        ].astype(str)
    )
    relevant_contracts = set(priced_rows["contract_id"].dropna().astype(str)) | set(
        priced_rows["underlying_contract_id"].dropna().astype(str)
    )
    relevant_contracts.discard("")
    earliest = pd.to_datetime(pairs["target_window_start_utc"], utc=True).min() - pd.Timedelta(
        seconds=max(0, int(max_underlying_staleness_seconds))
    )
    latest = pd.to_datetime(pairs["target_anchor_utc"], utc=True).max()
    occurrence_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for path in raw_files:
        file_sha = _sha256_file(path)
        rows_read = 0
        rows_valid = 0
        rows_matched = 0
        try:
            iterator = pd.read_csv(
                path,
                usecols=list(_RAW_COLUMNS),
                chunksize=max(1, int(chunk_size)),
                low_memory=False,
            )
            for chunk in iterator:
                start_ordinal = rows_read
                rows_read += len(chunk)
                chunk = chunk.copy()
                chunk["source_row_ordinal"] = np.arange(
                    start_ordinal, start_ordinal + len(chunk), dtype=np.int64
                )
                chunk["contract_id"] = chunk["#RIC"].fillna("").astype(str).str.strip()
                chunk["trade_datetime_utc"] = pd.to_datetime(
                    chunk["Date-Time"], errors="coerce", utc=True
                )
                chunk["price"] = pd.to_numeric(chunk["Price"], errors="coerce")
                chunk["volume"] = pd.to_numeric(chunk["Volume"], errors="coerce")
                valid = (
                    chunk["contract_id"].ne("")
                    & chunk["trade_datetime_utc"].notna()
                    & chunk["price"].gt(0)
                    & chunk["volume"].gt(0)
                )
                rows_valid += int(valid.sum())
                candidate = chunk.loc[
                    valid
                    & chunk["contract_id"].isin(relevant_contracts)
                    & chunk["trade_datetime_utc"].between(
                        earliest, latest, inclusive="left"
                    )
                ].copy()
                if candidate.empty:
                    continue
                candidate["trade_datetime_utc"] = candidate["trade_datetime_utc"].map(
                    _utc_string
                )
                candidate["row_fingerprint"] = [
                    _trade_fingerprint(contract_id, timestamp, price, volume)
                    for contract_id, timestamp, price, volume in candidate[
                        ["contract_id", "trade_datetime_utc", "price", "volume"]
                    ].itertuples(index=False, name=None)
                ]
                candidate["underlying_lookup_key"] = [
                    _underlying_identity(contract_id, timestamp, price)
                    for contract_id, timestamp, price in candidate[
                        ["contract_id", "trade_datetime_utc", "price"]
                    ].itertuples(index=False, name=None)
                ]
                is_option = candidate["row_fingerprint"].isin(option_fingerprints)
                is_underlying = candidate["underlying_lookup_key"].isin(underlying_keys)
                candidate = candidate.loc[is_option | is_underlying].copy()
                if candidate.empty:
                    continue
                rows_matched += len(candidate)
                for row in candidate.itertuples(index=False):
                    role_option = str(row.row_fingerprint) in option_fingerprints
                    role_underlying = str(row.underlying_lookup_key) in underlying_keys
                    occurrence_id = _sha256_bytes(
                        "|".join(
                            (
                                file_sha,
                                str(int(row.source_row_ordinal)),
                                str(row.row_fingerprint),
                            )
                        ).encode("utf-8")
                    )
                    occurrence_rows.append(
                        {
                            "raw_occurrence_id": occurrence_id,
                            "source_path": str(path),
                            "source_file_sha256": file_sha,
                            "source_row_ordinal": int(row.source_row_ordinal),
                            "source_line_number": int(row.source_row_ordinal) + 2,
                            "contract_id": str(row.contract_id),
                            "trade_datetime_utc": str(row.trade_datetime_utc),
                            "price": float(row.price),
                            "volume": float(row.volume),
                            "row_fingerprint": str(row.row_fingerprint),
                            "underlying_lookup_key": str(row.underlying_lookup_key),
                            "is_option_occurrence": bool(role_option),
                            "is_underlying_occurrence": bool(role_underlying),
                        }
                    )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise LabelReliabilityError(f"Unable to scan raw source {path}: {exc}") from exc
        source_rows.append(
            {
                "source_path": str(path),
                "source_file_sha256": file_sha,
                "size_bytes": path.stat().st_size,
                "rows_read": int(rows_read),
                "valid_positive_rows": int(rows_valid),
                "matched_lineage_occurrences": int(rows_matched),
            }
        )
    lineage = pd.DataFrame(occurrence_rows)
    if lineage.empty:
        raise LabelReliabilityError("No priced target rows could be linked to raw files.")
    if lineage["raw_occurrence_id"].duplicated().any():
        raise LabelReliabilityError("Raw occurrence IDs are not unique.")
    lineage["duplicate_count"] = lineage.groupby("row_fingerprint")[
        "raw_occurrence_id"
    ].transform("size")
    lineage["duplicate_ordinal"] = lineage.sort_values(
        ["source_path", "source_row_ordinal"]
    ).groupby("row_fingerprint").cumcount() + 1
    lineage = lineage.sort_values(
        ["trade_datetime_utc", "contract_id", "source_path", "source_row_ordinal"]
    ).reset_index(drop=True)
    return lineage, pd.DataFrame(source_rows)


def _attach_lineage(
    priced_rows: pd.DataFrame,
    pairs: pd.DataFrame,
    lineage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = priced_rows.loc[priced_rows["canonical_priced_row"]].copy()
    replay = pairs.merge(canonical, on="target_anchor_utc", how="left", validate="many_to_many")
    if replay["row_fingerprint"].isna().all():
        raise LabelReliabilityError("No canonical priced rows joined to pair manifests.")
    option_lineage = lineage.loc[lineage["is_option_occurrence"]].copy()
    occurrence_groups = option_lineage.groupby("row_fingerprint", sort=False)
    occurrence_count = occurrence_groups["raw_occurrence_id"].size()
    occurrence_ids = occurrence_groups["raw_occurrence_id"].apply(
        lambda values: json.dumps(sorted(str(value) for value in values), separators=(",", ":"))
    )
    replay["raw_option_occurrence_count"] = (
        replay["row_fingerprint"].map(occurrence_count).fillna(0).astype(int)
    )
    replay["raw_occurrence_ids_json"] = replay["row_fingerprint"].map(occurrence_ids).fillna("[]")
    underlying_lineage = lineage.loc[lineage["is_underlying_occurrence"]]
    underlying_count = underlying_lineage.groupby("underlying_lookup_key")[
        "raw_occurrence_id"
    ].size()
    replay["underlying_raw_occurrence_count"] = (
        replay["underlying_lookup_key"].map(underlying_count).fillna(0).astype(int)
    )
    replay["underlying_raw_locator_missing"] = (
        replay["underlying_lookup_key"].eq("")
        | replay["underlying_raw_occurrence_count"].eq(0)
    )
    missing_selected = replay["base_eligible"] & replay["raw_option_occurrence_count"].eq(0)
    if missing_selected.any():
        example = replay.loc[
            missing_selected,
            ["pair_id", "target_anchor_utc", "contract_id", "trade_datetime_utc"],
        ].head(5)
        raise LabelReliabilityError(
            "Selected cached-IV rows lack raw option locators: "
            + example.to_dict(orient="records").__repr__()
        )

    membership = replay.loc[
        replay["row_fingerprint"].ne(""),
        [
            "tolerance_minutes",
            "pair_id",
            "session_id",
            "effective_origin_utc",
            "target_anchor_utc",
            "target_window_start_utc",
            "row_fingerprint",
            "base_eligible",
        ],
    ].merge(
        option_lineage[
            [
                "raw_occurrence_id",
                "source_file_sha256",
                "source_row_ordinal",
                "row_fingerprint",
                "trade_datetime_utc",
                "duplicate_count",
                "duplicate_ordinal",
            ]
        ],
        on="row_fingerprint",
        how="left",
        validate="many_to_many",
    )
    times = pd.to_datetime(membership["trade_datetime_utc"], errors="coerce", utc=True)
    starts = pd.to_datetime(
        membership["target_window_start_utc"], errors="raise", utc=True
    )
    ends = pd.to_datetime(membership["target_anchor_utc"], errors="raise", utc=True)
    membership["inside_target_half_open_window"] = (times >= starts) & (times < ends)
    selected_outside = membership["base_eligible"] & ~membership[
        "inside_target_half_open_window"
    ]
    if selected_outside.any():
        raise LabelReliabilityError(
            "Selected raw occurrence violates the target [origin,anchor) window."
        )
    replay = replay.sort_values(
        ["tolerance_minutes", "pair_id", "trade_datetime_utc", "contract_id"]
    ).reset_index(drop=True)
    membership = membership.sort_values(
        ["tolerance_minutes", "pair_id", "trade_datetime_utc", "raw_occurrence_id"]
    ).reset_index(drop=True)
    return replay, membership


def _weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
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
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return float(np.median([value for value, _ in pairs]))
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def _apply_otm_preferred_itm_fallback(rows: pd.DataFrame) -> pd.DataFrame:
    selected = rows.loc[rows["bootstrap_weight"].gt(0)].copy()
    selected["surface_input_role_replay"] = ""
    if selected.empty:
        return selected
    has_otm = selected.groupby(["business_days", "strike"])["is_otm"].transform("any")
    selected = selected.loc[~(has_otm & ~selected["is_otm"])].copy()
    keep_indices: list[int] = []
    for _, maturity in selected.groupby("business_days", sort=False):
        otm = maturity.loc[maturity["is_otm"]]
        itm = maturity.loc[~maturity["is_otm"]]
        keep_indices.extend(int(index) for index in otm.index)
        otm_strikes = set(otm["strike"].astype(float))
        if itm.empty or not otm_strikes:
            continue
        ranked = sorted(
            set(itm["strike"].astype(float)),
            key=lambda strike: (
                float(
                    (itm.loc[itm["strike"].eq(strike), "percent_strike"] - 1.0)
                    .abs()
                    .min()
                ),
                strike,
            ),
        )
        allowed = set(ranked[: len(otm_strikes)])
        keep_indices.extend(int(index) for index in itm.loc[itm["strike"].isin(allowed)].index)
    result = selected.loc[sorted(set(keep_indices))].copy()
    result["surface_input_role_replay"] = np.where(result["is_otm"], "otm", "itm_fallback")
    return result


def _build_raw_surface(
    rows: pd.DataFrame,
    *,
    min_strikes_per_expiry: int,
    min_expiries_per_surface: int,
) -> tuple[dict[str, list[Any]] | None, pd.DataFrame, str]:
    selected = _apply_otm_preferred_itm_fallback(rows)
    if selected.empty:
        return None, selected, "no_selected_rows"
    points: list[dict[str, Any]] = []
    for (business_days, strike), bucket in selected.groupby(
        ["business_days", "strike"], sort=True
    ):
        weight_sum = float(bucket["bootstrap_weight"].sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            continue
        implied_vol = _weighted_median(
            bucket["implied_vol"], bucket["bootstrap_weight"]
        )
        percent_strike = float(
            np.average(bucket["percent_strike"], weights=bucket["bootstrap_weight"])
        )
        average_price = float(np.average(bucket["price"], weights=bucket["bootstrap_weight"]))
        average_spot = float(np.average(bucket["spot"], weights=bucket["bootstrap_weight"]))
        if not all(
            math.isfinite(value) and value > 0
            for value in (implied_vol, percent_strike, average_price, average_spot)
        ):
            continue
        points.append(
            {
                "business_days": int(round(float(business_days))),
                "strike": float(strike),
                "percent_strike": percent_strike,
                "implied_vol": implied_vol,
            }
        )
    point_frame = pd.DataFrame(points)
    if point_frame.empty:
        return None, selected.iloc[0:0].copy(), "no_aggregate_buckets"
    counts = point_frame.groupby("business_days")["strike"].transform("size")
    point_frame = point_frame.loc[counts.ge(int(min_strikes_per_expiry))].copy()
    if point_frame["business_days"].nunique() < int(min_expiries_per_surface):
        return None, selected.iloc[0:0].copy(), "insufficient_maturities"
    used_keys = set(
        point_frame[["business_days", "strike"]].itertuples(index=False, name=None)
    )
    selected["surface_bucket_used"] = [
        (int(round(float(day))), float(strike)) in used_keys
        for day, strike in selected[["business_days", "strike"]].itertuples(
            index=False, name=None
        )
    ]
    used_rows = selected.loc[selected["surface_bucket_used"]].copy()
    business_days: list[int] = []
    percent_strikes: list[list[float]] = []
    implied_vols: list[list[float]] = []
    for day, maturity in point_frame.groupby("business_days", sort=True):
        maturity = maturity.sort_values(["percent_strike", "strike"])
        business_days.append(int(day))
        percent_strikes.append(maturity["percent_strike"].astype(float).tolist())
        implied_vols.append(maturity["implied_vol"].astype(float).tolist())
    params = {
        "business_days": business_days,
        "percent_strikes": percent_strikes,
        "implied_vols": implied_vols,
    }
    try:
        parse_raw_surface_params(params)
    except ValueError as exc:
        return None, used_rows, f"invalid_raw_surface:{type(exc).__name__}"
    return params, used_rows, "ok"


def _uniform_from_key(seed: int, method: str, draw_id: int, unit_id: str) -> float:
    payload = f"{int(seed)}|{method}|{int(draw_id)}|{unit_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return (integer + 0.5) / float(2**64)


def _exp1_multiplier(seed: int, draw_id: int, unit_id: str) -> float:
    uniform = _uniform_from_key(
        seed, "support_preserving_bayesian", draw_id, unit_id
    )
    return -math.log1p(-uniform)


def _poisson1_multiplier(seed: int, draw_id: int, unit_id: str) -> int:
    uniform = _uniform_from_key(
        seed, "strike_cluster", draw_id, unit_id
    )
    probability = math.exp(-1.0)
    cumulative = probability
    outcome = 0
    while uniform > cumulative:
        outcome += 1
        probability /= float(outcome)
        cumulative += probability
        if outcome >= 32:
            break
    return outcome


def _cluster_unit_id(row: Any) -> str:
    maturity = str(getattr(row, "maturity_date", "")).strip()
    if not maturity:
        maturity = str(int(round(float(row.business_days))))
    trade_date = _utc_string(row.trade_datetime_utc)[:10]
    return f"{trade_date}|{maturity}|{_canonical_number(row.strike)}"


def _bootstrap_weights(
    rows: pd.DataFrame,
    *,
    method: str,
    seed: int,
    draw_id: int,
) -> np.ndarray:
    if method == "support_preserving_bayesian":
        multipliers = np.asarray(
            [
                _exp1_multiplier(seed, draw_id, str(fingerprint))
                for fingerprint in rows["row_fingerprint"]
            ],
            dtype=np.float64,
        )
    elif method == "strike_cluster":
        multipliers = np.asarray(
            [
                _poisson1_multiplier(seed, draw_id, _cluster_unit_id(row))
                for row in rows.itertuples(index=False)
            ],
            dtype=np.float64,
        )
    else:
        raise LabelReliabilityError(f"Unsupported bootstrap method: {method}")
    return rows["weight"].to_numpy(dtype=np.float64) * multipliers


def _surface_values(
    params: Mapping[str, Any],
    *,
    strike_grid: Sequence[float],
    maturity_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    values = reconstruct_raw_surface(
        params,
        strike_grid=strike_grid,
        maturity_days_grid=maturity_grid,
    ).astype(np.float64)
    support = raw_support_mask(
        params,
        strike_grid=strike_grid,
        maturity_days_grid=maturity_grid,
    )
    return values, support


def _canonical_bucket_counts(used_rows: pd.DataFrame) -> tuple[int, int]:
    if used_rows.empty:
        return 0, 0
    buckets = used_rows.groupby(["business_days", "strike"], sort=False)[
        "raw_option_occurrence_count"
    ].sum()
    return int(len(buckets)), int(buckets.ge(2).sum())


def _run_bootstraps(
    pairs: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    strike_grid: Sequence[float],
    maturity_grid: Sequence[float],
    draws: int,
    seed: int,
    methods: Sequence[str],
    min_strikes_per_expiry: int,
    min_expiries_per_surface: int,
    minimum_valid_draw_fraction: float,
    canonical_replay_tolerance: float,
    uncertainty_numeric_tolerance: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    draw_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    array_keys: list[str] = []
    array_values: dict[str, list[np.ndarray]] = {
        "canonical_target": [],
        "bootstrap_mean": [],
        "bootstrap_std": [],
        "bootstrap_p05": [],
        "bootstrap_p50": [],
        "bootstrap_p95": [],
        "bootstrap_mad": [],
        "valid_draw_count": [],
        "current_support": [],
        "canonical_target_support": [],
        "original_joint_support": [],
    }
    for pair in pairs.itertuples(index=False):
        pair_candidates = replay.loc[
            replay["tolerance_minutes"].eq(int(pair.tolerance_minutes))
            & replay["pair_id"].eq(str(pair.pair_id))
            & replay["base_eligible"]
        ].copy()
        if pair_candidates.empty:
            raise LabelReliabilityError(f"Pair {pair.pair_id} has no eligible target rows.")
        pair_candidates["bootstrap_weight"] = pair_candidates["weight"].astype(float)
        canonical_params, canonical_used, canonical_status = _build_raw_surface(
            pair_candidates,
            min_strikes_per_expiry=min_strikes_per_expiry,
            min_expiries_per_surface=min_expiries_per_surface,
        )
        if canonical_params is None:
            raise LabelReliabilityError(
                f"Canonical target replay failed for pair {pair.pair_id}: {canonical_status}"
            )
        current_params = parse_raw_surface_params(pair.current_surface_param_json)
        canonical_values, canonical_support = _surface_values(
            canonical_params,
            strike_grid=strike_grid,
            maturity_grid=maturity_grid,
        )
        current_values, current_support = _surface_values(
            current_params,
            strike_grid=strike_grid,
            maturity_grid=maturity_grid,
        )
        original_joint = current_support & canonical_support
        c_i = int(original_joint.sum())
        q_i, m_i = _canonical_bucket_counts(canonical_used)
        canonical_raw_occurrences = int(
            canonical_used["raw_option_occurrence_count"].sum()
        )
        canonical_unique_raw_rows = int(len(canonical_used))
        canonical_weights = canonical_used["weight"].to_numpy(dtype=np.float64)
        canonical_volume_ess = float(
            np.square(canonical_weights.sum()) / np.square(canonical_weights).sum()
        )
        singleton_bucket_count = 0
        for (business_days, strike), bucket in canonical_used.groupby(
            ["business_days", "strike"], sort=True
        ):
            occurrence_count = int(bucket["raw_option_occurrence_count"].sum())
            singleton = occurrence_count == 1
            singleton_bucket_count += int(singleton)
            weights = bucket["weight"].to_numpy(dtype=np.float64)
            bucket_ess = float(np.square(weights.sum()) / np.square(weights).sum())
            bucket_rows.append(
                {
                    "tolerance_minutes": int(pair.tolerance_minutes),
                    "pair_id": str(pair.pair_id),
                    "session_id": str(pair.session_id),
                    "effective_origin_utc": str(pair.effective_origin_utc),
                    "target_anchor_utc": str(pair.target_anchor_utc),
                    "business_days": int(round(float(business_days))),
                    "nominal_strike": float(strike),
                    "aggregated_percent_strike": float(
                        np.average(bucket["percent_strike"], weights=weights)
                    ),
                    "canonical_unique_raw_rows": int(len(bucket)),
                    "raw_occurrence_count": occurrence_count,
                    "volume_sum": float(weights.sum()),
                    "volume_ess": bucket_ess,
                    "singleton_raw_occurrence_bucket": bool(singleton),
                }
            )
        singleton_bucket_fraction = singleton_bucket_count / q_i if q_i else 1.0

        occurrence_input = pair_candidates.copy()
        occurrence_input["bootstrap_weight"] = (
            occurrence_input["weight"].astype(float)
            * occurrence_input["raw_option_occurrence_count"].astype(float)
        )
        occurrence_params, _, occurrence_status = _build_raw_surface(
            occurrence_input,
            min_strikes_per_expiry=min_strikes_per_expiry,
            min_expiries_per_surface=min_expiries_per_surface,
        )
        occurrence_valid = occurrence_params is not None
        occurrence_support_match = False
        occurrence_max_abs_error = np.nan
        occurrence_masked_mae = np.nan
        if occurrence_params is not None:
            occurrence_values, occurrence_support = _surface_values(
                occurrence_params,
                strike_grid=strike_grid,
                maturity_grid=maturity_grid,
            )
            occurrence_support_match = bool(
                np.array_equal(canonical_support, occurrence_support)
            )
            occurrence_eval_mask = canonical_support & occurrence_support
            if occurrence_eval_mask.any():
                occurrence_errors = np.abs(
                    occurrence_values[occurrence_eval_mask]
                    - canonical_values[occurrence_eval_mask]
                )
                occurrence_max_abs_error = float(np.max(occurrence_errors))
                occurrence_masked_mae = float(np.mean(occurrence_errors))
        official_replay_mae = np.nan
        official_replay_max_abs_error = np.nan
        official_support_match: bool | float = np.nan
        if str(pair.target_surface_param_json).strip():
            official_params = parse_raw_surface_params(pair.target_surface_param_json)
            official_values, official_support = _surface_values(
                official_params,
                strike_grid=strike_grid,
                maturity_grid=maturity_grid,
            )
            official_mask = canonical_support & official_support
            if not canonical_support.any():
                raise LabelReliabilityError(
                    f"Canonical target has zero frozen-grid support for pair {pair.pair_id}."
                )
            official_replay_max_abs_error = float(
                np.max(
                    np.abs(
                        canonical_values[canonical_support]
                        - official_values[canonical_support]
                    )
                )
            )
            if official_mask.any():
                official_replay_mae = float(
                    np.mean(np.abs(canonical_values[official_mask] - official_values[official_mask]))
                )
            official_support_match = bool(np.array_equal(canonical_support, official_support))
            if not official_support_match:
                raise LabelReliabilityError(
                    f"Canonical target support replay mismatch for pair {pair.pair_id}."
                )
            if official_replay_max_abs_error > canonical_replay_tolerance:
                raise LabelReliabilityError(
                    "Canonical target replay exceeds tolerance for pair "
                    f"{pair.pair_id}: max_abs_error={official_replay_max_abs_error:.17g}, "
                    f"tolerance={canonical_replay_tolerance:.17g}."
                )

        method_summaries: dict[str, dict[str, float | int]] = {}
        for method in methods:
            valid_surfaces: list[np.ndarray] = []
            valid_masks: list[np.ndarray] = []
            valid_maes: list[float] = []
            valid_retentions: list[float] = []
            for draw_id in range(draws):
                draw_input = pair_candidates.copy()
                draw_input["bootstrap_weight"] = _bootstrap_weights(
                    draw_input,
                    method=method,
                    seed=seed,
                    draw_id=draw_id,
                )
                params, used_rows, status = _build_raw_surface(
                    draw_input,
                    min_strikes_per_expiry=min_strikes_per_expiry,
                    min_expiries_per_surface=min_expiries_per_surface,
                )
                valid = False
                invalid_reason = status if params is None else ""
                eval_count = 0
                retention = 0.0
                mae = np.nan
                support_count = 0
                draw_q = 0
                if params is not None:
                    try:
                        draw_values, draw_support = _surface_values(
                            params,
                            strike_grid=strike_grid,
                            maturity_grid=maturity_grid,
                        )
                        evaluation_mask = original_joint & draw_support
                        eval_count = int(evaluation_mask.sum())
                        support_count = int(draw_support.sum())
                        draw_q = int(
                            used_rows.groupby(["business_days", "strike"]).ngroups
                        )
                        retention = eval_count / c_i if c_i > 0 else 0.0
                        if eval_count > 0:
                            mae = float(
                                np.mean(
                                    np.abs(
                                        draw_values[evaluation_mask]
                                        - canonical_values[evaluation_mask]
                                    )
                                )
                            )
                            valid = math.isfinite(mae)
                            invalid_reason = "" if valid else "nonfinite_masked_mae"
                            if (
                                valid
                                and method == "support_preserving_bayesian"
                                and not np.array_equal(draw_support, canonical_support)
                            ):
                                valid = False
                                invalid_reason = "bayesian_support_changed"
                        else:
                            invalid_reason = "zero_original_joint_draw_support"
                    except (ValueError, FloatingPointError) as exc:
                        invalid_reason = f"surface_reconstruction:{type(exc).__name__}"
                        draw_values = np.full_like(canonical_values, np.nan)
                        draw_support = np.zeros_like(canonical_support)
                    if valid:
                        valid_surfaces.append(draw_values)
                        valid_masks.append(draw_support)
                        valid_maes.append(mae)
                        valid_retentions.append(retention)
                draw_rows.append(
                    {
                        "tolerance_minutes": int(pair.tolerance_minutes),
                        "pair_id": str(pair.pair_id),
                        "session_id": str(pair.session_id),
                        "effective_origin_utc": str(pair.effective_origin_utc),
                        "target_anchor_utc": str(pair.target_anchor_utc),
                        "method": method,
                        "draw_id": int(draw_id),
                        "valid_draw": bool(valid),
                        "invalid_reason": invalid_reason,
                        "canonical_joint_cell_count": c_i,
                        "draw_target_support_cell_count": support_count,
                        "evaluation_cell_count": eval_count,
                        "original_joint_cell_retention_fraction": retention,
                        "masked_mae": mae,
                        "draw_surface_bucket_count": draw_q,
                    }
                )
            valid_count = len(valid_surfaces)
            invalid_count = draws - valid_count
            method_summaries[method] = {
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "mae_median": float(np.median(valid_maes)) if valid_maes else np.nan,
                "retention_mean": (
                    float(np.mean(valid_retentions)) if valid_retentions else 0.0
                ),
            }
            shape = canonical_values.shape
            stack = np.full((valid_count, *shape), np.nan, dtype=np.float64)
            for index, (surface, mask) in enumerate(zip(valid_surfaces, valid_masks)):
                stack[index][mask] = surface[mask]
            if valid_count:
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(stack, axis=0)
                    std = np.nanstd(stack, axis=0)
                    p05, p50, p95 = np.nanpercentile(stack, [5, 50, 95], axis=0)
                    mad = np.nanmedian(np.abs(stack - p50[None, :, :]), axis=0)
                cell_valid_count = np.isfinite(stack).sum(axis=0).astype(np.int32)
            else:
                mean = np.full(shape, np.nan)
                std = np.full(shape, np.nan)
                p05 = np.full(shape, np.nan)
                p50 = np.full(shape, np.nan)
                p95 = np.full(shape, np.nan)
                mad = np.full(shape, np.nan)
                cell_valid_count = np.zeros(shape, dtype=np.int32)
            key = f"{int(pair.tolerance_minutes):02d}m|{pair.pair_id}|{method}"
            array_keys.append(key)
            for name, value in (
                ("canonical_target", canonical_values),
                ("bootstrap_mean", mean),
                ("bootstrap_std", std),
                ("bootstrap_p05", p05),
                ("bootstrap_p50", p50),
                ("bootstrap_p95", p95),
                ("bootstrap_mad", mad),
                ("valid_draw_count", cell_valid_count),
                ("current_support", current_support),
                ("canonical_target_support", canonical_support),
                ("original_joint_support", original_joint),
            ):
                array_values[name].append(np.asarray(value))
            for maturity_index, maturity in enumerate(maturity_grid):
                for strike_index, strike in enumerate(strike_grid):
                    cell_rows.append(
                        {
                            "tolerance_minutes": int(pair.tolerance_minutes),
                            "pair_id": str(pair.pair_id),
                            "session_id": str(pair.session_id),
                            "effective_origin_utc": str(pair.effective_origin_utc),
                            "target_anchor_utc": str(pair.target_anchor_utc),
                            "method": method,
                            "maturity_index": int(maturity_index),
                            "strike_index": int(strike_index),
                            "maturity_days": float(maturity),
                            "percent_strike": float(strike),
                            "canonical_target_iv": float(
                                canonical_values[maturity_index, strike_index]
                            ),
                            "canonical_current_iv": float(
                                current_values[maturity_index, strike_index]
                            ),
                            "canonical_delta_iv": float(
                                canonical_values[maturity_index, strike_index]
                                - current_values[maturity_index, strike_index]
                            ),
                            "current_support": bool(
                                current_support[maturity_index, strike_index]
                            ),
                            "canonical_target_support": bool(
                                canonical_support[maturity_index, strike_index]
                            ),
                            "original_joint_support": bool(
                                original_joint[maturity_index, strike_index]
                            ),
                            "valid_draw_count": int(
                                cell_valid_count[maturity_index, strike_index]
                            ),
                            "support_probability": float(
                                cell_valid_count[maturity_index, strike_index] / draws
                            ),
                            "bootstrap_mean_iv": float(mean[maturity_index, strike_index]),
                            "bootstrap_std_iv": float(std[maturity_index, strike_index]),
                            "bootstrap_mad_iv": float(mad[maturity_index, strike_index]),
                            "bootstrap_p05_iv": float(p05[maturity_index, strike_index]),
                            "bootstrap_p50_iv": float(p50[maturity_index, strike_index]),
                            "bootstrap_p95_iv": float(p95[maturity_index, strike_index]),
                            "bootstrap_delta_mean": float(
                                mean[maturity_index, strike_index]
                                - current_values[maturity_index, strike_index]
                            ),
                            "bootstrap_delta_std": float(std[maturity_index, strike_index]),
                            "bootstrap_delta_p05": float(
                                p05[maturity_index, strike_index]
                                - current_values[maturity_index, strike_index]
                            ),
                            "bootstrap_delta_p50": float(
                                p50[maturity_index, strike_index]
                                - current_values[maturity_index, strike_index]
                            ),
                            "bootstrap_delta_p95": float(
                                p95[maturity_index, strike_index]
                                - current_values[maturity_index, strike_index]
                            ),
                            "pair_raw_occurrence_count": canonical_raw_occurrences,
                            "pair_canonical_unique_raw_rows": canonical_unique_raw_rows,
                            "pair_volume_ess": canonical_volume_ess,
                            "pair_singleton_bucket_fraction": singleton_bucket_fraction,
                        }
                    )
        bayesian = method_summaries["support_preserving_bayesian"]
        cluster = method_summaries["strike_cluster"]
        required_valid = max(1, int(math.ceil(draws * minimum_valid_draw_fraction)))
        u_value = float(bayesian["mae_median"])
        u_estimable = bool(
            int(bayesian["valid_count"]) >= required_valid
            and math.isfinite(u_value)
            and u_value > uncertainty_numeric_tolerance
            and m_i >= 1
        )
        formal_count = getattr(pair, "formal_joint_support_cell_count", np.nan)
        if not pd.isna(formal_count) and int(formal_count) != c_i:
            raise LabelReliabilityError(
                "Canonical joint support replay mismatch for pair "
                f"{pair.pair_id}: replay={c_i}, formal={int(formal_count)}."
            )
        pair_rows.append(
            {
                "pair_id": str(pair.pair_id),
                "tolerance_minutes": int(pair.tolerance_minutes),
                "session_id": str(pair.session_id),
                "effective_origin_utc": str(pair.effective_origin_utc),
                "target_anchor_utc": str(pair.target_anchor_utc),
                "c_i": c_i,
                "q_i": q_i,
                "m_i": m_i,
                "u_i": u_value,
                "u_estimable": u_estimable,
                "v_i": int(cluster["valid_count"]) / draws,
                "h_i": float(cluster["retention_mean"]),
                "requested_draws_per_method": int(draws),
                "bayesian_valid_draws": int(bayesian["valid_count"]),
                "bayesian_invalid_draws": int(bayesian["invalid_count"]),
                "cluster_valid_draws": int(cluster["valid_count"]),
                "cluster_invalid_draws": int(cluster["invalid_count"]),
                "minimum_valid_draws_for_u": int(required_valid),
                "uncertainty_numeric_tolerance": uncertainty_numeric_tolerance,
                "canonical_raw_occurrence_count": canonical_raw_occurrences,
                "canonical_unique_raw_rows": canonical_unique_raw_rows,
                "canonical_volume_ess": canonical_volume_ess,
                "singleton_bucket_count": singleton_bucket_count,
                "singleton_bucket_fraction": singleton_bucket_fraction,
                "formal_joint_support_cell_count": formal_count,
                "formal_joint_support_matches_replay": (
                    bool(int(formal_count) == c_i)
                    if not pd.isna(formal_count)
                    else np.nan
                ),
                "official_target_replay_masked_mae": official_replay_mae,
                "official_target_replay_max_abs_error": official_replay_max_abs_error,
                "official_target_support_matches_replay": official_support_match,
                "occurrence_weighted_replay_valid": occurrence_valid,
                "occurrence_weighted_replay_status": occurrence_status,
                "occurrence_weighted_support_matches_canonical": occurrence_support_match,
                "occurrence_weighted_max_abs_error": occurrence_max_abs_error,
                "occurrence_weighted_masked_mae": occurrence_masked_mae,
            }
        )
    arrays: dict[str, np.ndarray] = {
        "keys": np.asarray(array_keys, dtype=np.str_),
        "strike_grid": np.asarray(strike_grid, dtype=np.float64),
        "maturity_days_grid": np.asarray(maturity_grid, dtype=np.float64),
    }
    for name, values in array_values.items():
        arrays[name] = np.stack(values, axis=0)
    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(draw_rows),
        pd.DataFrame(cell_rows),
        pd.DataFrame(bucket_rows),
        arrays,
    )


def compute_reliability_scores(
    pair_metrics: pd.DataFrame | Sequence[Mapping[str, Any]],
    fold_train_pair_ids: Iterable[str],
    tolerance_minutes: int,
) -> pd.DataFrame:
    """Compute frozen fold-local reliability scores without reading Q3/Q4.

    ``tau_f`` is the Q75 of estimable ``u_i`` among the supplied fold-training
    IDs at one tolerance.  The returned frame contains only that tolerance and
    adds the six reliability factors, ``R_i`` and ``r_i``.
    """

    frame = (
        pair_metrics.copy()
        if isinstance(pair_metrics, pd.DataFrame)
        else pd.DataFrame(list(pair_metrics))
    )
    missing = sorted(set(PAIR_SCORE_COLUMNS) - set(frame.columns))
    if missing:
        raise LabelReliabilityError(f"Pair metrics are missing score columns: {missing}")
    tolerance = int(tolerance_minutes)
    work = frame.loc[
        pd.to_numeric(frame["tolerance_minutes"], errors="coerce").eq(tolerance)
    ].copy()
    if work.empty:
        raise LabelReliabilityError(f"No pair metrics for tolerance {tolerance}m.")
    if work["pair_id"].astype(str).duplicated().any():
        raise LabelReliabilityError("Pair metrics must be unique within tolerance.")
    train_ids = {str(pair_id) for pair_id in fold_train_pair_ids}
    if not train_ids:
        raise LabelReliabilityError("fold_train_pair_ids is empty.")
    u_values = pd.to_numeric(work["u_i"], errors="coerce")
    estimable = work["u_estimable"].map(_truthy) & np.isfinite(u_values) & u_values.ge(0)
    fold_mask = work["pair_id"].astype(str).isin(train_ids) & estimable
    missing_ids = train_ids - set(work["pair_id"].astype(str))
    if missing_ids:
        raise LabelReliabilityError(
            f"fold_train_pair_ids contains IDs outside {tolerance}m metrics: "
            f"{sorted(missing_ids)[:5]}"
        )
    if not fold_mask.any():
        raise LabelReliabilityError("No estimable training-pair uncertainty values.")
    tau_f = float(np.quantile(u_values.loc[fold_mask].to_numpy(dtype=float), 0.75))
    if not math.isfinite(tau_f) or tau_f <= 0:
        raise LabelReliabilityError(f"Fold uncertainty scale tau_f must be > 0, got {tau_f}.")
    numeric = {
        column: pd.to_numeric(work[column], errors="coerce").fillna(0.0).clip(lower=0.0)
        for column in ("c_i", "q_i", "m_i", "v_i", "h_i")
    }
    work["tau_f"] = tau_f
    work["support_factor_s_i"] = np.minimum(1.0, np.sqrt(numeric["c_i"] / 16.0))
    work["density_factor_d_i"] = np.minimum(1.0, np.sqrt(numeric["q_i"] / 8.0))
    work["replication_factor_e_i"] = np.minimum(1.0, np.sqrt(numeric["m_i"] / 4.0))
    work["uncertainty_factor_b_i"] = 0.0
    work.loc[estimable, "uncertainty_factor_b_i"] = 1.0 / (
        1.0 + np.square(u_values.loc[estimable] / tau_f)
    )
    work["validity_factor_v_i"] = numeric["v_i"].clip(upper=1.0)
    work["retention_factor_h_i"] = numeric["h_i"].clip(upper=1.0)
    product = (
        work["support_factor_s_i"]
        * work["density_factor_d_i"]
        * work["replication_factor_e_i"]
        * work["uncertainty_factor_b_i"]
        * work["validity_factor_v_i"]
        * work["retention_factor_h_i"]
    )
    work["R_i"] = np.power(product.clip(lower=0.0, upper=1.0), 1.0 / 6.0)
    work["r_i"] = 0.5 + 0.5 * work["R_i"]
    return work.reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    return path


def _write_deterministic_npz(arrays: Mapping[str, np.ndarray], path: Path) -> Path:
    """Write NPZ without wall-clock ZIP headers or pickle payloads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return path


def _artifact_hash_rows(paths: Sequence[Path], *, root: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "relative_path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(paths)
        ]
    )


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _jsonable(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _resume_existing(
    config: Mapping[str, Any], output_dir: Path, manifest_path: Path
) -> Path:
    if not manifest_path.is_file():
        raise LabelReliabilityError(
            f"Cannot resume incomplete reliability directory without manifest: {output_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise LabelReliabilityError("Resume manifest schema version mismatch.")
    if manifest.get("config_sha256") != _config_fingerprint(config):
        raise LabelReliabilityError("Resume config hash mismatch.")
    for source in manifest.get("inputs", []):
        path = Path(str(source["path"]))
        if not path.is_file() or _sha256_file(path) != str(source["sha256"]):
            raise LabelReliabilityError(f"Resume input hash mismatch: {path}")
    hash_path = output_dir / "output_hashes.csv.gz"
    if not hash_path.is_file():
        raise LabelReliabilityError("Resume output hash manifest is missing.")
    hashes = pd.read_csv(hash_path)
    for row in hashes.itertuples(index=False):
        path = output_dir / str(row.relative_path)
        if not path.is_file() or _sha256_file(path) != str(row.sha256):
            raise LabelReliabilityError(f"Resume output hash mismatch: {path}")
    return manifest_path


def run_label_reliability_bootstrap(
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    """Run the pre-Q3, target-only label reliability bootstrap.

    Args:
        config: Either the ``label_reliability`` mapping or a mapping containing
            that top-level key.  Required inputs are ``pair_manifests`` (exactly
            5m/30m), ``pre_q3_end_utc``, ``priced_target_rows``, ``raw_files``,
            and a frozen 16x16 grid (the shape is configurable for tests).
        output_dir: New immutable artifact directory.
        resume: Validate every recorded input/output hash and return the existing
            manifest.  Resume never appends or repairs a partial run.

    Returns:
        Path to ``label_bootstrap_manifest.json``.
    """

    payload = config.get("label_reliability", config)
    if not isinstance(payload, Mapping):
        raise LabelReliabilityError("label_reliability config must be a mapping.")
    resolved_config = dict(payload)
    target_dir = Path(output_dir).expanduser().resolve()
    manifest_path = target_dir / "label_bootstrap_manifest.json"
    if target_dir.exists() and any(target_dir.iterdir()):
        if resume:
            return _resume_existing(resolved_config, target_dir, manifest_path)
        raise FileExistsError(f"Reliability output directory is not empty: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    draws = int(resolved_config.get("draws", 1000))
    seed = int(resolved_config.get("seed", 42))
    methods = tuple(str(value) for value in resolved_config.get("methods", BOOTSTRAP_METHODS))
    if draws <= 0:
        raise LabelReliabilityError("draws must be a positive integer.")
    if set(methods) != set(BOOTSTRAP_METHODS) or len(methods) != len(BOOTSTRAP_METHODS):
        raise LabelReliabilityError(
            f"methods must contain exactly {list(BOOTSTRAP_METHODS)}."
        )
    minimum_valid_fraction = float(
        resolved_config.get("minimum_valid_draw_fraction", 0.8)
    )
    if not 0 < minimum_valid_fraction <= 1:
        raise LabelReliabilityError("minimum_valid_draw_fraction must be in (0, 1].")
    canonical_replay_tolerance = float(
        resolved_config.get("canonical_replay_tolerance", 1.0e-12)
    )
    uncertainty_numeric_tolerance = float(
        resolved_config.get("uncertainty_numeric_tolerance", 1.0e-15)
    )
    if canonical_replay_tolerance < 0 or not math.isfinite(canonical_replay_tolerance):
        raise LabelReliabilityError("canonical_replay_tolerance must be finite and >= 0.")
    if uncertainty_numeric_tolerance < 0 or not math.isfinite(
        uncertainty_numeric_tolerance
    ):
        raise LabelReliabilityError("uncertainty_numeric_tolerance must be finite and >= 0.")

    pairs, pair_paths, strikes, maturities, scope = _load_pair_manifests(
        resolved_config
    )
    priced_rows, priced_path, pricing_schema = _load_priced_rows(
        resolved_config, pairs
    )
    if "raw_files" not in resolved_config:
        raise LabelReliabilityError("raw_files is required.")
    raw_files = _resolve_raw_files(resolved_config["raw_files"])
    lineage, source_manifest = _scan_raw_occurrences(
        raw_files,
        priced_rows=priced_rows,
        pairs=pairs,
        chunk_size=int(resolved_config.get("raw_chunk_size", 250_000)),
        max_underlying_staleness_seconds=int(
            resolved_config.get("max_underlying_staleness_seconds", 60)
        ),
    )
    replay, membership = _attach_lineage(priced_rows, pairs, lineage)
    pair_metrics, draw_metrics, cell_metrics, bucket_metrics, arrays = _run_bootstraps(
        pairs,
        replay,
        strike_grid=strikes,
        maturity_grid=maturities,
        draws=draws,
        seed=seed,
        methods=methods,
        min_strikes_per_expiry=int(resolved_config.get("min_strikes_per_expiry", 2)),
        min_expiries_per_surface=int(
            resolved_config.get("min_expiries_per_surface", 2)
        ),
        minimum_valid_draw_fraction=minimum_valid_fraction,
        canonical_replay_tolerance=canonical_replay_tolerance,
        uncertainty_numeric_tolerance=uncertainty_numeric_tolerance,
    )
    missing_score_columns = sorted(set(PAIR_SCORE_COLUMNS) - set(pair_metrics.columns))
    if missing_score_columns:
        raise AssertionError(f"Internal pair score schema failure: {missing_score_columns}")

    input_paths = sorted(set([*pair_paths, priced_path, *raw_files]))
    input_rows = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in input_paths
    ]
    artifacts = [
        _write_csv(pairs, target_dir / "pre_q3_pair_manifest.csv.gz"),
        _write_csv(source_manifest, target_dir / "raw_source_manifest.csv.gz"),
        _write_csv(lineage, target_dir / "raw_trade_lineage.csv.gz"),
        _write_csv(membership, target_dir / "raw_window_membership.csv.gz"),
        _write_csv(replay, target_dir / "canonical_dedup_replay.csv.gz"),
        _write_csv(
            pair_metrics, target_dir / "label_reliability_pair_metrics.csv.gz"
        ),
        _write_csv(draw_metrics, target_dir / "bootstrap_draw_status.csv.gz"),
        _write_csv(
            cell_metrics, target_dir / "label_reliability_cell_metrics.csv.gz"
        ),
        _write_csv(
            bucket_metrics, target_dir / "label_reliability_bucket_metrics.csv.gz"
        ),
        _write_csv(pd.DataFrame(input_rows), target_dir / "input_hashes.csv.gz"),
    ]
    arrays_path = _write_deterministic_npz(arrays, target_dir / "cell_arrays.npz")
    artifacts.append(arrays_path)
    output_hashes = _artifact_hash_rows(artifacts, root=target_dir)
    output_hash_path = _write_csv(output_hashes, target_dir / "output_hashes.csv.gz")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis_scope": "pre_q3_target_only_label_reliability",
        "pre_q3_end_utc": scope["interval_end_utc"],
        "forecast_horizon_minutes": int(scope["forecast_horizon_minutes"]),
        "tolerance_semantics": (
            "5m/30m are news-origin alignment tolerances; both labels forecast the same 5m horizon"
        ),
        "target_cache_semantics": (
            "target rows are window_side=backward with calibration_datetime_utc equal to "
            "target_anchor_utc, covering [target_anchor-5m,target_anchor)"
        ),
        "config_sha256": _config_fingerprint(resolved_config),
        "draws_per_method": draws,
        "formal_1000_draw_contract_satisfied": draws == 1000,
        "seed": seed,
        "methods": {
            "support_preserving_bayesian": (
                "deterministic_raw-row_Exp(1)_weighted_bootstrap_v1"
            ),
            "strike_cluster": (
                "deterministic_Poisson(1)_trade-date_maturity_nominal-strike_cluster_v1"
            ),
        },
        "common_random_number_unit": (
            "canonical raw option row fingerprint; cluster IDs are stable across overlapping windows"
        ),
        "canonical_duplicate_policy": (
            "one cached-IV candidate per target anchor and semantic raw trade fingerprint; "
            "raw occurrence multiplicity is diagnostic only"
        ),
        "pricing_inversion_mode": "cached_black76_iv_not_independently_rerun",
        "pricing_context_columns": pricing_schema,
        "underlying_raw_locator_missing_rows": int(
            replay["underlying_raw_locator_missing"].sum()
        ),
        "rate_curve_raw_locator_scope": (
            "rate_curve_sha256 is frozen from precalibration; curve source row lineage is not replayed"
        ),
        "bootstrap_surfaces_training_access": "forbidden_summary_artifacts_only",
        "current_delta_diagnostic": (
            "current surface is fixed; target-current bootstrap summaries are diagnostic only "
            "and excluded from c/q/m/u/v/h scoring"
        ),
        "occurrence_multiplicity_sensitivity": (
            "canonical-vs-occurrence-weighted target differences are diagnostic only"
        ),
        "pair_count": int(len(pair_metrics)),
        "pair_counts_by_tolerance": {
            str(int(key)): int(value)
            for key, value in pair_metrics.groupby("tolerance_minutes")["pair_id"]
            .nunique()
            .items()
        },
        "raw_occurrence_count": int(len(lineage)),
        "canonical_replay_row_count": int(len(replay)),
        "invalid_draw_count": int((~draw_metrics["valid_draw"]).sum()),
        "grid": {
            "shape": [len(maturities), len(strikes)],
            "strike_grid": [float(value) for value in strikes],
            "maturity_days_grid": [float(value) for value in maturities],
        },
        "inputs": input_rows,
        "artifact_schema": {
            "raw_trade_lineage": "raw_trade_lineage.csv.gz",
            "draw_status": "bootstrap_draw_status.csv.gz",
            "pair_metrics": "label_reliability_pair_metrics.csv.gz",
            "cell_metrics": "label_reliability_cell_metrics.csv.gz",
            "bucket_metrics": "label_reliability_bucket_metrics.csv.gz",
            "cell_arrays": "cell_arrays.npz",
        },
        "output_hashes_path": str(output_hash_path.relative_to(target_dir)),
        "output_hashes_sha256": _sha256_file(output_hash_path),
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "BOOTSTRAP_METHODS",
    "LabelReliabilityError",
    "PAIR_SCORE_COLUMNS",
    "compute_reliability_scores",
    "run_label_reliability_bootstrap",
]
