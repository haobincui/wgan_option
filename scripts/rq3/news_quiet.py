"""News-arrival vs no-news quiet post-processing for RQ3."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import scripts._path_setup  # noqa: F401

from scripts.rq3.event_study import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    GAN_SHEET,
    METRIC_COLUMNS,
    WORKBOOK_EVENT_TEST_METRICS,
    _write_csv,
    _write_json,
    build_workbook_sample_metrics,
    parse_serialized_list,
    parse_timestamp_utc,
    quality_audit,
)
from wgan_option.merge_support import (
    DEFAULT_DAYS_IN_YEAR,
    DEFAULT_SOURCE_TIMEZONE,
    build_surface_from_params,
    load_news_base_frame,
    serialize_list,
)

NEWS_EVENT_GROUP = "news_event"
QUIET_GROUP = "quiet"
NEWS_QUIET_GROUPS = {NEWS_EVENT_GROUP, QUIET_GROUP}
PASSTHROUGH_COLUMNS = [
    "event_group",
    "has_news",
    "news_cluster_id",
    "quiet_buffer_minutes",
    "quiet_grid_minutes",
]
MAIN_RESULT_METRICS = [
    ("surface_mae", "mae"),
    ("short_atm_weighted_mae", "short_atm_weighted_mae"),
    ("atm_short_pure_mae", "atm_short_pure_mae"),
]


def _ceil_timestamp_to_grid(stamp: pd.Timestamp, *, grid_minutes: int) -> pd.Timestamp:
    grid = int(grid_minutes)
    if grid <= 0:
        raise ValueError("quiet_grid_minutes must be positive.")
    original = pd.Timestamp(stamp).tz_convert("UTC")
    floored = original.floor("min")
    minute_mod = int(floored.minute) % grid
    if minute_mod == 0 and original == floored:
        return floored
    delta = grid - minute_mod if minute_mod else grid
    return (floored + pd.Timedelta(minutes=delta)).floor("min")


def _floor_timestamp_to_grid(stamp: pd.Timestamp, *, grid_minutes: int) -> pd.Timestamp:
    grid = int(grid_minutes)
    if grid <= 0:
        raise ValueError("quiet_grid_minutes must be positive.")
    ts = pd.Timestamp(stamp).tz_convert("UTC").floor("min")
    return ts - pd.Timedelta(minutes=int(ts.minute) % grid)


def _utc_string(value: Any) -> str:
    return parse_timestamp_utc(value, field_name="timestamp").strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_key(stamp: pd.Timestamp) -> str:
    return pd.Timestamp(stamp).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_on_grid(stamp: pd.Timestamp, *, grid_minutes: int) -> bool:
    grid = int(grid_minutes)
    if grid <= 0:
        raise ValueError("quiet_grid_minutes must be positive.")
    return int(stamp.second) == 0 and int(stamp.microsecond) == 0 and int(stamp.minute) % grid == 0


def _nearest_delta_minutes(stamp: pd.Timestamp, news_ns: np.ndarray) -> float:
    if news_ns.size == 0:
        return float("inf")
    value = int(pd.Timestamp(stamp).value)
    idx = int(np.searchsorted(news_ns, value))
    candidates: list[int] = []
    if idx < news_ns.size:
        candidates.append(abs(int(news_ns[idx]) - value))
    if idx > 0:
        candidates.append(abs(value - int(news_ns[idx - 1])))
    if not candidates:
        return float("inf")
    return float(min(candidates) / 1_000_000_000.0 / 60.0)


def _load_all_surface_json(path: str | Path) -> dict[str, Mapping[str, Any]]:
    surface_path = Path(path).expanduser()
    if not surface_path.exists():
        raise FileNotFoundError(
            f"surface all JSON does not exist: {surface_path}. "
            "Run generate_surface with data_range=all first."
        )
    payload = json.loads(surface_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"surface all JSON must be a timestamp mapping: {surface_path}")
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_key, raw_entry in payload.items():
        if not isinstance(raw_entry, Mapping):
            continue
        if "surface_model" not in raw_entry or "surface_params" not in raw_entry:
            raise ValueError(
                f"{surface_path} does not look like data_range=all output. "
                "Expected timestamp -> {surface_model, surface_params}; "
                "window/excel payloads with backward/forward cannot define no-news quiet samples."
            )
        normalized[_utc_string(raw_key)] = raw_entry
    if not normalized:
        raise ValueError(f"No usable all-surface entries found in {surface_path}")
    return normalized


def _load_window_surface_json(path: str | Path) -> dict[str, Mapping[str, Any]]:
    surface_path = Path(path).expanduser()
    if not surface_path.exists():
        raise FileNotFoundError(f"window surface JSON does not exist: {surface_path}")
    payload = json.loads(surface_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"window surface JSON must be a timestamp mapping: {surface_path}")
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_key, raw_entry in payload.items():
        if not isinstance(raw_entry, Mapping):
            continue
        if "backward" not in raw_entry or "forward" not in raw_entry:
            raise ValueError(
                f"{surface_path} does not look like data_range=window output. "
                "Expected timestamp -> {backward, forward}."
            )
        normalized[_utc_string(raw_key)] = raw_entry
    if not normalized:
        raise ValueError(f"No usable window-surface entries found in {surface_path}")
    return normalized


def _surface_flat_from_entry(
    *,
    entry: Mapping[str, Any],
    timestamp_utc: str,
    strike_grid: Sequence[float],
    maturity_days_grid: Sequence[float],
) -> list[float]:
    surface_model = str(entry.get("surface_model") or "").strip()
    surface_params = entry.get("surface_params")
    if not surface_model or not isinstance(surface_params, Mapping):
        raise ValueError(f"Surface entry at {timestamp_utc} is missing surface_model/surface_params.")
    surface = build_surface_from_params(
        surface_model=surface_model,
        surface_params=surface_params,
        valuation_date=parse_timestamp_utc(timestamp_utc, field_name="surface timestamp").date(),
        days_in_year=DEFAULT_DAYS_IN_YEAR,
    )
    grid = surface.implied_vol_surface(
        percent_strikes=[float(value) for value in strike_grid],
        business_days=[int(round(float(value))) for value in maturity_days_grid],
        forward=1.0,
    )
    return [float(value) for row in grid for value in row]


def _surface_slice_count(entry: Mapping[str, Any]) -> int:
    params = entry.get("surface_params")
    if not isinstance(params, Mapping):
        return 0
    business_days = params.get("business_days")
    if isinstance(business_days, Sequence) and not isinstance(business_days, (str, bytes)):
        return len(business_days)
    return 0


def _embedding_dim(frame: pd.DataFrame, embedding_column: str, dim_column: str) -> int:
    if dim_column in frame.columns:
        numeric = pd.to_numeric(frame[dim_column], errors="coerce").dropna()
        if not numeric.empty and int(numeric.iloc[0]) > 0:
            return int(numeric.iloc[0])
    if embedding_column in frame.columns:
        for value in frame[embedding_column].tolist():
            parsed = parse_serialized_list(value)
            if parsed:
                return len(parsed)
    return 0


def _zero_embedding(dim: int) -> str:
    return serialize_list([0.0] * int(dim))


def _prepare_event_rows(source_frame: pd.DataFrame, *, quiet_buffer_minutes: int, quiet_grid_minutes: int) -> pd.DataFrame:
    frame = source_frame.copy()
    if "training_candidate_flag" in frame.columns:
        candidate_flags = pd.to_numeric(frame["training_candidate_flag"], errors="coerce").fillna(0).astype(int)
        filtered = frame[candidate_flags == 1].copy()
        if not filtered.empty:
            frame = filtered
    frame["event_group"] = NEWS_EVENT_GROUP
    frame["has_news"] = 1
    frame["news_cluster_id"] = frame["sample_id"].astype(str) if "sample_id" in frame.columns else ""
    frame["quiet_buffer_minutes"] = int(quiet_buffer_minutes)
    frame["quiet_grid_minutes"] = int(quiet_grid_minutes)
    return frame.reset_index(drop=True)


def _news_timestamps_from_xlsx(news_xlsx: str | Path, *, horizon_minutes: int) -> list[pd.Timestamp]:
    news_frame = load_news_base_frame(
        Path(news_xlsx).expanduser(),
        source_timezone=DEFAULT_SOURCE_TIMEZONE,
        offset_minutes=int(horizon_minutes),
    )
    timestamps: list[pd.Timestamp] = []
    for value in news_frame["timestamp_utc"].dropna().tolist():
        text = str(value).strip()
        if not text:
            continue
        timestamps.append(parse_timestamp_utc(text, field_name="timestamp_utc"))
    return sorted(timestamps)


def _load_event_rows(
    source_merged_vol: str | Path,
    *,
    sheet_name: str,
    quiet_buffer_minutes: int,
    quiet_grid_minutes: int,
) -> pd.DataFrame:
    source_path = Path(source_merged_vol).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"source merged-vol workbook does not exist: {source_path}")
    event_source = pd.read_excel(source_path, sheet_name=sheet_name, dtype=object)
    event_rows = _prepare_event_rows(
        event_source,
        quiet_buffer_minutes=int(quiet_buffer_minutes),
        quiet_grid_minutes=int(quiet_grid_minutes),
    )
    if event_rows.empty:
        raise ValueError(f"No usable news-event rows found in {source_path}")
    return event_rows


def _event_time_bounds(event_rows: pd.DataFrame, *, horizon_minutes: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_values: list[pd.Timestamp] = []
    end_values: list[pd.Timestamp] = []
    for column in ("current_snapshot_time_utc", "news_timestamp_utc"):
        if column in event_rows.columns:
            for value in event_rows[column].dropna().tolist():
                text = str(value).strip()
                if text and text.lower() not in {"nan", "nat", "none"}:
                    start_values.append(parse_timestamp_utc(text, field_name=column))
    for column in ("target_snapshot_time_utc", "news_timestamp_utc"):
        if column in event_rows.columns:
            for value in event_rows[column].dropna().tolist():
                text = str(value).strip()
                if text and text.lower() not in {"nan", "nat", "none"}:
                    end_values.append(parse_timestamp_utc(text, field_name=column))
    if not start_values or not end_values:
        raise ValueError("Could not infer timestamp bounds from source workbook.")
    start_ts = min(start_values)
    end_ts = max(end_values) - pd.Timedelta(minutes=int(horizon_minutes))
    if end_ts <= start_ts:
        raise ValueError(f"Invalid quiet target bounds: start={start_ts}, end={end_ts}")
    return start_ts, end_ts


def prepare_news_quiet_targets(
    *,
    source_merged_vol: str | Path,
    news_xlsx: str | Path,
    output_dir: str | Path,
    horizon_minutes: int = 5,
    quiet_grid_minutes: int = 5,
    quiet_buffer_minutes: int = 60,
    candidate_count: int = 12000,
    sample_seed: int = 20260625,
    sheet_name: str = GAN_SHEET,
) -> Path:
    event_rows = _load_event_rows(
        source_merged_vol,
        sheet_name=sheet_name,
        quiet_buffer_minutes=int(quiet_buffer_minutes),
        quiet_grid_minutes=int(quiet_grid_minutes),
    )
    start_ts, end_ts = _event_time_bounds(event_rows, horizon_minutes=int(horizon_minutes))
    start_ts = _ceil_timestamp_to_grid(start_ts, grid_minutes=int(quiet_grid_minutes))
    end_ts = _floor_timestamp_to_grid(end_ts, grid_minutes=int(quiet_grid_minutes))

    news_timestamps = _news_timestamps_from_xlsx(news_xlsx, horizon_minutes=int(horizon_minutes))
    news_ns = np.asarray([int(stamp.value) for stamp in news_timestamps], dtype=np.int64)
    news_ns.sort()

    rows: list[dict[str, Any]] = []
    eligible_keys: list[str] = []
    current = start_ts
    while current <= end_ts:
        target = current + pd.Timedelta(minutes=int(horizon_minutes))
        nearest_delta = _nearest_delta_minutes(current, news_ns)
        eligible = nearest_delta >= float(quiet_buffer_minutes)
        key = _timestamp_key(current)
        if eligible:
            eligible_keys.append(key)
        rows.append(
            {
                "candidate_timestamp_utc": key,
                "target_timestamp_utc": _timestamp_key(target),
                "nearest_news_delta_minutes": nearest_delta,
                "eligible": int(eligible),
                "selected_for_generation": 0,
                "reason": "eligible" if eligible else "within_news_buffer",
            }
        )
        current += pd.Timedelta(minutes=int(quiet_grid_minutes))

    if not eligible_keys:
        raise ValueError(
            "No quiet target candidates passed the news-buffer rule. "
            "Check source workbook bounds or quiet_buffer_minutes."
        )

    requested = int(candidate_count)
    if requested > 0 and len(eligible_keys) > requested:
        rng = np.random.default_rng(int(sample_seed))
        selected_idx = rng.choice(len(eligible_keys), size=requested, replace=False)
        selected_keys = {eligible_keys[int(idx)] for idx in selected_idx.tolist()}
    else:
        selected_keys = set(eligible_keys)

    for row in rows:
        if row["candidate_timestamp_utc"] in selected_keys:
            row["selected_for_generation"] = 1
            row["reason"] = "selected_for_generation"

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    selected = sorted(selected_keys)
    targets_path = output / "quiet_targets.txt"
    targets_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    _write_csv(audit, output / "quiet_target_audit.csv")
    _write_json(
        {
            "mode": "prepare_news_quiet_targets",
            "source_merged_vol": str(Path(source_merged_vol).expanduser()),
            "news_xlsx": str(Path(news_xlsx).expanduser()),
            "output_dir": str(output),
            "horizon_minutes": int(horizon_minutes),
            "quiet_grid_minutes": int(quiet_grid_minutes),
            "quiet_buffer_minutes": int(quiet_buffer_minutes),
            "candidate_count_requested": int(candidate_count),
            "sample_seed": int(sample_seed),
            "grid_candidate_count": int(len(rows)),
            "eligible_candidate_count": int(len(eligible_keys)),
            "selected_candidate_count": int(len(selected)),
            "start_timestamp_utc": _timestamp_key(start_ts),
            "end_timestamp_utc": _timestamp_key(end_ts),
        },
        output / "quiet_target_manifest.json",
    )
    return output


def build_news_quiet_workbook(
    *,
    source_merged_vol: str | Path,
    surface_all_json: str | Path,
    news_xlsx: str | Path,
    output_workbook: str | Path,
    horizon_minutes: int = 5,
    quiet_grid_minutes: int = 5,
    quiet_buffer_minutes: int = 60,
    sheet_name: str = GAN_SHEET,
) -> Path:
    source_path = Path(source_merged_vol).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"source merged-vol workbook does not exist: {source_path}")
    event_source = pd.read_excel(source_path, sheet_name=sheet_name, dtype=object)
    event_rows = _prepare_event_rows(
        event_source,
        quiet_buffer_minutes=int(quiet_buffer_minutes),
        quiet_grid_minutes=int(quiet_grid_minutes),
    )
    if event_rows.empty:
        raise ValueError(f"No usable news-event rows found in {source_path}")

    strike_grid = parse_serialized_list(event_rows.iloc[0]["strike_grid"])
    maturity_days_grid = parse_serialized_list(event_rows.iloc[0]["maturity_days_grid"])
    surface_shape_text = serialize_list([len(maturity_days_grid), len(strike_grid)])
    dims = {
        "hd": _embedding_dim(event_rows, "hd_embedding", "hd_dim"),
        "lp": _embedding_dim(event_rows, "lp_embedding", "lp_dim"),
        "bow": _embedding_dim(event_rows, "bow_embedding", "bow_dim"),
        "sentiment": _embedding_dim(event_rows, "sentiment_embedding", "sentiment_dim"),
    }

    surface_map = _load_all_surface_json(surface_all_json)
    news_timestamps = _news_timestamps_from_xlsx(news_xlsx, horizon_minutes=int(horizon_minutes))
    news_ns = np.asarray([int(stamp.value) for stamp in news_timestamps], dtype=np.int64)
    news_ns.sort()

    quiet_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for key in sorted(surface_map):
        timestamp = parse_timestamp_utc(key, field_name="surface timestamp")
        if not _is_on_grid(timestamp, grid_minutes=int(quiet_grid_minutes)):
            continue
        target_timestamp = timestamp + pd.Timedelta(minutes=int(horizon_minutes))
        target_key = _timestamp_key(target_timestamp)
        nearest_delta = _nearest_delta_minutes(timestamp, news_ns)
        selected = True
        reason = "selected"
        if target_key not in surface_map:
            selected = False
            reason = "missing_target_surface"
        elif nearest_delta < float(quiet_buffer_minutes):
            selected = False
            reason = "within_news_buffer"
        audit_row = {
            "candidate_timestamp_utc": key,
            "target_timestamp_utc": target_key,
            "selected": int(selected),
            "reason": reason,
            "nearest_news_delta_minutes": nearest_delta,
        }
        if not selected:
            audit_rows.append(audit_row)
            continue
        current_entry = surface_map[key]
        target_entry = surface_map[target_key]
        try:
            current_flat = _surface_flat_from_entry(
                entry=current_entry,
                timestamp_utc=key,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
            )
            target_flat = _surface_flat_from_entry(
                entry=target_entry,
                timestamp_utc=target_key,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
            )
        except Exception as exc:
            audit_row.update({"selected": 0, "reason": f"surface_reconstruction_failed: {exc}"})
            audit_rows.append(audit_row)
            continue

        row = {column: "" for column in event_rows.columns}
        row.update(
            {
                "sample_id": f"quiet_{key.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}",
                "news_timestamp_utc": key,
                "current_snapshot_time_utc": key,
                "target_snapshot_time_utc": target_key,
                "surface_model": str(current_entry.get("surface_model", "")),
                "strike_grid": serialize_list(strike_grid),
                "maturity_days_grid": serialize_list(maturity_days_grid),
                "surface_shape": surface_shape_text,
                "current_surface_flat": serialize_list(current_flat),
                "target_surface_flat": serialize_list(target_flat),
                "current_weighted_iv_rmse": "",
                "target_weighted_iv_rmse": "",
                "pair_quality_label": "quiet_no_news",
                "training_candidate_flag": 1,
                "event_group": QUIET_GROUP,
                "has_news": 0,
                "news_cluster_id": "",
                "quiet_buffer_minutes": int(quiet_buffer_minutes),
                "quiet_grid_minutes": int(quiet_grid_minutes),
            }
        )
        for column, value in {
            "current_has_surface": 1,
            "target_has_surface": 1,
            "current_has_svi": 1 if str(current_entry.get("surface_model", "")) == "svi" else "",
            "target_has_svi": 1 if str(target_entry.get("surface_model", "")) == "svi" else "",
            "current_surface_slice_count": _surface_slice_count(current_entry),
            "target_surface_slice_count": _surface_slice_count(target_entry),
            "current_svi_slice_count": _surface_slice_count(current_entry),
            "target_svi_slice_count": _surface_slice_count(target_entry),
        }.items():
            if column in row:
                row[column] = value
        if "hd_embedding" in row:
            row["hd_embedding"] = _zero_embedding(dims["hd"])
        if "lp_embedding" in row:
            row["lp_embedding"] = _zero_embedding(dims["lp"])
        if "bow_embedding" in row:
            row["bow_embedding"] = _zero_embedding(dims["bow"])
        if "sentiment_embedding" in row:
            row["sentiment_embedding"] = _zero_embedding(dims["sentiment"])
        for column, key_name in (("hd_dim", "hd"), ("lp_dim", "lp"), ("bow_dim", "bow"), ("sentiment_dim", "sentiment")):
            if column in row:
                row[column] = dims[key_name]
        quiet_rows.append(row)
        audit_rows.append(audit_row)

    if not quiet_rows:
        raise ValueError(
            "No no-news quiet rows were selected. "
            "Check surface_all_json coverage, quiet_grid_minutes, or quiet_buffer_minutes."
        )

    output = Path(output_workbook).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([event_rows, pd.DataFrame(quiet_rows)], ignore_index=True, sort=False)
    audit = pd.DataFrame(audit_rows)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name=GAN_SHEET, index=False)
        audit.to_excel(writer, sheet_name="news_quiet_audit", index=False)
    return output


def build_news_quiet_workbook_from_window(
    *,
    source_merged_vol: str | Path,
    window_surface_json: str | Path,
    output_workbook: str | Path,
    horizon_minutes: int = 5,
    quiet_grid_minutes: int = 5,
    quiet_buffer_minutes: int = 60,
    quiet_max_samples: int = 3711,
    quiet_sample_seed: int = 20260625,
    sheet_name: str = GAN_SHEET,
) -> Path:
    event_rows = _load_event_rows(
        source_merged_vol,
        sheet_name=sheet_name,
        quiet_buffer_minutes=int(quiet_buffer_minutes),
        quiet_grid_minutes=int(quiet_grid_minutes),
    )
    strike_grid = parse_serialized_list(event_rows.iloc[0]["strike_grid"])
    maturity_days_grid = parse_serialized_list(event_rows.iloc[0]["maturity_days_grid"])
    surface_shape_text = serialize_list([len(maturity_days_grid), len(strike_grid)])
    dims = {
        "hd": _embedding_dim(event_rows, "hd_embedding", "hd_dim"),
        "lp": _embedding_dim(event_rows, "lp_embedding", "lp_dim"),
        "bow": _embedding_dim(event_rows, "bow_embedding", "bow_dim"),
        "sentiment": _embedding_dim(event_rows, "sentiment_embedding", "sentiment_dim"),
    }

    surface_map = _load_window_surface_json(window_surface_json)
    valid_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    audit_rows: list[dict[str, Any]] = []

    for key in sorted(surface_map):
        bundle = surface_map[key]
        backward = bundle.get("backward")
        forward = bundle.get("forward")
        current_time = str(backward.get("snapshot_time_utc", key)) if isinstance(backward, Mapping) else key
        target_time = (
            str(forward.get("snapshot_time_utc", _timestamp_key(parse_timestamp_utc(key, field_name="target") + pd.Timedelta(minutes=int(horizon_minutes)))))
            if isinstance(forward, Mapping)
            else ""
        )
        audit_row = {
            "candidate_timestamp_utc": key,
            "current_snapshot_time_utc": current_time,
            "target_snapshot_time_utc": target_time,
            "selected": 0,
            "reason": "selected",
        }
        if not isinstance(backward, Mapping) or not isinstance(forward, Mapping):
            audit_row["reason"] = "missing_backward_or_forward"
            audit_rows.append(audit_row)
            continue
        if backward.get("surface_params") is None or forward.get("surface_params") is None:
            audit_row["reason"] = "missing_surface_params"
            audit_rows.append(audit_row)
            continue
        try:
            current_flat = _surface_flat_from_entry(
                entry=backward,
                timestamp_utc=current_time,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
            )
            target_flat = _surface_flat_from_entry(
                entry=forward,
                timestamp_utc=target_time,
                strike_grid=strike_grid,
                maturity_days_grid=maturity_days_grid,
            )
        except Exception as exc:
            audit_row["reason"] = f"surface_reconstruction_failed: {exc}"
            audit_rows.append(audit_row)
            continue

        row = {column: "" for column in event_rows.columns}
        row.update(
            {
                "sample_id": f"quiet_{current_time.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}",
                "news_timestamp_utc": current_time,
                "current_snapshot_time_utc": current_time,
                "target_snapshot_time_utc": target_time,
                "surface_model": str(backward.get("surface_model", "")),
                "strike_grid": serialize_list(strike_grid),
                "maturity_days_grid": serialize_list(maturity_days_grid),
                "surface_shape": surface_shape_text,
                "current_surface_flat": serialize_list(current_flat),
                "target_surface_flat": serialize_list(target_flat),
                "current_weighted_iv_rmse": "",
                "target_weighted_iv_rmse": "",
                "pair_quality_label": "quiet_no_news",
                "training_candidate_flag": 1,
                "event_group": QUIET_GROUP,
                "has_news": 0,
                "news_cluster_id": "",
                "quiet_buffer_minutes": int(quiet_buffer_minutes),
                "quiet_grid_minutes": int(quiet_grid_minutes),
            }
        )
        for column, value in {
            "current_has_surface": 1,
            "target_has_surface": 1,
            "current_has_svi": 1 if str(backward.get("surface_model", "")) == "svi" else "",
            "target_has_svi": 1 if str(forward.get("surface_model", "")) == "svi" else "",
            "current_surface_slice_count": _surface_slice_count(backward),
            "target_surface_slice_count": _surface_slice_count(forward),
            "current_svi_slice_count": _surface_slice_count(backward),
            "target_svi_slice_count": _surface_slice_count(forward),
        }.items():
            if column in row:
                row[column] = value
        if "hd_embedding" in row:
            row["hd_embedding"] = _zero_embedding(dims["hd"])
        if "lp_embedding" in row:
            row["lp_embedding"] = _zero_embedding(dims["lp"])
        if "bow_embedding" in row:
            row["bow_embedding"] = _zero_embedding(dims["bow"])
        if "sentiment_embedding" in row:
            row["sentiment_embedding"] = _zero_embedding(dims["sentiment"])
        for column, key_name in (("hd_dim", "hd"), ("lp_dim", "lp"), ("bow_dim", "bow"), ("sentiment_dim", "sentiment")):
            if column in row:
                row[column] = dims[key_name]
        valid_records.append((audit_row, row))

    requested = int(quiet_max_samples)
    if requested <= 0:
        selected_indices = set(range(len(valid_records)))
    elif len(valid_records) < requested:
        raise ValueError(
            f"Only {len(valid_records)} valid quiet rows were reconstructed from {window_surface_json}; "
            f"requested quiet_max_samples={requested}. Increase QUIET_CANDIDATE_COUNT or relax filters."
        )
    elif len(valid_records) > requested:
        rng = np.random.default_rng(int(quiet_sample_seed))
        selected_indices = {int(idx) for idx in rng.choice(len(valid_records), size=requested, replace=False).tolist()}
    else:
        selected_indices = set(range(len(valid_records)))

    quiet_rows: list[dict[str, Any]] = []
    for idx, (audit_row, row) in enumerate(valid_records):
        if idx in selected_indices:
            audit_row["selected"] = 1
            audit_row["reason"] = "selected_for_workbook"
            quiet_rows.append(row)
        else:
            audit_row["selected"] = 0
            audit_row["reason"] = "valid_not_sampled"
        audit_rows.append(audit_row)

    if not quiet_rows:
        raise ValueError("No no-news quiet rows were selected from window surface JSON.")

    output = Path(output_workbook).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([event_rows, pd.DataFrame(quiet_rows)], ignore_index=True, sort=False)
    audit = pd.DataFrame(audit_rows)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name=GAN_SHEET, index=False)
        audit.to_excel(writer, sheet_name="news_quiet_audit", index=False)
    return output


def _require_news_quiet_groups(frame: pd.DataFrame, *, source: str) -> None:
    if "event_group" not in frame.columns:
        raise ValueError(f"{source} must contain an event_group column with news_event/quiet labels.")
    groups = set(frame["event_group"].dropna().astype(str))
    missing = NEWS_QUIET_GROUPS - groups
    if missing:
        raise ValueError(f"{source} is missing required event_group values: {sorted(missing)}")


def news_quiet_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "surface_jump_mae",
        "surface_jump_rmse",
        "surface_jump_max_abs",
        "atm_short_abs_jump",
        "atm_short_signed_jump",
    ]
    rows: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby("event_group", dropna=False):
        row: dict[str, Any] = {
            "event_group": str(group),
            "sample_count": int(len(group_frame)),
            "has_news_mean": float(pd.to_numeric(group_frame.get("has_news", 0), errors="coerce").mean()),
        }
        for metric in metric_columns:
            if metric in group_frame.columns:
                values = pd.to_numeric(group_frame[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
                row[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
        rows.append(row)
    order = {NEWS_EVENT_GROUP: 0, QUIET_GROUP: 1}
    return pd.DataFrame(rows).sort_values("event_group", key=lambda series: series.map(order).fillna(99))


def _welch_test(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.size < 2 or right.size < 2:
        return {"t_stat": float("nan"), "welch_df": float("nan"), "p_two_sided": float("nan"), "p_news_event_greater": float("nan")}
    from scipy import stats

    result = stats.ttest_ind(left, right, equal_var=False, nan_policy="omit")
    p_greater = float(stats.ttest_ind(left, right, equal_var=False, alternative="greater", nan_policy="omit").pvalue)
    return {
        "t_stat": float(result.statistic),
        "welch_df": float("nan"),
        "p_two_sided": float(result.pvalue),
        "p_news_event_greater": p_greater,
    }


def news_quiet_tests(frame: pd.DataFrame, *, metrics: Sequence[str] = WORKBOOK_EVENT_TEST_METRICS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    news_frame = frame[frame["event_group"] == NEWS_EVENT_GROUP]
    quiet_frame = frame[frame["event_group"] == QUIET_GROUP]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for metric in metrics:
        news = pd.to_numeric(news_frame[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
        quiet = pd.to_numeric(quiet_frame[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "metric": metric,
            "difference": "news_event_minus_quiet",
            "interpretation_positive": "news-event samples have larger IVS jump",
            "news_event_n": int(news.size),
            "quiet_n": int(quiet.size),
            "news_event_mean": float(np.mean(news)) if news.size else float("nan"),
            "quiet_mean": float(np.mean(quiet)) if quiet.size else float("nan"),
        }
        if news.size >= 2 and quiet.size >= 2:
            diff = float(np.mean(news) - np.mean(quiet))
            boot = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)
            for idx in range(BOOTSTRAP_ITERATIONS):
                boot[idx] = float(
                    np.mean(news[rng.integers(0, news.size, size=news.size)])
                    - np.mean(quiet[rng.integers(0, quiet.size, size=quiet.size)])
                )
            low, high = np.percentile(boot, [2.5, 97.5])
            row.update(
                {
                    "news_event_minus_quiet_mean": diff,
                    **_welch_test(news, quiet),
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                }
            )
        else:
            row.update(
                {
                    "news_event_minus_quiet_mean": float("nan"),
                    "t_stat": float("nan"),
                    "welch_df": float("nan"),
                    "p_two_sided": float("nan"),
                    "p_news_event_greater": float("nan"),
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_ci95_low": float("nan"),
                    "bootstrap_ci95_high": float("nan"),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_news_quiet_workbook(
    *,
    workbook_path: str | Path,
    output_dir: str | Path,
    sheet_name: str = GAN_SHEET,
    split: str = "all",
    train_ratio: float = 0.8,
) -> Path:
    path = Path(workbook_path).expanduser()
    frame = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    metrics = build_workbook_sample_metrics(frame, split=split, train_ratio=train_ratio)
    _require_news_quiet_groups(metrics, source=str(path))
    metrics["is_announcement_window"] = metrics["event_group"].eq(NEWS_EVENT_GROUP)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics, output / "rq3_news_quiet_labeled_samples.csv")
    _write_csv(news_quiet_group_summary(metrics), output / "rq3_news_quiet_group_summary.csv")
    _write_csv(news_quiet_tests(metrics), output / "rq3_news_quiet_tests.csv")
    _write_csv(quality_audit(metrics), output / "rq3_news_quiet_quality_audit.csv")
    _write_json(
        {
            "mode": "news_quiet_workbook",
            "workbook_path": str(path),
            "output_dir": str(output),
            "split": split,
            "train_ratio": float(train_ratio),
            "sample_count": int(len(metrics)),
            "news_event_count": int(metrics["event_group"].eq(NEWS_EVENT_GROUP).sum()),
            "quiet_count": int(metrics["event_group"].eq(QUIET_GROUP).sum()),
        },
        output / "rq3_news_quiet_manifest.json",
    )
    return output


def _parse_result_specs(result_specs: Sequence[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in result_specs:
        if "=" not in str(spec):
            raise ValueError(f"--result must use label=path format, got: {spec!r}")
        label, raw_path = str(spec).split("=", 1)
        path = Path(raw_path).expanduser()
        if not label.strip() or not path.exists():
            raise ValueError(f"Invalid --result item: {spec!r}")
        parsed.append((label.strip(), path))
    if not parsed:
        raise ValueError("At least one --result label=path item is required.")
    return parsed


def _load_result_summaries(result_specs: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path in _parse_result_specs(result_specs):
        frame = pd.read_csv(path, dtype=object)
        if "event_group" not in frame.columns:
            raise ValueError(f"Result summary {path} is missing event_group metadata.")
        frame.insert(0, "model_label", label)
        frame.insert(1, "result_summary_path", str(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _infer_text_label(labels: Sequence[str], text_label: str | None) -> str:
    if text_label:
        if text_label not in labels:
            raise ValueError(f"text_label={text_label!r} not found in result labels: {list(labels)}")
        return text_label
    candidates = [label for label in labels if "text" in label.lower() and "no_text" not in label.lower()]
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer the LP text model label. Pass --text-label explicitly. "
            f"Candidates: {candidates}; labels: {list(labels)}"
        )
    return candidates[0]


def news_quiet_result_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_label, event_group), group in frame.groupby(["model_label", "event_group"], dropna=False):
        row: dict[str, Any] = {
            "model_label": str(model_label),
            "event_group": str(event_group),
            "sample_count": int(len(group)),
        }
        for column in METRIC_COLUMNS + [metric_column for _, metric_column in MAIN_RESULT_METRICS]:
            if column in group.columns:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                if not values.empty:
                    row[f"{column}_mean"] = float(values.mean())
                    row[f"{column}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_label", "event_group"]).reset_index(drop=True)


def _paired_text_vs_baselines(frame: pd.DataFrame, *, text_label: str) -> pd.DataFrame:
    labels = [str(value) for value in frame["model_label"].drop_duplicates().tolist()]
    keys = ["sample_id", "global_index", "news_timestamp_utc", "event_group"]
    text = frame[frame["model_label"] == text_label].copy()
    rows: list[dict[str, Any]] = []
    from scipy import stats

    for baseline in [label for label in labels if label != text_label]:
        other = frame[frame["model_label"] == baseline].copy()
        merged = text[keys + [column for _, column in MAIN_RESULT_METRICS]].merge(
            other[keys + [column for _, column in MAIN_RESULT_METRICS]],
            on=keys,
            suffixes=("_text", "_baseline"),
            how="inner",
        )
        for group in [NEWS_EVENT_GROUP, QUIET_GROUP, "all"]:
            group_frame = merged if group == "all" else merged[merged["event_group"] == group]
            for metric_name, metric_column in MAIN_RESULT_METRICS:
                diff = (
                    pd.to_numeric(group_frame[f"{metric_column}_baseline"], errors="coerce")
                    - pd.to_numeric(group_frame[f"{metric_column}_text"], errors="coerce")
                ).dropna().to_numpy(dtype=np.float64)
                row = {
                    "text_model": text_label,
                    "baseline_model": baseline,
                    "event_group": group,
                    "metric": metric_name,
                    "difference": "baseline_minus_text",
                    "interpretation_positive": "LP text lower MAE / better",
                    "n": int(diff.size),
                    "baseline_minus_text_mean": float(np.mean(diff)) if diff.size else float("nan"),
                    "text_better_rate": float(np.mean(diff > 0)) if diff.size else float("nan"),
                }
                if diff.size >= 2:
                    result = stats.ttest_1samp(diff, popmean=0.0, nan_policy="omit")
                    row.update(
                        {
                            "t_stat": float(result.statistic),
                            "df": int(diff.size - 1),
                            "p_two_sided": float(result.pvalue),
                            "p_text_better_one_sided": float(stats.t.sf(float(result.statistic), diff.size - 1)),
                        }
                    )
                else:
                    row.update(
                        {
                            "t_stat": float("nan"),
                            "df": float("nan"),
                            "p_two_sided": float("nan"),
                            "p_text_better_one_sided": float("nan"),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _text_advantage_did(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = pairwise[pairwise["event_group"].isin([NEWS_EVENT_GROUP, QUIET_GROUP])]
    for (baseline, metric), group in grouped.groupby(["baseline_model", "metric"], dropna=False):
        lookup = {str(row["event_group"]): row for _, row in group.iterrows()}
        if NEWS_EVENT_GROUP not in lookup or QUIET_GROUP not in lookup:
            continue
        news_row = lookup[NEWS_EVENT_GROUP]
        quiet_row = lookup[QUIET_GROUP]
        did = float(news_row["baseline_minus_text_mean"]) - float(quiet_row["baseline_minus_text_mean"])
        rows.append(
            {
                "baseline_model": str(baseline),
                "metric": str(metric),
                "difference": "text_advantage_news_event_minus_quiet",
                "interpretation_positive": "LP text advantage is larger for news-event samples",
                "news_event_text_advantage": float(news_row["baseline_minus_text_mean"]),
                "quiet_text_advantage": float(quiet_row["baseline_minus_text_mean"]),
                "did_mean": did,
                "news_event_n": int(news_row["n"]),
                "quiet_n": int(quiet_row["n"]),
            }
        )
    return pd.DataFrame(rows)


def analyze_news_quiet_results(
    *,
    result_specs: Sequence[str],
    output_dir: str | Path,
    text_label: str | None = None,
) -> Path:
    results = _load_result_summaries(result_specs)
    _require_news_quiet_groups(results, source="result summaries")
    labels = [str(value) for value in results["model_label"].drop_duplicates().tolist()]
    resolved_text_label = _infer_text_label(labels, text_label)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    group_metrics = news_quiet_result_group_metrics(results)
    pairwise = _paired_text_vs_baselines(results, text_label=resolved_text_label)
    did = _text_advantage_did(pairwise)
    _write_csv(results, output / "rq3_news_quiet_result_labeled_samples.csv")
    _write_csv(group_metrics, output / "rq3_news_quiet_result_group_metrics.csv")
    _write_csv(pairwise, output / "rq3_news_quiet_text_vs_baselines.csv")
    _write_csv(did, output / "rq3_news_quiet_text_advantage_did.csv")
    _write_json(
        {
            "mode": "news_quiet_result",
            "result_specs": list(result_specs),
            "output_dir": str(output),
            "text_label": resolved_text_label,
            "sample_count": int(len(results)),
            "news_event_count": int(results["event_group"].eq(NEWS_EVENT_GROUP).sum()),
            "quiet_count": int(results["event_group"].eq(QUIET_GROUP).sum()),
        },
        output / "rq3_news_quiet_result_manifest.json",
    )
    return output
