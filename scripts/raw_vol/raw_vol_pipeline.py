#!/usr/bin/env python3
"""Utilities for the raw-vol interpolation RQ1/RQ2 workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _weighted_median(values: list[float], weights: list[float]) -> float:
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
        ordered = sorted(value for value, _ in pairs)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return float((ordered[middle - 1] + ordered[middle]) / 2.0)
    threshold = total_weight / 2.0
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def _discover(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        exact = [candidate for candidate in matches if candidate.name == pattern]
        if len(exact) == 1:
            return exact[0]
    return matches[0]


def _surface_json_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "surface_json_path": "" if path is None else str(path),
            "surface_json_exists": False,
            "json_target_count": 0,
            "json_direction_count": 0,
            "json_directions_with_surface_params": 0,
            "json_targets_with_both_surfaces": 0,
        }
    payload = _read_json(path)
    target_count = len(payload) if isinstance(payload, dict) else 0
    direction_count = 0
    with_surface = 0
    targets_with_both_surfaces = 0
    model_counts: dict[str, int] = {}
    if isinstance(payload, dict):
        for target_payload in payload.values():
            if not isinstance(target_payload, dict):
                continue
            side_successes: list[bool] = []
            for direction in ("backward", "forward"):
                side = target_payload.get(direction)
                if not isinstance(side, dict):
                    side_successes.append(False)
                    continue
                direction_count += 1
                model = str(side.get("surface_model", "svi"))
                model_counts[model] = model_counts.get(model, 0) + 1
                has_surface = (
                    side.get("surface_params") is not None
                    or side.get("svi_params") is not None
                )
                side_successes.append(has_surface)
                if has_surface:
                    with_surface += 1
            if len(side_successes) == 2 and all(side_successes):
                targets_with_both_surfaces += 1
    return {
        "surface_json_path": str(path),
        "surface_json_exists": True,
        "json_target_count": target_count,
        "json_direction_count": direction_count,
        "json_directions_with_surface_params": with_surface,
        "json_targets_with_both_surfaces": targets_with_both_surfaces,
        "json_surface_model_counts": model_counts,
    }


def _workbook_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "workbook_path": str(path),
            "workbook_exists": False,
            "pair_rows": 0,
            "side_rows": 0,
            "gan_input_rows": 0,
            "usable_pairs": 0,
            "unique_surface_pairs": 0,
            "pair_level_training_samples": 0,
        }
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    summary: dict[str, Any] = {
        "workbook_path": str(path),
        "workbook_exists": True,
        "sheets": sheets,
    }
    if "news_surface_pair_audit" in sheets:
        pair = pd.read_excel(path, sheet_name="news_surface_pair_audit", engine="openpyxl")
        summary["pair_rows"] = int(len(pair))
        if "pair_quality_label" in pair:
            summary["pair_quality_counts"] = {
                str(key): int(value)
                for key, value in pair["pair_quality_label"].value_counts(dropna=False).items()
            }
        if "surface_model" in pair:
            summary["pair_surface_model_counts"] = {
                str(key): int(value)
                for key, value in pair["surface_model"].value_counts(dropna=False).items()
            }
        if "source_timezone" in pair:
            summary["news_source_timezones"] = sorted(
                {
                    str(value).strip()
                    for value in pair["source_timezone"].dropna().tolist()
                    if str(value).strip()
                }
            )
        if "timestamp_parse_status" in pair:
            summary["timestamp_parse_status_counts"] = {
                str(key): int(value)
                for key, value in pair["timestamp_parse_status"]
                .value_counts(dropna=False)
                .items()
            }
        if "training_candidate_flag" in pair:
            training_candidate = pd.to_numeric(
                pair["training_candidate_flag"],
                errors="coerce",
            ).fillna(0).eq(1)
            summary["usable_pairs"] = int(training_candidate.sum())
        else:
            training_candidate = pair.get(
                "pair_quality_label"
            ).eq("usable")
            summary["usable_pairs"] = int(training_candidate.sum())
        if "lp_text" in pair:
            nonempty_lp = (
                pair["lp_text"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            )
            strict_lineage = pair.loc[
                training_candidate & nonempty_lp
            ]
            summary["strict_text_lineage_article_rows"] = int(
                len(strict_lineage)
            )
            required_pair_columns = {
                "current_snapshot_time_utc",
                "target_snapshot_time_utc",
            }
            if required_pair_columns.issubset(strict_lineage.columns):
                summary["strict_text_lineage_pair_samples"] = int(
                    strict_lineage[
                        [
                            "current_snapshot_time_utc",
                            "target_snapshot_time_utc",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                )
            else:
                summary["strict_text_lineage_pair_samples"] = 0
    else:
        summary["pair_rows"] = 0
        summary["usable_pairs"] = 0
    if "surface_side_detail" in sheets:
        side = pd.read_excel(path, sheet_name="surface_side_detail", engine="openpyxl")
        summary["side_rows"] = int(len(side))
        if "side_quality_label" in side:
            summary["side_quality_counts"] = {
                str(key): int(value)
                for key, value in side["side_quality_label"].value_counts(dropna=False).items()
            }
    else:
        summary["side_rows"] = 0
    if "gan_input_ready" in sheets:
        gan = pd.read_excel(path, sheet_name="gan_input_ready", engine="openpyxl")
        summary["gan_input_rows"] = int(len(gan))
        required_pair_columns = {
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
        }
        if required_pair_columns.issubset(gan.columns):
            pair_frame = gan[
                ["current_snapshot_time_utc", "target_snapshot_time_utc"]
            ].copy()
            pair_frame["current_snapshot_time_utc"] = pd.to_datetime(
                pair_frame["current_snapshot_time_utc"],
                errors="coerce",
                utc=True,
            )
            pair_frame["target_snapshot_time_utc"] = pd.to_datetime(
                pair_frame["target_snapshot_time_utc"],
                errors="coerce",
                utc=True,
            )
            pair_frame = pair_frame.dropna(
                subset=["current_snapshot_time_utc", "target_snapshot_time_utc"]
            )
            pair_frame["current_quarter"] = (
                pair_frame["current_snapshot_time_utc"]
                .dt.tz_localize(None)
                .dt.to_period("Q")
                .astype(str)
            )
            unique_pairs = pair_frame.drop_duplicates(
                subset=[
                    "current_snapshot_time_utc",
                    "target_snapshot_time_utc",
                ]
            )
            unique_surface_pairs = int(len(unique_pairs))
            summary["unique_surface_pairs"] = unique_surface_pairs
            summary["pair_level_training_samples"] = unique_surface_pairs
            summary["duplicate_article_rows"] = int(
                len(pair_frame) - unique_surface_pairs
            )
            summary["article_rows_by_current_quarter"] = {
                str(key): int(value)
                for key, value in pair_frame["current_quarter"]
                .value_counts()
                .sort_index()
                .items()
            }
            summary["surface_pairs_by_current_quarter"] = {
                str(key): int(value)
                for key, value in unique_pairs["current_quarter"]
                .value_counts()
                .sort_index()
                .items()
            }
        else:
            summary["unique_surface_pairs"] = 0
            summary["pair_level_training_samples"] = 0
            summary["duplicate_article_rows"] = 0
            summary["article_rows_by_current_quarter"] = {}
            summary["surface_pairs_by_current_quarter"] = {}
        if "surface_model" in gan:
            summary["gan_surface_model_counts"] = {
                str(key): int(value)
                for key, value in gan["surface_model"].value_counts(dropna=False).items()
            }
    else:
        summary["gan_input_rows"] = 0
        summary["unique_surface_pairs"] = 0
        summary["pair_level_training_samples"] = 0
        summary["duplicate_article_rows"] = 0
        summary["article_rows_by_current_quarter"] = {}
        summary["surface_pairs_by_current_quarter"] = {}
    return summary


def _resolved_source_timezone(dataset_dir: Path) -> str:
    config_path = dataset_dir / "surface-resolved_config.yaml"
    if not config_path.is_file():
        return ""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return ""
    surface_builder = payload.get("surface_builder") or {}
    generate = (
        surface_builder.get("generate_surface")
        if isinstance(surface_builder, dict)
        else {}
    ) or {}
    return str(generate.get("source_timezone") or "").strip()


def _resolved_generate_settings(dataset_dir: Path) -> dict[str, Any]:
    config_path = dataset_dir / "surface-resolved_config.yaml"
    if not config_path.is_file():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    surface_builder = payload.get("surface_builder") or {}
    generate = (
        surface_builder.get("generate_surface")
        if isinstance(surface_builder, dict)
        else {}
    ) or {}
    return dict(generate) if isinstance(generate, dict) else {}


def _temporal_audit_summary(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "window_temporal_audit.json"
    if not path.is_file():
        return {
            "temporal_audit_path": str(path),
            "temporal_audit_exists": False,
            "temporal_audit_target_count": 0,
            "current_target_trade_overlap_count": -1,
            "post_origin_current_trade_count": -1,
            "temporal_audit_ok": False,
        }
    payload = _read_json(path)
    rows = list(payload.values()) if isinstance(payload, dict) else []
    overlap = sum(
        int(row.get("current_target_trade_overlap_count", 0))
        for row in rows
        if isinstance(row, dict)
    )
    post_origin = sum(
        int(row.get("post_origin_current_trade_count", 0))
        for row in rows
        if isinstance(row, dict)
    )
    statuses = {
        str(row.get("status", ""))
        for row in rows
        if isinstance(row, dict)
    }
    return {
        "temporal_audit_path": str(path),
        "temporal_audit_exists": True,
        "temporal_audit_target_count": len(rows),
        "current_target_trade_overlap_count": int(overlap),
        "post_origin_current_trade_count": int(post_origin),
        "temporal_audit_statuses": sorted(statuses),
        "temporal_audit_ok": bool(
            rows
            and overlap == 0
            and post_origin == 0
            and statuses <= {"ok"}
        ),
    }


def _precalibration_audit_summary(
    dataset_dir: Path,
    *,
    expected_rate_curve_sha256: str,
    option_filter_mode: str = "otm_only",
    max_itm_moneyness_distance: float = 0.05,
) -> dict[str, Any]:
    path = _discover(dataset_dir, "surface-raw-excel-precalib-points.csv")
    if path is None or not path.is_file():
        return {
            "precalibration_audit_path": "",
            "precalibration_audit_exists": False,
            "precalibration_accepted_rows": 0,
            "precalibration_corrected_inputs_ok": False,
        }
    columns = [
        "pricing_model",
        "underlying_match_mode",
        "underlying_staleness_seconds",
        "is_otm",
        "passes_precalib_filter",
        "weight",
        "rate_curve_sha256",
    ]
    fallback_mode = (
        str(option_filter_mode).strip().lower()
        == "otm_preferred_itm_fallback"
    )
    if fallback_mode:
        columns.extend(
            [
                "calibration_datetime_utc",
                "business_days",
                "strike",
                "percent_strike",
                "surface_input_role",
            ]
        )
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    accepted_flag = frame["passes_precalib_filter"].astype(str).str.lower().isin(
        {"true", "1", "1.0"}
    )
    accepted = frame[accepted_flag].copy()
    staleness = pd.to_numeric(
        accepted["underlying_staleness_seconds"],
        errors="coerce",
    )
    weights = pd.to_numeric(accepted["weight"], errors="coerce")
    is_otm = accepted["is_otm"].astype(str).str.lower().isin(
        {"true", "1", "1.0"}
    )
    accepted_itm_count = int((~is_otm).sum())
    selected_itm_within_range = True
    otm_preferred_per_strike = True
    itm_fallback_has_otm_anchor = True
    itm_fallback_strike_cap_ok = True
    surface_input_roles_valid = True
    if fallback_mode:
        percent_strike = pd.to_numeric(
            accepted["percent_strike"],
            errors="coerce",
        )
        selected_itm_within_range = bool(
            not accepted_itm_count
            or (
                percent_strike[~is_otm].notna().all()
                and percent_strike[~is_otm]
                .sub(1.0)
                .abs()
                .le(float(max_itm_moneyness_distance) + 1.0e-12)
                .all()
            )
        )
        roles = accepted["surface_input_role"].astype(str)
        surface_input_roles_valid = bool(
            roles[is_otm].eq("otm").all()
            and roles[~is_otm].eq("itm_fallback").all()
        )
        grouping = accepted[
            [
                "calibration_datetime_utc",
                "business_days",
                "strike",
            ]
        ].copy()
        grouping["business_days"] = pd.to_numeric(
            grouping["business_days"],
            errors="coerce",
        )
        grouping["strike"] = pd.to_numeric(
            grouping["strike"],
            errors="coerce",
        )
        grouping["is_otm"] = is_otm.to_numpy()
        grouping_ok = bool(
            grouping[
                [
                    "calibration_datetime_utc",
                    "business_days",
                    "strike",
                ]
            ]
            .notna()
            .all()
            .all()
        )
        if grouping_ok:
            maturity_keys = [
                "calibration_datetime_utc",
                "business_days",
            ]
            strike_keys = [*maturity_keys, "strike"]
            strike_flags = grouping.drop_duplicates(
                [*strike_keys, "is_otm"]
            )
            otm_preferred_per_strike = bool(
                strike_flags.groupby(strike_keys)["is_otm"]
                .nunique()
                .le(1)
                .all()
            )
            strike_counts = (
                strike_flags.groupby([*maturity_keys, "is_otm"])["strike"]
                .nunique()
                .unstack(fill_value=0)
            )
            otm_counts = strike_counts.get(
                True,
                pd.Series(0, index=strike_counts.index),
            )
            itm_counts = strike_counts.get(
                False,
                pd.Series(0, index=strike_counts.index),
            )
            itm_maturities = itm_counts.gt(0)
            itm_fallback_has_otm_anchor = bool(
                otm_counts[itm_maturities].ge(1).all()
            )
            itm_fallback_strike_cap_ok = bool(
                itm_counts[itm_maturities]
                .le(otm_counts[itm_maturities])
                .all()
            )
        else:
            otm_preferred_per_strike = False
            itm_fallback_has_otm_anchor = False
            itm_fallback_strike_cap_ok = False
    elif str(option_filter_mode).strip().lower() == "otm_only":
        selected_itm_within_range = bool(len(accepted) and is_otm.all())

    option_filter_policy_valid = bool(
        len(accepted)
        and (
            (
                str(option_filter_mode).strip().lower() == "otm_only"
                and is_otm.all()
            )
            or (
                fallback_mode
                and selected_itm_within_range
                and otm_preferred_per_strike
                and itm_fallback_has_otm_anchor
                and itm_fallback_strike_cap_ok
                and surface_input_roles_valid
            )
        )
    )
    checks = {
        "pricing_model_black76": bool(
            len(accepted)
            and accepted["pricing_model"].astype(str).str.lower().eq(
                "black76"
            ).all()
        ),
        "underlying_last_prior_trade": bool(
            len(accepted)
            and accepted["underlying_match_mode"].astype(str).eq(
                "last_prior_trade"
            ).all()
        ),
        "underlying_staleness_within_60s": bool(
            len(accepted)
            and staleness.notna().all()
            and staleness.ge(0.0).all()
            and staleness.le(60.0).all()
        ),
        "option_filter_policy_valid": option_filter_policy_valid,
        "positive_volume_weights": bool(
            len(accepted) and weights.notna().all() and weights.gt(0.0).all()
        ),
        "frozen_rate_curve_sha_present": bool(
            len(accepted)
            and accepted["rate_curve_sha256"].astype(str).str.len().eq(64).all()
        ),
        "frozen_rate_curve_sha_match": bool(
            len(accepted)
            and expected_rate_curve_sha256
            and accepted["rate_curve_sha256"]
            .astype(str)
            .eq(expected_rate_curve_sha256)
            .all()
        ),
    }
    return {
        "precalibration_audit_path": str(path),
        "precalibration_audit_exists": True,
        "precalibration_total_rows": int(len(frame)),
        "precalibration_accepted_rows": int(len(accepted)),
        "precalibration_selected_otm_rows": int(is_otm.sum()),
        "precalibration_selected_itm_fallback_rows": accepted_itm_count,
        "otm_only": bool(len(accepted) and is_otm.all()),
        "selected_itm_within_range": selected_itm_within_range,
        "otm_preferred_per_strike": otm_preferred_per_strike,
        "itm_fallback_has_otm_anchor": itm_fallback_has_otm_anchor,
        "itm_fallback_strike_cap_ok": itm_fallback_strike_cap_ok,
        "surface_input_roles_valid": surface_input_roles_valid,
        **checks,
        "precalibration_corrected_inputs_ok": all(checks.values()),
    }


def validate_dataset(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser()
    workbook_path = Path(args.workbook).expanduser() if args.workbook else dataset_dir / "merged_vol.xlsx"
    surface_json = Path(args.surface_json).expanduser() if args.surface_json else _discover(dataset_dir, "surface-raw-excel.json")
    if surface_json is None:
        surface_json = _discover(dataset_dir, "surface-*.json")

    workbook = _workbook_summary(workbook_path)
    usable_pairs = int(workbook.get("usable_pairs", 0))
    expected_timezone = str(args.source_timezone).strip()
    resolved_timezone = _resolved_source_timezone(dataset_dir)
    generate_settings = _resolved_generate_settings(dataset_dir)
    expected_lag = int(args.publication_availability_lag_minutes)
    resolved_lag = int(
        generate_settings.get("publication_availability_lag_minutes", -1)
    )
    rate_curve_value = str(generate_settings.get("rate_curve_path", "")).strip()
    rate_curve_path = Path(rate_curve_value).expanduser()
    if rate_curve_value and not rate_curve_path.is_absolute():
        rate_curve_path = Path.cwd() / rate_curve_path
    rate_curve_sha256 = (
        _sha256(rate_curve_path)
        if rate_curve_value and rate_curve_path.is_file()
        else ""
    )
    option_filter_mode = str(
        generate_settings.get("option_filter_mode", "")
    ).strip().lower()
    max_itm_moneyness_distance = float(
        generate_settings.get("max_itm_moneyness_distance", 0.05)
    )
    workbook_timezones = list(workbook.get("news_source_timezones") or [])
    timezone_ok = (
        resolved_timezone == expected_timezone
        and workbook_timezones == [expected_timezone]
    )
    corrected_config_ok = bool(
        str(generate_settings.get("pricing_model", "")).lower() == "black76"
        and str(generate_settings.get("underlying_match_mode", ""))
        == "last_prior_trade"
        and option_filter_mode
        in {"otm_only", "otm_preferred_itm_fallback"}
        and (
            option_filter_mode != "otm_preferred_itm_fallback"
            or abs(max_itm_moneyness_distance - 0.05) <= 1.0e-12
        )
        and str(generate_settings.get("iv_aggregation_mode", ""))
        == "volume_weighted_median"
        and int(generate_settings.get("window_minutes", -1))
        == int(args.window_minutes)
        and int(generate_settings.get("min_strikes_per_expiry", -1))
        == int(args.min_strikes_per_expiry)
        and int(generate_settings.get("min_expiries_per_minute", 0)) >= 2
        and int(
            generate_settings.get(
                "max_underlying_staleness_seconds",
                -1,
            )
        )
        == 60
        and bool(rate_curve_sha256)
        and resolved_lag == expected_lag
    )
    temporal = _temporal_audit_summary(dataset_dir)
    precalibration = _precalibration_audit_summary(
        dataset_dir,
        expected_rate_curve_sha256=rate_curve_sha256,
        option_filter_mode=option_filter_mode,
        max_itm_moneyness_distance=max_itm_moneyness_distance,
    )
    usable_ok = usable_pairs >= int(args.min_usable_pairs)
    corrected_inputs_ok = bool(
        corrected_config_ok
        and temporal["temporal_audit_ok"]
        and precalibration["precalibration_corrected_inputs_ok"]
    )
    validation = {
        "created_at_utc": _now_utc(),
        "dataset_dir": str(dataset_dir),
        "surface_model": "raw",
        "sample_policy": "raw_available_first",
        "interpolation_policy": "linear_percent_strike_and_linear_total_variance_by_maturity",
        "window_minutes": args.window_minutes,
        "min_strikes_per_expiry": args.min_strikes_per_expiry,
        "option_filter_mode": option_filter_mode,
        "max_itm_moneyness_distance": max_itm_moneyness_distance,
        "expected_news_source_timezone": expected_timezone,
        "surface_news_source_timezone": resolved_timezone,
        "expected_publication_availability_lag_minutes": expected_lag,
        "resolved_publication_availability_lag_minutes": resolved_lag,
        "resolved_rate_curve_path": str(rate_curve_path),
        "resolved_rate_curve_sha256": rate_curve_sha256,
        "workbook_news_source_timezones": workbook_timezones,
        "timezone_validation_ok": bool(timezone_ok),
        "corrected_generate_config_ok": corrected_config_ok,
        **temporal,
        **precalibration,
        "corrected_raw_inputs_ok": corrected_inputs_ok,
        **_surface_json_summary(surface_json),
        **workbook,
        "training_hard_stop_threshold": int(args.min_usable_pairs),
        "low_power_warning_threshold": int(args.warn_usable_pairs),
        "status": (
            "ok"
            if usable_ok and timezone_ok and corrected_inputs_ok
            else (
                "failed_timezone_mismatch"
                if not timezone_ok
                else (
                    "failed_corrected_input_audit"
                    if not corrected_inputs_ok
                    else "failed_low_usable_pairs"
                )
            )
        ),
        "warning": "low_power" if usable_pairs < int(args.warn_usable_pairs) else "",
    }
    output_json = Path(args.output_json).expanduser() if args.output_json else dataset_dir / "raw_vol_dataset_validation.json"
    _write_json(validation, output_json)

    if args.coverage_csv:
        coverage_path = Path(args.coverage_csv).expanduser()
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        exists = coverage_path.exists()
        with coverage_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "created_at_utc",
                    "dataset_dir",
                    "window_minutes",
                    "min_strikes_per_expiry",
                    "news_source_timezone",
                    "usable_pairs",
                    "gan_input_rows",
                    "json_target_count",
                    "json_directions_with_surface_params",
                    "status",
                    "warning",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "created_at_utc": validation["created_at_utc"],
                    "dataset_dir": validation["dataset_dir"],
                    "window_minutes": validation["window_minutes"],
                    "min_strikes_per_expiry": validation["min_strikes_per_expiry"],
                    "news_source_timezone": validation["surface_news_source_timezone"],
                    "usable_pairs": validation["usable_pairs"],
                    "gan_input_rows": validation["gan_input_rows"],
                    "json_target_count": validation["json_target_count"],
                    "json_directions_with_surface_params": validation["json_directions_with_surface_params"],
                    "status": validation["status"],
                    "warning": validation["warning"],
                }
            )
    print(f"Validation written to {output_json}")
    print(f"usable_article_rows={usable_pairs}")
    print(
        "pair_level_training_samples="
        f"{int(validation.get('pair_level_training_samples', 0))}"
    )
    print(
        "strict_text_lineage_article_rows="
        f"{int(validation.get('strict_text_lineage_article_rows', 0))}"
    )
    print(
        "strict_text_lineage_pair_samples="
        f"{int(validation.get('strict_text_lineage_pair_samples', 0))}"
    )
    return 0 if validation["status"] == "ok" or args.no_fail else 2


def rebuild_from_precalib(args: argparse.Namespace) -> int:
    """Reapply surface eligibility without repeating trade parsing or IV inversion."""

    source_dir = Path(args.source_dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("source_dataset_dir and output_dir must be different.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_surface = source_dir / "surface-raw-excel.json"
    source_precalib = source_dir / "surface-raw-excel-precalib-points.csv"
    source_config = source_dir / "surface-resolved_config.yaml"
    source_temporal = source_dir / "window_temporal_audit.json"
    for required in (
        source_surface,
        source_precalib,
        source_config,
        source_temporal,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required source artifact does not exist: {required}")

    config_payload = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    generate = (
        config_payload.get("surface_builder", {}).get("generate_surface", {})
        if isinstance(config_payload, dict)
        else {}
    )
    if str(generate.get("model", "")).strip().lower() != "raw":
        raise ValueError("rebuild-from-precalib only supports raw surface datasets.")
    if (
        str(generate.get("iv_aggregation_mode", "")).strip().lower()
        != "volume_weighted_median"
    ):
        raise ValueError(
            "Source dataset must use iv_aggregation_mode=volume_weighted_median."
        )

    columns = [
        "target_datetime_utc",
        "window_side",
        "business_days",
        "strike",
        "percent_strike",
        "implied_vol",
        "passes_precalib_filter",
        "weight",
    ]
    frame = pd.read_csv(
        source_precalib,
        usecols=columns,
        low_memory=False,
        float_precision="round_trip",
    )
    accepted = frame[
        frame["passes_precalib_filter"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "1.0"})
    ].copy()
    numeric_columns = [
        "business_days",
        "strike",
        "percent_strike",
        "implied_vol",
        "weight",
    ]
    for column in numeric_columns:
        accepted[column] = pd.to_numeric(accepted[column], errors="coerce")
    accepted = accepted.dropna(
        subset=["target_datetime_utc", "window_side", *numeric_columns]
    )
    accepted = accepted[
        accepted["business_days"].gt(0)
        & accepted["strike"].gt(0)
        & accepted["percent_strike"].gt(0)
        & accepted["implied_vol"].gt(0)
        & accepted["weight"].gt(0)
    ]

    buckets: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    for observation in accepted.itertuples(index=False):
        bucket_key = (
            str(observation.target_datetime_utc),
            str(observation.window_side),
            int(observation.business_days),
            float(observation.strike),
        )
        bucket = buckets.setdefault(
            bucket_key,
            {
                "weight_sum": 0.0,
                "percent_strike_weighted_sum": 0.0,
                "iv_observations": [],
            },
        )
        weight = float(observation.weight)
        bucket["weight_sum"] += weight
        bucket["percent_strike_weighted_sum"] += (
            float(observation.percent_strike) * weight
        )
        bucket["iv_observations"].append(
            (float(observation.implied_vol), weight)
        )

    slices_by_side: dict[
        tuple[str, str],
        dict[int, list[tuple[float, float]]],
    ] = {}
    for bucket_key, bucket in buckets.items():
        target_datetime, window_side, business_days, _strike = bucket_key
        weight_sum = float(bucket["weight_sum"])
        if weight_sum <= 0:
            continue
        percent_strike = (
            float(bucket["percent_strike_weighted_sum"]) / weight_sum
        )
        implied_vol = _weighted_median(
            [value for value, _ in bucket["iv_observations"]],
            [weight for _, weight in bucket["iv_observations"]],
        )
        if (
            not math.isfinite(percent_strike)
            or percent_strike <= 0
            or not math.isfinite(implied_vol)
            or implied_vol <= 0
        ):
            continue
        side_key = (str(target_datetime), str(window_side))
        slices_by_side.setdefault(side_key, {}).setdefault(
            int(business_days),
            [],
        ).append((percent_strike, implied_vol))

    min_strikes = int(args.min_strikes_per_expiry)
    min_expiries = int(args.min_expiries_per_side)
    params_by_side: dict[tuple[str, str], dict[str, Any]] = {}
    for side_key, maturity_slices in slices_by_side.items():
        business_days: list[int] = []
        percent_strikes: list[list[float]] = []
        implied_vols: list[list[float]] = []
        for maturity_day in sorted(maturity_slices):
            points = sorted(maturity_slices[maturity_day], key=lambda item: item[0])
            if len(points) < min_strikes:
                continue
            business_days.append(int(maturity_day))
            percent_strikes.append([float(point[0]) for point in points])
            implied_vols.append([float(point[1]) for point in points])
        if len(business_days) < min_expiries:
            continue
        params_by_side[side_key] = {
            "business_days": business_days,
            "percent_strikes": percent_strikes,
            "implied_vols": implied_vols,
        }

    surface_payload = _read_json(source_surface)
    if not isinstance(surface_payload, dict):
        raise ValueError(f"Expected a target-keyed surface JSON: {source_surface}")
    for target_datetime, target_payload in surface_payload.items():
        if not isinstance(target_payload, dict):
            continue
        for window_side in ("backward", "forward"):
            side_payload = target_payload.get(window_side)
            if not isinstance(side_payload, dict):
                continue
            side_payload["surface_model"] = "raw"
            side_payload.pop("svi_params", None)
            side_payload["surface_params"] = params_by_side.get(
                (str(target_datetime), window_side)
            )

    output_surface = output_dir / "surface-raw-excel.json"
    _write_json(surface_payload, output_surface)
    precalib_method = _link_or_copy(
        source_precalib,
        output_dir / source_precalib.name,
    )
    temporal_method = _link_or_copy(
        source_temporal,
        output_dir / source_temporal.name,
    )

    output_generate = config_payload["surface_builder"]["generate_surface"]
    run_ts = str(args.run_ts).strip() or output_dir.name
    output_generate.update(
        {
            "run_ts": run_ts,
            "output_dir": str(output_dir),
            "output_json": str(output_surface),
            "log_file": str(output_dir / "surface-raw-excel.log"),
            "resolved_config_path": str(output_dir / "surface-resolved_config.yaml"),
            "precalib_csv": str(output_dir / source_precalib.name),
            "window_audit_json": str(output_dir / source_temporal.name),
            "min_strikes_per_expiry": min_strikes,
            "min_expiries_per_minute": min_expiries,
        }
    )
    config_payload["runtime"] = {
        "mode": "rebuild_from_precalib",
        "created_at_utc": _now_utc(),
        "source_dataset_dir": str(source_dir),
        "source_surface_sha256": _sha256(source_surface),
        "source_precalib_sha256": _sha256(source_precalib),
    }
    output_config = output_dir / "surface-resolved_config.yaml"
    output_config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    successful_sides = sum(
        1
        for target_payload in surface_payload.values()
        if isinstance(target_payload, dict)
        for window_side in ("backward", "forward")
        if isinstance(target_payload.get(window_side), dict)
        and target_payload[window_side].get("surface_params") is not None
    )
    both_sides = sum(
        1
        for target_payload in surface_payload.values()
        if isinstance(target_payload, dict)
        and all(
            isinstance(target_payload.get(window_side), dict)
            and target_payload[window_side].get("surface_params") is not None
            for window_side in ("backward", "forward")
        )
    )
    manifest = {
        "created_at_utc": _now_utc(),
        "method": "rebuild_surface_eligibility_from_frozen_precalibration_rows",
        "source_dataset_dir": str(source_dir),
        "output_dataset_dir": str(output_dir),
        "min_strikes_per_expiry": min_strikes,
        "min_expiries_per_side": min_expiries,
        "precalibration_total_rows": int(len(frame)),
        "precalibration_accepted_rows": int(len(accepted)),
        "target_count": int(len(surface_payload)),
        "successful_surface_sides": int(successful_sides),
        "targets_with_both_sides": int(both_sides),
        "artifact_methods": {
            source_precalib.name: precalib_method,
            source_temporal.name: temporal_method,
        },
        "source_artifacts": {
            str(source_surface): _sha256(source_surface),
            str(source_precalib): _sha256(source_precalib),
            str(source_config): _sha256(source_config),
            str(source_temporal): _sha256(source_temporal),
        },
    }
    _write_json(manifest, output_dir / "precalib_rebuild_manifest.json")
    (output_dir / "surface-raw-excel.log").write_text(
        "Derived raw surface JSON from frozen pre-calibration audit rows.\n"
        f"source_dataset_dir={source_dir}\n"
        f"min_strikes_per_expiry={min_strikes}\n"
        f"min_expiries_per_side={min_expiries}\n"
        f"successful_surface_sides={successful_sides}\n"
        f"targets_with_both_sides={both_sides}\n",
        encoding="utf-8",
    )
    print(f"Rebuilt surface dataset: {output_dir}")
    print(f"successful_surface_sides={successful_sides}")
    print(f"targets_with_both_sides={both_sides}")
    return 0


def select_best(args: argparse.Namespace) -> int:
    coverage_csv = Path(args.coverage_csv).expanduser()
    frame = pd.read_csv(coverage_csv)
    if frame.empty:
        raise SystemExit(f"Coverage scan is empty: {coverage_csv}")
    frame["usable_pairs"] = pd.to_numeric(frame["usable_pairs"], errors="coerce").fillna(0).astype(int)
    frame["window_minutes"] = pd.to_numeric(frame["window_minutes"], errors="coerce").fillna(10**9).astype(int)
    frame["min_strikes_per_expiry"] = pd.to_numeric(frame["min_strikes_per_expiry"], errors="coerce").fillna(10**9).astype(int)
    best = frame.sort_values(
        ["usable_pairs", "window_minutes", "min_strikes_per_expiry"],
        ascending=[False, True, True],
    ).iloc[0]
    payload = {
        "created_at_utc": _now_utc(),
        "coverage_csv": str(coverage_csv),
        "selection_rule": "max usable_pairs, tie-break smaller window_minutes then min_strikes_per_expiry",
        "selected_dataset_dir": str(best["dataset_dir"]),
        "selected_window_minutes": int(best["window_minutes"]),
        "selected_min_strikes_per_expiry": int(best["min_strikes_per_expiry"]),
        "selected_usable_pairs": int(best["usable_pairs"]),
    }
    output_json = Path(args.output_json).expanduser()
    _write_json(payload, output_json)
    print(f"Best raw-vol dataset: {payload['selected_dataset_dir']}")
    print(f"Selection written to {output_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate one raw-vol merged workbook.")
    validate.add_argument("--dataset-dir", required=True)
    validate.add_argument("--workbook", default="")
    validate.add_argument("--surface-json", default="")
    validate.add_argument("--output-json", default="")
    validate.add_argument("--coverage-csv", default="")
    validate.add_argument("--window-minutes", type=int, default=0)
    validate.add_argument("--min-strikes-per-expiry", type=int, default=0)
    validate.add_argument("--source-timezone", default="Europe/London")
    validate.add_argument(
        "--publication-availability-lag-minutes",
        type=int,
        default=0,
    )
    validate.add_argument("--min-usable-pairs", type=int, default=100)
    validate.add_argument("--warn-usable-pairs", type=int, default=1000)
    validate.add_argument("--no-fail", action="store_true")
    validate.set_defaults(func=validate_dataset)

    rebuild = subparsers.add_parser(
        "rebuild-from-precalib",
        help="Reapply raw-surface strike/expiry eligibility to frozen IV audit rows.",
    )
    rebuild.add_argument("--source-dataset-dir", required=True)
    rebuild.add_argument("--output-dir", required=True)
    rebuild.add_argument("--run-ts", default="")
    rebuild.add_argument("--min-strikes-per-expiry", type=int, required=True)
    rebuild.add_argument("--min-expiries-per-side", type=int, default=2)
    rebuild.set_defaults(func=rebuild_from_precalib)

    select = subparsers.add_parser("select-best", help="Select the best dataset from a coverage CSV.")
    select.add_argument("--coverage-csv", required=True)
    select.add_argument("--output-json", required=True)
    select.set_defaults(func=select_best)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
