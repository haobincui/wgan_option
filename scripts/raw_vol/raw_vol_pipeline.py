#!/usr/bin/env python3
"""Utilities for the raw-vol interpolation RQ1/RQ2 workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
        }
    payload = _read_json(path)
    target_count = len(payload) if isinstance(payload, dict) else 0
    direction_count = 0
    with_surface = 0
    model_counts: dict[str, int] = {}
    if isinstance(payload, dict):
        for target_payload in payload.values():
            if not isinstance(target_payload, dict):
                continue
            for direction in ("backward", "forward"):
                side = target_payload.get(direction)
                if not isinstance(side, dict):
                    continue
                direction_count += 1
                model = str(side.get("surface_model", "svi"))
                model_counts[model] = model_counts.get(model, 0) + 1
                if side.get("surface_params") is not None or side.get("svi_params") is not None:
                    with_surface += 1
    return {
        "surface_json_path": str(path),
        "surface_json_exists": True,
        "json_target_count": target_count,
        "json_direction_count": direction_count,
        "json_directions_with_surface_params": with_surface,
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
            summary["usable_pairs"] = int(pd.to_numeric(pair["training_candidate_flag"], errors="coerce").fillna(0).sum())
        else:
            summary["usable_pairs"] = int((pair.get("pair_quality_label") == "usable").sum())
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
        if "surface_model" in gan:
            summary["gan_surface_model_counts"] = {
                str(key): int(value)
                for key, value in gan["surface_model"].value_counts(dropna=False).items()
            }
    else:
        summary["gan_input_rows"] = 0
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
        "otm_only": bool(len(accepted) and is_otm.all()),
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
    workbook_timezones = list(workbook.get("news_source_timezones") or [])
    timezone_ok = (
        resolved_timezone == expected_timezone
        and workbook_timezones == [expected_timezone]
    )
    corrected_config_ok = bool(
        str(generate_settings.get("pricing_model", "")).lower() == "black76"
        and str(generate_settings.get("underlying_match_mode", ""))
        == "last_prior_trade"
        and str(generate_settings.get("option_filter_mode", "")) == "otm_only"
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
    print(f"usable_pairs={usable_pairs}")
    return 0 if validation["status"] == "ok" or args.no_fail else 2


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
