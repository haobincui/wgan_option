#!/usr/bin/env python3
"""Validation, status, manifest, and packaging for corrected RQ1-RQ3 runs."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def write_status(args: argparse.Namespace) -> int:
    root = _resolve(args.pipeline_root)
    current = _read_json(root / "registry/pipeline_status.json")
    started = current.get("started_at_utc") or _now()
    payload = {
        **current,
        "pipeline_root": str(root),
        "status": str(args.status),
        "phase": str(args.phase),
        "message": str(args.message),
        "started_at_utc": started,
        "updated_at_utc": _now(),
        "finished_at_utc": (
            _now() if args.status in {"completed", "failed"} else ""
        ),
    }
    _write_json(root / "registry/pipeline_status.json", payload)
    return 0


def build_manifest(args: argparse.Namespace) -> int:
    root = _resolve(args.pipeline_root)
    output = root / "manifest.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(output)
    return 0


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(source)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != source_hash:
            raise FileExistsError(
                f"Snapshot destination differs from source: {destination}"
            )
        return "existing_verified"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def snapshot_inputs(args: argparse.Namespace) -> int:
    root = _resolve(args.pipeline_root)
    snapshot_root = root / "inputs/source_snapshot"
    rows: list[dict[str, Any]] = []
    for specification in args.input:
        if "=" not in specification:
            raise ValueError(
                "--input must use CATEGORY=PATH syntax; "
                f"received {specification!r}."
            )
        category, source_value = specification.split("=", 1)
        category = category.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", category):
            raise ValueError(f"Invalid snapshot category: {category!r}")
        source_text = source_value.strip()
        has_magic = glob.has_magic(source_text)
        if has_magic:
            pattern = (
                source_text
                if Path(source_text).is_absolute()
                else str(ROOT / source_text)
            )
            source_files = sorted(
                Path(value)
                for value in glob.glob(pattern, recursive=True)
                if Path(value).is_file()
            )
            if not source_files:
                raise FileNotFoundError(
                    f"Snapshot glob matched no files: {source_text}"
                )
            source_root = Path(
                os.path.commonpath(
                    [str(path.parent) for path in source_files]
                )
            )
        else:
            source = _resolve(source_text)
            if not source.exists():
                raise FileNotFoundError(source)
            source_files = (
                sorted(path for path in source.rglob("*") if path.is_file())
                if source.is_dir()
                else [source]
            )
            source_root = source if source.is_dir() else source.parent
        for source_file in source_files:
            relative = source_file.relative_to(source_root)
            destination = snapshot_root / category / relative
            method = _link_or_copy(source_file, destination)
            rows.append(
                {
                    "category": category,
                    "source_path": str(source_file.resolve()),
                    "snapshot_relative_path": str(
                        destination.relative_to(root)
                    ),
                    "size_bytes": int(source_file.stat().st_size),
                    "sha256": _sha256(source_file),
                    "import_method": method,
                }
            )
    if not rows:
        raise ValueError("No source files were supplied for snapshotting.")
    output = root / "inputs/source_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(output)
    return 0


def _count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _line in handle) - 1, 0)


def _support_hashes(root: Path) -> dict[str, str]:
    return {
        path.parent.name: _sha256(path)
        for path in sorted(
            (root / "inputs/folds").glob("*/raw_surface_support.json")
        )
    }


def validate_final(args: argparse.Namespace) -> int:
    pipeline_root = _resolve(args.pipeline_root)
    dataset = _resolve(args.dataset_dir)
    rq1 = _resolve(args.rq1_experiment)
    rq2 = _resolve(args.rq2_experiment)
    rq3 = _resolve(args.rq3_experiment)

    dataset_status = _read_json(dataset / "raw_vol_dataset_validation.json")
    rq1_status = _read_json(rq1 / "validation_summary.json")
    rq2_status = _read_json(rq2 / "validation_summary.json")
    rq3_status = _read_json(rq3 / "validation_summary.json")
    sentiment_audit = _read_json(
        pipeline_root / "inputs/sentiment_audit/sentiment_audit_manifest.json"
    )
    source_manifest_path = pipeline_root / "inputs/source_manifest.csv"
    source_manifest_rows: list[dict[str, str]] = []
    if source_manifest_path.is_file():
        with source_manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            source_manifest_rows = list(csv.DictReader(handle))
    source_categories = {
        row.get("category", "") for row in source_manifest_rows
    }
    raw_source_count = sum(
        row.get("category") == "raw_option_data"
        for row in source_manifest_rows
    )
    raw_source_names = {
        Path(str(row.get("source_path", ""))).name
        for row in source_manifest_rows
        if row.get("category") == "raw_option_data"
    }
    snapshot_hashes_match = bool(source_manifest_rows) and all(
        (
            pipeline_root / str(row.get("snapshot_relative_path", ""))
        ).is_file()
        and _sha256(
            pipeline_root / str(row.get("snapshot_relative_path", ""))
        )
        == str(row.get("sha256", ""))
        for row in source_manifest_rows
    )
    rq1_support = _support_hashes(rq1)
    rq2_support = _support_hashes(rq2)

    rq1_selected = _count_rows(
        rq1 / "checkpoint_selection/selected_checkpoints.csv"
    )
    rq2_selected = _count_rows(
        rq2 / "checkpoint_selection/selected_checkpoints.csv"
    )
    checks = {
        "dataset_validation_ok": dataset_status.get("status") == "ok",
        "dataset_corrected_raw_inputs_ok": bool(
            dataset_status.get("corrected_raw_inputs_ok")
        ),
        "dataset_temporal_audit_ok": bool(
            dataset_status.get("temporal_audit_ok")
        ),
        "dataset_black76_audit_ok": bool(
            dataset_status.get("precalibration_corrected_inputs_ok")
        ),
        "source_manifest_exists": source_manifest_path.is_file(),
        "source_manifest_required_categories": {
            "raw_option_data",
            "news_workbook",
            "text_features",
            "configs",
            "market_references",
            "event_calendar",
        }.issubset(source_categories),
        "source_snapshot_sha256_match": snapshot_hashes_match,
        "raw_option_source_file_count_728": raw_source_count == 728,
        "raw_option_aggregate_excluded": (
            "ty_plus_merged.csv.gz" not in raw_source_names
        ),
        "sentiment_audit_templates_exist": sentiment_audit.get("status")
        in {
            "pending_human_annotation_and_independent_repeat_scoring",
            "complete",
        },
        "rq1_validation_ok": rq1_status.get("status") == "ok",
        "rq2_validation_ok": rq2_status.get("status") == "ok",
        "rq3_validation_ok": rq3_status.get("status") == "ok",
        "rq1_selected_checkpoint_count_84": rq1_selected == 84,
        "rq2_selected_checkpoint_count_24": rq2_selected == 24,
        "rq1_four_support_artifacts": len(rq1_support) == 4,
        "rq2_four_support_artifacts": len(rq2_support) == 4,
        "rq1_rq2_support_sha_match": (
            bool(rq1_support) and rq1_support == rq2_support
        ),
        "rq1_primary_table_exists": (
            rq1
            / "final_tables/development_rq1_primary_controlled_incremental_text.csv"
        ).is_file(),
        "rq1_dm_hac_exists": (
            rq1 / "comparisons/development_dm_hac_tests.csv"
        ).is_file(),
        "rq2_primary_table_exists": (
            rq2
            / "final_tables/development_rq2_primary_lp_vs_baselines.csv"
        ).is_file(),
        "rq2_dm_hac_exists": (
            rq2 / "comparisons/development_rq2_dm_hac_tests.csv"
        ).is_file(),
        "rq3_primary_table_exists": (
            rq3
            / "final_tables/development_rq3_primary_conditional_robustness.csv"
        ).is_file(),
        "rq3_gw_exists": (
            rq3
            / "comparisons/development_rq3_giacomini_white_tests.csv"
        ).is_file(),
        "rq3_overall_dm_hac_exists": (
            rq3 / "comparisons/development_rq3_overall_dm_hac.csv"
        ).is_file(),
    }
    status = "ok" if all(checks.values()) else "failed"
    payload = {
        "created_at_utc": _now(),
        "status": status,
        "scope": "corrected_raw_vol_rq1_rq2_rq3",
        "dataset_dir": str(dataset),
        "rq1_experiment": str(rq1),
        "rq2_experiment": str(rq2),
        "rq3_experiment": str(rq3),
        "checks": checks,
        "rq1_selected_checkpoints": rq1_selected,
        "rq2_selected_checkpoints": rq2_selected,
        "rq1_support_sha256_by_fold": rq1_support,
        "rq2_support_sha256_by_fold": rq2_support,
        "publication_availability_lag_minutes": dataset_status.get(
            "resolved_publication_availability_lag_minutes"
        ),
        "source_manifest_rows": len(source_manifest_rows),
        "raw_option_source_file_count": raw_source_count,
        "sentiment_audit_status": sentiment_audit.get("status", "missing"),
        "news_source_timezone": dataset_status.get(
            "surface_news_source_timezone"
        ),
        "known_scope_limits": [
            "2023 rolling-development evidence; no 2024+ confirmation",
            "raw-vol metrics are evaluated only on fold-train observed support",
            "7-day ATM is not reported for raw-vol",
            "RQ3 is conditional predictive robustness, not causal identification",
            "arbitrage-constrained SSVI is intentionally deferred",
        ],
    }
    output = pipeline_root / "validation_summary.json"
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "ok" else 2


def monitor(args: argparse.Namespace) -> int:
    root = _resolve(args.pipeline_root)
    payload = _read_json(root / "registry/pipeline_status.json")
    pid_path = root / "registry/pipeline.pid"
    pid = (
        pid_path.read_text(encoding="utf-8").strip()
        if pid_path.is_file()
        else ""
    )
    running = False
    if pid.isdigit():
        try:
            os.kill(int(pid), 0)
            running = True
        except OSError:
            pass
    payload["pid"] = int(pid) if pid.isdigit() else None
    payload["process_running"] = running
    for name in ("rq1", "rq2", "rq3"):
        child = root / name
        payload[f"{name}_validation_status"] = _read_json(
            child / "validation_summary.json"
        ).get("status", "missing")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def summarize_lags(args: argparse.Namespace) -> int:
    root = _resolve(args.series_root)
    specifications = {
        "rq1": "rq1/final_tables/development_rq1_primary_controlled_incremental_text.csv",
        "rq2": "rq2/final_tables/development_rq2_primary_lp_vs_baselines.csv",
        "rq3": "rq3/final_tables/development_rq3_primary_conditional_robustness.csv",
    }
    rows: list[dict[str, Any]] = []
    for child in sorted(root.glob("lag_*")):
        if not child.is_dir():
            continue
        lag = child.name.removeprefix("lag_")
        for rq, relative in specifications.items():
            source = child / relative
            if not source.is_file():
                continue
            with source.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append(
                        {
                            "publication_availability_lag_minutes": lag,
                            "rq": rq,
                            "source_path": str(source),
                            **row,
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"No completed lag primary tables under {root}.")
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "publication_availability_lag_minutes",
        "rq",
        "source_path",
    ]
    fields = [*preferred, *[field for field in fields if field not in preferred]]
    output = root / "availability_lag_primary_results.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


def package(args: argparse.Namespace) -> int:
    root = _resolve(args.pipeline_root)
    output = (
        _resolve(args.output)
        if args.output
        else root.with_suffix(".zip")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = (
        zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    )
    with zipfile.ZipFile(
        output,
        "w",
        compression=compression,
        compresslevel=6 if args.compress else None,
        allowZip64=True,
    ) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(Path(root.name) / path.relative_to(root)),
                )
    print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("write-status")
    status.add_argument("--pipeline-root", required=True)
    status.add_argument("--status", required=True)
    status.add_argument("--phase", required=True)
    status.add_argument("--message", default="")
    status.set_defaults(func=write_status)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--pipeline-root", required=True)
    manifest.set_defaults(func=build_manifest)

    snapshot = subparsers.add_parser("snapshot-inputs")
    snapshot.add_argument("--pipeline-root", required=True)
    snapshot.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input specification in CATEGORY=PATH form; repeat as needed.",
    )
    snapshot.set_defaults(func=snapshot_inputs)

    validate = subparsers.add_parser("validate-final")
    validate.add_argument("--pipeline-root", required=True)
    validate.add_argument("--dataset-dir", required=True)
    validate.add_argument("--rq1-experiment", required=True)
    validate.add_argument("--rq2-experiment", required=True)
    validate.add_argument("--rq3-experiment", required=True)
    validate.set_defaults(func=validate_final)

    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--pipeline-root", required=True)
    monitor_parser.set_defaults(func=monitor)

    lags = subparsers.add_parser("summarize-lags")
    lags.add_argument("--series-root", required=True)
    lags.set_defaults(func=summarize_lags)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--pipeline-root", required=True)
    package_parser.add_argument("--output", default="")
    package_parser.add_argument("--compress", action="store_true")
    package_parser.set_defaults(func=package)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
