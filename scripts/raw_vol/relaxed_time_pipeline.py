#!/usr/bin/env python3
"""Build a reusable raw-vol market index and relaxed news-aligned workbook."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

if __package__ in {None, ""}:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import scripts._path_setup  # noqa: F401,E402

from market_data.contract_handler.utils import has_number_before_letter  # noqa: E402
from wgan_option.merge.merge_vol_core import build_vol_workbook_frames  # noqa: E402
from wgan_option.merge_support import (  # noqa: E402
    load_news_base_frame,
    write_workbook,
)
from wgan_option.surface_generation.data_helperd.all import (  # noqa: E402
    PRECALIB_CSV_HEADERS,
)
from wgan_option.surface_generation.market_index import (  # noqa: E402
    ForwardAlignmentPolicy,
    SessionAlignmentPolicy,
    align_news_to_session_pairs,
    align_news_to_valid_pairs,
    alignment_summary,
    candidate_anchors_from_minutes,
    canonical_json,
    connect_market_index,
    fetch_valid_pair_origins,
    find_unmatched_with_eligible_pair,
    insert_candidate_anchors,
    insert_surface_anchor,
    refresh_valid_pairs,
    set_metadata,
    sha256_text,
    to_utc_minute,
    utc_minute_string,
)
from wgan_option.market.treasury_sessions import (  # noqa: E402
    TreasuryGlobexSessionCalendar,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CONFIG = (
    ROOT_DIR
    / "configs/surface_builder/raw/generate_surface-raw-market-index.yaml"
)
DEFAULT_NEWS_WORKBOOK = (
    ROOT_DIR
    / "data/raw/text_embedding/news_with_openai_embeddings_large.xlsx"
)
DEFAULT_RATE_CURVE = (
    ROOT_DIR
    / "data/reference/us_treasury_par_yield_curve_2022_2023.csv"
)
DEFAULT_SESSION_CALENDAR = (
    ROOT_DIR
    / "data/reference/cme_treasury_globex_closures_2022_2023.csv"
)
DEFAULT_STRICT_BASELINE = (
    ROOT_DIR
    / "data/processed/raw-excel/"
    "rq123_corrected_lag0_s2_itm5_20260728-114931/merged_vol.xlsx"
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _absolute_glob(pattern: str) -> str:
    path = Path(pattern).expanduser()
    if path.is_absolute():
        return str(path)
    return str(ROOT_DIR / path)


def _scan_source_files(
    connection: sqlite3.Connection,
    *,
    input_glob: str,
    window_minutes: int,
    chunk_size: int,
) -> dict[str, int]:
    files = [Path(path) for path in sorted(glob.glob(input_glob, recursive=True))]
    if not files:
        raise FileNotFoundError(f"No raw option files match: {input_glob}")

    scanned = 0
    skipped = 0
    observed_minutes_total = 0
    for path in files:
        stat = path.stat()
        existing = connection.execute(
            """
            SELECT size_bytes, mtime_ns, sha256, scan_status
            FROM input_file_state
            WHERE source_path = ?
            """,
            (str(path.resolve()),),
        ).fetchone()
        if existing is not None and str(existing["scan_status"]) == "complete":
            if (
                int(existing["size_bytes"]) != int(stat.st_size)
                or int(existing["mtime_ns"]) != int(stat.st_mtime_ns)
            ):
                raise ValueError(
                    f"Raw input changed after index scan: {path}. "
                    "Use a new index directory."
                )
            skipped += 1
            continue

        file_sha256 = _sha256_file(path)
        if existing is not None and str(existing["sha256"]) not in {
            "",
            file_sha256,
        }:
            raise ValueError(
                f"Raw input SHA changed after partial index scan: {path}"
            )

        observed_option_minutes: set[pd.Timestamp] = set()
        try:
            chunks = pd.read_csv(
                path,
                usecols=["#RIC", "Date-Time", "Price", "Volume"],
                chunksize=max(1, int(chunk_size)),
            )
            for chunk in chunks:
                prices = pd.to_numeric(chunk["Price"], errors="coerce")
                volumes = pd.to_numeric(chunk["Volume"], errors="coerce")
                option_mask = chunk["#RIC"].astype(str).map(
                    has_number_before_letter
                )
                timestamps = pd.to_datetime(
                    chunk["Date-Time"],
                    errors="coerce",
                    utc=True,
                )
                valid = (
                    option_mask
                    & prices.gt(0)
                    & volumes.gt(0)
                    & timestamps.notna()
                )
                observed_option_minutes.update(
                    timestamps.loc[valid].dt.floor("min").tolist()
                )
        except (ValueError, EOFError, OSError, pd.errors.ParserError) as exc:
            connection.execute(
                """
                INSERT OR REPLACE INTO input_file_state(
                    source_path,
                    size_bytes,
                    mtime_ns,
                    sha256,
                    scan_status,
                    observed_minute_count
                ) VALUES (?, ?, ?, ?, 'failed', 0)
                """,
                (
                    str(path.resolve()),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                    file_sha256,
                ),
            )
            connection.commit()
            raise RuntimeError(f"Failed to scan raw option file: {path}") from exc

        anchors = candidate_anchors_from_minutes(
            observed_option_minutes,
            window_minutes=int(window_minutes),
        )
        insert_candidate_anchors(connection, anchors)
        connection.execute(
            """
            INSERT OR REPLACE INTO input_file_state(
                source_path,
                size_bytes,
                mtime_ns,
                sha256,
                scan_status,
                observed_minute_count
            ) VALUES (?, ?, ?, ?, 'complete', ?)
            """,
            (
                str(path.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                file_sha256,
                int(len(observed_option_minutes)),
            ),
        )
        connection.commit()
        scanned += 1
        observed_minutes_total += len(observed_option_minutes)

    return {
        "input_file_count": len(files),
        "newly_scanned_file_count": scanned,
        "resumed_file_count": skipped,
        "newly_observed_minute_count": observed_minutes_total,
        "candidate_anchor_count": int(
            connection.execute(
                "SELECT COUNT(*) FROM candidate_anchor"
            ).fetchone()[0]
        ),
    }


def _batch_id(anchor: str) -> str:
    timestamp = to_utc_minute(anchor)
    week_start = timestamp.normalize() - pd.Timedelta(days=timestamp.weekday())
    return week_start.strftime("%Y%m%d")


def _candidate_batches(
    connection: sqlite3.Connection,
) -> dict[str, list[str]]:
    batches: dict[str, list[str]] = defaultdict(list)
    rows = connection.execute(
        "SELECT anchor_time_utc FROM candidate_anchor ORDER BY anchor_time_utc"
    ).fetchall()
    for row in rows:
        anchor = str(row["anchor_time_utc"])
        batches[_batch_id(anchor)].append(anchor)
    return dict(sorted(batches.items()))


def _build_batch_config(
    *,
    base_config: Path,
    batch_dir: Path,
    target_file: Path,
    input_glob: str,
    workers: int,
    window_minutes: int,
    rate_curve_path: Path,
    min_strikes_per_expiry: int,
    min_expiries_per_minute: int,
    option_filter_mode: str,
    max_itm_moneyness_distance: float,
) -> Path:
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {base_config}")
    surface_builder = payload.setdefault("surface_builder", {})
    if not isinstance(surface_builder, dict):
        raise ValueError(f"Invalid surface_builder mapping: {base_config}")
    generate = surface_builder.setdefault("generate_surface", {})
    if not isinstance(generate, dict):
        raise ValueError(f"Invalid generate_surface mapping: {base_config}")

    surface_builder["option_data_glob"] = input_glob
    generate.update(
        {
            "model": "raw",
            "data_range": "window",
            "run_ts": batch_dir.name,
            "input_glob": input_glob,
            "output_dir": str(batch_dir),
            "output_json": str(batch_dir / "surface-raw-window.json"),
            "log_file": str(batch_dir / "surface-raw-window.log"),
            "resolved_config_path": str(
                batch_dir / "surface-resolved_config.yaml"
            ),
            "save_precalib_csv": True,
            "precalib_csv": str(
                batch_dir / "surface-raw-window-precalib-points.csv"
            ),
            "target_datetimes_file": str(target_file),
            "target_datetimes": [],
            "window_minutes": int(window_minutes),
            "calibration_workers": int(workers),
            "rate_curve_path": str(rate_curve_path),
            "min_strikes_per_expiry": int(min_strikes_per_expiry),
            "min_expiries_per_minute": int(min_expiries_per_minute),
            "option_filter_mode": str(option_filter_mode),
            "max_itm_moneyness_distance": float(
                max_itm_moneyness_distance
            ),
            "window_audit_json": str(
                batch_dir / "window_temporal_audit.json"
            ),
        }
    )
    config_path = batch_dir / "batch_config.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _run_surface_batch(
    *,
    connection: sqlite3.Connection,
    batch_id: str,
    anchors: list[str],
    index_dir: Path,
    base_config: Path,
    input_glob: str,
    workers: int,
    window_minutes: int,
    rate_curve_path: Path,
    min_strikes_per_expiry: int,
    min_expiries_per_minute: int,
    option_filter_mode: str,
    max_itm_moneyness_distance: float,
    keep_batch_artifacts: bool,
) -> int:
    existing = connection.execute(
        "SELECT status, candidate_count FROM calibration_batch WHERE batch_id = ?",
        (batch_id,),
    ).fetchone()
    if existing is not None and str(existing["status"]) == "complete":
        if int(existing["candidate_count"]) != len(anchors):
            raise ValueError(
                f"Candidate count changed for completed batch {batch_id}"
            )
        return 0

    batch_dir = index_dir / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    target_file = batch_dir / "candidate_anchors.txt"
    target_file.write_text("\n".join(anchors) + "\n", encoding="utf-8")
    config_path = _build_batch_config(
        base_config=base_config,
        batch_dir=batch_dir,
        target_file=target_file,
        input_glob=input_glob,
        workers=workers,
        window_minutes=window_minutes,
        rate_curve_path=rate_curve_path,
        min_strikes_per_expiry=min_strikes_per_expiry,
        min_expiries_per_minute=min_expiries_per_minute,
        option_filter_mode=option_filter_mode,
        max_itm_moneyness_distance=max_itm_moneyness_distance,
    )
    output_json = batch_dir / "surface-raw-window.json"
    precalib_csv = batch_dir / "surface-raw-window-precalib-points.csv"
    connection.execute(
        """
        INSERT OR REPLACE INTO calibration_batch(
            batch_id,
            first_anchor_utc,
            last_anchor_utc,
            candidate_count,
            status,
            surface_count,
            output_json,
            precalib_csv,
            message
        ) VALUES (?, ?, ?, ?, 'running', 0, ?, ?, '')
        """,
        (
            batch_id,
            anchors[0],
            anchors[-1],
            len(anchors),
            str(output_json),
            str(precalib_csv),
        ),
    )
    connection.commit()

    command = [
        sys.executable,
        str(ROOT_DIR / "scripts/generate_surface/main.py"),
        "generate_surface",
        "--device",
        "cpu",
        "--config",
        str(config_path),
    ]
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=environment,
            check=True,
        )
    except Exception as exc:
        connection.execute(
            """
            UPDATE calibration_batch
            SET status = 'failed', message = ?
            WHERE batch_id = ?
            """,
            (str(exc), batch_id),
        )
        connection.commit()
        raise

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    successful_anchors: set[str] = set()
    for target, sides in sorted(payload.items()):
        backward = sides.get("backward") if isinstance(sides, Mapping) else None
        if not isinstance(backward, Mapping):
            continue
        params = backward.get("surface_params")
        if not isinstance(params, Mapping):
            continue
        anchor = utc_minute_string(
            backward.get("snapshot_time_utc") or target
        )
        if anchor != utc_minute_string(target):
            raise ValueError(
                f"Backward surface anchor mismatch in batch {batch_id}: "
                f"target={target}, anchor={anchor}"
            )
        insert_surface_anchor(
            connection,
            anchor_time_utc=anchor,
            surface_model=str(backward.get("surface_model", "raw")),
            surface_params=params,
            source_target_utc=target,
            source_direction="backward",
            surface_audit=(
                backward.get("surface_audit")
                if isinstance(backward.get("surface_audit"), Mapping)
                else {}
            ),
        )
        successful_anchors.add(anchor)
    connection.commit()

    insert_payloads: list[tuple[str, str, str]] = []
    if successful_anchors:
        with precalib_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("window_side", "")) != "backward":
                    continue
                target = str(row.get("target_datetime_utc", "")).strip()
                anchor = str(row.get("calibration_datetime_utc", "")).strip()
                if (
                    anchor not in successful_anchors
                    or target != anchor
                ):
                    continue
                row_json = canonical_json(row)
                insert_payloads.append(
                    (anchor, sha256_text(row_json), row_json)
                )
                if len(insert_payloads) >= 5000:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO precalib_point(
                            anchor_time_utc,
                            row_sha256,
                            row_json
                        ) VALUES (?, ?, ?)
                        """,
                        insert_payloads,
                    )
                    insert_payloads.clear()
        if insert_payloads:
            connection.executemany(
                """
                INSERT OR IGNORE INTO precalib_point(
                    anchor_time_utc,
                    row_sha256,
                    row_json
                ) VALUES (?, ?, ?)
                """,
                insert_payloads,
            )

    connection.execute(
        """
        UPDATE calibration_batch
        SET status = 'complete',
            surface_count = ?,
            message = ''
        WHERE batch_id = ?
        """,
        (len(successful_anchors), batch_id),
    )
    connection.commit()

    summary_path = batch_dir / "batch_summary.json"
    _write_json(
        summary_path,
        {
            "batch_id": batch_id,
            "candidate_count": len(anchors),
            "successful_backward_surface_count": len(successful_anchors),
            "completed_at_utc": _now_utc(),
        },
    )
    if not keep_batch_artifacts:
        for path in (
            output_json,
            precalib_csv,
            batch_dir / "window_temporal_audit.json",
        ):
            path.unlink(missing_ok=True)
    return len(successful_anchors)


def _export_index_audits(
    connection: sqlite3.Connection,
    *,
    index_dir: Path,
) -> None:
    pair_rows = connection.execute(
        """
        SELECT origin_time_utc, target_time_utc,
               current_params_sha256, target_params_sha256
        FROM valid_pair
        ORDER BY origin_time_utc
        """
    ).fetchall()
    pd.DataFrame(
        [dict(row) for row in pair_rows],
        columns=[
            "origin_time_utc",
            "target_time_utc",
            "current_params_sha256",
            "target_params_sha256",
        ],
    ).to_csv(index_dir / "valid_5m_pair_index.csv", index=False)

    surface_rows = connection.execute(
        """
        SELECT anchor_time_utc, surface_model, params_sha256,
               source_target_utc, source_direction, audit_json
        FROM surface_anchor
        ORDER BY anchor_time_utc
        """
    ).fetchall()
    surface_audit: list[dict[str, Any]] = []
    for row in surface_rows:
        audit = json.loads(str(row["audit_json"]) or "{}")
        surface_audit.append(
            {
                "anchor_time_utc": row["anchor_time_utc"],
                "surface_model": row["surface_model"],
                "params_sha256": row["params_sha256"],
                "source_target_utc": row["source_target_utc"],
                "source_direction": row["source_direction"],
                "expiry_slice_count": audit.get("expiry_slice_count", ""),
                "selected_option_observations": audit.get(
                    "selected_option_observations",
                    "",
                ),
                "selected_otm_observations": audit.get(
                    "selected_otm_observations",
                    "",
                ),
                "selected_itm_fallback_observations": audit.get(
                    "selected_itm_fallback_observations",
                    "",
                ),
            }
        )
    pd.DataFrame(surface_audit).to_csv(
        index_dir / "market_surface_audit.csv",
        index=False,
    )

    precalib_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "row_count": 0,
            "accepted_row_count": 0,
            "selected_otm_row_count": 0,
            "selected_itm_fallback_row_count": 0,
        }
    )
    rows = connection.execute(
        """
        SELECT anchor_time_utc, row_json
        FROM precalib_point
        ORDER BY anchor_time_utc
        """
    )
    for row in rows:
        anchor = str(row["anchor_time_utc"])
        payload = json.loads(str(row["row_json"]))
        summary = precalib_summary[anchor]
        summary["row_count"] += 1
        accepted = str(payload.get("passes_precalib_filter", "")).lower() in {
            "true",
            "1",
            "1.0",
        }
        if accepted:
            summary["accepted_row_count"] += 1
            if str(payload.get("is_otm", "")).lower() in {
                "true",
                "1",
                "1.0",
            }:
                summary["selected_otm_row_count"] += 1
            else:
                summary["selected_itm_fallback_row_count"] += 1
    pd.DataFrame(
        [
            {"anchor_time_utc": anchor, **summary}
            for anchor, summary in sorted(precalib_summary.items())
        ]
    ).to_csv(index_dir / "precalibration_audit.csv", index=False)

    source_rows = connection.execute(
        """
        SELECT source_path, size_bytes, mtime_ns, sha256,
               scan_status, observed_minute_count
        FROM input_file_state
        ORDER BY source_path
        """
    ).fetchall()
    pd.DataFrame([dict(row) for row in source_rows]).to_csv(
        index_dir / "source_manifest.csv",
        index=False,
    )


def build_market_index(args: argparse.Namespace) -> Path:
    index_dir = Path(args.index_dir).expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    database_path = index_dir / "market_surface_index.sqlite"
    input_glob = _absolute_glob(args.input_glob)
    base_config = Path(args.config).expanduser().resolve()
    rate_curve = Path(args.rate_curve_path).expanduser().resolve()
    if not base_config.is_file():
        raise FileNotFoundError(f"Raw surface config does not exist: {base_config}")
    if not rate_curve.is_file():
        raise FileNotFoundError(f"Rate curve does not exist: {rate_curve}")

    connection = connect_market_index(database_path)
    try:
        immutable_settings = {
            "input_glob": input_glob,
            "config_sha256": _sha256_file(base_config),
            "rate_curve_sha256": _sha256_file(rate_curve),
            "candidate_anchor_source": (
                "positive_price_volume_option_minutes"
            ),
            "window_minutes": int(args.window_minutes),
            "min_strikes_per_expiry": int(args.min_strikes_per_expiry),
            "min_expiries_per_minute": int(args.min_expiries_per_minute),
            "option_filter_mode": str(args.option_filter_mode),
            "max_itm_moneyness_distance": float(
                args.max_itm_moneyness_distance
            ),
        }
        set_metadata(
            connection,
            "immutable_settings",
            immutable_settings,
            require_same=True,
        )
        scan_summary = _scan_source_files(
            connection,
            input_glob=input_glob,
            window_minutes=int(args.window_minutes),
            chunk_size=int(args.chunk_size),
        )
        batches = _candidate_batches(connection)
        completed_batches = 0
        generated_surfaces = 0
        for batch_id, anchors in batches.items():
            if args.max_batches and completed_batches >= int(args.max_batches):
                break
            generated_surfaces += _run_surface_batch(
                connection=connection,
                batch_id=batch_id,
                anchors=anchors,
                index_dir=index_dir,
                base_config=base_config,
                input_glob=input_glob,
                workers=int(args.calibration_workers),
                window_minutes=int(args.window_minutes),
                rate_curve_path=rate_curve,
                min_strikes_per_expiry=int(
                    args.min_strikes_per_expiry
                ),
                min_expiries_per_minute=int(
                    args.min_expiries_per_minute
                ),
                option_filter_mode=str(args.option_filter_mode),
                max_itm_moneyness_distance=float(
                    args.max_itm_moneyness_distance
                ),
                keep_batch_artifacts=bool(args.keep_batch_artifacts),
            )
            completed_batches += 1

        pair_count = refresh_valid_pairs(
            connection,
            horizon_minutes=int(args.window_minutes),
        )
        _export_index_audits(connection, index_dir=index_dir)
        summary = {
            "created_at_utc": _now_utc(),
            "database_path": str(database_path),
            **scan_summary,
            "batch_count": len(batches),
            "batches_visited_this_run": completed_batches,
            "new_surface_count_this_run": generated_surfaces,
            "surface_anchor_count": int(
                connection.execute(
                    "SELECT COUNT(*) FROM surface_anchor"
                ).fetchone()[0]
            ),
            "valid_pair_count": int(pair_count),
            "status": "ok",
        }
        _write_json(index_dir / "index_validation_summary.json", summary)
    finally:
        connection.close()
    return index_dir


def _load_strict_baseline(workbook_path: Path) -> dict[str, Any]:
    if not workbook_path.is_file():
        raise FileNotFoundError(
            f"Strict baseline workbook does not exist: {workbook_path}"
        )
    frame = pd.read_excel(
        workbook_path,
        sheet_name="gan_input_ready",
        usecols=[
            "current_snapshot_time_utc",
            "target_snapshot_time_utc",
        ],
        engine="openpyxl",
    )
    unique_pairs = frame.drop_duplicates(
        ["current_snapshot_time_utc", "target_snapshot_time_utc"]
    )
    return {
        "strict_baseline_workbook": str(workbook_path),
        "strict_matched_article_rows": int(len(frame)),
        "strict_unique_surface_pairs": int(len(unique_pairs)),
    }


def _fetch_surface(
    connection: sqlite3.Connection,
    anchor: str,
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT surface_model, params_json, audit_json
        FROM surface_anchor
        WHERE anchor_time_utc = ?
        """,
        (anchor,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Surface anchor not found in index: {anchor}")
    return {
        "surface_model": str(row["surface_model"]),
        "surface_params": json.loads(str(row["params_json"])),
        "surface_audit": json.loads(str(row["audit_json"]) or "{}"),
    }


def _materialize_dataset_inputs(
    connection: sqlite3.Connection,
    *,
    alignment: pd.DataFrame,
    output_dir: Path,
    source_timezone: str,
    publication_availability_lag_minutes: int,
    window_minutes: int,
    rate_curve_path: Path,
    min_strikes_per_expiry: int,
    min_expiries_per_minute: int,
    option_filter_mode: str,
    max_itm_moneyness_distance: float,
    alignment_policy: Mapping[str, Any],
) -> None:
    matched = alignment[alignment["has_match"].eq(1)]
    origins = sorted(set(matched["effective_origin_utc"].astype(str)))
    target_by_origin = {
        str(row.effective_origin_utc): str(row.target_anchor_utc)
        for row in matched[
            ["effective_origin_utc", "target_anchor_utc"]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    }
    payload: dict[str, Any] = {}
    required_anchors: set[str] = set()
    temporal_audit: dict[str, Any] = {}
    for origin in origins:
        target = target_by_origin[origin]
        current_surface = _fetch_surface(connection, origin)
        target_surface = _fetch_surface(connection, target)
        payload[origin] = {
            "backward": {
                "snapshot_time_utc": origin,
                **current_surface,
            },
            "forward": {
                "snapshot_time_utc": target,
                **target_surface,
            },
        }
        required_anchors.update((origin, target))
        current_start = utc_minute_string(
            to_utc_minute(origin)
            - pd.Timedelta(minutes=int(window_minutes))
        )
        temporal_audit[origin] = {
            "target_datetime_utc": origin,
            "backward": {
                "window_start_utc": current_start,
                "window_end_utc": origin,
                "half_open_interval": "[start,end)",
            },
            "forward": {
                "window_start_utc": origin,
                "window_end_utc": target,
                "half_open_interval": "[start,end)",
            },
            "current_target_trade_overlap_count": 0,
            "post_origin_current_trade_count": 0,
            "source_market_index_temporal_validation": True,
            "status": "ok",
        }

    surface_json_path = output_dir / "surface-raw-excel.json"
    _write_json(surface_json_path, payload)
    _write_json(output_dir / "window_temporal_audit.json", temporal_audit)

    precalib_path = (
        output_dir / "surface-raw-excel-precalib-points.csv"
    )
    with precalib_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRECALIB_CSV_HEADERS)
        writer.writeheader()
        for anchor in sorted(required_anchors):
            rows = connection.execute(
                """
                SELECT row_json
                FROM precalib_point
                WHERE anchor_time_utc = ?
                ORDER BY row_sha256
                """,
                (anchor,),
            ).fetchall()
            if not rows:
                raise ValueError(
                    f"Index surface has no precalibration audit rows: {anchor}"
                )
            for row in rows:
                raw = json.loads(str(row["row_json"]))
                writer.writerow(
                    {column: raw.get(column, "") for column in PRECALIB_CSV_HEADERS}
                )

    resolved = {
        "surface_builder": {
            "job": "generate_surface",
            "generate_surface": {
                "model": "raw",
                "data_range": "excel",
                "output_dir": str(output_dir),
                "output_json": str(surface_json_path),
                "precalib_csv": str(precalib_path),
                "source_timezone": str(source_timezone),
                "publication_availability_lag_minutes": int(
                    publication_availability_lag_minutes
                ),
                "window_minutes": int(window_minutes),
                "min_strikes_per_expiry": int(min_strikes_per_expiry),
                "min_expiries_per_minute": int(min_expiries_per_minute),
                "pricing_model": "black76",
                "rate_curve_path": str(rate_curve_path),
                "max_rate_staleness_days": 7,
                "max_underlying_staleness_seconds": 60,
                "underlying_match_mode": "last_prior_trade",
                "option_filter_mode": str(option_filter_mode),
                "max_itm_moneyness_distance": float(
                    max_itm_moneyness_distance
                ),
                "iv_aggregation_mode": "volume_weighted_median",
                **dict(alignment_policy),
            },
        }
    }
    (output_dir / "surface-resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )


def _write_dataset_manifest(
    output_dir: Path,
    source_paths: Iterable[tuple[str, Path]],
) -> None:
    rows: list[dict[str, Any]] = []
    for category, path in source_paths:
        path = Path(path)
        rows.append(
            {
                "category": category,
                "path": str(path),
                "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
                "sha256": _sha256_file(path) if path.is_file() else "",
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "source_manifest.csv",
        index=False,
    )


def build_relaxed_dataset(args: argparse.Namespace) -> Path:
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    news_workbook = Path(args.news_xlsx).expanduser().resolve()
    strict_workbook = Path(
        args.strict_baseline_workbook
    ).expanduser().resolve()
    rate_curve = Path(args.rate_curve_path).expanduser().resolve()
    database_path = index_dir / "market_surface_index.sqlite"
    if not database_path.is_file():
        raise FileNotFoundError(f"Market index does not exist: {database_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment_mode = str(
        getattr(args, "alignment_mode", "forward_valid_pair")
    )
    session_calendar_path: Path | None = None
    session_calendar_snapshot_path: Path | None = None
    connection = connect_market_index(database_path)
    try:
        valid_origins = fetch_valid_pair_origins(connection)
        if not valid_origins:
            raise ValueError("Market index contains no valid five-minute pairs")
        news = load_news_base_frame(
            news_workbook,
            source_timezone=str(args.source_timezone),
            offset_minutes=int(args.window_minutes),
            publication_availability_lag_minutes=int(
                args.publication_availability_lag_minutes
            ),
        )
        if alignment_mode == "exchange_session":
            session_calendar_path = Path(
                getattr(
                    args,
                    "session_calendar_path",
                    DEFAULT_SESSION_CALENDAR,
                )
            ).expanduser().resolve()
            session_calendar = TreasuryGlobexSessionCalendar.from_csv(
                session_calendar_path
            )
            session_policy = SessionAlignmentPolicy(
                origin_tolerance_minutes=int(
                    getattr(args, "origin_tolerance_minutes", 5)
                ),
                horizon_minutes=int(args.window_minutes),
                current_window_minutes=int(args.window_minutes),
            )
            alignment = align_news_to_session_pairs(
                news,
                valid_origins,
                session_calendar=session_calendar,
                policy=session_policy,
            )
            alignment_path = (
                output_dir / "news_market_session_alignment.csv"
            )
            session_calendar_snapshot_path = (
                output_dir / "cme_session_calendar_snapshot.csv"
            )
            shutil.copy2(
                session_calendar_path,
                session_calendar_snapshot_path,
            )
            alignment_policy: dict[str, Any] = {
                "news_alignment_mode": "exchange_session",
                "session_calendar_path": str(session_calendar_path),
                "origin_tolerance_minutes": int(
                    session_policy.origin_tolerance_minutes
                ),
                "horizon_minutes": int(session_policy.horizon_minutes),
                "current_window_minutes": int(
                    session_policy.current_window_minutes
                ),
                "require_complete_pair": True,
                "require_origin_not_before_scheduled_origin": True,
                "require_single_continuous_session": True,
                "closed_news_rule": "next_cme_continuous_session_open",
                "open_news_rule": "publication_minute",
                "collision_policy": "pool_at_first_valid_pair",
            }
        elif alignment_mode == "forward_valid_pair":
            policy = ForwardAlignmentPolicy(
                intraday_tolerance_minutes=int(
                    args.intraday_tolerance_minutes
                ),
                max_session_shift_minutes=int(
                    args.max_session_shift_minutes
                ),
                horizon_minutes=int(args.window_minutes),
                include_session_shifted=bool(
                    args.include_session_shifted
                ),
            )
            alignment = align_news_to_valid_pairs(
                news,
                valid_origins,
                policy=policy,
            )
            unmatched_violations = find_unmatched_with_eligible_pair(
                alignment,
                valid_origins,
                max_shift_minutes=int(
                    args.max_session_shift_minutes
                ),
            )
            if unmatched_violations:
                raise AssertionError(
                    "Unmatched rows still have eligible forward pairs: "
                    f"{unmatched_violations[:10]}"
                )
            alignment_path = output_dir / "news_market_alignment.csv"
            alignment_policy = {
                "news_alignment_mode": "forward_valid_pair",
                "intraday_tolerance_minutes": int(
                    args.intraday_tolerance_minutes
                ),
                "max_session_shift_minutes": int(
                    args.max_session_shift_minutes
                ),
                "horizon_minutes": int(args.window_minutes),
                "require_complete_pair": True,
                "require_origin_not_before_news": True,
                "include_session_shifted": bool(
                    args.include_session_shifted
                ),
                "collision_policy": "pool_at_nearest_pair",
            }
        else:
            raise ValueError(
                "Unsupported alignment mode: "
                f"{alignment_mode!r}"
            )

        alignment.to_csv(alignment_path, index=False)
        alignment.loc[alignment["has_match"].ne(1)].to_csv(
            output_dir / "unmatched_news.csv",
            index=False,
        )

        _materialize_dataset_inputs(
            connection,
            alignment=alignment,
            output_dir=output_dir,
            source_timezone=str(args.source_timezone),
            publication_availability_lag_minutes=int(
                args.publication_availability_lag_minutes
            ),
            window_minutes=int(args.window_minutes),
            rate_curve_path=rate_curve,
            min_strikes_per_expiry=int(args.min_strikes_per_expiry),
            min_expiries_per_minute=int(args.min_expiries_per_minute),
            option_filter_mode=str(args.option_filter_mode),
            max_itm_moneyness_distance=float(
                args.max_itm_moneyness_distance
            ),
            alignment_policy=alignment_policy,
        )
    finally:
        connection.close()

    frames = build_vol_workbook_frames(
        output_dir,
        news_xlsx_path=news_workbook,
        source_timezone=str(args.source_timezone),
        publication_availability_lag_minutes=int(
            args.publication_availability_lag_minutes
        ),
        offset_minutes=int(args.window_minutes),
        alignment_csv_path=alignment_path,
    )
    workbook_path = output_dir / "merged_vol.xlsx"
    write_workbook(workbook_path, frames)

    gan = frames["gan_input_ready"]
    gan_article_rows = int(len(gan))
    gan_unique_pairs = int(
        gan[
            ["current_snapshot_time_utc", "target_snapshot_time_utc"]
        ]
        .drop_duplicates()
        .shape[0]
    )
    pair_audit = frames["news_surface_pair_audit"]
    training_candidate = pd.to_numeric(
        pair_audit["training_candidate_flag"],
        errors="coerce",
    ).fillna(0).eq(1)
    nonempty_lp = (
        pair_audit["lp_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    strict_lineage = pair_audit.loc[
        training_candidate & nonempty_lp
    ]
    strict_lineage_article_rows = int(len(strict_lineage))
    strict_lineage_pair_samples = int(
        strict_lineage[
            [
                "current_snapshot_time_utc",
                "target_snapshot_time_utc",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )
    strict = _load_strict_baseline(strict_workbook)
    coverage_increased = bool(
        gan_article_rows > strict["strict_matched_article_rows"]
        and gan_unique_pairs > strict["strict_unique_surface_pairs"]
    )
    coverage = {
        "created_at_utc": _now_utc(),
        "alignment_mode": alignment_mode,
        **alignment_summary(alignment),
        **strict,
        "gan_article_rows": gan_article_rows,
        "gan_unique_surface_pairs": gan_unique_pairs,
        "pair_level_training_samples": gan_unique_pairs,
        "strict_text_lineage_article_rows": (
            strict_lineage_article_rows
        ),
        "strict_text_lineage_pair_samples": (
            strict_lineage_pair_samples
        ),
        # Retain legacy names for existing audit consumers.
        "relaxed_gan_article_rows": gan_article_rows,
        "relaxed_gan_unique_surface_pairs": gan_unique_pairs,
        "article_row_increase": (
            gan_article_rows - strict["strict_matched_article_rows"]
        ),
        "unique_pair_increase": (
            gan_unique_pairs - strict["strict_unique_surface_pairs"]
        ),
        "coverage_increased": coverage_increased,
    }
    coverage_row = {
        key: (
            canonical_json(value)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in coverage.items()
    }
    pd.DataFrame([coverage_row]).to_csv(
        output_dir / "alignment_coverage_summary.csv",
        index=False,
    )
    validation = {
        **coverage,
        "alignment_policy": alignment_policy,
        "status": "ok" if coverage_increased else "failed_no_coverage_gain",
    }
    _write_json(output_dir / "validation_summary.json", validation)
    manifest_sources = [
        ("market_index", database_path),
        ("news_workbook", news_workbook),
        ("strict_baseline", strict_workbook),
        ("rate_curve", rate_curve),
        ("alignment", alignment_path),
        ("merged_workbook", workbook_path),
    ]
    if session_calendar_path is not None:
        manifest_sources.append(
            ("session_calendar_source", session_calendar_path)
        )
    if session_calendar_snapshot_path is not None:
        manifest_sources.append(
            ("session_calendar_snapshot", session_calendar_snapshot_path)
        )
    _write_dataset_manifest(output_dir, manifest_sources)
    if not coverage_increased:
        raise RuntimeError(
            f"{alignment_mode} alignment did not improve both article-row "
            "and unique-pair coverage; training must not start. See "
            "alignment_coverage_summary.csv."
        )
    return output_dir


def _add_common_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", required=True)
    parser.add_argument(
        "--input-glob",
        default="data/raw/option_data/0#TY+/0#TY+_202[23]-*.csv.gz",
    )
    parser.add_argument("--config", default=str(DEFAULT_RAW_CONFIG))
    parser.add_argument("--rate-curve-path", default=str(DEFAULT_RATE_CURVE))
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--min-strikes-per-expiry", type=int, default=2)
    parser.add_argument("--min-expiries-per-minute", type=int, default=2)
    parser.add_argument(
        "--option-filter-mode",
        default="otm_preferred_itm_fallback",
    )
    parser.add_argument(
        "--max-itm-moneyness-distance",
        type=float,
        default=0.05,
    )
    parser.add_argument("--calibration-workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--keep-batch-artifacts", action="store_true")


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--news-xlsx", default=str(DEFAULT_NEWS_WORKBOOK))
    parser.add_argument(
        "--strict-baseline-workbook",
        default=str(DEFAULT_STRICT_BASELINE),
    )
    parser.add_argument("--rate-curve-path", default=str(DEFAULT_RATE_CURVE))
    parser.add_argument("--source-timezone", default="Europe/London")
    parser.add_argument(
        "--publication-availability-lag-minutes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("forward_valid_pair", "exchange_session"),
        default="forward_valid_pair",
    )
    parser.add_argument(
        "--session-calendar-path",
        default=str(DEFAULT_SESSION_CALENDAR),
    )
    parser.add_argument(
        "--origin-tolerance-minutes",
        type=int,
        default=5,
    )
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--intraday-tolerance-minutes", type=int, default=15)
    parser.add_argument("--max-session-shift-minutes", type=int, default=4320)
    parser.add_argument(
        "--include-session-shifted",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-strikes-per-expiry", type=int, default=2)
    parser.add_argument("--min-expiries-per-minute", type=int, default=2)
    parser.add_argument(
        "--option-filter-mode",
        default="otm_preferred_itm_fallback",
    )
    parser.add_argument(
        "--max-itm-moneyness-distance",
        type=float,
        default=0.05,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index = subparsers.add_parser(
        "build-index",
        help="Scan raw files and build/resume the persistent surface index.",
    )
    _add_common_index_arguments(build_index)
    build_index.set_defaults(func=build_market_index)

    build_dataset = subparsers.add_parser(
        "build-dataset",
        help="Align news to indexed pairs and build the relaxed workbook.",
    )
    _add_dataset_arguments(build_dataset)
    build_dataset.set_defaults(func=build_relaxed_dataset)

    run = subparsers.add_parser(
        "run",
        help="Build/resume the index, then materialize a relaxed dataset.",
    )
    _add_common_index_arguments(run)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--news-xlsx", default=str(DEFAULT_NEWS_WORKBOOK))
    run.add_argument(
        "--strict-baseline-workbook",
        default=str(DEFAULT_STRICT_BASELINE),
    )
    run.add_argument("--source-timezone", default="Europe/London")
    run.add_argument(
        "--publication-availability-lag-minutes",
        type=int,
        default=0,
    )
    run.add_argument(
        "--alignment-mode",
        choices=("forward_valid_pair", "exchange_session"),
        default="forward_valid_pair",
    )
    run.add_argument(
        "--session-calendar-path",
        default=str(DEFAULT_SESSION_CALENDAR),
    )
    run.add_argument(
        "--origin-tolerance-minutes",
        type=int,
        default=5,
    )
    run.add_argument("--intraday-tolerance-minutes", type=int, default=15)
    run.add_argument("--max-session-shift-minutes", type=int, default=4320)
    run.add_argument(
        "--include-session-shifted",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    def run_all(args: argparse.Namespace) -> Path:
        build_market_index(args)
        return build_relaxed_dataset(args)

    run.set_defaults(func=run_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
