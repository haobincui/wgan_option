"""Evaluate the frozen five-seed Text128+c32 checkpoints on 2023Q4.

This evaluator is intentionally separate from training.  ``prepare`` freezes
and hash-binds the five Q3-selected checkpoints while reading zero Q4 rows.
Only the explicit ``evaluate`` action opens the Q4 gate, materializes the
common 5-minute panel, runs MC64 inference, and writes retrospective results.
Q4 can never change a checkpoint, seed, epoch, or training setting.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import scripts._path_setup  # noqa: F401
from scripts.rq3 import (
    news_first_vol_film_unet_text128_c32_epoch240_5seed as training,
)
from scripts.rq3 import news_first_vol_comparison_analysis as comparison
from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
)


EXPERIMENT_KIND = "film_unet_mask_coords_text128_c32_epoch240_5seed_q4_v1"
CONFIRMATION_LABEL = "retrospective_frozen_exploratory"
DEFAULT_CONFIG = (
    "configs/rq3/news_first_vol_film_unet_text128_c32_epoch240_5seed_q4.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_film_unet_text128_c32_epoch240_5seed_q4_exact_ttm_v2"
)
SEEDS = training.SEEDS
MC_SAMPLES = 64
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260828
EXPECTED_COUNTS = {"rows": 167, "pairs": 143, "sessions": 45}
PANEL_NAME = "q4_common_05m"
EVALUATOR_PANEL_NAMESPACE = "core"
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()
SOURCE_PATHS = (
    ORCHESTRATOR_PATH,
    Path(training.__file__).resolve(),
    Path(comparison.__file__).resolve(),
    REPO_ROOT / "src/wgan_option/utils/inference_helpers.py",
    REPO_ROOT / "src/wgan_option/utils/merged_xlsx_parsing.py",
    REPO_ROOT / "src/wgan_option/utils/weighted_training.py",
    REPO_ROOT / "src/wgan_option/models/common.py",
    REPO_ROOT / "src/wgan_option/models/generator.py",
    REPO_ROOT / "src/wgan_option/models/gan_model.py",
    REPO_ROOT / "src/wgan_option/config.py",
)
EXPECTED_STRIKE_GRID = (
    0.970,
    0.974,
    0.978,
    0.982,
    0.986,
    0.990,
    0.994,
    0.998,
    1.002,
    1.006,
    1.010,
    1.014,
    1.018,
    1.022,
    1.026,
    1.030,
)
EXPECTED_MATURITY_GRID = (1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38)

_sha256_file = training._sha256_file
_payload_sha256 = training._payload_sha256
_write_json = training._write_json
_atomic_write_text = training._atomic_write_text
_utc_now = training._utc_now
_require_mapping = training._require_mapping


class Q4EvaluationError(ValueError):
    """Raised when a frozen Q4 input or output contract drifts."""


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve(strict=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise Q4EvaluationError(f"Self-hash field already exists: {field}")
    result = dict(payload)
    result[field] = _payload_sha256(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    observed = str(payload.get(field, ""))
    unsigned = {key: value for key, value in payload.items() if key != field}
    if not observed or observed != _payload_sha256(unsigned):
        raise Q4EvaluationError(f"{label} self-hash mismatch")


def _registry_path(root: Path) -> Path:
    return root / "registry/q4_evaluation.json"


def _status_path(root: Path) -> Path:
    return root / "control/status.json"


def _allowlist_path(root: Path) -> Path:
    return root / "registry/q4_checkpoint_allowlist.json"


def _allowlist_csv_path(root: Path) -> Path:
    return root / "registry/q4_checkpoint_allowlist.csv"


def _write_status(root: Path, state: str, **updates: Any) -> None:
    previous = _read_json(_status_path(root)) if _status_path(root).is_file() else {}
    payload = {
        **previous,
        "experiment_kind": EXPERIMENT_KIND,
        "state": str(state),
        "updated_at": _utc_now(),
        **updates,
    }
    _write_json(_status_path(root), payload)


def _validate_resolved(resolved: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "confirmation_label": CONFIRMATION_LABEL,
        "checkpoint_role": "best_learned",
        "q4_mc_samples": MC_SAMPLES,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for key, expected in exact.items():
        if resolved.get(key) != expected:
            raise Q4EvaluationError(f"{key} must be {expected!r}")
    if tuple(map(int, resolved.get("seeds", ()))) != SEEDS:
        raise Q4EvaluationError(f"seeds must be {SEEDS}")

    training_root = Path(str(resolved.get("training_root", "")))
    registry = training_root / "registry/jobs.json"
    summary = training_root / "analysis/five_seed_summary.json"
    if not registry.is_file() or not summary.is_file():
        raise FileNotFoundError("Completed five-seed training artifacts are missing")
    if _sha256_file(registry) != str(resolved.get("training_registry_sha256", "")):
        raise Q4EvaluationError("Frozen training registry SHA drifted")
    if _sha256_file(summary) != str(resolved.get("training_summary_sha256", "")):
        raise Q4EvaluationError("Frozen training summary SHA drifted")

    panel = _require_mapping(resolved.get("q4_panel"), "q4_panel")
    panel_exact = {
        "sheet_name": "gan_input_ready",
        "start_utc_inclusive": "2023-10-01T00:00:00Z",
        "end_utc_exclusive": "2024-01-01T00:00:00Z",
        "tolerance_minutes": 5,
        "support_mask_mode": "raw_joint",
        "surface_grid_profile": "exact_ttm_16x16_v1",
        "surface_grid_sha256": training.core.GRID_SHA256,
    }
    for key, expected in panel_exact.items():
        if panel.get(key) != expected:
            raise Q4EvaluationError(f"q4_panel.{key} must be {expected!r}")
    if {
        key: int(value) for key, value in panel.get("expected_counts", {}).items()
    } != EXPECTED_COUNTS:
        raise Q4EvaluationError("Q4 expected counts drifted")
    for key in ("source_workbook", "source_lineage_manifest"):
        if not Path(str(panel.get(key, ""))).is_file():
            raise FileNotFoundError(f"Missing frozen Q4 source: {panel.get(key)}")

    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if str(runtime.get("device")) != "cuda:0":
        raise Q4EvaluationError("runtime.device must be cuda:0")
    if int(runtime.get("sample_batch_size", -1)) != 32:
        raise Q4EvaluationError("runtime.sample_batch_size must be 32")
    if int(runtime.get("draw_batch_size", -1)) != 64:
        raise Q4EvaluationError("runtime.draw_batch_size must be 64")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _require_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    resolved = _require_mapping(
        root.get("film_unet_text128_c32_epoch240_5seed_q4"),
        "film_unet_text128_c32_epoch240_5seed_q4",
    )
    resolved["source_config_path"] = str(source)
    resolved["training_root"] = str(_resolve_repo_path(resolved["training_root"]))
    panel = _require_mapping(resolved.get("q4_panel"), "q4_panel")
    panel["source_workbook"] = str(_resolve_repo_path(panel["source_workbook"]))
    panel["source_lineage_manifest"] = str(
        _resolve_repo_path(panel["source_lineage_manifest"])
    )
    resolved["q4_panel"] = panel
    _validate_resolved(resolved)
    return resolved


def _source_hashes(resolved: Mapping[str, Any]) -> dict[str, str]:
    paths = (*SOURCE_PATHS, Path(str(resolved["source_config_path"])))
    return {
        f"source_sha256::{path.relative_to(REPO_ROOT)}": _sha256_file(path)
        for path in paths
    }


def _artifact(status: Mapping[str, Any], role: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in status.get("artifacts", ())
        if str(row.get("artifact_role")) == role
    ]
    if len(matches) != 1:
        raise Q4EvaluationError(f"Expected one {role} artifact")
    row = matches[0]
    path = Path(str(row.get("path", ""))).resolve(strict=False)
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or _sha256_file(path) != str(row.get("sha256", ""))
    ):
        raise Q4EvaluationError(f"Artifact drift for {role}: {path}")
    return row


def _training_allowlist(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(str(resolved["training_root"]))
    registry_path = root / "registry/jobs.json"
    summary_path = root / "analysis/five_seed_summary.json"
    if _sha256_file(registry_path) != str(resolved["training_registry_sha256"]):
        raise Q4EvaluationError("Training registry changed before allowlist freeze")
    if _sha256_file(summary_path) != str(resolved["training_summary_sha256"]):
        raise Q4EvaluationError("Training summary changed before allowlist freeze")
    training_registry = _read_json(registry_path)
    training_resolved = training.resolve_config(
        str(training_registry["source_config_path"])
    )
    training.prepare(training_resolved, root, resume=True)
    summary = _read_json(summary_path)
    q3_by_seed = {int(row["seed"]): row for row in summary.get("rows", ())}
    jobs = [dict(job) for job in training_registry.get("jobs", ())]
    if len(jobs) != len(SEEDS):
        raise Q4EvaluationError("Training registry must contain exactly five jobs")
    rows: list[dict[str, Any]] = []
    for job in jobs:
        seed = int(job["seed"])
        status = training._read_status(root, str(job["job_id"]))
        training._verify_status_binding(job, status, require_complete=True)
        metadata = _artifact(status, "best_learned")
        generator = _artifact(status, "generator_best_learned")
        discriminator = _artifact(status, "discriminator_best_learned")
        metadata_payload = _read_json(Path(str(metadata["path"])))
        artifacts = _require_mapping(
            metadata_payload.get("artifacts"), "checkpoint artifacts"
        )
        if (
            Path(str(artifacts.get("generator", ""))).resolve()
            != Path(str(generator["path"])).resolve()
        ):
            raise Q4EvaluationError(f"Generator pointer drift for seed={seed}")
        if (
            Path(str(artifacts.get("discriminator", ""))).resolve()
            != Path(str(discriminator["path"])).resolve()
        ):
            raise Q4EvaluationError(f"Discriminator pointer drift for seed={seed}")
        if int(metadata_payload.get("best_epoch", -1)) != int(status["best_epoch"]):
            raise Q4EvaluationError(f"Best epoch drift for seed={seed}")
        if (
            str(metadata_payload.get("generator_conditioning_mode"))
            != "film_unet_mask_coords_v1"
        ):
            raise Q4EvaluationError("Generator architecture drifted")
        if str(metadata_payload.get("critic_conditioning_mode")) != "lp_concat_v1":
            raise Q4EvaluationError("Critic architecture drifted")
        q3 = q3_by_seed.get(seed)
        if q3 is None or not math.isclose(
            float(q3["best_q3_mae"]),
            float(status["best_mae"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise Q4EvaluationError(f"Q3 selection binding drift for seed={seed}")
        rows.append(
            {
                "job_id": str(job["job_id"]),
                "job_spec_sha256": str(job["job_spec_sha256"]),
                "seed": seed,
                "best_epoch": int(status["best_epoch"]),
                "q3_best_mae": float(status["best_mae"]),
                "run_dir": str(Path(str(status["run_dir"])).resolve()),
                "checkpoint_metadata_path": str(Path(str(metadata["path"])).resolve()),
                "checkpoint_metadata_sha256": str(metadata["sha256"]),
                "checkpoint_metadata_size_bytes": int(metadata["size_bytes"]),
                "generator_checkpoint_path": str(
                    Path(str(generator["path"])).resolve()
                ),
                "generator_checkpoint_sha256": str(generator["sha256"]),
                "generator_checkpoint_size_bytes": int(generator["size_bytes"]),
                "discriminator_checkpoint_path": str(
                    Path(str(discriminator["path"])).resolve()
                ),
                "discriminator_checkpoint_sha256": str(discriminator["sha256"]),
                "discriminator_checkpoint_size_bytes": int(discriminator["size_bytes"]),
                "generator_conditioning_mode": str(job["generator_conditioning_mode"]),
                "critic_conditioning_mode": str(job["critic_conditioning_mode"]),
                "capacity_id": str(job["capacity_id"]),
                "wgan_parameters": int(job["wgan_parameters"]),
            }
        )
    rows.sort(key=lambda row: SEEDS.index(int(row["seed"])))
    if tuple(int(row["seed"]) for row in rows) != SEEDS:
        raise Q4EvaluationError("Allowlist seed coverage drifted")
    return rows


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise Q4EvaluationError("Cannot serialize an empty CSV")
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _validate_allowlist(
    root: Path, expected_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    payload = _read_json(_allowlist_path(root))
    _verify_self_hash(payload, "allowlist_payload_sha256", "checkpoint allowlist")
    rows = [dict(row) for row in payload.get("rows", ())]
    if rows != [dict(row) for row in expected_rows]:
        raise Q4EvaluationError("Checkpoint allowlist rows drifted")
    csv_path = _allowlist_csv_path(root)
    if not csv_path.is_file() or _sha256_file(csv_path) != str(
        payload.get("csv_sha256", "")
    ):
        raise Q4EvaluationError("Checkpoint allowlist CSV drifted")
    for row in rows:
        for prefix in (
            "checkpoint_metadata",
            "generator_checkpoint",
            "discriminator_checkpoint",
        ):
            path = Path(str(row[f"{prefix}_path"]))
            if (
                not path.is_file()
                or path.stat().st_size != int(row[f"{prefix}_size_bytes"])
                or _sha256_file(path) != str(row[f"{prefix}_sha256"])
            ):
                raise Q4EvaluationError(f"Frozen checkpoint drift: {path}")
    return rows


def prepare(resolved: Mapping[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    source_hashes = _source_hashes(resolved)
    allowlist_rows = _training_allowlist(resolved)
    registry_path = _registry_path(root)
    if registry_path.is_file():
        if not resume:
            raise FileExistsError(f"Q4 registry exists; pass --resume: {registry_path}")
        registry = _read_json(registry_path)
        if registry.get("experiment_kind") != EXPERIMENT_KIND:
            raise Q4EvaluationError("Q4 experiment kind drifted")
        if registry.get("source_hashes") != source_hashes:
            raise Q4EvaluationError("Q4 source/config code drifted")
        if registry.get("allowlist_rows_sha256") != _payload_sha256(allowlist_rows):
            raise Q4EvaluationError("Q4 checkpoint allowlist drifted")
        _validate_allowlist(root, allowlist_rows)
        if bool(registry.get("q4_evaluated")):
            _validate_completed(root, registry)
        return registry

    root.mkdir(parents=True, exist_ok=True)
    for directory in (
        "analysis",
        "analysis/predictions",
        "control",
        "data_windows/q4",
        "registry",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _atomic_write_text(_allowlist_csv_path(root), _csv_text(allowlist_rows))
    allowlist = _self_hashed(
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "selection_source": "Q3 best_learned checkpoints; Q4 not used for selection",
            "training_root": str(Path(str(resolved["training_root"])).resolve()),
            "training_registry_sha256": str(resolved["training_registry_sha256"]),
            "training_summary_sha256": str(resolved["training_summary_sha256"]),
            "row_count": len(allowlist_rows),
            "rows": allowlist_rows,
            "csv_path": str(_allowlist_csv_path(root).resolve()),
            "csv_sha256": _sha256_file(_allowlist_csv_path(root)),
            "frozen_at_utc": _utc_now(),
        },
        "allowlist_payload_sha256",
    )
    _write_json(_allowlist_path(root), allowlist)
    panel = _require_mapping(resolved["q4_panel"], "q4_panel")
    registry = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "confirmation_label": CONFIRMATION_LABEL,
        "created_at": _utc_now(),
        "source_config_path": str(resolved["source_config_path"]),
        "source_hashes": source_hashes,
        "training_root": str(Path(str(resolved["training_root"])).resolve()),
        "training_registry_sha256": str(resolved["training_registry_sha256"]),
        "training_summary_sha256": str(resolved["training_summary_sha256"]),
        "allowlist_path": str(_allowlist_path(root).resolve()),
        "allowlist_sha256": _sha256_file(_allowlist_path(root)),
        "allowlist_rows_sha256": _payload_sha256(allowlist_rows),
        "checkpoint_count": len(allowlist_rows),
        "q4_source_expected_path": str(panel["source_workbook"]),
        "q4_source_expected_sha256": str(panel["source_workbook_sha256"]),
        "q4_rows_read": 0,
        "q4_gate_open": False,
        "q4_window_materialized": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "state": "q4_locked",
    }
    _write_json(registry_path, registry)
    _write_status(
        root,
        "q4_locked",
        checkpoint_count=len(allowlist_rows),
        q4_rows_read=0,
        q4_gate_open=False,
        q4_window_materialized=False,
        q4_predictions_generated=False,
        q4_evaluated=False,
    )
    return registry


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].astype(str).nunique()),
        "sessions": int(frame["session_id"].astype(str).nunique()),
    }


def _universe_sha(frame: pd.DataFrame, *, include_sample: bool) -> str:
    columns = ["session_id", "pair_id"]
    if include_sample:
        if "sample_id" in frame.columns:
            columns.append("sample_id")
        elif "news_row_id" in frame.columns:
            columns.append("news_row_id")
        else:
            raise Q4EvaluationError("Q4 panel lacks a stable sample key")
    rows = sorted(
        tuple(str(value) for value in row)
        for row in frame[columns].itertuples(index=False, name=None)
    )
    return _payload_sha256(rows)


def _verify_grid(panel: pd.DataFrame) -> None:
    contracts = {
        "strike_grid": EXPECTED_STRIKE_GRID,
        "maturity_days_grid": EXPECTED_MATURITY_GRID,
        "surface_shape": (16, 16),
    }
    for column, expected in contracts.items():
        if column not in panel.columns:
            raise Q4EvaluationError(f"Q4 panel lacks {column}")
        values = panel[column].dropna().astype(str).unique().tolist()
        if len(values) != 1:
            raise Q4EvaluationError(f"Q4 panel contains multiple {column} contracts")
        observed = tuple(yaml.safe_load(values[0]))
        if column == "strike_grid":
            if len(observed) != len(expected) or any(
                abs(float(left) - float(right)) > 1e-6
                for left, right in zip(observed, expected)
            ):
                raise Q4EvaluationError("Q4 strike grid drifted")
        elif tuple(map(int, observed)) != tuple(map(int, expected)):
            raise Q4EvaluationError(f"Q4 {column} drifted")


def _load_q4_panel(
    path: Path, resolved: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel_config = _require_mapping(resolved["q4_panel"], "q4_panel")
    frame = pd.read_excel(path, sheet_name=str(panel_config["sheet_name"]))
    timestamps = pd.to_datetime(
        frame["effective_origin_utc"], errors="coerce", utc=True
    )
    start = pd.Timestamp(str(panel_config["start_utc_inclusive"]))
    end = pd.Timestamp(str(panel_config["end_utc_exclusive"]))
    if timestamps.isna().any():
        raise Q4EvaluationError("Q4 panel contains invalid timestamps")
    selected = frame.loc[(timestamps >= start) & (timestamps < end)].copy()
    if len(selected) != len(frame):
        raise Q4EvaluationError("Frozen Q4 source contains rows outside 2023Q4")
    panel, lineage, _ = _load_panel_source(
        selected,
        sheet_name=str(panel_config["sheet_name"]),
        panel_name=PANEL_NAME,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode="raw_joint",
    )
    if _counts(panel) != EXPECTED_COUNTS:
        raise Q4EvaluationError(f"Q4 supported counts drifted: {_counts(panel)}")
    _verify_grid(panel)
    lineage.update(
        {
            "interval_start_utc_inclusive": start.isoformat(),
            "interval_end_utc_exclusive": end.isoformat(),
            "counts": _counts(panel),
            "pair_universe_sha256": _universe_sha(panel, include_sample=False),
            "sample_universe_sha256": _universe_sha(panel, include_sample=True),
        }
    )
    return panel.reset_index(drop=True), lineage


def _materialize_q4(
    resolved: Mapping[str, Any], root: Path, registry: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not bool(registry.get("q4_gate_open")):
        raise RuntimeError("Q4 rows cannot be read before the explicit gate opens")
    if not _allowlist_path(root).is_file():
        raise RuntimeError("Q4 gate requires a frozen checkpoint allowlist")
    panel_config = _require_mapping(resolved["q4_panel"], "q4_panel")
    source = Path(str(panel_config["source_workbook"]))
    source_manifest = Path(str(panel_config["source_lineage_manifest"]))
    if _sha256_file(source) != str(panel_config["source_workbook_sha256"]):
        raise Q4EvaluationError("Frozen Q4 source workbook SHA drifted")
    if _sha256_file(source_manifest) != str(
        panel_config["source_lineage_manifest_sha256"]
    ):
        raise Q4EvaluationError("Frozen Q4 source lineage SHA drifted")
    target = root / "data_windows/q4/common_05m_q4.xlsx"
    manifest_path = root / "data_windows/q4/q4_panel_manifest.json"
    if target.exists() or manifest_path.exists():
        if not target.is_file() or not manifest_path.is_file():
            raise Q4EvaluationError("Partial Q4 panel materialization")
        manifest = _read_json(manifest_path)
        _verify_self_hash(manifest, "manifest_payload_sha256", "Q4 panel manifest")
        if _sha256_file(target) != str(manifest.get("workbook_sha256", "")):
            raise Q4EvaluationError("Materialized Q4 workbook drifted")
        panel, lineage = _load_q4_panel(target, resolved)
        if lineage["sample_universe_sha256"] != manifest.get("sample_universe_sha256"):
            raise Q4EvaluationError("Q4 panel sample universe drifted")
        return panel, manifest
    shutil.copy2(source, target)
    if _sha256_file(target) != str(panel_config["source_workbook_sha256"]):
        raise Q4EvaluationError("Q4 workbook copy SHA drifted")
    panel, lineage = _load_q4_panel(target, resolved)
    manifest = _self_hashed(
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "explicit_action": "evaluate",
            "source_path": str(source.resolve()),
            "source_sha256": _sha256_file(source),
            "source_lineage_manifest_path": str(source_manifest.resolve()),
            "source_lineage_manifest_sha256": _sha256_file(source_manifest),
            "workbook_path": str(target.resolve()),
            "workbook_sha256": _sha256_file(target),
            "raw_source_rows": int(
                pd.read_excel(
                    target,
                    sheet_name=str(panel_config["sheet_name"]),
                    usecols=["pair_id"],
                ).shape[0]
            ),
            "supported_counts": _counts(panel),
            "pair_universe_sha256": lineage["pair_universe_sha256"],
            "sample_universe_sha256": lineage["sample_universe_sha256"],
            "support_mask_mode": "raw_joint",
            "surface_grid_profile": "exact_ttm_16x16_v1",
            "surface_grid_sha256": training.core.GRID_SHA256,
            "q4_gate_open_before_read": True,
            "q4_window_materialized": True,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
            "materialized_at_utc": _utc_now(),
        },
        "manifest_payload_sha256",
    )
    _write_json(manifest_path, manifest)
    return panel, manifest


def _gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def _prediction_cache(
    root: Path,
    row: Mapping[str, Any],
    spec: RunSpec,
    panel: pd.DataFrame,
    panel_manifest: Mapping[str, Any],
    evaluator: TrainedRunEvaluator,
) -> pd.DataFrame:
    directory = root / "analysis/predictions"
    prediction_path = directory / f"seed_{int(row['seed'])}.csv.gz"
    manifest_path = directory / f"seed_{int(row['seed'])}.manifest.json"
    contract = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "job_id": str(row["job_id"]),
        "job_spec_sha256": str(row["job_spec_sha256"]),
        "seed": int(row["seed"]),
        "checkpoint_metadata_path": str(row["checkpoint_metadata_path"]),
        "checkpoint_metadata_sha256": str(row["checkpoint_metadata_sha256"]),
        "generator_checkpoint_sha256": str(row["generator_checkpoint_sha256"]),
        "panel_name": PANEL_NAME,
        "evaluator_panel_namespace": EVALUATOR_PANEL_NAMESPACE,
        "panel_sample_universe_sha256": str(panel_manifest["sample_universe_sha256"]),
        "mc_samples": MC_SAMPLES,
    }
    if prediction_path.exists() or manifest_path.exists():
        if not prediction_path.is_file() or not manifest_path.is_file():
            raise Q4EvaluationError(f"Partial prediction cache for seed={row['seed']}")
        manifest = _read_json(manifest_path)
        _verify_self_hash(manifest, "manifest_payload_sha256", "prediction manifest")
        unsigned = {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "prediction_path",
                "prediction_sha256",
                "row_count",
                "manifest_payload_sha256",
            }
        }
        if unsigned != contract:
            raise Q4EvaluationError(f"Prediction contract drift for seed={row['seed']}")
        if _sha256_file(prediction_path) != str(manifest["prediction_sha256"]):
            raise Q4EvaluationError(f"Prediction SHA drift for seed={row['seed']}")
        predictions = pd.read_csv(prediction_path, low_memory=False)
        if len(predictions) != int(manifest["row_count"]):
            raise Q4EvaluationError(f"Prediction row drift for seed={row['seed']}")
        return predictions
    predictions = evaluator(spec, EVALUATOR_PANEL_NAMESPACE, panel.copy())
    if (
        not isinstance(predictions, pd.DataFrame)
        or len(predictions) != EXPECTED_COUNTS["rows"]
    ):
        raise Q4EvaluationError(f"Prediction coverage drift for seed={row['seed']}")
    _gzip_csv(prediction_path, predictions)
    manifest = _self_hashed(
        {
            **contract,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": _sha256_file(prediction_path),
            "row_count": int(len(predictions)),
        },
        "manifest_payload_sha256",
    )
    _write_json(manifest_path, manifest)
    return predictions


def _run_spec_from_allowlist_row(row: Mapping[str, Any]) -> RunSpec:
    """Build the production evaluator contract from one frozen checkpoint row."""

    return RunSpec(
        run_id=str(row["job_id"]),
        run_dir=Path(str(row["run_dir"])),
        model="wgan",
        tolerance_minutes=5,
        seed=int(row["seed"]),
        checkpoint_path=Path(str(row["generator_checkpoint_path"])),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={
            "generator_conditioning_mode": row["generator_conditioning_mode"],
            "critic_conditioning_mode": row["critic_conditioning_mode"],
            "confirmation_label": CONFIRMATION_LABEL,
        },
    )


def _evaluate_checkpoints(
    resolved: Mapping[str, Any],
    root: Path,
    allowlist: Sequence[Mapping[str, Any]],
    panel: pd.DataFrame,
    panel_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    runtime = _require_mapping(resolved["runtime"], "runtime")
    evaluator = TrainedRunEvaluator(
        mc_samples=MC_SAMPLES,
        sample_batch_size=int(runtime["sample_batch_size"]),
        draw_batch_size=int(runtime["draw_batch_size"]),
        device=str(runtime["device"]),
    )
    parts: list[pd.DataFrame] = []
    coverage: set[tuple[str, str]] | None = None
    persistence: pd.Series | None = None
    for row in allowlist:
        spec = _run_spec_from_allowlist_row(row)
        predictions = _prediction_cache(
            root, row, spec, panel, panel_manifest, evaluator
        )
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            PANEL_NAME,
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise Q4EvaluationError(
                f"Prediction exclusions for seed={row['seed']}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        if _counts(pairs) != {"rows": 143, "pairs": 143, "sessions": 45}:
            raise Q4EvaluationError(f"Pair coverage drift for seed={row['seed']}")
        keys = set(
            pairs[["session_id", "pair_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if coverage is None:
            coverage = keys
            persistence = pairs.set_index("pair_id")["persistence_mae"].sort_index()
        elif keys != coverage:
            raise Q4EvaluationError("Q4 pair/session coverage differs across seeds")
        elif (
            not pairs.set_index("pair_id")["persistence_mae"]
            .sort_index()
            .equals(persistence)
        ):
            raise Q4EvaluationError("Q4 persistence vector differs across seeds")
        pairs.insert(0, "job_id", str(row["job_id"]))
        pairs.insert(1, "checkpoint_sha256", str(row["checkpoint_metadata_sha256"]))
        pairs.insert(2, "best_epoch", int(row["best_epoch"]))
        parts.append(pairs)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(SEEDS) * EXPECTED_COUNTS["pairs"]:
        raise Q4EvaluationError("Combined Q4 pair-metric row count drifted")
    if result.duplicated(["seed", "session_id", "pair_id"]).any():
        raise Q4EvaluationError("Duplicate Q4 seed/pair metric rows")
    return result


def seed_session_bootstrap(
    pair_metrics: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    required = {"seed", "session_id", "pair_id", "model_mae", "persistence_mae"}
    if missing := sorted(required - set(pair_metrics.columns)):
        raise Q4EvaluationError(f"Bootstrap input missing {missing}")
    frame = pair_metrics[list(required)].copy()
    frame["difference"] = pd.to_numeric(
        frame["model_mae"], errors="raise"
    ) - pd.to_numeric(frame["persistence_mae"], errors="raise")
    if not np.isfinite(frame["difference"].to_numpy(float)).all():
        raise Q4EvaluationError("Bootstrap differences are non-finite")
    seeds = tuple(sorted(frame["seed"].astype(int).unique().tolist()))
    if seeds != tuple(sorted(SEEDS)):
        raise Q4EvaluationError("Bootstrap requires all five seeds")
    if frame.duplicated(["seed", "session_id", "pair_id"]).any():
        raise Q4EvaluationError("Bootstrap input contains duplicate seed/pair rows")
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    coverages: list[frozenset[tuple[str, str]]] = []
    points: list[float] = []
    for seed in seeds:
        selected = frame[frame["seed"].astype(int).eq(seed)]
        coverages.append(
            frozenset(
                selected[["session_id", "pair_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
        grouped = selected.groupby("session_id", sort=True)["difference"].agg(
            ["sum", "count"]
        )
        sums = grouped["sum"].to_numpy(float)
        counts = grouped["count"].to_numpy(float)
        if len(sums) != EXPECTED_COUNTS["sessions"]:
            raise Q4EvaluationError("Bootstrap session coverage drifted")
        arrays[seed] = (sums, counts)
        points.append(float(sums.sum() / counts.sum()))
    if len(set(coverages)) != 1:
        raise Q4EvaluationError("Bootstrap pair coverage differs across seeds")
    rng = np.random.default_rng(int(random_seed))
    draws = np.empty(int(iterations), dtype=float)
    for draw_index in range(int(iterations)):
        means: list[float] = []
        for raw_seed_index in rng.integers(0, len(seeds), size=len(seeds)):
            sums, counts = arrays[seeds[int(raw_seed_index)]]
            sampled = rng.integers(0, len(sums), size=len(sums))
            means.append(float(sums[sampled].sum() / counts[sampled].sum()))
        draws[draw_index] = float(np.mean(means))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (float(np.sum(draws <= 0.0)) + 1.0) / (len(draws) + 1.0)
    p_upper = (float(np.sum(draws >= 0.0)) + 1.0) / (len(draws) + 1.0)
    return {
        "mean_difference_model_minus_persistence": float(np.mean(points)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(random_seed),
        "seed_count": len(seeds),
        "sessions_per_seed": EXPECTED_COUNTS["sessions"],
        "pairs_per_seed": EXPECTED_COUNTS["pairs"],
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _seed_rows(
    pair_metrics: pd.DataFrame, allowlist: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    frozen = {int(row["seed"]): row for row in allowlist}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        selected = pair_metrics[pair_metrics["seed"].astype(int).eq(seed)]
        model_mae = float(selected["model_mae"].mean())
        persistence_mae = float(selected["persistence_mae"].mean())
        ratio = model_mae / persistence_mae
        rows.append(
            {
                "seed": seed,
                "best_epoch": int(frozen[seed]["best_epoch"]),
                "q3_best_mae": float(frozen[seed]["q3_best_mae"]),
                "q4_model_mae": model_mae,
                "q4_persistence_mae": persistence_mae,
                "q4_log_mae_ratio_vs_persistence": math.log(ratio),
                "q4_improvement_vs_persistence_pct": (1.0 - ratio) * 100.0,
                "q4_win_rate": float(selected["win"].mean()),
                "pair_count": int(selected["pair_id"].nunique()),
                "session_count": int(selected["session_id"].nunique()),
                "checkpoint_metadata_sha256": str(
                    frozen[seed]["checkpoint_metadata_sha256"]
                ),
            }
        )
    return rows


def _finalize_panel_manifest(root: Path) -> dict[str, Any]:
    path = root / "data_windows/q4/q4_panel_manifest.json"
    manifest = _read_json(path)
    _verify_self_hash(manifest, "manifest_payload_sha256", "Q4 panel manifest")
    manifest.pop("manifest_payload_sha256")
    manifest.update(
        {
            "q4_predictions_generated": True,
            "q4_evaluated": True,
            "evaluated_at_utc": _utc_now(),
        }
    )
    manifest = _self_hashed(manifest, "manifest_payload_sha256")
    _write_json(path, manifest)
    return manifest


def _write_results(
    resolved: Mapping[str, Any],
    root: Path,
    registry: Mapping[str, Any],
    allowlist: Sequence[Mapping[str, Any]],
    pair_metrics: pd.DataFrame,
) -> dict[str, Any]:
    pair_path = root / "analysis/q4_pair_metrics.csv.gz"
    seed_path = root / "analysis/q4_seed_metrics.csv"
    _gzip_csv(pair_path, pair_metrics)
    seed_rows = _seed_rows(pair_metrics, allowlist)
    _atomic_write_text(seed_path, _csv_text(seed_rows))
    bootstrap = seed_session_bootstrap(pair_metrics)
    log_ratios = [float(row["q4_log_mae_ratio_vs_persistence"]) for row in seed_rows]
    model_maes = [float(row["q4_model_mae"]) for row in seed_rows]
    mean_log_ratio = float(np.mean(log_ratios))
    panel_manifest = _finalize_panel_manifest(root)
    artifacts: dict[str, str] = {
        "analysis/q4_pair_metrics.csv.gz": _sha256_file(pair_path),
        "analysis/q4_seed_metrics.csv": _sha256_file(seed_path),
        "data_windows/q4/common_05m_q4.xlsx": _sha256_file(
            root / "data_windows/q4/common_05m_q4.xlsx"
        ),
        "data_windows/q4/q4_panel_manifest.json": _sha256_file(
            root / "data_windows/q4/q4_panel_manifest.json"
        ),
    }
    for path in sorted((root / "analysis/predictions").glob("*")):
        if path.is_file():
            artifacts[str(path.relative_to(root))] = _sha256_file(path)
    summary = _self_hashed(
        {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "confirmation_label": CONFIRMATION_LABEL,
            "completed_at_utc": _utc_now(),
            "q4_evaluated": True,
            "q4_used_for_checkpoint_selection": False,
            "q4_used_for_epoch_selection": False,
            "selection_source": "Q3 best_learned checkpoint allowlist frozen before Q4 read",
            "training_root": str(resolved["training_root"]),
            "training_registry_sha256": str(resolved["training_registry_sha256"]),
            "allowlist_sha256": str(registry["allowlist_sha256"]),
            "q4_mc_samples": MC_SAMPLES,
            "panel_counts": EXPECTED_COUNTS,
            "panel_sample_universe_sha256": str(
                panel_manifest["sample_universe_sha256"]
            ),
            "seed_count": len(seed_rows),
            "mean_q4_model_mae": float(np.mean(model_maes)),
            "sample_std_q4_model_mae": float(np.std(model_maes, ddof=1)),
            "mean_q4_persistence_mae": float(
                np.mean([row["q4_persistence_mae"] for row in seed_rows])
            ),
            "mean_seed_log_mae_ratio_vs_persistence": mean_log_ratio,
            "geomean_improvement_vs_persistence_pct": (1.0 - math.exp(mean_log_ratio))
            * 100.0,
            "seeds_beating_persistence": sum(
                float(row["q4_model_mae"]) < float(row["q4_persistence_mae"])
                for row in seed_rows
            ),
            "bootstrap_model_minus_persistence": bootstrap,
            "statistical_support_vs_persistence": bool(
                float(bootstrap["mean_difference_model_minus_persistence"]) < 0.0
                and float(bootstrap["ci_95_upper"]) < 0.0
                and float(bootstrap["p_two_sided"]) < 0.05
            ),
            "seed_rows": seed_rows,
            "artifact_sha256": artifacts,
            "interpretation_boundary": (
                "Retrospective frozen exploratory Q4 evaluation; the 2023Q4 period "
                "has historical exposure and is not a confirmatory holdout."
            ),
        },
        "summary_payload_sha256",
    )
    summary_path = root / "analysis/q4_summary.json"
    _write_json(summary_path, summary)
    return summary


def _validate_completed(root: Path, registry: Mapping[str, Any]) -> None:
    if not all(
        bool(registry.get(field))
        for field in (
            "q4_gate_open",
            "q4_window_materialized",
            "q4_predictions_generated",
            "q4_evaluated",
        )
    ):
        raise Q4EvaluationError("Q4 registry is not terminal")
    allowlist = _read_json(_allowlist_path(root))
    _verify_self_hash(allowlist, "allowlist_payload_sha256", "checkpoint allowlist")
    panel_manifest = _read_json(root / "data_windows/q4/q4_panel_manifest.json")
    _verify_self_hash(panel_manifest, "manifest_payload_sha256", "Q4 panel manifest")
    if not bool(panel_manifest.get("q4_evaluated")):
        raise Q4EvaluationError("Q4 panel manifest is not terminal")
    summary_path = root / "analysis/q4_summary.json"
    summary = _read_json(summary_path)
    _verify_self_hash(summary, "summary_payload_sha256", "Q4 summary")
    if _sha256_file(summary_path) != str(registry.get("q4_summary_sha256", "")):
        raise Q4EvaluationError("Q4 summary SHA drifted")
    if str(summary.get("confirmation_label")) != CONFIRMATION_LABEL:
        raise Q4EvaluationError("Q4 confirmation label drifted")
    for relative, expected_sha in summary.get("artifact_sha256", {}).items():
        path = root / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected_sha):
            raise Q4EvaluationError(f"Q4 artifact drift: {path}")
    manifests = sorted((root / "analysis/predictions").glob("*.manifest.json"))
    predictions = sorted((root / "analysis/predictions").glob("*.csv.gz"))
    if len(manifests) != len(SEEDS) or len(predictions) != len(SEEDS):
        raise Q4EvaluationError("Q4 requires five prediction/manifests")
    for path in manifests:
        manifest = _read_json(path)
        _verify_self_hash(manifest, "manifest_payload_sha256", "prediction manifest")
        prediction = Path(str(manifest["prediction_path"]))
        if not prediction.is_file() or _sha256_file(prediction) != str(
            manifest["prediction_sha256"]
        ):
            raise Q4EvaluationError(f"Prediction artifact drift: {prediction}")
    pair_metrics = pd.read_csv(
        root / "analysis/q4_pair_metrics.csv.gz", low_memory=False
    )
    if len(pair_metrics) != len(SEEDS) * EXPECTED_COUNTS["pairs"]:
        raise Q4EvaluationError("Terminal Q4 pair metrics lost coverage")


def evaluate(
    resolved: Mapping[str, Any], root: Path, *, resume: bool
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "control/evaluate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another Q4 evaluator owns the lock") from exc
    registry_exists = _registry_path(root).is_file()
    registry = prepare(resolved, root, resume=resume if registry_exists else False)
    if bool(registry.get("q4_evaluated")):
        if not resume:
            raise RuntimeError("Q4 is already evaluated; pass --resume to validate")
        _validate_completed(root, registry)
        return _read_json(root / "analysis/q4_summary.json")
    allowlist_rows = _training_allowlist(resolved)
    allowlist = _validate_allowlist(root, allowlist_rows)
    registry.update(
        {
            "state": "q4_gate_open",
            "q4_gate_open": True,
            "q4_gate_opened_at_utc": _utc_now(),
        }
    )
    _write_json(_registry_path(root), registry)
    _write_status(root, "q4_gate_open", q4_gate_open=True)
    try:
        panel, panel_manifest = _materialize_q4(resolved, root, registry)
        registry.update(
            {
                "state": "q4_inference",
                "q4_rows_read": int(len(panel)),
                "q4_window_materialized": True,
                "q4_window_path": str(
                    (root / "data_windows/q4/common_05m_q4.xlsx").resolve()
                ),
                "q4_window_sha256": _sha256_file(
                    root / "data_windows/q4/common_05m_q4.xlsx"
                ),
                "q4_panel_manifest_path": str(
                    (root / "data_windows/q4/q4_panel_manifest.json").resolve()
                ),
                "q4_panel_manifest_sha256": _sha256_file(
                    root / "data_windows/q4/q4_panel_manifest.json"
                ),
            }
        )
        _write_json(_registry_path(root), registry)
        _write_status(
            root,
            "q4_inference",
            q4_rows_read=len(panel),
            q4_window_materialized=True,
        )
        pair_metrics = _evaluate_checkpoints(
            resolved, root, allowlist, panel, panel_manifest
        )
        summary = _write_results(resolved, root, registry, allowlist, pair_metrics)
        registry.update(
            {
                "state": "complete",
                "q4_predictions_generated": True,
                "q4_evaluated": True,
                "q4_summary_path": str((root / "analysis/q4_summary.json").resolve()),
                "q4_summary_sha256": _sha256_file(root / "analysis/q4_summary.json"),
                "q4_summary_payload_sha256": str(summary["summary_payload_sha256"]),
                "completed_at_utc": _utc_now(),
            }
        )
        _write_json(_registry_path(root), registry)
        _write_status(
            root,
            "complete",
            q4_rows_read=len(panel),
            q4_window_materialized=True,
            q4_predictions_generated=True,
            q4_evaluated=True,
            prediction_count=len(SEEDS),
        )
        _validate_completed(root, registry)
        return summary
    except BaseException as exc:
        _write_status(
            root,
            "failed_after_q4_gate_open",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def status(root: Path) -> dict[str, Any]:
    if not _registry_path(root).is_file():
        return {"state": "not_prepared", "output_root": str(root)}
    registry = _read_json(_registry_path(root))
    payload = _read_json(_status_path(root)) if _status_path(root).is_file() else {}
    payload.update(
        {
            "output_root": str(root),
            "checkpoint_count": int(registry.get("checkpoint_count", 0)),
            "q4_rows_read": int(registry.get("q4_rows_read", 0)),
            "q4_gate_open": bool(registry.get("q4_gate_open")),
            "q4_window_materialized": bool(registry.get("q4_window_materialized")),
            "q4_predictions_generated": bool(registry.get("q4_predictions_generated")),
            "q4_evaluated": bool(registry.get("q4_evaluated")),
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "evaluate", "status"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", default=DEFAULT_CONFIG)
        sub.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
        if action in {"prepare", "evaluate"}:
            sub.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    root = _resolve_repo_path(args.output_dir)
    if args.action == "status":
        result = status(root)
    else:
        resolved = resolve_config(args.config)
        if args.action == "prepare":
            registry = prepare(resolved, root, resume=bool(args.resume))
            result = {
                "output_root": str(root),
                "state": registry["state"],
                "checkpoint_count": registry["checkpoint_count"],
                "q4_rows_read": registry["q4_rows_read"],
                "q4_gate_open": registry["q4_gate_open"],
            }
        else:
            result = evaluate(resolved, root, resume=bool(args.resume))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
