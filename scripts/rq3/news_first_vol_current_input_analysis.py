"""Q3-only paired analysis for Generator current-support input masking.

The focal checkpoints differ from the immutable reference only in what enters
the Generator's current-surface encoder.  ``current_support_masked`` feeds
``current_surface * current_support_mask`` to that encoder; both treatments
retain the original unmasked current surface as the residual/persistence
anchor.  The discriminator and all fitted losses continue to use the future-
aware raw-joint mask.

This module deliberately has no Q4 evaluation path.  It freezes and validates
an explicit full-current reference manifest, evaluates best-learned masked
checkpoints on the common Q3 panel, and performs paired seed-then-CME-session
bootstrap inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.rq3.news_first_vol_comparison_analysis import (
    RunSpec,
    TrainedRunEvaluator,
    _load_panel_source,
    aggregate_pair_metrics,
    compute_sample_metrics,
    holm_adjust,
)
from wgan_option.models.common import (
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    generator_current_input_fingerprint,
)
from wgan_option.utils.inference_helpers import (
    resolve_checkpoint_generator_current_input_contract,
)


DEFAULT_REFERENCE_ROOT = Path(
    "outputs/experiments/rq3_news_first_vol_coverage_completion_q097_103_ttm07_38_v1"
)
REFERENCE_STAGE_ID = "stage_3_wgan_low_lr_capacity"
REFERENCE_PAIR_METRICS_RELATIVE_PATH = Path(
    "analysis/stages/stage_3_wgan_low_lr_capacity/q3_pair_metrics.csv.gz"
)
REFERENCE_MANIFEST_FILENAME = "full_current_reference_manifest.csv"
REFERENCE_MANIFEST_HASH_FILENAME = "full_current_reference_manifest.sha256"
Q3_START_UTC = "2023-07-01T00:00:00Z"
Q3_END_UTC = "2023-10-01T00:00:00Z"
SELECTION_PANEL = "common_validation_05m"
SUPPORT_MASK_MODE = "raw_joint"
CAPACITY_PROFILE = "small"
PARAMETER_COUNT = 149_333
FIXED_LEARNING_RATE = 5.0e-7
LR_PROFILE = "lr_5e_07"
TEXT_MODE = "real_text"
SEEDS = (42, 202, 404)
TOLERANCES = (5, 30)
EXPECTED_PAIR_COUNT = 123
EXPECTED_SESSION_COUNT = 33
EXPECTED_JOB_COUNT = len(SEEDS) * len(TOLERANCES)
EXPECTED_PAIR_ROWS = EXPECTED_JOB_COUNT * EXPECTED_PAIR_COUNT
PERSISTENCE_MATCH_ATOL = 1.0e-12
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260820


class CurrentInputAnalysisError(ValueError):
    """Raised when current-input evidence cannot support paired inference."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CurrentInputAnalysisError(f"Expected a JSON mapping: {target}")
    return dict(value)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    value = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise CurrentInputAnalysisError(f"Expected a YAML mapping: {target}")
    if set(value) == {"training"} and isinstance(value["training"], Mapping):
        value = value["training"]
    return dict(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    compression: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression=compression)
    return path


def _registry(root: Path) -> dict[str, Any]:
    path = root / "registry" / "jobs.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_json(path)


def _reference_job_id(seed: int, tolerance: int) -> str:
    return (
        f"s3_wgan_small_lr_5e_07_seed_{int(seed):03d}_real_text_{int(tolerance):02d}m"
    )


def _artifact_map(status: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = status.get("artifacts") or []
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        role = str(row.get("artifact_role", row.get("role", ""))).strip()
        if not role or role in output:
            raise CurrentInputAnalysisError(
                "Status contains a missing/duplicate artifact role"
            )
        output[role] = row
    return output


def _validate_hashed_path(path: Any, sha256: Any, *, label: str) -> Path:
    target = Path(str(path)).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    observed = _sha256_file(target)
    if observed != str(sha256):
        raise CurrentInputAnalysisError(
            f"{label} hash drifted: expected={sha256}, observed={observed}"
        )
    return target


def _best_learned_contract(
    metadata_path: Path,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    epoch = int(metadata.get("best_learned_epoch_ge_1", metadata.get("best_epoch", -1)))
    if epoch < 1 or str(metadata.get("selection_scope", "")) != "trained_epochs_only":
        raise CurrentInputAnalysisError(
            f"Analysis requires best_learned epoch >=1: {metadata_path}"
        )
    saved_mode = (
        str(metadata.get("generator_current_input_mode", expected_mode)).strip().lower()
    )
    if saved_mode != expected_mode:
        raise CurrentInputAnalysisError(
            f"Best-learned current-input mode drifted: {metadata_path}"
        )
    saved_fingerprint = str(
        metadata.get(
            "generator_current_input_fingerprint",
            generator_current_input_fingerprint(expected_mode),
        )
    )
    if saved_fingerprint != generator_current_input_fingerprint(expected_mode):
        raise CurrentInputAnalysisError(
            f"Best-learned current-input fingerprint drifted: {metadata_path}"
        )
    return metadata


def _checkpoint_current_input_contract(
    checkpoint_path: Path,
    *,
    expected_mode: str,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode, fingerprint = resolve_checkpoint_generator_current_input_contract(checkpoint)
    expected_fingerprint = generator_current_input_fingerprint(expected_mode)
    if mode != expected_mode or fingerprint != expected_fingerprint:
        raise CurrentInputAnalysisError(
            f"Checkpoint current-input contract drifted: {checkpoint_path}"
        )
    return fingerprint


def freeze_full_current_reference_manifest(
    experiment_root: str | Path,
    *,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
) -> Path:
    """Freeze the exact six immutable full-current reference cells.

    This helper is intended for the orchestrator's prepare phase.  Formal
    analysis never calls it implicitly: a missing manifest is an error rather
    than permission to discover a new/latest reference.
    """

    root = Path(experiment_root).resolve(strict=False)
    reference = Path(reference_root).resolve()
    registry_path = reference / "registry" / "jobs.json"
    registry = _read_json(registry_path)
    jobs = {str(row["job_id"]): dict(row) for row in registry.get("jobs", [])}
    pair_path = reference / REFERENCE_PAIR_METRICS_RELATIVE_PATH
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for tolerance in TOLERANCES:
            job_id = _reference_job_id(seed, tolerance)
            if job_id not in jobs:
                raise CurrentInputAnalysisError(
                    f"Reference registry lacks frozen job {job_id}"
                )
            job = jobs[job_id]
            expected_axes = {
                "stage_id": REFERENCE_STAGE_ID,
                "model_family": "wgan",
                "capacity_profile": CAPACITY_PROFILE,
                "lr_profile": LR_PROFILE,
                "seed": seed,
                "text_ablation_mode": TEXT_MODE,
                "tolerance_minutes": tolerance,
                "support_mask_mode": SUPPORT_MASK_MODE,
            }
            for field, expected in expected_axes.items():
                if job.get(field) != expected:
                    raise CurrentInputAnalysisError(
                        f"Reference axis drifted for {job_id}: {field}"
                    )
            if not math.isclose(
                float(job.get("initial_learning_rate", float("nan"))),
                FIXED_LEARNING_RATE,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            ):
                raise CurrentInputAnalysisError(f"Reference LR drifted for {job_id}")
            status_path = reference / "registry" / "jobs" / f"{job_id}.status.json"
            status = _read_json(status_path)
            if status.get("status") != "completed" or status.get(
                "config_sha256"
            ) != job.get("config_sha256"):
                raise CurrentInputAnalysisError(f"Reference status invalid: {job_id}")
            artifacts = _artifact_map(status)
            required = {
                "generator_best_learned",
                "best_learned_checkpoint",
                "resolved_training_config",
            }
            if not required.issubset(artifacts):
                raise CurrentInputAnalysisError(
                    f"Reference learned artifacts missing: {job_id}"
                )
            for role in required:
                _validate_hashed_path(
                    artifacts[role]["path"],
                    artifacts[role]["sha256"],
                    label=f"reference {job_id}/{role}",
                )
            metadata_path = Path(artifacts["best_learned_checkpoint"]["path"])
            metadata = _best_learned_contract(
                metadata_path,
                expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
            )
            checkpoint_path = Path(artifacts["generator_best_learned"]["path"])
            fingerprint = _checkpoint_current_input_contract(
                checkpoint_path,
                expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
            )
            config_path = Path(artifacts["resolved_training_config"]["path"])
            config = _read_yaml(config_path)
            if (
                str(config.get("support_mask_mode", "")).strip().lower()
                != SUPPORT_MASK_MODE
            ):
                raise CurrentInputAnalysisError(
                    f"Reference config mask drifted: {job_id}"
                )
            rows.append(
                {
                    "seed": seed,
                    "tolerance_minutes": tolerance,
                    "reference_job_id": job_id,
                    "generator_current_input_mode": FULL_CURRENT_GENERATOR_INPUT_MODE,
                    "generator_current_input_fingerprint": fingerprint,
                    "reference_root": str(reference),
                    "reference_registry_path": str(registry_path),
                    "reference_registry_sha256": _sha256_file(registry_path),
                    "reference_status_path": str(status_path),
                    "reference_status_sha256": _sha256_file(status_path),
                    "reference_training_config_path": str(config_path),
                    "reference_training_config_sha256": _sha256_file(config_path),
                    "best_learned_epoch": int(
                        metadata.get(
                            "best_learned_epoch_ge_1", metadata.get("best_epoch")
                        )
                    ),
                    "best_learned_metadata_path": str(metadata_path),
                    "best_learned_metadata_sha256": _sha256_file(metadata_path),
                    "generator_checkpoint_path": str(checkpoint_path),
                    "generator_checkpoint_sha256": _sha256_file(checkpoint_path),
                    "reference_q3_pair_metrics_path": str(pair_path),
                    "reference_q3_pair_metrics_sha256": _sha256_file(pair_path),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["seed", "tolerance_minutes"], kind="stable"
    )
    manifest_path = _write_csv(manifest, root / REFERENCE_MANIFEST_FILENAME)
    manifest_sha = _sha256_file(manifest_path)
    hash_path = root / REFERENCE_MANIFEST_HASH_FILENAME
    hash_path.write_text(manifest_sha + "\n", encoding="utf-8")
    return manifest_path


def _load_reference_manifest(
    experiment_root: Path,
    reference_root: Path,
) -> tuple[pd.DataFrame, str]:
    manifest_path = experiment_root / REFERENCE_MANIFEST_FILENAME
    hash_path = experiment_root / REFERENCE_MANIFEST_HASH_FILENAME
    if not manifest_path.is_file() or not hash_path.is_file():
        raise CurrentInputAnalysisError(
            "Formal analysis requires the frozen full-current reference manifest"
        )
    manifest_sha = _sha256_file(manifest_path)
    if hash_path.read_text(encoding="utf-8").strip() != manifest_sha:
        raise CurrentInputAnalysisError("Reference manifest companion hash mismatch")
    registry_path = experiment_root / "registry" / "jobs.json"
    if registry_path.is_file():
        recorded = str(
            _read_json(registry_path).get("full_current_reference_manifest_sha256", "")
        )
        if recorded and recorded != manifest_sha:
            raise CurrentInputAnalysisError("Registry reference-manifest hash mismatch")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "seed",
        "tolerance_minutes",
        "reference_job_id",
        "generator_current_input_mode",
        "generator_current_input_fingerprint",
        "reference_root",
        "reference_registry_path",
        "reference_registry_sha256",
        "reference_status_path",
        "reference_status_sha256",
        "reference_training_config_path",
        "reference_training_config_sha256",
        "best_learned_epoch",
        "best_learned_metadata_path",
        "best_learned_metadata_sha256",
        "generator_checkpoint_path",
        "generator_checkpoint_sha256",
        "reference_q3_pair_metrics_path",
        "reference_q3_pair_metrics_sha256",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise CurrentInputAnalysisError(f"Reference manifest missing {missing}")
    if len(manifest) != EXPECTED_JOB_COUNT:
        raise CurrentInputAnalysisError("Reference manifest must contain six rows")
    roots = {str(Path(value).resolve()) for value in manifest["reference_root"]}
    expected_root = str(reference_root.resolve())
    if roots != {expected_root}:
        raise CurrentInputAnalysisError(
            f"Frozen reference root differs from requested root: {sorted(roots)}"
        )
    expected_cells = {(seed, tolerance) for seed in SEEDS for tolerance in TOLERANCES}
    observed_cells = set(
        zip(
            pd.to_numeric(manifest["seed"], errors="coerce").astype(int),
            pd.to_numeric(manifest["tolerance_minutes"], errors="coerce").astype(int),
        )
    )
    if (
        observed_cells != expected_cells
        or manifest.duplicated(["seed", "tolerance_minutes"]).any()
    ):
        raise CurrentInputAnalysisError("Reference manifest cell matrix drifted")
    expected_fingerprint = generator_current_input_fingerprint(
        FULL_CURRENT_GENERATOR_INPUT_MODE
    )
    if set(manifest["generator_current_input_mode"].astype(str)) != {
        FULL_CURRENT_GENERATOR_INPUT_MODE
    } or set(manifest["generator_current_input_fingerprint"].astype(str)) != {
        expected_fingerprint
    }:
        raise CurrentInputAnalysisError("Reference input-mode lineage drifted")
    for row in manifest.to_dict(orient="records"):
        for path_field, hash_field in (
            ("reference_registry_path", "reference_registry_sha256"),
            ("reference_status_path", "reference_status_sha256"),
            ("reference_training_config_path", "reference_training_config_sha256"),
            ("best_learned_metadata_path", "best_learned_metadata_sha256"),
            ("generator_checkpoint_path", "generator_checkpoint_sha256"),
            ("reference_q3_pair_metrics_path", "reference_q3_pair_metrics_sha256"),
        ):
            _validate_hashed_path(
                row[path_field], row[hash_field], label=f"manifest {path_field}"
            )
        _best_learned_contract(
            Path(str(row["best_learned_metadata_path"])),
            expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
        )
        _checkpoint_current_input_contract(
            Path(str(row["generator_checkpoint_path"])),
            expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
        )
    return manifest.reset_index(drop=True), manifest_sha


def _masked_jobs(root: Path) -> list[dict[str, Any]]:
    registry = _registry(root)
    jobs: list[dict[str, Any]] = []
    for raw in registry.get("jobs", []):
        job = dict(raw)
        try:
            selected = (
                str(job.get("model_family", "")).lower() == "wgan"
                and str(job.get("capacity_profile", "")).lower() == CAPACITY_PROFILE
                and math.isclose(
                    float(job.get("initial_learning_rate", float("nan"))),
                    FIXED_LEARNING_RATE,
                    rel_tol=0.0,
                    abs_tol=1.0e-18,
                )
                and int(job.get("seed", -1)) in SEEDS
                and int(job.get("tolerance_minutes", -1)) in TOLERANCES
                and str(job.get("text_ablation_mode", "")).lower() == TEXT_MODE
                and str(job.get("support_mask_mode", "")).lower() == SUPPORT_MASK_MODE
                and str(job.get("generator_current_input_mode", "")).lower()
                == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            )
        except (TypeError, ValueError):
            selected = False
        if selected:
            status_path = root / "registry" / "jobs" / f"{job['job_id']}.status.json"
            status = _read_json(status_path)
            if status.get("status") != "completed" or status.get(
                "config_sha256"
            ) != job.get("config_sha256"):
                raise CurrentInputAnalysisError(
                    f"Masked status invalid: {job['job_id']}"
                )
            artifacts = _artifact_map(status)
            for role in (
                "generator_best_learned",
                "best_learned_checkpoint",
                "resolved_training_config",
            ):
                if role not in artifacts:
                    raise CurrentInputAnalysisError(
                        f"Masked job lacks {role}: {job['job_id']}"
                    )
                _validate_hashed_path(
                    artifacts[role]["path"],
                    artifacts[role]["sha256"],
                    label=f"masked {job['job_id']}/{role}",
                )
            jobs.append(
                {
                    **job,
                    "status_path": str(status_path),
                    "status_sha256": _sha256_file(status_path),
                    "run_dir": str(status["run_dir"]),
                    "artifacts_by_role": artifacts,
                }
            )
    cells = {(int(job["seed"]), int(job["tolerance_minutes"])) for job in jobs}
    expected = {(seed, tolerance) for seed in SEEDS for tolerance in TOLERANCES}
    if len(jobs) != EXPECTED_JOB_COUNT or cells != expected:
        raise CurrentInputAnalysisError(
            "Masked analysis requires exactly 3 seeds x {5m,30m} real-text jobs"
        )
    return sorted(
        jobs, key=lambda row: (int(row["seed"]), int(row["tolerance_minutes"]))
    )


def _masked_run_spec(job: Mapping[str, Any]) -> tuple[RunSpec, dict[str, Any]]:
    artifacts = dict(job["artifacts_by_role"])
    metadata_path = Path(str(artifacts["best_learned_checkpoint"]["path"]))
    metadata = _best_learned_contract(
        metadata_path,
        expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    )
    checkpoint_path = Path(str(artifacts["generator_best_learned"]["path"]))
    fingerprint = _checkpoint_current_input_contract(
        checkpoint_path,
        expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    )
    config_path = Path(str(artifacts["resolved_training_config"]["path"]))
    config = _read_yaml(config_path)
    contracts = {
        "support_mask_mode": SUPPORT_MASK_MODE,
        "news_first_text_ablation_mode": TEXT_MODE,
        "news_first_capacity_profile": CAPACITY_PROFILE,
        "generator_current_input_mode": CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    }
    for field, expected in contracts.items():
        if str(config.get(field, "")).strip().lower() != expected:
            raise CurrentInputAnalysisError(
                f"Masked training config drifted {job['job_id']}: {field}"
            )
    if int(config.get("seed", -1)) != int(job["seed"]):
        raise CurrentInputAnalysisError(f"Masked seed config drifted: {job['job_id']}")
    if int(config.get("news_first_dataset_tolerance_minutes", -1)) != int(
        job["tolerance_minutes"]
    ):
        raise CurrentInputAnalysisError(
            f"Masked tolerance config drifted: {job['job_id']}"
        )
    if not math.isclose(
        float(config.get("learning_rate", float("nan"))),
        FIXED_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1.0e-18,
    ):
        raise CurrentInputAnalysisError(f"Masked LR config drifted: {job['job_id']}")
    spec = RunSpec(
        run_id=str(job["job_id"]),
        run_dir=Path(str(job["run_dir"])),
        model="wgan",
        tolerance_minutes=int(job["tolerance_minutes"]),
        seed=int(job["seed"]),
        checkpoint_path=checkpoint_path,
        text_ablation_mode=TEXT_MODE,
        support_mask_mode=SUPPORT_MASK_MODE,
        generator_current_input_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        manifest_path=config_path,
        metadata={
            **config,
            "generator_current_input_fingerprint": fingerprint,
            "best_learned_metadata_path": str(metadata_path),
        },
    )
    selection = {
        "generator_current_input_mode": CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        "generator_current_input_fingerprint": fingerprint,
        "job_id": str(job["job_id"]),
        "seed": int(job["seed"]),
        "tolerance_minutes": int(job["tolerance_minutes"]),
        "best_learned_epoch": int(
            metadata.get("best_learned_epoch_ge_1", metadata.get("best_epoch"))
        ),
        "selection_scope": "trained_epochs_only",
        "selection_metadata_path": str(metadata_path),
        "selection_metadata_sha256": _sha256_file(metadata_path),
        "generator_checkpoint_path": str(checkpoint_path),
        "generator_checkpoint_sha256": _sha256_file(checkpoint_path),
        "training_config_path": str(config_path),
        "training_config_sha256": _sha256_file(config_path),
        "status_path": str(job["status_path"]),
        "status_sha256": str(job["status_sha256"]),
    }
    return spec, selection


def _q3_panel(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources: set[tuple[str, str, str]] = set()
    for job in jobs:
        artifacts = dict(job["artifacts_by_role"])
        config = _read_yaml(artifacts["resolved_training_config"]["path"])
        sources.add(
            (
                str(config.get("news_first_common_eval_data_path", "")),
                str(config.get("sheet_name", "gan_input_ready")),
                str(config.get("support_mask_mode", "none")).strip().lower(),
            )
        )
    if len(sources) != 1:
        raise CurrentInputAnalysisError("Masked jobs disagree on the common Q3 panel")
    raw_path, sheet_name, support_mode = next(iter(sources))
    if support_mode != SUPPORT_MASK_MODE:
        raise CurrentInputAnalysisError("Q3 analysis requires raw_joint support")
    workbook = Path(raw_path).resolve()
    raw = pd.read_excel(workbook, sheet_name=sheet_name)
    timestamps = pd.to_datetime(
        raw.get("effective_origin_utc"), errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise CurrentInputAnalysisError("Invalid common-panel timestamps")
    start, end = pd.Timestamp(Q3_START_UTC), pd.Timestamp(Q3_END_UTC)
    q3 = raw.loc[(timestamps >= start) & (timestamps < end)].copy()
    panel, lineage, _ = _load_panel_source(
        q3,
        sheet_name=sheet_name,
        panel_name=SELECTION_PANEL,
        freeze_test_window=False,
        enforce_expected_counts=False,
        support_mask_mode=support_mode,
    )
    selected = pd.to_datetime(panel["effective_origin_utc"], errors="coerce", utc=True)
    if selected.isna().any() or bool(
        (selected < start).any() or (selected >= end).any()
    ):
        raise CurrentInputAnalysisError("A non-Q3 row reached the masked evaluator")
    lineage.update(
        {
            "path": str(workbook),
            "sha256": _sha256_file(workbook),
            "interval_start_utc": Q3_START_UTC,
            "interval_end_utc_exclusive": Q3_END_UTC,
            "source_rows_materialized_before_q3_filter": int(len(raw)),
            "q3_rows_passed_to_evaluator": int(len(panel)),
            "q4_rows_passed_to_evaluator": 0,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
        }
    )
    return panel, lineage


def evaluate_masked_q3_pair_metrics(
    experiment_root: str | Path,
    *,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    prediction_artifact_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Evaluate the six best-learned masked checkpoints on common Q3 only."""

    root = Path(experiment_root).resolve()
    jobs = _masked_jobs(root)
    panel, lineage = _q3_panel(jobs)
    production = evaluator or TrainedRunEvaluator(mc_samples=16)
    expected_fingerprint = generator_current_input_fingerprint(
        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
    )
    pair_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_artifacts: list[dict[str, Any]] = []
    for job in jobs:
        spec, selection = _masked_run_spec(job)
        selection_rows.append(selection)
        predictions = production(spec, "q3", panel.copy())
        required = {
            "prediction_mc_samples",
            "prediction_fallback",
            "generator_current_input_mode",
            "generator_current_input_fingerprint",
            "current_support_cell_count",
            "current_support_mask_fingerprint",
        }
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise CurrentInputAnalysisError(
                f"Masked prediction lineage missing {missing}: {spec.run_id}"
            )
        if len(predictions) != len(panel):
            raise CurrentInputAnalysisError(
                f"Masked evaluator lost Q3 rows: {spec.run_id}"
            )
        if (
            not pd.to_numeric(predictions["prediction_mc_samples"], errors="coerce")
            .eq(16)
            .all()
        ):
            raise CurrentInputAnalysisError("Masked Gaussian evaluation requires MC16")
        if predictions["prediction_fallback"].astype(bool).any():
            raise CurrentInputAnalysisError("Prediction fallback is forbidden")
        if set(predictions["generator_current_input_mode"].astype(str)) != {
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        } or set(predictions["generator_current_input_fingerprint"].astype(str)) != {
            expected_fingerprint
        }:
            raise CurrentInputAnalysisError("Masked prediction input lineage drifted")
        counts = pd.to_numeric(
            predictions["current_support_cell_count"], errors="coerce"
        )
        fingerprints = predictions["current_support_mask_fingerprint"].astype(str)
        if (
            counts.isna().any()
            or bool((counts <= 0).any())
            or not fingerprints.str.fullmatch(r"[0-9a-f]{64}").all()
        ):
            raise CurrentInputAnalysisError("Masked prediction support lineage invalid")
        if prediction_artifact_dir is not None:
            export = predictions.copy()
            export.insert(0, "run_id", spec.run_id)
            export.insert(1, "panel", SELECTION_PANEL)
            export.insert(2, "seed", spec.seed)
            export.insert(3, "tolerance_minutes", spec.tolerance_minutes)
            path = _write_csv(
                export,
                Path(prediction_artifact_dir) / f"{spec.run_id}_q3_predictions.csv.gz",
                compression="gzip",
            )
            prediction_artifacts.append(
                {
                    "role": f"masked_q3_predictions:{spec.run_id}",
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": int(len(export)),
                }
            )
        samples, exclusions, _ = compute_sample_metrics(
            spec,
            SELECTION_PANEL,
            panel,
            predictions,
            evaluate_embedded_atm_skew=False,
        )
        if not exclusions.empty:
            raise CurrentInputAnalysisError(
                f"Masked Q3 exclusions for {spec.run_id}: "
                f"{exclusions['exclusion_code'].value_counts().to_dict()}"
            )
        pairs = aggregate_pair_metrics(samples)
        pairs = pairs[
            pairs["stratum_type"].astype(str).eq("overall")
            & pairs["stratum_value"].astype(str).eq("all")
        ].copy()
        pairs["generator_current_input_mode"] = (
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        )
        pairs["generator_current_input_fingerprint"] = expected_fingerprint
        pairs["parameter_count"] = PARAMETER_COUNT
        pairs["initial_learning_rate"] = FIXED_LEARNING_RATE
        pairs["capacity_profile"] = CAPACITY_PROFILE
        pairs["evidence_source"] = "new_masked_best_learned_q3"
        pairs["selection_metadata_sha256"] = selection["selection_metadata_sha256"]
        pairs["generator_checkpoint_sha256"] = selection["generator_checkpoint_sha256"]
        pair_parts.append(pairs)
    output = validate_mode_pair_metrics(
        pd.concat(pair_parts, ignore_index=True),
        expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    )
    lineage.update(
        {
            "evaluated_run_count": EXPECTED_JOB_COUNT,
            "prediction_artifacts": prediction_artifacts,
            "q4_used_for_checkpoint_selection": False,
        }
    )
    return output, lineage, pd.DataFrame(selection_rows)


def validate_mode_pair_metrics(
    frame: pd.DataFrame,
    *,
    expected_mode: str,
) -> pd.DataFrame:
    """Validate one exact 3-seed x 2-tolerance x 123-pair Q3 mode matrix."""

    required = {
        "run_id",
        "model",
        "text_ablation_mode",
        "support_mask_mode",
        "generator_current_input_mode",
        "seed",
        "tolerance_minutes",
        "panel",
        "stratum_type",
        "stratum_value",
        "pair_id",
        "session_id",
        "model_mae",
        "persistence_mae",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CurrentInputAnalysisError(f"Pair metrics missing {missing}")
    output = frame.copy()
    output["seed"] = pd.to_numeric(output["seed"], errors="coerce").astype(int)
    output["tolerance_minutes"] = pd.to_numeric(
        output["tolerance_minutes"], errors="coerce"
    ).astype(int)
    expected_fingerprint = generator_current_input_fingerprint(expected_mode)
    if set(output["generator_current_input_mode"].astype(str)) != {expected_mode}:
        raise CurrentInputAnalysisError("Pair-metric input mode drifted")
    if "generator_current_input_fingerprint" not in output.columns:
        output["generator_current_input_fingerprint"] = expected_fingerprint
    if set(output["generator_current_input_fingerprint"].astype(str)) != {
        expected_fingerprint
    }:
        raise CurrentInputAnalysisError("Pair-metric input fingerprint drifted")
    fixed_sets = {
        "model": {"wgan"},
        "text_ablation_mode": {TEXT_MODE},
        "support_mask_mode": {SUPPORT_MASK_MODE},
        "panel": {SELECTION_PANEL},
        "stratum_type": {"overall"},
        "stratum_value": {"all"},
    }
    for column, expected in fixed_sets.items():
        if set(output[column].astype(str).str.lower()) != expected:
            raise CurrentInputAnalysisError(f"Pair-metric axis drifted: {column}")
    cells = set(
        output[["seed", "tolerance_minutes"]].itertuples(index=False, name=None)
    )
    expected_cells = {(seed, tolerance) for seed in SEEDS for tolerance in TOLERANCES}
    key = [
        "generator_current_input_mode",
        "seed",
        "tolerance_minutes",
        "pair_id",
    ]
    if (
        cells != expected_cells
        or len(output) != EXPECTED_PAIR_ROWS
        or output.duplicated(key).any()
    ):
        raise CurrentInputAnalysisError("Pair metrics are not the exact 6x123 matrix")
    coverage: list[frozenset[tuple[str, str]]] = []
    for cell, group in output.groupby(
        ["generator_current_input_mode", "seed", "tolerance_minutes"], sort=False
    ):
        if (
            group["pair_id"].nunique() != EXPECTED_PAIR_COUNT
            or group["session_id"].nunique() != EXPECTED_SESSION_COUNT
        ):
            raise CurrentInputAnalysisError(f"Q3 cell coverage drifted: {cell}")
        coverage.append(
            frozenset(
                group[["pair_id", "session_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
    if len(set(coverage)) != 1:
        raise CurrentInputAnalysisError("Mode cells do not share one paired Q3 panel")
    for column in ("model_mae", "persistence_mae"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
        values = output[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or bool((values <= 0.0).any()):
            raise CurrentInputAnalysisError(f"Invalid {column}")
    persistence_span = output.groupby("pair_id")["persistence_mae"].agg(
        lambda values: float(values.max() - values.min())
    )
    if bool((persistence_span > PERSISTENCE_MATCH_ATOL).any()):
        raise CurrentInputAnalysisError("Persistence differs across mode cells")
    return output.sort_values(
        [
            "generator_current_input_mode",
            "seed",
            "tolerance_minutes",
            "session_id",
            "pair_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def load_full_current_reference_evidence(
    experiment_root: str | Path,
    *,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load only the six cells named by the frozen in-root reference manifest."""

    root = Path(experiment_root).resolve()
    reference = Path(reference_root).resolve()
    manifest, manifest_sha = _load_reference_manifest(root, reference)
    sources = manifest[
        ["reference_q3_pair_metrics_path", "reference_q3_pair_metrics_sha256"]
    ].drop_duplicates()
    if len(sources) != 1:
        raise CurrentInputAnalysisError("Reference pair-metric lineage differs by cell")
    source = sources.iloc[0]
    pair_path = _validate_hashed_path(
        source["reference_q3_pair_metrics_path"],
        source["reference_q3_pair_metrics_sha256"],
        label="reference Q3 pair metrics",
    )
    all_pairs = pd.read_csv(pair_path, low_memory=False)
    run_ids = set(manifest["reference_job_id"].astype(str))
    selected = all_pairs[all_pairs["run_id"].astype(str).isin(run_ids)].copy()
    selected["generator_current_input_mode"] = FULL_CURRENT_GENERATOR_INPUT_MODE
    selected["generator_current_input_fingerprint"] = (
        generator_current_input_fingerprint(FULL_CURRENT_GENERATOR_INPUT_MODE)
    )
    selected["parameter_count"] = PARAMETER_COUNT
    selected["initial_learning_rate"] = FIXED_LEARNING_RATE
    selected["capacity_profile"] = CAPACITY_PROFILE
    selected["evidence_source"] = "immutable_full_current_reference_q3"
    manifest_by_run = manifest.set_index("reference_job_id")
    selected["selection_metadata_sha256"] = selected["run_id"].map(
        manifest_by_run["best_learned_metadata_sha256"]
    )
    selected["generator_checkpoint_sha256"] = selected["run_id"].map(
        manifest_by_run["generator_checkpoint_sha256"]
    )
    output = validate_mode_pair_metrics(
        selected,
        expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
    )
    selection = manifest.rename(
        columns={
            "reference_job_id": "job_id",
            "best_learned_metadata_path": "selection_metadata_path",
            "best_learned_metadata_sha256": "selection_metadata_sha256",
        }
    ).copy()
    selection["selection_scope"] = "trained_epochs_only"
    lineage = {
        "reference_root": str(reference),
        "reference_manifest_path": str(root / REFERENCE_MANIFEST_FILENAME),
        "reference_manifest_sha256": manifest_sha,
        "reference_pair_metrics_path": str(pair_path),
        "reference_pair_metrics_sha256": _sha256_file(pair_path),
        "reference_run_count": EXPECTED_JOB_COUNT,
    }
    return output, selection, lineage


def build_paired_pair_metrics(
    masked_pair_metrics: pd.DataFrame,
    full_current_pair_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Join masked/full rows one-to-one on seed, tolerance and Q3 pair."""

    masked = validate_mode_pair_metrics(
        masked_pair_metrics,
        expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    )
    full = validate_mode_pair_metrics(
        full_current_pair_metrics,
        expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
    )
    keys = ["seed", "tolerance_minutes", "pair_id", "session_id"]
    left = masked[
        keys
        + [
            "run_id",
            "model_mae",
            "persistence_mae",
            "generator_current_input_mode",
            "selection_metadata_sha256",
            "generator_checkpoint_sha256",
        ]
    ].rename(
        columns={
            "run_id": "masked_run_id",
            "model_mae": "masked_mae",
            "persistence_mae": "masked_persistence_mae",
            "generator_current_input_mode": "masked_generator_current_input_mode",
            "selection_metadata_sha256": "masked_selection_metadata_sha256",
            "generator_checkpoint_sha256": "masked_generator_checkpoint_sha256",
        }
    )
    right = full[
        keys
        + [
            "run_id",
            "model_mae",
            "persistence_mae",
            "generator_current_input_mode",
            "selection_metadata_sha256",
            "generator_checkpoint_sha256",
        ]
    ].rename(
        columns={
            "run_id": "full_current_run_id",
            "model_mae": "full_current_mae",
            "persistence_mae": "full_current_persistence_mae",
            "generator_current_input_mode": "full_generator_current_input_mode",
            "selection_metadata_sha256": "full_selection_metadata_sha256",
            "generator_checkpoint_sha256": "full_generator_checkpoint_sha256",
        }
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(paired) != EXPECTED_PAIR_ROWS:
        raise CurrentInputAnalysisError("Masked/full pairing lost Q3 rows")
    if not np.allclose(
        paired["masked_persistence_mae"],
        paired["full_current_persistence_mae"],
        rtol=0.0,
        atol=PERSISTENCE_MATCH_ATOL,
    ):
        raise CurrentInputAnalysisError("Paired persistence differs by input mode")
    paired["persistence_mae"] = paired["masked_persistence_mae"]
    paired["masked_minus_full_current"] = (
        paired["masked_mae"] - paired["full_current_mae"]
    )
    paired["masked_minus_persistence"] = (
        paired["masked_mae"] - paired["persistence_mae"]
    )
    paired["full_current_minus_persistence"] = (
        paired["full_current_mae"] - paired["persistence_mae"]
    )
    paired["comparison_generator_current_input_modes"] = (
        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        + "_minus_"
        + FULL_CURRENT_GENERATOR_INPUT_MODE
    )
    # Keep the candidate mode in the canonical analysis key even though this
    # table is already a paired comparison.  The explicitly named reference
    # column prevents the comparison row from being mistaken for a pooled
    # single-mode observation downstream.
    paired["generator_current_input_mode"] = CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
    paired["reference_generator_current_input_mode"] = FULL_CURRENT_GENERATOR_INPUT_MODE
    return paired.sort_values(keys, kind="stable").reset_index(drop=True)


def build_cell_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Return one pair-balanced row for each seed x tolerance cell."""

    rows: list[dict[str, Any]] = []
    for (seed, tolerance), group in paired.groupby(
        ["seed", "tolerance_minutes"], sort=True
    ):
        masked = float(group["masked_mae"].mean())
        full = float(group["full_current_mae"].mean())
        persistence = float(group["persistence_mae"].mean())
        rows.append(
            {
                "generator_current_input_mode": (
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
                "reference_generator_current_input_mode": (
                    FULL_CURRENT_GENERATOR_INPUT_MODE
                ),
                "seed": int(seed),
                "tolerance_minutes": int(tolerance),
                "pair_count": int(group["pair_id"].nunique()),
                "session_count": int(group["session_id"].nunique()),
                "masked_generator_current_input_mode": (
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
                "masked_mae": masked,
                "full_current_mae": full,
                "persistence_mae": persistence,
                "masked_minus_full_current": masked - full,
                "masked_minus_persistence": masked - persistence,
                "full_current_minus_persistence": full - persistence,
                "masked_beats_full_pair_rate": float(
                    group["masked_minus_full_current"].lt(0.0).mean()
                ),
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != EXPECTED_JOB_COUNT:
        raise CurrentInputAnalysisError("Cell summary must have six rows")
    return output


def build_cross_seed_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize the three independent seed point estimates per tolerance."""

    rows: list[dict[str, Any]] = []
    metrics = (
        "masked_mae",
        "full_current_mae",
        "persistence_mae",
        "masked_minus_full_current",
        "masked_minus_persistence",
        "full_current_minus_persistence",
    )
    for tolerance, group in cell_summary.groupby("tolerance_minutes", sort=True):
        if tuple(sorted(group["seed"].astype(int))) != SEEDS:
            raise CurrentInputAnalysisError("Cross-seed summary lost a seed")
        row: dict[str, Any] = {
            "generator_current_input_mode": (
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            ),
            "reference_generator_current_input_mode": (
                FULL_CURRENT_GENERATOR_INPUT_MODE
            ),
            "tolerance_minutes": int(tolerance),
            "seed_count": len(SEEDS),
            "pair_count_per_seed": EXPECTED_PAIR_COUNT,
            "session_count_per_seed": EXPECTED_SESSION_COUNT,
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}_across_seeds"] = float(values.std(ddof=1))
        rows.append(row)
    output = pd.DataFrame(rows)
    if len(output) != len(TOLERANCES):
        raise CurrentInputAnalysisError("Cross-seed summary must have two rows")
    return output


def build_combined_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the single equal-seed/equal-tolerance combined point estimate."""

    if len(cell_summary) != EXPECTED_JOB_COUNT:
        raise CurrentInputAnalysisError("Combined summary requires all six cells")
    metrics = (
        "masked_mae",
        "full_current_mae",
        "persistence_mae",
        "masked_minus_full_current",
        "masked_minus_persistence",
        "full_current_minus_persistence",
    )
    row: dict[str, Any] = {
        "generator_current_input_mode": (CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE),
        "reference_generator_current_input_mode": (FULL_CURRENT_GENERATOR_INPUT_MODE),
        "tolerance_scope": "combined_05m_30m",
        "seed_count": len(SEEDS),
        "tolerance_count": len(TOLERANCES),
        "pair_rows_per_seed": EXPECTED_PAIR_COUNT * len(TOLERANCES),
        "session_count_per_seed": EXPECTED_SESSION_COUNT,
    }
    for metric in metrics:
        per_seed = cell_summary.groupby("seed")[metric].mean().to_numpy(dtype=float)
        row[metric] = float(per_seed.mean())
        row[f"sd_{metric}_across_seeds"] = float(per_seed.std(ddof=1))
    return pd.DataFrame([row])


def _two_level_bootstrap(
    differences: pd.DataFrame,
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Resample seeds first, then paired complete CME-session clusters."""

    if int(iterations) < 2:
        raise CurrentInputAnalysisError("Bootstrap requires at least two draws")
    required = {"seed", "session_id", "pair_id", "difference"}
    missing = sorted(required - set(differences.columns))
    if missing:
        raise CurrentInputAnalysisError(f"Bootstrap input missing {missing}")
    frame = differences.copy()
    frame["difference"] = pd.to_numeric(frame["difference"], errors="coerce")
    if not np.isfinite(frame["difference"].to_numpy(dtype=float)).all():
        raise CurrentInputAnalysisError("Bootstrap differences are non-finite")
    seeds = tuple(sorted(frame["seed"].astype(int).unique().tolist()))
    if seeds != SEEDS:
        raise CurrentInputAnalysisError("Bootstrap lost a training seed")
    extras = ["tolerance_minutes"] if "tolerance_minutes" in frame.columns else []
    if frame.duplicated(["seed", "session_id", "pair_id", *extras]).any():
        raise CurrentInputAnalysisError("Duplicate paired bootstrap row")
    coverage: list[frozenset[tuple[str, ...]]] = []
    arrays: dict[int, tuple[np.ndarray, np.ndarray, int, int]] = {}
    points: list[float] = []
    for seed in seeds:
        selected = frame[frame["seed"].astype(int).eq(seed)]
        coverage.append(
            frozenset(
                selected[["session_id", "pair_id", *extras]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
        )
        grouped = selected.groupby("session_id", sort=True).agg(
            difference_sum=("difference", "sum"),
            row_count=("difference", "size"),
        )
        sums = grouped["difference_sum"].to_numpy(dtype=float)
        counts = grouped["row_count"].to_numpy(dtype=float)
        arrays[seed] = (sums, counts, len(grouped), len(selected))
        points.append(float(sums.sum() / counts.sum()))
    if len(set(coverage)) != 1:
        raise CurrentInputAnalysisError(
            "Two-level bootstrap requires identical paired seed coverage"
        )
    if any(value[2] != EXPECTED_SESSION_COUNT for value in arrays.values()):
        raise CurrentInputAnalysisError("Bootstrap session count drifted")
    rng = np.random.default_rng(int(random_seed))
    seed_indexes = rng.integers(0, len(seeds), size=(iterations, len(seeds)))
    session_draw_means = np.empty((len(seeds), iterations, len(seeds)), dtype=float)
    for seed_index, seed in enumerate(seeds):
        sums, counts, session_count, _ = arrays[seed]
        indexes = rng.integers(
            0,
            session_count,
            size=(iterations, len(seeds), session_count),
        )
        session_draw_means[seed_index] = sums[indexes].sum(axis=2) / counts[
            indexes
        ].sum(axis=2)
    iteration_indexes = np.arange(iterations)[:, None]
    slot_indexes = np.arange(len(seeds))[None, :]
    draws = session_draw_means[
        seed_indexes,
        iteration_indexes,
        slot_indexes,
    ].mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    p_lower = (float(np.sum(draws <= 0.0)) + 1.0) / (iterations + 1.0)
    p_upper = (float(np.sum(draws >= 0.0)) + 1.0) / (iterations + 1.0)
    return {
        "mean_diff": float(np.mean(points)),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "p_two_sided": float(min(1.0, 2.0 * min(p_lower, p_upper))),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(random_seed),
        "seed_count": len(SEEDS),
        "session_count_per_seed": int(arrays[seeds[0]][2]),
        "pair_rows_per_seed": int(arrays[seeds[0]][3]),
        "resampling_method": "seed_then_paired_CME_session_cluster",
    }


def _difference_frame(paired: pd.DataFrame, column: str) -> pd.DataFrame:
    output = paired[["seed", "tolerance_minutes", "session_id", "pair_id"]].copy()
    output["difference"] = paired[column].to_numpy(dtype=float)
    return output


def build_current_input_bootstrap(
    paired: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Build one combined primary and two Holm-adjusted tolerance tests."""

    base = _difference_frame(paired, "masked_minus_full_current")
    rows: list[dict[str, Any]] = []
    scopes = [("combined_05m_30m", base)] + [
        (
            f"{tolerance:02d}m",
            base[base["tolerance_minutes"].astype(int).eq(tolerance)],
        )
        for tolerance in TOLERANCES
    ]
    for index, (scope, selected) in enumerate(scopes):
        primary = scope == "combined_05m_30m"
        rows.append(
            {
                "comparison": "current_support_masked_minus_full_current",
                "generator_current_input_mode": (
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
                "candidate_generator_current_input_mode": (
                    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                ),
                "reference_generator_current_input_mode": (
                    FULL_CURRENT_GENERATOR_INPUT_MODE
                ),
                "difference_definition": "masked_mae_minus_full_current_mae",
                "negative_means_candidate_better": True,
                "tolerance_scope": scope,
                "analysis_role": "primary" if primary else "secondary_tolerance",
                "holm_family": (
                    "primary_combined_single"
                    if primary
                    else "secondary_05m_30m_two_tests"
                ),
                **_two_level_bootstrap(
                    selected,
                    iterations=iterations,
                    random_seed=random_seed + index,
                ),
            }
        )
    output = pd.DataFrame(rows)
    primary_index = output.index[output["analysis_role"].eq("primary")]
    output.loc[primary_index, "holm_adjusted_p"] = output.loc[
        primary_index, "p_two_sided"
    ]
    secondary_index = output.index[output["analysis_role"].eq("secondary_tolerance")]
    output.loc[secondary_index, "holm_adjusted_p"] = holm_adjust(
        output.loc[secondary_index, "p_two_sided"].tolist()
    )
    return output.reset_index(drop=True)


def build_persistence_bootstrap(
    paired: pd.DataFrame,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    random_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Contrast masked and full-current models separately with persistence."""

    rows: list[dict[str, Any]] = []
    for mode_index, (mode, column) in enumerate(
        (
            (
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
                "masked_minus_persistence",
            ),
            (FULL_CURRENT_GENERATOR_INPUT_MODE, "full_current_minus_persistence"),
        )
    ):
        base = _difference_frame(paired, column)
        scopes = [("combined_05m_30m", base)] + [
            (
                f"{tolerance:02d}m",
                base[base["tolerance_minutes"].astype(int).eq(tolerance)],
            )
            for tolerance in TOLERANCES
        ]
        for scope_index, (scope, selected) in enumerate(scopes):
            combined = scope == "combined_05m_30m"
            rows.append(
                {
                    "comparison": f"{mode}_minus_persistence",
                    "generator_current_input_mode": mode,
                    "reference": "persistence",
                    "difference_definition": f"{column}_mae",
                    "negative_means_model_better": True,
                    "tolerance_scope": scope,
                    "analysis_role": (
                        "secondary_combined" if combined else "secondary_tolerance"
                    ),
                    "holm_family": (
                        f"{mode}_combined_single"
                        if combined
                        else f"{mode}_tolerance_two_tests"
                    ),
                    **_two_level_bootstrap(
                        selected,
                        iterations=iterations,
                        random_seed=(random_seed + 100 + mode_index * 10 + scope_index),
                    ),
                }
            )
    output = pd.DataFrame(rows)
    for (_, role), indexes in output.groupby(
        ["generator_current_input_mode", "analysis_role"], sort=False
    ).groups.items():
        if role == "secondary_tolerance":
            output.loc[indexes, "holm_adjusted_p"] = holm_adjust(
                output.loc[indexes, "p_two_sided"].tolist()
            )
        else:
            output.loc[indexes, "holm_adjusted_p"] = output.loc[indexes, "p_two_sided"]
    return output.sort_values(
        ["generator_current_input_mode", "analysis_role", "tolerance_scope"],
        kind="stable",
    ).reset_index(drop=True)


def _canonical_pair_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_id", "session_id", "persistence_mae"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CurrentInputAnalysisError(f"Validation panel missing {missing}")
    selected = frame[["pair_id", "session_id", "persistence_mae"]].copy()
    selected["pair_id"] = selected["pair_id"].astype(str)
    selected["session_id"] = selected["session_id"].astype(str)
    selected["persistence_mae"] = pd.to_numeric(
        selected["persistence_mae"], errors="coerce"
    )
    if (
        selected.duplicated(["pair_id", "session_id"]).any()
        or not np.isfinite(selected["persistence_mae"].to_numpy(dtype=float)).all()
    ):
        raise CurrentInputAnalysisError("Validation panel keys/values are invalid")
    return selected.sort_values(["session_id", "pair_id"], kind="stable").reset_index(
        drop=True
    )


def _panel_hashes(frame: pd.DataFrame) -> tuple[str, str]:
    selected = _canonical_pair_panel(frame)
    panel_records = (
        selected[["pair_id", "session_id"]].astype(str).to_dict(orient="records")
    )
    persistence_records = [
        {
            "pair_id": str(row.pair_id),
            "session_id": str(row.session_id),
            "persistence_mae": format(float(row.persistence_mae), ".17g"),
        }
        for row in selected.itertuples(index=False)
    ]
    return _payload_sha256(panel_records), _payload_sha256(persistence_records)


def _validation_lineage(
    masked: pd.DataFrame,
    full: pd.DataFrame,
) -> dict[str, Any]:
    canonical = _canonical_pair_panel(
        full[
            full["seed"].astype(int).eq(SEEDS[0])
            & full["tolerance_minutes"].astype(int).eq(TOLERANCES[0])
        ]
    )
    if len(canonical) != EXPECTED_PAIR_COUNT:
        raise CurrentInputAnalysisError("Canonical validation cell is incomplete")
    canonical_panel_sha, canonical_persistence_sha = _panel_hashes(canonical)
    canonical_keys = canonical[["pair_id", "session_id"]]
    canonical_persistence = canonical["persistence_mae"].to_numpy(dtype=float)
    cells: list[dict[str, Any]] = []
    panel_hashes: set[str] = set()
    observed_persistence_hashes: set[str] = set()
    maximum_persistence_difference = 0.0
    for mode, frame in (
        (CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE, masked),
        (FULL_CURRENT_GENERATOR_INPUT_MODE, full),
    ):
        for (seed, tolerance), group in frame.groupby(
            ["seed", "tolerance_minutes"], sort=True
        ):
            observed = _canonical_pair_panel(group)
            panel_sha, observed_persistence_sha = _panel_hashes(observed)
            panel_hashes.add(panel_sha)
            observed_persistence_hashes.add(observed_persistence_sha)
            if len(observed) != len(canonical) or not observed[
                ["pair_id", "session_id"]
            ].equals(canonical_keys):
                raise CurrentInputAnalysisError(
                    "Validation pair/session panel differs by cell: "
                    f"{mode}/seed={int(seed)}/tolerance={int(tolerance)}"
                )
            persistence_difference = np.abs(
                observed["persistence_mae"].to_numpy(dtype=float)
                - canonical_persistence
            )
            cell_maximum = float(persistence_difference.max(initial=0.0))
            maximum_persistence_difference = max(
                maximum_persistence_difference, cell_maximum
            )
            if cell_maximum > PERSISTENCE_MATCH_ATOL:
                raise CurrentInputAnalysisError(
                    "Validation persistence differs by more than "
                    f"{PERSISTENCE_MATCH_ATOL:.1e}: "
                    f"{mode}/seed={int(seed)}/tolerance={int(tolerance)} "
                    f"max_abs_diff={cell_maximum:.17g}"
                )
            cells.append(
                {
                    "generator_current_input_mode": mode,
                    "seed": int(seed),
                    "tolerance_minutes": int(tolerance),
                    "pair_count": int(group["pair_id"].nunique()),
                    "session_count": int(group["session_id"].nunique()),
                    "pair_session_panel_sha256": panel_sha,
                    "persistence_vector_sha256": canonical_persistence_sha,
                    "observed_persistence_vector_sha256": (observed_persistence_sha),
                    "max_abs_persistence_diff_vs_canonical": cell_maximum,
                }
            )
    if panel_hashes != {canonical_panel_sha}:
        raise CurrentInputAnalysisError("Validation panel hashes differ by cell")
    return {
        "schema_version": 1,
        "panel": SELECTION_PANEL,
        "interval_start_utc": Q3_START_UTC,
        "interval_end_utc_exclusive": Q3_END_UTC,
        "pair_count": EXPECTED_PAIR_COUNT,
        "session_count": EXPECTED_SESSION_COUNT,
        "pair_session_panel_sha256": canonical_panel_sha,
        "persistence_vector_sha256": canonical_persistence_sha,
        "persistence_match_atol": PERSISTENCE_MATCH_ATOL,
        "persistence_match_rtol": 0.0,
        "max_abs_persistence_diff_vs_canonical": maximum_persistence_difference,
        "observed_persistence_vector_sha256_count": len(observed_persistence_hashes),
        "persistence_canonical_cell": {
            "generator_current_input_mode": FULL_CURRENT_GENERATOR_INPUT_MODE,
            "seed": SEEDS[0],
            "tolerance_minutes": TOLERANCES[0],
            "canonical_sort": ["session_id", "pair_id"],
            "canonical_float_format": ".17g",
        },
        "cells": cells,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "q4_used_for_checkpoint_selection": False,
    }


def _fixture_selection_lineage() -> pd.DataFrame:
    rows = []
    for mode in (
        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        FULL_CURRENT_GENERATOR_INPUT_MODE,
    ):
        for seed in SEEDS:
            for tolerance in TOLERANCES:
                rows.append(
                    {
                        "generator_current_input_mode": mode,
                        "generator_current_input_fingerprint": (
                            generator_current_input_fingerprint(mode)
                        ),
                        "job_id": f"fixture_{mode}_{seed}_{tolerance}",
                        "seed": seed,
                        "tolerance_minutes": tolerance,
                        "best_learned_epoch": 1,
                        "selection_scope": "trained_epochs_only",
                        "selection_metadata_path": "fixture",
                        "selection_metadata_sha256": "fixture",
                        "generator_checkpoint_path": "fixture",
                        "generator_checkpoint_sha256": "fixture",
                    }
                )
    return pd.DataFrame(rows)


def _validate_analysis_mode_keys(
    *,
    masked: pd.DataFrame,
    full: pd.DataFrame,
    paired: pd.DataFrame,
    cell_summary: pd.DataFrame,
    cross_seed: pd.DataFrame,
    combined: pd.DataFrame,
    comparison: pd.DataFrame,
    persistence: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    """Fail closed if an exported analytical grain drops its input-mode key."""

    candidate_frames = {
        "masked pairs": masked,
        "paired pairs": paired,
        "cell summary": cell_summary,
        "cross-seed summary": cross_seed,
        "combined summary": combined,
        "masked/full bootstrap": comparison,
    }
    for label, frame in candidate_frames.items():
        if "generator_current_input_mode" not in frame.columns or set(
            frame["generator_current_input_mode"].astype(str)
        ) != {CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE}:
            raise CurrentInputAnalysisError(
                f"{label} dropped the masked input-mode key"
            )
    if "generator_current_input_mode" not in full.columns or set(
        full["generator_current_input_mode"].astype(str)
    ) != {FULL_CURRENT_GENERATOR_INPUT_MODE}:
        raise CurrentInputAnalysisError(
            "Full-current pairs dropped their input-mode key"
        )
    expected_modes = {
        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        FULL_CURRENT_GENERATOR_INPUT_MODE,
    }
    for label, frame in (
        ("persistence bootstrap", persistence),
        ("selection lineage", selection),
    ):
        if (
            "generator_current_input_mode" not in frame.columns
            or set(frame["generator_current_input_mode"].astype(str)) != expected_modes
        ):
            raise CurrentInputAnalysisError(f"{label} dropped an input-mode key")


def run_current_input_analysis(
    experiment_root: str | Path,
    *,
    reference_root: str | Path = DEFAULT_REFERENCE_ROOT,
    masked_pair_metrics: pd.DataFrame | None = None,
    full_current_pair_metrics: pd.DataFrame | None = None,
    evaluator: Callable[[RunSpec, str, pd.DataFrame], pd.DataFrame] | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Path:
    """Run the complete Q3-only masked-versus-full paired analysis."""

    if int(bootstrap_iterations) != DEFAULT_BOOTSTRAP_ITERATIONS:
        raise CurrentInputAnalysisError("Formal analysis is frozen to 10,000 draws")
    injected = masked_pair_metrics is not None or full_current_pair_metrics is not None
    if injected and (masked_pair_metrics is None or full_current_pair_metrics is None):
        raise CurrentInputAnalysisError(
            "Fixture injection requires both masked and full-current pair metrics"
        )
    root = Path(experiment_root).resolve(strict=False)
    analysis_dir = root / "analysis" / "current_input_ablation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if injected:
        masked = validate_mode_pair_metrics(
            masked_pair_metrics,
            expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        )
        full = validate_mode_pair_metrics(
            full_current_pair_metrics,
            expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
        )
        selection = _fixture_selection_lineage()
        panel_lineage: dict[str, Any] = {
            "fixture_injected": True,
            "interval_start_utc": Q3_START_UTC,
            "interval_end_utc_exclusive": Q3_END_UTC,
            "q4_rows_passed_to_evaluator": 0,
            "q4_predictions_generated": False,
            "q4_evaluated": False,
        }
        reference_lineage: dict[str, Any] = {"fixture_injected": True}
        registry_sha = "fixture"
    else:
        # Validate the already-frozen in-root reference before inspecting new
        # jobs; this prevents accidental latest-directory reference discovery.
        full, full_selection, reference_lineage = load_full_current_reference_evidence(
            root,
            reference_root=reference_root,
        )
        masked, panel_lineage, masked_selection = evaluate_masked_q3_pair_metrics(
            root,
            evaluator=evaluator,
            prediction_artifact_dir=analysis_dir / "q3_predictions",
        )
        selection = pd.concat(
            [masked_selection, full_selection], ignore_index=True, sort=False
        )
        registry_sha = _sha256_file(root / "registry" / "jobs.json")
    paired = build_paired_pair_metrics(masked, full)
    cell_summary = build_cell_summary(paired)
    cross_seed = build_cross_seed_summary(cell_summary)
    combined = build_combined_summary(cell_summary)
    comparison = build_current_input_bootstrap(
        paired,
        iterations=bootstrap_iterations,
        random_seed=bootstrap_seed,
    )
    persistence = build_persistence_bootstrap(
        paired,
        iterations=bootstrap_iterations,
        random_seed=bootstrap_seed,
    )
    if (
        len(comparison) != 3
        or len(comparison[comparison["analysis_role"].eq("primary")]) != 1
    ):
        raise CurrentInputAnalysisError("Primary/secondary comparison family drifted")
    if len(persistence) != 6:
        raise CurrentInputAnalysisError("Persistence comparison family drifted")
    validation_lineage = _validation_lineage(masked, full)
    selection = selection.sort_values(
        ["generator_current_input_mode", "seed", "tolerance_minutes"],
        kind="stable",
    ).reset_index(drop=True)
    _validate_analysis_mode_keys(
        masked=masked,
        full=full,
        paired=paired,
        cell_summary=cell_summary,
        cross_seed=cross_seed,
        combined=combined,
        comparison=comparison,
        persistence=persistence,
        selection=selection,
    )

    paths: dict[str, Path] = {
        "masked_q3_pair_metrics": _write_csv(
            masked,
            analysis_dir / "masked_q3_pair_metrics.csv.gz",
            compression="gzip",
        ),
        "full_current_q3_pair_metrics": _write_csv(
            full,
            analysis_dir / "full_current_q3_pair_metrics.csv.gz",
            compression="gzip",
        ),
        "paired_q3_pair_metrics": _write_csv(
            paired,
            analysis_dir / "current_input_paired_q3_pair_metrics.csv.gz",
            compression="gzip",
        ),
        "cell_summary": _write_csv(
            cell_summary, analysis_dir / "current_input_cell_summary.csv"
        ),
        "cross_seed_summary": _write_csv(
            cross_seed, analysis_dir / "current_input_cross_seed_summary.csv"
        ),
        "combined_summary": _write_csv(
            combined, analysis_dir / "current_input_combined_summary.csv"
        ),
        "masked_vs_full_bootstrap": _write_csv(
            comparison, analysis_dir / "current_input_masked_vs_full_bootstrap.csv"
        ),
        "models_vs_persistence_bootstrap": _write_csv(
            persistence,
            analysis_dir / "current_input_models_vs_persistence_bootstrap.csv",
        ),
        "selection_lineage": _write_csv(
            selection, analysis_dir / "current_input_selection_lineage.csv"
        ),
    }
    validation_lineage["selection_lineage_sha256"] = _sha256_file(
        paths["selection_lineage"]
    )
    validation_lineage["selection_rows"] = int(len(selection))
    validation_lineage["validation_lineage_sha256"] = _payload_sha256(
        validation_lineage
    )
    paths["validation_lineage"] = _write_json(
        analysis_dir / "current_input_validation_lineage.json",
        validation_lineage,
    )
    for artifact in panel_lineage.get("prediction_artifacts", []):
        paths[str(artifact["role"])] = Path(str(artifact["path"]))
    primary = comparison[comparison["analysis_role"].eq("primary")].iloc[0]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "generator_current_input_q3_paired_ablation",
        "masked_generator_current_input_mode": (
            CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
        ),
        "masked_generator_current_input_fingerprint": (
            generator_current_input_fingerprint(
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            )
        ),
        "reference_generator_current_input_mode": FULL_CURRENT_GENERATOR_INPUT_MODE,
        "reference_generator_current_input_fingerprint": (
            generator_current_input_fingerprint(FULL_CURRENT_GENERATOR_INPUT_MODE)
        ),
        "generator_input_change": "encoder_surface_only",
        "masked_encoder_input": "current_surface * current_support_mask",
        "mask_channel_added": False,
        "residual_anchor": "original_unmasked_full_current_surface",
        "critic_and_loss_mask": "raw_joint",
        "capacity_profile": CAPACITY_PROFILE,
        "parameter_count": PARAMETER_COUNT,
        "learning_rate": FIXED_LEARNING_RATE,
        "text_ablation_mode": TEXT_MODE,
        "seeds": list(SEEDS),
        "tolerances_minutes": list(TOLERANCES),
        "q3_pair_count_per_cell": EXPECTED_PAIR_COUNT,
        "q3_session_count_per_cell": EXPECTED_SESSION_COUNT,
        "masked_pair_rows": int(len(masked)),
        "full_current_pair_rows": int(len(full)),
        "paired_rows": int(len(paired)),
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_layers": ["training_seed", "CME_session"],
        "bootstrap_method": "seed_then_paired_CME_session_cluster",
        "primary_family": "one combined 5m/30m masked-minus-full comparison",
        "secondary_family": "5m and 30m masked-minus-full comparisons; Holm over 2",
        "reference_evidence_design": "historical_immutable_coverage_experiment",
        "reference_retrained_concurrently": False,
        "reference_exact_pairing_controls": (
            "training_config_epoch0_tensors_q3_panel_and_artifact_hashes"
        ),
        "historical_reference_residual_confounding": (
            "historical_code_and_concurrent_runtime_environment_may_differ"
        ),
        "q3_workbook_materialization": (
            "full_sheet_materialized_then_q3_filtered_before_evaluator"
        ),
        "primary_result": primary.to_dict(),
        "panel_lineage": panel_lineage,
        "reference_lineage": reference_lineage,
        "masked_registry_sha256": registry_sha,
        "selection_lineage_sha256": _sha256_file(paths["selection_lineage"]),
        "validation_lineage_sha256": _sha256_file(paths["validation_lineage"]),
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "q4_used_for_checkpoint_selection": False,
        "artifacts": {
            role: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for role, path in paths.items()
        },
        "created_at_utc": _utc_now(),
    }
    summary["analysis_sha256"] = _payload_sha256(summary)
    summary_path = _write_json(
        analysis_dir / "current_input_analysis_summary.json", summary
    )
    validation = {
        "schema_version": 1,
        "status": "pass",
        "analysis_sha256": summary["analysis_sha256"],
        "selection_lineage_sha256": summary["selection_lineage_sha256"],
        "validation_lineage_sha256": summary["validation_lineage_sha256"],
        "generator_current_input_mode_is_in_all_analysis_keys": True,
        "pair_rows_per_mode": EXPECTED_PAIR_ROWS,
        "pairs_per_cell": EXPECTED_PAIR_COUNT,
        "sessions_per_cell": EXPECTED_SESSION_COUNT,
        "primary_comparison_count": 1,
        "secondary_tolerance_comparison_count": 2,
        "bootstrap_iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "validated_at_utc": _utc_now(),
    }
    validation["validation_sha256"] = _payload_sha256(validation)
    _write_json(analysis_dir / "current_input_validation_summary.json", validation)
    return summary_path


__all__ = [
    "CurrentInputAnalysisError",
    "DEFAULT_REFERENCE_ROOT",
    "build_cell_summary",
    "build_combined_summary",
    "build_cross_seed_summary",
    "build_current_input_bootstrap",
    "build_paired_pair_metrics",
    "build_persistence_bootstrap",
    "evaluate_masked_q3_pair_metrics",
    "freeze_full_current_reference_manifest",
    "load_full_current_reference_evidence",
    "run_current_input_analysis",
    "validate_mode_pair_metrics",
]
