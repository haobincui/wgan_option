"""One-cell ungated low-projection-LR supplement for the FiLM probe."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from scripts.rq3 import news_first_vol_film_unet_text_signal_probe as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    "configs/rq3/news_first_vol_film_unet_ungated_proj_lr5e7_seed42_f2.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_film_unet_ungated_proj_lr5e7_seed42_f2_5m_v1"
)
EXPERIMENT_KIND = "film_unet_ungated_proj_lr5e7_seed42_f2_5m_probe_v1"
REGISTRY_KIND = "film_unet_ungated_proj_lr5e7_probe_task_registry_v1"
TERMINAL_QA_KIND = "film_unet_ungated_proj_lr5e7_probe_terminal_qa_v1"
INTERPRETATION = "single_seed_mechanism_validation"
JOB_ID = "ungated_global_proj_lr5e7"
ACTIONS = (
    "prepare",
    "canary",
    "launch",
    "analyze",
    "report",
    "qa",
    "status",
    "run-pipeline",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = _resolve(path)
    config = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "config")
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = base._sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    source = _mapping(config.get("source_probe"), "source_probe")
    comparators = _mapping(config.get("frozen_comparators"), "comparators")
    data = _mapping(config.get("data_contract"), "data_contract")
    model = _mapping(config.get("model_contract"), "model_contract")
    training = _mapping(config.get("training"), "training")
    analysis = _mapping(config.get("analysis"), "analysis")
    runtime = _mapping(config.get("runtime"), "runtime")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    if experiment.get("interpretation") != INTERPRETATION:
        raise ValueError("Interpretation drift")
    if (data.get("tolerance_minutes"), data.get("fold_id"), data.get("seed")) != (
        5,
        "f2_2023q2",
        42,
    ):
        raise ValueError("Supplement must remain seed42/f2/5m")
    if tuple(
        int(data[key])
        for key in (
            "train_pairs",
            "train_sessions",
            "validation_pairs",
            "validation_sessions",
        )
    ) != (526, 133, 110, 34):
        raise ValueError("Supplement pair/session universe drift")
    if data.get("allowed_partitions") != ["train", "validation"]:
        raise ValueError("Only train/validation partitions are permitted")
    if int(data.get("test_loader_count", -1)) or int(data.get("q3_q4_input_rows", -1)):
        raise ValueError("Test and Q3/Q4 access must remain zero")
    expected_model = {
        "generator_conditioning_mode": "film_unet_mask_coords_v1",
        "critic_conditioning_mode": "lp_disabled_same_shape_v1",
        "capacity_profile": "c32_text128",
        "factor_code": "1000",
        "spatial_rank": 0,
        "gate_enabled": False,
        "generator_parameters": 827_745,
        "critic_parameters": 729_157,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model_contract.{key} drift")
    expected_training = {
        "job_id": JOB_ID,
        "epochs": 60,
        "physical_gpu_id": 0,
        "text_encoder_learning_rate": 2.5e-6,
        "global_film_learning_rate": 5.0e-7,
        "early_stopping": False,
        "scheduler": False,
        "checkpoint_selection": "final_epoch",
        "shared_initialization": True,
        "shared_batch_order": True,
        "shared_noise_bank": True,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} drift")
    if int(analysis.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Analysis requires 10,000 bootstrap replicates")
    expected_analysis = {
        "confidence_level": 0.95,
        "bootstrap_seed": 20260831,
        "primary_metric": "log_mae_matched_over_zero",
        "secondary_metric": "log_mae_matched_over_wrong",
        "retention_metric": "positive_maximin_magnitude_relative_to_control_v1",
        "interpretation": INTERPRETATION,
    }
    for key, expected in expected_analysis.items():
        if analysis.get(key) != expected:
            raise ValueError(f"analysis.{key} drift")
    if (
        not 0.0
        < float(analysis["low_lr_dominant_maximum_retention"])
        < float(analysis["gate_dominant_minimum_retention"])
        < 1.0
    ):
        raise ValueError("Diagnostic retention thresholds drift")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0,):
        raise ValueError("Supplement must use physical GPU0 like the control")
    expected_runtime = {
        "workers_per_gpu": 1,
        "cpu_threads_per_job": 1,
        "poll_interval_seconds": 1,
        "resource_sample_interval_seconds": 2,
        "maximum_peak_gpu_memory_gib": 20.0,
        "maximum_host_ram_fraction": 0.85,
        "canary_epochs": 1,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise ValueError(f"runtime.{key} drift")
    for key in (
        "input_manifest_path",
        "resolved_config_path",
        "terminal_qa_path",
    ):
        if not _resolve(str(source[key])).is_file():
            raise FileNotFoundError(_resolve(str(source[key])))
    for role in (
        "task_registry",
        "terminal_qa",
        "output_hashes",
        "control",
        "gated_lr5e7",
    ):
        row = _mapping(comparators.get(role), f"comparators.{role}")
        path_keys = [key for key in row if key.endswith("path")]
        if not path_keys:
            raise ValueError(f"Comparator {role} has no path binding")
        for path_key in path_keys:
            sha_key = path_key.replace("path", "sha256")
            if (
                not _resolve(str(row[path_key])).is_file()
                or len(str(row[sha_key])) != 64
            ):
                raise ValueError(f"Comparator binding drift: {role}.{path_key}")


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _write_registry(root: Path, payload: Mapping[str, Any]) -> Path:
    value = dict(payload)
    value["updated_at_utc"] = base._utc_now()
    return base._write_json(_registry_path(root), base._signed_payload(value))


def _read_registry(root: Path) -> dict[str, Any]:
    value = base._read_json(_registry_path(root))
    base._verify_signed_payload(value, "ungated-LR task registry")
    if value.get("kind") != REGISTRY_KIND:
        raise ValueError("Ungated-LR registry kind drift")
    return value


def _completed_registry(
    root: Path, *, resume: bool, action: str
) -> dict[str, Any] | None:
    if not _registry_path(root).is_file():
        return None
    registry = _read_registry(root)
    if registry.get("status") != "completed":
        return None
    if not resume:
        raise FileExistsError(f"Completed root requires --resume for {action}")
    base._verify_output_hash_manifest(root / "qa/output_hashes.csv", root)
    return registry


def _bound_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = _mapping(config["source_probe"], "source_probe")
    for label in ("input_manifest", "resolved_config", "terminal_qa"):
        path = _resolve(str(source[f"{label}_path"]))
        if base._sha256_file(path) != str(source[f"{label}_sha256"]):
            raise ValueError(f"Frozen source {label} SHA drift")
        rows.append(base._file_row(f"source_{label}", path))
    comparators = _mapping(config["frozen_comparators"], "comparators")
    for role in (
        "task_registry",
        "terminal_qa",
        "output_hashes",
        "control",
        "gated_lr5e7",
    ):
        value = _mapping(comparators[role], f"comparators.{role}")
        for path_key in sorted(key for key in value if key.endswith("path")):
            path = _resolve(str(value[path_key]))
            expected = str(value[path_key.replace("path", "sha256")])
            if base._sha256_file(path) != expected:
                raise ValueError(f"Frozen comparator SHA drift: {role}.{path_key}")
            rows.append(base._file_row(f"{role}_{path_key}", path))
    terminal = base._read_json(_resolve(str(comparators["terminal_qa"]["path"])))
    if terminal.get("status") != "passed":
        raise ValueError("Frozen comparator terminal QA is not passing")
    base._verify_output_hash_manifest(
        _resolve(str(comparators["output_hashes"]["path"])),
        _resolve(str(comparators["experiment_root"])),
    )
    input_manifest = base._read_json(_resolve(str(source["input_manifest_path"])))
    if input_manifest.get("test_loader_materialized") is not False:
        raise ValueError("Frozen source input materialized a test loader")
    return rows


def _source_manifest(root: Path, config: Mapping[str, Any]) -> Path:
    rows = _bound_rows(config)
    for role, path in (
        ("source_config", config["source_config_path"]),
        ("orchestrator", Path(__file__).resolve()),
        (
            "base_orchestrator",
            REPO_ROOT / "scripts/rq3/news_first_vol_film_unet_text_signal_probe.py",
        ),
        (
            "worker_core",
            REPO_ROOT
            / "scripts/rq3/news_first_vol_film_unet_text_signal_probe_core.py",
        ),
        (
            "adapter_model",
            REPO_ROOT / "src/wgan_option/models/text_signal_adapter.py",
        ),
    ):
        rows.append(base._file_row(role, path))
    return base._write_json(
        root / "contracts/source_manifest.json",
        base._signed_payload(
            {
                "schema_version": 1,
                "kind": "film_unet_ungated_proj_lr5e7_source_manifest_v1",
                "rows": rows,
            }
        ),
    )


def _verify_source_manifest(root: Path) -> dict[str, Any]:
    value = base._read_json(root / "contracts/source_manifest.json")
    base._verify_signed_payload(value, "ungated-LR source manifest")
    for row in value.get("rows") or []:
        base._verify_file_row(_mapping(row, "source row"), "source row")
    return value


def _build_record(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(config["source_probe"], "source_probe")
    model = _mapping(config["model_contract"], "model_contract")
    training = _mapping(config["training"], "training")
    source_config = base._read_json(_resolve(str(source["resolved_config_path"])))
    output_dir = (root / "runs/screen" / JOB_ID).resolve()
    payload = {
        "schema_version": 1,
        "kind": base.JOB_SPEC_KIND,
        "job_id": JOB_ID,
        "profile_id": JOB_ID,
        "stage": "screen",
        "seed": 42,
        "fold_id": "f2_2023q2",
        "tolerance_minutes": 5,
        "factor_code": "1000",
        "text_assignment": "matched",
        "text_output_dimension": 128,
        "spatial_rank": 0,
        "physical_gpu_id": 0,
        "max_epochs": 60,
        "output_dir": str(output_dir),
        "evaluation_partition": "validation",
        "allowed_partitions": ["train", "validation"],
        "core_module": base.CORE_MODULE,
        "input_manifest_path": str(_resolve(str(source["input_manifest_path"]))),
        "input_manifest_sha256": str(source["input_manifest_sha256"]),
        "calibrate_auxiliary_weight": False,
        "alignment_auxiliary_weight": 0.0,
        "expected_generator_parameters": int(model["generator_parameters"]),
        "text_encoder_learning_rate": float(training["text_encoder_learning_rate"]),
        "global_film_learning_rate": float(training["global_film_learning_rate"]),
        "baseline_generator": deepcopy(
            source_config["baseline"]["generator_checkpoint"]
        ),
        "baseline_critic": deepcopy(source_config["baseline"]["critic_checkpoint"]),
        "training_contract": deepcopy(training),
        "model_contract": deepcopy(model),
        "test_loader_count": 0,
        "q3_q4_input_rows": 0,
    }
    spec_path, spec_sha = base._write_job_spec(
        root / "registry/job_specs/screen" / f"{JOB_ID}.json", payload
    )
    return {
        "job_id": JOB_ID,
        "profile_id": JOB_ID,
        "stage": "screen",
        "factor_code": "1000",
        "physical_gpu_id": 0,
        "gate_enabled": False,
        "job_spec_path": str(spec_path.resolve()),
        "job_spec_sha256": spec_sha,
        "output_dir": str(output_dir),
    }


def _validate_prepared(root: Path, config: Mapping[str, Any]) -> None:
    if base._read_json(root / "contracts/resolved_config.json") != dict(config):
        raise ValueError("Resolved supplement config drift")
    _bound_rows(config)
    _verify_source_manifest(root)
    registry = _read_registry(root)
    record = _mapping(registry.get("job"), "registry job")
    if record.get("job_id") != JOB_ID:
        raise ValueError("Supplement registry job drift")
    base._verify_job_spec(record["job_spec_path"], record["job_spec_sha256"])


def prepare(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        if _registry_path(root).is_file():
            if not resume:
                raise FileExistsError(f"Supplement already prepared: {root}")
            _validate_prepared(root, config)
            if _completed_registry(root, resume=True, action="prepare") is not None:
                return root
            base._journal(root, "prepare", "completed", resumed=True)
            return root
        base._journal(root, "prepare", "running")
        if root.exists() and any(root.iterdir()) and not resume:
            raise FileExistsError(f"Non-empty root requires --resume: {root}")
        for relative in (
            "contracts",
            "registry/job_specs/screen",
            "registry/job_status",
            "runs/screen",
            "analysis",
            "report",
            "qa",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        base._write_json(root / "contracts/resolved_config.json", config)
        source_manifest = _source_manifest(root, config)
        record = _build_record(root, config)
        _write_registry(
            root,
            {
                "schema_version": 1,
                "kind": REGISTRY_KIND,
                "experiment_kind": EXPERIMENT_KIND,
                "interpretation": INTERPRETATION,
                "status": "prepared",
                "root": str(root),
                "job": record,
                "job_count": 1,
                "source_manifest_path": str(source_manifest.resolve()),
                "source_manifest_sha256": base._sha256_file(source_manifest),
                "partitions_materialized": ["train", "validation"],
                "test_loader_count": 0,
                "prediction_count": 0,
                "q3_q4_input_rows": 0,
                "created_at_utc": base._utc_now(),
            },
        )
        _validate_prepared(root, config)
        base._journal(root, "prepare", "completed", jobs=1)
        return root


def _canary_path(root: Path) -> Path:
    return base._control_dir(root) / "canary_result.json"


def _canary_record(
    root: Path, record: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    canary_root = base._control_dir(root) / "canary"
    formal_spec = base._verify_job_spec(
        record["job_spec_path"], record["job_spec_sha256"]
    )
    output_dir = (canary_root / "runs" / f"canary_{JOB_ID}").resolve()
    payload = {
        **formal_spec,
        "job_id": f"canary_{JOB_ID}",
        "stage": "canary",
        "max_epochs": 1,
        "output_dir": str(output_dir),
    }
    payload.pop("job_spec_sha256", None)
    spec_path, spec_sha = base._write_job_spec(
        canary_root / "job_specs" / f"canary_{JOB_ID}.json", payload
    )
    return canary_root, {
        **dict(record),
        "job_id": f"canary_{JOB_ID}",
        "stage": "canary",
        "job_spec_path": str(spec_path.resolve()),
        "job_spec_sha256": spec_sha,
        "output_dir": str(output_dir),
    }


def _check_resources(summary: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    runtime = _mapping(config["runtime"], "runtime")
    if float(summary["peak_host_ram_fraction"]) >= float(
        runtime["maximum_host_ram_fraction"]
    ):
        raise RuntimeError("Host-RAM gate failed")
    if max(map(float, summary["peak_gpu_memory_gib"].values())) >= float(
        runtime["maximum_peak_gpu_memory_gib"]
    ):
        raise RuntimeError("GPU-memory gate failed")


def _verify_canary(root: Path, path: Path) -> dict[str, Any]:
    value = base._read_json(path)
    base._verify_signed_payload(value, "ungated-LR canary")
    canary_root = Path(str(value.get("canary_root", ""))).resolve()
    record = _mapping(value.get("job"), "canary job")
    if value.get("status") != "passed" or not base._completed_valid(
        record, canary_root
    ):
        raise ValueError("Frozen supplement canary drifted")
    if canary_root != base._control_dir(root) / "canary":
        raise ValueError("Canary root drift")
    registry = _read_registry(root)
    formal_spec = base._verify_job_spec(
        registry["job"]["job_spec_path"], registry["job"]["job_spec_sha256"]
    )
    canary_spec = base._verify_job_spec(
        record["job_spec_path"], record["job_spec_sha256"]
    )
    expected = {
        **formal_spec,
        "job_id": f"canary_{JOB_ID}",
        "stage": "canary",
        "max_epochs": 1,
        "output_dir": str((canary_root / "runs" / f"canary_{JOB_ID}").resolve()),
    }
    expected.pop("job_spec_sha256", None)
    observed = dict(canary_spec)
    observed.pop("job_spec_sha256", None)
    if observed != expected:
        raise ValueError("Canary spec is not the exact one-epoch formal-spec variant")
    return value


def canary(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        completed = _completed_registry(root, resume=resume, action="canary")
        if completed is not None:
            path = Path(str(completed["canary_path"])).resolve()
            if base._sha256_file(path) != completed.get("canary_sha256"):
                raise ValueError("Frozen canary binding drift")
            _verify_canary(root, path)
            return path
        path = _canary_path(root)
        if path.is_file():
            _verify_canary(root, path)
            registry = _read_registry(root)
            registry.update(
                status="canary_passed",
                canary_path=str(path.resolve()),
                canary_sha256=base._sha256_file(path),
            )
            _write_registry(root, registry)
            return path
        base._journal(root, "canary", "running")
        registry = _read_registry(root)
        canary_root, record = _canary_record(root, registry["job"])
        summary = base._launch_records(
            [record],
            canary_root,
            config,
            workers_per_gpu=1,
            resume=resume,
        )
        _check_resources(summary, config)
        diagnostics = base._read_json(Path(record["output_dir"]) / "diagnostics.json")
        reference = base._read_json(
            _resolve(config["frozen_comparators"]["control"]["diagnostics_path"])
        )
        if diagnostics["initial_parameter_group_sha256"] != {
            key: reference["initial_parameter_group_sha256"][key]
            for key in ("text_encoder", "global_film")
        }:
            raise ValueError("Canary does not share the frozen control initialization")
        if (
            diagnostics["initial_parameter_sha256"]
            != reference["initial_parameter_sha256"]
        ):
            raise ValueError("Canary full initialization SHA drift")
        if int(diagnostics["parameters"]["generator_total"]) != 827_745:
            raise ValueError("Canary parameter-count drift")
        if diagnostics["final_global_gate_values"]:
            raise ValueError("Ungated canary unexpectedly contains gate values")
        value = base._signed_payload(
            {
                "schema_version": 1,
                "kind": "film_unet_ungated_proj_lr5e7_canary_v1",
                "status": "passed",
                "canary_root": str(canary_root.resolve()),
                "job": record,
                "resources": summary,
                "test_loader_count": 0,
                "q3_q4_input_rows": 0,
                "completed_at_utc": base._utc_now(),
            }
        )
        base._write_json(path, value)
        registry.update(
            status="canary_passed",
            canary_path=str(path.resolve()),
            canary_sha256=base._sha256_file(path),
        )
        _write_registry(root, registry)
        base._journal(root, "canary", "completed", jobs=1)
        return path


def launch(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        if _completed_registry(root, resume=resume, action="launch") is not None:
            return root / "resource_summary_screen.json"
        canary_path = _canary_path(root)
        registry = _read_registry(root)
        if registry.get("canary_path") != str(canary_path.resolve()) or registry.get(
            "canary_sha256"
        ) != base._sha256_file(canary_path):
            raise ValueError("Registered canary binding drift")
        _verify_canary(root, canary_path)
        base._journal(root, "launch", "running")
        summary = base._launch_records(
            [registry["job"]],
            root,
            config,
            workers_per_gpu=1,
            resume=resume,
        )
        _check_resources(summary, config)
        if int(summary["completed_jobs"]) != 1:
            raise RuntimeError("Supplement training did not complete")
        registry.update(status="trained", resource_summary=summary)
        _write_registry(root, registry)
        base._journal(root, "launch", "completed", jobs=1)
        return root / "resource_summary_screen.json"


def _metrics_path(record: Mapping[str, Any]) -> Path:
    spec = base._verify_job_spec(record["job_spec_path"], record["job_spec_sha256"])
    result = base._verify_result(spec, Path(str(record["output_dir"])))
    for artifact in result["artifacts"]:
        if artifact.get("role") == "validation_pair_metrics":
            return Path(str(artifact["path"])).resolve()
    raise ValueError("Supplement result lacks validation metrics")


def _session_mae(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    sessions = frame.groupby("session_id", sort=True)[
        ["mae_matched", "mae_wrong", "mae_zero"]
    ].mean()
    return {column: float(sessions[column].mean()) for column in sessions}


def _effect_stats(
    path: Path,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    probe_core = base._load_core()
    sessions = probe_core._session_frame(path)
    matched_zero = probe_core._bootstrap_ratio(
        sessions,
        "mae_matched",
        "mae_zero",
        replicates=replicates,
        confidence=confidence,
        seed=seed,
    )
    matched_wrong = probe_core._bootstrap_ratio(
        sessions,
        "mae_matched",
        "mae_wrong",
        replicates=replicates,
        confidence=confidence,
        seed=seed + 1,
    )
    constraint_increases = {
        name: float(
            sessions[f"{name}_matched"].mean() - sessions[f"{name}_zero"].mean()
        )
        for name in ("calendar_violation", "butterfly_violation")
    }
    return {
        "session_mae": _session_mae(path),
        "matched_vs_zero": matched_zero,
        "matched_vs_wrong": matched_wrong,
        "constraint_violation_rate_increases": constraint_increases,
        "maximin": max(matched_zero["estimate"], matched_wrong["estimate"]),
    }


def _bootstrap_effect_contrast(
    focal_path: Path,
    reference_path: Path,
    *,
    denominator: str,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    focal = (
        pd.read_csv(focal_path)
        .groupby("session_id", sort=True)[["mae_matched", denominator]]
        .mean()
    )
    reference = (
        pd.read_csv(reference_path)
        .groupby("session_id", sort=True)[["mae_matched", denominator]]
        .mean()
    )
    if not focal.index.equals(reference.index) or len(focal) != 34:
        raise ValueError("Effect contrast requires the same 34 sessions")
    estimate = math.log(
        focal.mae_matched.mean() / focal[denominator].mean()
    ) - math.log(reference.mae_matched.mean() / reference[denominator].mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    count = len(focal)
    focal_values = focal.to_numpy(dtype=np.float64)
    reference_values = reference.to_numpy(dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, count, size=count)
        left = focal_values[sampled].mean(axis=0)
        right = reference_values[sampled].mean(axis=0)
        draws[index] = math.log(left[0] / left[1]) - math.log(right[0] / right[1])
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "replicates": replicates,
        "confidence_level": confidence,
    }


def _classify_retention(
    retention: float,
    *,
    low_lr_maximum: float,
    gate_minimum: float,
) -> str:
    if not math.isfinite(retention) or retention < 0.0:
        raise ValueError("Text-effect retention must be finite and nonnegative")
    if not 0.0 <= low_lr_maximum < gate_minimum <= 1.0:
        raise ValueError("Invalid retention-classification thresholds")
    if retention >= gate_minimum:
        return "gate_is_primary_text_suppressor"
    if retention <= low_lr_maximum:
        return "low_projection_lr_is_primary_text_suppressor"
    return "mixed_or_ambiguous"


def analyze(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        completed = _completed_registry(root, resume=resume, action="analyze")
        if completed is not None:
            return Path(str(completed["analysis_path"])).resolve()
        registry = _read_registry(root)
        record = registry["job"]
        if not base._completed_valid(record, root):
            raise RuntimeError("Analysis requires the completed supplement")
        base._journal(root, "analyze", "running")
        settings = _mapping(config["analysis"], "analysis")
        replicates = int(settings["bootstrap_replicates"])
        confidence = float(settings["confidence_level"])
        seed = int(settings["bootstrap_seed"])
        supplement_path = _metrics_path(record)
        comparator_config = config["frozen_comparators"]
        control_path = _resolve(comparator_config["control"]["metrics_path"])
        gated_path = _resolve(comparator_config["gated_lr5e7"]["metrics_path"])
        pair_keys = [
            tuple(
                pd.read_csv(path)[["pair_id", "session_id"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
            for path in (supplement_path, control_path, gated_path)
        ]
        if pair_keys[1:] != [pair_keys[0], pair_keys[0]]:
            raise ValueError(
                "Supplement and comparators have different pair/session order"
            )
        effects = {
            "supplement": _effect_stats(
                supplement_path,
                replicates=replicates,
                confidence=confidence,
                seed=seed,
            ),
            "control": _effect_stats(
                control_path,
                replicates=replicates,
                confidence=confidence,
                seed=seed + 10,
            ),
            "gated_lr5e7": _effect_stats(
                gated_path,
                replicates=replicates,
                confidence=confidence,
                seed=seed + 20,
            ),
        }
        control_magnitude = max(0.0, -float(effects["control"]["maximin"]))
        supplement_magnitude = max(0.0, -float(effects["supplement"]["maximin"]))
        retention = (
            supplement_magnitude / control_magnitude
            if control_magnitude > 0.0
            else math.nan
        )
        diagnosis = _classify_retention(
            retention,
            low_lr_maximum=float(settings["low_lr_dominant_maximum_retention"]),
            gate_minimum=float(settings["gate_dominant_minimum_retention"]),
        )
        probe_core = base._load_core()
        direct = {
            "supplement_vs_control": probe_core._paired_job_comparison(
                supplement_path, control_path, seed=seed + 100
            ),
            "supplement_vs_gated_lr5e7": probe_core._paired_job_comparison(
                supplement_path, gated_path, seed=seed + 101
            ),
        }
        contrasts = {}
        for comparator_name, comparator_path in (
            ("control", control_path),
            ("gated_lr5e7", gated_path),
        ):
            for denominator in ("mae_zero", "mae_wrong"):
                contrasts[f"supplement_vs_{comparator_name}_{denominator}"] = (
                    _bootstrap_effect_contrast(
                        supplement_path,
                        comparator_path,
                        denominator=denominator,
                        replicates=replicates,
                        confidence=confidence,
                        seed=seed + 200 + len(contrasts),
                    )
                )
        payload = base._signed_payload(
            {
                "schema_version": 1,
                "kind": "film_unet_ungated_proj_lr5e7_analysis_v1",
                "interpretation": INTERPRETATION,
                "effects": effects,
                "direct_comparisons": direct,
                "text_effect_contrasts": contrasts,
                "text_effect_retention_vs_control": retention,
                "descriptive_diagnosis": diagnosis,
                "diagnosis_rule": {
                    "low_lr_dominant_maximum_retention": settings[
                        "low_lr_dominant_maximum_retention"
                    ],
                    "gate_dominant_minimum_retention": settings[
                        "gate_dominant_minimum_retention"
                    ],
                },
                "causal_limitations": [
                    "single seed mechanism validation",
                    "retention thresholds are descriptive diagnostics",
                ],
                "test_loader_count": 0,
                "q3_q4_input_rows": 0,
            }
        )
        path = root / "analysis/ungated_proj_lr5e7_analysis.json"
        base._write_json(path, payload)
        flat_rows = []
        for name, value in effects.items():
            flat_rows.append(
                {
                    "profile": name,
                    "matched_mae": value["session_mae"]["mae_matched"],
                    "matched_vs_zero": value["matched_vs_zero"]["estimate"],
                    "matched_vs_zero_ci_lower": value["matched_vs_zero"]["ci_lower"],
                    "matched_vs_zero_ci_upper": value["matched_vs_zero"]["ci_upper"],
                    "matched_vs_wrong": value["matched_vs_wrong"]["estimate"],
                    "matched_vs_wrong_ci_lower": value["matched_vs_wrong"]["ci_lower"],
                    "matched_vs_wrong_ci_upper": value["matched_vs_wrong"]["ci_upper"],
                    "maximin": value["maximin"],
                }
            )
        csv_path = root / "analysis/ungated_proj_lr5e7_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
        registry.update(
            status="analyzed",
            analysis_path=str(path.resolve()),
            analysis_sha256=base._sha256_file(path),
        )
        _write_registry(root, registry)
        base._journal(root, "analyze", "completed", diagnosis=diagnosis)
        return path


def report(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        completed = _completed_registry(root, resume=resume, action="report")
        if completed is not None:
            return Path(str(completed["report_path"])).resolve()
        registry = _read_registry(root)
        analysis_path = root / "analysis/ungated_proj_lr5e7_analysis.json"
        if registry.get("analysis_path") != str(
            analysis_path.resolve()
        ) or registry.get("analysis_sha256") != base._sha256_file(analysis_path):
            raise ValueError("Frozen analysis binding drift")
        analysis = base._read_json(analysis_path)
        base._verify_signed_payload(analysis, "ungated-LR analysis")
        lines = [
            "# Ungated Global-FiLM projection-LR supplement",
            "",
            "Seed 42；5m / f2_2023q2；60 epochs；train/validation only。",
            "",
            "| Profile | Matched MAE | matched/zero log-ratio | matched/wrong log-ratio | Maximin |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, value in analysis["effects"].items():
            zero = value["matched_vs_zero"]
            wrong = value["matched_vs_wrong"]
            lines.append(
                "| {name} | {mae:.10g} | {zero:.8g} [{zero_lo:.8g}, {zero_hi:.8g}] "
                "| {wrong:.8g} [{wrong_lo:.8g}, {wrong_hi:.8g}] | {score:.8g} |".format(
                    name=name,
                    mae=value["session_mae"]["mae_matched"],
                    zero=zero["estimate"],
                    zero_lo=zero["ci_lower"],
                    zero_hi=zero["ci_upper"],
                    wrong=wrong["estimate"],
                    wrong_lo=wrong["ci_lower"],
                    wrong_hi=wrong["ci_upper"],
                    score=value["maximin"],
                )
            )
        lines.extend(
            [
                "",
                "## Direct matched-MAE comparisons",
                "",
                "| Comparison | log(MAE focal/reference) | 95% CI | Non-worse sessions |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, value in analysis["direct_comparisons"].items():
            lines.append(
                "| {name} | {estimate:.8g} | [{lower:.8g}, {upper:.8g}] | "
                "{nonworse}/34 |".format(
                    name=name,
                    estimate=value["estimate"],
                    lower=value["ci_lower"],
                    upper=value["ci_upper"],
                    nonworse=value["nonworse_sessions"],
                )
            )
        lines.extend(
            [
                "",
                "## Text-effect contrasts",
                "",
                "Negative values mean the supplement has a stronger matched-text effect.",
                "",
                "| Contrast | Effect difference | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for name, value in analysis["text_effect_contrasts"].items():
            lines.append(
                "| {name} | {estimate:.8g} | [{lower:.8g}, {upper:.8g}] |".format(
                    name=name,
                    estimate=value["estimate"],
                    lower=value["ci_lower"],
                    upper=value["ci_upper"],
                )
            )
        lines.extend(
            [
                "",
                "Text-effect retention vs control: "
                f"`{analysis['text_effect_retention_vs_control']:.6g}`.",
                "",
                f"Descriptive diagnosis: `{analysis['descriptive_diagnosis']}`.",
                "",
                "Only a single seed was tested; the diagnosis is descriptive, not confirmatory.",
                "",
                "Test loader / prediction / Q3-Q4 input rows are all zero.",
            ]
        )
        path = root / "report/ungated_proj_lr5e7_supplement.md"
        base._atomic_write_text(path, "\n".join(lines) + "\n")
        registry.update(
            status="reported",
            report_path=str(path.resolve()),
            report_sha256=base._sha256_file(path),
        )
        _write_registry(root, registry)
        base._journal(root, "report", "completed")
        return path


def qa(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        registry = _read_registry(root)
        if registry.get("status") == "completed":
            if not resume:
                raise FileExistsError("Completed root requires --resume")
            base._verify_output_hash_manifest(root / "qa/output_hashes.csv", root)
            return Path(str(registry["terminal_qa_path"]))
        if registry.get("status") != "reported":
            raise RuntimeError("QA requires a completed report")
        base._journal(root, "qa", "running")
        record = registry["job"]
        if not base._completed_valid(record, root):
            raise RuntimeError("Supplement completion artifacts are invalid")
        output = Path(str(record["output_dir"]))
        diagnostics = base._read_json(output / "diagnostics.json")
        metrics = pd.read_csv(output / "training_metrics.csv")
        pairs = pd.read_csv(output / "validation_pair_metrics.csv")
        control_diagnostics = base._read_json(
            _resolve(config["frozen_comparators"]["control"]["diagnostics_path"])
        )
        control_pairs = pd.read_csv(
            _resolve(config["frozen_comparators"]["control"]["metrics_path"])
        )
        if len(metrics) != 60 or list(metrics["epoch"]) != list(range(1, 61)):
            raise ValueError("Training epoch trace drift")
        if len(pairs) != 110 or pairs["session_id"].nunique() != 34:
            raise ValueError("Validation universe drift")
        if tuple(pairs["pair_id"].astype(str)) != tuple(
            control_pairs["pair_id"].astype(str)
        ) or tuple(pairs["session_id"].astype(str)) != tuple(
            control_pairs["session_id"].astype(str)
        ):
            raise ValueError("Supplement/control pair/session order drift")
        if not np.array_equal(
            pairs["mae_zero"].to_numpy(), control_pairs["mae_zero"].to_numpy()
        ):
            raise ValueError("Supplement/control zero-text metrics differ")
        if diagnostics["initial_parameter_group_sha256"] != {
            key: control_diagnostics["initial_parameter_group_sha256"][key]
            for key in ("text_encoder", "global_film")
        }:
            raise ValueError("Common adapter initialization drift")
        if (
            diagnostics["initial_parameter_sha256"]
            != control_diagnostics["initial_parameter_sha256"]
        ):
            raise ValueError("Common full initialization drift")
        contract = diagnostics["optimizer_contract"]
        if (
            contract["mode"] != "separate_text_and_film_lr_v1"
            or float(contract["text_encoder_learning_rate"]) != 2.5e-6
            or float(contract["global_film_learning_rate"]) != 5.0e-7
            or contract["global_gate_learning_rate"] is not None
        ):
            raise ValueError("Supplement optimizer contract drift")
        if int(diagnostics["parameters"]["generator_total"]) != 827_745:
            raise ValueError("Supplement parameter count drift")
        if (
            diagnostics["gate_schedule"]["mode"] != "disabled"
            or diagnostics["final_global_gate_values"]
        ):
            raise ValueError("Supplement unexpectedly enabled a gate")
        if set(diagnostics["trainable_parameter_names"]) != {
            "text_encoder",
            "global_film",
        }:
            raise ValueError("Supplement trainable parameter groups drift")
        gradients = diagnostics["first_update_gradient_norms"]
        if any(
            not math.isfinite(float(gradients[name])) or float(gradients[name]) <= 0.0
            for name in ("text_encoder", "global_film")
        ):
            raise ValueError(
                "Text/FiLM first-update gradients must be finite and nonzero"
            )
        if (
            diagnostics["final_parameter_sha256"]
            == diagnostics["initial_parameter_sha256"]
        ):
            raise ValueError("Supplement adapter parameters did not update")
        if float(diagnostics["zero_text_max_abs_error"]) > 1.0e-7:
            raise ValueError("Zero-text equivalence failed")
        if (
            not math.isfinite(float(diagnostics["max_gradient_norm"]))
            or float(diagnostics["max_gradient_norm"]) > 1.0e3
        ):
            raise ValueError("Gradient norm is invalid or exploded")
        if diagnostics.get("test_loader_materialized") is not False or int(
            diagnostics.get("q3_q4_input_rows", -1)
        ):
            raise ValueError("Forbidden evaluation access detected")
        numeric = metrics.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Training metrics contain nonfinite values")
        analysis_path = root / "analysis/ungated_proj_lr5e7_analysis.json"
        report_path = root / "report/ungated_proj_lr5e7_supplement.md"
        if registry.get("analysis_path") != str(
            analysis_path.resolve()
        ) or registry.get("analysis_sha256") != base._sha256_file(analysis_path):
            raise ValueError("Terminal QA analysis binding drift")
        if registry.get("report_path") != str(report_path.resolve()) or registry.get(
            "report_sha256"
        ) != base._sha256_file(report_path):
            raise ValueError("Terminal QA report binding drift")
        analysis = base._read_json(analysis_path)
        base._verify_signed_payload(analysis, "ungated-LR analysis")
        if (
            max(
                analysis["effects"]["supplement"][
                    "constraint_violation_rate_increases"
                ].values()
            )
            > 0.01
        ):
            raise ValueError("Supplement increases constraint violations by over 1%")
        terminal = base._signed_payload(
            {
                "schema_version": 1,
                "kind": TERMINAL_QA_KIND,
                "status": "passed",
                "jobs": 1,
                "validation_pairs": 110,
                "validation_sessions": 34,
                "descriptive_diagnosis": analysis["descriptive_diagnosis"],
                "test_loader_count": 0,
                "prediction_count": 0,
                "q3_q4_input_rows": 0,
                "output_hash_manifest_path": str(
                    (root / "qa/output_hashes.csv").resolve()
                ),
                "completed_at_utc": base._utc_now(),
            }
        )
        terminal_path = base._write_json(root / "qa/terminal_qa.json", terminal)
        registry.update(
            status="completed",
            terminal_qa_path=str(terminal_path.resolve()),
            terminal_qa_sha256=base._sha256_file(terminal_path),
        )
        _write_registry(root, registry)
        base._journal(root, "qa", "completed")
        hash_path = base._output_hash_manifest(root)
        base._verify_output_hash_manifest(hash_path, root)
        return terminal_path


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if not _registry_path(root).is_file():
        return {
            "root": str(root),
            "status": "not_prepared",
            "completed_jobs": 0,
            "total_jobs": 1,
        }
    registry = _read_registry(root)
    record = registry["job"]
    return {
        "root": str(root),
        "status": registry.get("status"),
        "completed_jobs": int(base._completed_valid(record, root)),
        "total_jobs": 1,
        "test_loader_count": registry.get("test_loader_count", 0),
        "prediction_count": registry.get("prediction_count", 0),
        "q3_q4_input_rows": registry.get("q3_q4_input_rows", 0),
    }


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    if _registry_path(root).is_file():
        current = _read_registry(root)
        if current.get("status") == "completed":
            if not resume:
                raise FileExistsError("Completed root requires --resume")
            config = load_config(config_path)
            _validate_prepared(root, config)
            base._verify_output_hash_manifest(root / "qa/output_hashes.csv", root)
            return Path(str(current["terminal_qa_path"]))
    with base._exclusive_lock(root):
        control = base._control_dir(root)
        base._atomic_write_text(control / "pipeline.pid", f"{os.getpid()}\n")
        try:
            prepare(config_path, output_dir, resume=resume)
            canary(config_path, output_dir, resume=resume)
            launch(config_path, output_dir, resume=resume)
            analyze(config_path, output_dir, resume=resume)
            report(config_path, output_dir, resume=resume)
            result = qa(config_path, output_dir, resume=resume)
            base._journal(root, "run-pipeline", "completed")
            return result
        except BaseException as exc:
            base._journal(
                root,
                "run-pipeline",
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            (control / "pipeline.pid").unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "status":
        print(json.dumps(status(args.output_dir), indent=2, sort_keys=True))
        return 0
    handlers = {
        "prepare": prepare,
        "canary": canary,
        "launch": launch,
        "analyze": analyze,
        "report": report,
        "qa": qa,
        "run-pipeline": run_pipeline,
    }
    result = handlers[args.action](args.config, args.output_dir, resume=args.resume)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "JOB_ID",
    "_classify_retention",
    "analyze",
    "canary",
    "launch",
    "load_config",
    "main",
    "prepare",
    "qa",
    "report",
    "run_pipeline",
    "status",
    "validate_config",
]
