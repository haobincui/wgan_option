"""Three-cell gated Global-FiLM learning-rate mechanism probe.

This experiment reuses the frozen train/validation inputs from the completed
text-signal probe.  It intentionally has no test, prediction, Q3, or Q4 path.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from scripts.rq3 import news_first_vol_film_unet_text_signal_probe as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_gated_global_lr_seed42_f2.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/rq3_news_first_vol_film_unet_gated_global_lr_seed42_f2_5m_v1"
)
EXPERIMENT_KIND = "film_unet_gated_global_lr_seed42_f2_5m_probe_v1"
REGISTRY_KIND = "film_unet_gated_global_lr_probe_task_registry_v1"
TERMINAL_QA_KIND = "film_unet_gated_global_lr_probe_terminal_qa_v1"
INTERPRETATION = "single_seed_mechanism_validation"
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
        raise ValueError("The probe is frozen to seed42/f2/5m")
    expected_counts = (526, 133, 110, 34)
    observed_counts = tuple(
        int(data[key])
        for key in (
            "train_pairs",
            "train_sessions",
            "validation_pairs",
            "validation_sessions",
        )
    )
    if observed_counts != expected_counts:
        raise ValueError(f"Pair/session universe drift: {observed_counts}")
    if data.get("allowed_partitions") != ["train", "validation"]:
        raise ValueError("Only train/validation partitions are permitted")
    if (
        int(data.get("test_loader_count", -1)) != 0
        or int(data.get("q3_q4_input_rows", -1)) != 0
    ):
        raise ValueError("Test and Q3/Q4 access must remain zero")
    expected_model = {
        "generator_conditioning_mode": "film_unet_mask_coords_v1",
        "critic_conditioning_mode": "lp_disabled_same_shape_v1",
        "factor_code": "1000",
        "spatial_rank": 0,
        "baseline_generator_parameters": 827_745,
        "gated_generator_parameters": 827_751,
        "critic_parameters": 729_157,
        "gate_mode": "direct_hard_clamped_per_site_v1",
        "gate_sites": 6,
        "gate_freeze_epochs": 10,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model_contract.{key} drift")
    if not math.isclose(float(model["gate_initial"]), 0.01) or not math.isclose(
        float(model["gate_maximum"]), 0.1
    ):
        raise ValueError("Gate bounds drift")
    if int(training.get("epochs", -1)) != 60:
        raise ValueError("Training must use exactly 60 epochs")
    if not math.isclose(float(training["text_encoder_learning_rate"]), 2.5e-6):
        raise ValueError("Text-encoder LR drift")
    jobs = list(training.get("jobs") or [])
    if len(jobs) != 3 or len({str(row.get("job_id")) for row in jobs}) != 3:
        raise ValueError("Exactly three unique jobs are required")
    expected_jobs = {
        "control_1000_uniform_2p5e6": (False, None, None),
        "gated_global_proj_lr5e7": (True, 5.0e-7, 5.0e-7),
        "gated_global_proj_lr2p5e7": (True, 2.5e-7, 2.5e-7),
    }
    for row in jobs:
        job_id = str(row.get("job_id"))
        if job_id not in expected_jobs:
            raise ValueError(f"Unexpected job: {job_id}")
        enabled, film_lr, gate_lr = expected_jobs[job_id]
        if bool(row.get("gate_enabled")) is not enabled:
            raise ValueError(f"Gate flag drift for {job_id}")
        if enabled and (
            not math.isclose(float(row["global_film_learning_rate"]), film_lr)
            or not math.isclose(float(row["global_gate_learning_rate"]), gate_lr)
        ):
            raise ValueError(f"Grouped LR drift for {job_id}")
    if int(analysis.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Analysis requires 10,000 bootstrap replicates")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Both physical GPUs must be configured")
    if int(runtime.get("workers_per_gpu", -1)) != 2:
        raise ValueError("This small matrix is frozen to two workers/GPU")
    for key in (
        "input_manifest_path",
        "resolved_config_path",
        "terminal_qa_path",
    ):
        if not _resolve(str(source[key])).is_file():
            raise FileNotFoundError(_resolve(str(source[key])))


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _write_registry(root: Path, payload: Mapping[str, Any]) -> Path:
    value = dict(payload)
    value["updated_at_utc"] = base._utc_now()
    return base._write_json(_registry_path(root), base._signed_payload(value))


def _read_registry(root: Path) -> dict[str, Any]:
    value = base._read_json(_registry_path(root))
    base._verify_signed_payload(value, "gated-LR task registry")
    if value.get("kind") != REGISTRY_KIND:
        raise ValueError("Gated-LR registry kind drift")
    return value


def _verify_source_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(config["source_probe"], "source_probe")
    rows = {}
    for label in ("input_manifest", "resolved_config", "terminal_qa"):
        path = _resolve(str(source[f"{label}_path"]))
        expected = str(source[f"{label}_sha256"])
        if base._sha256_file(path) != expected:
            raise ValueError(f"Frozen source {label} SHA drift")
        rows[label] = base._file_row(label, path)
    manifest = base._read_json(rows["input_manifest"]["path"])
    if manifest.get("test_loader_materialized") is not False:
        raise ValueError("Source probe materialized a test loader")
    if (
        int(manifest.get("test_prediction_rows", -1)) != 0
        or int(manifest.get("q3_q4_input_rows", -1)) != 0
    ):
        raise ValueError("Source probe contains forbidden evaluation rows")
    old_terminal = base._read_json(rows["terminal_qa"]["path"])
    if old_terminal.get("status") != "passed":
        raise ValueError("Source probe terminal QA is not passing")
    return rows


def _source_manifest(root: Path, config: Mapping[str, Any]) -> Path:
    rows = list(_verify_source_probe(config).values())
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
    payload = base._signed_payload(
        {
            "schema_version": 1,
            "kind": "film_unet_gated_global_lr_source_manifest_v1",
            "rows": rows,
        }
    )
    return base._write_json(root / "contracts/source_manifest.json", payload)


def _verify_source_manifest(root: Path) -> dict[str, Any]:
    path = root / "contracts/source_manifest.json"
    value = base._read_json(path)
    base._verify_signed_payload(value, "gated-LR source manifest")
    for row in value.get("rows") or []:
        base._verify_file_row(_mapping(row, "source row"), "source row")
    return value


def _build_job_records(
    root: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = _mapping(config["source_probe"], "source_probe")
    training = _mapping(config["training"], "training")
    model = _mapping(config["model_contract"], "model_contract")
    input_path = _resolve(str(source["input_manifest_path"]))
    source_config = base._read_json(_resolve(str(source["resolved_config_path"])))
    records: list[dict[str, Any]] = []
    for job in training["jobs"]:
        row = _mapping(job, "training job")
        job_id = str(row["job_id"])
        output_dir = (root / "runs/screen" / job_id).resolve()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": base.JOB_SPEC_KIND,
            "job_id": job_id,
            "profile_id": job_id,
            "stage": "screen",
            "seed": 42,
            "fold_id": "f2_2023q2",
            "tolerance_minutes": 5,
            "factor_code": "1000",
            "text_assignment": "matched",
            "text_output_dimension": 128,
            "spatial_rank": 0,
            "physical_gpu_id": int(row["physical_gpu_id"]),
            "max_epochs": int(training["epochs"]),
            "output_dir": str(output_dir),
            "evaluation_partition": "validation",
            "allowed_partitions": ["train", "validation"],
            "core_module": base.CORE_MODULE,
            "input_manifest_path": str(input_path),
            "input_manifest_sha256": str(source["input_manifest_sha256"]),
            "calibrate_auxiliary_weight": False,
            "alignment_auxiliary_weight": 0.0,
            "expected_generator_parameters": (
                int(model["gated_generator_parameters"])
                if row["gate_enabled"]
                else int(model["baseline_generator_parameters"])
            ),
            "baseline_generator": deepcopy(
                source_config["baseline"]["generator_checkpoint"]
            ),
            "baseline_critic": deepcopy(source_config["baseline"]["critic_checkpoint"]),
            "training_contract": deepcopy(training),
            "model_contract": deepcopy(model),
            "test_loader_count": 0,
            "q3_q4_input_rows": 0,
        }
        if bool(row["gate_enabled"]):
            payload.update(
                text_encoder_learning_rate=float(
                    training["text_encoder_learning_rate"]
                ),
                global_film_learning_rate=float(row["global_film_learning_rate"]),
                global_gate_learning_rate=float(row["global_gate_learning_rate"]),
                global_residual_gate_initial=float(model["gate_initial"]),
                global_residual_gate_max=float(model["gate_maximum"]),
                global_residual_gate_freeze_epochs=int(model["gate_freeze_epochs"]),
            )
        spec_path, spec_sha = base._write_job_spec(
            root / "registry/job_specs/screen" / f"{job_id}.json", payload
        )
        records.append(
            {
                "job_id": job_id,
                "profile_id": job_id,
                "stage": "screen",
                "factor_code": "1000",
                "physical_gpu_id": int(row["physical_gpu_id"]),
                "gate_enabled": bool(row["gate_enabled"]),
                "job_spec_path": str(spec_path.resolve()),
                "job_spec_sha256": spec_sha,
                "output_dir": str(output_dir),
            }
        )
    return records


def _validate_prepared(root: Path, config: Mapping[str, Any]) -> None:
    frozen = base._read_json(root / "contracts/resolved_config.json")
    if frozen != dict(config):
        raise ValueError("Resolved gated-LR config drift")
    _verify_source_probe(config)
    _verify_source_manifest(root)
    registry = _read_registry(root)
    records = list(registry.get("jobs") or [])
    if len(records) != 3:
        raise ValueError("Prepared registry must contain three jobs")
    for record in records:
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
        base._journal(root, "prepare", "running")
        if _registry_path(root).is_file():
            if not resume:
                raise FileExistsError(f"Experiment already prepared: {root}")
            _validate_prepared(root, config)
            base._journal(root, "prepare", "completed", resumed=True)
            return root
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
        records = _build_job_records(root, config)
        registry = {
            "schema_version": 1,
            "kind": REGISTRY_KIND,
            "experiment_kind": EXPERIMENT_KIND,
            "interpretation": INTERPRETATION,
            "status": "prepared",
            "root": str(root),
            "jobs": records,
            "job_count": 3,
            "source_manifest_path": str(source_manifest.resolve()),
            "source_manifest_sha256": base._sha256_file(source_manifest),
            "partitions_materialized": ["train", "validation"],
            "test_loader_count": 0,
            "prediction_count": 0,
            "q3_q4_input_rows": 0,
            "created_at_utc": base._utc_now(),
        }
        _write_registry(root, registry)
        _validate_prepared(root, config)
        base._journal(root, "prepare", "completed", jobs=3)
        return root


def _canary_path(root: Path) -> Path:
    return base._control_dir(root) / "canary_result.json"


def _verify_canary(root: Path, path: Path) -> dict[str, Any]:
    value = base._read_json(path)
    base._verify_signed_payload(value, "gated-LR canary")
    if value.get("status") != "passed":
        raise ValueError("Frozen canary is not passing")
    canary_root = Path(str(value.get("canary_root", ""))).resolve()
    records = list(value.get("jobs") or [])
    if len(records) != 3 or any(
        not base._completed_valid(record, canary_root) for record in records
    ):
        raise ValueError("Frozen canary completion artifacts drifted")
    if canary_root != base._control_dir(root) / "canary":
        raise ValueError("Frozen canary root drifted")
    return value


def _canary_records(
    root: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[Path, list[dict[str, Any]]]:
    canary_root = base._control_dir(root) / "canary"
    output: list[dict[str, Any]] = []
    for record in records:
        formal_spec = base._verify_job_spec(
            record["job_spec_path"], record["job_spec_sha256"]
        )
        job_id = "canary_" + str(record["job_id"])
        output_dir = (canary_root / "runs" / job_id).resolve()
        payload = {
            **formal_spec,
            "job_id": job_id,
            "profile_id": str(record["job_id"]),
            "stage": "canary",
            "max_epochs": 1,
            "output_dir": str(output_dir),
        }
        payload.pop("job_spec_sha256", None)
        spec_path, spec_sha = base._write_job_spec(
            canary_root / "job_specs" / f"{job_id}.json", payload
        )
        output.append(
            {
                **dict(record),
                "job_id": job_id,
                "stage": "canary",
                "job_spec_path": str(spec_path.resolve()),
                "job_spec_sha256": spec_sha,
                "output_dir": str(output_dir),
            }
        )
    return canary_root, output


def _check_resources(summary: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    runtime = _mapping(config["runtime"], "runtime")
    if float(summary["peak_host_ram_fraction"]) >= float(
        runtime["maximum_host_ram_fraction"]
    ):
        raise RuntimeError("Canary/formal host-RAM gate failed")
    if max(map(float, summary["peak_gpu_memory_gib"].values())) >= float(
        runtime["maximum_peak_gpu_memory_gib"]
    ):
        raise RuntimeError("Canary/formal GPU-memory gate failed")


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
        path = _canary_path(root)
        if path.is_file():
            _verify_canary(root, path)
            return path
        base._journal(root, "canary", "running")
        registry = _read_registry(root)
        canary_root, records = _canary_records(root, registry["jobs"])
        summary = base._launch_records(
            records,
            canary_root,
            config,
            workers_per_gpu=int(config["runtime"]["workers_per_gpu"]),
            resume=resume,
        )
        _check_resources(summary, config)
        for record in records:
            diagnostics = base._read_json(
                Path(record["output_dir"]) / "diagnostics.json"
            )
            expected = 827_751 if record["gate_enabled"] else 827_745
            if int(diagnostics["parameters"]["generator_total"]) != expected:
                raise ValueError("Canary parameter-count drift")
            if float(diagnostics["zero_text_max_abs_error"]) > 1.0e-7:
                raise ValueError("Canary zero-text equivalence failed")
            if record["gate_enabled"]:
                gates = list(diagnostics["final_global_gate_values"])
                if len(gates) != 6 or any(
                    not math.isclose(float(value), 0.01, abs_tol=1.0e-8)
                    for value in gates
                ):
                    raise ValueError("Frozen one-epoch canary gate moved")
                gradients = diagnostics["first_update_gradient_norms"]
                if float(gradients.get("global_gate", 0.0)) <= 0.0:
                    raise ValueError("Canary gate did not receive a raw gradient")
        value = base._signed_payload(
            {
                "schema_version": 1,
                "kind": "film_unet_gated_global_lr_canary_v1",
                "status": "passed",
                "canary_root": str(canary_root.resolve()),
                "jobs": records,
                "resources": summary,
                "test_loader_count": 0,
                "q3_q4_input_rows": 0,
                "completed_at_utc": base._utc_now(),
            }
        )
        base._write_json(path, value)
        registry.update(status="canary_passed", canary_path=str(path.resolve()))
        _write_registry(root, registry)
        base._journal(root, "canary", "completed", jobs=3)
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
        if not _canary_path(root).is_file():
            raise RuntimeError("Formal launch requires a passing canary")
        _verify_canary(root, _canary_path(root))
        base._journal(root, "launch", "running")
        registry = _read_registry(root)
        summary = base._launch_records(
            registry["jobs"],
            root,
            config,
            workers_per_gpu=int(config["runtime"]["workers_per_gpu"]),
            resume=resume,
        )
        _check_resources(summary, config)
        if int(summary["completed_jobs"]) != 3:
            raise RuntimeError("Formal launch did not complete all three jobs")
        registry.update(status="trained", resource_summary=summary)
        _write_registry(root, registry)
        base._journal(root, "launch", "completed", jobs=3)
        return root / "resource_summary_screen.json"


def _result_metrics_path(record: Mapping[str, Any]) -> Path:
    spec = base._verify_job_spec(record["job_spec_path"], record["job_spec_sha256"])
    result = base._verify_result(spec, Path(str(record["output_dir"])))
    for artifact in result["artifacts"]:
        if artifact.get("role") == "validation_pair_metrics":
            return Path(str(artifact["path"])).resolve()
    raise ValueError(f"Missing validation metrics for {record['job_id']}")


def analyze(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    del resume
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        registry = _read_registry(root)
        if not all(base._completed_valid(row, root) for row in registry["jobs"]):
            raise RuntimeError("Analysis requires all three completed jobs")
        base._journal(root, "analyze", "running")
        analysis = _mapping(config["analysis"], "analysis")
        replicates = int(analysis["bootstrap_replicates"])
        confidence = float(analysis["confidence_level"])
        seed = int(analysis["bootstrap_seed"])
        probe_core = base._load_core()
        metric_paths = {
            str(row["job_id"]): _result_metrics_path(row) for row in registry["jobs"]
        }
        pair_universes = [
            tuple(pd.read_csv(path)["pair_id"].astype(str))
            for path in metric_paths.values()
        ]
        if any(values != pair_universes[0] for values in pair_universes[1:]):
            raise ValueError("Three cells do not share the same validation pair order")
        detailed_rows: list[dict[str, Any]] = []
        flat_rows: list[dict[str, Any]] = []
        for index, record in enumerate(registry["jobs"]):
            job_id = str(record["job_id"])
            sessions = probe_core._session_frame(metric_paths[job_id])
            matched_zero = probe_core._bootstrap_ratio(
                sessions,
                "mae_matched",
                "mae_zero",
                replicates=replicates,
                confidence=confidence,
                seed=seed + index * 10,
            )
            matched_wrong = probe_core._bootstrap_ratio(
                sessions,
                "mae_matched",
                "mae_wrong",
                replicates=replicates,
                confidence=confidence,
                seed=seed + index * 10 + 1,
            )
            constraints = {
                name: float(
                    sessions[f"{name}_matched"].mean() - sessions[f"{name}_zero"].mean()
                )
                for name in ("calendar_violation", "butterfly_violation")
            }
            score = max(matched_zero["estimate"], matched_wrong["estimate"])
            detailed_rows.append(
                {
                    "job_id": job_id,
                    "gate_enabled": bool(record["gate_enabled"]),
                    "selection_score": score,
                    "matched_vs_zero": matched_zero,
                    "matched_vs_wrong": matched_wrong,
                    "constraint_violation_rate_increases": constraints,
                }
            )
            flat_rows.append(
                {
                    "job_id": job_id,
                    "gate_enabled": bool(record["gate_enabled"]),
                    "selection_score": score,
                    "matched_vs_zero_log_ratio": matched_zero["estimate"],
                    "matched_vs_zero_ci_lower": matched_zero["ci_lower"],
                    "matched_vs_zero_ci_upper": matched_zero["ci_upper"],
                    "matched_vs_zero_nonworse_sessions": matched_zero[
                        "nonworse_sessions"
                    ],
                    "matched_vs_wrong_log_ratio": matched_wrong["estimate"],
                    "matched_vs_wrong_ci_lower": matched_wrong["ci_lower"],
                    "matched_vs_wrong_ci_upper": matched_wrong["ci_upper"],
                    "matched_vs_wrong_nonworse_sessions": matched_wrong[
                        "nonworse_sessions"
                    ],
                }
            )
        reference_id = str(analysis["direct_reference_job"])
        comparison_pairs = (
            ("gated_global_proj_lr5e7", reference_id),
            ("gated_global_proj_lr2p5e7", reference_id),
            ("gated_global_proj_lr2p5e7", "gated_global_proj_lr5e7"),
        )
        comparisons = []
        for index, (focal, reference) in enumerate(comparison_pairs):
            result = probe_core._paired_job_comparison(
                metric_paths[focal], metric_paths[reference], seed=seed + 100 + index
            )
            comparisons.append(
                {"focal_job": focal, "reference_job": reference, **result}
            )
        by_job = {row["job_id"]: row for row in detailed_rows}
        direct_by_focal = {row["focal_job"]: row for row in comparisons[:2]}
        threshold = float(analysis["minimum_log_mae_improvement"])
        maximum_constraint = float(
            analysis["maximum_constraint_violation_increase_fraction"]
        )
        minimum_sessions = int(analysis["minimum_nonworse_sessions"])
        supported = []
        for job_id in ("gated_global_proj_lr5e7", "gated_global_proj_lr2p5e7"):
            row = by_job[job_id]
            direct = direct_by_focal[job_id]
            if (
                row["matched_vs_zero"]["estimate"] <= threshold
                and row["matched_vs_wrong"]["estimate"] <= threshold
                and row["matched_vs_zero"]["ci_upper"] < 0.0
                and row["matched_vs_wrong"]["ci_upper"] < 0.0
                and row["matched_vs_zero"]["nonworse_sessions"] >= minimum_sessions
                and row["matched_vs_wrong"]["nonworse_sessions"] >= minimum_sessions
                and max(row["constraint_violation_rate_increases"].values())
                <= maximum_constraint
                and direct["estimate"] < 0.0
                and direct["ci_upper"] < 0.0
            ):
                supported.append(job_id)
        leader = min(detailed_rows, key=lambda row: row["selection_score"])["job_id"]
        payload = base._signed_payload(
            {
                "schema_version": 1,
                "kind": "film_unet_gated_global_lr_analysis_v1",
                "interpretation": INTERPRETATION,
                "rows": detailed_rows,
                "direct_comparisons": comparisons,
                "descriptive_leader": leader,
                "mechanism_supported": bool(supported),
                "supported_gated_jobs": supported,
                "causal_attribution_limitation": (
                    "Gated cells jointly change the gate and projection LR; this "
                    "three-cell probe cannot isolate the gate main effect."
                ),
                "test_loader_count": 0,
                "q3_q4_input_rows": 0,
            }
        )
        summary_path = root / "analysis/gated_global_lr_selection.json"
        base._write_json(summary_path, payload)
        for path, rows in (
            (root / "analysis/gated_global_lr_summary.csv", flat_rows),
            (root / "analysis/gated_global_lr_comparisons.csv", comparisons),
        ):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        registry.update(
            status="analyzed",
            analysis_path=str(summary_path.resolve()),
            analysis_sha256=base._sha256_file(summary_path),
        )
        _write_registry(root, registry)
        base._journal(root, "analyze", "completed", leader=leader)
        return summary_path


def report(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    del resume
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with base._exclusive_lock(root):
        _validate_prepared(root, config)
        selection = base._read_json(root / "analysis/gated_global_lr_selection.json")
        base._verify_signed_payload(selection, "gated-LR analysis")
        rows = selection["rows"]
        lines = [
            "# Gated Global FiLM learning-rate probe",
            "",
            "Seed 42；5m / f2_2023q2；只使用 train 与 validation；60 epochs。",
            "",
            "| 配置 | matched vs zero log-ratio | matched vs wrong log-ratio | maximin |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {job} | {zero:.8g} | {wrong:.8g} | {score:.8g} |".format(
                    job=row["job_id"],
                    zero=row["matched_vs_zero"]["estimate"],
                    wrong=row["matched_vs_wrong"]["estimate"],
                    score=row["selection_score"],
                )
            )
        lines.extend(
            [
                "",
                f"Descriptive leader: `{selection['descriptive_leader']}`.",
                "",
                "Mechanism supported: "
                f"`{str(selection['mechanism_supported']).lower()}`.",
                "",
                "本实验是 single-seed mechanism validation，且 gated arms 同时改变 gate "
                "与 projection LR，因此不能把差异单独归因于 gate。",
                "",
                "Test loader / prediction / Q3-Q4 input rows 均为 0。",
            ]
        )
        path = root / "report/gated_global_lr_probe.md"
        base._atomic_write_text(path, "\n".join(lines) + "\n")
        registry = _read_registry(root)
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
        group_shas: dict[str, list[str]] = {"text_encoder": [], "global_film": []}
        pair_ids: list[tuple[str, ...]] = []
        for record in registry["jobs"]:
            if not base._completed_valid(record, root):
                raise RuntimeError(f"Invalid job completion: {record['job_id']}")
            output = Path(record["output_dir"])
            diagnostics = base._read_json(output / "diagnostics.json")
            metrics = pd.read_csv(output / "training_metrics.csv")
            pairs = pd.read_csv(output / "validation_pair_metrics.csv")
            if len(metrics) != 60 or list(metrics["epoch"]) != list(range(1, 61)):
                raise ValueError("Training epoch trace drift")
            if len(pairs) != 110 or pairs["session_id"].nunique() != 34:
                raise ValueError("Validation universe drift")
            pair_ids.append(tuple(pairs["pair_id"].astype(str)))
            if float(diagnostics["zero_text_max_abs_error"]) > 1.0e-7:
                raise ValueError("Zero-text equivalence QA failed")
            if (
                diagnostics.get("test_loader_materialized") is not False
                or int(diagnostics.get("q3_q4_input_rows", -1)) != 0
            ):
                raise ValueError("Forbidden evaluation access detected")
            for group in group_shas:
                group_shas[group].append(
                    diagnostics["initial_parameter_group_sha256"][group]
                )
            if record["gate_enabled"]:
                if int(diagnostics["parameters"]["generator_total"]) != 827_751:
                    raise ValueError("Gated parameter count drift")
                if len(diagnostics["trainable_parameter_names"]["global_gate"]) != 6:
                    raise ValueError("Gated model does not have six gates")
                if not bool(metrics.loc[:9, "global_gate_frozen"].all()) or bool(
                    metrics.loc[10:, "global_gate_frozen"].any()
                ):
                    raise ValueError("Gate freeze/release trace drift")
                if not bool((metrics.loc[:9, "global_gate_learning_rate"] == 0).all()):
                    raise ValueError("Frozen gate LR is not zero")
                gate_columns = [
                    column
                    for column in metrics.columns
                    if column.startswith("global_gate_site_")
                ]
                if len(gate_columns) != 6:
                    raise ValueError("Per-site gate trace is incomplete")
                gates = metrics[gate_columns]
                if not bool(((gates >= -1.0e-8) & (gates <= 0.10000001)).all().all()):
                    raise ValueError("Gate escaped its frozen bounds")
                if not bool((gates.iloc[:10] == gates.iloc[0]).all().all()):
                    raise ValueError("A gate moved during the ten-epoch freeze")
                if bool((gates.iloc[-1] == gates.iloc[9]).all()):
                    raise ValueError("No gate changed after release")
                if float(diagnostics.get("gate_release_gradient_norm") or 0.0) <= 0.0:
                    raise ValueError("Gate release gradient QA failed")
            elif int(diagnostics["parameters"]["generator_total"]) != 827_745:
                raise ValueError("Control parameter count drift")
        if any(values != pair_ids[0] for values in pair_ids[1:]):
            raise ValueError("Pair IDs differ across cells")
        if any(len(set(values)) != 1 for values in group_shas.values()):
            raise ValueError("Common adapter initialization drifted across cells")
        terminal = base._signed_payload(
            {
                "schema_version": 1,
                "kind": TERMINAL_QA_KIND,
                "status": "passed",
                "jobs": 3,
                "validation_pairs": 110,
                "validation_sessions": 34,
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
            "total_jobs": 3,
        }
    registry = _read_registry(root)
    jobs = list(registry.get("jobs") or [])
    return {
        "root": str(root),
        "status": registry.get("status"),
        "completed_jobs": sum(base._completed_valid(row, root) for row in jobs),
        "total_jobs": len(jobs),
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
    result = handlers[args.action](
        args.config,
        args.output_dir,
        resume=args.resume,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
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
