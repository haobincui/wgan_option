"""Fail-closed multi-seed capacity experiment for FiLM-G + NoLP Critic.

The module deliberately reuses the already tested conditioning, exact-TTM,
training, and frozen-epoch/LR replay contracts.  It owns a new branch-local
registry because the 36-cell capacity matrix and all-capacity Q4 refit are a
different estimand from the earlier single-seed architecture factorial.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

from scripts.rq3 import news_first_vol_generator_film_critic_factorial as factorial
from scripts.rq3 import news_first_vol_training as training
from wgan_option.models.common import (
    critic_conditioning_fingerprint,
    generator_conditioning_fingerprint,
)
from wgan_option.utils.text_ablation import REAL_TEXT


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "film_nolp_capacity_seed_q4_refit"
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_nolp_capacity_seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/rq3_news_first_vol_film_nolp_capacity_seed_exact_ttm_v1"
)
DEVELOPMENT_STAGE = "development_q3_capacity_selection"
REFIT_STAGE = "full_history_capacity_refit"
BENCHMARK_STAGE = "benchmark"
REFIT_MODE = "frozen_epoch_lr_replay_v1"
GENERATOR_MODE = "film_conv_bottleneck_concat_v1"
CRITIC_MODE = "lp_disabled_same_shape_v1"
PROFILES = tuple(training.CAPACITY_PROFILE_NAMES)
SEEDS = (42, 202, 404)
TOLERANCES = (5, 30)
EXPECTED_STAGE_JOBS = 36
EXPECTED_TOTAL_JOBS = 72
EXPECTED_Q4_COMMON_COUNTS = {"rows": 167, "pairs": 143, "sessions": 45}
DEVELOPMENT_ARTIFACT_ROLES = factorial.DEVELOPMENT_ARTIFACT_ROLES
REFIT_ARTIFACT_ROLES = factorial.REFIT_ARTIFACT_ROLES

Q4_RECOVERY_INCIDENT = "factorial_q4_materializer_kind_guard_v1"
Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SHA256 = (
    "a2bed0e5c3e13693dbccee9ea733655f10a414c008056f18165aedae95ee09aa"
)
Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SIZE = 71_072
Q4_RECOVERY_FAILURE = "ValueError: Experiment root is not a Generator/Critic factorial"
Q4_RECOVERY_DIRECTORY = "registry/q4_recovery_v1"
Q4_RECOVERY_LEDGER = "ledger.json"
Q4_RECOVERY_CURRENT_CODE_HASHES = "code_hashes_current.csv"
Q4_RECOVERY_PRE_OUTPUT_HASHES = "pre_recovery_output_hashes.csv"
Q4_RECOVERY_ORIGINAL_MANIFEST_SHA256 = {
    "source_hashes.csv": (
        "723c57ce463c7762bbcc004da4ff2dc126e091e74724fc324e9692567d5e9781"
    ),
    "code_hashes.csv": (
        "66a88b9fa67d433ae3841a5eda3a26b905207282606eb303cf29e7357e07df45"
    ),
    "config_hashes.csv": (
        "7f6195adb3bd3a085f8d87e504349a3e81fb6e14012207084cb900f34600c7dc"
    ),
}
Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256 = (
    "758882ace96d5fed69863c9ffdb83113610d3b8c1933cfdf3ba08ffb645d7f1e"
)
Q4_RECOVERY_ORIGINAL_STATUS_SHA256 = (
    "30fcf84ef43775b3b8af6a7b950d241841b8ab4f89b91be679957705fda2561e"
)

# The formal root reached a fully evaluated Q4 state under q4_recovery_v1, then
# postprocessing stopped before writing resource_summary.csv because the
# telemetry producer uses the canonical ``utilization_gpu_pct`` column while
# the consumer requested the obsolete ``utilization_gpu_percent`` spelling.
# These anchors authorize one additional, orchestrator-only repair without
# rewriting either the prepared manifests or the q4_recovery_v1 bundle.
POSTPROCESS_RECOVERY_INCIDENT = "resource_telemetry_column_name_v2"
POSTPROCESS_RECOVERY_DIRECTORY = "registry/postprocess_recovery_v2"
POSTPROCESS_RECOVERY_LEDGER = "ledger.json"
POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES = "code_hashes_current.csv"
POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES = "pre_recovery_output_hashes.csv"
POSTPROCESS_RECOVERY_PREVIOUS_ORCHESTRATOR_SHA256 = (
    "75838aa09c23fdf3df18a3ea8eba7a36bb0fc4e21737b7ab28ab1579ec77e97e"
)
POSTPROCESS_RECOVERY_PREVIOUS_ORCHESTRATOR_SIZE = 112_644
POSTPROCESS_RECOVERY_V1_ANCHORS = {
    Q4_RECOVERY_LEDGER: (
        "21d33f2ce4ea0d4b8077d41e2f5b49e2d43c082ee532f321807368ef6c490b3b"
    ),
    Q4_RECOVERY_CURRENT_CODE_HASHES: (
        "d8ef04bf94772379d1de28e62bb2ac9bf12723c39b433c5292afb59f86a9eccd"
    ),
    Q4_RECOVERY_PRE_OUTPUT_HASHES: (
        "94637c55155ce0491e7bc08e4d38732995e11f8465627a84fbd3b1b02e18ead9"
    ),
}
POSTPROCESS_RECOVERY_PRE_REGISTRY_SHA256 = (
    "5cd2979d2c907f42e185cbacdec095ffd52f1e2314e24ffb215a7d8f92c0bf46"
)
POSTPROCESS_RECOVERY_PRE_STATUS_SHA256 = (
    "c66fa4cd63c52de4b128ebbeda581f5cf821b20880716373c49c2fce9190fb48"
)
POSTPROCESS_RECOVERY_PARTIAL_FILE_SHA256 = {
    "report/film_nolp_capacity_conclusion.md": (
        "ffef428ce49f37487903ce20032d46ba0cfe5a4da18aecf8b838a630bc42c2b4"
    ),
    "report/film_nolp_capacity_conclusion.html": (
        "b11621e74bb192f706c3fab7435df944e96f06db1318113a007b80779b4bf07d"
    ),
    "resource_usage.csv": (
        "dd44a9d66c7348388dd6a9539fd4928d2aaef69187dee477b9ba3b1c34270f21"
    ),
    "analysis/film_nolp_capacity_q4_summary.json": (
        "8fab8ede61caf635d05a17560ef435ee572a9202f53fd4477d12482337c82b25"
    ),
    "data_windows/q4/q4_window_manifest.json": (
        "7fd061d85289be08e48f4fe257a7d4fd4b012a9d068f024bf6f8efbad10759fa"
    ),
    "data_windows/q4/common_05m_q4.xlsx": (
        "d86414013f086f64f2f4ab672b85afba478d2bb8c263e70c865f4a695b2d8c1c"
    ),
}
POSTPROCESS_RECOVERY_ABSENT_PATHS = (
    "resource_summary.csv",
    "output_hashes.csv",
    "qa.json",
    "registry/final_registry_snapshot.json",
)

_read_json = training._read_json
_write_json = training._write_json
_write_yaml = training._write_yaml
_write_csv = training._write_csv
_sha256_file = training._sha256_file
_payload_sha256 = training._payload_sha256
_utc_now = training._utc_now
_require_mapping = training._require_mapping
_resolve_repo_path = training._resolve_repo_path
_job_status_path = training._job_status_path


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    return _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)


def _sweep(resolved: Mapping[str, Any]) -> dict[str, Any]:
    return _require_mapping(
        resolved.get("film_nolp_capacity_seed_sweep"),
        "film_nolp_capacity_seed_sweep",
    )


def _profiles(resolved: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw = _require_mapping(_sweep(resolved).get("profiles"), "profiles")
    if tuple(raw) != PROFILES:
        raise ValueError(f"Capacity order must be {PROFILES}")
    expected_fields = {
        *training.CAPACITY_PROFILE_SHAPE_FIELDS,
        "expected_generator_parameters",
        "expected_critic_parameters",
        "expected_wgan_parameters",
    }
    result: dict[str, dict[str, int]] = {}
    for name, values_raw in raw.items():
        values = _require_mapping(values_raw, f"profiles.{name}")
        if set(values) != expected_fields:
            raise ValueError(f"Profile fields drifted for {name}")
        result[name] = {key: int(value) for key, value in values.items()}
        frozen = training.FROZEN_CAPACITY_PROFILES[name]
        for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
            if result[name][field] != int(frozen[field]):
                raise ValueError(f"Frozen capacity width drift: {name}.{field}")
    return result


def _validate_config(resolved: Mapping[str, Any]) -> None:
    datasets = _require_mapping(resolved.get("datasets"), "datasets")
    split = _require_mapping(resolved.get("split"), "split")
    sweep = _sweep(resolved)
    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if (
        not bool(sweep.get("enabled"))
        or sweep.get("experiment_kind") != EXPERIMENT_KIND
    ):
        raise ValueError("Capacity experiment kind/enabled contract drift")
    if tuple(map(int, datasets.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("Dataset tolerances must be [5, 30]")
    if tuple(map(str, datasets.get("text_ablation_modes", ()))) != (REAL_TEXT,):
        raise ValueError("Only real_text is permitted")
    if str(datasets.get("support_mask_mode")) != "raw_joint":
        raise ValueError("support_mask_mode must be raw_joint")
    if tuple(map(str, sweep.get("capacity_profiles", ()))) != PROFILES:
        raise ValueError("Capacity matrix drift")
    if tuple(map(int, sweep.get("seeds", ()))) != SEEDS:
        raise ValueError("Seeds must be 42/202/404")
    if tuple(map(int, sweep.get("tolerances_minutes", ()))) != TOLERANCES:
        raise ValueError("Sweep tolerances must be 5/30")
    if tuple(map(str, sweep.get("text_ablation_modes", ()))) != (REAL_TEXT,):
        raise ValueError("Sweep text mode must be real_text")
    if sweep.get("generator_conditioning_mode") != GENERATOR_MODE:
        raise ValueError("Generator mode drift")
    if sweep.get("critic_conditioning_mode") != CRITIC_MODE:
        raise ValueError("Critic mode drift")
    if sweep.get("refit_mode") != REFIT_MODE:
        raise ValueError("Refit mode drift")
    for key, expected in (
        ("development_train_end_utc", "2023-07-01T00:00:00Z"),
        ("development_validation_end_utc", "2023-10-01T00:00:00Z"),
        ("refit_train_end_utc", "2023-10-01T00:00:00Z"),
        ("q4_start_utc", "2023-10-01T00:00:00Z"),
        ("q4_end_utc", "2024-01-01T00:00:00Z"),
    ):
        if str(split.get(key)) != expected:
            raise ValueError(f"Frozen split boundary drift: {key}")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("Validation MC must be 16")
    if int(split.get("q4_mc_samples", -1)) != 64:
        raise ValueError("Q4 MC must be 64")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Exactly physical GPU0/GPU1 are required")
    if int(runtime.get("benchmark_workers_per_gpu", -1)) != 18:
        raise ValueError("Benchmark concurrency must be 18 workers/GPU")
    if int(runtime.get("fallback_slots_per_gpu", -1)) != 12:
        raise ValueError("Fallback concurrency must be 12 workers/GPU")
    if int(runtime.get("slots_per_gpu", -1)) not in {12, 18}:
        raise ValueError("Formal concurrency must be 12 or 18 workers/GPU")
    configured_q4_counts = {
        str(key): int(value)
        for key, value in _require_mapping(
            sweep.get("expected_q4_common_5m_counts"),
            "expected_q4_common_5m_counts",
        ).items()
    }
    if configured_q4_counts != EXPECTED_Q4_COMMON_COUNTS:
        raise ValueError("Frozen Q4 common-panel counts drifted")
    _profiles(resolved)
    model = _require_mapping(
        _require_mapping(resolved.get("models"), "models").get("wgan"), "wgan"
    )
    train = _require_mapping(model.get("training"), "wgan.training")
    frozen = {
        "embedding_dim": 1024,
        "noise_dim": 32,
        "generator_noise_mode": "gaussian",
        "generator_current_input_mode": "current_support_masked",
        "generator_conditioning_mode": GENERATOR_MODE,
        "critic_conditioning_mode": CRITIC_MODE,
        "critic_normalization_mode": "legacy_instance_norm_v1",
        "residual_output_mode": "identity_softplus_residual",
        "batch_size": 16,
        "discriminator_iter": 5,
        "news_first_label_reliability_mode": "none",
    }
    for key, expected in frozen.items():
        if train.get(key) != expected:
            raise ValueError(f"Frozen training contract drift: {key}")
    for key, expected in (
        ("generator_learning_rate", 5e-7),
        ("discriminator_learning_rate", 5e-7),
        ("reduce_lr_min_lr", 5e-8),
    ):
        value = float(train.get(key, math.nan))
        if not math.isfinite(value) or value != expected:
            raise ValueError(f"Frozen LR contract drift: {key}")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _load_yaml(source, "capacity config")
    resolved = deepcopy(_require_mapping(root.get(ROOT_KEY), ROOT_KEY))
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    resolved["source_config_path"] = str(source)
    _validate_config(resolved)
    return resolved


def _grid_contract(resolved: Mapping[str, Any]) -> dict[str, Any]:
    sweep = _sweep(resolved)
    payload = {
        "schema_version": 1,
        "support_method": "raw_bracket_intersection_v1",
        "strike_grid": [float(value) for value in sweep["strike_grid"]],
        "maturity_days_grid": [int(value) for value in sweep["maturity_days_grid"]],
        "surface_shape": [16, 16],
    }
    output = {
        **payload,
        "surface_grid_profile": str(sweep["surface_grid_profile"]),
        "surface_grid_sha256": _payload_sha256(payload),
    }
    expected = factorial._surface_grid_contract()
    if output["surface_grid_sha256"] != expected["surface_grid_sha256"]:
        raise ValueError("Exact-TTM grid SHA drift")
    return output


def _model_contract(resolved: Mapping[str, Any], profile: str) -> dict[str, Any]:
    values = _profiles(resolved)[profile]
    shape = {field: values[field] for field in training.CAPACITY_PROFILE_SHAPE_FIELDS}
    architecture = {
        "schema_version": 1,
        "capacity_profile": profile,
        **shape,
    }
    architecture_sha = _payload_sha256(architecture)
    grid = _grid_contract(resolved)
    conditioning = {
        "generator_conditioning_mode": GENERATOR_MODE,
        "generator_conditioning_fingerprint": generator_conditioning_fingerprint(
            GENERATOR_MODE
        ),
        "critic_conditioning_mode": CRITIC_MODE,
        "critic_conditioning_fingerprint": critic_conditioning_fingerprint(CRITIC_MODE),
    }
    payload = {
        "schema_version": 1,
        **conditioning,
        "conditioning_contract_sha256": _payload_sha256(conditioning),
        "capacity_profile": profile,
        "architecture_profile_sha256": architecture_sha,
        "architecture": architecture,
        "surface_grid_profile": grid["surface_grid_profile"],
        "surface_grid_sha256": grid["surface_grid_sha256"],
        "embedding_mode": "lp",
        "embedding_dim": 1024,
        "generator_current_input_mode": "current_support_masked",
        "generator_noise_mode": "gaussian",
        "noise_dim": 32,
        "initial_learning_rate": 5e-7,
        "expected_generator_parameters": values["expected_generator_parameters"],
        "expected_critic_parameters": values["expected_critic_parameters"],
        "expected_wgan_parameters": values["expected_wgan_parameters"],
    }
    payload["model_contract_sha256"] = _payload_sha256(payload)
    return payload


def experiment_specs(stage: str) -> list[dict[str, Any]]:
    if stage not in {DEVELOPMENT_STAGE, REFIT_STAGE, BENCHMARK_STAGE}:
        raise ValueError(f"Unknown stage: {stage}")
    return [
        {
            "experiment_stage": stage,
            "capacity_profile": profile,
            "seed": seed,
            "tolerance_minutes": tolerance,
            "text_ablation_mode": REAL_TEXT,
            "generator_conditioning_mode": GENERATOR_MODE,
            "critic_conditioning_mode": CRITIC_MODE,
        }
        for profile in PROFILES
        for seed in SEEDS
        for tolerance in TOLERANCES
    ]


def _assign(
    specs: Sequence[Mapping[str, Any]], *, slots_per_gpu: int, invert: bool
) -> list[dict[str, Any]]:
    counts = {0: 0, 1: 0}
    rows: list[dict[str, Any]] = []
    for raw in specs:
        row = dict(raw)
        parity = (
            PROFILES.index(str(row["capacity_profile"]))
            + SEEDS.index(int(row["seed"]))
            + TOLERANCES.index(int(row["tolerance_minutes"]))
            + int(invert)
        ) % 2
        gpu = (0, 1)[parity]
        local = counts[gpu]
        counts[gpu] += 1
        row.update(
            gpu_id=gpu,
            gpu_slot=local % int(slots_per_gpu),
            wave=local // int(slots_per_gpu) + 1,
        )
        rows.append(row)
    _assert_gpu_balance(rows)
    return rows


def _assert_gpu_balance(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_STAGE_JOBS:
        raise ValueError(f"Expected {EXPECTED_STAGE_JOBS} jobs")
    for factor in ("capacity_profile", "seed", "tolerance_minutes"):
        for value in {row[factor] for row in rows}:
            counts = [
                sum(row[factor] == value and int(row["gpu_id"]) == gpu for row in rows)
                for gpu in (0, 1)
            ]
            if counts[0] != counts[1]:
                raise ValueError(f"GPU imbalance for {factor}={value}: {counts}")


def _job_id(stage: str, spec: Mapping[str, Any]) -> str:
    prefix = {DEVELOPMENT_STAGE: "dev", REFIT_STAGE: "refit", BENCHMARK_STAGE: "bench"}[
        stage
    ]
    return (
        f"{prefix}_film_nolp_{spec['capacity_profile']}_lr_5e_07_"
        f"seed_{int(spec['seed']):03d}_real_{int(spec['tolerance_minutes']):02d}m"
    )


def _training_payload(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    spec: Mapping[str, Any],
    stage: str,
    recipe_path: Path | None = None,
    benchmark: bool = False,
) -> dict[str, Any]:
    base = factorial._training_payload(
        resolved,
        root,
        spec=spec,
        stage=factorial.REFIT_STAGE
        if stage == REFIT_STAGE
        else factorial.DEVELOPMENT_STAGE,
        recipe_path=recipe_path,
        benchmark=benchmark,
    )
    profile = str(spec["capacity_profile"])
    values = _profiles(resolved)[profile]
    contract = _model_contract(resolved, profile)
    for field in training.CAPACITY_PROFILE_SHAPE_FIELDS:
        base[field] = values[field]
    run_stage = BENCHMARK_STAGE if benchmark else stage
    base.update(
        {
            "news_first_capacity_profile": profile,
            "news_first_architecture_profile_sha256": contract[
                "architecture_profile_sha256"
            ],
            "news_first_model_contract_sha256": contract["model_contract_sha256"],
            "news_first_surface_grid_profile": contract["surface_grid_profile"],
            "news_first_surface_grid_sha256": contract["surface_grid_sha256"],
            "seed": int(spec["seed"]),
            "output_root": str(
                (
                    root
                    / "runs"
                    / run_stage
                    / profile
                    / "lr_5e_07"
                    / f"seed_{int(spec['seed']):03d}"
                    / "real_text"
                    / f"tolerance_{int(spec['tolerance_minutes']):02d}m"
                ).resolve()
            ),
        }
    )
    return base


def _manifest_row(role: str, path: Path) -> dict[str, Any]:
    return {
        "artifact_role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_hash_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_csv(path, [dict(row) for row in rows], tuple(rows[0]))


def _verify_hash_rows(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty hash manifest: {path}")
    for row in rows:
        target = Path(row["path"])
        if (
            not target.is_file()
            or target.stat().st_size != int(row["size_bytes"])
            or _sha256_file(target) != row["sha256"]
        ):
            raise ValueError(f"Hash drift: {target}")


def _read_hash_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Empty hash manifest: {path}")
    roles = [row["artifact_role"] for row in rows if "artifact_role" in row]
    paths = [row["path"] for row in rows if "path" in row]
    if (roles and len(roles) != len(set(roles))) or (
        paths and len(paths) != len(set(paths))
    ):
        raise ValueError(f"Hash-manifest uniqueness drift: {path}")
    return rows


def _recovery_directory(root: Path) -> Path:
    return root / Q4_RECOVERY_DIRECTORY


def _recovery_ledger_path(root: Path) -> Path:
    return _recovery_directory(root) / Q4_RECOVERY_LEDGER


def _validate_original_recovery_manifest_anchors(root: Path) -> None:
    for name, expected_sha256 in Q4_RECOVERY_ORIGINAL_MANIFEST_SHA256.items():
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"Original prepared manifest anchor drift: {path}")


def _validate_recovery_code_candidate(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Allow exactly the known orchestrator fix and no other code drift."""

    original_path = root / "code_hashes.csv"
    original_rows = _read_hash_rows(original_path)
    target_path = Path(__file__).resolve()
    target_role = "code:scripts/rq3/news_first_vol_film_nolp_capacity_seed.py"
    changed: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    for row in original_rows:
        path = Path(row["path"])
        if not path.is_file():
            raise ValueError(f"Recovery code file is missing: {path}")
        actual = _manifest_row(str(row["artifact_role"]), path)
        current_rows.append(actual)
        differs = (
            int(row["size_bytes"]) != int(actual["size_bytes"])
            or row["sha256"] != actual["sha256"]
        )
        if not differs:
            continue
        if (
            row["artifact_role"] != target_role
            or path.resolve() != target_path
            or int(row["size_bytes"]) != Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SIZE
            or row["sha256"] != Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SHA256
        ):
            raise ValueError(f"Unapproved recovery code drift: {path}")
        changed.append(
            {
                "artifact_role": target_role,
                "path": str(target_path),
                "original_size_bytes": int(row["size_bytes"]),
                "original_sha256": row["sha256"],
                "recovery_size_bytes": int(actual["size_bytes"]),
                "recovery_sha256": actual["sha256"],
            }
        )
    if len(changed) != 1:
        raise ValueError("Q4 recovery requires exactly one approved code drift")
    return current_rows, changed[0]


def _validate_registry_job_lineage(root: Path, registry: Mapping[str, Any]) -> None:
    seen: set[str] = set()
    for job in registry.get("jobs", []):
        if job["job_id"] in seen or _job_spec_sha(job) != job["job_spec_sha256"]:
            raise ValueError("Job registry uniqueness/spec SHA drift")
        seen.add(str(job["job_id"]))
        config = Path(job["training_config_path"])
        dataset = Path(job["dataset_path"])
        if (
            not config.is_file()
            or not dataset.is_file()
            or _sha256_file(config) != job["config_sha256"]
            or _sha256_file(dataset) != job["dataset_sha256"]
        ):
            raise ValueError(f"Job config/dataset drift: {job['job_id']}")


def _pre_recovery_output_rows(root: Path) -> list[dict[str, Any]]:
    recovery = _recovery_directory(root).resolve()
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and recovery not in path.resolve().parents
        and not path.name.startswith(".q4_recovery_v1.")
    )
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _recovery_training_evidence(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    jobs = [dict(job) for job in registry.get("jobs", [])]
    if len(jobs) != EXPECTED_TOTAL_JOBS:
        raise ValueError("Recovery requires exactly 72 registered jobs")
    status_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    checkpoint_count = 0
    for job in jobs:
        status_path = _job_status_path(root, str(job["job_id"]))
        status = _require_mapping(_read_json(status_path), "job status")
        expected_roles = (
            DEVELOPMENT_ARTIFACT_ROLES
            if job["experiment_stage"] == DEVELOPMENT_STAGE
            else REFIT_ARTIFACT_ROLES
        )
        artifacts = list(status.get("artifacts") or [])
        if (
            status.get("status") != "completed"
            or status.get("config_sha256") != job["config_sha256"]
            or tuple(row.get("artifact_role") for row in artifacts) != expected_roles
            or not _completed_valid(job, status)
        ):
            raise ValueError(f"Recovery job artifact drift: {job['job_id']}")
        status_rows.append(
            {
                "job_id": str(job["job_id"]),
                "path": str(status_path.resolve()),
                "sha256": _sha256_file(status_path),
            }
        )
        for artifact in artifacts:
            normalized = {
                "job_id": str(job["job_id"]),
                "artifact_role": str(artifact["artifact_role"]),
                "path": str(Path(artifact["path"]).resolve()),
                "size_bytes": int(artifact["size_bytes"]),
                "sha256": str(artifact["sha256"]),
            }
            artifact_rows.append(normalized)
            if Path(normalized["path"]).suffix == ".pt":
                checkpoint_count += 1
    if len(status_rows) != 72 or len(artifact_rows) != 756:
        raise ValueError("Recovery requires 72 statuses and 756 training artifacts")
    if checkpoint_count != 360:
        raise ValueError("Recovery requires exactly 360 checkpoint artifacts")
    return {
        "completed_job_status_count": len(status_rows),
        "required_training_artifact_count": len(artifact_rows),
        "checkpoint_artifact_count": checkpoint_count,
        "job_status_universe_sha256": _payload_sha256(
            sorted(status_rows, key=lambda row: row["job_id"])
        ),
        "training_artifact_universe_sha256": _payload_sha256(
            sorted(
                artifact_rows,
                key=lambda row: (row["job_id"], row["artifact_role"]),
            )
        ),
    }


def _recovery_q3_prediction_evidence(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    directory = root / "analysis" / "predictions" / "q3_development_best_learned"
    jobs = {
        str(job["job_id"]): dict(job)
        for job in registry.get("jobs", [])
        if job.get("experiment_stage") == DEVELOPMENT_STAGE
    }
    if len(jobs) != EXPECTED_STAGE_JOBS:
        raise ValueError("Recovery requires all 36 development jobs")
    csv_paths = {
        path.name.removesuffix(".csv.gz"): path for path in directory.glob("*.csv.gz")
    }
    manifest_paths = {
        path.name.removesuffix(".manifest.json"): path
        for path in directory.glob("*.manifest.json")
    }
    if set(csv_paths) != set(jobs) or set(manifest_paths) != set(jobs):
        raise ValueError("Recovery requires 36 Q3 predictions and 36 manifests")
    evidence_rows: list[dict[str, Any]] = []
    panel_hashes: set[str] = set()
    for job_id, job in jobs.items():
        csv_path = csv_paths[job_id]
        manifest_path = manifest_paths[job_id]
        manifest = _require_mapping(_read_json(manifest_path), "Q3 prediction manifest")
        unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        status = _require_mapping(
            _read_json(_job_status_path(root, job_id)), "development job status"
        )
        learned = [
            row
            for row in status.get("artifacts", [])
            if row.get("artifact_role") == "generator_best_learned"
        ]
        if (
            len(learned) != 1
            or _payload_sha256(unsigned) != manifest.get("manifest_sha256")
            or manifest.get("job_id") != job_id
            or manifest.get("job_spec_sha256") != job["job_spec_sha256"]
            or manifest.get("checkpoint_sha256") != learned[0]["sha256"]
            or int(manifest.get("mc_samples", -1)) != 16
            or Path(str(manifest.get("prediction_path", ""))).resolve()
            != csv_path.resolve()
            or _sha256_file(csv_path) != manifest.get("prediction_sha256")
            or int(manifest.get("row_count", -1)) != 148
        ):
            raise ValueError(f"Q3 prediction lineage drift: {job_id}")
        panel_hashes.add(str(manifest.get("panel_universe_sha256")))
        evidence_rows.append(
            {
                "job_id": job_id,
                "prediction_sha256": str(manifest["prediction_sha256"]),
                "manifest_sha256": _sha256_file(manifest_path),
                "manifest_payload_sha256": str(manifest["manifest_sha256"]),
            }
        )
    if len(panel_hashes) != 1:
        raise ValueError("Q3 prediction panel universe drift")
    return {
        "q3_prediction_count": len(csv_paths),
        "q3_prediction_manifest_count": len(manifest_paths),
        "q3_panel_universe_sha256": next(iter(panel_hashes)),
        "q3_prediction_universe_sha256": _payload_sha256(
            sorted(evidence_rows, key=lambda row: row["job_id"])
        ),
    }


def _recovery_immutable_evidence(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_registry_job_lineage(root, registry)
    _validate_stage(root, DEVELOPMENT_STAGE)
    _validate_stage(root, REFIT_STAGE)
    _validate_frozen_selection(root)
    allowlist = _validate_allowlist(root)
    if len(allowlist) != 72:
        raise ValueError("Recovery requires exactly 72 allowlist rows")
    allowlist_evidence = [
        {
            "job_id": row["job_id"],
            "checkpoint_role": row["checkpoint_role"],
            "checkpoint_path": str(Path(row["checkpoint_path"]).resolve()),
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        for row in allowlist
    ]
    return {
        **_recovery_training_evidence(root, registry),
        **_recovery_q3_prediction_evidence(root, registry),
        "q4_allowlist_row_count": len(allowlist),
        "q4_allowlist_universe_sha256": _payload_sha256(
            sorted(
                allowlist_evidence,
                key=lambda row: (row["job_id"], row["checkpoint_role"]),
            )
        ),
    }


def _code_paths() -> list[Path]:
    relatives = (
        "scripts/rq3/main.py",
        "scripts/rq3/news_first_vol_film_nolp_capacity_seed.py",
        "scripts/rq3/news_first_vol_film_nolp_capacity_seed_analysis.py",
        "scripts/rq3/news_first_vol_film_nolp_capacity_seed_report.py",
        "scripts/rq3/news_first_vol_generator_film_critic_factorial.py",
        "scripts/rq3/news_first_vol_comparison_analysis.py",
        "scripts/rq3/news_first_vol_training.py",
        "src/wgan_option/config.py",
        "src/wgan_option/models/common.py",
        "src/wgan_option/models/generator.py",
        "src/wgan_option/models/discriminator.py",
        "src/wgan_option/models/gan_model.py",
        "src/wgan_option/train_vol_xlsx.py",
        "src/wgan_option/utils/news_first_dataloaders.py",
        "src/wgan_option/utils/inference_helpers.py",
    )
    return [(REPO_ROOT / relative).resolve() for relative in relatives]


def _job_spec_sha(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _build_jobs(
    resolved: Mapping[str, Any],
    root: Path,
    *,
    stage: str,
    slots_per_gpu: int,
    recipes: Mapping[str, Path] | None = None,
    benchmark: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec_stage = BENCHMARK_STAGE if benchmark else stage
    specs = _assign(
        experiment_specs(spec_stage),
        slots_per_gpu=slots_per_gpu,
        invert=stage == REFIT_STAGE,
    )
    jobs: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for spec in specs:
        profile = str(spec["capacity_profile"])
        development_id = _job_id(DEVELOPMENT_STAGE, spec)
        recipe = None if recipes is None else recipes.get(development_id)
        payload = _training_payload(
            resolved,
            root,
            spec=spec,
            stage=stage,
            recipe_path=recipe,
            benchmark=benchmark,
        )
        job_stage = BENCHMARK_STAGE if benchmark else stage
        job_id = _job_id(job_stage, spec)
        config_path = (root / "configs" / job_stage / f"{job_id}.yaml").resolve()
        _write_yaml(config_path, payload)
        contract = _model_contract(resolved, profile)
        job = {
            "job_id": job_id,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": job_stage,
            "model_family": "wgan",
            "capacity_profile": profile,
            "capacity_profile_sha256": contract["architecture_profile_sha256"],
            "lr_profile": "lr_5e_07",
            "generator_conditioning_mode": GENERATOR_MODE,
            "critic_conditioning_mode": CRITIC_MODE,
            "generator_conditioning_fingerprint": contract[
                "generator_conditioning_fingerprint"
            ],
            "critic_conditioning_fingerprint": contract[
                "critic_conditioning_fingerprint"
            ],
            "conditioning_contract_sha256": contract["conditioning_contract_sha256"],
            "architecture_profile_sha256": contract["architecture_profile_sha256"],
            "model_contract_sha256": contract["model_contract_sha256"],
            "surface_grid_profile": contract["surface_grid_profile"],
            "surface_grid_sha256": contract["surface_grid_sha256"],
            "expected_generator_parameters": contract["expected_generator_parameters"],
            "expected_critic_parameters": contract["expected_critic_parameters"],
            "expected_wgan_parameters": contract["expected_wgan_parameters"],
            "text_ablation_mode": REAL_TEXT,
            "support_mask_mode": "raw_joint",
            "tolerance_minutes": int(spec["tolerance_minutes"]),
            "seed": int(spec["seed"]),
            "gpu_id": int(spec["gpu_id"]),
            "gpu_slot": int(spec["gpu_slot"]),
            "wave": int(spec["wave"]),
            "training_config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "dataset_path": str(Path(payload["data_path"]).resolve()),
            "dataset_sha256": _sha256_file(Path(payload["data_path"])),
            "output_root": str(Path(payload["output_root"]).resolve()),
            "development_job_id": development_id,
            "parent_development_job_id": development_id if stage == REFIT_STAGE else "",
            "refit_recipe_path": str(recipe.resolve()) if recipe else "",
            "refit_recipe_sha256": _sha256_file(recipe) if recipe else "",
        }
        job["job_spec_sha256"] = _job_spec_sha(job)
        jobs.append(job)
        config_rows.append(_manifest_row(f"training_config:{job_id}", config_path))
    if len(jobs) != EXPECTED_STAGE_JOBS:
        raise ValueError("Job matrix count drift")
    if len({job["job_id"] for job in jobs}) != EXPECTED_STAGE_JOBS:
        raise ValueError("Duplicate job ID")
    if len({job["output_root"] for job in jobs}) != EXPECTED_STAGE_JOBS:
        raise ValueError("Duplicate job output root")
    return jobs, config_rows


def _initial_status(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "experiment_stage": job["experiment_stage"],
        "status": "prepared",
        "attempt": 0,
        "config_sha256": job["config_sha256"],
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluator_rows": 0,
        "artifacts": [],
    }


def _registry(root: Path) -> dict[str, Any]:
    return _require_mapping(_read_json(root / "registry" / "jobs.json"), "registry")


def _write_registry(root: Path, payload: Mapping[str, Any]) -> None:
    _write_json(root / "registry" / "jobs.json", dict(payload))
    rows = []
    for job in payload.get("jobs", []):
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        rows.append(
            {**job, "status": status.get("status"), "attempt": status.get("attempt")}
        )
    if rows:
        _write_csv(root / "task_registry.csv", rows, tuple(rows[0]))


def _write_experiment_status(root: Path, status: str, **details: Any) -> Path:
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": status,
        **details,
        "updated_at_utc": _utc_now(),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _write_json(root / "registry" / "experiment_status.json", payload)


def _benchmark_root(formal_root: Path) -> Path:
    return formal_root.with_name(formal_root.name + "_benchmark")


def _benchmark_result_path(formal_root: Path) -> Path:
    return _benchmark_root(formal_root) / "benchmark_result.json"


def _snapshot_sha(paths: Sequence[Path]) -> str:
    return _payload_sha256(
        [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in paths
        ]
    )


def _prepare_root(
    config_path: str | Path,
    root: Path,
    *,
    slots_per_gpu: int,
    benchmark: bool,
) -> Path:
    resolved = resolve_config(config_path)
    resolved["runtime"]["slots_per_gpu"] = int(slots_per_gpu)
    _validate_config(resolved)
    for relative in (
        "analysis",
        "configs",
        "data_windows",
        "logs",
        "registry/jobs",
        "report",
        "runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    resolved_path = _write_yaml(root / "resolved_config.yaml", {ROOT_KEY: resolved})
    window_rows, window_manifest = factorial._materialize_pre_q4_windows(resolved, root)
    split_manifest = factorial._write_split_manifest(resolved, root)
    jobs, job_configs = _build_jobs(
        resolved,
        root,
        stage=DEVELOPMENT_STAGE,
        slots_per_gpu=slots_per_gpu,
        benchmark=benchmark,
    )
    contracts = [_model_contract(resolved, profile) for profile in PROFILES]
    model_manifest = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "profiles": contracts,
    }
    model_manifest["payload_sha256"] = _payload_sha256(model_manifest)
    model_path = _write_json(root / "model_contract_manifest.json", model_manifest)
    source_rows = [
        _manifest_row("source_config", _resolve_repo_path(config_path)),
        *[
            _manifest_row(
                f"source_workbook_{tol:02d}m",
                Path(resolved["datasets"]["root"])
                / str(resolved["datasets"]["workbook_template"]).format(
                    tolerance=tol, tolerance02=f"{tol:02d}"
                ),
            )
            for tol in TOLERANCES
        ],
        *window_rows,
    ]
    code_rows = [
        _manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in _code_paths()
    ]
    config_rows = [
        _manifest_row("resolved_config", resolved_path),
        _manifest_row("rolling_split_manifest", split_manifest),
        _manifest_row("pre_q4_window_manifest", window_manifest),
        _manifest_row("model_contract_manifest", model_path),
        *job_configs,
    ]
    _write_hash_rows(root / "source_hashes.csv", source_rows)
    _write_hash_rows(root / "code_hashes.csv", code_rows)
    _write_hash_rows(root / "config_hashes.csv", config_rows)
    registry = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "benchmark_prepared" if benchmark else "development_prepared",
        "benchmark_root": bool(benchmark),
        "slots_per_gpu": int(slots_per_gpu),
        "development_job_count": 0 if benchmark else EXPECTED_STAGE_JOBS,
        "benchmark_job_count": EXPECTED_STAGE_JOBS if benchmark else 0,
        "refit_job_count": 0,
        "selection_frozen": False,
        "refit_complete": False,
        "q4_gate_open": False,
        "q4_window_materialized": False,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "jobs": jobs,
        "created_at_utc": _utc_now(),
    }
    _write_json(root / "registry" / "jobs.json", registry)
    for job in jobs:
        _write_json(_job_status_path(root, str(job["job_id"])), _initial_status(job))
    _write_registry(root, registry)
    _write_experiment_status(
        root,
        registry["status"],
        current_stage=BENCHMARK_STAGE if benchmark else DEVELOPMENT_STAGE,
    )
    _validate_root(root)
    return root


def _recovery_snapshot_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "relative_path": str(row["relative_path"]),
            "path": str(row["path"]),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in rows
    ]
    return _payload_sha256(normalized)


def _validate_recovery_bundle(
    root: Path, *, require_registered: bool
) -> dict[str, Any]:
    directory = _recovery_directory(root)
    ledger_path = directory / Q4_RECOVERY_LEDGER
    current_code_path = directory / Q4_RECOVERY_CURRENT_CODE_HASHES
    snapshot_path = directory / Q4_RECOVERY_PRE_OUTPUT_HASHES
    for path in (ledger_path, current_code_path, snapshot_path):
        if not path.is_file():
            raise ValueError(f"Partial Q4 recovery bundle: {path}")
    ledger = _require_mapping(_read_json(ledger_path), "Q4 recovery ledger")
    unsigned = {key: value for key, value in ledger.items() if key != "payload_sha256"}
    if (
        ledger.get("schema_version") != 1
        or ledger.get("incident") != Q4_RECOVERY_INCIDENT
        or _payload_sha256(unsigned) != ledger.get("payload_sha256")
        or ledger.get("training_jobs_rewritten") is not False
        or int(ledger.get("q4_rows_read_before_recovery", -1)) != 0
        or ledger.get("pre_recovery_registry_sha256")
        != Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256
        or ledger.get("pre_recovery_experiment_status_sha256")
        != Q4_RECOVERY_ORIGINAL_STATUS_SHA256
    ):
        raise ValueError("Q4 recovery ledger contract drift")
    _validate_original_recovery_manifest_anchors(root)
    anchored = _require_mapping(ledger.get("anchored_manifests"), "anchored manifests")
    for name in ("source_hashes.csv", "config_hashes.csv", "code_hashes.csv"):
        path = root / name
        anchor = _require_mapping(anchored.get(name), f"anchored manifest {name}")
        if (
            anchor.get("path") != str(path.resolve())
            or int(anchor.get("size_bytes", -1)) != path.stat().st_size
            or anchor.get("sha256") != _sha256_file(path)
        ):
            raise ValueError(f"Original recovery manifest drift: {path}")
    _verify_hash_rows(root / "source_hashes.csv")
    _verify_hash_rows(root / "config_hashes.csv")
    current_rows, approved_change = _validate_recovery_code_candidate(root)
    recorded_changes = list(ledger.get("approved_code_changes") or [])
    if recorded_changes != [approved_change]:
        raise ValueError("Approved Q4 recovery code change drift")
    if str(current_code_path.resolve()) != ledger.get(
        "current_code_manifest_path"
    ) or _sha256_file(current_code_path) != ledger.get("current_code_manifest_sha256"):
        raise ValueError("Current recovery code manifest anchor drift")
    _verify_hash_rows(current_code_path)
    recorded_current = _read_hash_rows(current_code_path)
    normalized_current = [
        {
            "artifact_role": str(row["artifact_role"]),
            "path": str(Path(row["path"]).resolve()),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in recorded_current
    ]
    if normalized_current != current_rows:
        raise ValueError("Current recovery code universe drift")
    if str(snapshot_path.resolve()) != ledger.get(
        "pre_recovery_output_manifest_path"
    ) or _sha256_file(snapshot_path) != ledger.get(
        "pre_recovery_output_manifest_sha256"
    ):
        raise ValueError("Pre-recovery output snapshot anchor drift")
    snapshot_rows = _read_hash_rows(snapshot_path)
    if len(snapshot_rows) != int(
        ledger.get("pre_recovery_output_file_count", -1)
    ) or _recovery_snapshot_fingerprint(snapshot_rows) != ledger.get(
        "pre_recovery_output_universe_sha256"
    ):
        raise ValueError("Pre-recovery output snapshot contract drift")
    registry = _registry(root)
    evidence = _recovery_immutable_evidence(root, registry)
    if evidence != ledger.get("immutable_evidence"):
        raise ValueError("Recovered training/Q3 evidence drift")
    if require_registered:
        if (
            not bool(registry.get("q4_recovery_applied"))
            or registry.get("q4_recovery_incident") != Q4_RECOVERY_INCIDENT
            or registry.get("q4_recovery_ledger_path") != str(ledger_path.resolve())
            or registry.get("q4_recovery_ledger_sha256") != _sha256_file(ledger_path)
            or registry.get("q4_recovery_current_code_manifest_sha256")
            != _sha256_file(current_code_path)
            or registry.get("q4_recovery_pre_output_manifest_sha256")
            != _sha256_file(snapshot_path)
        ):
            raise ValueError("Registered Q4 recovery lineage drift")
    return ledger


def _validate_known_q4_recovery_candidate(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate the exact no-Q4-read failure before authorizing recovery."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    if _recovery_directory(root).exists():
        raise ValueError("Q4 recovery bundle already exists")
    _validate_original_recovery_manifest_anchors(root)
    if (
        _sha256_file(root / "registry" / "jobs.json")
        != Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256
        or _sha256_file(root / "registry" / "experiment_status.json")
        != Q4_RECOVERY_ORIGINAL_STATUS_SHA256
    ):
        raise ValueError("Original Q4 failure registry/status anchor drift")
    _verify_hash_rows(root / "source_hashes.csv")
    _verify_hash_rows(root / "config_hashes.csv")
    current_rows, approved_change = _validate_recovery_code_candidate(root)
    registry = _registry(root)
    if (
        registry.get("experiment_kind") != EXPERIMENT_KIND
        or registry.get("status") != "refit_complete_q4_locked"
        or not bool(registry.get("selection_frozen"))
        or not bool(registry.get("refit_complete"))
        or not bool(registry.get("q4_gate_open"))
        or any(
            bool(registry.get(field))
            for field in (
                "q4_window_materialized",
                "q4_loader_created",
                "q4_predictions_generated",
                "q4_evaluated",
            )
        )
        or len(registry.get("jobs", [])) != EXPECTED_TOTAL_JOBS
    ):
        raise ValueError("Root is not the exact recoverable Q4 failure state")
    status_path = root / "registry" / "experiment_status.json"
    status = _require_mapping(_read_json(status_path), "failed experiment status")
    unsigned_status = {
        key: value for key, value in status.items() if key != "payload_sha256"
    }
    if (
        status.get("experiment_kind") != EXPERIMENT_KIND
        or status.get("status") != "failed"
        or status.get("current_stage") != "q4"
        or status.get("error") != Q4_RECOVERY_FAILURE
        or _payload_sha256(unsigned_status) != status.get("payload_sha256")
    ):
        raise ValueError("Experiment status is not the known Q4 failure")
    forbidden = [
        root / "data_windows" / "q4",
        root / "analysis" / "predictions" / "q4_refit_final",
        root / "analysis" / "film_nolp_capacity_q4_pair_metrics.csv.gz",
        root / "analysis" / "film_nolp_capacity_q4_persistence_contrasts.csv",
        root / "analysis" / "film_nolp_capacity_q4_pairwise_contrasts.csv",
        root / "analysis" / "film_nolp_capacity_q4_summary.json",
    ]
    if any(path.exists() for path in forbidden):
        raise ValueError("Q4 recovery requires zero materialized Q4 artifacts")
    _recovery_immutable_evidence(root, registry)
    return registry, current_rows, approved_change


def _is_known_q4_failure_status(payload: Mapping[str, Any]) -> bool:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return bool(
        payload.get("experiment_kind") == EXPERIMENT_KIND
        and payload.get("status") == "failed"
        and payload.get("current_stage") == "q4"
        and payload.get("error") == Q4_RECOVERY_FAILURE
        and _payload_sha256(unsigned) == payload.get("payload_sha256")
    )


def _register_q4_recovery(root: Path) -> Path:
    """Atomically anchor the known code fix without touching trained artifacts."""

    final_directory = _recovery_directory(root)
    if final_directory.exists():
        ledger = _validate_recovery_bundle(root, require_registered=False)
    else:
        registry, current_code_rows, approved_change = (
            _validate_known_q4_recovery_candidate(root)
        )
        snapshot_rows = _pre_recovery_output_rows(root)
        temporary = root / "registry" / f".q4_recovery_v1.{os.getpid()}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        snapshot_path = _write_csv(
            temporary / Q4_RECOVERY_PRE_OUTPUT_HASHES,
            snapshot_rows,
            tuple(snapshot_rows[0]),
        )
        current_code_path = _write_csv(
            temporary / Q4_RECOVERY_CURRENT_CODE_HASHES,
            current_code_rows,
            tuple(current_code_rows[0]),
        )
        final_snapshot = final_directory / Q4_RECOVERY_PRE_OUTPUT_HASHES
        final_current_code = final_directory / Q4_RECOVERY_CURRENT_CODE_HASHES
        immutable_evidence = _recovery_immutable_evidence(root, registry)
        ledger = {
            "schema_version": 1,
            "incident": Q4_RECOVERY_INCIDENT,
            "reason": Q4_RECOVERY_FAILURE,
            "compatibility_boundary": "branch_local_unreleased_experiment_schema",
            "approved_code_changes": [approved_change],
            "anchored_manifests": {
                name: {
                    "path": str((root / name).resolve()),
                    "size_bytes": (root / name).stat().st_size,
                    "sha256": _sha256_file(root / name),
                }
                for name in (
                    "source_hashes.csv",
                    "config_hashes.csv",
                    "code_hashes.csv",
                )
            },
            "pre_recovery_registry_sha256": _sha256_file(
                root / "registry" / "jobs.json"
            ),
            "pre_recovery_experiment_status_sha256": _sha256_file(
                root / "registry" / "experiment_status.json"
            ),
            "pre_recovery_output_manifest_path": str(final_snapshot.resolve()),
            "pre_recovery_output_manifest_sha256": _sha256_file(snapshot_path),
            "pre_recovery_output_file_count": len(snapshot_rows),
            "pre_recovery_output_universe_sha256": (
                _recovery_snapshot_fingerprint(snapshot_rows)
            ),
            "current_code_manifest_path": str(final_current_code.resolve()),
            "current_code_manifest_sha256": _sha256_file(current_code_path),
            "immutable_evidence": immutable_evidence,
            "training_jobs_rewritten": False,
            "q4_rows_read_before_recovery": 0,
            "q4_broad_30m_rows_read_before_recovery": 0,
            "created_at_utc": _utc_now(),
        }
        ledger["payload_sha256"] = _payload_sha256(ledger)
        _write_json(temporary / Q4_RECOVERY_LEDGER, ledger)
        os.replace(temporary, final_directory)
        ledger = _validate_recovery_bundle(root, require_registered=False)
    registry = _registry(root)
    if bool(registry.get("q4_recovery_applied")):
        _validate_recovery_bundle(root, require_registered=True)
        status = _require_mapping(
            _read_json(root / "registry" / "experiment_status.json"),
            "experiment status",
        )
        if _is_known_q4_failure_status(status):
            _write_experiment_status(
                root,
                "refit_complete_q4_recovered_locked",
                current_stage="q4_recovery",
            )
        return _recovery_ledger_path(root)
    if _sha256_file(root / "registry" / "jobs.json") != ledger.get(
        "pre_recovery_registry_sha256"
    ) or _sha256_file(root / "registry" / "experiment_status.json") != ledger.get(
        "pre_recovery_experiment_status_sha256"
    ):
        raise ValueError("Recovery registration state changed after snapshot")
    registry.update(
        status="refit_complete_q4_recovered_locked",
        q4_gate_open=False,
        q4_recovery_applied=True,
        q4_recovery_incident=Q4_RECOVERY_INCIDENT,
        q4_recovery_ledger_path=str(_recovery_ledger_path(root).resolve()),
        q4_recovery_ledger_sha256=_sha256_file(_recovery_ledger_path(root)),
        q4_recovery_current_code_manifest_path=str(
            (_recovery_directory(root) / Q4_RECOVERY_CURRENT_CODE_HASHES).resolve()
        ),
        q4_recovery_current_code_manifest_sha256=_sha256_file(
            _recovery_directory(root) / Q4_RECOVERY_CURRENT_CODE_HASHES
        ),
        q4_recovery_pre_output_manifest_path=str(
            (_recovery_directory(root) / Q4_RECOVERY_PRE_OUTPUT_HASHES).resolve()
        ),
        q4_recovery_pre_output_manifest_sha256=_sha256_file(
            _recovery_directory(root) / Q4_RECOVERY_PRE_OUTPUT_HASHES
        ),
        q4_recovered_at_utc=_utc_now(),
    )
    _write_json(root / "registry" / "jobs.json", registry)
    _write_experiment_status(
        root, "refit_complete_q4_recovered_locked", current_stage="q4_recovery"
    )
    _validate_recovery_bundle(root, require_registered=True)
    return _recovery_ledger_path(root)


def _postprocess_recovery_directory(root: Path) -> Path:
    return root / POSTPROCESS_RECOVERY_DIRECTORY


def _postprocess_recovery_ledger_path(root: Path) -> Path:
    return _postprocess_recovery_directory(root) / POSTPROCESS_RECOVERY_LEDGER


def _reject_postprocess_recovery_temporary_directories(root: Path) -> None:
    registry_directory = root / "registry"
    temporary = sorted(
        path
        for path in registry_directory.glob(".postprocess_recovery_v2.*.tmp")
        if path.exists()
    )
    if temporary:
        raise ValueError(
            "Interrupted postprocess recovery temporary directory requires "
            f"explicit inspection/removal: {[str(path) for path in temporary]}"
        )


def _normalized_manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "artifact_role": str(row["artifact_role"]),
            "path": str(Path(row["path"]).resolve()),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in _read_hash_rows(path)
    ]


def _single_code_transition(
    previous_rows: Sequence[Mapping[str, Any]],
    *,
    target_role: str,
    target_path: Path,
    previous_size_bytes: int,
    previous_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate exactly one live-code transition from a historical manifest."""

    normalized_target = target_path.resolve()
    current_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for previous_raw in previous_rows:
        previous = {
            "artifact_role": str(previous_raw["artifact_role"]),
            "path": str(Path(str(previous_raw["path"])).resolve()),
            "size_bytes": int(previous_raw["size_bytes"]),
            "sha256": str(previous_raw["sha256"]),
        }
        path = Path(previous["path"])
        if not path.is_file():
            raise ValueError(f"Postprocess recovery code file is missing: {path}")
        actual = _manifest_row(previous["artifact_role"], path)
        current_rows.append(actual)
        differs = previous["size_bytes"] != int(actual["size_bytes"]) or previous[
            "sha256"
        ] != str(actual["sha256"])
        if not differs:
            continue
        if (
            previous["artifact_role"] != target_role
            or path.resolve() != normalized_target
            or previous["size_bytes"] != int(previous_size_bytes)
            or previous["sha256"] != previous_sha256
        ):
            raise ValueError(f"Unapproved postprocess recovery code drift: {path}")
        changes.append(
            {
                "artifact_role": target_role,
                "path": str(normalized_target),
                "previous_size_bytes": previous["size_bytes"],
                "previous_sha256": previous["sha256"],
                "recovery_size_bytes": int(actual["size_bytes"]),
                "recovery_sha256": str(actual["sha256"]),
            }
        )
    if len(changes) != 1:
        raise ValueError(
            "Postprocess recovery requires exactly one approved code drift"
        )
    return current_rows, changes[0]


def _validate_formal_q4_recovery_v1_anchor(root: Path) -> dict[str, Any]:
    """Validate the immutable v1 bundle while allowing only its target to advance."""

    directory = _recovery_directory(root)
    for name, expected_sha256 in POSTPROCESS_RECOVERY_V1_ANCHORS.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"Q4 recovery v1 anchor drift: {path}")
    ledger_path = directory / Q4_RECOVERY_LEDGER
    current_code_path = directory / Q4_RECOVERY_CURRENT_CODE_HASHES
    snapshot_path = directory / Q4_RECOVERY_PRE_OUTPUT_HASHES
    ledger = _require_mapping(_read_json(ledger_path), "Q4 recovery v1 ledger")
    unsigned = {key: value for key, value in ledger.items() if key != "payload_sha256"}
    if (
        ledger.get("schema_version") != 1
        or ledger.get("incident") != Q4_RECOVERY_INCIDENT
        or _payload_sha256(unsigned) != ledger.get("payload_sha256")
        or ledger.get("training_jobs_rewritten") is not False
        or int(ledger.get("q4_rows_read_before_recovery", -1)) != 0
    ):
        raise ValueError("Q4 recovery v1 ledger contract drift")
    _validate_original_recovery_manifest_anchors(root)
    _verify_hash_rows(root / "source_hashes.csv")
    _verify_hash_rows(root / "config_hashes.csv")

    original_rows = _normalized_manifest_rows(root / "code_hashes.csv")
    v1_rows = _normalized_manifest_rows(current_code_path)
    if [row["artifact_role"] for row in original_rows] != [
        row["artifact_role"] for row in v1_rows
    ]:
        raise ValueError("Q4 recovery v1 code universe drift")
    target_role = "code:scripts/rq3/news_first_vol_film_nolp_capacity_seed.py"
    target_path = Path(__file__).resolve()
    approved_changes: list[dict[str, Any]] = []
    for original, recovered in zip(original_rows, v1_rows, strict=True):
        if original["path"] != recovered["path"]:
            raise ValueError("Q4 recovery v1 code path drift")
        path = Path(recovered["path"])
        if recovered["artifact_role"] == target_role:
            if (
                path.resolve() != target_path
                or original["size_bytes"] != Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SIZE
                or original["sha256"] != Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SHA256
                or recovered["size_bytes"]
                != POSTPROCESS_RECOVERY_PREVIOUS_ORCHESTRATOR_SIZE
                or recovered["sha256"]
                != POSTPROCESS_RECOVERY_PREVIOUS_ORCHESTRATOR_SHA256
            ):
                raise ValueError("Q4 recovery v1 orchestrator transition drift")
            approved_changes.append(
                {
                    "artifact_role": target_role,
                    "path": str(target_path),
                    "original_size_bytes": original["size_bytes"],
                    "original_sha256": original["sha256"],
                    "recovery_size_bytes": recovered["size_bytes"],
                    "recovery_sha256": recovered["sha256"],
                }
            )
            continue
        if original != recovered:
            raise ValueError(f"Unexpected Q4 recovery v1 code change: {path}")
        actual = _manifest_row(recovered["artifact_role"], path)
        if actual != recovered:
            raise ValueError(f"Unapproved code drift after Q4 recovery v1: {path}")
    if ledger.get("approved_code_changes") != approved_changes:
        raise ValueError("Q4 recovery v1 approved change drift")
    if (
        ledger.get("current_code_manifest_path") != str(current_code_path.resolve())
        or ledger.get("current_code_manifest_sha256") != _sha256_file(current_code_path)
        or ledger.get("pre_recovery_output_manifest_path")
        != str(snapshot_path.resolve())
        or ledger.get("pre_recovery_output_manifest_sha256")
        != _sha256_file(snapshot_path)
    ):
        raise ValueError("Q4 recovery v1 manifest anchor drift")
    snapshot_rows = _read_hash_rows(snapshot_path)
    if len(snapshot_rows) != int(
        ledger.get("pre_recovery_output_file_count", -1)
    ) or _recovery_snapshot_fingerprint(snapshot_rows) != ledger.get(
        "pre_recovery_output_universe_sha256"
    ):
        raise ValueError("Q4 recovery v1 output snapshot drift")
    registry = _registry(root)
    if (
        not bool(registry.get("q4_recovery_applied"))
        or registry.get("q4_recovery_incident") != Q4_RECOVERY_INCIDENT
        or registry.get("q4_recovery_ledger_path") != str(ledger_path.resolve())
        or registry.get("q4_recovery_ledger_sha256") != _sha256_file(ledger_path)
        or registry.get("q4_recovery_current_code_manifest_sha256")
        != _sha256_file(current_code_path)
        or registry.get("q4_recovery_pre_output_manifest_sha256")
        != _sha256_file(snapshot_path)
    ):
        raise ValueError("Registered Q4 recovery v1 lineage drift")
    evidence = _recovery_immutable_evidence(root, registry)
    if evidence != ledger.get("immutable_evidence"):
        raise ValueError("Q4 recovery v1 immutable evidence drift")
    return ledger


def _postprocess_recovery_code_candidate(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_rows = _normalized_manifest_rows(
        _recovery_directory(root) / Q4_RECOVERY_CURRENT_CODE_HASHES
    )
    return _single_code_transition(
        previous_rows,
        target_role="code:scripts/rq3/news_first_vol_film_nolp_capacity_seed.py",
        target_path=Path(__file__).resolve(),
        previous_size_bytes=POSTPROCESS_RECOVERY_PREVIOUS_ORCHESTRATOR_SIZE,
        previous_sha256=POSTPROCESS_RECOVERY_PREVIOUS_ORCHESTRATOR_SHA256,
    )


def _pre_postprocess_recovery_output_rows(root: Path) -> list[dict[str, Any]]:
    recovery = _postprocess_recovery_directory(root).resolve()
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and recovery not in path.resolve().parents
        and not path.name.startswith(".postprocess_recovery_v2.")
    )
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _postprocess_immutable_evidence(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    _validate_completed_q4_evaluation(root, resolved, registry)
    base = _recovery_immutable_evidence(root, registry)
    prediction_directory = root / "analysis/predictions/q4_refit_final"
    prediction_paths = sorted(prediction_directory.glob("*.csv.gz"))
    prediction_manifests = sorted(prediction_directory.glob("*.manifest.json"))
    if len(prediction_paths) != 36 or len(prediction_manifests) != 36:
        raise ValueError("Postprocess recovery requires 36 Q4 prediction caches")
    q4_artifacts = (
        prediction_paths
        + prediction_manifests
        + [
            root / "analysis/film_nolp_capacity_q4_pair_metrics.csv.gz",
            root / "analysis/film_nolp_capacity_q4_scores.csv",
            root / "analysis/film_nolp_capacity_q4_persistence_contrasts.csv",
            root / "analysis/film_nolp_capacity_q4_pairwise_contrasts.csv",
            root / "analysis/film_nolp_capacity_q4_30m_secondary_scores.csv",
            root / "analysis/film_nolp_capacity_q4_30m_secondary_persistence.csv",
            root / "analysis/film_nolp_capacity_q4_30m_secondary_pairwise.csv",
            root / "analysis/film_nolp_capacity_q4_summary.json",
            root / "data_windows/q4/common_05m_q4.xlsx",
            root / "data_windows/q4/q4_window_manifest.json",
            root / "report/film_nolp_capacity_conclusion.md",
            root / "report/film_nolp_capacity_conclusion.html",
            root / "resource_usage.csv",
        ]
    )
    if any(not path.is_file() for path in q4_artifacts):
        missing = [str(path) for path in q4_artifacts if not path.is_file()]
        raise ValueError(f"Postprocess recovery immutable artifacts missing: {missing}")
    rows = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(q4_artifacts)
    ]
    return {
        **base,
        "q4_prediction_cache_count": len(prediction_paths),
        "q4_prediction_manifest_count": len(prediction_manifests),
        "q4_and_partial_artifact_count": len(rows),
        "q4_and_partial_artifact_universe_sha256": _payload_sha256(rows),
    }


def _validate_postprocess_recovery_bundle(
    root: Path, *, require_registered: bool
) -> dict[str, Any]:
    _reject_postprocess_recovery_temporary_directories(root)
    _validate_formal_q4_recovery_v1_anchor(root)
    directory = _postprocess_recovery_directory(root)
    ledger_path = directory / POSTPROCESS_RECOVERY_LEDGER
    current_code_path = directory / POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES
    snapshot_path = directory / POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES
    for path in (ledger_path, current_code_path, snapshot_path):
        if not path.is_file():
            raise ValueError(f"Partial postprocess recovery bundle: {path}")
    ledger = _require_mapping(_read_json(ledger_path), "postprocess recovery ledger")
    unsigned = {key: value for key, value in ledger.items() if key != "payload_sha256"}
    if (
        ledger.get("schema_version") != 2
        or ledger.get("incident") != POSTPROCESS_RECOVERY_INCIDENT
        or _payload_sha256(unsigned) != ledger.get("payload_sha256")
        or ledger.get("training_jobs_rewritten") is not False
        or ledger.get("q4_recomputed") is not False
        or ledger.get("pre_recovery_registry_sha256")
        != POSTPROCESS_RECOVERY_PRE_REGISTRY_SHA256
        or ledger.get("pre_recovery_experiment_status_sha256")
        != POSTPROCESS_RECOVERY_PRE_STATUS_SHA256
    ):
        raise ValueError("Postprocess recovery ledger contract drift")
    anchored_v1 = _require_mapping(
        ledger.get("anchored_q4_recovery_v1"), "anchored Q4 recovery v1"
    )
    for name, expected_sha256 in POSTPROCESS_RECOVERY_V1_ANCHORS.items():
        path = _recovery_directory(root) / name
        anchor = _require_mapping(anchored_v1.get(name), f"Q4 recovery v1 {name}")
        if (
            anchor.get("path") != str(path.resolve())
            or int(anchor.get("size_bytes", -1)) != path.stat().st_size
            or anchor.get("sha256") != expected_sha256
            or _sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"Postprocess recovery v1 anchor drift: {path}")
    current_rows, approved_change = _postprocess_recovery_code_candidate(root)
    if ledger.get("approved_code_changes") != [approved_change]:
        raise ValueError("Approved postprocess recovery code change drift")
    if ledger.get("current_code_manifest_path") != str(
        current_code_path.resolve()
    ) or ledger.get("current_code_manifest_sha256") != _sha256_file(current_code_path):
        raise ValueError("Postprocess current-code manifest anchor drift")
    _verify_hash_rows(current_code_path)
    if _normalized_manifest_rows(current_code_path) != current_rows:
        raise ValueError("Postprocess recovery current-code universe drift")
    if ledger.get("pre_recovery_output_manifest_path") != str(
        snapshot_path.resolve()
    ) or ledger.get("pre_recovery_output_manifest_sha256") != _sha256_file(
        snapshot_path
    ):
        raise ValueError("Postprocess pre-recovery snapshot anchor drift")
    snapshot_rows = _read_hash_rows(snapshot_path)
    if len(snapshot_rows) != int(
        ledger.get("pre_recovery_output_file_count", -1)
    ) or _recovery_snapshot_fingerprint(snapshot_rows) != ledger.get(
        "pre_recovery_output_universe_sha256"
    ):
        raise ValueError("Postprocess pre-recovery output snapshot drift")
    snapshot_relative = {str(row["relative_path"]) for row in snapshot_rows}
    if tuple(ledger.get("pre_recovery_absent_paths") or ()) != (
        POSTPROCESS_RECOVERY_ABSENT_PATHS
    ) or any(path in snapshot_relative for path in POSTPROCESS_RECOVERY_ABSENT_PATHS):
        raise ValueError("Postprocess pre-recovery absence contract drift")
    partial = _require_mapping(
        ledger.get("partial_report_artifacts"), "partial reports"
    )
    for relative, expected_sha256 in POSTPROCESS_RECOVERY_PARTIAL_FILE_SHA256.items():
        path = root / relative
        anchor = _require_mapping(partial.get(relative), f"partial artifact {relative}")
        if (
            not path.is_file()
            or anchor.get("path") != str(path.resolve())
            or int(anchor.get("size_bytes", -1)) != path.stat().st_size
            or anchor.get("sha256") != expected_sha256
            or _sha256_file(path) != expected_sha256
        ):
            raise ValueError(f"Postprocess partial artifact drift: {path}")
    registry = _registry(root)
    evidence = _postprocess_immutable_evidence(root, registry)
    if evidence != ledger.get("immutable_evidence"):
        raise ValueError("Postprocess recovery immutable evidence drift")
    if require_registered and (
        not bool(registry.get("postprocess_recovery_applied"))
        or registry.get("postprocess_recovery_incident")
        != POSTPROCESS_RECOVERY_INCIDENT
        or registry.get("postprocess_recovery_ledger_path")
        != str(ledger_path.resolve())
        or registry.get("postprocess_recovery_ledger_sha256")
        != _sha256_file(ledger_path)
        or registry.get("postprocess_recovery_current_code_manifest_sha256")
        != _sha256_file(current_code_path)
        or registry.get("postprocess_recovery_pre_output_manifest_sha256")
        != _sha256_file(snapshot_path)
    ):
        raise ValueError("Registered postprocess recovery lineage drift")
    return ledger


def _validate_known_postprocess_recovery_candidate(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _reject_postprocess_recovery_temporary_directories(root)
    if _postprocess_recovery_directory(root).exists():
        raise ValueError("Postprocess recovery bundle already exists")
    _validate_formal_q4_recovery_v1_anchor(root)
    if (
        _sha256_file(root / "registry/jobs.json")
        != POSTPROCESS_RECOVERY_PRE_REGISTRY_SHA256
        or _sha256_file(root / "registry/experiment_status.json")
        != POSTPROCESS_RECOVERY_PRE_STATUS_SHA256
    ):
        raise ValueError("Postprocess recovery registry/status anchor drift")
    registry = _registry(root)
    status = _require_mapping(
        _read_json(root / "registry/experiment_status.json"), "experiment status"
    )
    unsigned_status = {
        key: value for key, value in status.items() if key != "payload_sha256"
    }
    if (
        registry.get("experiment_kind") != EXPERIMENT_KIND
        or registry.get("status") != "q4_evaluated"
        or not bool(registry.get("selection_frozen"))
        or not bool(registry.get("refit_complete"))
        or not all(
            bool(registry.get(field))
            for field in (
                "q4_window_materialized",
                "q4_loader_created",
                "q4_predictions_generated",
                "q4_evaluated",
            )
        )
        or status.get("experiment_kind") != EXPERIMENT_KIND
        or status.get("status") != "q4_evaluated"
        or status.get("current_stage") != "q4"
        or _payload_sha256(unsigned_status) != status.get("payload_sha256")
    ):
        raise ValueError("Root is not the exact recoverable postprocess failure state")
    for relative, expected_sha256 in POSTPROCESS_RECOVERY_PARTIAL_FILE_SHA256.items():
        path = root / relative
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(f"Postprocess recovery partial-state drift: {path}")
    present = [
        relative
        for relative in POSTPROCESS_RECOVERY_ABSENT_PATHS
        if (root / relative).exists()
    ]
    if present:
        raise ValueError(f"Postprocess recovery expected absent artifacts: {present}")
    current_rows, approved_change = _postprocess_recovery_code_candidate(root)
    _postprocess_immutable_evidence(root, registry)
    return registry, current_rows, approved_change


def _register_postprocess_recovery(root: Path) -> Path:
    """Atomically register the telemetry-column repair after frozen Q4."""

    final_directory = _postprocess_recovery_directory(root)
    if final_directory.exists():
        ledger = _validate_postprocess_recovery_bundle(root, require_registered=False)
    else:
        registry, current_rows, approved_change = (
            _validate_known_postprocess_recovery_candidate(root)
        )
        snapshot_rows = _pre_postprocess_recovery_output_rows(root)
        temporary = root / "registry" / f".postprocess_recovery_v2.{os.getpid()}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        snapshot_path = _write_csv(
            temporary / POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES,
            snapshot_rows,
            tuple(snapshot_rows[0]),
        )
        current_code_path = _write_csv(
            temporary / POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES,
            current_rows,
            tuple(current_rows[0]),
        )
        final_snapshot = final_directory / POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES
        final_current_code = final_directory / POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES
        partial = {
            relative: {
                "path": str((root / relative).resolve()),
                "size_bytes": (root / relative).stat().st_size,
                "sha256": _sha256_file(root / relative),
            }
            for relative in POSTPROCESS_RECOVERY_PARTIAL_FILE_SHA256
        }
        ledger = {
            "schema_version": 2,
            "incident": POSTPROCESS_RECOVERY_INCIDENT,
            "reason": (
                "KeyError: utilization_gpu_percent; telemetry uses utilization_gpu_pct"
            ),
            "compatibility_boundary": "branch_local_unreleased_experiment_schema",
            "approved_code_changes": [approved_change],
            "anchored_q4_recovery_v1": {
                name: {
                    "path": str((_recovery_directory(root) / name).resolve()),
                    "size_bytes": (_recovery_directory(root) / name).stat().st_size,
                    "sha256": _sha256_file(_recovery_directory(root) / name),
                }
                for name in POSTPROCESS_RECOVERY_V1_ANCHORS
            },
            "pre_recovery_registry_sha256": _sha256_file(root / "registry/jobs.json"),
            "pre_recovery_experiment_status_sha256": _sha256_file(
                root / "registry/experiment_status.json"
            ),
            "pre_recovery_output_manifest_path": str(final_snapshot.resolve()),
            "pre_recovery_output_manifest_sha256": _sha256_file(snapshot_path),
            "pre_recovery_output_file_count": len(snapshot_rows),
            "pre_recovery_output_universe_sha256": _recovery_snapshot_fingerprint(
                snapshot_rows
            ),
            "current_code_manifest_path": str(final_current_code.resolve()),
            "current_code_manifest_sha256": _sha256_file(current_code_path),
            "partial_report_artifacts": partial,
            "pre_recovery_absent_paths": list(POSTPROCESS_RECOVERY_ABSENT_PATHS),
            "immutable_evidence": _postprocess_immutable_evidence(root, registry),
            "training_jobs_rewritten": False,
            "q4_recomputed": False,
            "created_at_utc": _utc_now(),
        }
        ledger["payload_sha256"] = _payload_sha256(ledger)
        _write_json(temporary / POSTPROCESS_RECOVERY_LEDGER, ledger)
        os.replace(temporary, final_directory)
        ledger = _validate_postprocess_recovery_bundle(root, require_registered=False)
    registry = _registry(root)
    if bool(registry.get("postprocess_recovery_applied")):
        _validate_postprocess_recovery_bundle(root, require_registered=True)
        status = _require_mapping(
            _read_json(root / "registry/experiment_status.json"), "experiment status"
        )
        if status.get("status") == "q4_evaluated":
            _write_experiment_status(
                root,
                "q4_evaluated_postprocess_recovered_locked",
                current_stage="postprocess_recovery",
            )
        return _postprocess_recovery_ledger_path(root)
    if _sha256_file(root / "registry/jobs.json") != ledger.get(
        "pre_recovery_registry_sha256"
    ) or _sha256_file(root / "registry/experiment_status.json") != ledger.get(
        "pre_recovery_experiment_status_sha256"
    ):
        raise ValueError("Postprocess recovery registration state changed")
    registry.update(
        postprocess_recovery_applied=True,
        postprocess_recovery_incident=POSTPROCESS_RECOVERY_INCIDENT,
        postprocess_recovery_ledger_path=str(
            _postprocess_recovery_ledger_path(root).resolve()
        ),
        postprocess_recovery_ledger_sha256=_sha256_file(
            _postprocess_recovery_ledger_path(root)
        ),
        postprocess_recovery_current_code_manifest_path=str(
            (final_directory / POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES).resolve()
        ),
        postprocess_recovery_current_code_manifest_sha256=_sha256_file(
            final_directory / POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES
        ),
        postprocess_recovery_pre_output_manifest_path=str(
            (final_directory / POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES).resolve()
        ),
        postprocess_recovery_pre_output_manifest_sha256=_sha256_file(
            final_directory / POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES
        ),
        postprocess_recovered_at_utc=_utc_now(),
    )
    # The recovery is lineage-only: do not regenerate task_registry.csv or any
    # training artifact while anchoring the additional registry fields.
    _write_json(root / "registry/jobs.json", registry)
    _write_experiment_status(
        root,
        "q4_evaluated_postprocess_recovered_locked",
        current_stage="postprocess_recovery",
    )
    _validate_postprocess_recovery_bundle(root, require_registered=True)
    return _postprocess_recovery_ledger_path(root)


def _validate_root(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in ("source_hashes.csv", "config_hashes.csv"):
        _verify_hash_rows(root / name)
    try:
        _verify_hash_rows(root / "code_hashes.csv")
    except ValueError:
        if _postprocess_recovery_directory(root).exists():
            _validate_postprocess_recovery_bundle(root, require_registered=True)
        else:
            _validate_recovery_bundle(root, require_registered=True)
    registry = _registry(root)
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    _validate_registry_job_lineage(root, registry)
    return _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )


def _validated_benchmark(formal_root: Path, config_path: str | Path) -> dict[str, Any]:
    path = _benchmark_result_path(formal_root)
    payload = _require_mapping(_read_json(path), "benchmark result")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload.get("status") != "passed"
        or int(payload.get("worker_count", -1)) != EXPECTED_STAGE_JOBS
        or _payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("config_sha256") != _sha256_file(_resolve_repo_path(config_path))
        or payload.get("code_snapshot_sha256") != _snapshot_sha(_code_paths())
        or int(payload.get("selected_slots_per_gpu", -1)) not in {12, 18}
    ):
        raise ValueError("Benchmark result/config/code contract drift")
    return payload


def prepare_experiment(
    config_path: str | Path, output_dir: str | Path, *, reuse: bool = False
) -> Path:
    root = Path(output_dir).resolve()
    if root.exists():
        if not reuse:
            raise FileExistsError(root)
        _validate_root(root)
        return root
    benchmark = _validated_benchmark(root, config_path)
    _prepare_root(
        config_path,
        root,
        slots_per_gpu=int(benchmark["selected_slots_per_gpu"]),
        benchmark=False,
    )
    registry = _registry(root)
    benchmark_path = _benchmark_result_path(root)
    registry.update(
        benchmark_result_path=str(benchmark_path.resolve()),
        benchmark_result_sha256=_sha256_file(benchmark_path),
    )
    source_rows: list[dict[str, Any]] = []
    with (root / "source_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        source_rows.extend(csv.DictReader(handle))
    source_rows.append(_manifest_row("concurrency_benchmark", benchmark_path))
    _write_hash_rows(root / "source_hashes.csv", source_rows)
    _write_registry(root, registry)
    _validate_root(root)
    return root


def _completed_valid(job: Mapping[str, Any], status: Mapping[str, Any]) -> bool:
    artifacts = list(status.get("artifacts") or [])
    return bool(
        status.get("status") == "completed"
        and status.get("config_sha256") == job["config_sha256"]
        and artifacts
        and all(
            Path(str(row["path"])).is_file()
            and Path(str(row["path"])).stat().st_size == int(row["size_bytes"])
            and _sha256_file(Path(str(row["path"]))) == row["sha256"]
            for row in artifacts
        )
    )


def _artifact_rows(job: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = run_dir / "checkpoints"
    metrics = run_dir / "metrics"
    if job["experiment_stage"] == REFIT_STAGE:
        required = {
            "generator_final": checkpoints / "generator.pt",
            "discriminator_final": checkpoints / "discriminator.pt",
            "training_metrics_csv": metrics / "training_metrics.csv",
            "training_metrics_json": metrics / "training_metrics.json",
            "resolved_training_config": metrics / "training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
        }
        expected_roles = REFIT_ARTIFACT_ROLES
    else:
        required = {
            "generator_initial_epoch0": checkpoints / "generator_initial_epoch0.pt",
            "discriminator_initial_epoch0": checkpoints
            / "discriminator_initial_epoch0.pt",
            "generator_best": checkpoints / "generator_best.pt",
            "discriminator_best": checkpoints / "discriminator_best.pt",
            "generator_best_learned": checkpoints / "generator_best_learned.pt",
            "discriminator_best_learned": checkpoints / "discriminator_best_learned.pt",
            "generator_final": checkpoints / "generator.pt",
            "discriminator_final": checkpoints / "discriminator.pt",
            "training_metrics_csv": metrics / "training_metrics.csv",
            "training_metrics_json": metrics / "training_metrics.json",
            "best_checkpoint": metrics / "best_checkpoint.json",
            "initial_checkpoint": metrics / "initial_checkpoint.json",
            "best_learned_checkpoint": metrics / "best_learned_checkpoint.json",
            "resolved_training_config": metrics / "training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
        }
        expected_roles = DEVELOPMENT_ARTIFACT_ROLES
    if tuple(required) != expected_roles:
        raise AssertionError("Artifact role ordering drift")
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training completed without artifacts: {missing}")
    return [_manifest_row(role, path) for role, path in required.items()]


def _execute_job(
    job: Mapping[str, Any], *, dry_run: bool
) -> tuple[Path, list[dict[str, Any]]]:
    from wgan_option.config import load_config
    from wgan_option.train_vol_xlsx import VolSurfaceXlsxTrainer

    config = load_config(str(job["training_config_path"]))
    trainer = VolSurfaceXlsxTrainer(
        config, config_path=str(job["training_config_path"])
    )
    result = trainer.dry_run() if dry_run else trainer.train()
    if result is None or trainer.model is None:
        raise RuntimeError("Trainer did not return an instantiated WGAN run")
    generator_parameters = sum(
        parameter.numel() for parameter in trainer.model.G.parameters()
    )
    critic_parameters = sum(
        parameter.numel() for parameter in trainer.model.D.parameters()
    )
    if generator_parameters != int(job["expected_generator_parameters"]):
        raise RuntimeError("Generator parameter contract drift")
    if critic_parameters != int(job["expected_critic_parameters"]):
        raise RuntimeError("Critic parameter contract drift")
    if generator_parameters + critic_parameters != int(job["expected_wgan_parameters"]):
        raise RuntimeError("WGAN parameter contract drift")
    if (
        trainer.model.G.generator_conditioning_fingerprint
        != job["generator_conditioning_fingerprint"]
    ):
        raise RuntimeError("Generator conditioning fingerprint drift")
    if (
        trainer.model.D.critic_conditioning_fingerprint
        != job["critic_conditioning_fingerprint"]
    ):
        raise RuntimeError("Critic conditioning fingerprint drift")
    if str(config.news_first_model_contract_sha256) != job["model_contract_sha256"]:
        raise RuntimeError("Model-contract SHA drift")
    run_dir = Path(result).resolve()
    return run_dir, [] if dry_run else _artifact_rows(job, run_dir)


def _find_job(root: Path, job_id: str) -> dict[str, Any]:
    matches = [job for job in _registry(root)["jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise KeyError(job_id)
    return dict(matches[0])


def run_worker(
    output_dir: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    _validate_root(root)
    job = _find_job(root, job_id)
    registry = _registry(root)
    if registry.get("q4_gate_open") or registry.get("q4_evaluated"):
        raise RuntimeError("Training is forbidden after Q4 opens")
    if job["experiment_stage"] == REFIT_STAGE and not registry.get("selection_frozen"):
        raise RuntimeError("Refit requires frozen selection")
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if _completed_valid(job, previous):
        if resume and not dry_run:
            return Path(previous["run_dir"])
        raise RuntimeError(f"Job already completed: {job_id}")
    if previous.get("status") == "running" and training._pid_is_live(
        previous.get("pid")
    ):
        raise RuntimeError(f"Duplicate live job: {job_id}")
    if previous.get("status") in {"running", "failed"} and not (resume or dry_run):
        raise RuntimeError(f"Interrupted job requires --resume: {job_id}")
    running = {
        **_initial_status(job),
        "status": "running",
        "attempt": int(previous.get("attempt", 0)) + 1,
        "dry_run": bool(dry_run),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": _utc_now(),
    }
    _write_json(status_path, running)
    try:
        run_dir, artifacts = _execute_job(job, dry_run=dry_run)
        completed = {
            **running,
            "status": "dry_run_passed" if dry_run else "completed",
            "run_dir": str(run_dir),
            "artifacts": artifacts,
            "completed_at_utc": _utc_now(),
        }
        _write_json(status_path, completed)
        _write_registry(root, _registry(root))
        return run_dir
    except BaseException as exc:
        _write_json(
            status_path,
            {
                **running,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at_utc": _utc_now(),
            },
        )
        raise


def _worker_command(
    root: Path, job: Mapping[str, Any], *, dry_run: bool, resume: bool
) -> list[str]:
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    command = [
        str(resolved["runtime"]["python_executable"]),
        str(REPO_ROOT / "scripts/rq3/main.py"),
        "train-news-first-vol-film-nolp-capacity-seed-sweep",
        "worker",
        "--config",
        str(resolved["source_config_path"]),
        "--output-dir",
        str(root),
        "--job-id",
        str(job["job_id"]),
    ]
    if dry_run:
        command.append("--worker-dry-run")
    if resume:
        command.append("--resume")
    return command


def _host_ram_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
    total = values.get("MemTotal", 0)
    return 1.0 - values.get("MemAvailable", 0) / total if total else 1.0


def _run_wave(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    wave: int,
    dry_run: bool,
    resume: bool,
) -> float:
    if not jobs:
        return _host_ram_fraction()
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    runtime = resolved["runtime"]
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime.get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 2)),
        wave=wave,
    )
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    peak_host = _host_ram_fraction()
    try:
        monitor.start()
        for job in jobs:
            previous = _read_json(_job_status_path(root, str(job["job_id"])))
            log = (
                root
                / "logs"
                / f"{job['job_id']}.attempt_{int(previous.get('attempt', 0)) + 1:02d}.log"
            )
            handle = log.open("a", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["NEWS_FIRST_JOB_LOG_PATH"] = str(log)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = threads
            processes.append(
                subprocess.Popen(
                    _worker_command(root, job, dry_run=dry_run, resume=resume),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        pending = set(range(len(processes)))
        while pending:
            peak_host = max(peak_host, _host_ram_fraction())
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    training._terminate_processes(processes)
                    raise RuntimeError(
                        f"Wave {wave} job {jobs[index]['job_id']} exited {code}"
                    )
            if pending:
                time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        training._terminate_processes(processes)
        raise
    finally:
        monitor.stop()
        for handle in handles:
            handle.close()
    return peak_host


def _stage_jobs(root: Path, stage: str) -> list[dict[str, Any]]:
    jobs = [
        dict(job) for job in _registry(root)["jobs"] if job["experiment_stage"] == stage
    ]
    if len(jobs) != EXPECTED_STAGE_JOBS:
        raise ValueError(f"Expected {EXPECTED_STAGE_JOBS} {stage} jobs")
    _assert_gpu_balance(jobs)
    return jobs


def _launch_stage(root: Path, stage: str, *, dry_run: bool, resume: bool) -> float:
    _validate_root(root)
    jobs = _stage_jobs(root, stage)
    peak = 0.0
    for wave in sorted({int(job["wave"]) for job in jobs}):
        selected = []
        for job in jobs:
            if int(job["wave"]) != wave:
                continue
            status = _read_json(_job_status_path(root, str(job["job_id"])))
            if not dry_run and _completed_valid(job, status):
                if resume:
                    continue
                raise RuntimeError(f"Completed job requires --resume: {job['job_id']}")
            if dry_run and status.get("status") == "dry_run_passed" and resume:
                continue
            selected.append(job)
        peak = max(
            peak, _run_wave(root, selected, wave=wave, dry_run=dry_run, resume=resume)
        )
    failures = []
    for job in jobs:
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_valid(job, status)
        )
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status')}")
    if failures:
        raise RuntimeError(f"Incomplete stage: {failures}")
    _write_registry(root, _registry(root))
    return peak


def _resource_peaks(root: Path) -> dict[int, float]:
    peaks = {0: 0.0, 1: 0.0}
    samples = {0: 0, 1: 0}
    with (root / "resource_usage.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("sample_status") != "ok":
                continue
            gpu = int(row["gpu_index"])
            if gpu in peaks:
                samples[gpu] += 1
                peaks[gpu] = max(peaks[gpu], float(row["memory_used_mib"]))
    if any(samples[gpu] == 0 for gpu in peaks):
        raise RuntimeError(f"Missing GPU telemetry: {samples}")
    return peaks


def _checkpoint_tensors(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in (
            "state_dict",
            "model_state_dict",
            "generator_state_dict",
            "discriminator_state_dict",
        ):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and nested:
                return {
                    str(name): value
                    for name, value in nested.items()
                    if torch.is_tensor(value)
                }
        tensors = {
            str(name): value
            for name, value in payload.items()
            if torch.is_tensor(value)
        }
        if tensors:
            return tensors
    raise ValueError(f"No tensor state dict in {path}")


def _assert_benchmark_updates(root: Path) -> None:
    import torch

    for job in _stage_jobs(root, BENCHMARK_STAGE):
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        by_role = {
            row["artifact_role"]: Path(row["path"]) for row in status["artifacts"]
        }
        for initial_role, final_role in (
            ("generator_initial_epoch0", "generator_final"),
            ("discriminator_initial_epoch0", "discriminator_final"),
        ):
            initial = _checkpoint_tensors(by_role[initial_role])
            final = _checkpoint_tensors(by_role[final_role])
            if set(initial) != set(final):
                raise ValueError(f"Benchmark state keys drift: {job['job_id']}")
            changed = any(not torch.equal(initial[key], final[key]) for key in initial)
            if not changed:
                raise ValueError(
                    f"Benchmark parameters did not update: {job['job_id']}/{final_role}"
                )


def run_benchmark(
    config_path: str | Path, output_dir: str | Path, *, resume: bool = False
) -> Path:
    formal = Path(output_dir).resolve()
    if formal.exists():
        raise FileExistsError("Benchmark must precede formal root")
    root = _benchmark_root(formal)
    if root.exists():
        if not resume:
            raise FileExistsError(f"Benchmark root exists; use --resume: {root}")
        _validate_root(root)
    else:
        _prepare_root(config_path, root, slots_per_gpu=18, benchmark=True)
    _write_experiment_status(root, "benchmark_running", current_stage=BENCHMARK_STAGE)
    try:
        peak_host = _launch_stage(root, BENCHMARK_STAGE, dry_run=False, resume=resume)
        _assert_benchmark_updates(root)
        resolved = _validate_root(root)
        peaks = _resource_peaks(root)
        memory_limit = (
            float(resolved["runtime"]["preflight_max_peak_gpu_memory_gib"]) * 1024.0
        )
        ram_limit = float(resolved["runtime"]["preflight_max_host_ram_fraction"])
        selected = (
            18
            if all(value < memory_limit for value in peaks.values())
            and peak_host < ram_limit
            else 12
        )
        result = {
            "schema_version": 1,
            "experiment_kind": EXPERIMENT_KIND,
            "status": "passed",
            "worker_count": EXPECTED_STAGE_JOBS,
            "workers_per_gpu": 18,
            "num_epochs": 1,
            "peak_gpu_memory_mib": {str(key): value for key, value in peaks.items()},
            "peak_host_ram_fraction": peak_host,
            "gpu_memory_limit_mib_exclusive": memory_limit,
            "host_ram_limit_fraction_exclusive": ram_limit,
            "selected_slots_per_gpu": selected,
            "config_sha256": _sha256_file(_resolve_repo_path(config_path)),
            "code_snapshot_sha256": _snapshot_sha(_code_paths()),
            "completed_at_utc": _utc_now(),
        }
        result["payload_sha256"] = _payload_sha256(result)
        _write_json(root / "benchmark_result.json", result)
        _write_experiment_status(
            root, "benchmark_completed", selected_slots_per_gpu=selected
        )
    except BaseException as exc:
        _write_experiment_status(root, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return root


def launch_development(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    root = prepare_experiment(
        config_path, output_dir, reuse=bool(resume or Path(output_dir).exists())
    )
    if _registry(root).get("selection_frozen") and not dry_run:
        raise RuntimeError("Development is frozen")
    status = "development_dry_running" if dry_run else "development_running"
    _write_experiment_status(root, status, current_stage=DEVELOPMENT_STAGE)
    try:
        _launch_stage(root, DEVELOPMENT_STAGE, dry_run=dry_run, resume=resume)
        _write_experiment_status(
            root,
            "development_dry_run_passed" if dry_run else "development_complete",
            current_stage=DEVELOPMENT_STAGE,
        )
    except BaseException as exc:
        _write_experiment_status(
            root,
            "failed",
            current_stage=DEVELOPMENT_STAGE,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return root


def _artifact_path(status: Mapping[str, Any], role: str) -> Path:
    matches = [
        Path(row["path"])
        for row in status.get("artifacts", [])
        if row.get("artifact_role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {role} artifact")
    return matches[0]


def _validate_stage(root: Path, stage: str) -> None:
    failures = []
    for job in _stage_jobs(root, stage):
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        expected_roles = (
            DEVELOPMENT_ARTIFACT_ROLES
            if stage == DEVELOPMENT_STAGE
            else REFIT_ARTIFACT_ROLES
        )
        observed_roles = tuple(
            row.get("artifact_role") for row in status.get("artifacts", [])
        )
        if not _completed_valid(job, status) or observed_roles != expected_roles:
            failures.append(f"{job['job_id']}={status.get('status')}")
    if failures:
        raise RuntimeError(f"Incomplete {stage}: {failures}")


def _validate_recipe(path: Path) -> dict[str, Any]:
    recipe = _require_mapping(_read_json(path), "refit recipe")
    factorial._validate_refit_recipe(recipe)
    return recipe


def _recipe_paths(root: Path) -> dict[str, Path]:
    directory = root / "analysis" / "refit_recipes"
    paths = {path.stem: path.resolve() for path in directory.glob("*.json")}
    expected = {job["job_id"] for job in _stage_jobs(root, DEVELOPMENT_STAGE)}
    if set(paths) != expected:
        raise ValueError("Refit recipes do not cover all 36 development jobs")
    for path in paths.values():
        _validate_recipe(path)
    return paths


def _validate_frozen_selection(root: Path) -> Path:
    registry = _registry(root)
    selection = Path(str(registry.get("selection_path", "")))
    manifest = Path(str(registry.get("refit_recipe_manifest_path", "")))
    if (
        not selection.is_file()
        or _sha256_file(selection) != registry.get("selection_sha256")
        or not manifest.is_file()
        or _sha256_file(manifest) != registry.get("refit_recipe_manifest_sha256")
    ):
        raise ValueError("Frozen selection/recipe manifest drift")
    recipes = _recipe_paths(root)
    payload = _require_mapping(_read_json(manifest), "recipe manifest")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if (
        _payload_sha256(unsigned) != payload.get("manifest_sha256")
        or int(payload.get("recipe_count", -1)) != EXPECTED_STAGE_JOBS
        or payload.get("selection_sha256") != _sha256_file(selection)
        or len(payload.get("recipes", [])) != EXPECTED_STAGE_JOBS
    ):
        raise ValueError("Recipe manifest contract drift")
    indexed = {row["development_job_id"]: row for row in payload["recipes"]}
    if set(indexed) != set(recipes):
        raise ValueError("Recipe manifest job universe drift")
    for job_id, path in recipes.items():
        if indexed[job_id]["recipe_sha256"] != _sha256_file(path):
            raise ValueError(f"Recipe SHA drift: {job_id}")
    return selection


def freeze_selection(output_dir: str | Path) -> Path:
    from scripts.rq3.news_first_vol_film_nolp_capacity_seed_analysis import (
        freeze_refit_recipes,
        run_q3_analysis,
    )

    root = Path(output_dir).resolve()
    resolved = _validate_root(root)
    registry = _registry(root)
    if registry.get("selection_frozen"):
        _validate_frozen_selection(root)
        return root
    _validate_stage(root, DEVELOPMENT_STAGE)
    selection_path = root / "analysis" / "film_nolp_capacity_q3_selection.json"
    recipe_manifest = root / "analysis" / "refit_recipe_manifest.json"
    if not selection_path.is_file():
        run_q3_analysis(root)
    if not recipe_manifest.is_file():
        freeze_refit_recipes(root, selection_path)
    recipes = _recipe_paths(root)
    manifest_payload = _require_mapping(_read_json(recipe_manifest), "recipe manifest")
    if int(manifest_payload.get("recipe_count", -1)) != EXPECTED_STAGE_JOBS:
        raise ValueError("Recipe manifest count drift")
    slots = int(registry["slots_per_gpu"])
    refit_jobs, config_rows = _build_jobs(
        resolved,
        root,
        stage=REFIT_STAGE,
        slots_per_gpu=slots,
        recipes=recipes,
    )
    existing_ids = {job["job_id"] for job in registry["jobs"]}
    if existing_ids & {job["job_id"] for job in refit_jobs}:
        raise ValueError("Refit job ID collides with development")
    for job in refit_jobs:
        _write_json(_job_status_path(root, str(job["job_id"])), _initial_status(job))
    with (root / "config_hashes.csv").open("r", encoding="utf-8", newline="") as handle:
        config_manifest = list(csv.DictReader(handle))
    config_manifest.extend(config_rows)
    _write_hash_rows(root / "config_hashes.csv", config_manifest)
    registry.update(
        status="selection_frozen_refit_prepared",
        selection_frozen=True,
        selection_path=str(selection_path.resolve()),
        selection_sha256=_sha256_file(selection_path),
        refit_recipe_manifest_path=str(recipe_manifest.resolve()),
        refit_recipe_manifest_sha256=_sha256_file(recipe_manifest),
        refit_job_count=EXPECTED_STAGE_JOBS,
        jobs=[*registry["jobs"], *refit_jobs],
        selection_frozen_at_utc=_utc_now(),
    )
    _write_registry(root, registry)
    _write_experiment_status(
        root, "selection_frozen_refit_prepared", current_stage="selection"
    )
    _validate_root(root)
    _validate_frozen_selection(root)
    return root


def _freeze_q4_allowlist(root: Path) -> Path:
    registry = _registry(root)
    _validate_stage(root, REFIT_STAGE)
    rows: list[dict[str, Any]] = []
    for job in _stage_jobs(root, REFIT_STAGE):
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        for role in ("generator_final", "discriminator_final"):
            checkpoint = _artifact_path(status, role)
            rows.append(
                {
                    "job_id": job["job_id"],
                    "parent_development_job_id": job["parent_development_job_id"],
                    "capacity_profile": job["capacity_profile"],
                    "seed": job["seed"],
                    "tolerance_minutes": job["tolerance_minutes"],
                    "generator_conditioning_mode": GENERATOR_MODE,
                    "critic_conditioning_mode": CRITIC_MODE,
                    "checkpoint_role": role,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": _sha256_file(checkpoint),
                    "q4_allowed": True,
                }
            )
    if len(rows) != 72 or len({row["checkpoint_path"] for row in rows}) != 72:
        raise ValueError("Q4 allowlist requires 72 unique checkpoints")
    path = _write_csv(root / "q4_checkpoint_allowlist.csv", rows, tuple(rows[0]))
    registry.update(
        status="refit_complete_q4_locked",
        refit_complete=True,
        q4_allowlist_path=str(path.resolve()),
        q4_allowlist_sha256=_sha256_file(path),
        q4_gate_open=False,
        q4_window_materialized=False,
        q4_loader_created=False,
        q4_predictions_generated=False,
        q4_evaluated=False,
    )
    _write_registry(root, registry)
    _write_experiment_status(
        root, "refit_complete_q4_locked", current_stage="q4_locked"
    )
    return path


def _validate_allowlist(root: Path) -> list[dict[str, str]]:
    registry = _registry(root)
    path = Path(str(registry.get("q4_allowlist_path", "")))
    if not path.is_file() or _sha256_file(path) != registry.get("q4_allowlist_sha256"):
        raise ValueError("Q4 allowlist drift")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 72:
        raise ValueError("Q4 allowlist row-count drift")
    for row in rows:
        checkpoint = Path(row["checkpoint_path"])
        if (
            not checkpoint.is_file()
            or _sha256_file(checkpoint) != row["checkpoint_sha256"]
            or row["q4_allowed"].lower() != "true"
        ):
            raise ValueError(f"Q4 checkpoint drift: {checkpoint}")
    return rows


def launch_refit(
    config_path: str | Path, output_dir: str | Path, *, resume: bool = False
) -> Path:
    root = Path(output_dir).resolve()
    _validate_root(root)
    registry = _registry(root)
    if not registry.get("selection_frozen"):
        raise RuntimeError("Refit requires freeze-selection")
    _validate_frozen_selection(root)
    if registry.get("q4_gate_open") or registry.get("q4_evaluated"):
        raise RuntimeError("Refit forbidden after Q4 access")
    if registry.get("refit_complete"):
        if resume:
            _validate_allowlist(root)
            return root
        raise RuntimeError("Refit already complete")
    _write_experiment_status(root, "refit_running", current_stage=REFIT_STAGE)
    try:
        _launch_stage(root, REFIT_STAGE, dry_run=False, resume=resume)
        _freeze_q4_allowlist(root)
    except BaseException as exc:
        _write_experiment_status(
            root,
            "failed",
            current_stage=REFIT_STAGE,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    return root


def _validate_existing_q4_window(
    resolved: Mapping[str, Any], root: Path
) -> tuple[Path, Path]:
    workbook = root / "data_windows" / "q4" / "common_05m_q4.xlsx"
    manifest_path = root / "data_windows" / "q4" / "q4_window_manifest.json"
    if not workbook.is_file() or not manifest_path.is_file():
        raise ValueError("Partial Q4 window materialization")
    manifest = _require_mapping(_read_json(manifest_path), "Q4 window manifest")
    unsigned = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    source = factorial._dataset_path(resolved, 5)
    if (
        _payload_sha256(unsigned) != manifest.get("payload_sha256")
        or manifest.get("experiment_kind") != EXPERIMENT_KIND
        or manifest.get("explicit_action") != "evaluate-q4"
        or int(manifest.get("tolerance_minutes", -1)) != 5
        or manifest.get("supported_counts") != EXPECTED_Q4_COMMON_COUNTS
        or manifest.get("workbook_path") != str(workbook.resolve())
        or manifest.get("workbook_sha256") != _sha256_file(workbook)
        or manifest.get("source_workbook_path") != str(source.resolve())
        or manifest.get("source_workbook_sha256") != _sha256_file(source)
        or int(manifest.get("broad_30m_q4_rows_read", -1)) != 0
        or not bool(manifest.get("q4_window_materialized"))
    ):
        raise ValueError("Existing Q4 window lineage drift")
    supported = factorial._supported_frame(resolved, workbook, 5)
    observed = factorial._counts(supported)
    if observed != EXPECTED_Q4_COMMON_COUNTS:
        raise ValueError(
            f"Frozen Q4 common counts drifted: {observed} != "
            f"{EXPECTED_Q4_COMMON_COUNTS}"
        )
    return workbook, manifest_path


def _materialize_q4_common_window(
    resolved: Mapping[str, Any], root: Path, *, resume: bool
) -> tuple[Path, Path]:
    """Materialize only the frozen common-5m Q4 panel for this experiment kind."""

    import pandas as pd

    registry = _registry(root)
    if registry.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Q4 materialization experiment kind drift")
    if not bool(registry.get("q4_gate_open")):
        raise RuntimeError("Q4 materialization requires the explicit open gate")
    if not bool(registry.get("selection_frozen")) or not bool(
        registry.get("refit_complete")
    ):
        raise RuntimeError("Q4 materialization requires frozen selection/refits")
    # These checks deliberately precede the first Q4 source read.
    _validate_frozen_selection(root)
    _validate_allowlist(root)
    directory = root / "data_windows" / "q4"
    if directory.exists():
        if not resume:
            raise RuntimeError("Q4 window already exists; use --resume to validate")
        return _validate_existing_q4_window(resolved, root)
    source = factorial._dataset_path(resolved, 5)
    sheet = str(resolved["datasets"]["sheet_name"])
    frame = pd.read_excel(source, sheet_name=sheet)
    timestamp = pd.to_datetime(frame["effective_origin_utc"], errors="coerce", utc=True)
    if timestamp.isna().any():
        raise ValueError("Invalid Q4 timestamps in common 5m workbook")
    start = pd.Timestamp(resolved["split"]["q4_start_utc"])
    end = pd.Timestamp(resolved["split"]["q4_end_utc"])
    selected = frame.loc[(timestamp >= start) & (timestamp < end)].copy()
    if selected.empty:
        raise ValueError("Q4 common window is empty")
    directory.mkdir(parents=True, exist_ok=False)
    workbook = directory / "common_05m_q4.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        selected.to_excel(writer, sheet_name=sheet, index=False)
    supported = factorial._supported_frame(resolved, workbook, 5)
    observed = factorial._counts(supported)
    if observed != EXPECTED_Q4_COMMON_COUNTS:
        raise ValueError(
            f"Frozen Q4 common counts drifted: {observed} != "
            f"{EXPECTED_Q4_COMMON_COUNTS}"
        )
    manifest = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "explicit_action": "evaluate-q4",
        "source_sheet_scanned_for_window_materialization": True,
        "source_workbook_path": str(source.resolve()),
        "source_workbook_sha256": _sha256_file(source),
        "start_utc_inclusive": str(resolved["split"]["q4_start_utc"]),
        "end_utc_exclusive": str(resolved["split"]["q4_end_utc"]),
        "tolerance_minutes": 5,
        "supported_counts": observed,
        "workbook_path": str(workbook.resolve()),
        "workbook_sha256": _sha256_file(workbook),
        "q4_window_materialized": True,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "broad_30m_q4_rows_read": 0,
        "materialized_at_utc": _utc_now(),
    }
    manifest["payload_sha256"] = _payload_sha256(manifest)
    manifest_path = _write_json(directory / "q4_window_manifest.json", manifest)
    return workbook, manifest_path


def evaluate_q4(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_nolp_capacity_seed_analysis import (
        run_q4_analysis,
    )

    root = Path(output_dir).resolve()
    try:
        resolved = _validate_root(root)
    except ValueError:
        if not resume:
            raise
        _register_q4_recovery(root)
        resolved = _validate_root(root)
    registry = _registry(root)
    if not registry.get("refit_complete"):
        raise RuntimeError("Q4 requires all 36 refits")
    _validate_allowlist(root)
    if registry.get("q4_evaluated"):
        if resume:
            _validate_completed_q4_evaluation(root, resolved, registry)
            return root
        raise RuntimeError("Q4 already evaluated")
    if registry.get("q4_gate_open") and not resume:
        raise RuntimeError("Q4 gate is already open; use --resume")
    registry.update(q4_gate_open=True, q4_gate_opened_at_utc=_utc_now())
    _write_registry(root, registry)
    try:
        workbook, window_manifest = _materialize_q4_common_window(
            resolved, root, resume=resume
        )
        registry = _registry(root)
        registry.update(
            q4_window_materialized=True,
            q4_window_path=str(workbook.resolve()),
            q4_window_sha256=_sha256_file(workbook),
            q4_window_manifest_path=str(window_manifest.resolve()),
            q4_window_manifest_sha256=_sha256_file(window_manifest),
            q4_loader_created=False,
        )
        _write_registry(root, registry)
        summary = Path(run_q4_analysis(root))
        if not summary.is_file():
            raise FileNotFoundError(summary)
        window_payload = _require_mapping(
            _read_json(window_manifest), "Q4 window manifest"
        )
        window_payload.update(
            q4_loader_created=True,
            q4_predictions_generated=True,
            q4_evaluated=True,
        )
        window_payload.pop("payload_sha256", None)
        window_payload["payload_sha256"] = _payload_sha256(window_payload)
        _write_json(window_manifest, window_payload)
        registry = _registry(root)
        registry.update(
            status="q4_evaluated",
            q4_loader_created=True,
            q4_predictions_generated=True,
            q4_evaluated=True,
            q4_summary_path=str(summary.resolve()),
            q4_summary_sha256=_sha256_file(summary),
            q4_window_manifest_sha256=_sha256_file(window_manifest),
            q4_evaluated_at_utc=_utc_now(),
        )
        _write_registry(root, registry)
        _write_experiment_status(root, "q4_evaluated", current_stage="q4")
    except BaseException as exc:
        _write_experiment_status(
            root, "failed", current_stage="q4", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    return root


def _resource_summary(root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    usage = root / "resource_usage.csv"
    if not usage.is_file():
        raise FileNotFoundError(usage)
    with usage.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        required = {"sample_status", "gpu_index", "memory_used_mib"}
        if not required.issubset(fieldnames):
            raise ValueError("Resource telemetry required columns are missing")
        utilization_columns = tuple(
            name
            for name in ("utilization_gpu_pct", "utilization_gpu_percent")
            if name in fieldnames
        )
        if len(utilization_columns) != 1:
            raise ValueError(
                "Resource telemetry must contain exactly one GPU utilization column"
            )
        utilization_column = utilization_columns[0]
        samples = [row for row in reader if row.get("sample_status") == "ok"]
    if not samples:
        raise ValueError("Resource telemetry contains no successful samples")
    observed_gpus = {int(row["gpu_index"]) for row in samples}
    if observed_gpus != {0, 1}:
        raise ValueError(f"Resource telemetry GPU universe drift: {observed_gpus}")
    for gpu in (0, 1):
        selected = [row for row in samples if int(row["gpu_index"]) == gpu]
        rows.append(
            {
                "gpu_id": gpu,
                "sample_count": len(selected),
                "peak_memory_mib": max(
                    float(row["memory_used_mib"]) for row in selected
                ),
                "mean_utilization_percent": (
                    sum(float(row[utilization_column]) for row in selected)
                    / len(selected)
                ),
            }
        )
    return _write_csv(root / "resource_summary.csv", rows, tuple(rows[0]))


def _all_local_files(root: Path) -> list[Path]:
    excluded = {
        (root / "output_hashes.csv").resolve(),
        (root / "registry/jobs.json").resolve(),
    }
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    )


def _validate_prediction_cache_set(
    root: Path,
    *,
    stage_name: str,
    jobs: Sequence[Mapping[str, Any]],
    checkpoint_role: str,
    mc_samples: int,
    row_count: int,
) -> None:
    directory = root / "analysis" / "predictions" / stage_name
    expected = {str(job["job_id"]): dict(job) for job in jobs}
    csv_paths = {
        path.name.removesuffix(".csv.gz"): path for path in directory.glob("*.csv.gz")
    }
    manifest_paths = {
        path.name.removesuffix(".manifest.json"): path
        for path in directory.glob("*.manifest.json")
    }
    if set(csv_paths) != set(expected) or set(manifest_paths) != set(expected):
        raise ValueError(f"Incomplete {stage_name} prediction cache")
    panel_hashes: set[str] = set()
    for job_id, job in expected.items():
        prediction = csv_paths[job_id]
        manifest_path = manifest_paths[job_id]
        manifest = _require_mapping(_read_json(manifest_path), "prediction manifest")
        unsigned = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        status = _require_mapping(
            _read_json(_job_status_path(root, job_id)), "job status"
        )
        checkpoint = [
            row
            for row in status.get("artifacts", [])
            if row.get("artifact_role") == checkpoint_role
        ]
        if (
            len(checkpoint) != 1
            or _payload_sha256(unsigned) != manifest.get("manifest_sha256")
            or manifest.get("job_id") != job_id
            or manifest.get("job_spec_sha256") != job["job_spec_sha256"]
            or manifest.get("checkpoint_sha256") != checkpoint[0]["sha256"]
            or int(manifest.get("mc_samples", -1)) != mc_samples
            or int(manifest.get("row_count", -1)) != row_count
            or Path(str(manifest.get("prediction_path", ""))).resolve()
            != prediction.resolve()
            or manifest.get("prediction_sha256") != _sha256_file(prediction)
        ):
            raise ValueError(f"Prediction cache lineage drift: {job_id}")
        panel_hashes.add(str(manifest.get("panel_universe_sha256")))
    if len(panel_hashes) != 1:
        raise ValueError(f"{stage_name} panel-universe drift")


def _validate_q4_summary(root: Path) -> None:
    summary_path = root / "analysis" / "film_nolp_capacity_q4_summary.json"
    summary = _require_mapping(_read_json(summary_path), "Q4 summary")
    unsigned = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if (
        summary.get("experiment_kind") != EXPERIMENT_KIND
        or _payload_sha256(unsigned) != summary.get("summary_sha256")
        or not all(
            bool(summary.get(field))
            for field in (
                "q4_loader_created",
                "q4_predictions_generated",
                "q4_evaluated",
            )
        )
        or bool(summary.get("q4_used_for_selection"))
    ):
        raise ValueError("Q4 summary contract drift")
    artifacts = _require_mapping(summary.get("artifact_sha256"), "Q4 artifacts")
    for name, sha256 in artifacts.items():
        path = root / "analysis" / str(name)
        if not path.is_file() or _sha256_file(path) != sha256:
            raise ValueError(f"Q4 summary artifact drift: {path}")
    historical = _require_mapping(
        summary.get("historical_q4_exposure"), "historical Q4 exposure"
    )
    historical_path = Path(str(historical.get("historical_prediction_path", "")))
    expected_historical_path = (
        REPO_ROOT
        / "outputs/experiments/rq3_news_first_vol_training_q097_103_ttm07_38_v1"
        / "analysis/predictions/regression_05m_core.csv.gz"
    ).resolve()
    if (
        historical_path.resolve() != expected_historical_path
        or not historical_path.is_file()
        or _sha256_file(historical_path)
        != historical.get("historical_prediction_sha256")
        or int(historical.get("current_pair_count", -1)) != 143
        or int(historical.get("overlapping_pair_count", -1)) != 143
        or historical.get("pair_overlap_label") != "143/143"
        or historical.get("historically_exposed") is not True
        or historical.get("interpretation")
        != "retrospective_frozen_exploratory_not_confirmatory"
    ):
        raise ValueError("Historical Q4 exposure evidence/path/hash drift")


def _validate_completed_q4_evaluation(
    root: Path, resolved: Mapping[str, Any], registry: Mapping[str, Any]
) -> None:
    if not all(
        bool(registry.get(field))
        for field in (
            "q4_window_materialized",
            "q4_loader_created",
            "q4_predictions_generated",
            "q4_evaluated",
        )
    ):
        raise ValueError("Completed Q4 registry flags are incomplete")
    _, window_manifest = _validate_existing_q4_window(resolved, root)
    if registry.get("q4_window_manifest_path") != str(
        window_manifest.resolve()
    ) or registry.get("q4_window_manifest_sha256") != _sha256_file(window_manifest):
        raise ValueError("Q4 registry window-manifest anchor drift")
    window_payload = _require_mapping(_read_json(window_manifest), "Q4 window manifest")
    if not all(
        bool(window_payload.get(field))
        for field in (
            "q4_loader_created",
            "q4_predictions_generated",
            "q4_evaluated",
        )
    ):
        raise ValueError("Q4 window manifest is not terminal")
    summary_path = root / "analysis" / "film_nolp_capacity_q4_summary.json"
    if registry.get("q4_summary_path") != str(summary_path.resolve()) or registry.get(
        "q4_summary_sha256"
    ) != _sha256_file(summary_path):
        raise ValueError("Q4 registry summary anchor drift")
    _validate_q4_summary(root)
    refit = [
        job
        for job in registry.get("jobs", [])
        if job.get("experiment_stage") == REFIT_STAGE
    ]
    if len(refit) != EXPECTED_STAGE_JOBS:
        raise ValueError("Completed Q4 requires all 36 refit jobs")
    _validate_prediction_cache_set(
        root,
        stage_name="q4_refit_final",
        jobs=refit,
        checkpoint_role="generator_final",
        mc_samples=64,
        row_count=167,
    )


def _validate_final_registry_snapshot(root: Path) -> None:
    path = root / "registry" / "final_registry_snapshot.json"
    payload = _require_mapping(_read_json(path), "final registry snapshot")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    snapshot_registry = _require_mapping(payload.get("registry"), "snapshot registry")
    snapshot_status = _require_mapping(
        payload.get("experiment_status"), "snapshot experiment status"
    )
    live_registry = _registry(root)
    terminal_anchor_fields = {
        "terminal_output_manifest_path",
        "terminal_output_manifest_sha256",
        "terminal_output_artifact_count",
    }
    normalized_live_registry = {
        key: value
        for key, value in live_registry.items()
        if key not in terminal_anchor_fields
    }
    live_status = _require_mapping(
        _read_json(root / "registry" / "experiment_status.json"),
        "terminal experiment status",
    )
    unsigned_status = {
        key: value for key, value in live_status.items() if key != "payload_sha256"
    }
    if (
        payload.get("experiment_kind") != EXPERIMENT_KIND
        or _payload_sha256(unsigned) != payload.get("payload_sha256")
        or snapshot_registry.get("experiment_kind") != EXPERIMENT_KIND
        or snapshot_registry.get("status") != "completed"
        or not bool(snapshot_registry.get("q4_evaluated"))
        or normalized_live_registry != snapshot_registry
        or snapshot_status.get("experiment_kind") != EXPERIMENT_KIND
        or snapshot_status.get("status") != "completed"
        or live_status != snapshot_status
        or _payload_sha256(unsigned_status) != live_status.get("payload_sha256")
    ):
        raise ValueError("Final registry snapshot contract drift")


def _validate_terminal_manifest_registry_anchor(
    root: Path, registry: Mapping[str, Any], manifest: Path, row_count: int
) -> None:
    if (
        registry.get("status") != "completed"
        or registry.get("terminal_output_manifest_path") != str(manifest.resolve())
        or registry.get("terminal_output_manifest_sha256") != _sha256_file(manifest)
        or int(registry.get("terminal_output_artifact_count", -1)) != row_count
    ):
        raise ValueError("Terminal output manifest registry anchor drift")


def _repair_unanchored_terminal_manifest(root: Path) -> None:
    """Repair only the atomic gap after output_hashes but before its registry anchor."""

    manifest = root / "output_hashes.csv"
    registry = _registry(root)
    anchor_fields = (
        "terminal_output_manifest_path",
        "terminal_output_manifest_sha256",
        "terminal_output_artifact_count",
    )
    present = tuple(field in registry for field in anchor_fields)
    if any(present):
        raise ValueError("Terminal manifest repair requires all anchor fields absent")
    if registry.get("status") != "completed" or not manifest.is_file():
        raise ValueError(
            "Terminal manifest repair requires completed unanchored output"
        )
    _validate_final_registry_snapshot(root)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Terminal manifest repair requires a non-empty manifest")
    relative_paths = [row["relative_path"] for row in rows]
    absolute_paths = [row["path"] for row in rows]
    roles = [row["artifact_role"] for row in rows]
    expected_paths = {
        path.relative_to(root).as_posix() for path in _all_local_files(root)
    }
    if (
        len(rows) != len(set(relative_paths))
        or len(rows) != len(set(absolute_paths))
        or len(rows) != len(set(roles))
        or set(relative_paths) != expected_paths
    ):
        raise ValueError("Unanchored terminal output universe drift")
    for row in rows:
        path = root / row["relative_path"]
        if (
            row["artifact_role"] != f"output:{row['relative_path']}"
            or row["path"] != str(path.resolve())
            or not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"Unanchored terminal output drift: {path}")
    qa_path = root / "qa.json"
    qa = _require_mapping(_read_json(qa_path), "terminal QA")
    unsigned_qa = {key: value for key, value in qa.items() if key != "payload_sha256"}
    if (
        qa.get("status") != "passed"
        or _payload_sha256(unsigned_qa) != qa.get("payload_sha256")
        or int(qa.get("terminal_output_artifact_count", -1)) != len(rows)
    ):
        raise ValueError("Unanchored terminal QA/output contract drift")
    registry.update(
        terminal_output_manifest_path=str(manifest.resolve()),
        terminal_output_manifest_sha256=_sha256_file(manifest),
        terminal_output_artifact_count=len(rows),
    )
    # jobs.json is intentionally outside output_hashes, so this is the only
    # authorized mutation in this exact interrupted-write state.
    _write_json(root / "registry/jobs.json", registry)


def qa_experiment(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    _validate_root(root)
    registry = _registry(root)
    development = [
        job for job in registry["jobs"] if job["experiment_stage"] == DEVELOPMENT_STAGE
    ]
    refit = [job for job in registry["jobs"] if job["experiment_stage"] == REFIT_STAGE]
    if len(development) != 36 or len(refit) != 36:
        raise ValueError("Terminal registry must contain 36+36 jobs")
    _validate_stage(root, DEVELOPMENT_STAGE)
    _validate_stage(root, REFIT_STAGE)
    _validate_frozen_selection(root)
    _validate_allowlist(root)
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    _validate_completed_q4_evaluation(root, resolved, registry)
    required = (
        "analysis/film_nolp_capacity_q3_pair_metrics.csv.gz",
        "analysis/film_nolp_capacity_q3_scores.csv",
        "analysis/film_nolp_capacity_q3_persistence_contrasts.csv",
        "analysis/film_nolp_capacity_q3_pairwise_contrasts.csv",
        "analysis/film_nolp_capacity_training_diagnostics.csv",
        "analysis/film_nolp_capacity_q3_selection.json",
        "analysis/refit_recipe_manifest.json",
        "analysis/film_nolp_capacity_q4_pair_metrics.csv.gz",
        "analysis/film_nolp_capacity_q4_scores.csv",
        "analysis/film_nolp_capacity_q4_persistence_contrasts.csv",
        "analysis/film_nolp_capacity_q4_pairwise_contrasts.csv",
        "analysis/film_nolp_capacity_q4_30m_secondary_scores.csv",
        "analysis/film_nolp_capacity_q4_30m_secondary_persistence.csv",
        "analysis/film_nolp_capacity_q4_30m_secondary_pairwise.csv",
        "analysis/film_nolp_capacity_q4_summary.json",
        "data_windows/q4/common_05m_q4.xlsx",
        "data_windows/q4/q4_window_manifest.json",
        "report/film_nolp_capacity_conclusion.md",
        "report/film_nolp_capacity_conclusion.html",
        "q4_checkpoint_allowlist.csv",
        "resource_summary.csv",
        "task_registry.csv",
        "resolved_config.yaml",
        "source_hashes.csv",
        "code_hashes.csv",
        "config_hashes.csv",
        "rolling_split_manifest.csv",
        "model_contract_manifest.json",
        "registry/final_registry_snapshot.json",
        "registry/experiment_status.json",
    )
    recovery_required = (
        (
            f"{Q4_RECOVERY_DIRECTORY}/{Q4_RECOVERY_LEDGER}",
            f"{Q4_RECOVERY_DIRECTORY}/{Q4_RECOVERY_CURRENT_CODE_HASHES}",
            f"{Q4_RECOVERY_DIRECTORY}/{Q4_RECOVERY_PRE_OUTPUT_HASHES}",
        )
        if registry.get("q4_recovery_applied")
        else ()
    )
    postprocess_recovery_required = (
        (
            f"{POSTPROCESS_RECOVERY_DIRECTORY}/{POSTPROCESS_RECOVERY_LEDGER}",
            (
                f"{POSTPROCESS_RECOVERY_DIRECTORY}/"
                f"{POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES}"
            ),
            (
                f"{POSTPROCESS_RECOVERY_DIRECTORY}/"
                f"{POSTPROCESS_RECOVERY_PRE_OUTPUT_HASHES}"
            ),
        )
        if registry.get("postprocess_recovery_applied")
        else ()
    )
    missing = [
        relative
        for relative in (
            *required,
            *recovery_required,
            *postprocess_recovery_required,
        )
        if not (root / relative).is_file()
    ]
    if missing:
        raise ValueError(f"Terminal artifacts missing: {missing}")
    _validate_final_registry_snapshot(root)
    _validate_prediction_cache_set(
        root,
        stage_name="q3_development_best_learned",
        jobs=development,
        checkpoint_role="generator_best_learned",
        mc_samples=16,
        row_count=148,
    )
    _validate_prediction_cache_set(
        root,
        stage_name="q4_refit_final",
        jobs=refit,
        checkpoint_role="generator_final",
        mc_samples=64,
        row_count=167,
    )
    q3_predictions = development
    q4_predictions = refit
    if registry.get("q4_recovery_applied"):
        if registry.get("postprocess_recovery_applied"):
            _validate_postprocess_recovery_bundle(root, require_registered=True)
        else:
            _validate_recovery_bundle(root, require_registered=True)
    manifest = root / "output_hashes.csv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        _validate_terminal_manifest_registry_anchor(root, registry, manifest, len(rows))
        if not (root / "qa.json").is_file():
            raise FileNotFoundError(root / "qa.json")
        relative_paths = [row["relative_path"] for row in rows]
        absolute_paths = [row["path"] for row in rows]
        roles = [row["artifact_role"] for row in rows]
        expected_paths = {
            path.relative_to(root).as_posix() for path in _all_local_files(root)
        }
        if (
            len(rows) != len(set(relative_paths))
            or len(rows) != len(set(absolute_paths))
            or len(rows) != len(set(roles))
            or set(relative_paths) != expected_paths
        ):
            raise ValueError("Duplicate output manifest path")
        for row in rows:
            path = root / row["relative_path"]
            if (
                row["artifact_role"] != f"output:{row['relative_path']}"
                or row["path"] != str(path.resolve())
                or not path.is_file()
                or path.stat().st_size != int(row["size_bytes"])
                or _sha256_file(path) != row["sha256"]
            ):
                raise ValueError(f"Terminal output drift: {path}")
        qa_path = root / "qa.json"
        qa = _require_mapping(_read_json(qa_path), "terminal QA")
        unsigned_qa = {
            key: value for key, value in qa.items() if key != "payload_sha256"
        }
        if (
            qa.get("status") != "passed"
            or _payload_sha256(unsigned_qa) != qa.get("payload_sha256")
            or int(qa.get("terminal_output_artifact_count", -1)) != len(rows)
        ):
            raise ValueError("Terminal QA/output manifest contract drift")
        return qa_path
    qa_path = root / "qa.json"
    artifact_count = len(_all_local_files(root)) + (0 if qa_path.is_file() else 1)
    report = {
        "schema_version": 1,
        "status": "passed",
        "development_jobs": 36,
        "refit_jobs": 36,
        "q3_prediction_cells": len(q3_predictions),
        "q4_prediction_cells": len(q4_predictions),
        "q4_interpretation": "retrospective_frozen_exploratory_not_confirmatory",
        "terminal_output_artifact_count": artifact_count,
        "completed_at_utc": _utc_now(),
    }
    report["payload_sha256"] = _payload_sha256(report)
    return _write_json(qa_path, report)


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    from scripts.rq3.news_first_vol_film_nolp_capacity_seed_report import render_report

    root = Path(output_dir).resolve()
    try:
        _validate_root(root)
    except ValueError:
        if not resume:
            raise
        _register_postprocess_recovery(root)
        _validate_root(root)
    registry = _registry(root)
    if registry.get("status") == "completed":
        if not (root / "output_hashes.csv").is_file() and not resume:
            raise ValueError("Completed registry is missing terminal output manifest")
        if (root / "output_hashes.csv").is_file():
            anchor_fields = (
                "terminal_output_manifest_path",
                "terminal_output_manifest_sha256",
                "terminal_output_artifact_count",
            )
            present = tuple(field in registry for field in anchor_fields)
            if not all(present):
                if not resume or any(present):
                    raise ValueError("Completed registry has partial terminal anchors")
                _repair_unanchored_terminal_manifest(root)
            return qa_experiment(root)
    if not registry.get("q4_evaluated"):
        raise RuntimeError("Postprocess requires completed Q4 evaluation")
    render_report(root)
    _resource_summary(root)
    registry.update(status="completed", completed_at_utc=_utc_now())
    _write_registry(root, registry)
    _write_experiment_status(root, "completed", current_stage="terminal")
    snapshot = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "registry": _registry(root),
        "experiment_status": _read_json(root / "registry/experiment_status.json"),
        "created_at_utc": _utc_now(),
    }
    snapshot["payload_sha256"] = _payload_sha256(snapshot)
    _write_json(root / "registry/final_registry_snapshot.json", snapshot)
    qa_experiment(root)
    files = _all_local_files(root)
    rows = [
        {
            "artifact_role": f"output:{path.relative_to(root).as_posix()}",
            "relative_path": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    _write_csv(root / "output_hashes.csv", rows, tuple(rows[0]))
    registry = _registry(root)
    registry.update(
        terminal_output_manifest_path=str((root / "output_hashes.csv").resolve()),
        terminal_output_manifest_sha256=_sha256_file(root / "output_hashes.csv"),
        terminal_output_artifact_count=len(rows),
    )
    # jobs.json is intentionally outside output_hashes to avoid recursive
    # self-reference through its terminal-manifest anchor.
    _write_json(root / "registry/jobs.json", registry)
    return qa_experiment(root)


def status_experiment(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if not root.exists():
        return {"status": "absent", "root": str(root)}
    registry = _registry(root)
    counts: dict[str, int] = {}
    for job in registry.get("jobs", []):
        state = str(
            _read_json(_job_status_path(root, str(job["job_id"]))).get("status")
        )
        counts[state] = counts.get(state, 0) + 1
    return {
        "status": registry.get("status"),
        "root": str(root),
        "job_status_counts": counts,
        "selection_frozen": bool(registry.get("selection_frozen")),
        "refit_complete": bool(registry.get("refit_complete")),
        "q4_evaluated": bool(registry.get("q4_evaluated")),
    }


def _pipeline_lock(root: Path) -> tuple[Path, int]:
    control = root.with_name(root.name + "_control")
    control.mkdir(parents=True, exist_ok=True)
    lock = control / "pipeline.lock"
    if lock.is_file():
        try:
            previous_pid = int(lock.read_text(encoding="utf-8").strip())
        except ValueError:
            previous_pid = -1
        if training._pid_is_live(previous_pid):
            raise RuntimeError(f"Live pipeline supervisor lock exists: {lock}")
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(descriptor, f"{os.getpid()}\n".encode())
    os.close(descriptor)
    (control / "pipeline.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    return lock, os.getpid()


def run_pipeline(
    config_path: str | Path, output_dir: str | Path, *, resume: bool = False
) -> Path:
    root = Path(output_dir).resolve()
    lock, _ = _pipeline_lock(root)
    try:
        if not _benchmark_result_path(root).is_file():
            run_benchmark(config_path, root, resume=resume)
        prepare_experiment(config_path, root, reuse=resume)
        if not _registry(root).get("selection_frozen"):
            launch_development(config_path, root, resume=True, dry_run=True)
            launch_development(config_path, root, resume=True, dry_run=False)
            freeze_selection(root)
        if not _registry(root).get("refit_complete"):
            launch_refit(config_path, root, resume=True)
        if not _registry(root).get("q4_evaluated"):
            evaluate_q4(root, resume=True)
        postprocess(root, resume=resume)
        return root
    finally:
        lock.unlink(missing_ok=True)


def run_news_first_vol_film_nolp_capacity_seed(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "prepare",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
) -> Path | dict[str, Any]:
    normalized = str(action).strip().lower()
    if normalized == "benchmark":
        return run_benchmark(config_path, output_dir, resume=resume)
    if normalized == "prepare":
        return prepare_experiment(config_path, output_dir, reuse=bool(reuse or resume))
    if normalized == "dry-run":
        return launch_development(config_path, output_dir, resume=resume, dry_run=True)
    if normalized == "launch-development":
        return launch_development(config_path, output_dir, resume=resume)
    if normalized == "freeze-selection":
        return freeze_selection(output_dir)
    if normalized == "launch-refit":
        return launch_refit(config_path, output_dir, resume=resume)
    if normalized == "evaluate-q4":
        return evaluate_q4(output_dir, resume=resume)
    if normalized == "postprocess":
        return postprocess(output_dir, resume=resume)
    if normalized == "qa":
        return qa_experiment(output_dir)
    if normalized == "status":
        return status_experiment(output_dir)
    if normalized == "run-pipeline":
        return run_pipeline(config_path, output_dir, resume=resume)
    if normalized == "worker":
        if not job_id:
            raise ValueError("--job-id is required for worker")
        return run_worker(output_dir, job_id, dry_run=worker_dry_run, resume=resume)
    raise ValueError(f"Unsupported action: {action}")


__all__ = [
    "BENCHMARK_STAGE",
    "DEVELOPMENT_STAGE",
    "REFIT_STAGE",
    "experiment_specs",
    "resolve_config",
    "run_news_first_vol_film_nolp_capacity_seed",
]
