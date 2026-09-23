"""Background FiLM-G + LP-Critic capacity/LR/seed development sweep.

This branch-local orchestrator owns an isolated 54-job Q3 development matrix.
It deliberately does not reuse the frozen NoLP capacity registry or read Q4.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Mapping, Sequence

import yaml

import scripts._path_setup  # noqa: F401
from scripts.rq3 import news_first_vol_training as training
from wgan_option.models.common import (
    critic_conditioning_fingerprint,
    generator_conditioning_fingerprint,
)


EXPERIMENT_KIND = "film_lp_capacity_lr_seed_q3_development_v1"
WARMUP_EXPERIMENT_KIND = "film_lp_capacity_seed_lr5e7_warmup10_q3_development_v1"
EXPERIMENT_KINDS = frozenset({EXPERIMENT_KIND, WARMUP_EXPERIMENT_KIND})
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_lp_critic_capacity_lr_3seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_film_lp_critic_capacity_lr_3seed_exact_ttm_v1"
)
GENERATOR_MODE = "film_conv_bottleneck_concat_v1"
CRITIC_MODE = "lp_concat_v1"
CAPACITIES = ("micro", "tiny", "small", "medium", "large", "legacy")
LEARNING_RATES = (5e-7, 2.5e-7, 1.25e-7)
SEEDS = (42, 202, 404)
GPU_IDS = (0, 1)
EXPECTED_JOB_COUNT = len(CAPACITIES) * len(LEARNING_RATES) * len(SEEDS)
GRID_SHA256 = "7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8"
SHAPE_FIELDS = (
    "gen_base_channels",
    "gen_text_hidden_dim",
    "gen_text_out_dim",
    "gen_hidden_dim",
    "disc_base_channels",
    "disc_text_hidden_dim",
    "disc_hidden_dim",
)
PARAMETER_FIELDS = (
    "expected_generator_parameters",
    "expected_critic_parameters",
    "expected_wgan_parameters",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = Path(__file__).resolve()

_payload_sha256 = training._payload_sha256
_sha256_file = training._sha256_file
_utc_now = training._utc_now
_write_json = training._write_json
_write_yaml = training._write_yaml
_atomic_write_text = training._atomic_write_text


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    return _require_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _lr_slug(value: float) -> str:
    text = f"{float(value):.8g}".replace(".", "p").replace("-", "_")
    return text


def _validate_resolved(resolved: Mapping[str, Any]) -> None:
    if int(resolved.get("schema_version", -1)) != 1:
        raise ValueError("Sweep schema_version must be 1")
    experiment_kind = str(resolved.get("experiment_kind"))
    if experiment_kind not in EXPERIMENT_KINDS:
        raise ValueError(f"experiment_kind must be one of {sorted(EXPERIMENT_KINDS)}")
    if tuple(map(str, resolved.get("capacities", ()))) != CAPACITIES:
        raise ValueError(f"capacities must be {CAPACITIES}")
    learning_rates = tuple(map(float, resolved.get("learning_rates", ())))
    if not learning_rates or len(set(learning_rates)) != len(learning_rates):
        raise ValueError("learning_rates must be non-empty and unique")
    if any(not math.isfinite(value) or value <= 0.0 for value in learning_rates):
        raise ValueError("learning_rates must be finite and positive")
    if tuple(map(int, resolved.get("seeds", ()))) != SEEDS:
        raise ValueError(f"seeds must be {SEEDS}")
    if int(resolved.get("max_epochs", -1)) != 30:
        raise ValueError("max_epochs must be 30")
    if int(resolved.get("early_stopping_min_epochs", -1)) != 15:
        raise ValueError("early_stopping_min_epochs must be 15")
    if int(resolved.get("early_stopping_patience", -1)) != 16:
        raise ValueError("early_stopping_patience must be 16")
    if int(resolved.get("validation_mc_samples", -1)) != 16:
        raise ValueError("validation_mc_samples must be 16")
    warmup_epochs = int(resolved.get("lr_warmup_epochs", 0))
    warmup_start_factor = float(resolved.get("lr_warmup_start_factor", 0.1))
    if warmup_epochs < 0 or warmup_epochs > int(resolved["max_epochs"]):
        raise ValueError("lr_warmup_epochs must be between 0 and max_epochs")
    if not math.isfinite(warmup_start_factor) or not (0.0 < warmup_start_factor <= 1.0):
        raise ValueError("lr_warmup_start_factor must be finite and in (0, 1]")
    if experiment_kind == EXPERIMENT_KIND:
        if learning_rates != LEARNING_RATES or warmup_epochs != 0:
            raise ValueError(
                "The original experiment kind requires its frozen three-LR grid "
                "and lr_warmup_epochs=0"
            )
    elif (
        learning_rates != (5e-7,)
        or warmup_epochs != 10
        or not math.isclose(warmup_start_factor, 0.1)
    ):
        raise ValueError(
            "The warmup experiment kind requires LR=5e-7, 10 warmup epochs, "
            "and start factor=0.1"
        )

    profiles = _require_mapping(resolved.get("profiles"), "profiles")
    if tuple(profiles) != CAPACITIES:
        raise ValueError("Capacity profile order drifted")
    expected_profile_fields = {*SHAPE_FIELDS, *PARAMETER_FIELDS}
    for capacity in CAPACITIES:
        profile = _require_mapping(profiles.get(capacity), f"profiles.{capacity}")
        if set(profile) != expected_profile_fields:
            raise ValueError(f"Profile fields drifted for {capacity}")
        if any(int(profile[field]) <= 0 for field in expected_profile_fields):
            raise ValueError(f"Profile values must be positive for {capacity}")

    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != GPU_IDS:
        raise ValueError("gpu_ids must be [0, 1]")
    if not 1 <= int(runtime.get("workers_per_gpu", -1)) <= 12:
        raise ValueError("workers_per_gpu must be between 1 and 12")
    if int(runtime.get("cpu_threads_per_job", -1)) != 1:
        raise ValueError("cpu_threads_per_job must be 1")
    if float(runtime.get("poll_interval_seconds", 0)) <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    if float(runtime.get("resource_sample_interval_seconds", 0)) <= 0:
        raise ValueError("resource_sample_interval_seconds must be positive")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    root = _require_mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    resolved = _require_mapping(
        root.get("film_lp_capacity_lr_seed_sweep"),
        "film_lp_capacity_lr_seed_sweep",
    )
    resolved.setdefault("lr_warmup_epochs", 0)
    resolved.setdefault("lr_warmup_start_factor", 0.1)
    resolved["source_config_path"] = str(source)
    resolved["base_training_config"] = str(
        _resolve_repo_path(str(resolved["base_training_config"]))
    )
    resolved["runtime"] = _require_mapping(resolved["runtime"], "runtime")
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(str(resolved["runtime"]["python_executable"]))
    )
    _validate_resolved(resolved)
    return resolved


def _architecture_contract(
    resolved: Mapping[str, Any], capacity: str
) -> dict[str, Any]:
    profile = _require_mapping(resolved["profiles"][capacity], capacity)
    payload = {
        "schema_version": 1,
        "capacity_profile": capacity,
        **{field: int(profile[field]) for field in SHAPE_FIELDS},
    }
    return {**payload, "architecture_profile_sha256": _payload_sha256(payload)}


def _model_contract(
    resolved: Mapping[str, Any], capacity: str, learning_rate: float
) -> dict[str, Any]:
    profile = _require_mapping(resolved["profiles"][capacity], capacity)
    architecture = _architecture_contract(resolved, capacity)
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
        "capacity_profile": capacity,
        "architecture_profile_sha256": architecture["architecture_profile_sha256"],
        "architecture": {
            key: value
            for key, value in architecture.items()
            if key != "architecture_profile_sha256"
        },
        "surface_grid_profile": "exact_ttm_16x16_v1",
        "surface_grid_sha256": GRID_SHA256,
        "embedding_mode": "lp",
        "embedding_dim": 1024,
        "generator_current_input_mode": "current_support_masked",
        "generator_noise_mode": "gaussian",
        "noise_dim": 32,
        "initial_learning_rate": float(learning_rate),
        **{field: int(profile[field]) for field in PARAMETER_FIELDS},
    }
    if int(resolved.get("lr_warmup_epochs", 0)) > 0:
        payload.update(
            {
                "lr_warmup_epochs": int(resolved["lr_warmup_epochs"]),
                "lr_warmup_start_factor": float(resolved["lr_warmup_start_factor"]),
            }
        )
    payload["model_contract_sha256"] = _payload_sha256(payload)
    return payload


def _job_id(
    capacity: str,
    learning_rate: float,
    seed: int,
    *,
    warmup_epochs: int = 0,
) -> str:
    warmup = f"_warmup{warmup_epochs}" if warmup_epochs > 0 else ""
    return f"film_lp_{capacity}_lr_{_lr_slug(learning_rate)}{warmup}_seed_{seed:03d}"


def experiment_specs(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    index = 0
    learning_rates = tuple(map(float, resolved["learning_rates"]))
    seeds = tuple(map(int, resolved["seeds"]))
    warmup_epochs = int(resolved.get("lr_warmup_epochs", 0))
    warmup_start_factor = float(resolved.get("lr_warmup_start_factor", 0.1))
    for capacity in CAPACITIES:
        for learning_rate in learning_rates:
            for seed in seeds:
                contract = _model_contract(resolved, capacity, learning_rate)
                spec = {
                    "job_id": _job_id(
                        capacity,
                        learning_rate,
                        seed,
                        warmup_epochs=warmup_epochs,
                    ),
                    "experiment_kind": str(resolved["experiment_kind"]),
                    "capacity_profile": capacity,
                    "learning_rate": learning_rate,
                    "lr_slug": _lr_slug(learning_rate),
                    "lr_warmup_epochs": warmup_epochs,
                    "lr_warmup_start_factor": warmup_start_factor,
                    "seed": seed,
                    "gpu_id": GPU_IDS[index % len(GPU_IDS)],
                    "generator_conditioning_mode": GENERATOR_MODE,
                    "critic_conditioning_mode": CRITIC_MODE,
                    "architecture_profile_sha256": contract[
                        "architecture_profile_sha256"
                    ],
                    "model_contract_sha256": contract["model_contract_sha256"],
                    "expected_generator_parameters": contract[
                        "expected_generator_parameters"
                    ],
                    "expected_critic_parameters": contract[
                        "expected_critic_parameters"
                    ],
                    "expected_wgan_parameters": contract["expected_wgan_parameters"],
                }
                spec["job_spec_sha256"] = _payload_sha256(spec)
                specs.append(spec)
                index += 1
    expected_job_count = len(CAPACITIES) * len(learning_rates) * len(seeds)
    if len(specs) != expected_job_count:
        raise AssertionError("Unexpected job count")
    if len({spec["job_id"] for spec in specs}) != expected_job_count:
        raise AssertionError("Duplicate job IDs")
    counts = {gpu: sum(spec["gpu_id"] == gpu for spec in specs) for gpu in GPU_IDS}
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError(f"Unbalanced GPU assignment: {counts}")
    return specs


def _training_payload(
    resolved: Mapping[str, Any], output_root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    base_path = Path(str(resolved["base_training_config"]))
    payload = _require_mapping(
        yaml.safe_load(base_path.read_text(encoding="utf-8")), "base"
    )
    profile = _require_mapping(
        resolved["profiles"][str(spec["capacity_profile"])], "profile"
    )
    for field in SHAPE_FIELDS:
        payload[field] = int(profile[field])
    learning_rate = float(spec["learning_rate"])
    payload.update(
        {
            "generator_conditioning_mode": GENERATOR_MODE,
            "critic_conditioning_mode": CRITIC_MODE,
            "learning_rate": learning_rate,
            "generator_learning_rate": learning_rate,
            "discriminator_learning_rate": learning_rate,
            "lr_warmup_epochs": int(spec["lr_warmup_epochs"]),
            "lr_warmup_start_factor": float(spec["lr_warmup_start_factor"]),
            "num_epochs": int(resolved["max_epochs"]),
            "early_stopping_min_epochs": int(resolved["early_stopping_min_epochs"]),
            "early_stopping_patience": int(resolved["early_stopping_patience"]),
            "validation_mc_samples": int(resolved["validation_mc_samples"]),
            "seed": int(spec["seed"]),
            "news_first_capacity_profile": str(spec["capacity_profile"]),
            "news_first_lr_profile": (
                f"capacity_seed_{spec['lr_slug']}"
                + (
                    f"_warmup{int(spec['lr_warmup_epochs'])}"
                    if int(spec["lr_warmup_epochs"]) > 0
                    else ""
                )
            ),
            "news_first_architecture_profile_sha256": str(
                spec["architecture_profile_sha256"]
            ),
            "news_first_model_contract_sha256": str(spec["model_contract_sha256"]),
            "news_first_materialize_validation_loader": True,
            "news_first_materialize_test_loader": False,
            "output_root": str(
                (
                    output_root
                    / "runs"
                    / str(spec["capacity_profile"])
                    / str(spec["lr_slug"])
                    / f"seed_{int(spec['seed']):03d}"
                ).resolve()
            ),
        }
    )
    return payload


def _registry_path(output_root: Path) -> Path:
    return output_root / "registry/jobs.json"


def _job_status_path(output_root: Path, job_id: str) -> Path:
    return output_root / f"registry/jobs/{job_id}.status.json"


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "artifact_role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def prepare(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> dict[str, Any]:
    registry_path = _registry_path(output_root)
    source_path = Path(str(resolved["source_config_path"]))
    base_path = Path(str(resolved["base_training_config"]))
    source_hashes = {
        "source_config_sha256": _sha256_file(source_path),
        "base_training_config_sha256": _sha256_file(base_path),
        "orchestrator_sha256": _sha256_file(ORCHESTRATOR_PATH),
    }
    specs = experiment_specs(resolved)

    if registry_path.is_file():
        if not resume:
            raise FileExistsError(
                f"Registry already exists; pass --resume: {registry_path}"
            )
        registry = _read_json(registry_path)
        if registry.get("experiment_kind") != resolved["experiment_kind"]:
            raise ValueError("Existing registry experiment kind drifted")
        if any(registry.get(key) != value for key, value in source_hashes.items()):
            raise ValueError("Source/config drift detected; resume refused")
        if registry.get("jobs_payload_sha256") != _payload_sha256(specs):
            raise ValueError("Job matrix drift detected; resume refused")
        for job in registry["jobs"]:
            config_path = Path(str(job["config_path"]))
            if (
                not config_path.is_file()
                or _sha256_file(config_path) != job["config_sha256"]
            ):
                raise ValueError(f"Generated job config drift: {config_path}")
        return registry

    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("configs", "control", "logs", "registry/jobs", "analysis"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for spec in specs:
        config_path = output_root / f"configs/{spec['job_id']}.yaml"
        payload = _training_payload(resolved, output_root, spec)
        _write_yaml(config_path, payload)
        job = {
            **spec,
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256_file(config_path),
            "run_root": str(Path(str(payload["output_root"])).resolve()),
        }
        jobs.append(job)
        _write_json(
            _job_status_path(output_root, str(spec["job_id"])),
            {
                "job_id": spec["job_id"],
                "state": "pending",
                "attempt": 0,
                "updated_at": _utc_now(),
                "job_spec_sha256": spec["job_spec_sha256"],
            },
        )

    registry = {
        "schema_version": 1,
        "experiment_kind": str(resolved["experiment_kind"]),
        "created_at": _utc_now(),
        "source_config_path": str(source_path),
        "base_training_config_path": str(base_path),
        **source_hashes,
        "expected_job_count": len(specs),
        "jobs_payload_sha256": _payload_sha256(specs),
        "jobs": jobs,
    }
    _write_json(registry_path, registry)
    return registry


def _verify_artifacts(status: Mapping[str, Any]) -> None:
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, Sequence) or not artifacts:
        raise ValueError("Completed status has no artifacts")
    for raw in artifacts:
        artifact = _require_mapping(raw, "artifact")
        path = Path(str(artifact["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["size_bytes"])
            or _sha256_file(path) != artifact["sha256"]
        ):
            raise ValueError(f"Artifact drift: {path}")


def _discover_completed(job: Mapping[str, Any]) -> dict[str, Any] | None:
    run_root = Path(str(job["run_root"]))
    candidates = sorted(run_root.glob("*/metrics/best_learned_checkpoint.json"))
    for best_path in reversed(candidates):
        run_dir = best_path.parents[1]
        paths = {
            "best_learned": best_path,
            "generator_best_learned": run_dir / "checkpoints/generator_best_learned.pt",
            "discriminator_best_learned": run_dir
            / "checkpoints/discriminator_best_learned.pt",
            "training_metrics": run_dir / "metrics/training_metrics.csv",
            "resolved_config": run_dir / "metrics/training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
        }
        if not all(path.is_file() for path in paths.values()):
            continue
        if "Training complete" not in paths["run_log"].read_text(
            encoding="utf-8", errors="replace"
        ):
            continue
        best = _read_json(best_path)
        config = _require_mapping(
            yaml.safe_load(paths["resolved_config"].read_text(encoding="utf-8")),
            "resolved config",
        )
        expected = {
            "seed": int(job["seed"]),
            "news_first_capacity_profile": str(job["capacity_profile"]),
            "news_first_architecture_profile_sha256": str(
                job["architecture_profile_sha256"]
            ),
            "news_first_model_contract_sha256": str(job["model_contract_sha256"]),
            "generator_conditioning_mode": GENERATOR_MODE,
            "critic_conditioning_mode": CRITIC_MODE,
            "lr_warmup_epochs": int(job.get("lr_warmup_epochs", 0)),
            "lr_warmup_start_factor": float(job.get("lr_warmup_start_factor", 0.1)),
            "news_first_materialize_test_loader": False,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            continue
        if not (
            math.isclose(
                float(config["generator_learning_rate"]), float(job["learning_rate"])
            )
            and math.isclose(
                float(config["discriminator_learning_rate"]),
                float(job["learning_rate"]),
            )
            and str(best.get("model_contract_sha256"))
            == str(job["model_contract_sha256"])
            and int(best.get("lr_warmup_epochs", 0))
            == int(job.get("lr_warmup_epochs", 0))
            and math.isclose(
                float(best.get("lr_warmup_start_factor", 0.1)),
                float(job.get("lr_warmup_start_factor", 0.1)),
            )
        ):
            continue
        with paths["training_metrics"].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        epochs = [int(row["epoch"]) for row in rows]
        if not epochs or epochs[0] != 0 or epochs != list(range(epochs[-1] + 1)):
            continue
        if not all(math.isfinite(float(row["val_recon"])) for row in rows):
            continue
        warmup_epochs = int(job.get("lr_warmup_epochs", 0))
        if warmup_epochs > 0:
            start_factor = float(job["lr_warmup_start_factor"])
            target_lr = float(job["learning_rate"])
            warmup_rows_valid = True
            for row in rows:
                epoch = int(row["epoch"])
                if epoch > warmup_epochs:
                    continue
                schedule_epoch = max(1, epoch)
                progress = (
                    1.0
                    if warmup_epochs == 1
                    else float(schedule_epoch - 1) / float(warmup_epochs - 1)
                )
                expected_lr = target_lr * (
                    start_factor + progress * (1.0 - start_factor)
                )
                if not (
                    math.isclose(float(row["g_lr"]), expected_lr, abs_tol=1e-15)
                    and math.isclose(float(row["d_lr"]), expected_lr, abs_tol=1e-15)
                ):
                    warmup_rows_valid = False
                    break
            if not warmup_rows_valid:
                continue
        metrics = _require_mapping(best.get("metrics"), "best metrics")
        artifacts = [_artifact(path, role) for role, path in paths.items()]
        return {
            "run_dir": str(run_dir.resolve()),
            "best_epoch": int(best["best_epoch"]),
            "best_mae": float(metrics["val_recon"]),
            "persistence_mae": float(metrics["val_current_recon"]),
            "completed_epochs": epochs[-1],
            "artifacts": artifacts,
        }
    return None


def _read_status(output_root: Path, job_id: str) -> dict[str, Any]:
    return _read_json(_job_status_path(output_root, job_id))


def _write_status(output_root: Path, job_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(_job_status_path(output_root, job_id), dict(payload))


def _counts(registry: Mapping[str, Any], output_root: Path) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "complete": 0, "failed": 0}
    for job in registry["jobs"]:
        state = str(_read_status(output_root, str(job["job_id"])).get("state"))
        counts[state] = counts.get(state, 0) + 1
    return counts


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _write_supervisor_status(
    registry: Mapping[str, Any],
    output_root: Path,
    *,
    state: str,
    active: Mapping[str, Any],
    message: str = "",
) -> None:
    _write_json(
        output_root / "control/status.json",
        {
            "experiment_kind": registry["experiment_kind"],
            "state": state,
            "pid": os.getpid(),
            "pid_alive": True,
            "updated_at": _utc_now(),
            "counts": _counts(registry, output_root),
            "active_jobs": sorted(active),
            "message": message,
        },
    )


def _resource_snapshot(output_root: Path) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "timestamp": _utc_now(),
        "returncode": result.returncode,
        "rows": result.stdout.strip().splitlines(),
    }
    path = output_root / "control/resource_snapshots.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_summary(registry: Mapping[str, Any], output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        status = _read_status(output_root, str(job["job_id"]))
        if status.get("state") != "complete":
            raise ValueError("Cannot summarize an incomplete registry")
        persistence = float(status["persistence_mae"])
        mae = float(status["best_mae"])
        rows.append(
            {
                "job_id": job["job_id"],
                "capacity_profile": job["capacity_profile"],
                "learning_rate": job["learning_rate"],
                "lr_warmup_epochs": job.get("lr_warmup_epochs", 0),
                "lr_warmup_start_factor": job.get("lr_warmup_start_factor", 0.1),
                "seed": job["seed"],
                "gpu_id": job["gpu_id"],
                "total_parameters": job["expected_wgan_parameters"],
                "best_epoch": status["best_epoch"],
                "completed_epochs": status["completed_epochs"],
                "best_mae": mae,
                "persistence_mae": persistence,
                "improvement_vs_persistence_pct": (persistence - mae)
                / persistence
                * 100.0,
                "run_dir": status["run_dir"],
            }
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(output_root / "analysis/job_summary.csv", buffer.getvalue())


def run_pipeline(
    resolved: Mapping[str, Any], output_root: Path, *, resume: bool
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "control/supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            "Another supervisor already holds the experiment lock"
        ) from exc

    registry = prepare(resolved, output_root, resume=resume)
    _atomic_write_text(output_root / "control/supervisor.pid", f"{os.getpid()}\n")
    stop_requested = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    pending_by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in GPU_IDS}
    for raw_job in registry["jobs"]:
        job = dict(raw_job)
        status = _read_status(output_root, str(job["job_id"]))
        if status.get("state") == "complete":
            _verify_artifacts(status)
            continue
        recovered = _discover_completed(job)
        if recovered is not None:
            _write_status(
                output_root,
                str(job["job_id"]),
                {
                    "job_id": job["job_id"],
                    "state": "complete",
                    "attempt": int(status.get("attempt", 0)),
                    "updated_at": _utc_now(),
                    "job_spec_sha256": job["job_spec_sha256"],
                    **recovered,
                },
            )
            continue
        pending_by_gpu[int(job["gpu_id"])].append(job)

    runtime = _require_mapping(resolved["runtime"], "runtime")
    workers_per_gpu = int(runtime["workers_per_gpu"])
    python_executable = str(runtime["python_executable"])
    poll_interval = float(runtime["poll_interval_seconds"])
    resource_interval = float(runtime["resource_sample_interval_seconds"])
    active: dict[str, dict[str, Any]] = {}
    failed_message = ""
    last_resource = 0.0
    _write_supervisor_status(registry, output_root, state="running", active=active)

    while any(pending_by_gpu.values()) or active:
        if stop_requested or failed_message:
            for record in active.values():
                record["process"].terminate()
            for record in active.values():
                try:
                    record["process"].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    record["process"].kill()
                record["handle"].close()
            state = "interrupted" if stop_requested else "failed"
            _write_supervisor_status(
                registry,
                output_root,
                state=state,
                active={},
                message=failed_message or "Supervisor received a stop signal",
            )
            raise RuntimeError(failed_message or "Supervisor interrupted")

        for gpu in GPU_IDS:
            running_on_gpu = sum(record["gpu_id"] == gpu for record in active.values())
            while running_on_gpu < workers_per_gpu and pending_by_gpu[gpu]:
                job = pending_by_gpu[gpu].pop(0)
                job_id = str(job["job_id"])
                previous = _read_status(output_root, job_id)
                attempt = int(previous.get("attempt", 0)) + 1
                log_path = output_root / f"logs/{job_id}.attempt_{attempt:02d}.log"
                handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONPATH"] = "src:."
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["OMP_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
                env["MKL_NUM_THREADS"] = str(runtime["cpu_threads_per_job"])
                command = [
                    python_executable,
                    "scripts/train/train_vol.py",
                    "--config",
                    str(job["config_path"]),
                    "--train-only",
                ]
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[job_id] = {
                    "process": process,
                    "handle": handle,
                    "gpu_id": gpu,
                    "job": job,
                    "attempt": attempt,
                    "log_path": log_path,
                }
                _write_status(
                    output_root,
                    job_id,
                    {
                        "job_id": job_id,
                        "state": "running",
                        "attempt": attempt,
                        "pid": process.pid,
                        "gpu_id": gpu,
                        "started_at": _utc_now(),
                        "updated_at": _utc_now(),
                        "job_spec_sha256": job["job_spec_sha256"],
                        "stdout_log": str(log_path.resolve()),
                    },
                )
                print(
                    f"launched {job_id} pid={process.pid} gpu={gpu} attempt={attempt}",
                    flush=True,
                )
                running_on_gpu += 1

        for job_id, record in list(active.items()):
            process = record["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            record["handle"].close()
            job = record["job"]
            if returncode != 0:
                failed_message = (
                    f"{job_id} exited with code {returncode}; see {record['log_path']}"
                )
                _write_status(
                    output_root,
                    job_id,
                    {
                        "job_id": job_id,
                        "state": "failed",
                        "attempt": record["attempt"],
                        "returncode": returncode,
                        "updated_at": _utc_now(),
                        "job_spec_sha256": job["job_spec_sha256"],
                        "stdout_log": str(record["log_path"].resolve()),
                    },
                )
            else:
                completed = _discover_completed(job)
                if completed is None:
                    failed_message = f"{job_id} exited 0 but artifacts are incomplete"
                else:
                    _write_status(
                        output_root,
                        job_id,
                        {
                            "job_id": job_id,
                            "state": "complete",
                            "attempt": record["attempt"],
                            "returncode": 0,
                            "updated_at": _utc_now(),
                            "job_spec_sha256": job["job_spec_sha256"],
                            "stdout_log": str(record["log_path"].resolve()),
                            **completed,
                        },
                    )
                    print(
                        f"completed {job_id} epoch={completed['best_epoch']} "
                        f"mae={completed['best_mae']:.12g}",
                        flush=True,
                    )
            del active[job_id]

        now = time.monotonic()
        if now - last_resource >= resource_interval:
            _resource_snapshot(output_root)
            last_resource = now
        _write_supervisor_status(
            registry,
            output_root,
            state="running",
            active=active,
            message=failed_message,
        )
        if any(pending_by_gpu.values()) or active:
            time.sleep(poll_interval)

    _write_summary(registry, output_root)
    _write_supervisor_status(registry, output_root, state="complete", active={})
    print(f"all {len(registry['jobs'])} jobs completed", flush=True)


def status(output_root: Path) -> dict[str, Any]:
    registry_path = _registry_path(output_root)
    if not registry_path.is_file():
        return {"state": "not_prepared", "output_root": str(output_root)}
    registry = _read_json(registry_path)
    status_path = output_root / "control/status.json"
    payload = _read_json(status_path) if status_path.is_file() else {}
    pid_path = output_root / "control/supervisor.pid"
    pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.is_file() else 0
    payload.update(
        {
            "output_root": str(output_root),
            "pid": pid,
            "pid_alive": bool(pid and _pid_alive(pid)),
            "counts": _counts(registry, output_root),
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run-pipeline", "status"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", default=DEFAULT_CONFIG)
        sub.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
        if action in {"prepare", "run-pipeline"}:
            sub.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_root = _resolve_repo_path(args.output_dir)
    if args.action == "status":
        print(json.dumps(status(output_root), indent=2, sort_keys=True))
        return 0
    resolved = resolve_config(args.config)
    if args.action == "prepare":
        registry = prepare(resolved, output_root, resume=bool(args.resume))
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "job_count": len(registry["jobs"]),
                    "gpu_counts": {
                        str(gpu): sum(job["gpu_id"] == gpu for job in registry["jobs"])
                        for gpu in GPU_IDS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    run_pipeline(resolved, output_root, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
