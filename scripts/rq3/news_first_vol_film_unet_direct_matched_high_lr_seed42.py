"""High FiLM-projection LR sweep for direct matched-LP training.

This branch-local profile is deliberately additive.  It delegates the full
training/evaluation lifecycle to the already-audited direct matched-LP LR
orchestrator while replacing only the experiment identity and five FiLM
projection learning rates.  It never consumes a parent or continuation
checkpoint.

The shared prediction core historically derived the text-overlay mode from a
closed arm-name table.  These high-LR arm names are branch-local, so this
profile also installs its matched-LP overlay resolver into the prediction core
for the duration of each call and restores every patched symbol on exit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42 as low_lr,
)


DEFAULT_CONFIG = (
    "configs/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.yaml"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_high_lr_"
    "seed42_exact_ttm_rolling_v1"
)
EXPERIMENT_KIND = (
    "rq12_news_first_vol_film_unet_text128_c32_nolp_direct_matched_high_lr_"
    "seed42_rolling_v1"
)
INTERPRETATION = "retrospective_rolling_development_single_seed_descriptive"
DIRECT_ARMS = (
    "film_lr_5e6",
    "film_lr_1e5",
    "film_lr_2p5e5",
    "film_lr_5e5",
    "film_lr_1e4",
)
FILM_LEARNING_RATES: Mapping[str, float] = {
    "film_lr_5e6": 5.0e-6,
    "film_lr_1e5": 1.0e-5,
    "film_lr_2p5e5": 2.5e-5,
    "film_lr_5e5": 5.0e-5,
    "film_lr_1e4": 1.0e-4,
}
BRIDGE_ARM = "film_lr_5e6"
STRESS_ARM = "film_lr_1e4"
STRESS_CANARY_EPOCHS = 5
EXPECTED_TRAINING_JOBS = 20
EXPECTED_PREDICTION_CELLS = 20
EXPECTED_PAIR_METRIC_ROWS = 2_500
WORKER_MODULE = "scripts.rq3.news_first_vol_film_unet_direct_matched_high_lr_seed42"
INFERENCE_DETERMINISM_KIND = "direct_matched_high_film_lr_inference_determinism_v1"
BENCHMARK_RESULT_KIND = "direct_matched_high_film_lr_full_matrix_epoch1_benchmark_v1"
STRESS_CANARY_RESULT_KIND = "direct_matched_high_film_lr_1e4_epoch5_canary_v1"
REGISTRY_KIND = "direct_matched_high_film_lr_task_registry_v1"
QA_KIND = "direct_matched_high_film_lr_terminal_qa_v1"
HIGH_ANALYSIS_MODULE = (
    "scripts.rq3.news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis"
)

SOURCE_CODE_RELATIVE_PATHS = tuple(
    dict.fromkeys(
        (
            "scripts/rq3/news_first_vol_film_unet_direct_matched_high_lr_seed42.py",
            "scripts/rq3/"
            "news_first_vol_film_unet_direct_matched_high_lr_seed42_analysis.py",
            *low_lr.SOURCE_CODE_RELATIVE_PATHS,
        )
    )
)


_BASE_BENCHMARK = low_lr.benchmark
_BASE_PREPARE_EXPERIMENT = low_lr.prepare_experiment
_BASE_PREPARE = low_lr.prepare
_BASE_DRY_RUN = low_lr.dry_run
_BASE_WORKER = low_lr.worker
_BASE_LAUNCH = low_lr.launch
_BASE_FREEZE_EVALUATION = low_lr.freeze_evaluation
_BASE_PREDICT = low_lr.predict
_BASE_STATUS = low_lr.status
_BASE_POSTPROCESS = low_lr.postprocess
_BASE_QA = low_lr.qa
_BASE_RUN_PIPELINE = low_lr.run_pipeline
_BASE_MODEL_CONTRACT = low_lr.model_contract
_BASE_VALIDATE_GPU_BALANCE = low_lr.validate_gpu_balance


def _analysis_module() -> Any:
    return importlib.import_module(HIGH_ANALYSIS_MODULE)


def _overlay_mode(arm: str) -> str:
    if str(arm) not in DIRECT_ARMS:
        raise ValueError(f"Unknown matched high-FiLM-LR arm: {arm}")
    return "lp_mean_l2"


def _stress_result_path(formal_root: Path) -> Path:
    return formal_root.with_name(formal_root.name + "_control") / (
        "stress_canary_result.json"
    )


def _stress_root(formal_root: Path) -> Path:
    return formal_root.with_name(formal_root.name + "_stress_canary_1e4_epoch5")


def _validate_stress_metrics(metrics: pd.DataFrame, *, job_id: str) -> None:
    required = {"epoch", "g_lr_film_projection"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Stress-canary metrics missing {missing}: {job_id}")
    epochs = pd.to_numeric(metrics["epoch"], errors="raise").astype(int)
    if epochs.tolist() != list(range(STRESS_CANARY_EPOCHS + 1)):
        raise ValueError(f"Stress-canary epoch trace drift: {job_id}")
    # Epoch zero intentionally has no train-loss values. Learned epochs must
    # nevertheless be completely finite across every numeric metric.
    learned_numeric = metrics.loc[epochs >= 1].select_dtypes(include="number")
    if learned_numeric.empty or not np.isfinite(learned_numeric.to_numpy(float)).all():
        raise ValueError(f"Stress-canary contains NaN/Inf: {job_id}")
    film_lr = pd.to_numeric(metrics["g_lr_film_projection"], errors="raise").astype(
        float
    )
    if not math.isclose(
        float(film_lr.iloc[0]),
        FILM_LEARNING_RATES[STRESS_ARM],
        rel_tol=0.0,
        abs_tol=1e-18,
    ):
        raise ValueError(f"Stress-canary initial FiLM LR drift: {job_id}")


def _verify_stress_result(
    config: Mapping[str, Any], formal_root: Path
) -> dict[str, Any]:
    result_path = _stress_result_path(formal_root)
    payload = low_lr.direct.read_json(result_path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if (
        payload.get("kind") != STRESS_CANARY_RESULT_KIND
        or payload.get("status") != "passed"
        or low_lr.direct.payload_sha256(unsigned) != payload.get("payload_sha256")
        or payload.get("source_config_sha256") != config["source_config_sha256"]
        or payload.get("stress_arm") != STRESS_ARM
        or int(payload.get("epoch_count", -1)) != STRESS_CANARY_EPOCHS
        or int(payload.get("job_count", -1)) != len(low_lr.direct.FOLDS)
    ):
        raise ValueError("High-FiLM-LR stress-canary result drift")
    registry_path = low_lr.direct._verify_frozen_file(
        payload["canary_root_registry_path"],
        payload["canary_root_registry_sha256"],
    )
    root = Path(str(payload["canary_root"])).resolve()
    if registry_path != (root / "registry/task_registry.json").resolve():
        raise ValueError("Stress-canary registry path drift")
    low_lr.direct.validate_root(root)
    registry = low_lr.direct.read_registry(root)
    selected = [dict(job) for job in registry["jobs"] if str(job["arm"]) == STRESS_ARM]
    if len(selected) != len(low_lr.direct.FOLDS):
        raise ValueError("Stress canary must contain one 1e-4 job per fold")
    observed_ids = sorted(str(job["job_id"]) for job in selected)
    if observed_ids != sorted(map(str, payload.get("job_ids") or [])):
        raise ValueError("Stress-canary job universe drift")
    for job in selected:
        status = low_lr.direct.read_json(
            low_lr.direct._status_path(root, str(job["job_id"]))
        )
        if not low_lr.direct._completed_valid(job, status):
            raise ValueError(f"Incomplete stress-canary job: {job['job_id']}")
        metrics_artifact = low_lr.direct._artifact(status, "training_metrics_csv")
        metrics = pd.read_csv(metrics_artifact["path"])
        _validate_stress_metrics(metrics, job_id=str(job["job_id"]))
    return payload


def _stress_canary_active(
    config: Mapping[str, Any], formal_root: Path, *, resume: bool
) -> Path:
    """Run four five-epoch 1e-4 jobs while a high-LR profile is active."""

    result_path = _stress_result_path(formal_root)
    if result_path.is_file():
        _verify_stress_result(config, formal_root)
        return result_path
    root = _stress_root(formal_root)
    if not root.exists():
        workers_per_gpu = int(config["runtime"]["fallback_workers_per_gpu"])
        _BASE_PREPARE(
            config,
            root,
            workers_per_gpu=workers_per_gpu,
            num_epochs=STRESS_CANARY_EPOCHS,
            root_mode="benchmark",
        )
    elif not resume:
        raise FileExistsError(root)
    low_lr.direct.validate_root(root)
    registry = low_lr.direct.read_registry(root)
    selected = [dict(job) for job in registry["jobs"] if str(job["arm"]) == STRESS_ARM]
    if len(selected) != len(low_lr.direct.FOLDS):
        raise ValueError("Stress canary requires four 1e-4 fold jobs")
    peak_ram = 0.0
    for wave in sorted({int(job["wave"]) for job in selected}):
        pending = []
        for job in selected:
            if int(job["wave"]) != wave:
                continue
            status = low_lr.direct.read_json(
                low_lr.direct._status_path(root, str(job["job_id"]))
            )
            if low_lr.direct._completed_valid(job, status):
                if resume:
                    continue
                raise RuntimeError(
                    f"Completed stress-canary job requires --resume: {job['job_id']}"
                )
            pending.append(job)
        peak_ram = max(
            peak_ram,
            low_lr.direct._run_wave(root, pending, wave=wave, resume=resume),
        )
    resources = low_lr.direct._resource_peaks(root)
    runtime = config["runtime"]
    if resources["peak_gpu_memory_gib"] >= float(
        runtime["preflight_max_peak_gpu_memory_gib"]
    ):
        raise RuntimeError("Stress-canary GPU-memory gate failed")
    if peak_ram >= float(runtime["preflight_max_host_ram_fraction"]):
        raise RuntimeError("Stress-canary host-RAM gate failed")
    registry_path = root / "registry/task_registry.json"
    payload = {
        "schema_version": 1,
        "kind": STRESS_CANARY_RESULT_KIND,
        "status": "passed",
        "source_config_sha256": config["source_config_sha256"],
        "stress_arm": STRESS_ARM,
        "film_learning_rate": FILM_LEARNING_RATES[STRESS_ARM],
        "epoch_count": STRESS_CANARY_EPOCHS,
        "job_count": len(selected),
        "job_ids": sorted(str(job["job_id"]) for job in selected),
        "canary_root": str(root.resolve()),
        "canary_root_registry_path": str(registry_path.resolve()),
        "canary_root_registry_sha256": low_lr.direct.sha256_file(registry_path),
        "peak_host_ram_fraction": float(peak_ram),
        **resources,
        "completed_at_utc": low_lr.direct.utc_now(),
    }
    payload["payload_sha256"] = low_lr.direct.payload_sha256(payload)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = low_lr.direct.write_json(result_path, payload)
    _verify_stress_result(config, formal_root)
    return result


def _benchmark_with_stress(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    benchmark_path = _BASE_BENCHMARK(config_path, output_dir, resume=resume)
    config = low_lr.load_config(config_path)
    _stress_canary_active(config, Path(output_dir).resolve(), resume=resume)
    return benchmark_path


def _prepare_experiment_with_stress(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = low_lr.load_config(config_path)
    formal_root = Path(output_dir).resolve()
    stress = _verify_stress_result(config, formal_root)
    result = _BASE_PREPARE_EXPERIMENT(config_path, formal_root, resume=resume)
    registry = low_lr.direct.read_registry(result)
    evidence = {
        "path": str(_stress_result_path(formal_root).resolve()),
        "sha256": low_lr.direct.sha256_file(_stress_result_path(formal_root)),
        "kind": STRESS_CANARY_RESULT_KIND,
        "stress_arm": STRESS_ARM,
        "epoch_count": STRESS_CANARY_EPOCHS,
        "job_count": int(stress["job_count"]),
    }
    if registry.get("stress_canary_evidence") not in (None, evidence):
        raise ValueError("Formal registry stress-canary evidence drift")
    if registry.get("stress_canary_evidence") is None:
        registry["stress_canary_evidence"] = evidence
        low_lr.direct.write_registry(result, registry)
    return result


@contextmanager
def high_lr_profile() -> Iterator[None]:
    """Install the high-LR profile into low/direct/core and restore it safely."""

    analysis_module = _analysis_module()
    replacements: dict[str, Any] = {
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "DEFAULT_OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "EXPERIMENT_KIND": EXPERIMENT_KIND,
        "INTERPRETATION": INTERPRETATION,
        "DIRECT_ARMS": DIRECT_ARMS,
        "FILM_LEARNING_RATES": FILM_LEARNING_RATES,
        "EXPECTED_TRAINING_JOBS": EXPECTED_TRAINING_JOBS,
        "EXPECTED_PREDICTION_CELLS": EXPECTED_PREDICTION_CELLS,
        "EXPECTED_PAIR_METRIC_ROWS": EXPECTED_PAIR_METRIC_ROWS,
        "WORKER_MODULE": WORKER_MODULE,
        "INFERENCE_DETERMINISM_KIND": INFERENCE_DETERMINISM_KIND,
        "BENCHMARK_RESULT_KIND": BENCHMARK_RESULT_KIND,
        "REGISTRY_KIND": REGISTRY_KIND,
        "QA_KIND": QA_KIND,
        "SOURCE_CODE_RELATIVE_PATHS": SOURCE_CODE_RELATIVE_PATHS,
        "lr_analysis": analysis_module,
        "_overlay_mode": _overlay_mode,
        "benchmark": _benchmark_with_stress,
        "prepare_experiment": _prepare_experiment_with_stress,
    }
    originals = {name: getattr(low_lr, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(low_lr, name, value)
        with low_lr.lr_sweep_profile():
            core = low_lr.direct.core
            original_core_overlay_mode = core._overlay_mode
            core._overlay_mode = _overlay_mode
            try:
                yield
            finally:
                core._overlay_mode = original_core_overlay_mode
    finally:
        for name, value in originals.items():
            setattr(low_lr, name, value)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with high_lr_profile():
        config = low_lr.load_config(config_path)
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis must be a mapping")
    if analysis.get("bridge_arm") != BRIDGE_ARM:
        raise ValueError("High-LR bridge arm drift")
    if analysis.get("stress_endpoint_arm") != STRESS_ARM:
        raise ValueError("High-LR stress endpoint drift")
    if bool(analysis.get("test_based_lr_selection_permitted", True)):
        raise ValueError("Test-based LR selection must remain disabled")
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    with high_lr_profile():
        low_lr.validate_config(config)
    analysis = config.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis must be a mapping")
    if (
        analysis.get("bridge_arm") != BRIDGE_ARM
        or analysis.get("stress_endpoint_arm") != STRESS_ARM
        or bool(analysis.get("test_based_lr_selection_permitted", True))
    ):
        raise ValueError("High-LR analysis boundary drift")


def planned_specs(config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    resolved = load_config() if config is None else dict(config)
    validate_config(resolved)
    with high_lr_profile():
        return low_lr.planned_specs(resolved)


def _training_payload(
    config: Mapping[str, Any],
    root: Path,
    spec: Mapping[str, Any],
    *,
    num_epochs: int | None = None,
    slots_per_gpu: int | None = None,
) -> dict[str, Any]:
    with high_lr_profile():
        return low_lr._training_payload(
            config,
            root,
            spec,
            num_epochs=num_epochs,
            slots_per_gpu=slots_per_gpu,
        )


def benchmark(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    with high_lr_profile():
        return _benchmark_with_stress(config_path, output_dir, resume=resume)


def stress_canary(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    with high_lr_profile():
        config = low_lr.load_config(config_path)
        return _stress_canary_active(config, Path(output_dir).resolve(), resume=resume)


def prepare_experiment(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    with high_lr_profile():
        return _prepare_experiment_with_stress(config_path, output_dir, resume=resume)


def prepare(
    config_or_path: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    **kwargs: Any,
) -> Path:
    with high_lr_profile():
        return _BASE_PREPARE(config_or_path, output_root, **kwargs)


def dry_run(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    return benchmark(config_path, output_dir, resume=resume)


def worker(
    output_dir: str | Path,
    job_id: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    with high_lr_profile():
        return _BASE_WORKER(output_dir, job_id, resume=resume, dry_run=dry_run)


def launch(output_dir: str | Path, *, resume: bool = False) -> Path:
    with high_lr_profile():
        return _BASE_LAUNCH(output_dir, resume=resume)


def freeze_evaluation(output_dir: str | Path, *, resume: bool = False) -> Path:
    with high_lr_profile():
        return _BASE_FREEZE_EVALUATION(output_dir, resume=resume)


def predict(output_dir: str | Path, *, resume: bool = False) -> Path:
    with high_lr_profile():
        return _BASE_PREDICT(output_dir, resume=resume)


def postprocess(output_dir: str | Path, *, resume: bool = False) -> Path:
    with high_lr_profile():
        return _BASE_POSTPROCESS(output_dir, resume=resume)


def qa(output_dir: str | Path, *, resume: bool = False) -> Path:
    with high_lr_profile():
        return _BASE_QA(output_dir, resume=resume)


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    with high_lr_profile():
        return _BASE_STATUS(output_dir)


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    with high_lr_profile():
        return _BASE_RUN_PIPELINE(config_path, output_dir, resume=resume)


def model_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    with high_lr_profile():
        return _BASE_MODEL_CONTRACT(config)


def validate_gpu_balance(specs: Sequence[Mapping[str, Any]]) -> None:
    with high_lr_profile():
        _BASE_VALIDATE_GPU_BALANCE(specs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "benchmark",
            "stress-canary",
            "prepare",
            "dry-run",
            "worker",
            "launch",
            "freeze-evaluation",
            "predict",
            "postprocess",
            "qa",
            "status",
            "run-pipeline",
        ),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    action = str(args.action)
    if action == "benchmark":
        result: Any = benchmark(args.config, args.output_dir, resume=args.resume)
    elif action == "stress-canary":
        result = stress_canary(args.config, args.output_dir, resume=args.resume)
    elif action == "prepare":
        result = prepare_experiment(args.config, args.output_dir, resume=args.resume)
    elif action == "dry-run":
        result = dry_run(args.config, args.output_dir, resume=args.resume)
    elif action == "worker":
        if not args.job_id:
            raise SystemExit("worker requires --job-id")
        result = worker(
            args.output_dir,
            args.job_id,
            resume=args.resume,
            dry_run=args.worker_dry_run,
        )
    elif action == "launch":
        result = launch(args.output_dir, resume=args.resume)
    elif action == "freeze-evaluation":
        result = freeze_evaluation(args.output_dir, resume=args.resume)
    elif action == "predict":
        result = predict(args.output_dir, resume=args.resume)
    elif action == "postprocess":
        result = postprocess(args.output_dir, resume=args.resume)
    elif action == "qa":
        result = qa(args.output_dir, resume=args.resume)
    elif action == "status":
        result = status(args.output_dir)
    else:
        result = run_pipeline(args.config, args.output_dir, resume=args.resume)
    print(
        json.dumps(result, indent=2, sort_keys=True)
        if isinstance(result, Mapping)
        else result
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BRIDGE_ARM",
    "DIRECT_ARMS",
    "EXPECTED_PAIR_METRIC_ROWS",
    "EXPECTED_PREDICTION_CELLS",
    "EXPECTED_TRAINING_JOBS",
    "FILM_LEARNING_RATES",
    "STRESS_ARM",
    "STRESS_CANARY_EPOCHS",
    "benchmark",
    "freeze_evaluation",
    "high_lr_profile",
    "load_config",
    "model_contract",
    "planned_specs",
    "predict",
    "prepare",
    "run_pipeline",
    "stress_canary",
    "validate_config",
    "validate_gpu_balance",
]
