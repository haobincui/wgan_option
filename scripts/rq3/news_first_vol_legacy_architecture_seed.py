"""Legacy-width 2x2 Generator/Critic architecture experiment on Q3.

Nine cells are trained locally.  The three FiLM-Generator + NoLP-Critic cells
are immutable references to the completed formal capacity experiment.  This
module never materializes, loads, predicts, or evaluates Q4.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml

from scripts.rq3 import news_first_vol_film_nolp_capacity_seed as capacity
from scripts.rq3 import news_first_vol_generator_film_critic_factorial as factorial
from scripts.rq3 import news_first_vol_training as training
from wgan_option.models.common import (
    critic_conditioning_fingerprint,
    generator_conditioning_fingerprint,
)
from wgan_option.utils.text_ablation import REAL_TEXT


ROOT_KEY = training.ROOT_KEY
REPO_ROOT = training.REPO_ROOT
EXPERIMENT_KIND = "legacy_generator_critic_architecture_seed_q3"
DEFAULT_CONFIG = "configs/rq3/news_first_vol_legacy_architecture_seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/rq3_news_first_vol_legacy_architecture_seed_exact_ttm_v1"
)
STAGE = "development_q3_architecture_comparison"
GENERATOR_MODES = (
    "bottleneck_concat_v1",
    "film_conv_bottleneck_concat_v1",
)
CRITIC_MODES = ("lp_concat_v1", "lp_disabled_same_shape_v1")
ANCHOR = (GENERATOR_MODES[0], CRITIC_MODES[0])
REFERENCE_CELL = (GENERATOR_MODES[1], CRITIC_MODES[1])
SEEDS = (42, 202, 404)
TOLERANCE = 5
EXPECTED_LOCAL_JOBS = 9
EXPECTED_REFERENCES = 3
EXPECTED_CELLS = 12

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
        resolved.get("legacy_architecture_seed_sweep"),
        "legacy_architecture_seed_sweep",
    )


def _exact_float(value: Any, expected: float, label: str) -> None:
    observed = float(value)
    if not math.isfinite(observed) or observed != expected:
        raise ValueError(f"{label} must equal {expected}; observed={observed}")


def _validate_config(resolved: Mapping[str, Any]) -> None:
    datasets = _require_mapping(resolved.get("datasets"), "datasets")
    split = _require_mapping(resolved.get("split"), "split")
    sweep = _sweep(resolved)
    runtime = _require_mapping(resolved.get("runtime"), "runtime")
    if (
        not bool(sweep.get("enabled"))
        or sweep.get("experiment_kind") != EXPERIMENT_KIND
    ):
        raise ValueError("Legacy architecture experiment kind/enabled drift")
    if (
        tuple(map(str, sweep.get("generator_conditioning_modes", ())))
        != GENERATOR_MODES
    ):
        raise ValueError("Generator mode/order drift")
    if tuple(map(str, sweep.get("critic_conditioning_modes", ()))) != CRITIC_MODES:
        raise ValueError("Critic mode/order drift")
    anchor = _require_mapping(sweep.get("anchor"), "anchor")
    if (
        anchor.get("generator_conditioning_mode"),
        anchor.get("critic_conditioning_mode"),
    ) != ANCHOR:
        raise ValueError("Anchor must be Concat-G + LP-Critic")
    reference = _require_mapping(sweep.get("external_reference"), "external_reference")
    if (
        reference.get("generator_conditioning_mode"),
        reference.get("critic_conditioning_mode"),
    ) != REFERENCE_CELL:
        raise ValueError("External reference must be FiLM-G + NoLP-Critic")
    if tuple(map(int, sweep.get("seeds", ()))) != SEEDS:
        raise ValueError("Seeds must be 42/202/404")
    if int(sweep.get("tolerance_minutes", -1)) != TOLERANCE:
        raise ValueError("Only the common 5m lane is permitted")
    if str(sweep.get("text_ablation_mode")) != REAL_TEXT:
        raise ValueError("Only real_text is permitted")
    if str(datasets.get("support_mask_mode")) != "raw_joint":
        raise ValueError("raw_joint support is frozen")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("Common evaluation tolerance must be 5m")
    if str(split.get("development_train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("Training cutoff drift")
    if str(split.get("development_validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("Q3 cutoff drift")
    if int(split.get("validation_mc_samples", -1)) != 16:
        raise ValueError("Validation MC must be 16")
    _exact_float(sweep.get("initial_learning_rate"), 5e-7, "initial LR")
    _exact_float(sweep.get("scheduler_min_lr"), 5e-8, "scheduler floor")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("Exactly GPU0/GPU1 are required")
    if int(runtime.get("slots_per_gpu", -1)) != 5:
        raise ValueError("The nine local tasks use five slots/GPU")
    expected_grid = factorial._surface_grid_contract()
    if [int(value) for value in sweep.get("maturity_days_grid", ())] != expected_grid[
        "maturity_days_grid"
    ]:
        raise ValueError("Exact-TTM maturity axis drift")
    strikes = [float(value) for value in sweep.get("strike_grid", ())]
    if len(strikes) != 16 or any(
        abs(left - right) > 1e-12
        for left, right in zip(strikes, expected_grid["strike_grid"])
    ):
        raise ValueError("Exact-TTM strike axis drift")
    model = _require_mapping(
        _require_mapping(resolved.get("models"), "models").get("wgan"), "wgan"
    )
    values = _require_mapping(model.get("training"), "wgan.training")
    frozen = {
        "embedding_dim": 1024,
        "noise_dim": 32,
        "generator_noise_mode": "gaussian",
        "generator_current_input_mode": "current_support_masked",
        "critic_normalization_mode": "legacy_instance_norm_v1",
        "gen_base_channels": 32,
        "gen_res_blocks": 0,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
        "gen_hidden_dim": 1024,
        "disc_base_channels": 32,
        "disc_res_blocks": 0,
        "disc_text_hidden_dim": 128,
        "disc_hidden_dim": 786,
        "residual_output_mode": "identity_softplus_residual",
        "batch_size": 16,
        "discriminator_iter": 5,
        "news_first_label_reliability_mode": "none",
        "news_first_materialize_validation_loader": True,
        "news_first_materialize_test_loader": False,
    }
    for key, expected in frozen.items():
        if values.get(key) != expected:
            raise ValueError(f"Frozen training contract drift: {key}")


def resolve_config(config_path: str | Path) -> dict[str, Any]:
    source = _resolve_repo_path(config_path)
    raw = _load_yaml(source, "legacy architecture config")
    resolved = deepcopy(_require_mapping(raw.get(ROOT_KEY), ROOT_KEY))
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    resolved["runtime"]["python_executable"] = str(
        _resolve_repo_path(resolved["runtime"].get("python_executable", sys.executable))
    )
    reference = resolved["legacy_architecture_seed_sweep"]["external_reference"]
    reference["experiment_root"] = str(_resolve_repo_path(reference["experiment_root"]))
    resolved["source_config_path"] = str(source)
    _validate_config(resolved)
    return resolved


def _architecture_slug(generator: str, critic: str) -> str:
    return "_".join(
        (
            "gconcat" if generator == GENERATOR_MODES[0] else "gfilm",
            "dlp" if critic == CRITIC_MODES[0] else "dnolp",
        )
    )


def all_specs() -> list[dict[str, Any]]:
    return [
        {
            "generator_conditioning_mode": generator,
            "critic_conditioning_mode": critic,
            "seed": seed,
            "tolerance_minutes": TOLERANCE,
            "text_ablation_mode": REAL_TEXT,
            "external_reference": (generator, critic) == REFERENCE_CELL,
        }
        for generator in GENERATOR_MODES
        for critic in CRITIC_MODES
        for seed in SEEDS
    ]


def local_specs() -> list[dict[str, Any]]:
    return [spec for spec in all_specs() if not spec["external_reference"]]


def _job_id(spec: Mapping[str, Any]) -> str:
    slug = _architecture_slug(
        str(spec["generator_conditioning_mode"]),
        str(spec["critic_conditioning_mode"]),
    )
    return f"dev_legacy_{slug}_lr_5e_07_seed_{int(spec['seed']):03d}_real_05m"


def _expected_parameters(resolved: Mapping[str, Any], generator: str) -> dict[str, int]:
    raw = _require_mapping(
        _require_mapping(
            _sweep(resolved).get("expected_parameters"), "expected_parameters"
        ).get(generator),
        f"expected_parameters.{generator}",
    )
    return {
        "generator": int(raw["generator"]),
        "critic": int(raw["critic"]),
        "total": int(raw["total"]),
    }


def _model_contract(
    resolved: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    generator = str(spec["generator_conditioning_mode"])
    critic = str(spec["critic_conditioning_mode"])
    parameters = _expected_parameters(resolved, generator)
    shape = {
        "gen_base_channels": 32,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
        "gen_hidden_dim": 1024,
        "disc_base_channels": 32,
        "disc_text_hidden_dim": 128,
        "disc_hidden_dim": 786,
    }
    architecture = {
        "schema_version": 1,
        "capacity_profile": "legacy",
        **shape,
    }
    architecture_sha = _payload_sha256(architecture)
    conditioning = {
        "generator_conditioning_mode": generator,
        "generator_conditioning_fingerprint": generator_conditioning_fingerprint(
            generator
        ),
        "critic_conditioning_mode": critic,
        "critic_conditioning_fingerprint": critic_conditioning_fingerprint(critic),
    }
    grid = factorial._surface_grid_contract()
    contract = {
        "schema_version": 1,
        **conditioning,
        "conditioning_contract_sha256": _payload_sha256(conditioning),
        "capacity_profile": "legacy",
        "architecture_profile_sha256": architecture_sha,
        "architecture": architecture,
        "surface_grid_profile": str(_sweep(resolved)["surface_grid_profile"]),
        "surface_grid_sha256": grid["surface_grid_sha256"],
        "embedding_mode": "lp",
        "embedding_dim": 1024,
        "generator_current_input_mode": "current_support_masked",
        "generator_noise_mode": "gaussian",
        "noise_dim": 32,
        "initial_learning_rate": 5e-7,
        "expected_generator_parameters": parameters["generator"],
        "expected_critic_parameters": parameters["critic"],
        "expected_wgan_parameters": parameters["total"],
    }
    contract["model_contract_sha256"] = _payload_sha256(contract)
    return contract


def _training_payload(
    resolved: Mapping[str, Any], root: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    base = factorial._training_payload(
        resolved,
        root,
        spec=spec,
        stage=factorial.DEVELOPMENT_STAGE,
        benchmark=False,
    )
    contract = _model_contract(resolved, spec)
    base.update(
        {
            "data_path": str(
                (root / "data_windows/tolerance_05m_pre_q4.xlsx").resolve()
            ),
            "news_first_common_eval_data_path": str(
                (root / "data_windows/tolerance_05m_pre_q4.xlsx").resolve()
            ),
            "news_first_capacity_profile": "legacy",
            "generator_conditioning_mode": spec["generator_conditioning_mode"],
            "critic_conditioning_mode": spec["critic_conditioning_mode"],
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
                    / STAGE
                    / _architecture_slug(
                        str(spec["generator_conditioning_mode"]),
                        str(spec["critic_conditioning_mode"]),
                    )
                    / f"seed_{int(spec['seed']):03d}"
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


def _artifact(status: Mapping[str, Any], role: str) -> dict[str, Any]:
    rows = [
        row for row in status.get("artifacts", ()) if row.get("artifact_role") == role
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one {role} artifact")
    row = dict(rows[0])
    path = Path(str(row["path"]))
    if not path.is_file() or _sha256_file(path) != row["sha256"]:
        raise ValueError(f"Artifact drift: {path}")
    return row


def _snapshot_file(
    source: Path, target: Path, expected_sha: str, label: str
) -> dict[str, Any]:
    if not source.is_file() or _sha256_file(source) != expected_sha:
        raise ValueError(f"Frozen {label} SHA drift: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if _sha256_file(target) != expected_sha:
        raise ValueError(f"Snapshot {label} SHA drift: {target}")
    return _manifest_row(label, target)


def _reference_manifest(
    resolved: Mapping[str, Any], *, snapshot_root: Path | None = None
) -> dict[str, Any]:
    contract = _require_mapping(
        _sweep(resolved)["external_reference"], "external_reference"
    )
    source_root = Path(str(contract["experiment_root"]))
    registry_path = source_root / "registry/jobs.json"
    registry = _require_mapping(_read_json(registry_path), "source registry")
    if registry.get("experiment_kind") != capacity.EXPERIMENT_KIND:
        raise ValueError("Reference root is not the formal capacity experiment")
    capacity_selection_source = (
        source_root / "analysis/film_nolp_capacity_q3_selection.json"
    )
    capacity_selection = _require_mapping(
        _read_json(capacity_selection_source), "capacity selection"
    )
    if (
        _sha256_file(capacity_selection_source) != contract["capacity_selection_sha256"]
        or capacity_selection.get("selection_sha256")
        != contract["capacity_selection_payload_sha256"]
        or _payload_sha256(
            {
                key: value
                for key, value in capacity_selection.items()
                if key != "selection_sha256"
            }
        )
        != contract["capacity_selection_payload_sha256"]
        or capacity_selection.get("point_leader") != "legacy"
        or capacity_selection.get("candidate_statistical_support") != "descriptive_only"
        or capacity_selection.get("q4_used_for_capacity_selection") is not False
    ):
        raise ValueError("Capacity-selection motivation lineage drift")
    capacity_selection_path = capacity_selection_source
    if snapshot_root is not None:
        capacity_selection_snapshot = _snapshot_file(
            capacity_selection_source,
            snapshot_root / "capacity_q3_selection.json",
            str(contract["capacity_selection_sha256"]),
            "capacity_q3_selection",
        )
        capacity_selection_path = Path(capacity_selection_snapshot["path"])
    expected_seeds = _require_mapping(contract.get("seeds"), "external_reference.seeds")
    rows: list[dict[str, Any]] = []
    expected_reference_gpu = dict(zip(SEEDS, (1, 0, 1)))
    for seed in SEEDS:
        expected = _require_mapping(
            expected_seeds.get(seed) or expected_seeds.get(str(seed)),
            f"reference seed {seed}",
        )
        source_id = str(expected["job_id"])
        matches = [
            dict(job) for job in registry["jobs"] if job.get("job_id") == source_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Reference job is not unique: {source_id}")
        job = matches[0]
        required_job = {
            "job_spec_sha256": expected["job_spec_sha256"],
            "config_sha256": expected["config_sha256"],
            "capacity_profile": "legacy",
            "generator_conditioning_mode": REFERENCE_CELL[0],
            "critic_conditioning_mode": REFERENCE_CELL[1],
            "text_ablation_mode": REAL_TEXT,
            "tolerance_minutes": 5,
            "seed": seed,
            "gpu_id": expected_reference_gpu[seed],
            "dataset_sha256": contract["source_dataset_sha256"],
            "surface_grid_sha256": contract["surface_grid_sha256"],
            "model_contract_sha256": contract["model_contract_sha256"],
            "expected_generator_parameters": int(
                contract["expected_generator_parameters"]
            ),
            "expected_critic_parameters": int(contract["expected_critic_parameters"]),
            "expected_wgan_parameters": int(contract["expected_wgan_parameters"]),
        }
        for key, value in required_job.items():
            if job.get(key) != value:
                raise ValueError(f"Reference {source_id} {key} drift")
        status_path = _job_status_path(source_root, source_id)
        if _sha256_file(status_path) != expected["status_sha256"]:
            raise ValueError(f"Reference status SHA drift: {source_id}")
        status = _require_mapping(_read_json(status_path), "reference status")
        if status.get("status") != "completed":
            raise ValueError(f"Reference job is not complete: {source_id}")
        training_config_path = Path(job["training_config_path"])
        if (
            not training_config_path.is_file()
            or _sha256_file(training_config_path) != expected["config_sha256"]
        ):
            raise ValueError(f"Reference training config SHA drift: {source_id}")
        artifacts = {}
        for role in (
            "generator_best_learned",
            "discriminator_best_learned",
            "best_learned_checkpoint",
            "training_metrics_csv",
        ):
            row = _artifact(status, role)
            expected_key = f"{role}_sha256"
            if expected_key in expected and row["sha256"] != expected[expected_key]:
                raise ValueError(f"Reference {role} SHA drift: {source_id}")
            artifacts[role] = row
        metadata = _require_mapping(
            _read_json(Path(artifacts["best_learned_checkpoint"]["path"])),
            "reference checkpoint metadata",
        )
        best_epoch = int(
            metadata.get("best_learned_epoch_ge_1", metadata.get("best_epoch", 0))
        )
        if best_epoch != int(expected["best_learned_epoch"]):
            raise ValueError(f"Reference best epoch drift: {source_id}")
        prediction_dir = (
            source_root / "analysis/predictions/q3_development_best_learned"
        )
        prediction_path = prediction_dir / f"{source_id}.csv.gz"
        prediction_manifest_path = prediction_dir / f"{source_id}.manifest.json"
        if (
            _sha256_file(prediction_path) != expected["prediction_sha256"]
            or _sha256_file(prediction_manifest_path)
            != expected["prediction_manifest_sha256"]
        ):
            raise ValueError(f"Reference prediction SHA drift: {source_id}")
        prediction_manifest = _require_mapping(
            _read_json(prediction_manifest_path), "reference prediction manifest"
        )
        prediction_unsigned = {
            key: value
            for key, value in prediction_manifest.items()
            if key != "manifest_sha256"
        }
        if prediction_manifest.get("manifest_sha256") != _payload_sha256(
            prediction_unsigned
        ):
            raise ValueError(f"Reference prediction self-hash drift: {source_id}")
        expected_prediction = {
            "manifest_sha256": expected["prediction_manifest_payload_sha256"],
            "panel_universe_sha256": contract["q3_panel_universe_sha256"],
            "row_count": 148,
            "checkpoint_sha256": expected["generator_best_learned_sha256"],
            "mc_samples": 16,
            "seed": seed,
            "tolerance_minutes": 5,
        }
        for key, value in expected_prediction.items():
            if prediction_manifest.get(key) != value:
                raise ValueError(f"Reference prediction {key} drift: {source_id}")
        snapshots: dict[str, Any] = {}
        if snapshot_root is not None:
            directory = snapshot_root / f"seed_{seed:03d}"
            snapshots["source_status"] = _snapshot_file(
                status_path,
                directory / "source_status.json",
                str(expected["status_sha256"]),
                "source_status",
            )
            snapshots["training_config"] = _snapshot_file(
                training_config_path,
                directory / "training_config.yaml",
                str(expected["config_sha256"]),
                "training_config",
            )
            copied_artifacts = {}
            for role, row in artifacts.items():
                source = Path(row["path"])
                suffix = (
                    "json"
                    if role == "best_learned_checkpoint"
                    else ("csv" if role == "training_metrics_csv" else "pt")
                )
                copied_artifacts[role] = _snapshot_file(
                    source,
                    directory / f"{role}.{suffix}",
                    str(row["sha256"]),
                    role,
                )
            artifacts = copied_artifacts
            snapshots["prediction"] = _snapshot_file(
                prediction_path,
                directory / "q3_prediction.csv.gz",
                str(expected["prediction_sha256"]),
                "q3_prediction",
            )
            snapshots["prediction_manifest"] = _snapshot_file(
                prediction_manifest_path,
                directory / "q3_prediction_source_manifest.json",
                str(expected["prediction_manifest_sha256"]),
                "q3_prediction_source_manifest",
            )
            status_path = Path(snapshots["source_status"]["path"])
            training_config_path = Path(snapshots["training_config"]["path"])
        pseudo = {
            "job_id": _job_id(
                {
                    "generator_conditioning_mode": REFERENCE_CELL[0],
                    "critic_conditioning_mode": REFERENCE_CELL[1],
                    "seed": seed,
                }
            ),
            "source_job_id": source_id,
            "source_job_spec_sha256": job["job_spec_sha256"],
            "source_config_sha256": job["config_sha256"],
            "source_gpu_id": int(job["gpu_id"]),
            "source_training_config_path": str(training_config_path.resolve()),
            "source_status_path": str(status_path.resolve()),
            "source_status_sha256": expected["status_sha256"],
            "source_run_dir": str(Path(status["run_dir"]).resolve()),
            "generator_conditioning_mode": REFERENCE_CELL[0],
            "critic_conditioning_mode": REFERENCE_CELL[1],
            "seed": seed,
            "tolerance_minutes": 5,
            "text_ablation_mode": REAL_TEXT,
            "external_reference": True,
            "dataset_sha256": contract["source_dataset_sha256"],
            "surface_grid_sha256": contract["surface_grid_sha256"],
            "model_contract_sha256": contract["model_contract_sha256"],
            "expected_generator_parameters": int(
                contract["expected_generator_parameters"]
            ),
            "expected_critic_parameters": int(contract["expected_critic_parameters"]),
            "expected_wgan_parameters": int(contract["expected_wgan_parameters"]),
            "best_learned_epoch": best_epoch,
            "artifacts": artifacts,
            "prediction": {
                "path": str(
                    Path(
                        snapshots.get("prediction", {"path": prediction_path})["path"]
                    ).resolve()
                ),
                "sha256": expected["prediction_sha256"],
                "source_manifest_path": str(
                    Path(
                        snapshots.get(
                            "prediction_manifest", {"path": prediction_manifest_path}
                        )["path"]
                    ).resolve()
                ),
                "source_manifest_sha256": expected["prediction_manifest_sha256"],
                "source_manifest_payload_sha256": expected[
                    "prediction_manifest_payload_sha256"
                ],
                "panel_universe_sha256": contract["q3_panel_universe_sha256"],
                "row_count": 148,
                "mc_samples": 16,
            },
        }
        pseudo["reference_spec_sha256"] = _payload_sha256(pseudo)
        rows.append(pseudo)
    output = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "source_experiment_root": str(source_root.resolve()),
        "source_registry_path_at_freeze": str(registry_path.resolve()),
        "source_registry_sha256_at_freeze": _sha256_file(registry_path),
        "capacity_selection": {
            "path": str(capacity_selection_path.resolve()),
            "sha256": contract["capacity_selection_sha256"],
            "payload_sha256": contract["capacity_selection_payload_sha256"],
            "point_leader": "legacy",
            "candidate_statistical_support": "descriptive_only",
            "q3_post_selection_motivation_only": True,
        },
        "references": rows,
        "reference_count": len(rows),
        "frozen_at_utc": _utc_now(),
    }
    output["payload_sha256"] = _payload_sha256(output)
    return output


def _job_spec_sha(job: Mapping[str, Any]) -> str:
    return _payload_sha256(
        {key: value for key, value in job.items() if key != "job_spec_sha256"}
    )


def _assigned_specs() -> list[dict[str, Any]]:
    rows = []
    counts = {0: 0, 1: 0}
    for spec in local_specs():
        generator_index = GENERATOR_MODES.index(
            str(spec["generator_conditioning_mode"])
        )
        critic_index = CRITIC_MODES.index(str(spec["critic_conditioning_mode"]))
        seed_index = SEEDS.index(int(spec["seed"]))
        # For each seed, G/D main-effect comparisons remain on the same
        # physical GPU.  The nine local tasks split 4/5 overall.
        gpu = (generator_index + critic_index + seed_index) % 2
        local = counts[gpu]
        counts[gpu] += 1
        rows.append({**spec, "gpu_id": gpu, "gpu_slot": local, "wave": 1})
    if sorted(counts.values()) != [4, 5]:
        raise ValueError(f"GPU assignment imbalance: {counts}")
    return rows


def _initial_status(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "experiment_stage": STAGE,
        "status": "prepared",
        "attempt": 0,
        "config_sha256": job["config_sha256"],
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluator_rows": 0,
        "artifacts": [],
    }


def _code_paths() -> list[Path]:
    names = (
        "scripts/rq3/main.py",
        "scripts/rq3/news_first_vol_legacy_architecture_seed.py",
        "scripts/rq3/news_first_vol_legacy_architecture_seed_analysis.py",
        "scripts/rq3/news_first_vol_legacy_architecture_seed_report.py",
        "scripts/rq3/news_first_vol_generator_film_critic_factorial.py",
        "scripts/rq3/news_first_vol_film_nolp_capacity_seed.py",
        "scripts/rq3/news_first_vol_comparison_analysis.py",
        "scripts/rq3/news_first_vol_training.py",
        "src/wgan_option/config.py",
        "src/wgan_option/models/common.py",
        "src/wgan_option/models/generator.py",
        "src/wgan_option/models/discriminator.py",
        "src/wgan_option/models/gan_model.py",
        "src/wgan_option/train_vol_xlsx.py",
        "src/wgan_option/utils/inference_helpers.py",
    )
    return [(REPO_ROOT / name).resolve() for name in names]


def _write_hash_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_csv(path, [dict(row) for row in rows], tuple(rows[0]))


def _verify_hash_manifest(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty hash manifest: {path}")
    roles = [row["artifact_role"] for row in rows]
    paths = [row["path"] for row in rows]
    if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
        raise ValueError(f"Hash manifest role/path uniqueness drift: {path}")
    for row in rows:
        target = Path(row["path"])
        if (
            not target.is_file()
            or target.stat().st_size != int(row["size_bytes"])
            or _sha256_file(target) != row["sha256"]
        ):
            raise ValueError(f"Hash drift: {target}")


def _validate_split(resolved: Mapping[str, Any], workbook: Path) -> Path:
    import pandas as pd

    frame = factorial._supported_frame(resolved, workbook, TOLERANCE)
    train_end = pd.Timestamp(resolved["split"]["development_train_end_utc"])
    validation_end = pd.Timestamp(resolved["split"]["development_validation_end_utc"])
    train = frame[frame["effective_origin_utc"] < train_end]
    q3 = frame[
        (frame["effective_origin_utc"] >= train_end)
        & (frame["effective_origin_utc"] < validation_end)
    ]
    rows = []
    for name, selected, expected_key in (
        ("development_train", train, "expected_development_counts"),
        ("development_q3", q3, "expected_q3_counts"),
    ):
        counts = factorial._counts(selected)
        expected = {
            key: int(value)
            for key, value in _require_mapping(
                _sweep(resolved)[expected_key], expected_key
            ).items()
        }
        if counts != expected:
            raise ValueError(f"{name} counts drift: {counts} != {expected}")
        rows.append({"split": name, "tolerance_minutes": 5, **counts})
    if set(train["pair_id"]) & set(q3["pair_id"]):
        raise ValueError("Pair leakage between train and Q3")
    if set(train["session_id"]) & set(q3["session_id"]):
        raise ValueError("Session leakage between train and Q3")
    return _write_csv(workbook.parent / "split_manifest.csv", rows, tuple(rows[0]))


def prepare_experiment(
    config_path: str | Path, output_dir: str | Path, *, reuse: bool = False
) -> Path:
    root = Path(output_dir).resolve()
    if root.exists():
        if not reuse:
            raise FileExistsError(root)
        _validate_root(root)
        return root
    resolved = resolve_config(config_path)
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
    source_root = Path(_sweep(resolved)["external_reference"]["experiment_root"])
    source_workbook = source_root / "data_windows/pre_q4/tolerance_05m_pre_q4.xlsx"
    expected_dataset_sha = str(
        _sweep(resolved)["external_reference"]["source_dataset_sha256"]
    )
    if (
        not source_workbook.is_file()
        or _sha256_file(source_workbook) != expected_dataset_sha
    ):
        raise ValueError("Frozen 5m pre-Q4 source workbook drift")
    workbook = root / "data_windows/tolerance_05m_pre_q4.xlsx"
    shutil.copy2(source_workbook, workbook)
    if _sha256_file(workbook) != expected_dataset_sha:
        raise ValueError("Copied dataset SHA drift")
    split_manifest = _validate_split(resolved, workbook)
    reference_manifest = _write_json(
        root / "reference_manifest.json",
        _reference_manifest(resolved, snapshot_root=root / "references"),
    )
    jobs = []
    config_rows = []
    for spec in _assigned_specs():
        payload = _training_payload(resolved, root, spec)
        job_id = _job_id(spec)
        config_path_out = root / "configs" / f"{job_id}.yaml"
        _write_yaml(config_path_out, payload)
        contract = _model_contract(resolved, spec)
        job = {
            "job_id": job_id,
            "experiment_kind": EXPERIMENT_KIND,
            "experiment_stage": STAGE,
            "capacity_profile": "legacy",
            "generator_conditioning_mode": spec["generator_conditioning_mode"],
            "critic_conditioning_mode": spec["critic_conditioning_mode"],
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
            "seed": int(spec["seed"]),
            "tolerance_minutes": 5,
            "text_ablation_mode": REAL_TEXT,
            "support_mask_mode": "raw_joint",
            "external_reference": False,
            "gpu_id": int(spec["gpu_id"]),
            "gpu_slot": int(spec["gpu_slot"]),
            "wave": int(spec["wave"]),
            "training_config_path": str(config_path_out.resolve()),
            "config_sha256": _sha256_file(config_path_out),
            "dataset_path": str(workbook.resolve()),
            "dataset_sha256": _sha256_file(workbook),
            "output_root": str(Path(payload["output_root"]).resolve()),
        }
        job["job_spec_sha256"] = _job_spec_sha(job)
        jobs.append(job)
        config_rows.append(_manifest_row(f"training_config:{job_id}", config_path_out))
    if (
        len(jobs) != EXPECTED_LOCAL_JOBS
        or len({job["job_id"] for job in jobs}) != EXPECTED_LOCAL_JOBS
    ):
        raise ValueError("Local job matrix drift")
    source_rows = [
        _manifest_row("source_config", _resolve_repo_path(config_path)),
        _manifest_row("resolved_config", resolved_path),
        _manifest_row("frozen_pre_q4_05m", workbook),
        _manifest_row("split_manifest", split_manifest),
        _manifest_row("external_reference_manifest", reference_manifest),
    ]
    code_rows = [
        _manifest_row(f"code:{path.relative_to(REPO_ROOT)}", path)
        for path in _code_paths()
    ]
    _write_hash_manifest(root / "source_hashes.csv", source_rows)
    _write_hash_manifest(root / "code_hashes.csv", code_rows)
    _write_hash_manifest(root / "config_hashes.csv", config_rows)
    registry = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "prepared_q4_forbidden",
        "local_training_job_count": len(jobs),
        "external_reference_count": EXPECTED_REFERENCES,
        "total_evaluation_cell_count": EXPECTED_CELLS,
        "q4_access": "forbidden",
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "jobs": jobs,
        "created_at_utc": _utc_now(),
    }
    for name in ("source_hashes.csv", "code_hashes.csv", "config_hashes.csv"):
        registry[f"{name.removesuffix('.csv')}_sha256"] = _sha256_file(root / name)
    registry["resolved_config_sha256"] = _payload_sha256(resolved)
    _write_json(root / "registry/jobs.json", registry)
    for job in jobs:
        _write_json(_job_status_path(root, job["job_id"]), _initial_status(job))
    _write_registry_exports(root)
    _write_experiment_status(root, "prepared_q4_forbidden")
    _validate_root(root)
    return root


def _registry(root: Path) -> dict[str, Any]:
    payload = _require_mapping(_read_json(root / "registry/jobs.json"), "registry")
    if payload.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    return payload


def _write_registry_exports(root: Path) -> None:
    registry = _registry(root)
    rows = []
    for job in registry["jobs"]:
        status = _read_json(_job_status_path(root, job["job_id"]))
        rows.append(
            {**job, "status": status.get("status"), "attempt": status.get("attempt")}
        )
    _write_csv(root / "task_registry.csv", rows, tuple(rows[0]))


def _write_experiment_status(root: Path, status: str, **details: Any) -> Path:
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": status,
        "local_completed": sum(
            _read_json(_job_status_path(root, job["job_id"])).get("status")
            == "completed"
            for job in _registry(root)["jobs"]
        ),
        "local_total": EXPECTED_LOCAL_JOBS,
        "external_reference_count": EXPECTED_REFERENCES,
        "q4_access": "forbidden",
        **details,
        "updated_at_utc": _utc_now(),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _write_json(root / "registry/experiment_status.json", payload)


def _validate_references(root: Path) -> list[dict[str, Any]]:
    manifest = _require_mapping(
        _read_json(root / "reference_manifest.json"), "reference manifest"
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    if manifest.get("payload_sha256") != _payload_sha256(unsigned):
        raise ValueError("Reference manifest payload SHA drift")
    rows = [dict(row) for row in manifest.get("references", ())]
    if len(rows) != EXPECTED_REFERENCES:
        raise ValueError("Reference count drift")
    capacity_selection = _require_mapping(
        manifest.get("capacity_selection"), "capacity selection snapshot"
    )
    capacity_selection_path = Path(capacity_selection["path"])
    capacity_selection_payload = _require_mapping(
        _read_json(capacity_selection_path), "capacity selection snapshot"
    )
    if (
        not capacity_selection_path.resolve().is_relative_to(root.resolve())
        or _sha256_file(capacity_selection_path) != capacity_selection["sha256"]
        or capacity_selection_payload.get("selection_sha256")
        != capacity_selection["payload_sha256"]
        or capacity_selection_payload.get("point_leader") != "legacy"
        or capacity_selection_payload.get("candidate_statistical_support")
        != "descriptive_only"
    ):
        raise ValueError("Frozen capacity-selection snapshot drift")
    for row in rows:
        if (
            int(row.get("source_gpu_id", -1))
            != dict(zip(SEEDS, (1, 0, 1)))[int(row["seed"])]
        ):
            raise ValueError("Frozen reference GPU lineage drift")
        status_path = Path(row["source_status_path"])
        if (
            not status_path.resolve().is_relative_to(root.resolve())
            or _sha256_file(status_path) != row["source_status_sha256"]
        ):
            raise ValueError("Frozen reference status drift")
        for artifact in row["artifacts"].values():
            path = Path(artifact["path"])
            if (
                not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size != int(artifact["size_bytes"])
                or _sha256_file(path) != artifact["sha256"]
            ):
                raise ValueError(f"Frozen reference artifact drift: {path}")
        config_path = Path(row["source_training_config_path"])
        if (
            not config_path.is_file()
            or not config_path.resolve().is_relative_to(root.resolve())
            or _sha256_file(config_path) != row["source_config_sha256"]
        ):
            raise ValueError(f"Frozen reference config drift: {config_path}")
        prediction = _require_mapping(row.get("prediction"), "reference prediction")
        prediction_path = Path(prediction["path"])
        source_manifest_path = Path(prediction["source_manifest_path"])
        if (
            not prediction_path.is_file()
            or not prediction_path.resolve().is_relative_to(root.resolve())
            or _sha256_file(prediction_path) != prediction["sha256"]
            or not source_manifest_path.is_file()
            or not source_manifest_path.resolve().is_relative_to(root.resolve())
            or _sha256_file(source_manifest_path)
            != prediction["source_manifest_sha256"]
            or int(prediction["row_count"]) != 148
            or int(prediction["mc_samples"]) != 16
        ):
            raise ValueError("Frozen reference prediction drift")
        source_manifest = _require_mapping(
            _read_json(source_manifest_path), "reference prediction source manifest"
        )
        source_manifest_unsigned = {
            key: value
            for key, value in source_manifest.items()
            if key != "manifest_sha256"
        }
        if (
            source_manifest.get("manifest_sha256")
            != prediction["source_manifest_payload_sha256"]
            or source_manifest.get("manifest_sha256")
            != _payload_sha256(source_manifest_unsigned)
            or source_manifest.get("panel_universe_sha256")
            != prediction["panel_universe_sha256"]
        ):
            raise ValueError("Frozen reference prediction manifest drift")
        expected_reference_sha = row["reference_spec_sha256"]
        if (
            _payload_sha256(
                {
                    key: value
                    for key, value in row.items()
                    if key != "reference_spec_sha256"
                }
            )
            != expected_reference_sha
        ):
            raise ValueError("Frozen reference spec SHA drift")
    return rows


def _validate_root(root: Path) -> dict[str, Any]:
    registry = _registry(root)
    for name in ("source_hashes.csv", "code_hashes.csv", "config_hashes.csv"):
        if _sha256_file(root / name) != registry[f"{name.removesuffix('.csv')}_sha256"]:
            raise ValueError(f"Manifest anchor drift: {name}")
        _verify_hash_manifest(root / name)
    resolved = _require_mapping(
        _load_yaml(root / "resolved_config.yaml", "resolved").get(ROOT_KEY), ROOT_KEY
    )
    if _payload_sha256(resolved) != registry["resolved_config_sha256"]:
        raise ValueError("Resolved config drift")
    if len(registry["jobs"]) != EXPECTED_LOCAL_JOBS:
        raise ValueError("Local registry count drift")
    for job in registry["jobs"]:
        if _job_spec_sha(job) != job["job_spec_sha256"]:
            raise ValueError(f"Job spec drift: {job['job_id']}")
        for path_key, sha_key in (
            ("training_config_path", "config_sha256"),
            ("dataset_path", "dataset_sha256"),
        ):
            path = Path(job[path_key])
            if not path.is_file() or _sha256_file(path) != job[sha_key]:
                raise ValueError(f"Job input drift: {job['job_id']} {path_key}")
    _validate_references(root)
    return resolved


def _find_job(root: Path, job_id: str) -> dict[str, Any]:
    rows = [dict(job) for job in _registry(root)["jobs"] if job["job_id"] == job_id]
    if len(rows) != 1:
        raise KeyError(job_id)
    return rows[0]


def _completed_local_job_valid(
    root: Path, job: Mapping[str, Any], status: Mapping[str, Any]
) -> bool:
    artifacts = [dict(row) for row in status.get("artifacts", ())]
    roles = tuple(row.get("artifact_role") for row in artifacts)
    if (
        status.get("status") != "completed"
        or status.get("config_sha256") != job["config_sha256"]
        or roles != factorial.DEVELOPMENT_ARTIFACT_ROLES
        or len(roles) != len(set(roles))
    ):
        return False
    for row in artifacts:
        path = Path(str(row.get("path", "")))
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(path) != row.get("sha256")
        ):
            return False
    return True


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
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if _completed_local_job_valid(root, job, previous):
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
        run_dir, artifacts = capacity._execute_job(job, dry_run=dry_run)
        _write_json(
            status_path,
            {
                **running,
                "status": "dry_run_passed" if dry_run else "completed",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
                "completed_at_utc": _utc_now(),
            },
        )
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
        "train-news-first-vol-legacy-architecture-seed-sweep",
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


def _launch(root: Path, *, dry_run: bool, resume: bool) -> Path:
    resolved = _validate_root(root)
    processes = []
    handles = []
    monitor = training._ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(resolved["runtime"].get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(
            resolved["runtime"].get("resource_sample_interval_seconds", 2)
        ),
        wave=1,
    )
    selected = []
    for job in _registry(root)["jobs"]:
        status = _read_json(_job_status_path(root, job["job_id"]))
        if not dry_run and _completed_local_job_valid(root, job, status):
            if resume:
                continue
            raise RuntimeError(f"Job already complete; use --resume: {job['job_id']}")
        selected.append(job)
    try:
        monitor.start()
        for job in selected:
            status = _read_json(_job_status_path(root, job["job_id"]))
            log_path = (
                root
                / "logs"
                / f"{job['job_id']}.attempt_{int(status.get('attempt', 0)) + 1:02d}.log"
            )
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(resolved["runtime"].get("cpu_threads_per_job", 1)))
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
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code:
                    training._terminate_processes(processes)
                    raise RuntimeError(f"Job {selected[index]['job_id']} exited {code}")
            if pending:
                time.sleep(0.5)
    finally:
        monitor.stop()
        for handle in handles:
            handle.close()
    failures = []
    for job in _registry(root)["jobs"]:
        status = _read_json(_job_status_path(root, job["job_id"]))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_local_job_valid(root, job, status)
        )
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status')}")
    if failures:
        raise RuntimeError(f"Incomplete local matrix: {failures}")
    _write_registry_exports(root)
    state = "dry_run_passed" if dry_run else "training_complete_q4_forbidden"
    registry = _registry(root)
    registry["status"] = state
    _write_json(root / "registry/jobs.json", registry)
    _write_experiment_status(root, state)
    return root


def analyze_experiment(output_dir: str | Path) -> Path:
    from scripts.rq3.news_first_vol_legacy_architecture_seed_analysis import (
        run_q3_analysis,
    )

    root = Path(output_dir).resolve()
    _validate_root(root)
    return run_q3_analysis(root)


def _forbidden_q4_artifacts(root: Path) -> list[Path]:
    forbidden_names = {
        "q4_common_05m.xlsx",
        "q4_window_manifest.json",
        "legacy_architecture_q4_pair_metrics.csv.gz",
        "legacy_architecture_q4_summary.json",
    }
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.lower() in forbidden_names
            or "analysis/predictions/q4" in path.as_posix().lower()
            or "data_windows/q4" in path.as_posix().lower()
        )
    ]


def _validate_selection_artifact(root: Path) -> dict[str, Any]:
    path = root / "analysis/legacy_architecture_q3_selection.json"
    selection = _require_mapping(_read_json(path), "Q3 selection")
    unsigned = {
        key: value for key, value in selection.items() if key != "payload_sha256"
    }
    if selection.get("payload_sha256") != _payload_sha256(unsigned):
        raise ValueError("Selection payload SHA drift")
    expected_outputs = {
        "legacy_architecture_q3_pair_metrics.csv.gz",
        "legacy_architecture_q3_cell_scores.csv",
        "legacy_architecture_q3_candidate_anchor_contrasts.csv",
        "legacy_architecture_q3_persistence_contrasts.csv",
        "legacy_architecture_q3_secondary_factorial_effects.csv",
        "legacy_architecture_training_diagnostics.csv",
    }
    output_hashes = _require_mapping(
        selection.get("output_sha256"), "selection output SHA"
    )
    if set(output_hashes) != expected_outputs:
        raise ValueError("Selection output family drift")
    for name, expected_sha in output_hashes.items():
        output = root / "analysis" / name
        if not output.is_file() or _sha256_file(output) != expected_sha:
            raise ValueError(f"Selection output SHA drift: {name}")
    return selection


def qa_experiment(output_dir: str | Path) -> Path:
    root = Path(output_dir).resolve()
    _validate_root(root)
    registry = _registry(root)
    failures = []
    completed = 0
    for job in registry["jobs"]:
        status = _read_json(_job_status_path(root, job["job_id"]))
        if _completed_local_job_valid(root, job, status):
            completed += 1
        else:
            failures.append(f"{job['job_id']}={status.get('status')}")
        if (
            status.get("q4_loader_created")
            or status.get("q4_predictions_generated")
            or int(status.get("q4_evaluator_rows", 0))
        ):
            failures.append(f"Q4 access recorded by {job['job_id']}")
    forbidden = _forbidden_q4_artifacts(root)
    if forbidden:
        failures.append(f"Q4 artifacts forbidden: {[str(path) for path in forbidden]}")
    required = (
        "resolved_config.yaml",
        "source_hashes.csv",
        "code_hashes.csv",
        "config_hashes.csv",
        "data_windows/split_manifest.csv",
        "reference_manifest.json",
        "task_registry.csv",
        "registry/experiment_status.json",
        "analysis/legacy_architecture_q3_pair_metrics.csv.gz",
        "analysis/legacy_architecture_q3_cell_scores.csv",
        "analysis/legacy_architecture_q3_candidate_anchor_contrasts.csv",
        "analysis/legacy_architecture_q3_persistence_contrasts.csv",
        "analysis/legacy_architecture_q3_secondary_factorial_effects.csv",
        "analysis/legacy_architecture_training_diagnostics.csv",
        "analysis/legacy_architecture_q3_selection.json",
        "report/legacy_architecture_q3_conclusion.md",
        "report/legacy_architecture_q3_conclusion.html",
        "resource_usage.csv",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        failures.append(f"Missing terminal artifacts: {missing}")
    status_paths = [
        _job_status_path(root, str(job["job_id"])) for job in registry["jobs"]
    ]
    if len(status_paths) != EXPECTED_LOCAL_JOBS or any(
        not path.is_file() for path in status_paths
    ):
        failures.append("Nine local job status files are required")
    local_prediction_dir = root / "analysis/predictions/q3"
    local_predictions = list(local_prediction_dir.glob("*.csv.gz"))
    local_manifests = list(local_prediction_dir.glob("*.manifest.json"))
    if (
        len(local_predictions) != EXPECTED_LOCAL_JOBS
        or len(local_manifests) != EXPECTED_LOCAL_JOBS
    ):
        failures.append("Local prediction matrix must contain 9 CSVs and 9 manifests")
    reference_rows = _validate_references(root)
    if (
        sum(bool(row.get("prediction")) for row in reference_rows)
        != EXPECTED_REFERENCES
    ):
        failures.append("Reference prediction matrix must contain three snapshots")
    selection_path = root / "analysis/legacy_architecture_q3_selection.json"
    if selection_path.is_file():
        try:
            selection = _validate_selection_artifact(root)
        except (ValueError, FileNotFoundError) as exc:
            failures.append(str(exc))
            selection = {}
        if (
            selection.get("q4_read") is not False
            or selection.get("q4_used_for_selection") is not False
            or selection.get("panel_counts")
            != {"rows": 148, "pairs": 135, "sessions": 33}
        ):
            failures.append("Selection Q3/Q4 contract drift")
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "passed" if not failures else "failed",
        "local_job_count": len(registry["jobs"]),
        "local_completed": completed,
        "external_reference_count": len(reference_rows),
        "evaluation_cell_count": len(registry["jobs"]) + len(reference_rows),
        "prediction_count": len(local_predictions) + len(reference_rows),
        "prediction_manifest_count": len(local_manifests) + len(reference_rows),
        "q4_access": "forbidden",
        "predicted_terminal_output_artifact_count": (
            _predicted_terminal_output_artifact_count(root)
        ),
        "failures": failures,
        "checked_at_utc": _utc_now(),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    path = _write_json(root / "qa.json", payload)
    if failures:
        raise RuntimeError(f"QA failed: {failures}")
    return path


def postprocess(output_dir: str | Path) -> Path:
    from scripts.rq3.news_first_vol_legacy_architecture_seed_report import render_report

    root = Path(output_dir).resolve()
    terminal_manifest = root / "terminal_output_sha256.csv"
    existing_report = root / "report/legacy_architecture_q3_conclusion.html"
    if terminal_manifest.is_file():
        _validate_terminal_output_manifest(root, repair_registry_anchor=True)
        return existing_report
    selection = analyze_experiment(root)
    report = render_report(root)
    registry = _registry(root)
    registry.update(
        status="completed_q3_exploratory_q4_forbidden",
        selection_path=str(selection.resolve()),
        selection_sha256=_sha256_file(selection),
        report_path=str(report.resolve()),
        report_sha256=_sha256_file(report),
    )
    _write_json(root / "registry/jobs.json", registry)
    _write_experiment_status(root, "completed_q3_exploratory_q4_forbidden")
    qa_path = qa_experiment(root)
    _write_final_registry_snapshot(root, qa_path=qa_path)
    files = _terminal_manifest_candidates(root)
    rows = [
        _manifest_row(f"terminal_output:{path.relative_to(root).as_posix()}", path)
        for path in files
    ]
    _write_csv(terminal_manifest, rows, tuple(rows[0]))
    _validate_terminal_output_manifest(root, repair_registry_anchor=True)
    return report


def _write_final_registry_snapshot(root: Path, *, qa_path: Path) -> Path:
    registry = _registry(root)
    frozen_registry = _strip_terminal_registry_fields(registry)
    experiment_status = root / "registry/experiment_status.json"
    task_registry = root / "task_registry.csv"
    payload = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "completed_q3_exploratory_q4_forbidden",
        "job_count": len(frozen_registry["jobs"]),
        "job_universe_sha256": _payload_sha256(
            sorted(str(job["job_spec_sha256"]) for job in frozen_registry["jobs"])
        ),
        "frozen_registry": frozen_registry,
        "frozen_registry_sha256": _payload_sha256(frozen_registry),
        "experiment_status_path": str(experiment_status.resolve()),
        "experiment_status_sha256": _sha256_file(experiment_status),
        "task_registry_path": str(task_registry.resolve()),
        "task_registry_sha256": _sha256_file(task_registry),
        "qa_path": str(qa_path.resolve()),
        "qa_sha256": _sha256_file(qa_path),
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    return _write_json(root / "registry/final_registry_snapshot.json", payload)


def _terminal_manifest_candidates(root: Path) -> list[Path]:
    excluded = {
        (root / "terminal_output_sha256.csv").resolve(),
        (root / "registry/jobs.json").resolve(),
        (root / "registry/pipeline.lock").resolve(),
    }
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    )


def _predicted_terminal_output_artifact_count(root: Path) -> int:
    """Return the exact terminal row count before QA/snapshot are materialized.

    This is interruption-safe: a resumed postprocess may already have either
    file, in which case it is already included by ``_terminal_manifest_candidates``.
    """

    qa_path = root / "qa.json"
    snapshot_path = root / "registry/final_registry_snapshot.json"
    return (
        len(_terminal_manifest_candidates(root))
        + int(not qa_path.is_file())
        + int(not snapshot_path.is_file())
    )


def _strip_terminal_registry_fields(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the mutable terminal-manifest anchor from a registry."""

    return {
        key: deepcopy(value)
        for key, value in registry.items()
        if not key.startswith("terminal_output_manifest")
        and key != "terminal_output_artifact_count"
    }


def _validate_terminal_output_manifest(
    root: Path, *, repair_registry_anchor: bool = False
) -> None:
    path = root / "terminal_output_sha256.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_paths = {
        str(candidate.resolve()) for candidate in _terminal_manifest_candidates(root)
    }
    observed_paths = {row["path"] for row in rows}
    roles = [row["artifact_role"] for row in rows]
    if (
        not rows
        or len(roles) != len(set(roles))
        or len(observed_paths) != len(rows)
        or observed_paths != expected_paths
    ):
        raise ValueError("Terminal output manifest coverage/uniqueness drift")
    for row in rows:
        target = Path(row["path"])
        expected_role = (
            f"terminal_output:{target.resolve().relative_to(root.resolve()).as_posix()}"
        )
        if (
            row["artifact_role"] != expected_role
            or target.stat().st_size != int(row["size_bytes"])
            or _sha256_file(target) != row["sha256"]
        ):
            raise ValueError(f"Terminal output hash drift: {target}")
    snapshot_path = root / "registry/final_registry_snapshot.json"
    snapshot = _require_mapping(_read_json(snapshot_path), "final registry snapshot")
    snapshot_unsigned = {
        key: value for key, value in snapshot.items() if key != "payload_sha256"
    }
    frozen_registry = _require_mapping(
        snapshot.get("frozen_registry"), "frozen terminal registry"
    )
    canonical_snapshot_paths = {
        "experiment_status_path": root / "registry/experiment_status.json",
        "task_registry_path": root / "task_registry.csv",
        "qa_path": root / "qa.json",
    }
    if any(
        Path(str(snapshot.get(key, ""))).resolve() != expected.resolve()
        for key, expected in canonical_snapshot_paths.items()
    ):
        raise ValueError("Final registry snapshot canonical path drift")
    if (
        snapshot.get("payload_sha256") != _payload_sha256(snapshot_unsigned)
        or snapshot.get("status") != "completed_q3_exploratory_q4_forbidden"
        or int(snapshot.get("job_count", -1)) != EXPECTED_LOCAL_JOBS
        or snapshot.get("frozen_registry_sha256") != _payload_sha256(frozen_registry)
        or frozen_registry.get("experiment_kind") != EXPERIMENT_KIND
        or frozen_registry.get("status") != "completed_q3_exploratory_q4_forbidden"
        or len(frozen_registry.get("jobs", ())) != EXPECTED_LOCAL_JOBS
        or snapshot.get("job_universe_sha256")
        != _payload_sha256(
            sorted(
                str(job["job_spec_sha256"]) for job in frozen_registry.get("jobs", ())
            )
        )
        or _sha256_file(canonical_snapshot_paths["experiment_status_path"])
        != snapshot["experiment_status_sha256"]
        or _sha256_file(canonical_snapshot_paths["task_registry_path"])
        != snapshot["task_registry_sha256"]
        or _sha256_file(canonical_snapshot_paths["qa_path"]) != snapshot["qa_sha256"]
    ):
        raise ValueError("Final registry snapshot drift")
    experiment_status = _require_mapping(
        _read_json(canonical_snapshot_paths["experiment_status_path"]),
        "terminal experiment status",
    )
    experiment_status_unsigned = {
        key: value
        for key, value in experiment_status.items()
        if key != "payload_sha256"
    }
    if (
        experiment_status.get("payload_sha256")
        != _payload_sha256(experiment_status_unsigned)
        or int(experiment_status.get("schema_version", -1)) != 1
        or experiment_status.get("experiment_kind") != EXPERIMENT_KIND
        or experiment_status.get("status") != "completed_q3_exploratory_q4_forbidden"
    ):
        raise ValueError("Terminal experiment status drift")
    qa = _require_mapping(
        _read_json(canonical_snapshot_paths["qa_path"]), "terminal QA"
    )
    qa_unsigned = {key: value for key, value in qa.items() if key != "payload_sha256"}
    expected_qa = {
        "schema_version": 1,
        "experiment_kind": EXPERIMENT_KIND,
        "status": "passed",
        "failures": [],
        "local_job_count": EXPECTED_LOCAL_JOBS,
        "local_completed": EXPECTED_LOCAL_JOBS,
        "external_reference_count": EXPECTED_REFERENCES,
        "evaluation_cell_count": EXPECTED_CELLS,
        "prediction_count": EXPECTED_CELLS,
        "prediction_manifest_count": EXPECTED_CELLS,
        "q4_access": "forbidden",
        "predicted_terminal_output_artifact_count": len(rows),
    }
    if qa.get("payload_sha256") != _payload_sha256(qa_unsigned) or any(
        qa.get(key) != value for key, value in expected_qa.items()
    ):
        raise ValueError("Terminal QA payload/count drift")
    registry = _registry(root)
    if _strip_terminal_registry_fields(registry) != frozen_registry:
        raise ValueError("Live registry drift from frozen terminal snapshot")
    expected_anchor = {
        "terminal_output_manifest_path": str(path.resolve()),
        "terminal_output_manifest_sha256": _sha256_file(path),
        "terminal_output_artifact_count": len(rows),
    }
    anchor_matches = all(
        registry.get(key) == value for key, value in expected_anchor.items()
    )
    if not anchor_matches and repair_registry_anchor:
        if registry.get("status") != "completed_q3_exploratory_q4_forbidden":
            raise ValueError("Terminal registry status cannot be repaired")
        registry.update(expected_anchor)
        _write_json(root / "registry/jobs.json", registry)
        registry = _registry(root)
    if registry.get("status") != "completed_q3_exploratory_q4_forbidden" or any(
        registry.get(key) != value for key, value in expected_anchor.items()
    ):
        raise ValueError("Terminal output registry anchor drift")


def status_experiment(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    registry = _registry(root)
    counts: dict[str, int] = {}
    for job in registry["jobs"]:
        state = str(_read_json(_job_status_path(root, job["job_id"])).get("status"))
        counts[state] = counts.get(state, 0) + 1
    return {
        "experiment_root": str(root),
        "status": registry.get("status"),
        "local_jobs": counts,
        "external_references": EXPECTED_REFERENCES,
        "q4_access": "forbidden",
    }


def run_pipeline(
    config_path: str | Path, output_dir: str | Path, *, resume: bool = False
) -> Path:
    root = Path(output_dir).resolve()
    if not root.exists():
        prepare_experiment(config_path, root)
    else:
        _validate_root(root)
    terminal_manifest = root / "terminal_output_sha256.csv"
    if terminal_manifest.is_file():
        _validate_terminal_output_manifest(root, repair_registry_anchor=True)
        report = root / "report/legacy_architecture_q3_conclusion.html"
        if not report.is_file():
            raise FileNotFoundError(f"Completed report is missing: {report}")
        return report
    lock = root / "registry/pipeline.lock"
    if lock.exists():
        previous = _read_json(lock)
        if training._pid_is_live(previous.get("pid")):
            raise RuntimeError(
                f"Pipeline already running under PID {previous.get('pid')}"
            )
        lock.unlink()
    _write_json(
        lock,
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at_utc": _utc_now(),
        },
    )
    try:
        _launch(root, dry_run=False, resume=resume)
        return postprocess(root)
    finally:
        if lock.exists():
            lock.unlink()


def run_news_first_vol_legacy_architecture_seed(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    action: str = "prepare",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
) -> Path | dict[str, Any]:
    if action == "prepare":
        return prepare_experiment(config_path, output_dir, reuse=reuse)
    if action == "dry-run":
        root = prepare_experiment(config_path, output_dir, reuse=True)
        return _launch(root, dry_run=True, resume=True)
    if action == "launch":
        root = prepare_experiment(config_path, output_dir, reuse=True)
        return _launch(root, dry_run=False, resume=resume)
    if action == "worker":
        if not job_id:
            raise ValueError("worker requires --job-id")
        return run_worker(output_dir, job_id, dry_run=worker_dry_run, resume=resume)
    if action == "analyze":
        return analyze_experiment(output_dir)
    if action == "postprocess":
        return postprocess(output_dir)
    if action == "qa":
        return qa_experiment(output_dir)
    if action == "status":
        return status_experiment(output_dir)
    if action == "run-pipeline":
        return run_pipeline(config_path, output_dir, resume=resume)
    raise ValueError(f"Unsupported legacy architecture action: {action}")


__all__ = [
    "ANCHOR",
    "CRITIC_MODES",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "EXPERIMENT_KIND",
    "GENERATOR_MODES",
    "REFERENCE_CELL",
    "SEEDS",
    "all_specs",
    "local_specs",
    "prepare_experiment",
    "resolve_config",
    "run_news_first_vol_legacy_architecture_seed",
]
