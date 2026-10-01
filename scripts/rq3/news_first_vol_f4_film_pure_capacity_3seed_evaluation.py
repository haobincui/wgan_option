"""Freeze and evaluate the F4 FiLM/Pure-CNN capacity experiment.

This module owns the one-way test-data gate.  ``freeze_checkpoints`` only
inspects training registries and artifacts.  No configured 2023Q4 path is
resolved or opened until the resulting 72-row G/D allowlist has been read
back, hash checked, and every selected checkpoint has been verified.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd
import yaml

import scripts._path_setup  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KIND = "news_first_vol_f4_film_pure_capacity_3seed_v1"
EVALUATION_KIND = "news_first_vol_f4_film_pure_capacity_3seed_evaluation_v1"
DEFAULT_CONFIG = "configs/rq3/news_first_vol_f4_film_pure_capacity_3seed.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/"
    "rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1"
)
ARCHITECTURES = ("film_cnn", "pure_cnn")
CAPACITIES = ("c08", "c12", "c16", "c24", "c32", "c48")
SEEDS = (42, 202, 404)
FOLD_ID = "f4_2023q4"
EXPECTED_JOBS = 36
EXPECTED_PAIRS = 143
EXPECTED_SESSIONS = 45
EXPECTED_PAIR_ROWS = EXPECTED_JOBS * EXPECTED_PAIRS
CHECKPOINT_ROLES = (
    "generator_best_learned",
    "discriminator_best_learned",
)
ALLOWLIST_COLUMNS = (
    "job_id",
    "architecture",
    "capacity_id",
    "seed",
    "checkpoint_role",
    "checkpoint_path",
    "size_bytes",
    "checkpoint_sha256",
    "job_spec_sha256",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_or_verify_bytes(path: Path, payload: bytes) -> Path:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen artifact differs from reproducible payload: {path}")
        return path
    return _atomic_bytes(path, payload)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    return _atomic_bytes(path, _json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g").encode("utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _write_or_verify_bytes(path, _csv_bytes(frame))


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _write_or_verify_bytes(
        path,
        gzip.compress(_csv_bytes(frame), compresslevel=9, mtime=0),
    )


def _registry_path(root: Path) -> Path:
    return root / "registry/jobs.json"


def _status_path(root: Path, job_id: str) -> Path:
    return root / f"registry/jobs/{job_id}.status.json"


def _state_path(root: Path) -> Path:
    return root / "registry/f4_evaluation_state.json"


def _allowlist_path(root: Path) -> Path:
    return root / "registry/f4_checkpoint_allowlist.csv"


def _checkpoint_manifest_path(root: Path) -> Path:
    return root / "analysis/f4_checkpoint_manifest.csv"


def _determinism_path(root: Path) -> Path:
    return root / "evaluation/inference_determinism_contract.json"


def _input_manifest_path(root: Path) -> Path:
    return root / "evaluation/f4_test_input_manifest.json"


def _prediction_manifest_path(root: Path) -> Path:
    return root / "evaluation/f4_prediction_manifest.csv"


def _pair_metrics_path(root: Path) -> Path:
    return root / "evaluation/f4_pair_metrics.csv.gz"


def _evaluation_config_path(root: Path) -> Path:
    return root / "evaluation/f4_resolved_evaluation_config.json"


def _signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return {**unsigned, "payload_sha256": _payload_sha256(unsigned)}


def _read_signed_json(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != _payload_sha256(unsigned):
        raise ValueError(f"Self-hash drift: {path}")
    return payload


def _write_state(root: Path, payload: Mapping[str, Any]) -> Path:
    return _write_json(_state_path(root), _signed_payload(payload))


def _load_training_registry(root: Path) -> dict[str, Any]:
    path = _registry_path(root)
    if not path.is_file():
        raise FileNotFoundError(path)
    registry = _read_json(path)
    jobs = registry.get("jobs")
    if (
        registry.get("experiment_kind") != EXPERIMENT_KIND
        or int(registry.get("expected_job_count", -1)) != EXPECTED_JOBS
        or not isinstance(jobs, list)
        or len(jobs) != EXPECTED_JOBS
    ):
        raise ValueError("Training registry is not the frozen 36-job F4 experiment")
    expected = {
        (architecture, capacity, seed)
        for architecture in ARCHITECTURES
        for capacity in CAPACITIES
        for seed in SEEDS
    }
    actual = {
        (str(job["architecture"]), str(job["capacity_id"]), int(job["seed"]))
        for job in jobs
    }
    if actual != expected or len({str(job["job_id"]) for job in jobs}) != EXPECTED_JOBS:
        raise ValueError("Training registry job universe drift")
    for job in jobs:
        config_path = Path(str(job["config_path"]))
        if (
            not config_path.is_file()
            or _sha256_file(config_path) != str(job["config_sha256"])
        ):
            raise ValueError(f"Training config drift: {config_path}")
    return registry


def _verified_status_artifacts(
    root: Path, job: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    status_path = _status_path(root, str(job["job_id"]))
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = _read_json(status_path)
    if (
        status.get("state") != "complete"
        or status.get("job_id") != job["job_id"]
        or status.get("job_spec_sha256") != job["job_spec_sha256"]
        or not bool(status.get("generator_parameters_updated"))
        or not bool(status.get("critic_parameters_updated"))
    ):
        raise ValueError(f"Training status is not complete and valid: {job['job_id']}")
    raw_artifacts = status.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError(f"Training status has no artifacts: {job['job_id']}")
    artifacts: dict[str, dict[str, Any]] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Malformed training artifact: {job['job_id']}")
        role = str(raw.get("artifact_role", ""))
        path = Path(str(raw.get("path", ""))).resolve()
        if not role or role in artifacts:
            raise ValueError(f"Duplicate/empty artifact role: {job['job_id']}/{role}")
        if (
            not path.is_file()
            or path.stat().st_size != int(raw.get("size_bytes", -1))
            or _sha256_file(path) != str(raw.get("sha256", ""))
        ):
            raise ValueError(f"Training artifact drift: {path}")
        artifacts[role] = dict(raw)
    if not set(CHECKPOINT_ROLES).issubset(artifacts):
        raise ValueError(f"Selected checkpoint pair is absent: {job['job_id']}")
    return artifacts


def _inference_contract() -> dict[str, Any]:
    return _signed_payload(
        {
            "schema_version": 1,
            "kind": "f4_capacity_shared_mc64_inference_v1",
            "seed_scope": "training_seed_shared_across_architectures_and_capacities_v1",
            "method": "stable_noise_for_keys_v1",
            "sample_key": "sample_id",
            "prediction_mc_samples": 64,
            "noise_dim": 32,
            "python_numpy_torch_seeded": True,
            "torch_cuda_manual_seed_all": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        }
    )


def _validate_checkpoint_allowlist(root: Path) -> dict[str, dict[str, str]]:
    state_path = _state_path(root)
    if not state_path.is_file():
        raise RuntimeError("Q4 access denied: checkpoint allowlist is not frozen")
    state = _read_signed_json(state_path)
    if not bool(state.get("checkpoints_frozen")):
        raise RuntimeError("Q4 access denied: checkpoint freeze is incomplete")
    registry_path = _registry_path(root)
    if (
        state.get("training_registry_path") != str(registry_path.resolve())
        or state.get("training_registry_sha256") != _sha256_file(registry_path)
    ):
        raise ValueError("Training registry changed after checkpoint freeze")
    allowlist_path = Path(str(state.get("checkpoint_allowlist_path", "")))
    manifest_path = Path(str(state.get("checkpoint_manifest_path", "")))
    determinism_path = Path(str(state.get("inference_determinism_contract_path", "")))
    if (
        allowlist_path != _allowlist_path(root).resolve()
        or not allowlist_path.is_file()
        or _sha256_file(allowlist_path) != state.get("checkpoint_allowlist_sha256")
        or manifest_path != _checkpoint_manifest_path(root).resolve()
        or not manifest_path.is_file()
        or _sha256_file(manifest_path) != state.get("checkpoint_manifest_sha256")
        or determinism_path != _determinism_path(root).resolve()
        or not determinism_path.is_file()
        or _sha256_file(determinism_path)
        != state.get("inference_determinism_contract_sha256")
        or _read_signed_json(determinism_path) != _inference_contract()
    ):
        raise ValueError("Frozen checkpoint lineage drift")
    registry = _load_training_registry(root)
    expected_jobs = {str(job["job_id"]): job for job in registry["jobs"]}
    frame = pd.read_csv(allowlist_path, dtype=str, keep_default_na=False)
    if (
        tuple(frame.columns) != ALLOWLIST_COLUMNS
        or len(frame) != EXPECTED_JOBS * len(CHECKPOINT_ROLES)
        or frame.duplicated(["job_id", "checkpoint_role"]).any()
        or set(frame["job_id"]) != set(expected_jobs)
    ):
        raise ValueError("Checkpoint allowlist must contain exactly 36 G/D pairs")
    expected_role_pairs = {
        (job_id, role) for job_id in expected_jobs for role in CHECKPOINT_ROLES
    }
    if set(zip(frame["job_id"], frame["checkpoint_role"], strict=True)) != expected_role_pairs:
        raise ValueError("Checkpoint allowlist role universe drift")
    for row in frame.itertuples(index=False):
        job = expected_jobs[str(row.job_id)]
        path = Path(str(row.checkpoint_path)).resolve()
        if (
            str(row.architecture) != str(job["architecture"])
            or str(row.capacity_id) != str(job["capacity_id"])
            or int(row.seed) != int(job["seed"])
            or str(row.job_spec_sha256) != str(job["job_spec_sha256"])
            or not path.is_file()
            or path.stat().st_size != int(row.size_bytes)
            or _sha256_file(path) != str(row.checkpoint_sha256)
        ):
            raise ValueError(f"Allowlisted checkpoint drift: {path}")
    generators = frame.loc[frame["checkpoint_role"].eq("generator_best_learned")]
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if len(manifest) != EXPECTED_JOBS or set(manifest["job_id"]) != set(expected_jobs):
        raise ValueError("Generator checkpoint manifest must contain 36 rows")
    expected_manifest = generators.drop(columns="checkpoint_role").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(manifest, expected_manifest, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Generator manifest differs from frozen allowlist") from exc
    return {
        str(row.job_id): {
            "path": str(Path(row.checkpoint_path).resolve()),
            "sha256": str(row.checkpoint_sha256),
            "role": "generator_best_learned",
        }
        for row in generators.itertuples(index=False)
    }


def freeze_checkpoints(formal_root: str | Path) -> Path:
    """Freeze the 36 selected Generator checkpoints plus their 36 Critics.

    This function has no configuration argument by design, so its execution
    path cannot discover or open the configured Q4 panel or overlays.
    """

    root = _resolve(formal_root)
    if _state_path(root).is_file():
        _validate_checkpoint_allowlist(root)
        return _allowlist_path(root)
    registry = _load_training_registry(root)
    rows: list[dict[str, Any]] = []
    for job in sorted(registry["jobs"], key=lambda row: str(row["job_id"])):
        artifacts = _verified_status_artifacts(root, job)
        for role in CHECKPOINT_ROLES:
            artifact = artifacts[role]
            rows.append(
                {
                    "job_id": str(job["job_id"]),
                    "architecture": str(job["architecture"]),
                    "capacity_id": str(job["capacity_id"]),
                    "seed": int(job["seed"]),
                    "checkpoint_role": role,
                    "checkpoint_path": str(Path(str(artifact["path"])).resolve()),
                    "size_bytes": int(artifact["size_bytes"]),
                    "checkpoint_sha256": str(artifact["sha256"]),
                    "job_spec_sha256": str(job["job_spec_sha256"]),
                }
            )
    allowlist = pd.DataFrame(rows, columns=ALLOWLIST_COLUMNS)
    if len(allowlist) != 72 or allowlist["checkpoint_path"].nunique() != 72:
        raise ValueError("Expected 72 distinct selected G/D checkpoint files")
    allowlist_path = _write_csv(_allowlist_path(root), allowlist)
    generator_manifest = allowlist.loc[
        allowlist["checkpoint_role"].eq("generator_best_learned")
    ].drop(columns="checkpoint_role")
    manifest_path = _write_csv(_checkpoint_manifest_path(root), generator_manifest)
    determinism_path = _write_json(_determinism_path(root), _inference_contract())
    _write_state(
        root,
        {
            "schema_version": 1,
            "kind": EVALUATION_KIND,
            "checkpoints_frozen": True,
            "checkpoints_frozen_at_utc": _utc_now(),
            "training_registry_path": str(_registry_path(root).resolve()),
            "training_registry_sha256": _sha256_file(_registry_path(root)),
            "completed_training_jobs": EXPECTED_JOBS,
            "checkpoint_allowlist_path": str(allowlist_path.resolve()),
            "checkpoint_allowlist_sha256": _sha256_file(allowlist_path),
            "checkpoint_allowlist_rows": 72,
            "checkpoint_manifest_path": str(manifest_path.resolve()),
            "checkpoint_manifest_sha256": _sha256_file(manifest_path),
            "generator_checkpoint_count": EXPECTED_JOBS,
            "inference_determinism_contract_path": str(determinism_path.resolve()),
            "inference_determinism_contract_sha256": _sha256_file(determinism_path),
            "q4_data_opened": False,
            "predictions_frozen": False,
        },
    )
    _validate_checkpoint_allowlist(root)
    return allowlist_path


def _load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        payload = dict(config)
    else:
        path = _resolve(config)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"Config root must be a mapping: {path}")
        payload = dict(raw)
    if payload.get("experiment", {}).get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Evaluation config experiment kind drift")
    matrix = payload.get("matrix", {})
    if (
        tuple(matrix.get("architectures", ())) != ARCHITECTURES
        or tuple(matrix.get("capacities", ())) != CAPACITIES
        or tuple(map(int, matrix.get("seeds", ()))) != SEEDS
        or str(matrix.get("fold")) != FOLD_ID
    ):
        raise ValueError("Evaluation matrix drift")
    if int(payload.get("training", {}).get("prediction_mc_samples", -1)) != 64:
        raise ValueError("F4 evaluation requires MC64")
    return payload


def _frozen_input_destinations(root: Path) -> dict[str, Path]:
    base = root / "evaluation/frozen_inputs"
    return {
        "panel": base / "f4_2023q4.csv.gz",
        "film_overlay": base / "film_cnn.json",
        "pure_overlay": base / "pure_cnn.json",
    }


def _verify_panel_and_overlays(
    config: Mapping[str, Any], destinations: Mapping[str, Path]
) -> pd.DataFrame:
    from wgan_option.utils.news_first_experiment_core import (
        load_pair_text_overlay_manifest,
    )

    frozen = config["data"]["frozen_f4_inputs"]
    panel = pd.read_csv(destinations["panel"], low_memory=False)
    required = {
        "pair_id",
        "session_id",
        "sample_id",
        "effective_origin_utc",
        "current_surface_flat",
        "target_surface_flat",
    }
    if (
        len(panel) != EXPECTED_PAIRS
        or not required.issubset(panel.columns)
        or panel["pair_id"].astype(str).nunique() != EXPECTED_PAIRS
        or panel["session_id"].astype(str).nunique() != EXPECTED_SESSIONS
        or panel["sample_id"].astype(str).duplicated().any()
    ):
        raise ValueError("Frozen F4 panel must contain 143 pairs / 45 sessions")
    origins = pd.to_datetime(panel["effective_origin_utc"], errors="coerce", utc=True)
    if (
        origins.isna().any()
        or (origins < pd.Timestamp("2023-10-01T00:00:00Z")).any()
        or (origins >= pd.Timestamp("2024-01-01T00:00:00Z")).any()
    ):
        raise ValueError("Frozen panel is not exclusively 2023Q4")
    pair_ids = set(panel["pair_id"].astype(str))
    session_by_pair = dict(
        zip(
            panel["pair_id"].astype(str),
            panel["session_id"].astype(str),
            strict=True,
        )
    )
    declarations = (
        (
            "film_overlay",
            "film_overlay_sha256",
            "film_overlay_profile_sha256",
            "lp_mean_l2",
        ),
        (
            "pure_overlay",
            "pure_overlay_sha256",
            "pure_overlay_profile_sha256",
            "current_only",
        ),
    )
    for name, hash_key, profile_key, mode in declarations:
        manifest = load_pair_text_overlay_manifest(
            destinations[name],
            str(frozen[hash_key]),
            str(frozen[profile_key]),
            expected_mode=mode,
        )
        if set(manifest.embeddings) != pair_ids or any(
            manifest.sessions[pair_id] != session_by_pair[pair_id]
            for pair_id in pair_ids
        ):
            raise ValueError(f"Frozen F4 {name} pair/session universe drift")
    return panel


def _materialize_q4_inputs(
    config: Mapping[str, Any], root: Path
) -> tuple[pd.DataFrame, dict[str, Path]]:
    # The gate is deliberately the first operation.  In particular, no path
    # under data.frozen_f4_inputs is resolved or stat'ed before it succeeds.
    _validate_checkpoint_allowlist(root)
    state = _read_signed_json(_state_path(root))
    frozen = config["data"]["frozen_f4_inputs"]
    sources = {
        "panel": (_resolve(frozen["panel_path"]), str(frozen["panel_sha256"])),
        "film_overlay": (
            _resolve(frozen["film_overlay_path"]),
            str(frozen["film_overlay_sha256"]),
        ),
        "pure_overlay": (
            _resolve(frozen["pure_overlay_path"]),
            str(frozen["pure_overlay_sha256"]),
        ),
    }
    destinations = _frozen_input_destinations(root)
    source_rows: list[dict[str, Any]] = []
    for role, (source, expected_sha) in sources.items():
        if not source.is_file() or _sha256_file(source) != expected_sha:
            raise ValueError(f"Configured frozen Q4 input drift: {source}")
        payload = source.read_bytes()
        _write_or_verify_bytes(destinations[role], payload)
        if _sha256_file(destinations[role]) != expected_sha:
            raise ValueError(f"Materialized frozen Q4 input drift: {destinations[role]}")
        source_rows.append(
            {
                "role": role,
                "source_path": str(source),
                "destination_path": str(destinations[role].resolve()),
                "sha256": expected_sha,
                "size_bytes": len(payload),
            }
        )
    panel = _verify_panel_and_overlays(config, destinations)
    manifest = _signed_payload(
        {
            "schema_version": 1,
            "kind": "f4_capacity_frozen_test_inputs_v1",
            "fold_id": FOLD_ID,
            "pair_count": EXPECTED_PAIRS,
            "session_count": EXPECTED_SESSIONS,
            "sources": source_rows,
        }
    )
    manifest_path = _input_manifest_path(root)
    if manifest_path.is_file():
        if _read_signed_json(manifest_path) != manifest:
            raise ValueError("Frozen F4 input manifest drift")
    else:
        _write_json(manifest_path, manifest)
    if bool(state.get("q4_data_opened")):
        if (
            state.get("test_input_manifest_path") != str(manifest_path.resolve())
            or state.get("test_input_manifest_sha256") != _sha256_file(manifest_path)
        ):
            raise ValueError("Frozen F4 input state drift")
    else:
        unsigned = {key: value for key, value in state.items() if key != "payload_sha256"}
        unsigned.update(
            q4_data_opened=True,
            q4_data_opened_at_utc=_utc_now(),
            test_input_manifest_path=str(manifest_path.resolve()),
            test_input_manifest_sha256=_sha256_file(manifest_path),
        )
        _write_state(root, unsigned)
    return panel, destinations


def _panel_with_overlay(
    config: Mapping[str, Any],
    panel: pd.DataFrame,
    destinations: Mapping[str, Path],
    architecture: str,
) -> pd.DataFrame:
    from wgan_option.utils.news_first_experiment_core import (
        load_pair_text_overlay_manifest,
    )

    frozen = config["data"]["frozen_f4_inputs"]
    is_film = architecture == "film_cnn"
    path = destinations["film_overlay" if is_film else "pure_overlay"]
    manifest = load_pair_text_overlay_manifest(
        path,
        str(frozen["film_overlay_sha256" if is_film else "pure_overlay_sha256"]),
        str(
            frozen[
                "film_overlay_profile_sha256"
                if is_film
                else "pure_overlay_profile_sha256"
            ]
        ),
        expected_mode="lp_mean_l2" if is_film else "current_only",
    )
    result = panel.copy()
    pair_ids = result["pair_id"].astype(str)
    result["lp_embedding"] = [
        json.dumps(
            manifest.embeddings[pair_id].astype(float).tolist(),
            separators=(",", ":"),
        )
        for pair_id in pair_ids
    ]
    result["hd_embedding"] = ""
    result["sample_id"] = pair_ids.map(lambda value: f"pair::{value}")
    result["sample_weight"] = 1.0
    return result


def _noise_bank_profile(seed: int, sample_ids: Sequence[str]) -> str:
    return _payload_sha256(
        {
            "schema_version": 1,
            "method": "stable_noise_for_keys_v1",
            "fold_id": FOLD_ID,
            "seed": int(seed),
            "sample_ids": sorted(map(str, sample_ids)),
            "draws": 64,
            "noise_dim": 32,
        }
    )


def _configure_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from wgan_option.utils.reproducibility import seed_everything

    seed_everything(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _prediction_paths(root: Path, job: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    base = (
        root
        / "evaluation/predictions"
        / str(job["architecture"])
        / str(job["capacity_id"])
        / f"seed_{int(job['seed']):03d}"
    )
    return (
        base / "predictions.csv.gz",
        base / "pair_metrics.csv",
        base / "manifest.json",
    )


def _evaluate_cell(
    config: Mapping[str, Any],
    root: Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    panel: pd.DataFrame,
    destinations: Mapping[str, Path],
) -> tuple[dict[str, Any], pd.DataFrame]:
    import torch
    from scripts.rq3.news_first_vol_comparison_analysis import (
        RunSpec,
        TrainedRunEvaluator,
        _enforce_formal_run_coverage,
        _prediction_export_frame,
        aggregate_pair_metrics,
        compute_sample_metrics,
    )

    _configure_determinism(int(job["seed"]))
    architecture = str(job["architecture"])
    evaluation_panel = _panel_with_overlay(
        config, panel, destinations, architecture
    )
    noise_sha = _noise_bank_profile(
        int(job["seed"]), evaluation_panel["sample_id"].astype(str).tolist()
    )
    status = _read_json(_status_path(root, str(job["job_id"])))
    run = RunSpec(
        run_id=str(job["job_id"]),
        run_dir=Path(str(status["run_dir"])),
        model="wgan",
        tolerance_minutes=5,
        seed=int(job["seed"]),
        checkpoint_path=Path(checkpoint["path"]),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={
            "fold": FOLD_ID,
            "architecture": architecture,
            "capacity_id": str(job["capacity_id"]),
        },
    )
    device = (
        os.environ.get("F4_EVALUATION_DEVICE", f"cuda:{int(job['gpu_id'])}")
        if torch.cuda.is_available()
        else "cpu"
    )
    evaluator = TrainedRunEvaluator(
        mc_samples=64,
        sample_batch_size=32,
        draw_batch_size=64,
        device=device,
    )
    panel_name = "f4_2023q4_test_05m"
    predictions = evaluator(run, panel_name, evaluation_panel)
    samples, exclusions, metric_exclusions = compute_sample_metrics(
        run,
        panel_name,
        evaluation_panel,
        predictions,
        evaluate_embedded_atm_skew=False,
    )
    export = _prediction_export_frame(
        run, panel_name, evaluation_panel, predictions, samples
    )
    _enforce_formal_run_coverage(
        run, panel_name, evaluation_panel, samples, exclusions, export
    )
    prediction_path, evidence_path, manifest_path = _prediction_paths(root, job)
    overlay_key = "film_overlay" if architecture == "film_cnn" else "pure_overlay"
    export = export.assign(
        job_id=str(job["job_id"]),
        fold_id=FOLD_ID,
        architecture=architecture,
        capacity_id=str(job["capacity_id"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
        panel_sha256=_sha256_file(destinations["panel"]),
        test_overlay_sha256=_sha256_file(destinations[overlay_key]),
        noise_bank_profile_sha256=noise_sha,
    )
    _write_gzip_csv(prediction_path, export)
    prediction_sha = _sha256_file(prediction_path)
    standard = aggregate_pair_metrics(samples)
    standard = standard.loc[
        standard["stratum_type"].eq("overall")
        & standard["stratum_value"].eq("all")
    ].copy()
    if len(standard) != EXPECTED_PAIRS:
        raise ValueError(f"F4 pair-metric count drift: {job['job_id']}")
    evidence = pd.DataFrame(
        {
            "fold_id": FOLD_ID,
            "fold": FOLD_ID,
            "architecture": architecture,
            "capacity_id": str(job["capacity_id"]),
            "is_current_capacity": str(job["capacity_id"]) == "c32",
            "generator_parameters": int(job["generator_parameters"]),
            "critic_parameters": int(job["critic_parameters"]),
            "total_wgan_parameters": int(job["wgan_parameters"]),
            "seed": int(job["seed"]),
            "pair_id": standard["pair_id"].astype(str),
            "session_id": standard["session_id"].astype(str),
            "target_mae": standard["model_mae"].astype(float),
            "persistence_mae": standard["persistence_mae"].astype(float),
            "prediction_mc_samples": 64,
            "checkpoint_sha256": str(checkpoint["sha256"]),
            "prediction_sha256": prediction_sha,
            "noise_bank_profile_sha256": noise_sha,
            "panel_sha256": _sha256_file(destinations["panel"]),
            "test_overlay_sha256": _sha256_file(destinations[overlay_key]),
            "job_spec_sha256": str(job["job_spec_sha256"]),
        }
    ).sort_values("pair_id", kind="stable")
    if (
        not evidence["target_mae"].map(math.isfinite).all()
        or not evidence["persistence_mae"].map(math.isfinite).all()
    ):
        raise ValueError(f"Non-finite F4 metric: {job['job_id']}")
    _write_csv(evidence_path, evidence)
    manifest = _signed_payload(
        {
            "schema_version": 1,
            "kind": "f4_capacity_prediction_cell_v1",
            "job_id": str(job["job_id"]),
            "job_spec_sha256": str(job["job_spec_sha256"]),
            "architecture": architecture,
            "capacity_id": str(job["capacity_id"]),
            "seed": int(job["seed"]),
            "fold_id": FOLD_ID,
            "checkpoint_path": str(Path(checkpoint["path"]).resolve()),
            "checkpoint_sha256": str(checkpoint["sha256"]),
            "panel_path": str(destinations["panel"].resolve()),
            "panel_sha256": _sha256_file(destinations["panel"]),
            "test_overlay_path": str(destinations[overlay_key].resolve()),
            "test_overlay_sha256": _sha256_file(destinations[overlay_key]),
            "noise_bank_profile_sha256": noise_sha,
            "prediction_mc_samples": 64,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": prediction_sha,
            "pair_metrics_path": str(evidence_path.resolve()),
            "pair_metrics_sha256": _sha256_file(evidence_path),
            "pair_metric_rows": EXPECTED_PAIRS,
            "optional_metric_exclusion_count": int(len(metric_exclusions)),
        }
    )
    _write_json(manifest_path, manifest)
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest, evidence


def evaluate_f4_worker(
    config: Mapping[str, Any] | str | Path,
    formal_root: str | Path,
    job_id: str,
    *,
    resume: bool,
) -> Path:
    """Evaluate one isolated cell in a process bound to one visible GPU."""

    root = _resolve(formal_root)
    checkpoints = _validate_checkpoint_allowlist(root)
    resolved = _load_config(config)
    # The supervisor has already frozen these destinations.  Calling the
    # idempotent materializer here revalidates hashes after the allowlist gate.
    panel, destinations = _materialize_q4_inputs(resolved, root)
    registry = _load_training_registry(root)
    matches = [job for job in registry["jobs"] if str(job["job_id"]) == str(job_id)]
    if len(matches) != 1:
        raise ValueError(f"Unknown evaluation job: {job_id}")
    job = matches[0]
    checkpoint = checkpoints[str(job["job_id"])]
    paths = _prediction_paths(root, job)
    present = [path.is_file() for path in paths]
    if all(present):
        if not resume:
            raise FileExistsError(f"Prediction exists; pass --resume: {job_id}")
        _validate_prediction_cell(root, job, checkpoint, panel)
    else:
        if any(present) and not resume:
            raise ValueError(f"Partial prediction requires --resume: {job_id}")
        _evaluate_cell(
            resolved,
            root,
            job,
            checkpoint,
            panel,
            destinations,
        )
        _validate_prediction_cell(root, job, checkpoint, panel)
    return paths[2]


def _launch_evaluation_workers(
    config: Mapping[str, Any],
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    resume: bool,
) -> None:
    """Launch every pending cell at once, balanced 18/18 over physical GPUs."""

    config_path = _evaluation_config_path(root)
    _write_or_verify_bytes(config_path, _json_bytes(dict(config)))
    log_dir = root / "evaluation/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    active: dict[str, tuple[subprocess.Popen[bytes], io.BufferedWriter, Path]] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        log_path = log_dir / f"{job_id}.log"
        handle = log_path.open("wb")
        env = os.environ.copy()
        env.update(
            PYTHONPATH="src:.",
            CUDA_VISIBLE_DEVICES=str(int(job["gpu_id"])),
            F4_EVALUATION_DEVICE="cuda:0",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            "-m",
            "scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed_evaluation",
            "evaluate-worker",
            "--config",
            str(config_path),
            "--output-dir",
            str(root),
            "--job-id",
            job_id,
        ]
        if resume:
            command.append("--resume")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        active[job_id] = (process, handle, log_path)
    failures: list[str] = []
    for job_id, (process, handle, log_path) in active.items():
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append(f"{job_id} (exit={returncode}, log={log_path})")
    if failures:
        raise RuntimeError("F4 evaluation worker failures: " + "; ".join(failures))


def _validate_frozen_prediction_bundle(
    config: Mapping[str, Any],
    root: Path,
    registry: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, str]],
    panel: pd.DataFrame,
) -> Path:
    state = _read_signed_json(_state_path(root))
    manifest_path = Path(str(state.get("prediction_manifest_path", "")))
    pair_path = Path(str(state.get("pair_metrics_path", "")))
    if (
        not bool(state.get("predictions_frozen"))
        or int(state.get("prediction_cell_count", -1)) != EXPECTED_JOBS
        or int(state.get("pair_metric_rows", -1)) != EXPECTED_PAIR_ROWS
        or manifest_path != _prediction_manifest_path(root).resolve()
        or not manifest_path.is_file()
        or _sha256_file(manifest_path) != state.get("prediction_manifest_sha256")
        or pair_path != _pair_metrics_path(root).resolve()
        or not pair_path.is_file()
        or _sha256_file(pair_path) != state.get("pair_metrics_sha256")
    ):
        raise ValueError("Frozen F4 prediction bundle lineage drift")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    jobs = [dict(job) for job in registry["jobs"]]
    expected_jobs = {str(job["job_id"]) for job in jobs}
    if (
        len(manifest) != EXPECTED_JOBS
        or manifest["job_id"].duplicated().any()
        or set(manifest["job_id"]) != expected_jobs
    ):
        raise ValueError("Frozen F4 prediction manifest must contain 36 jobs")
    manifest_by_job = manifest.set_index("job_id", drop=False)
    cell_frames: list[pd.DataFrame] = []
    for job in jobs:
        job_id = str(job["job_id"])
        cell_manifest, evidence = _validate_prediction_cell(
            root, job, checkpoints[job_id], panel
        )
        global_row = manifest_by_job.loc[job_id]
        local_manifest_path = _prediction_paths(root, job)[2]
        if (
            global_row["checkpoint_sha256"] != cell_manifest["checkpoint_sha256"]
            or global_row["noise_bank_profile_sha256"]
            != cell_manifest["noise_bank_profile_sha256"]
            or global_row["prediction_sha256"] != cell_manifest["prediction_sha256"]
            or global_row["pair_metrics_sha256"]
            != cell_manifest["pair_metrics_sha256"]
            or global_row["manifest_path"] != str(local_manifest_path.resolve())
            or global_row["manifest_sha256"] != _sha256_file(local_manifest_path)
        ):
            raise ValueError(f"Global/cell prediction manifest drift: {job_id}")
        cell_frames.append(evidence)
    cell_combined = pd.concat(cell_frames, ignore_index=True).sort_values(
        ["architecture", "capacity_id", "seed", "pair_id"], kind="stable"
    ).reset_index(drop=True)
    global_evidence = pd.read_csv(pair_path).sort_values(
        ["architecture", "capacity_id", "seed", "pair_id"], kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            global_evidence,
            cell_combined,
            check_dtype=False,
            # Each layer is independently CSV-hash-bound.  A second
            # parse/write/parse can move a binary float by one final decimal
            # place, so compare numeric columns at well below metric precision.
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as exc:
        raise ValueError("Global/cell frozen F4 pair evidence mismatch") from exc
    from scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed_analysis import (
        summarize_pair_metrics,
    )

    summarize_pair_metrics(
        global_evidence,
        parameter_counts=config["expected_parameter_counts"],
    )
    return pair_path


def _validate_prediction_cell(
    root: Path,
    job: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    panel: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    prediction_path, evidence_path, manifest_path = _prediction_paths(root, job)
    if not all(path.is_file() for path in (prediction_path, evidence_path, manifest_path)):
        raise ValueError(f"Partial prediction artifacts: {job['job_id']}")
    manifest = _read_signed_json(manifest_path)
    expected_noise = _noise_bank_profile(
        int(job["seed"]), panel["sample_id"].astype(str).tolist()
    )
    if (
        manifest.get("job_id") != job["job_id"]
        or manifest.get("job_spec_sha256") != job["job_spec_sha256"]
        or manifest.get("checkpoint_sha256") != checkpoint["sha256"]
        or manifest.get("prediction_sha256") != _sha256_file(prediction_path)
        or manifest.get("pair_metrics_sha256") != _sha256_file(evidence_path)
        or manifest.get("noise_bank_profile_sha256") != expected_noise
        or int(manifest.get("prediction_mc_samples", -1)) != 64
        or int(manifest.get("pair_metric_rows", -1)) != EXPECTED_PAIRS
    ):
        raise ValueError(f"Prediction-cell manifest drift: {job['job_id']}")
    evidence = pd.read_csv(evidence_path)
    if (
        len(evidence) != EXPECTED_PAIRS
        or evidence["pair_id"].astype(str).nunique() != EXPECTED_PAIRS
        or evidence["session_id"].astype(str).nunique() != EXPECTED_SESSIONS
        or set(evidence["pair_id"].astype(str)) != set(panel["pair_id"].astype(str))
        or set(evidence["noise_bank_profile_sha256"].astype(str)) != {expected_noise}
        or set(evidence["checkpoint_sha256"].astype(str)) != {checkpoint["sha256"]}
    ):
        raise ValueError(f"Prediction-cell evidence drift: {job['job_id']}")
    return manifest, evidence


def evaluate_f4(
    config: Mapping[str, Any] | str | Path,
    formal_root: str | Path,
    resume: bool = False,
) -> Path:
    """Run frozen-panel MC64 inference for all 36 architecture/capacity cells."""

    root = _resolve(formal_root)
    freeze_checkpoints(root)
    # Validate the full allowlist immediately before the only Q4-opening call.
    checkpoints = _validate_checkpoint_allowlist(root)
    resolved = _load_config(config)
    panel, destinations = _materialize_q4_inputs(resolved, root)
    registry = _load_training_registry(root)
    state = _read_signed_json(_state_path(root))
    if bool(state.get("predictions_frozen")):
        return _validate_frozen_prediction_bundle(
            resolved, root, registry, checkpoints, panel
        )
    jobs = sorted(
        registry["jobs"],
        key=lambda row: (
            int(row["seed"]),
            str(row["architecture"]),
            CAPACITIES.index(str(row["capacity_id"])),
        ),
    )
    pending: list[Mapping[str, Any]] = []
    for job in jobs:
        checkpoint = checkpoints[str(job["job_id"])]
        paths = _prediction_paths(root, job)
        present = [path.is_file() for path in paths]
        if all(present):
            if not resume:
                raise FileExistsError(
                    f"Prediction exists; pass resume=True: {job['job_id']}"
                )
            _validate_prediction_cell(root, job, checkpoint, panel)
        else:
            if any(present) and not resume:
                raise ValueError(
                    f"Partial prediction requires resume=True: {job['job_id']}"
                )
            pending.append(job)
    if pending:
        _launch_evaluation_workers(resolved, root, pending, resume=resume)
    prediction_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    for job in jobs:
        checkpoint = checkpoints[str(job["job_id"])]
        manifest, evidence = _validate_prediction_cell(
            root, job, checkpoint, panel
        )
        prediction_rows.append(
            {
                "job_id": str(job["job_id"]),
                "architecture": str(job["architecture"]),
                "capacity_id": str(job["capacity_id"]),
                "seed": int(job["seed"]),
                "checkpoint_sha256": str(checkpoint["sha256"]),
                "noise_bank_profile_sha256": str(
                    manifest["noise_bank_profile_sha256"]
                ),
                "prediction_path": str(manifest["prediction_path"]),
                "prediction_sha256": str(manifest["prediction_sha256"]),
                "pair_metrics_path": str(manifest["pair_metrics_path"]),
                "pair_metrics_sha256": str(manifest["pair_metrics_sha256"]),
                "manifest_path": str(_prediction_paths(root, job)[2].resolve()),
                "manifest_sha256": _sha256_file(_prediction_paths(root, job)[2]),
            }
        )
        evidence_frames.append(evidence)
    combined = pd.concat(evidence_frames, ignore_index=True)
    if len(combined) != EXPECTED_PAIR_ROWS:
        raise ValueError(f"Expected {EXPECTED_PAIR_ROWS} F4 pair rows")
    noise_counts = combined.groupby("seed")["noise_bank_profile_sha256"].nunique()
    if not noise_counts.eq(1).all():
        raise ValueError("Architectures/capacities did not share one MC64 bank per seed")
    from scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed_analysis import (
        summarize_pair_metrics,
    )

    summarize_pair_metrics(
        combined,
        parameter_counts=resolved["expected_parameter_counts"],
    )
    manifest_path = _write_csv(
        _prediction_manifest_path(root), pd.DataFrame(prediction_rows)
    )
    pair_path = _write_gzip_csv(
        _pair_metrics_path(root),
        combined.sort_values(
            ["architecture", "capacity_id", "seed", "pair_id"], kind="stable"
        ).reset_index(drop=True),
    )
    state = _read_signed_json(_state_path(root))
    unsigned = {key: value for key, value in state.items() if key != "payload_sha256"}
    unsigned.update(
        predictions_frozen=True,
        predictions_frozen_at_utc=_utc_now(),
        prediction_cell_count=EXPECTED_JOBS,
        prediction_manifest_path=str(manifest_path.resolve()),
        prediction_manifest_sha256=_sha256_file(manifest_path),
        pair_metrics_path=str(pair_path.resolve()),
        pair_metrics_sha256=_sha256_file(pair_path),
        pair_metric_rows=EXPECTED_PAIR_ROWS,
    )
    _write_state(root, unsigned)
    return pair_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("freeze-checkpoints", "evaluate-f4", "evaluate-worker"),
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.action == "freeze-checkpoints":
        result = freeze_checkpoints(args.output_dir)
    elif args.action == "evaluate-worker":
        if not args.job_id:
            raise ValueError("evaluate-worker requires --job-id")
        result = evaluate_f4_worker(
            args.config,
            args.output_dir,
            args.job_id,
            resume=bool(args.resume),
        )
    else:
        result = evaluate_f4(args.config, args.output_dir, resume=bool(args.resume))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_f4", "evaluate_f4_worker", "freeze_checkpoints"]
