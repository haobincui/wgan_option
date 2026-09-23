"""Core implementation for the single-seed FiLM text-signal probe.

This module intentionally lives behind the thin probe orchestrator.  It owns
only the experiment-local data preparation, frozen-backbone adapter training,
validation analysis, and report generation.  In particular, it never
materializes a test loader and it never writes to the source ten-seed root.
"""

from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import yaml

from wgan_option.config import Config
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.models.text_signal_adapter import (
    SMALL_NORMAL_STARTUP_MODE,
    TEXT_SIGNAL_ADAPTER_STATE_SCHEMA,
    ZERO_STARTUP_MODE,
    FrozenFiLMTextAdapterGenerator,
    convert_film_unet_to_text_signal_adapter,
)
from wgan_option.utils.news_first_dataloaders import (
    NewsFirstSplitSpec,
    create_news_first_vol_surface_dataloaders,
)
from wgan_option.utils.news_first_experiment_core import pair_universe_sha256
from wgan_option.utils.text_signal_probe_data import (
    baseline_unique_article_mean_l2,
    build_wrong_text_donor_plan,
    deduplicate_lp_articles,
    fit_improved_lp_transform,
    load_wrong_text_donor_manifest,
    save_improved_lp_transform,
    transform_improved_lp,
    transform_then_derange,
    write_wrong_text_donor_manifest,
)
from wgan_option.utils.weighted_training import (
    apply_surface_mask,
    masked_mean_per_sample,
    stable_key_to_int64,
    stable_noise_for_keys,
    training_weighted_mean,
    validated_sample_weights,
    validated_surface_mask,
)


PROBE_INPUT_SCHEMA = "film_unet_text_signal_probe_input_v1"
PROBE_RESULT_SCHEMA = "film_unet_text_signal_probe_result_v1"
PROBE_SELECTION_SCHEMA = "film_unet_text_signal_probe_selection_v1"
_PAIR_INDEX_SCHEMA = "film_unet_text_signal_pair_index_v1"
_OVERLAY_SCHEMA = "film_unet_text_signal_overlay_v1"
_INDEPENDENT_SHUFFLE_SEED = 2026083002
_WRONG_DONOR_SEED = 2026083001
_NOISE_SEED = 420042
_EPSILON = 1.0e-12


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: object) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            encoded = _canonical_json_bytes(payload)
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(destination)


def _atomic_npy(path: str | Path, values: np.ndarray) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(destination)


def _artifact(
    path: str | Path, *, role: str, root: Path | None = None
) -> dict[str, Any]:
    artifact_path = Path(path).resolve()
    # The orchestrator verifies artifact rows without an experiment-root
    # argument, so persist absolute paths. ``root`` remains accepted to keep
    # call sites self-documenting about artifact ownership.
    del root
    return {
        "role": role,
        "path": str(artifact_path),
        "size_bytes": int(artifact_path.stat().st_size),
        "sha256": _sha256_file(artifact_path),
    }


def _resolve_artifact_path(record: Mapping[str, Any], root: Path) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else root / path


def _verify_artifacts(records: Sequence[Mapping[str, Any]], root: Path) -> None:
    for record in records:
        path = _resolve_artifact_path(record, root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {record.get('role', 'artifact')}: {path}")
        if int(path.stat().st_size) != int(record["size_bytes"]):
            raise ValueError(f"Artifact size drift: {path}")
        if _sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Artifact SHA drift: {path}")


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def _as_config_mapping(
    resolved_config: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(resolved_config, Mapping):
        return dict(resolved_config)
    return _load_yaml_mapping(resolved_config)


def _absolute(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Invalid model checkpoint: {path}")
    if not isinstance(payload.get("config"), dict):
        raise ValueError(f"Checkpoint lacks a resolved config: {path}")
    return payload


def _source_overlay_path(config: Mapping[str, Any]) -> Path:
    source_root = _absolute(config["baseline"]["source_experiment_root"])
    return (
        source_root
        / "inputs"
        / "pair_text_overlays"
        / "tolerance_05m"
        / "f2_2023q2"
        / "lp_matched.json"
    )


def _pair_universe_path(config: Mapping[str, Any]) -> Path:
    return (
        _absolute(config["baseline"]["source_experiment_root"])
        / "inputs"
        / "pair_universes.csv"
    )


def _read_pair_universe(config: Mapping[str, Any]) -> pd.DataFrame:
    path = _pair_universe_path(config)
    frame = pd.read_csv(path)
    required = {"tolerance_minutes", "fold", "partition", "pair_id", "session_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair universe misses columns: {missing}")
    selected = frame.loc[
        (frame["tolerance_minutes"].astype(int) == 5)
        & (frame["fold"].astype(str) == "f2_2023q2")
        & (frame["partition"].astype(str).isin(["train", "validation"]))
    ].copy()
    selected["pair_id"] = selected["pair_id"].astype(str)
    selected["session_id"] = selected["session_id"].astype(str)
    if selected["pair_id"].duplicated().any():
        raise ValueError("Probe pair universe contains duplicate pair IDs")
    expected = config["data"]["expected_counts"]
    for partition in ("train", "validation"):
        rows = selected.loc[selected["partition"] == partition]
        expected_pairs = int(expected[f"{partition}_pairs"])
        expected_sessions = int(expected[f"{partition}_sessions"])
        if (
            len(rows) != expected_pairs
            or rows["session_id"].nunique() != expected_sessions
        ):
            raise ValueError(
                f"Unexpected {partition} pair/session counts: "
                f"{len(rows)}/{rows['session_id'].nunique()} != "
                f"{expected_pairs}/{expected_sessions}"
            )
    return selected.sort_values(["partition", "pair_id"], kind="stable").reset_index(
        drop=True
    )


def _read_probe_workbook(
    config: Mapping[str, Any], universe: pd.DataFrame
) -> pd.DataFrame:
    workbook = _absolute(config["data"]["workbook_path"])
    frame = pd.read_excel(workbook, sheet_name=str(config["data"]["sheet_name"]))
    pair_ids = set(universe["pair_id"])
    frame["pair_id"] = frame["pair_id"].astype(str)
    frame = frame.loc[frame["pair_id"].isin(pair_ids)].copy()
    if set(frame["pair_id"]) != pair_ids:
        raise ValueError("Workbook does not cover the frozen train/validation universe")
    partition_by_pair = dict(
        zip(universe["pair_id"], universe["partition"], strict=True)
    )
    expected_session = dict(
        zip(universe["pair_id"], universe["session_id"], strict=True)
    )
    frame["partition"] = frame["pair_id"].map(partition_by_pair)
    if any(
        str(row.session_id) != expected_session[str(row.pair_id)]
        for row in frame[["pair_id", "session_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        raise ValueError("Workbook session lineage differs from the frozen universe")
    return frame


def _source_overlay_vectors(path: Path) -> tuple[dict[str, np.ndarray], str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Source LP overlay lacks records")
    vectors = {
        str(row["pair_id"]): np.asarray(row["embedding"], dtype=np.float32)
        for row in records
    }
    return vectors, _sha256_file(path), str(payload.get("profile_sha256", ""))


def _write_overlay(
    output_dir: Path,
    *,
    name: str,
    vectors: Mapping[str, np.ndarray],
    universe: pd.DataFrame,
    method: str,
    lineage: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_ids = sorted(vectors)
    if pair_ids != sorted(universe["pair_id"].astype(str)):
        raise ValueError(f"{name} overlay universe mismatch")
    matrix = np.stack(
        [np.asarray(vectors[pair_id], dtype=np.float32) for pair_id in pair_ids]
    )
    if matrix.shape != (len(pair_ids), 1024) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} overlay has an invalid matrix")
    array_path = output_dir / f"{name}_embeddings.npy"
    array_sha = _atomic_npy(array_path, matrix)
    partition_by_pair = dict(
        zip(universe["pair_id"], universe["partition"], strict=True)
    )
    session_by_pair = dict(
        zip(universe["pair_id"], universe["session_id"], strict=True)
    )
    profile = {
        "schema_version": _OVERLAY_SCHEMA,
        "name": name,
        "method": method,
        "embedding_dim": 1024,
        "pair_count": len(pair_ids),
        "pair_universe_sha256": pair_universe_sha256(pair_ids),
        "array": {
            "path": array_path.name,
            "shape": [len(pair_ids), 1024],
            "dtype": "float32",
            "sha256": array_sha,
        },
        "records": [
            {
                "row": index,
                "pair_id": pair_id,
                "partition": partition_by_pair[pair_id],
                "session_id": session_by_pair[pair_id],
            }
            for index, pair_id in enumerate(pair_ids)
        ],
        "lineage": dict(lineage),
    }
    profile_sha = hashlib.sha256(_canonical_json_bytes(profile)).hexdigest()
    manifest_path = output_dir / f"{name}_overlay.json"
    _atomic_json(manifest_path, {**profile, "profile_sha256": profile_sha})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, [
        _artifact(
            manifest_path, role=f"{name}_overlay_manifest", root=output_dir.parent
        ),
        _artifact(array_path, role=f"{name}_overlay_array", root=output_dir.parent),
    ]


def _write_shuffle_manifest(
    path: Path,
    mapping: Mapping[str, str],
    universe: pd.DataFrame,
) -> dict[str, Any]:
    partition_by_pair = dict(
        zip(universe["pair_id"], universe["partition"], strict=True)
    )
    payload = {
        "schema_version": "film_unet_text_signal_independent_shuffle_v1",
        "master_seed": _INDEPENDENT_SHUFFLE_SEED,
        "namespace": "confirmation_independent_shuffle",
        "pair_universe_sha256": pair_universe_sha256(sorted(mapping)),
        "records": [
            {
                "receiver_pair_id": receiver,
                "donor_pair_id": donor,
                "partition": partition_by_pair[receiver],
            }
            for receiver, donor in sorted(mapping.items())
        ],
    }
    _atomic_json(path, payload)
    return payload


def prepare_probe_inputs(
    root: str | Path, resolved_config: Mapping[str, Any] | str | Path
) -> Path:
    """Freeze the train/validation-only text inputs and their complete lineage."""

    output_root = Path(root).resolve()
    config = _as_config_mapping(resolved_config)
    inputs = output_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    baseline_cfg = config["baseline"]
    source_artifacts: list[dict[str, Any]] = []
    for model_role in ("generator_checkpoint", "critic_checkpoint"):
        record = baseline_cfg[model_role]
        path = _absolute(record["path"])
        if int(path.stat().st_size) != int(record["size_bytes"]):
            raise ValueError(f"Frozen {model_role} size mismatch")
        if _sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Frozen {model_role} SHA mismatch")
        source_artifacts.append(_artifact(path, role=model_role, root=output_root))
    for record in config["data"].get("source_files", []):
        path = _absolute(record["path"])
        if _sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"Frozen data source SHA mismatch: {path}")
        source_artifacts.append(
            _artifact(path, role=str(record["role"]), root=output_root)
        )

    universe = _read_pair_universe(config)
    workbook = _read_probe_workbook(config, universe)
    pair_ids = sorted(universe["pair_id"])
    train_pair_ids = sorted(universe.loc[universe["partition"] == "train", "pair_id"])
    pair_articles = deduplicate_lp_articles(
        workbook,
        required_pair_ids=pair_ids,
        embedding_dim=1024,
    )
    baseline_vectors = baseline_unique_article_mean_l2(pair_articles)
    source_overlay = _source_overlay_path(config)
    source_vectors, source_overlay_sha, source_overlay_profile_sha = (
        _source_overlay_vectors(source_overlay)
    )
    if set(source_vectors) != set(baseline_vectors):
        raise ValueError(
            "Historical LP overlay universe differs from the probe universe"
        )
    maximum_difference = max(
        float(np.max(np.abs(source_vectors[pair_id] - baseline_vectors[pair_id])))
        for pair_id in pair_ids
    )
    if maximum_difference > 1.0e-7:
        raise ValueError(
            "Rebuilt historical LP aggregation differs from the frozen overlay: "
            f"max_abs={maximum_difference:.9g}"
        )
    # Use the rebuilt vectors after proving equivalence.  This binds article
    # lineage into the new probe rather than merely copying an opaque artifact.
    transform = fit_improved_lp_transform(
        pair_articles, train_pair_ids, half_life_minutes=5.0
    )
    transform_artifacts = save_improved_lp_transform(transform, inputs / "lp_transform")
    improved_vectors = transform_improved_lp(pair_articles, transform)

    baseline_manifest, baseline_artifacts = _write_overlay(
        inputs,
        name="baseline_lp",
        vectors=baseline_vectors,
        universe=universe,
        method="unique_article_lp_mean_l2_v1",
        lineage={
            "source_overlay_path": str(source_overlay),
            "source_overlay_sha256": source_overlay_sha,
            "source_overlay_profile_sha256": source_overlay_profile_sha,
            "rebuilt_source_max_abs_difference": maximum_difference,
        },
    )
    improved_manifest, improved_artifacts = _write_overlay(
        inputs,
        name="improved_lp",
        vectors=improved_vectors,
        universe=universe,
        method="article_l2_recency_center_pc1_l2_v1",
        lineage={
            "transform_manifest_path": transform_artifacts["manifest_path"],
            "transform_manifest_sha256": transform_artifacts["manifest_sha256"],
            "transform_sha256": transform.transform_sha256,
            "fit_partition": "train",
        },
    )

    donor_plan = build_wrong_text_donor_plan(
        workbook,
        master_seed=_WRONG_DONOR_SEED,
        namespace="probe_wrong_text_anchor",
    )
    wrong_path = inputs / "wrong_text_donors.json"
    wrong_info = write_wrong_text_donor_manifest(donor_plan, wrong_path)
    partition_by_pair = dict(
        zip(universe["pair_id"], universe["partition"], strict=True)
    )
    _, independent_mapping = transform_then_derange(
        baseline_vectors,
        partition_by_pair,
        master_seed=_INDEPENDENT_SHUFFLE_SEED,
        namespace="confirmation_independent_shuffle",
    )
    shuffle_path = inputs / "independent_shuffle.json"
    shuffle_manifest = _write_shuffle_manifest(
        shuffle_path, independent_mapping, universe
    )

    pair_index_path = inputs / "pair_index.json"
    pair_index = {
        "schema_version": _PAIR_INDEX_SCHEMA,
        "fold_id": "f2_2023q2",
        "tolerance_minutes": 5,
        "partitions": ["train", "validation"],
        "pair_universe_sha256": pair_universe_sha256(pair_ids),
        "records": universe[["pair_id", "session_id", "partition"]]
        .sort_values(["partition", "pair_id"], kind="stable")
        .to_dict(orient="records"),
    }
    _atomic_json(pair_index_path, pair_index)

    artifacts = [
        *source_artifacts,
        _artifact(
            _pair_universe_path(config), role="source_pair_universe", root=output_root
        ),
        _artifact(source_overlay, role="source_baseline_lp_overlay", root=output_root),
        *baseline_artifacts,
        *improved_artifacts,
        _artifact(
            transform_artifacts["manifest_path"],
            role="improved_lp_transform",
            root=output_root,
        ),
        _artifact(
            transform_artifacts["train_mean_path"],
            role="improved_lp_train_mean",
            root=output_root,
        ),
        _artifact(
            transform_artifacts["top_pc1_path"],
            role="improved_lp_top_pc1",
            root=output_root,
        ),
        _artifact(wrong_path, role="wrong_text_donor_manifest", root=output_root),
        _artifact(shuffle_path, role="independent_shuffle_manifest", root=output_root),
        _artifact(pair_index_path, role="pair_index", root=output_root),
    ]
    manifest = {
        "schema_version": PROBE_INPUT_SCHEMA,
        "interpretation": "single_seed_mechanism_validation",
        "fold_id": "f2_2023q2",
        "tolerance_minutes": 5,
        "partitions": ["train", "validation"],
        "test_loader_materialized": False,
        "test_prediction_rows": 0,
        "q3_q4_input_rows": 0,
        "counts": {
            "train_pairs": 526,
            "train_sessions": 133,
            "validation_pairs": 110,
            "validation_sessions": 34,
        },
        "pair_universe_sha256": pair_index["pair_universe_sha256"],
        "baseline_overlay": {
            "manifest_path": str(
                (inputs / "baseline_lp_overlay.json").relative_to(output_root)
            ),
            "manifest_sha256": _sha256_file(inputs / "baseline_lp_overlay.json"),
            "profile_sha256": baseline_manifest["profile_sha256"],
        },
        "improved_overlay": {
            "manifest_path": str(
                (inputs / "improved_lp_overlay.json").relative_to(output_root)
            ),
            "manifest_sha256": _sha256_file(inputs / "improved_lp_overlay.json"),
            "profile_sha256": improved_manifest["profile_sha256"],
        },
        "wrong_text": {
            "manifest_path": str(wrong_path.relative_to(output_root)),
            "manifest_sha256": wrong_info["manifest_sha256"],
            "profile_sha256": wrong_info["profile_sha256"],
        },
        "independent_shuffle": {
            "manifest_path": str(shuffle_path.relative_to(output_root)),
            "manifest_sha256": _sha256_file(shuffle_path),
            "master_seed": shuffle_manifest["master_seed"],
        },
        "pair_index_path": str(pair_index_path.relative_to(output_root)),
        "source_baseline": {
            "generator_sha256": baseline_cfg["generator_checkpoint"]["sha256"],
            "critic_sha256": baseline_cfg["critic_checkpoint"]["sha256"],
        },
        "artifacts": artifacts,
    }
    manifest_path = inputs / "probe_input_manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _load_overlay(
    root: Path, input_manifest: Mapping[str, Any], which: str
) -> dict[str, np.ndarray]:
    record = input_manifest[f"{which}_overlay"]
    manifest_path = root / str(record["manifest_path"])
    if _sha256_file(manifest_path) != str(record["manifest_sha256"]):
        raise ValueError(f"{which} overlay manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile_sha256") != str(record["profile_sha256"]):
        raise ValueError(f"{which} overlay profile SHA mismatch")
    array_path = manifest_path.parent / str(manifest["array"]["path"])
    if _sha256_file(array_path) != str(manifest["array"]["sha256"]):
        raise ValueError(f"{which} overlay array SHA mismatch")
    matrix = np.load(array_path, allow_pickle=False)
    if tuple(matrix.shape) != tuple(manifest["array"]["shape"]):
        raise ValueError(f"{which} overlay array shape mismatch")
    return {
        str(row["pair_id"]): np.asarray(matrix[int(row["row"])], dtype=np.float32)
        for row in manifest["records"]
    }


def _load_mapping(path: Path, *, receiver_key: str, donor_key: str) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row[receiver_key]): str(row[donor_key]) for row in payload["records"]}


def _probe_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    root: Path,
) -> Config:
    source = Config(**checkpoint["config"])
    overlay_path = _source_overlay_path(config)
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    return replace(
        source,
        data_path=str(_absolute(config["data"]["workbook_path"])),
        news_first_common_eval_data_path=str(
            _absolute(config["data"]["workbook_path"])
        ),
        news_first_train_end_utc="2023-01-01T00:00:00Z",
        news_first_validation_end_utc="2023-04-01T00:00:00Z",
        news_first_materialize_validation_loader=True,
        news_first_materialize_test_loader=False,
        news_first_data_window_start_utc_inclusive="",
        news_first_data_window_end_utc_exclusive="",
        news_first_pair_text_overlay_mode="lp_mean_l2",
        news_first_pair_text_manifest_path=str(overlay_path),
        news_first_pair_text_manifest_sha256=_sha256_file(overlay_path),
        news_first_pair_text_profile_sha256=str(overlay["profile_sha256"]),
        news_first_refit_mode="none",
        news_first_refit_recipe_path="",
        news_first_refit_recipe_sha256="",
        news_first_full_training_state_mode="none",
        news_first_full_training_state_contract_path="",
        news_first_full_training_state_contract_sha256="",
        num_workers=0,
        seed=42,
        num_epochs=1,
        use_early_stopping=False,
        use_reduce_lr_on_plateau=False,
        lr_scheduler_type="none",
        output_root=str(root),
        models_path=str(root / "checkpoints"),
        outputs_path=str(root / "checkpoints"),
        samples_path=str(root / "samples"),
        metrics_path=str(root / "metrics"),
        normalization_stats_path=str(root / "metrics" / "normalization_stats.json"),
    )


def _load_bundle_and_model(
    root: Path,
    config: Mapping[str, Any],
    *,
    startup_mode: str,
    spatial_rank: int,
    text_out_dim: int,
    global_residual_gate_initial: float | None = None,
    global_residual_gate_max: float = 0.1,
) -> tuple[
    Any,
    WGAN_GP,
    FrozenFiLMTextAdapterGenerator,
    torch.nn.Module,
    Config,
]:
    generator_path = _absolute(config["baseline"]["generator_checkpoint"]["path"])
    critic_path = _absolute(config["baseline"]["critic_checkpoint"]["path"])
    generator_checkpoint = _checkpoint_payload(generator_path)
    critic_checkpoint = _checkpoint_payload(critic_path)
    model_config = _probe_config_from_checkpoint(generator_checkpoint, config, root)
    bundle = create_news_first_vol_surface_dataloaders(
        model_config,
        model_config.news_first_common_eval_data_path,
        split_spec=NewsFirstSplitSpec(
            train_end_utc="2023-01-01T00:00:00Z",
            validation_end_utc="2023-04-01T00:00:00Z",
        ),
    )
    if bundle.test_loader is not None or bundle.test_samples or bundle.test_items:
        raise RuntimeError(
            "Probe contract violation: a test loader/item was materialized"
        )
    if (bundle.train_samples, bundle.val_samples) != (526, 110):
        raise ValueError("Probe loader pair counts drifted")
    model = WGAN_GP(
        model_config,
        bundle.strike_grid,
        bundle.maturity_grid_days,
        bundle.embedding_dim,
    )
    model.G.load_state_dict(generator_checkpoint["state_dict"], strict=True)
    model.D.load_state_dict(critic_checkpoint["state_dict"], strict=True)
    model.G.eval()
    model.D.eval()
    for parameter in model.D.parameters():
        parameter.requires_grad_(False)
    source_generator = model.G
    adapter = convert_film_unet_to_text_signal_adapter(
        source_generator,
        startup_mode=startup_mode,
        spatial_rank=spatial_rank,
        adapter_text_out_dim=text_out_dim,
        baseline_sha256=str(config["baseline"]["generator_checkpoint"]["sha256"]),
        global_residual_gate_initial=global_residual_gate_initial,
        global_residual_gate_max=global_residual_gate_max,
    ).to(model.device)
    model.G = adapter
    return bundle, model, adapter, source_generator, model_config


def _validate_bundle_pair_lineage(
    bundle: Any,
    experiment_root: Path,
    input_manifest: Mapping[str, Any],
) -> None:
    index_path = experiment_root / str(input_manifest["pair_index_path"])
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _PAIR_INDEX_SCHEMA:
        raise ValueError("Probe pair-index schema mismatch")
    records = list(payload.get("records") or [])
    for partition, items in (
        ("train", bundle.train_items),
        ("validation", bundle.val_items),
    ):
        expected = [
            (str(row["pair_id"]), str(row["session_id"]))
            for row in records
            if str(row["partition"]) == partition
        ]
        observed = [(str(item.pair_id), str(item.session_id)) for item in items]
        if sorted(observed) != sorted(expected):
            raise ValueError(
                f"{partition} loader pair/session lineage differs from pair_index"
            )
    observed_all = sorted(
        str(item.pair_id) for item in (*bundle.train_items, *bundle.val_items)
    )
    if pair_universe_sha256(observed_all) != input_manifest["pair_universe_sha256"]:
        raise ValueError("Loader pair-universe SHA differs from frozen probe inputs")


class _ProbeDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Any],
        matched: Mapping[str, np.ndarray],
        wrong_mapping: Mapping[str, str],
        *,
        assignment_mapping: Mapping[str, str] | None = None,
    ) -> None:
        self.items = list(items)
        self.pair_ids = [str(item.pair_id) for item in self.items]
        if len(self.pair_ids) != len(set(self.pair_ids)):
            raise ValueError("Probe dataset must contain exactly one row per pair")
        assignment = assignment_mapping or {
            pair_id: pair_id for pair_id in self.pair_ids
        }
        self.matched = [
            np.asarray(matched[assignment[pair_id]], dtype=np.float32)
            for pair_id in self.pair_ids
        ]
        self.wrong = [
            np.asarray(matched[wrong_mapping[pair_id]], dtype=np.float32)
            for pair_id in self.pair_ids
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        item = self.items[index]
        support = item.support_mask
        current_support = item.current_support_mask
        if support is None or current_support is None:
            raise ValueError("Probe requires raw joint and current-only support masks")
        return (
            torch.as_tensor(item.current_surface, dtype=torch.float32),
            torch.as_tensor(self.matched[index], dtype=torch.float32),
            torch.as_tensor(self.wrong[index], dtype=torch.float32),
            torch.as_tensor(item.target_surface, dtype=torch.float32),
            torch.tensor(float(item.sample_weight), dtype=torch.float32),
            torch.tensor(
                stable_key_to_int64(item.stable_sample_key), dtype=torch.int64
            ),
            torch.as_tensor(support, dtype=torch.float32),
            torch.as_tensor(current_support, dtype=torch.float32),
            torch.tensor(index, dtype=torch.int64),
        )


def _loader(dataset: _ProbeDataset, *, epoch: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42_000_000 + int(epoch))
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _to_device(
    batch: Sequence[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device=device, non_blocking=True) for value in batch)


def _base_loss(
    model: WGAN_GP,
    fake: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    text: torch.Tensor,
    sample_weight: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    support = validated_surface_mask(support, reference_surface=fake)
    batch_size = int(fake.shape[0])
    adv_per_sample = (
        -model.D(
            apply_surface_mask(fake, support),
            apply_surface_mask(current, support),
            text,
        )
        .reshape(batch_size, -1)
        .mean(dim=1)
    )
    recon_per_sample = masked_mean_per_sample(torch.abs(fake - target), support)
    adv = training_weighted_mean(adv_per_sample, sample_weight)
    recon = training_weighted_mean(recon_per_sample, sample_weight)
    calendar = model.calendar_arbitrage_penalty(
        fake, sample_weight, training_weights=True, support_mask=support
    )
    butterfly = model.butterfly_arbitrage_penalty(
        fake, sample_weight, training_weights=True, support_mask=support
    )
    smooth = model.smoothness_penalty(
        fake, sample_weight, training_weights=True, support_mask=support
    )
    total = (
        adv
        + model.lambda_recon * recon
        + model.lambda_calendar * calendar
        + model.lambda_butterfly * butterfly
        + model.lambda_smooth * smooth
    )
    return total, {
        "adv": adv,
        "recon": recon,
        "calendar": calendar,
        "butterfly": butterfly,
        "smooth": smooth,
    }


def _alignment_loss(
    fake_matched: torch.Tensor,
    fake_zero: torch.Tensor,
    fake_wrong: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive = masked_mean_per_sample(torch.abs(fake_matched - target), support)
    zero = masked_mean_per_sample(torch.abs(fake_zero - target), support)
    wrong = masked_mean_per_sample(torch.abs(fake_wrong - target), support)
    scale = max(float(tau), 1.0e-8)
    rank_zero = scale * F.softplus((positive - zero.detach()) / scale)
    rank_wrong = scale * F.softplus((positive - wrong.detach()) / scale)
    wrong_anchor = masked_mean_per_sample(torch.abs(fake_wrong - fake_zero), support)
    total_per_sample = rank_zero + rank_wrong + 0.25 * wrong_anchor
    return training_weighted_mean(total_per_sample, sample_weight), {
        "rank_zero": training_weighted_mean(rank_zero, sample_weight),
        "rank_wrong": training_weighted_mean(rank_wrong, sample_weight),
        "wrong_anchor": training_weighted_mean(wrong_anchor, sample_weight),
    }


def _gradient_norm(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    squared = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().pow(2).sum()
    return float(torch.sqrt(squared).cpu())


def _initial_tau(
    dataset: _ProbeDataset,
    adapter: FrozenFiLMTextAdapterGenerator,
    device: torch.device,
) -> float:
    errors: list[float] = []
    adapter.eval()
    with torch.no_grad():
        for raw_batch in _loader(dataset, epoch=0, shuffle=False):
            (
                current,
                _matched,
                _wrong,
                target,
                _weight,
                keys,
                support,
                current_support,
                _,
            ) = _to_device(raw_batch, device)
            noise = stable_noise_for_keys(
                keys,
                noise_dim=32,
                base_seed=_NOISE_SEED,
                draw_index=0,
                device=device,
                dtype=current.dtype,
            )
            zero = torch.zeros((len(current), 1024), device=device, dtype=current.dtype)
            fake = adapter(
                current, zero, noise=noise, current_support_mask=current_support
            )
            values = masked_mean_per_sample(torch.abs(fake - target), support)
            errors.extend(float(value) for value in values.cpu())
    if not errors:
        raise ValueError("Cannot calibrate tau from an empty train set")
    return float(np.median(np.asarray(errors, dtype=np.float64)))


def _adapter_parameter_sha(adapter: FrozenFiLMTextAdapterGenerator) -> str:
    digest = hashlib.sha256()
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        values = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _adapter_parameter_group_shas(
    adapter: FrozenFiLMTextAdapterGenerator,
) -> dict[str, str]:
    named_parameters = dict(adapter.named_parameters())
    output: dict[str, str] = {}
    for group, names in adapter.trainable_parameter_names().items():
        digest = hashlib.sha256()
        for name in names:
            values = named_parameters[name].detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(values.numpy().tobytes())
        output[group] = digest.hexdigest()
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_zero_equivalence(
    source: torch.nn.Module,
    adapter: FrozenFiLMTextAdapterGenerator,
    dataset: _ProbeDataset,
    device: torch.device,
) -> float:
    batch = _to_device(next(iter(_loader(dataset, epoch=0, shuffle=False))), device)
    current, _matched, _wrong, _target, _weight, keys, _support, current_support, _ = (
        batch
    )
    noise = stable_noise_for_keys(
        keys,
        noise_dim=32,
        base_seed=_NOISE_SEED,
        draw_index=0,
        device=device,
        dtype=current.dtype,
    )
    zero = torch.zeros((len(current), 1024), device=device, dtype=current.dtype)
    source.eval()
    adapter.eval()
    with torch.no_grad():
        expected = source(
            current, zero, noise=noise, current_support_mask=current_support
        )
        observed = adapter(
            current, zero, noise=noise, current_support_mask=current_support
        )
    return float(torch.max(torch.abs(expected - observed)).cpu())


def _evaluate(
    model: WGAN_GP,
    adapter: FrozenFiLMTextAdapterGenerator,
    dataset: _ProbeDataset,
    items: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter.eval()
    rows: list[dict[str, Any]] = []
    spatial_values: list[float] = []
    with torch.no_grad():
        for raw_batch in _loader(dataset, epoch=0, shuffle=False):
            (
                current,
                matched,
                wrong,
                target,
                _weight,
                keys,
                support,
                current_support,
                indices,
            ) = _to_device(raw_batch, model.device)
            predictions: dict[str, torch.Tensor] = {}
            for text_name, text in (
                ("matched", matched),
                ("wrong", wrong),
                ("zero", torch.zeros_like(matched)),
            ):
                prediction_sum = torch.zeros_like(target)
                for draw_index in range(16):
                    noise = stable_noise_for_keys(
                        keys,
                        noise_dim=32,
                        base_seed=_NOISE_SEED,
                        draw_index=draw_index,
                        device=model.device,
                        dtype=current.dtype,
                    )
                    prediction_sum += adapter(
                        current,
                        text,
                        noise=noise,
                        current_support_mask=current_support,
                    )
                predictions[text_name] = prediction_sum / 16.0
            metrics: dict[str, dict[str, torch.Tensor]] = {}
            for name, prediction in predictions.items():
                metrics[name] = {
                    "mae": masked_mean_per_sample(
                        torch.abs(prediction - target), support
                    ),
                    "calendar": model._calendar_penalty_per_sample(prediction, support),
                    "butterfly": model._butterfly_penalty_per_sample(
                        prediction, support
                    ),
                }
                metrics[name]["calendar_violation"] = (
                    metrics[name]["calendar"] > 1.0e-10
                ).to(torch.float32)
                metrics[name]["butterfly_violation"] = (
                    metrics[name]["butterfly"] > 1.0e-10
                ).to(torch.float32)
            if adapter.spatial_rank:
                diagnostic = adapter.activation_diagnostics(matched)
                for site in diagnostic["spatial"]:
                    gamma = site["gamma"]
                    beta = site["beta"]
                    spatial_values.append(float(gamma.var(dim=(-2, -1)).mean().cpu()))
                    spatial_values.append(float(beta.var(dim=(-2, -1)).mean().cpu()))
            for row_offset, item_index in enumerate(indices.detach().cpu().tolist()):
                item = items[int(item_index)]
                row: dict[str, Any] = {
                    "pair_id": str(item.pair_id),
                    "session_id": str(item.session_id),
                    "partition": "validation",
                }
                for text_name in ("matched", "wrong", "zero"):
                    for metric_name, values in metrics[text_name].items():
                        row[f"{metric_name}_{text_name}"] = float(
                            values[row_offset].detach().cpu()
                        )
                rows.append(row)
    if len(rows) != 110 or len({row["session_id"] for row in rows}) != 34:
        raise ValueError("Validation metric universe drifted")
    diagnostics = {
        "spatial_modulation_variance_mean": (
            float(np.mean(spatial_values)) if spatial_values else None
        ),
        "spatial_modulation_nonzero": (
            bool(spatial_values and max(spatial_values) > 0.0)
            if adapter.spatial_rank
            else None
        ),
    }
    return rows, diagnostics


def _save_adapter_state(
    path: Path,
    adapter: FrozenFiLMTextAdapterGenerator,
    optimizer: Adam,
    *,
    epoch: int,
    learning_rate: float,
    baseline_generator_sha: str,
    baseline_critic_sha: str,
    optimizer_contract: Mapping[str, Any] | None = None,
    gate_schedule: Mapping[str, Any] | None = None,
) -> None:
    parameter_groups = adapter.trainable_parameter_names()
    payload = {
        "schema_version": TEXT_SIGNAL_ADAPTER_STATE_SCHEMA,
        "phase": "end_of_epoch_after_optimizer_step",
        "epoch": int(epoch),
        "adapter": adapter.extract_adapter_state(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate": float(learning_rate),
        "parameter_group_names": parameter_groups,
        "trainable_parameter_sha256": _adapter_parameter_sha(adapter),
        "baseline_generator_sha256": baseline_generator_sha,
        "baseline_critic_sha256": baseline_critic_sha,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        },
    }
    if optimizer_contract is not None:
        payload["optimizer_contract"] = dict(optimizer_contract)
    if gate_schedule is not None:
        payload["gate_schedule"] = dict(gate_schedule)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_probe_adapter_state(
    path: str | Path,
    adapter: FrozenFiLMTextAdapterGenerator,
    optimizer: Adam,
    *,
    expected_generator_sha256: str,
    expected_critic_sha256: str,
    expected_optimizer_contract: Mapping[str, Any] | None = None,
    expected_gate_schedule: Mapping[str, Any] | None = None,
) -> int:
    """Restore an adapter-only state and its exact next-update RNG contract."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Probe adapter state must be a mapping")
    if payload.get("schema_version") != TEXT_SIGNAL_ADAPTER_STATE_SCHEMA:
        raise ValueError("Probe adapter state schema mismatch")
    if payload.get("phase") != "end_of_epoch_after_optimizer_step":
        raise ValueError("Probe adapter state phase mismatch")
    if payload.get("baseline_generator_sha256") != expected_generator_sha256:
        raise ValueError("Probe adapter Generator binding mismatch")
    if payload.get("baseline_critic_sha256") != expected_critic_sha256:
        raise ValueError("Probe adapter Critic binding mismatch")
    observed_optimizer_contract = payload.get("optimizer_contract")
    observed_gate_schedule = payload.get("gate_schedule")
    gated_state = adapter.global_residual_gate_initial is not None
    if gated_state and (
        expected_optimizer_contract is None or expected_gate_schedule is None
    ):
        raise ValueError(
            "Gated probe restore requires optimizer and gate-schedule contracts"
        )
    if expected_optimizer_contract is not None and observed_optimizer_contract != dict(
        expected_optimizer_contract
    ):
        raise ValueError("Probe adapter optimizer contract mismatch")
    if expected_gate_schedule is not None and observed_gate_schedule != dict(
        expected_gate_schedule
    ):
        raise ValueError("Probe adapter gate-schedule contract mismatch")
    adapter.load_adapter_state(payload["adapter"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng = payload.get("rng")
    if not isinstance(rng, Mapping):
        raise ValueError("Probe adapter RNG state is missing")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch_cpu"])
    cuda_states = list(rng.get("torch_cuda") or [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("Probe state contains CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)
    if _adapter_parameter_sha(adapter) != payload.get("trainable_parameter_sha256"):
        raise ValueError("Restored adapter parameter SHA mismatch")
    expected_groups = {
        str(key): tuple(value)
        for key, value in dict(payload["parameter_group_names"]).items()
    }
    if adapter.trainable_parameter_names() != expected_groups:
        raise ValueError("Restored adapter parameter-group contract mismatch")
    epoch = int(payload.get("epoch", 0))
    if epoch <= 0:
        raise ValueError("Probe adapter state epoch must be positive")
    return epoch


def _job_factor_code(spec: Mapping[str, Any]) -> str:
    code = str(spec.get("factor_code") or spec.get("factors") or "0000")
    if len(code) != 4 or any(value not in "01" for value in code):
        raise ValueError(f"Invalid factor code: {code!r}")
    return code


def _job_epochs(spec: Mapping[str, Any]) -> int:
    epochs = int(
        spec.get("max_epochs") or spec.get("epochs") or spec.get("num_epochs") or 60
    )
    if epochs not in {1, 60, 240}:
        raise ValueError(f"Probe jobs support 1, 60, or 240 epochs, got {epochs}")
    return epochs


def _set_global_gate_optimizer_phase(
    optimizer: Adam,
    *,
    enabled: bool,
    epoch: int,
    freeze_epochs: int,
    target_learning_rate: float,
) -> bool:
    """Freeze gate Adam state/LR, then release it on the first later epoch."""

    frozen = bool(enabled and epoch <= freeze_epochs)
    observed_groups = 0
    for group in optimizer.param_groups:
        if group.get("group_name") != "global_gate":
            continue
        observed_groups += 1
        group["lr"] = 0.0 if frozen else float(target_learning_rate)
    if enabled and observed_groups != 1:
        raise RuntimeError(
            "Enabled Global FiLM gates require exactly one optimizer group"
        )
    if not enabled and observed_groups:
        raise RuntimeError("Disabled Global FiLM gates leaked into the optimizer")
    return frozen


def _read_job_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Job spec must be a JSON object")
    return payload


def run_probe_job(job_spec_path: str | Path, output_dir: str | Path) -> Path:
    """Train one adapter-only probe and persist a hash-closed result."""

    spec_path = Path(job_spec_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _read_job_spec(spec_path)
    spec_sha = str(spec.get("job_spec_sha256", ""))
    if not spec_sha:
        raise ValueError("Job spec lacks its signed payload SHA")
    job_id = str(spec.get("job_id", output.name))
    input_manifest_path = Path(str(spec.get("input_manifest_path", ""))).resolve()
    if not input_manifest_path.is_file():
        raise FileNotFoundError("Job spec input_manifest_path is missing")
    root = input_manifest_path.parents[1]
    config_path = root / "contracts" / "resolved_config.json"
    config = _load_yaml_mapping(config_path)
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("schema_version") != PROBE_INPUT_SCHEMA:
        raise ValueError("Probe input manifest schema mismatch")
    expected_input_sha = str(spec.get("input_manifest_sha256", ""))
    if expected_input_sha and _sha256_file(input_manifest_path) != expected_input_sha:
        raise ValueError("Job/input manifest SHA mismatch")
    _verify_artifacts(input_manifest["artifacts"], root)
    if input_manifest.get("test_loader_materialized") is not False:
        raise ValueError("Probe input manifest does not prohibit test loading")

    factor_code = _job_factor_code(spec)
    factor_a, factor_b, factor_c, factor_d = (int(value) for value in factor_code)
    text_out_dim = int(
        spec.get("text_output_dimension") or spec.get("adapter_text_out_dim") or 128
    )
    spatial_rank = int(spec.get("spatial_rank", 2 if factor_d else 0))
    if text_out_dim == 328:
        spatial_rank = 0
    startup_mode = SMALL_NORMAL_STARTUP_MODE if factor_a else ZERO_STARTUP_MODE
    learning_rate = 2.5e-6 if factor_a else 5.0e-7
    gate_initial_raw = spec.get("global_residual_gate_initial")
    global_residual_gate_initial = (
        None if gate_initial_raw is None else float(gate_initial_raw)
    )
    global_residual_gate_max = float(spec.get("global_residual_gate_max", 0.1))
    global_residual_gate_freeze_epochs = int(
        spec.get("global_residual_gate_freeze_epochs", 0)
    )
    if global_residual_gate_initial is not None:
        if factor_d or spatial_rank:
            raise ValueError("Bounded Global FiLM gate jobs must disable Spatial FiLM")
        if not 0.0 < global_residual_gate_initial < global_residual_gate_max:
            raise ValueError("Invalid bounded Global FiLM gate range")
        if global_residual_gate_freeze_epochs < 0:
            raise ValueError("Gate freeze epochs must be nonnegative")
    elif global_residual_gate_freeze_epochs:
        raise ValueError("Gate freeze epochs require an enabled gate")
    text_encoder_learning_rate = float(
        spec.get("text_encoder_learning_rate", learning_rate)
    )
    global_film_learning_rate = float(
        spec.get("global_film_learning_rate", learning_rate)
    )
    global_gate_learning_rate = float(
        spec.get("global_gate_learning_rate", global_film_learning_rate)
    )
    for label, value in (
        ("text_encoder_learning_rate", text_encoder_learning_rate),
        ("global_film_learning_rate", global_film_learning_rate),
        ("global_gate_learning_rate", global_gate_learning_rate),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    epochs = _job_epochs(spec)
    raw_auxiliary_weight = spec.get(
        "alignment_auxiliary_weight", spec.get("auxiliary_weight", 0.0)
    )
    auxiliary_weight = (
        float(raw_auxiliary_weight)
        if factor_b and raw_auxiliary_weight is not None
        else 0.0
    )
    calibrate_auxiliary = bool(spec.get("calibrate_auxiliary_weight", False)) and bool(
        factor_b
    )
    assignment_mode = str(
        spec.get("text_assignment") or spec.get("assignment") or "matched"
    )

    baseline_vectors = _load_overlay(root, input_manifest, "baseline")
    improved_vectors = _load_overlay(root, input_manifest, "improved")
    vectors = improved_vectors if factor_c else baseline_vectors
    wrong_path = root / str(input_manifest["wrong_text"]["manifest_path"])
    wrong_plan = load_wrong_text_donor_manifest(
        wrong_path,
        expected_manifest_sha256=input_manifest["wrong_text"]["manifest_sha256"],
        expected_profile_sha256=input_manifest["wrong_text"]["profile_sha256"],
    )
    assignment_mapping = None
    if assignment_mode in {"independent_shuffle", "shuffle"}:
        shuffle_path = root / str(
            input_manifest["independent_shuffle"]["manifest_path"]
        )
        if (
            _sha256_file(shuffle_path)
            != input_manifest["independent_shuffle"]["manifest_sha256"]
        ):
            raise ValueError("Independent shuffle manifest SHA mismatch")
        assignment_mapping = _load_mapping(
            shuffle_path,
            receiver_key="receiver_pair_id",
            donor_key="donor_pair_id",
        )
    elif assignment_mode != "matched":
        raise ValueError(f"Unknown text assignment: {assignment_mode}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    bundle, model, adapter, source_generator, _model_config = _load_bundle_and_model(
        output,
        config,
        startup_mode=startup_mode,
        spatial_rank=spatial_rank,
        text_out_dim=text_out_dim,
        global_residual_gate_initial=global_residual_gate_initial,
        global_residual_gate_max=global_residual_gate_max,
    )
    _validate_bundle_pair_lineage(bundle, root, input_manifest)
    train_dataset = _ProbeDataset(
        bundle.train_items,
        vectors,
        wrong_plan.mapping,
        assignment_mapping=assignment_mapping,
    )
    validation_dataset = _ProbeDataset(
        bundle.val_items,
        vectors,
        wrong_plan.mapping,
        assignment_mapping=assignment_mapping,
    )
    zero_error = _validate_zero_equivalence(
        source_generator, adapter, validation_dataset, model.device
    )
    tolerance = float(config["model"]["zero_text_max_abs_tolerance"])
    if zero_error > tolerance:
        raise RuntimeError(
            f"G(x,0) preservation gate failed: {zero_error:.9g} > {tolerance:.9g}"
        )
    expected_parameters = int(
        spec.get(
            "expected_generator_parameters",
            (
                int(config["model"]["expected_text328_global_generator_parameters"])
                if text_out_dim == 328
                else (
                    int(config["model"]["expected_spatial_r2_generator_parameters"])
                    if spatial_rank
                    else int(config["model"]["frozen_generator_parameters"])
                )
            ),
        )
    )
    if adapter.parameter_count != expected_parameters:
        raise RuntimeError(
            f"Adapter parameter count mismatch: {adapter.parameter_count} != {expected_parameters}"
        )

    trainable_parameters = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    named_parameter_groups = adapter.trainable_parameter_groups()
    use_grouped_optimizer = global_residual_gate_initial is not None or any(
        key in spec
        for key in (
            "text_encoder_learning_rate",
            "global_film_learning_rate",
            "global_gate_learning_rate",
        )
    )
    if use_grouped_optimizer:
        configured_lrs = {
            "text_encoder": text_encoder_learning_rate,
            "global_film": global_film_learning_rate,
            "global_gate": global_gate_learning_rate,
            "spatial_film": global_film_learning_rate,
        }
        optimizer_groups = [
            {
                "params": list(parameters),
                "lr": configured_lrs[group],
                "group_name": group,
            }
            for group, parameters in named_parameter_groups.items()
        ]
        optimizer = Adam(optimizer_groups, betas=(0.5, 0.9))
    else:
        optimizer = Adam(trainable_parameters, lr=learning_rate, betas=(0.5, 0.9))
    optimizer_contract = {
        "mode": (
            "separate_text_and_film_lr_v1"
            if use_grouped_optimizer
            else "uniform_adapter_lr_v1"
        ),
        "text_encoder_learning_rate": text_encoder_learning_rate,
        "global_film_learning_rate": global_film_learning_rate,
        "global_gate_learning_rate": (
            global_gate_learning_rate
            if global_residual_gate_initial is not None
            else None
        ),
    }
    gate_schedule = {
        "mode": (
            "fixed_then_learned_direct_hard_clamped_v1"
            if global_residual_gate_initial is not None
            else "disabled"
        ),
        "initial": global_residual_gate_initial,
        "maximum": (
            global_residual_gate_max
            if global_residual_gate_initial is not None
            else None
        ),
        "freeze_epochs": global_residual_gate_freeze_epochs,
    }
    gate_parameters = tuple(named_parameter_groups.get("global_gate", ()))
    tau = _initial_tau(train_dataset, adapter, model.device)
    metrics: list[dict[str, Any]] = []
    calibrated_weight: float | None = None
    max_gradient_norm = 0.0
    first_update_gradient_norms: dict[str, float] | None = None
    initial_parameter_sha = _adapter_parameter_sha(adapter)
    initial_parameter_group_shas = _adapter_parameter_group_shas(adapter)
    gate_release_gradient_norm: float | None = None
    training_start = time.time()
    adapter.train()
    for epoch in range(1, epochs + 1):
        gate_frozen_for_epoch = _set_global_gate_optimizer_phase(
            optimizer,
            enabled=bool(gate_parameters),
            epoch=epoch,
            freeze_epochs=global_residual_gate_freeze_epochs,
            target_learning_rate=global_gate_learning_rate,
        )
        epoch_sums = {
            "total": 0.0,
            "base": 0.0,
            "auxiliary": 0.0,
            "recon": 0.0,
            "adv": 0.0,
            "calendar": 0.0,
            "butterfly": 0.0,
            "smooth": 0.0,
        }
        batches = 0
        for batch_index, raw_batch in enumerate(
            _loader(train_dataset, epoch=epoch, shuffle=True)
        ):
            (
                current,
                matched,
                wrong,
                target,
                sample_weight,
                keys,
                support,
                current_support,
                _,
            ) = _to_device(raw_batch, model.device)
            weights = validated_sample_weights(
                sample_weight,
                batch_size=len(current),
                device=model.device,
                dtype=current.dtype,
            )
            noise = stable_noise_for_keys(
                keys,
                noise_dim=32,
                base_seed=_NOISE_SEED,
                draw_index=epoch - 1,
                device=model.device,
                dtype=current.dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            fake_matched = adapter(
                current, matched, noise=noise, current_support_mask=current_support
            )
            base_loss, components = _base_loss(
                model,
                fake_matched,
                current,
                target,
                matched,
                weights,
                support,
            )
            auxiliary = base_loss.new_zeros(())
            if factor_b:
                zero = torch.zeros_like(matched)
                fake_zero = adapter(
                    current, zero, noise=noise, current_support_mask=current_support
                )
                fake_wrong = adapter(
                    current, wrong, noise=noise, current_support_mask=current_support
                )
                auxiliary, _aux_components = _alignment_loss(
                    fake_matched,
                    fake_zero,
                    fake_wrong,
                    target,
                    support,
                    weights,
                    tau=tau,
                )
                if calibrate_auxiliary and calibrated_weight is None:
                    base_norm = _gradient_norm(base_loss, trainable_parameters)
                    auxiliary_norm = _gradient_norm(auxiliary, trainable_parameters)
                    if (
                        not math.isfinite(base_norm)
                        or not math.isfinite(auxiliary_norm)
                        or auxiliary_norm <= 0.0
                    ):
                        raise RuntimeError(
                            "Cannot calibrate a finite nonzero auxiliary gradient"
                        )
                    calibrated_weight = float(
                        np.clip(0.05 * base_norm / auxiliary_norm, 1.0e-6, 100.0)
                    )
                    auxiliary_weight = calibrated_weight
            total = base_loss + auxiliary_weight * auxiliary
            total.backward()
            if (
                gate_parameters
                and epoch == global_residual_gate_freeze_epochs + 1
                and batch_index == 0
            ):
                gate_release_gradient_norm = math.sqrt(
                    sum(
                        float(parameter.grad.detach().pow(2).sum().cpu())
                        for parameter in gate_parameters
                        if parameter.grad is not None
                    )
                )
                if (
                    not math.isfinite(gate_release_gradient_norm)
                    or gate_release_gradient_norm <= 0.0
                ):
                    raise RuntimeError(
                        "Released Global FiLM gate has no finite nonzero gradient"
                    )
            if epoch == 1 and batch_index == 0:
                named_parameters = dict(adapter.named_parameters())
                first_update_gradient_norms = {}
                for group, names in adapter.trainable_parameter_names().items():
                    group_squared = 0.0
                    for name in names:
                        gradient = named_parameters[name].grad
                        if gradient is not None:
                            group_squared += float(gradient.detach().pow(2).sum().cpu())
                    first_update_gradient_norms[group] = math.sqrt(group_squared)
                if factor_a:
                    required_groups = {"text_encoder", "global_film"}
                    if spatial_rank:
                        required_groups.add("spatial_film")
                    invalid_groups = [
                        group
                        for group in sorted(required_groups)
                        if not math.isfinite(
                            first_update_gradient_norms.get(group, 0.0)
                        )
                        or first_update_gradient_norms.get(group, 0.0) <= 0.0
                    ]
                    if invalid_groups:
                        raise RuntimeError(
                            "A=1 first update did not reach all text groups: "
                            f"{invalid_groups}"
                        )
            if gate_frozen_for_epoch:
                for parameter in gate_parameters:
                    parameter.grad = None
            squared_norm = 0.0
            for parameter in trainable_parameters:
                if parameter.grad is not None:
                    squared_norm += float(parameter.grad.detach().pow(2).sum().cpu())
            gradient_norm = math.sqrt(squared_norm)
            if not math.isfinite(gradient_norm) or gradient_norm > 1.0e3:
                raise RuntimeError(
                    f"Adapter gradient explosion/nonfinite: {gradient_norm}"
                )
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            optimizer.step()
            adapter.clamp_global_residual_gates_()
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in trainable_parameters
            ):
                raise RuntimeError("Adapter parameters became nonfinite")
            epoch_sums["total"] += float(total.detach().cpu())
            epoch_sums["base"] += float(base_loss.detach().cpu())
            epoch_sums["auxiliary"] += float(auxiliary.detach().cpu())
            for name in ("recon", "adv", "calendar", "butterfly", "smooth"):
                epoch_sums[name] += float(components[name].detach().cpu())
            batches += 1
        if batches == 0:
            raise RuntimeError("Training loader produced zero batches")
        gate_values = [
            float(value.detach().cpu())
            for value in adapter.global_residual_gate_values()
        ]
        metrics.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "text_encoder_learning_rate": text_encoder_learning_rate,
                "global_film_learning_rate": global_film_learning_rate,
                "global_gate_learning_rate": (
                    (0.0 if gate_frozen_for_epoch else global_gate_learning_rate)
                    if gate_parameters
                    else None
                ),
                "global_gate_target_learning_rate": (
                    global_gate_learning_rate if gate_parameters else None
                ),
                "global_gate_frozen": bool(gate_frozen_for_epoch),
                "global_gate_mean": (
                    float(np.mean(gate_values)) if gate_values else None
                ),
                **{
                    f"global_gate_site_{index}": value
                    for index, value in enumerate(gate_values)
                },
                "auxiliary_weight": auxiliary_weight,
                **{name: value / batches for name, value in epoch_sums.items()},
                "max_gradient_norm_so_far": max_gradient_norm,
            }
        )

    final_parameter_sha = _adapter_parameter_sha(adapter)
    if final_parameter_sha == initial_parameter_sha:
        raise RuntimeError("No adapter parameter changed during training")
    pair_rows, activation_diagnostics = _evaluate(
        model, adapter, validation_dataset, bundle.val_items
    )
    if spatial_rank and not activation_diagnostics["spatial_modulation_nonzero"]:
        raise RuntimeError("Spatial modulation collapsed to zero")

    metrics_path = output / "training_metrics.csv"
    pair_metrics_path = output / "validation_pair_metrics.csv"
    diagnostics_path = output / "diagnostics.json"
    state_path = output / "text_adapter_probe_state.pt"
    _write_csv(metrics_path, metrics)
    _write_csv(pair_metrics_path, pair_rows)
    diagnostics = {
        "job_id": job_id,
        "factor_code": factor_code,
        "epochs": epochs,
        "assignment": assignment_mode,
        "zero_text_max_abs_error": zero_error,
        "zero_text_tolerance": tolerance,
        "tau": tau,
        "auxiliary_weight": auxiliary_weight,
        "calibrated_auxiliary_weight": calibrated_weight,
        "max_gradient_norm": max_gradient_norm,
        "first_update_gradient_norms": first_update_gradient_norms,
        "gate_release_gradient_norm": gate_release_gradient_norm,
        "optimizer_contract": optimizer_contract,
        "gate_schedule": gate_schedule,
        "final_global_gate_values": [
            float(value.detach().cpu())
            for value in adapter.global_residual_gate_values()
        ],
        "parameters": {
            "generator_total": adapter.parameter_count,
            "adapter_trainable": sum(
                int(parameter.numel()) for parameter in trainable_parameters
            ),
            "spatial_added": adapter.spatial_parameter_count,
        },
        "trainable_parameter_names": adapter.trainable_parameter_names(),
        "initial_parameter_sha256": initial_parameter_sha,
        "initial_parameter_group_sha256": initial_parameter_group_shas,
        "final_parameter_sha256": final_parameter_sha,
        "training_seconds": time.time() - training_start,
        **activation_diagnostics,
        "test_loader_materialized": False,
        "test_prediction_rows": 0,
        "q3_q4_input_rows": 0,
    }
    _atomic_json(diagnostics_path, diagnostics)
    _save_adapter_state(
        state_path,
        adapter,
        optimizer,
        epoch=epochs,
        learning_rate=learning_rate,
        baseline_generator_sha=config["baseline"]["generator_checkpoint"]["sha256"],
        baseline_critic_sha=config["baseline"]["critic_checkpoint"]["sha256"],
        optimizer_contract=optimizer_contract,
        gate_schedule=gate_schedule,
    )
    artifacts = [
        _artifact(metrics_path, role="training_metrics", root=output),
        _artifact(pair_metrics_path, role="validation_pair_metrics", root=output),
        _artifact(diagnostics_path, role="diagnostics", root=output),
        _artifact(state_path, role="adapter_state", root=output),
    ]
    result = {
        "schema_version": 1,
        "kind": "film_unet_text_signal_probe_job_result_v1",
        "result_contract": PROBE_RESULT_SCHEMA,
        "status": "completed",
        "job_id": job_id,
        "job_spec_path": str(spec_path),
        "job_spec_sha256": spec_sha,
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "factor_code": factor_code,
        "epochs": epochs,
        "assignment": assignment_mode,
        "allowed_partitions": ["train", "validation"],
        "test_loader_count": 0,
        "prediction_count": 0,
        "calibrated_auxiliary_weight": calibrated_weight,
        "effective_auxiliary_weight": auxiliary_weight,
        "optimizer_contract": optimizer_contract,
        "gate_schedule": gate_schedule,
        "artifacts": artifacts,
        "interpretation": "single_seed_mechanism_validation",
    }
    result_path = output / "result.json"
    _atomic_json(result_path, result)
    return result_path


def _session_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "pair_id",
        "session_id",
        "mae_matched",
        "mae_wrong",
        "mae_zero",
        "calendar_violation_matched",
        "calendar_violation_zero",
        "butterfly_violation_matched",
        "butterfly_violation_zero",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair metrics lack columns: {missing}")
    if len(frame) != 110 or frame["session_id"].nunique() != 34:
        raise ValueError("Screen analysis expected 110 pairs / 34 sessions")
    numeric = [column for column in required if column not in {"pair_id", "session_id"}]
    return frame.groupby("session_id", sort=True)[numeric].mean().reset_index()


def _bootstrap_ratio(
    sessions: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    lhs = sessions[numerator].to_numpy(dtype=np.float64)
    rhs = sessions[denominator].to_numpy(dtype=np.float64)
    estimate = float(np.log(lhs.mean() / rhs.mean()))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(lhs), size=(replicates, len(lhs)))
    draws = np.log(lhs[indices].mean(axis=1) / rhs[indices].mean(axis=1))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": estimate,
        "ci_lower": float(np.quantile(draws, alpha)),
        "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
        "bootstrap_se": float(np.std(draws, ddof=1)),
        "nonworse_sessions": int(np.sum(lhs <= rhs)),
        "replicates": int(replicates),
        "confidence_level": float(confidence),
    }


def _bootstrap_screen_maximin(
    sessions: pd.DataFrame,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    """Bootstrap both screen contrasts on one shared session resample bank."""

    matched = sessions["mae_matched"].to_numpy(dtype=np.float64)
    zero = sessions["mae_zero"].to_numpy(dtype=np.float64)
    wrong = sessions["mae_wrong"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matched), size=(replicates, len(matched)))
    matched_draw = matched[indices].mean(axis=1)
    zero_draws = np.log(matched_draw / zero[indices].mean(axis=1))
    wrong_draws = np.log(matched_draw / wrong[indices].mean(axis=1))
    alpha = (1.0 - confidence) / 2.0

    def summarize(draws: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
        return {
            "estimate": float(np.log(matched.mean() / reference.mean())),
            "ci_lower": float(np.quantile(draws, alpha)),
            "ci_upper": float(np.quantile(draws, 1.0 - alpha)),
            "bootstrap_se": float(np.std(draws, ddof=1)),
            "nonworse_sessions": int(np.sum(matched <= reference)),
            "replicates": int(replicates),
            "confidence_level": float(confidence),
            "shared_session_resample_bank": True,
        }

    maximin_draws = np.maximum(zero_draws, wrong_draws)
    return (
        summarize(zero_draws, zero),
        summarize(wrong_draws, wrong),
        float(np.std(maximin_draws, ddof=1)),
    )


def _discover_results(
    root: Path, *, epochs: int, assignment: str | None = "matched"
) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "completed"
            and int(payload.get("epochs", -1)) == epochs
            and (
                assignment is None
                or str(payload.get("assignment", "matched")) == assignment
            )
        ):
            matches.append(path)
    return sorted(matches)


def _result_artifact(result_path: Path, role: str) -> Path:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for record in payload["artifacts"]:
        if record["role"] == role:
            path = Path(record["path"])
            return path if path.is_absolute() else result_path.parent / path
    raise ValueError(f"Result lacks artifact role {role}: {result_path}")


def analyze_probe_screen(root: str | Path) -> Path:
    """Apply the frozen maximin gates to the complete 16-cell screen."""

    output_root = Path(root).resolve()
    result_paths = _discover_results(output_root, epochs=60, assignment="matched")
    by_code: dict[str, Path] = {}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        code = str(result["factor_code"])
        if code in by_code:
            raise ValueError(f"Duplicate 60-epoch result for factor code {code}")
        by_code[code] = path
    expected_codes = {f"{value:04b}" for value in range(16)}
    if set(by_code) != expected_codes:
        missing = sorted(expected_codes - set(by_code))
        raise ValueError(f"Screen is incomplete; missing factor codes: {missing}")

    rows: list[dict[str, Any]] = []
    for code in sorted(by_code):
        sessions = _session_frame(
            _result_artifact(by_code[code], "validation_pair_metrics")
        )
        zero, wrong, score_se = _bootstrap_screen_maximin(
            sessions,
            replicates=10_000,
            confidence=0.90,
            seed=20260830 + int(code, 2),
        )
        constraint_increases = {
            name: float(
                sessions[f"{name}_matched"].mean() - sessions[f"{name}_zero"].mean()
            )
            for name in ("calendar_violation", "butterfly_violation")
        }
        diagnostics = json.loads(
            _result_artifact(by_code[code], "diagnostics").read_text(encoding="utf-8")
        )
        gates = {
            "point_estimates": zero["estimate"] <= -5.0e-4
            and wrong["estimate"] <= -5.0e-4,
            "session_consistency": zero["nonworse_sessions"] >= 21
            and wrong["nonworse_sessions"] >= 21,
            "bootstrap_ci": zero["ci_upper"] < 0.0 and wrong["ci_upper"] < 0.0,
            "constraints": max(constraint_increases.values()) <= 0.01,
            "diagnostics": (
                math.isfinite(float(diagnostics["max_gradient_norm"]))
                and float(diagnostics["max_gradient_norm"]) <= 1.0e3
                and (
                    diagnostics["spatial_modulation_nonzero"] is not False
                    if code[3] == "1"
                    else True
                )
            ),
        }
        score = max(float(zero["estimate"]), float(wrong["estimate"]))
        rows.append(
            {
                "factor_code": code,
                "selection_score": score,
                "selection_score_bootstrap_se": score_se,
                "matched_vs_zero": zero,
                "matched_vs_wrong": wrong,
                "constraint_violation_rate_increases": constraint_increases,
                "gates": gates,
                "passes_all_gates": all(gates.values()),
                "parameter_count": int(diagnostics["parameters"]["generator_total"]),
                "result_path": str(by_code[code]),
            }
        )
    passing = [row for row in rows if row["passes_all_gates"]]
    winner: dict[str, Any] | None = None
    if passing:
        point_leader = min(passing, key=lambda row: row["selection_score"])
        threshold = (
            point_leader["selection_score"]
            + point_leader["selection_score_bootstrap_se"]
        )
        eligible = [row for row in passing if row["selection_score"] <= threshold]
        winner = min(
            eligible,
            key=lambda row: (
                row["parameter_count"],
                row["selection_score"],
                row["factor_code"],
            ),
        )
    payload = {
        "schema_version": PROBE_SELECTION_SCHEMA,
        "status": "completed",
        "interpretation": "single_seed_mechanism_validation",
        "screen_jobs": 16,
        "confirmation_eligible": winner is not None,
        "winner_factor_code": winner["factor_code"] if winner else None,
        "winner": winner,
        "cells": rows,
        "test_prediction_rows": 0,
        "q3_q4_input_rows": 0,
    }
    analysis_dir = output_root / "analysis"
    summary_path = analysis_dir / "screen_summary.csv"
    _write_csv(
        summary_path,
        [
            {
                "factor_code": row["factor_code"],
                "selection_score": row["selection_score"],
                "selection_score_bootstrap_se": row["selection_score_bootstrap_se"],
                "matched_vs_zero_log_ratio": row["matched_vs_zero"]["estimate"],
                "matched_vs_zero_ci_upper": row["matched_vs_zero"]["ci_upper"],
                "matched_vs_wrong_log_ratio": row["matched_vs_wrong"]["estimate"],
                "matched_vs_wrong_ci_upper": row["matched_vs_wrong"]["ci_upper"],
                "passes_all_gates": row["passes_all_gates"],
                "parameter_count": row["parameter_count"],
            }
            for row in rows
        ],
    )
    payload["artifacts"] = [
        _artifact(summary_path, role="screen_summary", root=output_root)
    ]
    selection_path = analysis_dir / "screen_selection.json"
    _atomic_json(selection_path, payload)
    return selection_path


def _paired_job_comparison(
    focal_path: Path,
    reference_path: Path,
    *,
    focal_column: str = "mae_matched",
    reference_column: str = "mae_matched",
    seed: int,
) -> dict[str, Any]:
    focal = pd.read_csv(focal_path)[["pair_id", "session_id", focal_column]].rename(
        columns={focal_column: "focal_error"}
    )
    reference = pd.read_csv(reference_path)[
        ["pair_id", "session_id", reference_column]
    ].rename(columns={reference_column: "reference_error"})
    merged = focal.merge(
        reference,
        on=["pair_id", "session_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 110 or merged["session_id"].nunique() != 34:
        raise ValueError("Confirmation comparison pair/session universe drifted")
    sessions = (
        merged.groupby("session_id", sort=True)[["focal_error", "reference_error"]]
        .mean()
        .reset_index()
    )
    return _bootstrap_ratio(
        sessions,
        "focal_error",
        "reference_error",
        replicates=10_000,
        confidence=0.95,
        seed=seed,
    )


def _confirmation_summary(root: Path) -> dict[str, Any] | None:
    selection_path = root / "analysis" / "screen_selection.json"
    if not selection_path.exists():
        return None
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner_code = selection.get("winner_factor_code")
    if not winner_code:
        return {
            "status": "not_run_no_screen_winner",
            "winner_factor_code": None,
        }
    results = _discover_results(root, epochs=240, assignment=None)
    by_role: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = _read_job_spec(Path(payload["job_spec_path"]))
        role = str(spec.get("confirmation_role", ""))
        if not role:
            continue
        if role in by_role:
            raise ValueError(f"Duplicate confirmation role: {role}")
        by_role[role] = path
        rows.append(
            {
                "role": role,
                "job_id": payload["job_id"],
                "factor_code": payload["factor_code"],
                "assignment": payload["assignment"],
                "result_path": str(path),
            }
        )
    required = {
        "baseline_0000_matched",
        "winner_matched",
        "winner_independent_shuffle",
    }
    if results and not required.issubset(by_role):
        raise ValueError(
            f"Confirmation jobs are incomplete: {sorted(required - set(by_role))}"
        )
    comparisons: dict[str, Any] = {}
    if required.issubset(by_role):
        winner_metrics = _result_artifact(
            by_role["winner_matched"], "validation_pair_metrics"
        )
        comparisons["winner_vs_frozen_no_text"] = _paired_job_comparison(
            winner_metrics,
            winner_metrics,
            focal_column="mae_matched",
            reference_column="mae_zero",
            seed=20260901,
        )
        comparisons["winner_vs_independently_trained_shuffle"] = _paired_job_comparison(
            winner_metrics,
            _result_artifact(
                by_role["winner_independent_shuffle"],
                "validation_pair_metrics",
            ),
            seed=20260902,
        )
        if "winner_text328_global_capacity_control" in by_role:
            comparisons["winner_vs_parameter_matched_global_text328"] = (
                _paired_job_comparison(
                    winner_metrics,
                    _result_artifact(
                        by_role["winner_text328_global_capacity_control"],
                        "validation_pair_metrics",
                    ),
                    seed=20260903,
                )
            )
    success = bool(comparisons) and all(
        comparison["estimate"] < 0.0 and comparison["ci_upper"] < 0.0
        for comparison in comparisons.values()
    )
    return {
        "status": "completed" if comparisons else "pending",
        "winner_factor_code": winner_code,
        "jobs": rows,
        "comparisons": comparisons,
        "success": success,
        "success_rule": "all_required_log_mae_estimates_and_95pct_ci_upper_below_zero",
        "interpretation": "single_seed_mechanism_validation",
    }


def write_probe_report(root: str | Path) -> Path:
    """Write a compact durable report without implying multi-seed evidence."""

    output_root = Path(root).resolve()
    selection_path = output_root / "analysis" / "screen_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Screen selection has not been written")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    confirmation = _confirmation_summary(output_root)
    _atomic_json(output_root / "analysis" / "confirmation_analysis.json", confirmation)
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "text_signal_probe_report.md"
    lines = [
        "# FiLM text-signal single-seed mechanism validation",
        "",
        "This experiment is labelled `single_seed_mechanism_validation`. It uses only",
        "the 5-minute `f2_2023q2` train/validation split and does not create a test loader,",
        "Q3 prediction, or Q4 prediction.",
        "",
        f"- Screen jobs completed: {selection.get('screen_jobs', 0)}/16",
        f"- Confirmation eligible: `{selection.get('confirmation_eligible', False)}`",
        f"- Winner factor code: `{selection.get('winner_factor_code')}`",
        "",
        "## Screen ranking",
        "",
        "| Code | maximin log-MAE | Pass | Parameters |",
        "|---|---:|:---:|---:|",
    ]
    for row in sorted(
        selection.get("cells", []), key=lambda value: value["selection_score"]
    ):
        lines.append(
            f"| {row['factor_code']} | {row['selection_score']:.8f} | "
            f"{'yes' if row['passes_all_gates'] else 'no'} | {row['parameter_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Confirmation",
            "",
            "```json",
            json.dumps(confirmation, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


__all__ = [
    "analyze_probe_screen",
    "load_probe_adapter_state",
    "prepare_probe_inputs",
    "run_probe_job",
    "write_probe_report",
]
