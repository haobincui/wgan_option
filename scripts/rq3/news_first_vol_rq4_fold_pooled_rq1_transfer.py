"""RQ4 fold-pooled transfer evaluation for frozen RQ1 FiLM/Pure checkpoints.

The pipeline is evaluation-only.  It freezes four RQ4 tolerance-30m panels,
imports the SHA-bound RQ1 checkpoint allowlists, runs 80 spawn-isolated GPU
prediction cells, publishes pair-level raw-joint MAE, and separates the
descriptive train+validation+test result from the test-only rolling-OOS table.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/rq3/news_first_vol_rq4_fold_pooled_rq1_transfer_10seed.yaml"
)
EXPERIMENT_KIND = "rq4_fold_pooled_rq1_film_pure_transfer_10seed_t30_v1"
WORKER_ENTRYPOINT = (
    "scripts.rq3.news_first_vol_rq4_fold_pooled_prediction_worker:"
    "run_prediction_unit"
)
CANONICAL_MODELS = ("film_cnn", "pure_cnn")
CANONICAL_FOLDS = (
    "f1_2023q1",
    "f2_2023q2",
    "f3_2023q3",
    "f4_2023q4",
)
CANONICAL_SEEDS = (
    42,
    202,
    404,
    382624741,
    1607127774,
    1662128673,
    2041145538,
    2014889368,
    1343862330,
    779214671,
)
EXPECTED_GRID_SHA256 = (
    "f5ca187a550582267b450f27a73a675ab8a2465dfb907396fc565eb94114539e"
)
EXPECTED_TASKS = 80
EXPECTED_ROWS = 11_380
EXPECTED_TEST_ROWS = 1_900
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class RQ4FoldPooledTransferError(RuntimeError):
    """Raised when the frozen transfer-evaluation contract drifts."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RQ4FoldPooledTransferError(
            "Pipeline payload is not finite canonical JSON"
        ) from exc


def payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _require_sha(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA_RE.fullmatch(digest) is None:
        raise RQ4FoldPooledTransferError(f"{label} must be a lowercase SHA-256")
    return digest


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_file(
    value: object, *, label: str, expected_sha256: object | None = None
) -> Path:
    path = resolve_path(str(value or ""))
    if not path.is_file():
        raise RQ4FoldPooledTransferError(f"{label} is missing: {path}")
    if expected_sha256 is not None:
        expected = _require_sha(expected_sha256, f"{label} SHA")
        if sha256_file(path) != expected:
            raise RQ4FoldPooledTransferError(f"{label} SHA drift: {path}")
    return path


def _atomic_bytes(path: Path, content: bytes, *, replace: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if not path.is_file() or path.read_bytes() != content:
            raise RQ4FoldPooledTransferError(f"Existing artifact drift: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> Path:
    content = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _atomic_bytes(path, content, replace=replace)


def _read_signed(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RQ4FoldPooledTransferError(f"Invalid signed manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RQ4FoldPooledTransferError(f"Manifest is not a mapping: {path}")
    unsigned = dict(payload)
    observed = str(unsigned.pop("payload_sha256", ""))
    if observed != payload_sha256(unsigned):
        raise RQ4FoldPooledTransferError(f"Manifest payload SHA drift: {path}")
    unsigned["payload_sha256"] = observed
    return unsigned


def _dataframe_csv_bytes(frame: Any, *, compressed: bool) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n")
    if not compressed:
        return text.encode("utf-8")
    target = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
        stream.write(text.encode("utf-8"))
    return target.getvalue()


def _write_dataframe(
    path: Path, frame: Any, *, compressed: bool = False, replace: bool = False
) -> Path:
    return _atomic_bytes(
        path,
        _dataframe_csv_bytes(frame, compressed=compressed),
        replace=replace,
    )


def _artifact_record(role: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RQ4FoldPooledTransferError(f"Artifact is missing: {path}")
    return {
        "role": str(role),
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_artifact_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or sha256_file(path) != _require_sha(record.get("sha256"), "artifact SHA")
    ):
        raise RQ4FoldPooledTransferError(f"Artifact record drift: {path}")
    return path


def load_config(
    config_path: str | Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    import yaml

    path = _require_file(config_path, label="RQ4 transfer config")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RQ4FoldPooledTransferError(f"Invalid YAML config: {path}") from exc
    if not isinstance(config, Mapping):
        raise RQ4FoldPooledTransferError("Config root must be a mapping")
    resolved = dict(config)
    _validate_config(resolved)
    return resolved, path


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RQ4FoldPooledTransferError(f"{label} must be a mapping")
    return dict(value)


def _validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    models = _mapping(config.get("models"), "models")
    matrix = _mapping(config.get("matrix"), "matrix")
    prediction = _mapping(config.get("prediction"), "prediction")
    analysis = _mapping(config.get("analysis"), "analysis")
    runtime = _mapping(config.get("runtime"), "runtime")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise RQ4FoldPooledTransferError("Experiment kind drift")
    if set(models) != set(CANONICAL_MODELS):
        raise RQ4FoldPooledTransferError("Model universe must be FiLM-CNN/Pure-CNN")
    if tuple(map(int, matrix.get("seeds", ()))) != CANONICAL_SEEDS:
        raise RQ4FoldPooledTransferError("Canonical 10-seed order drift")
    if tuple(map(str, matrix.get("folds", ()))) != CANONICAL_FOLDS:
        raise RQ4FoldPooledTransferError("Canonical rolling-fold order drift")
    if (
        int(matrix.get("expected_checkpoint_count", -1)) != EXPECTED_TASKS
        or int(matrix.get("expected_prediction_tasks", -1)) != EXPECTED_TASKS
        or int(matrix.get("expected_prediction_rows", -1)) != EXPECTED_ROWS
    ):
        raise RQ4FoldPooledTransferError("Frozen task/row counts drift")
    if (
        int(data.get("alignment_tolerance_minutes", -1)) != 30
        or int(data.get("forecast_horizon_minutes", -1)) != 5
        or int(data.get("surface_cell_count", -1)) != 256
        or int(data.get("embedding_dimension", -1)) != 1024
        or str(data.get("support_mask_mode")) != "raw_joint"
    ):
        raise RQ4FoldPooledTransferError("Data representation contract drift")
    if (
        int(prediction.get("mc_samples", -1)) != 64
        or int(prediction.get("noise_dim", -1)) != 32
        or not bool(prediction.get("shared_noise_across_models"))
        or bool(prediction.get("save_individual_draws"))
    ):
        raise RQ4FoldPooledTransferError("Prediction/MC contract drift")
    if (
        bool(analysis.get("combined_inference_enabled"))
        or str(analysis.get("oos_source_split")) != "test"
        or int(analysis.get("bootstrap_replicates", -1)) != 10_000
    ):
        raise RQ4FoldPooledTransferError("Analysis-scope contract drift")
    gpu_ids = tuple(map(int, runtime.get("gpu_ids", ())))
    if gpu_ids != (0, 1) or int(runtime.get("workers_per_gpu", -1)) != 4:
        raise RQ4FoldPooledTransferError(
            "Formal runtime requires GPUs 0,1 and 4 workers/GPU"
        )


def _output_root(config: Mapping[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return resolve_path(override)
    experiment = _mapping(config["experiment"], "experiment")
    return resolve_path(str(experiment["output_root"]))


def _source_files(config: Mapping[str, Any]) -> dict[str, Path]:
    data = _mapping(config["data"], "data")
    fields = {
        "merged_vol": "merged_vol_path",
        "support_audit": "support_audit_path",
        "pair_universes": "pair_universes_path",
        "frozen_scheduled_events": "frozen_scheduled_events_path",
        "market_jump_pairs": "market_jump_pairs_path",
    }
    result: dict[str, Path] = {}
    for role, key in fields.items():
        result[role] = _require_file(
            data.get(key),
            label=role,
            expected_sha256=data.get(f"{role}_sha256"),
        )
    return result


def _expected_panel_counts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    counts = _mapping(config.get("expected_fold_counts"), "expected_fold_counts")
    if tuple(counts) != CANONICAL_FOLDS:
        raise RQ4FoldPooledTransferError("Expected fold-count order drift")
    return {
        str(key): _mapping(value, f"expected_fold_counts.{key}")
        for key, value in counts.items()
    }


def _resolved_config_bytes(
    config: Mapping[str, Any], config_path: Path, root: Path
) -> bytes:
    import yaml

    payload = dict(config)
    payload["_resolved"] = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "output_root": str(root.resolve()),
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")


def _freeze_checkpoints(
    config: Mapping[str, Any], destination: Path
) -> tuple[Any, list[dict[str, Any]]]:
    import pandas as pd
    import torch

    models = _mapping(config["models"], "models")
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for model_name in CANONICAL_MODELS:
        model = _mapping(models[model_name], f"models.{model_name}")
        source_root = resolve_path(str(model["source_root"]))
        if not source_root.is_dir():
            raise RQ4FoldPooledTransferError(
                f"Source model root is missing: {source_root}"
            )
        qa_path = _require_file(
            model["source_qa_path"],
            label=f"{model_name} source QA",
            expected_sha256=model["source_qa_sha256"],
        )
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if str(qa.get("status")) != "passed":
            raise RQ4FoldPooledTransferError(f"{model_name} source QA did not pass")
        manifest_path = _require_file(
            model["checkpoint_manifest_path"],
            label=f"{model_name} checkpoint manifest",
            expected_sha256=model["checkpoint_manifest_sha256"],
        )
        frame = pd.read_csv(manifest_path, low_memory=False)
        required = {
            "job_id",
            "arm",
            "fold",
            "seed",
            "tolerance_minutes",
            "checkpoint_path",
            "checkpoint_sha256",
            "size_bytes",
        }
        if not required.issubset(frame.columns):
            raise RQ4FoldPooledTransferError(
                f"{model_name} checkpoint manifest schema drift"
            )
        selected = frame.loc[
            frame["arm"].astype(str).eq(str(model["source_arm"]))
        ].copy()
        selected["seed"] = pd.to_numeric(selected["seed"], errors="raise").astype(int)
        selected["fold"] = selected["fold"].astype(str)
        expected_cells = {
            (seed, fold) for seed in CANONICAL_SEEDS for fold in CANONICAL_FOLDS
        }
        observed_cells = set(
            selected[["seed", "fold"]].itertuples(index=False, name=None)
        )
        if (
            len(selected) != 40
            or observed_cells != expected_cells
            or selected.duplicated(["seed", "fold"]).any()
            or set(pd.to_numeric(selected["tolerance_minutes"], errors="raise")) != {5}
        ):
            raise RQ4FoldPooledTransferError(
                f"{model_name} checkpoint cell universe drift"
            )
        for row in selected.sort_values(["seed", "fold"], kind="stable").itertuples(
            index=False
        ):
            checkpoint = Path(str(row.checkpoint_path)).expanduser().resolve()
            if source_root not in checkpoint.parents:
                raise RQ4FoldPooledTransferError(
                    f"Checkpoint escaped {model_name} source root: {checkpoint}"
                )
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size != int(row.size_bytes)
                or sha256_file(checkpoint)
                != _require_sha(row.checkpoint_sha256, "checkpoint SHA")
            ):
                raise RQ4FoldPooledTransferError(
                    f"Checkpoint artifact drift: {checkpoint}"
                )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("state_dict"), Mapping
            ):
                raise RQ4FoldPooledTransferError(
                    f"Checkpoint payload drift: {checkpoint}"
                )
            parameter_count = sum(
                int(value.numel()) for value in payload["state_dict"].values()
            )
            if (
                str(payload.get("surface_grid_sha256")) != EXPECTED_GRID_SHA256
                or str(payload.get("generator_conditioning_mode"))
                != str(model["generator_conditioning_mode"])
                or parameter_count != int(model["expected_generator_parameters"])
            ):
                raise RQ4FoldPooledTransferError(
                    f"Checkpoint model/grid contract drift: {checkpoint}"
                )
            records.append(
                {
                    "model": model_name,
                    "effective_text_use": model_name == "film_cnn",
                    "source_arm": str(model["source_arm"]),
                    "source_job_id": str(row.job_id),
                    "seed": int(row.seed),
                    "checkpoint_fold": str(row.fold),
                    "source_tolerance_minutes": 5,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": str(row.checkpoint_sha256),
                    "checkpoint_size_bytes": int(row.size_bytes),
                    "source_root": str(source_root),
                    "source_qa_path": str(qa_path),
                    "source_qa_sha256": str(model["source_qa_sha256"]),
                    "source_checkpoint_manifest_path": str(manifest_path),
                    "source_checkpoint_manifest_sha256": str(
                        model["checkpoint_manifest_sha256"]
                    ),
                    "generator_conditioning_mode": str(
                        model["generator_conditioning_mode"]
                    ),
                    "generator_parameter_count": parameter_count,
                    "surface_grid_sha256": EXPECTED_GRID_SHA256,
                }
            )
        sources.extend(
            [
                _artifact_record(f"{model_name}_source_qa", qa_path),
                _artifact_record(
                    f"{model_name}_source_checkpoint_manifest", manifest_path
                ),
            ]
        )
    result = pd.DataFrame(records).sort_values(
        ["model", "seed", "checkpoint_fold"], kind="stable"
    )
    if len(result) != EXPECTED_TASKS:
        raise RQ4FoldPooledTransferError("Frozen checkpoint count drift")
    _write_dataframe(destination, result)
    return result.reset_index(drop=True), sources


def _panel_index(
    config: Mapping[str, Any], panel_root: Path
) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    from scripts.rq3.news_first_vol_rq4_fold_pooled_panels import (
        materialize_fold_pooled_panels,
        validate_fold_pooled_bundle,
    )

    paths = materialize_fold_pooled_panels(config, panel_root)
    validate_fold_pooled_bundle(panel_root, config=config)
    expected = _expected_panel_counts(config)
    rows: list[dict[str, Any]] = []
    panels: dict[str, Any] = {}
    for fold in CANONICAL_FOLDS:
        path = Path(paths[fold]).resolve()
        panel = pd.read_csv(path, low_memory=False)
        item = expected[fold]
        split_counts = item["splits"]
        observed_split = panel.groupby("source_split")["pair_id"].nunique().to_dict()
        if (
            len(panel) != int(item["pairs"])
            or panel["pair_id"].astype(str).nunique() != int(item["pairs"])
            or panel["session_id"].astype(str).nunique() != int(item["sessions"])
            or any(
                int(observed_split.get(split, -1)) != int(split_counts[split]["pairs"])
                for split in ("train", "validation", "test")
            )
        ):
            raise RQ4FoldPooledTransferError(f"Panel count drift: {fold}")
        panels[fold] = panel
        rows.append(
            {
                "checkpoint_fold": fold,
                "panel_path": str(path),
                "panel_sha256": sha256_file(path),
                "panel_size_bytes": path.stat().st_size,
                "row_count": len(panel),
                "pair_count": panel["pair_id"].astype(str).nunique(),
                "session_count": panel["session_id"].astype(str).nunique(),
                "train_pair_count": int(observed_split["train"]),
                "validation_pair_count": int(observed_split["validation"]),
                "test_pair_count": int(observed_split["test"]),
            }
        )
    return pd.DataFrame(rows), {
        "paths": paths,
        "panels": panels,
    }


def _prediction_units(
    config: Mapping[str, Any],
    root: Path,
    checkpoints: Any,
    panel_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from scripts.rq3.news_first_vol_rq4_fold_pooled_prediction_worker import (
        noise_bank_profile_sha256,
    )

    prediction = _mapping(config["prediction"], "prediction")
    panels = panel_state["panels"]
    paths = panel_state["paths"]
    units: list[dict[str, Any]] = []
    for row in checkpoints.itertuples(index=False):
        fold = str(row.checkpoint_fold)
        panel = panels[fold]
        panel_path = Path(paths[fold]).resolve()
        noise_sha = noise_bank_profile_sha256(
            seed=int(row.seed),
            fold=fold,
            sample_ids=panel["sample_id"].astype(str).tolist(),
            draws=64,
        )
        cell_dir = (
            root / "predictions" / str(row.model) / fold / f"seed_{int(row.seed)}"
        )
        units.append(
            {
                "prediction_unit_id": (
                    f"rq4_fold_pooled::{row.model}::seed_{int(row.seed)}::{fold}"
                ),
                "prediction_kind": "rq4_fold_pooled",
                "model_name": str(row.model),
                "effective_text_use": bool(row.effective_text_use),
                "source_arm": str(row.source_arm),
                "source_job_id": str(row.source_job_id),
                "seed": int(row.seed),
                "fold": fold,
                "checkpoint_fold": fold,
                "checkpoint_path": str(row.checkpoint_path),
                "checkpoint_sha256": str(row.checkpoint_sha256),
                "checkpoint_size_bytes": int(row.checkpoint_size_bytes),
                "source_root": str(row.source_root),
                "source_qa_path": str(row.source_qa_path),
                "source_qa_sha256": str(row.source_qa_sha256),
                "panel_path": str(panel_path),
                "panel_sha256": sha256_file(panel_path),
                "panel_row_count": int(len(panel)),
                "panel_pair_count": int(panel["pair_id"].astype(str).nunique()),
                "panel_session_count": int(panel["session_id"].astype(str).nunique()),
                "alignment_tolerance_minutes": 30,
                "forecast_horizon_minutes": 5,
                "mc_samples": 64,
                "noise_dim": 32,
                "sample_batch_size": int(prediction["sample_batch_size"]),
                "draw_batch_size": int(prediction["draw_batch_size"]),
                "support_mask_mode": "raw_joint",
                "generator_current_input_mode": "current_support_masked",
                "text_ablation_mode": "real_text",
                "noise_bank_namespace": (
                    f"rq4/fold_pooled/t30/{fold}/seed_{int(row.seed)}"
                ),
                "noise_bank_profile_sha256": noise_sha,
                "artifact_root": str(root.resolve()),
                "experiment_root": str(root.resolve()),
                "cell_output_dir": str(cell_dir.resolve()),
                "expected_artifact_roles": [
                    "predicted_surfaces",
                    "pair_metrics",
                    "prediction_manifest",
                ],
            }
        )
    units.sort(key=lambda item: (item["seed"], item["fold"], item["model_name"]))
    identifiers = [str(item["prediction_unit_id"]) for item in units]
    if len(units) != EXPECTED_TASKS or len(set(identifiers)) != EXPECTED_TASKS:
        raise RQ4FoldPooledTransferError("Prediction-unit universe drift")
    profiles: dict[tuple[int, str], set[str]] = {}
    for unit in units:
        profiles.setdefault((int(unit["seed"]), str(unit["fold"])), set()).add(
            str(unit["noise_bank_profile_sha256"])
        )
    if any(len(values) != 1 for values in profiles.values()):
        raise RQ4FoldPooledTransferError("Models do not share paired MC noise")
    return units


def _unit_manifest(path: Path, units: Sequence[Mapping[str, Any]]) -> Path:
    payload = _signed(
        {
            "schema_version": 1,
            "kind": "rq4_fold_pooled_prediction_units_v1",
            "prediction_unit_count": len(units),
            "units": [dict(unit) for unit in units],
        }
    )
    return _write_json(path, payload)


def _read_units(path: Path) -> list[dict[str, Any]]:
    payload = _read_signed(path)
    rows = payload.get("units")
    if (
        payload.get("kind") != "rq4_fold_pooled_prediction_units_v1"
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_TASKS
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise RQ4FoldPooledTransferError("Prediction-unit manifest drift")
    return [dict(row) for row in rows]


def prepare(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Freeze source hashes, four panels, 80 checkpoints, and 80 GPU units."""

    import pandas as pd

    config, config_file = load_config(config_path)
    root = _output_root(config, output_dir)
    registry_path = root / "registry.json"
    if registry_path.is_file():
        if not resume:
            raise RQ4FoldPooledTransferError(
                f"Existing prepared experiment requires --resume: {root}"
            )
        return validate_prepared(config_path=config_file, output_dir=root)
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    resolved_path = inputs / "resolved_config.yaml"
    _atomic_bytes(resolved_path, _resolved_config_bytes(config, config_file, root))
    sources = _source_files(config)
    source_rows = [_artifact_record(role, path) for role, path in sources.items()]

    panel_index, panel_state = _panel_index(config, inputs / "fold_pooled")
    panel_index_path = _write_dataframe(inputs / "panel_index.csv", panel_index)
    checkpoint_path = inputs / "checkpoint_manifest.csv"
    checkpoints, model_sources = _freeze_checkpoints(config, checkpoint_path)
    source_rows.extend(model_sources)
    source_hashes_path = _write_dataframe(
        inputs / "source_hashes.csv",
        pd.DataFrame(source_rows).sort_values("role", kind="stable"),
    )
    units = _prediction_units(config, root, checkpoints, panel_state)
    units_path = _unit_manifest(inputs / "prediction_units.json", units)
    unit_rows = pd.DataFrame(units).copy()
    unit_rows["expected_artifact_roles"] = unit_rows["expected_artifact_roles"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    units_csv_path = _write_dataframe(inputs / "prediction_units.csv", unit_rows)
    determinism = _signed(
        {
            "schema_version": 1,
            "kind": "rq4_fold_pooled_inference_determinism_v1",
            "seed_scope": "checkpoint_seed_before_each_prediction_cell",
            "python_numpy_torch_seeded": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
            "noise_method": "stable_noise_for_keys_v1",
            "prediction_mc_samples": 64,
            "noise_dim": 32,
            "models_share_sample_keys_and_noise_profile": True,
        }
    )
    determinism_path = _write_json(
        inputs / "inference_determinism_contract.json", determinism
    )
    registry = _signed(
        {
            "schema_version": 1,
            "kind": "rq4_fold_pooled_transfer_registry_v1",
            "experiment_kind": EXPERIMENT_KIND,
            "config_path": str(config_file.resolve()),
            "config_sha256": sha256_file(config_file),
            "output_root": str(root.resolve()),
            "expected_prediction_tasks": EXPECTED_TASKS,
            "expected_prediction_rows": EXPECTED_ROWS,
            "artifacts": {
                "resolved_config": _artifact_record("resolved_config", resolved_path),
                "source_hashes": _artifact_record("source_hashes", source_hashes_path),
                "panel_bundle_manifest": _artifact_record(
                    "panel_bundle_manifest", Path(panel_state["paths"]["manifest"])
                ),
                "panel_index": _artifact_record("panel_index", panel_index_path),
                "checkpoint_manifest": _artifact_record(
                    "checkpoint_manifest", checkpoint_path
                ),
                "prediction_units_json": _artifact_record(
                    "prediction_units_json", units_path
                ),
                "prediction_units_csv": _artifact_record(
                    "prediction_units_csv", units_csv_path
                ),
                "inference_determinism": _artifact_record(
                    "inference_determinism", determinism_path
                ),
            },
        }
    )
    _write_json(registry_path, registry)
    return validate_prepared(config_path=config_file, output_dir=root)


def validate_prepared(
    *, config_path: str | Path = DEFAULT_CONFIG, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Revalidate every frozen input and return the unit universe."""

    import pandas as pd

    config, config_file = load_config(config_path)
    root = _output_root(config, output_dir)
    registry_path = root / "registry.json"
    if not registry_path.is_file():
        raise RQ4FoldPooledTransferError(f"Experiment is not prepared: {root}")
    registry = _read_signed(registry_path)
    if (
        registry.get("kind") != "rq4_fold_pooled_transfer_registry_v1"
        or registry.get("experiment_kind") != EXPERIMENT_KIND
        or Path(str(registry.get("config_path"))).resolve() != config_file.resolve()
        or registry.get("config_sha256") != sha256_file(config_file)
        or Path(str(registry.get("output_root"))).resolve() != root.resolve()
    ):
        raise RQ4FoldPooledTransferError("Prepared registry/config lineage drift")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RQ4FoldPooledTransferError("Prepared registry artifacts are invalid")
    artifact_paths = {
        role: _validate_artifact_record(record)
        for role, record in artifacts.items()
        if isinstance(record, Mapping)
    }
    if len(artifact_paths) != len(artifacts):
        raise RQ4FoldPooledTransferError("Prepared registry has invalid artifact rows")
    from scripts.rq3.news_first_vol_rq4_fold_pooled_panels import (
        validate_fold_pooled_bundle,
    )

    panel_paths = validate_fold_pooled_bundle(
        root / "inputs/fold_pooled", config=config
    )
    checkpoints = pd.read_csv(artifact_paths["checkpoint_manifest"], low_memory=False)
    if (
        len(checkpoints) != EXPECTED_TASKS
        or checkpoints.duplicated(["model", "seed", "checkpoint_fold"]).any()
    ):
        raise RQ4FoldPooledTransferError("Prepared checkpoint manifest drift")
    for row in checkpoints.itertuples(index=False):
        path = Path(str(row.checkpoint_path)).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(row.checkpoint_size_bytes)
            or sha256_file(path) != str(row.checkpoint_sha256)
        ):
            raise RQ4FoldPooledTransferError(f"Prepared checkpoint drift: {path}")
    units = _read_units(artifact_paths["prediction_units_json"])
    panel_cache = {
        fold: pd.read_csv(panel_paths[fold], usecols=["sample_id"], low_memory=False)
        for fold in CANONICAL_FOLDS
    }
    from scripts.rq3.news_first_vol_rq4_fold_pooled_prediction_worker import (
        noise_bank_profile_sha256,
    )

    for unit in units:
        fold = str(unit["fold"])
        panel_path = Path(str(unit["panel_path"])).resolve()
        if (
            panel_path != Path(panel_paths[fold]).resolve()
            or sha256_file(panel_path) != str(unit["panel_sha256"])
            or noise_bank_profile_sha256(
                seed=int(unit["seed"]),
                fold=fold,
                sample_ids=panel_cache[fold]["sample_id"].astype(str).tolist(),
            )
            != str(unit["noise_bank_profile_sha256"])
        ):
            raise RQ4FoldPooledTransferError(
                f"Prepared prediction-unit lineage drift: {unit['prediction_unit_id']}"
            )
    return {
        "config": config,
        "config_path": config_file,
        "output_root": root,
        "registry": registry,
        "registry_path": registry_path,
        "artifacts": artifact_paths,
        "panel_paths": panel_paths,
        "checkpoints": checkpoints,
        "units": units,
    }


def predict(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    resume: bool = False,
    gpu_ids: Sequence[int] | None = None,
    workers_per_gpu: int | None = None,
) -> dict[str, Any]:
    """Run all 80 prediction cells on two isolated GPUs."""

    state = validate_prepared(config_path=config_path, output_dir=output_dir)
    runtime = _mapping(state["config"]["runtime"], "runtime")
    resolved_gpus = tuple(
        map(int, gpu_ids if gpu_ids is not None else runtime["gpu_ids"])
    )
    resolved_workers = int(
        workers_per_gpu if workers_per_gpu is not None else runtime["workers_per_gpu"]
    )
    if not resolved_gpus or resolved_workers <= 0:
        raise RQ4FoldPooledTransferError("GPU IDs/workers must be non-empty/positive")
    from scripts.rq3 import (
        news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction as parallel,
    )

    return parallel.run_parallel_prediction_units(
        state["units"],
        worker_entrypoint=WORKER_ENTRYPOINT,
        control_dir=state["output_root"] / "control/prediction",
        artifact_root=state["output_root"],
        gpu_ids=resolved_gpus,
        workers_per_gpu=resolved_workers,
        resume=bool(resume),
        experiment_kind=EXPERIMENT_KIND,
    )


def _collect_prediction_outputs(state: Mapping[str, Any]) -> dict[str, Path]:
    import pandas as pd

    from scripts.rq3 import news_first_vol_rq4_fold_pooled_prediction_worker as worker

    prediction_parts: list[Any] = []
    metric_parts: list[Any] = []
    manifest_rows: list[dict[str, Any]] = []
    for unit in state["units"]:
        paths = worker._artifact_paths(unit)
        manifest = worker._validate_complete_bundle(unit, *paths)
        predictions = pd.read_csv(paths[0], low_memory=False)
        metrics = pd.read_csv(paths[1], low_memory=False)
        prediction_parts.append(predictions)
        metric_parts.append(metrics)
        manifest_rows.append(
            {
                "prediction_unit_id": str(unit["prediction_unit_id"]),
                "model": str(unit["model_name"]),
                "seed": int(unit["seed"]),
                "checkpoint_fold": str(unit["fold"]),
                "prediction_manifest_path": str(paths[2]),
                "prediction_manifest_sha256": sha256_file(paths[2]),
                "predicted_surface_rows": int(manifest["prediction_row_count"]),
                "pair_metric_rows": int(manifest["pair_metrics_row_count"]),
            }
        )
    predictions = pd.concat(
        prediction_parts, ignore_index=True, sort=False
    ).sort_values(["model", "seed", "checkpoint_fold", "pair_id"], kind="stable")
    metrics = pd.concat(metric_parts, ignore_index=True, sort=False).sort_values(
        ["model", "seed", "checkpoint_fold", "pair_id"], kind="stable"
    )
    keys = ["model", "seed", "checkpoint_fold", "pair_id"]
    if (
        len(predictions) != EXPECTED_ROWS
        or len(metrics) != EXPECTED_ROWS
        or predictions.duplicated(keys).any()
        or metrics.duplicated(keys).any()
        or len(metrics.loc[metrics["source_split"].astype(str).eq("test")])
        != EXPECTED_TEST_ROWS
    ):
        raise RQ4FoldPooledTransferError("Global prediction/metric coverage drift")
    evaluation = Path(state["output_root"]) / "evaluation"
    prediction_path = _write_dataframe(
        evaluation / "rq4_fold_pooled_predictions.csv.gz",
        predictions,
        compressed=True,
        replace=True,
    )
    metrics_path = _write_dataframe(
        evaluation / "rq4_fold_pooled_pair_metrics.csv.gz",
        metrics,
        compressed=True,
        replace=True,
    )
    test_metrics = metrics.loc[metrics["source_split"].astype(str).eq("test")].copy()
    if len(test_metrics) != EXPECTED_TEST_ROWS:
        raise RQ4FoldPooledTransferError("Test-only pair-metric coverage drift")
    test_metrics_path = _write_dataframe(
        evaluation / "rq4_test_only_pair_metrics.csv.gz",
        test_metrics,
        compressed=True,
        replace=True,
    )
    manifest_frame = pd.DataFrame(manifest_rows).sort_values(
        ["model", "seed", "checkpoint_fold"], kind="stable"
    )
    manifest_csv = _write_dataframe(
        evaluation / "prediction_manifest.csv", manifest_frame, replace=True
    )
    collection = _signed(
        {
            "schema_version": 1,
            "kind": "rq4_fold_pooled_prediction_collection_v1",
            "prediction_task_count": EXPECTED_TASKS,
            "prediction_row_count": EXPECTED_ROWS,
            "pair_metric_row_count": EXPECTED_ROWS,
            "test_pair_metric_row_count": EXPECTED_TEST_ROWS,
            "artifacts": {
                "predictions": _artifact_record("predictions", prediction_path),
                "pair_metrics": _artifact_record("pair_metrics", metrics_path),
                "test_only_pair_metrics": _artifact_record(
                    "test_only_pair_metrics", test_metrics_path
                ),
                "prediction_manifest": _artifact_record(
                    "prediction_manifest", manifest_csv
                ),
            },
        }
    )
    collection_path = _write_json(
        evaluation / "prediction_collection_manifest.json",
        collection,
        replace=True,
    )
    return {
        "predictions": prediction_path,
        "pair_metrics": metrics_path,
        "test_only_pair_metrics": test_metrics_path,
        "prediction_manifest": manifest_csv,
        "collection_manifest": collection_path,
    }


def analyze(
    *, config_path: str | Path = DEFAULT_CONFIG, output_dir: str | Path | None = None
) -> dict[str, Path]:
    """Collect cell artifacts and publish combined/OOS analysis tables."""

    state = validate_prepared(config_path=config_path, output_dir=output_dir)
    execution = (
        state["output_root"]
        / "control/prediction/parallel_prediction_execution_manifest.json"
    )
    if not execution.is_file():
        raise RQ4FoldPooledTransferError("Prediction stage is incomplete")
    collected = _collect_prediction_outputs(state)
    settings = _mapping(state["config"]["analysis"], "analysis")
    from scripts.rq3.news_first_vol_rq4_fold_pooled_analysis import (
        analyze as analyze_metrics,
    )

    paths = analyze_metrics(
        collected["pair_metrics"],
        state["output_root"] / "analysis",
        expected_seeds=CANONICAL_SEEDS,
        expected_folds=CANONICAL_FOLDS,
        bootstrap_iterations=int(settings["bootstrap_replicates"]),
        bootstrap_seed=int(settings["bootstrap_seed"]),
        strict_design_counts=True,
    )
    return {**collected, **{f"analysis_{key}": value for key, value in paths.items()}}


def qa(
    *, config_path: str | Path = DEFAULT_CONFIG, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Run terminal scientific-lineage and coverage checks."""

    import numpy as np
    import pandas as pd

    state = validate_prepared(config_path=config_path, output_dir=output_dir)
    root = Path(state["output_root"])
    collection_path = root / "evaluation/prediction_collection_manifest.json"
    analysis_path = root / "analysis/analysis_manifest.json"
    execution_path = (
        root / "control/prediction/parallel_prediction_execution_manifest.json"
    )
    for path in (collection_path, analysis_path, execution_path):
        if not path.is_file():
            raise RQ4FoldPooledTransferError(
                f"Required terminal artifact missing: {path}"
            )
    collection = _read_signed(collection_path)
    metric_path = _validate_artifact_record(collection["artifacts"]["pair_metrics"])
    prediction_path = _validate_artifact_record(collection["artifacts"]["predictions"])
    test_metric_path = _validate_artifact_record(
        collection["artifacts"]["test_only_pair_metrics"]
    )
    metrics = pd.read_csv(metric_path, low_memory=False)
    predictions = pd.read_csv(prediction_path, low_memory=False)
    test_metrics = pd.read_csv(test_metric_path, low_memory=False)
    keys = ["seed", "checkpoint_fold", "pair_id"]

    def grouped_columns_identical(frame: Any, columns: Sequence[str]) -> bool:
        if not set(columns).issubset(frame.columns):
            return False
        counts = frame.groupby(keys, sort=False)[list(columns)].nunique(dropna=False)
        return bool(counts.eq(1).all().all())

    pair_counts = metrics.groupby(keys)["model"].nunique()
    prediction_pair_counts = predictions.groupby(keys)["model"].nunique()
    paired_input_lineage = grouped_columns_identical(
        predictions,
        (
            "sample_id",
            "panel_sha256",
            "source_split",
            "event_regime",
        ),
    )
    paired_support_lineage = grouped_columns_identical(
        predictions,
        (
            "support_mask_flat",
            "supported_cell_count",
            "support_method",
            "support_grid_fingerprint",
        ),
    )
    paired_current_input_lineage = grouped_columns_identical(
        predictions,
        (
            "current_support_cell_count",
            "current_support_mask_fingerprint",
            "generator_current_input_mode",
            "generator_current_input_fingerprint",
        ),
    )
    paired_noise_lineage = grouped_columns_identical(
        predictions,
        (
            "noise_bank_profile_sha256",
            "generator_noise_mode",
            "generator_noise_fingerprint",
            "prediction_mc_samples",
        ),
    )
    finite_metrics = np.isfinite(
        metrics[["target_mae", "persistence_mae", "supported_cell_count"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    ).all()
    surface_lengths = predictions["predicted_surface_flat"].map(
        lambda value: len(json.loads(value))
    )
    surface_finite = predictions["predicted_surface_flat"].map(
        lambda value: bool(
            np.isfinite(np.asarray(json.loads(value), dtype=float)).all()
        )
    )
    test = metrics.loc[metrics["source_split"].astype(str).eq("test")].reset_index(
        drop=True
    )
    test_unique = test[
        [
            "checkpoint_fold",
            "pair_id",
            "session_id",
            "scheduled_event",
            "market_jump",
            "event_regime",
        ]
    ].drop_duplicates(["checkpoint_fold", "pair_id"])
    scheduled_test = (
        test_unique["scheduled_event"].astype(str).str.lower().isin({"true", "1"})
    )
    jump_test = test_unique["market_jump"].astype(str).str.lower().isin({"true", "1"})

    expected_panels = _expected_panel_counts(state["config"])
    panel_counts_exact = True
    split_time_ranges_disjoint = True
    for fold in CANONICAL_FOLDS:
        panel = pd.read_csv(state["panel_paths"][fold], low_memory=False)
        expected = expected_panels[fold]
        scheduled = panel["scheduled_event"].astype(str).str.lower().isin({"true", "1"})
        jump = panel["market_jump"].astype(str).str.lower().isin({"true", "1"})
        panel_counts_exact = panel_counts_exact and all(
            (
                len(panel) == int(expected["pairs"]),
                panel["pair_id"].astype(str).nunique() == int(expected["pairs"]),
                panel["session_id"].astype(str).nunique() == int(expected["sessions"]),
                int(scheduled.sum()) == int(expected["scheduled"]),
                int(jump.sum()) == int(expected["jump"]),
                int((scheduled & jump).sum()) == int(expected["both"]),
                set(panel["checkpoint_fold"].astype(str)) == {fold},
            )
        )
        origin = pd.to_datetime(
            panel["effective_origin_utc"], errors="coerce", utc=True
        )
        split_times = {
            split: origin.loc[panel["source_split"].astype(str).eq(split)]
            for split in ("train", "validation", "test")
        }
        split_time_ranges_disjoint = split_time_ranges_disjoint and all(
            (
                origin.notna().all(),
                split_times["train"].max() < split_times["validation"].min(),
                split_times["validation"].max() < split_times["test"].min(),
            )
        )
        for split in ("train", "validation", "test"):
            selected = panel.loc[panel["source_split"].astype(str).eq(split)]
            split_expected = expected["splits"][split]
            panel_counts_exact = panel_counts_exact and all(
                (
                    selected["pair_id"].astype(str).nunique()
                    == int(split_expected["pairs"]),
                    selected["session_id"].astype(str).nunique()
                    == int(split_expected["sessions"]),
                )
            )

    source_qa_passed = True
    source_qa_rows = state["checkpoints"][["source_qa_path", "source_qa_sha256"]]
    for row in source_qa_rows.drop_duplicates().itertuples(index=False):
        source_qa = Path(str(row.source_qa_path)).resolve()
        if not source_qa.is_file() or sha256_file(source_qa) != str(
            row.source_qa_sha256
        ):
            source_qa_passed = False
            continue
        try:
            source_qa_payload = json.loads(source_qa.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_qa_passed = False
            continue
        source_qa_passed = source_qa_passed and (
            str(source_qa_payload.get("status")) == "passed"
        )

    execution = _read_signed(execution_path)
    execution_cells_valid = True
    gpu_task_counts = {0: 0, 1: 0}
    gpu_worker_pids: dict[int, set[int]] = {0: set(), 1: set()}
    for record in execution.get("cell_manifests", []):
        cell_path = execution_path.parent / str(record.get("relative_path", ""))
        execution_cells_valid = execution_cells_valid and (
            cell_path.is_file()
            and sha256_file(cell_path) == str(record.get("sha256", ""))
        )
        if not cell_path.is_file():
            continue
        try:
            cell = _read_signed(cell_path)
            physical_gpu = int(cell.get("physical_gpu_id", -1))
            worker_pid = int(cell.get("worker_pid", -1))
        except (OSError, TypeError, ValueError, RQ4FoldPooledTransferError):
            execution_cells_valid = False
            continue
        execution_cells_valid = execution_cells_valid and all(
            (
                cell.get("kind") == "backbone_text_effect_prediction_cell_manifest_v1",
                str(cell.get("prediction_unit_id"))
                == str(record.get("prediction_unit_id")),
                physical_gpu in {0, 1},
                str(cell.get("cuda_visible_devices")) == str(physical_gpu),
                int(cell.get("logical_device", -1)) == 0,
                worker_pid > 0,
            )
        )
        if physical_gpu in gpu_task_counts and worker_pid > 0:
            gpu_task_counts[physical_gpu] += 1
            gpu_worker_pids[physical_gpu].add(worker_pid)
    scheduler_topology_valid = all(
        (
            execution.get("kind")
            == "backbone_text_effect_parallel_prediction_manifest_v1",
            int(execution.get("prediction_unit_count", -1)) == EXPECTED_TASKS,
            len(execution.get("cell_manifests", [])) == EXPECTED_TASKS,
            execution.get("physical_gpu_ids") == [0, 1],
            execution.get("workers_per_gpu") == {"0": 4, "1": 4},
            len(execution.get("shared_noise_profiles", {})) == 40,
            gpu_task_counts == {0: 40, 1: 40},
            {gpu: len(pids) for gpu, pids in gpu_worker_pids.items()} == {0: 4, 1: 4},
            execution_cells_valid,
        )
    )

    from scripts.rq3 import news_first_vol_rq4_fold_pooled_analysis as analysis_module

    analysis_bundle_valid = True
    analysis_input_hash_matches = False
    report_present = False
    try:
        analysis_manifest = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis_bundle_valid = (
            analysis_manifest.get("kind") == "rq4_fold_pooled_analysis_manifest_v1"
        )
        normalised_metrics, _ = analysis_module.normalise_pair_metrics(
            metrics,
            expected_seeds=CANONICAL_SEEDS,
            expected_folds=CANONICAL_FOLDS,
            strict_design_counts=True,
        )
        analysis_input_hash_matches = (
            analysis_manifest.get("input_canonical_sha256")
            == analysis_module._canonical_frame_sha256(normalised_metrics)
            and Path(str(analysis_manifest.get("input_path", ""))).resolve()
            == metric_path.resolve()
        )
        expected_analysis_rows = {
            "combined_cell_summary": 80,
            "combined_model_summary": 2,
            "combined_cell_comparisons": 40,
            "combined_comparison_summary": 1,
            "combined_strata_summary": 22,
            "test_only_oos_summary": 2,
            "test_only_bootstrap": 3,
            "test_only_direction_consistency": 42,
        }
        analysis_records = analysis_manifest.get("artifacts", {})
        analysis_bundle_valid = analysis_bundle_valid and set(analysis_records) == set(
            analysis_module.ARTIFACT_FILENAMES
        )
        for role, filename in analysis_module.ARTIFACT_FILENAMES.items():
            record = analysis_records.get(role, {})
            artifact = root / "analysis" / filename
            table = (
                pd.read_csv(artifact, low_memory=False) if artifact.is_file() else None
            )
            analysis_bundle_valid = analysis_bundle_valid and all(
                (
                    artifact.is_file(),
                    Path(str(record.get("path", ""))).resolve() == artifact.resolve(),
                    artifact.is_file()
                    and sha256_file(artifact) == str(record.get("sha256", "")),
                    table is not None
                    and len(table)
                    == int(record.get("rows", -1))
                    == expected_analysis_rows[role],
                    table is not None
                    and list(table.columns) == list(record.get("columns", [])),
                )
            )
        report_record = analysis_manifest.get("report", {})
        report_path = root / "analysis/rq4_fold_pooled_report.md"
        report_present = all(
            (
                report_path.is_file(),
                Path(str(report_record.get("path", ""))).resolve()
                == report_path.resolve(),
                report_path.is_file()
                and report_path.stat().st_size
                == int(report_record.get("size_bytes", -1)),
                report_path.is_file()
                and sha256_file(report_path) == str(report_record.get("sha256", "")),
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        analysis_bundle_valid = False
        analysis_input_hash_matches = False
        report_present = False
    checks = {
        "source_model_qa_passed": bool(source_qa_passed),
        "fold_pooled_panel_count": len(CANONICAL_FOLDS) == 4,
        "fold_pooled_panel_counts_exact": bool(panel_counts_exact),
        "source_split_time_ranges_disjoint": bool(split_time_ranges_disjoint),
        "checkpoint_count": len(state["checkpoints"]) == EXPECTED_TASKS,
        "dual_gpu_worker_topology": bool(scheduler_topology_valid),
        "prediction_task_count": int(collection["prediction_task_count"])
        == EXPECTED_TASKS,
        "prediction_row_count": len(predictions) == EXPECTED_ROWS,
        "pair_metric_row_count": len(metrics) == EXPECTED_ROWS,
        "test_metric_row_count": len(test) == EXPECTED_TEST_ROWS,
        "test_only_artifact_matches": len(test_metrics) == EXPECTED_TEST_ROWS
        and test_metrics.equals(test),
        "test_unique_pair_count": test_unique["pair_id"].nunique() == 95,
        "test_unique_session_count": test_unique["session_id"].nunique() == 44,
        "test_event_counts_exact": int(scheduled_test.sum()) == 70
        and int(jump_test.sum()) == 31
        and int((scheduled_test & jump_test).sum()) == 6,
        "paired_model_coverage": bool(pair_counts.eq(2).all()),
        "paired_prediction_coverage": bool(prediction_pair_counts.eq(2).all()),
        "paired_input_lineage": bool(paired_input_lineage),
        "paired_support_mask_lineage": bool(paired_support_lineage),
        "paired_current_input_lineage": bool(paired_current_input_lineage),
        "paired_noise_lineage": bool(paired_noise_lineage),
        "finite_nonnegative_metrics": bool(finite_metrics)
        and bool((pd.to_numeric(metrics["target_mae"]) >= 0.0).all()),
        "positive_raw_joint_support": bool(
            (pd.to_numeric(metrics["supported_cell_count"]) > 0).all()
        ),
        "prediction_surface_width_256": bool(surface_lengths.eq(256).all()),
        "finite_prediction_surfaces": bool(surface_finite.all()),
        "mc64_without_fallback": bool(
            pd.to_numeric(predictions["prediction_mc_samples"], errors="coerce")
            .eq(64)
            .all()
        )
        and not predictions["prediction_fallback"]
        .astype(str)
        .str.lower()
        .isin({"true", "1"})
        .any(),
        "combined_inference_disabled": not bool(
            state["config"]["analysis"]["combined_inference_enabled"]
        ),
        "analysis_input_hash_matches": bool(analysis_input_hash_matches),
        "analysis_bundle_hashes_and_schema_valid": bool(analysis_bundle_valid),
        "human_readable_report_present": bool(report_present),
    }
    failed = sorted(key for key, value in checks.items() if not bool(value))
    if failed:
        raise RQ4FoldPooledTransferError(f"Terminal QA failed: {failed}")
    hash_rows: list[dict[str, Any]] = []
    excluded = {root / "qa.json", root / "output_hashes.csv"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in excluded or path.name.endswith(".lock"):
            continue
        hash_rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    hashes_path = _write_dataframe(
        root / "output_hashes.csv", pd.DataFrame(hash_rows), replace=True
    )
    payload = _signed(
        {
            "schema_version": 1,
            "kind": "rq4_fold_pooled_transfer_terminal_qa_v1",
            "status": "passed",
            "experiment_kind": EXPERIMENT_KIND,
            "checks": checks,
            "counts": {
                "panels": 4,
                "checkpoints": EXPECTED_TASKS,
                "prediction_tasks": EXPECTED_TASKS,
                "predictions": len(predictions),
                "pair_metrics": len(metrics),
                "test_pair_metrics": len(test),
                "test_unique_pairs": int(test_unique["pair_id"].nunique()),
                "test_unique_sessions": int(test_unique["session_id"].nunique()),
                "test_scheduled_pairs": int(scheduled_test.sum()),
                "test_jump_pairs": int(jump_test.sum()),
                "test_both_pairs": int((scheduled_test & jump_test).sum()),
            },
            "interpretation": {
                "combined": "descriptive_train_validation_test_mixture",
                "test_only": "retrospective_rolling_out_of_sample_development_evidence",
                "market_jump": "post_hoc_target_dependent_stress_regime",
            },
            "output_hashes": _artifact_record("output_hashes", hashes_path),
        }
    )
    _write_json(root / "qa.json", payload, replace=True)
    return payload


def run_pipeline(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    resume: bool = False,
    gpu_ids: Sequence[int] | None = None,
    workers_per_gpu: int | None = None,
) -> dict[str, Any]:
    prepare(
        config_path=config_path,
        output_dir=output_dir,
        resume=resume,
    )
    predict(
        config_path=config_path,
        output_dir=output_dir,
        resume=resume,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
    )
    analyze(config_path=config_path, output_dir=output_dir)
    return qa(config_path=config_path, output_dir=output_dir)


def status(
    *, config_path: str | Path = DEFAULT_CONFIG, output_dir: str | Path | None = None
) -> dict[str, Any]:
    config, _ = load_config(config_path)
    root = _output_root(config, output_dir)
    registry = root / "registry.json"
    cells = root / "control/prediction/prediction_cells"
    completed = len(list(cells.glob("*.json"))) if cells.is_dir() else 0
    qa_path = root / "qa.json"
    return {
        "experiment_root": str(root),
        "prepared": registry.is_file(),
        "completed_prediction_tasks": completed,
        "expected_prediction_tasks": EXPECTED_TASKS,
        "prediction_complete": (
            root / "control/prediction/parallel_prediction_execution_manifest.json"
        ).is_file(),
        "analysis_complete": (root / "analysis/analysis_manifest.json").is_file(),
        "qa_status": (
            json.loads(qa_path.read_text(encoding="utf-8")).get("status")
            if qa_path.is_file()
            else "pending"
        ),
    }


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU IDs must be a unique comma list")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "predict", "analyze", "qa", "run-pipeline", "status"),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids)
    parser.add_argument("--workers-per-gpu", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {"config_path": args.config, "output_dir": args.output_dir}
    if args.command == "prepare":
        result = prepare(**common, resume=args.resume)
        output: Any = {
            "output_root": str(result["output_root"]),
            "prediction_tasks": len(result["units"]),
        }
    elif args.command == "predict":
        output = predict(
            **common,
            resume=args.resume,
            gpu_ids=args.gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
        )
    elif args.command == "analyze":
        output = {key: str(value) for key, value in analyze(**common).items()}
    elif args.command == "qa":
        output = qa(**common)
    elif args.command == "run-pipeline":
        output = run_pipeline(
            **common,
            resume=args.resume,
            gpu_ids=args.gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
        )
    else:
        output = status(**common)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CANONICAL_FOLDS",
    "CANONICAL_MODELS",
    "CANONICAL_SEEDS",
    "DEFAULT_CONFIG",
    "EXPERIMENT_KIND",
    "RQ4FoldPooledTransferError",
    "analyze",
    "load_config",
    "main",
    "payload_sha256",
    "predict",
    "prepare",
    "qa",
    "run_pipeline",
    "sha256_file",
    "status",
    "validate_prepared",
]
