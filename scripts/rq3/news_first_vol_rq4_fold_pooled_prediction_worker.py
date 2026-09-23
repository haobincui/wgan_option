"""Spawn-safe prediction worker for the RQ4 fold-pooled transfer experiment.

Only standard-library modules are imported before the scheduler isolates a
physical GPU.  Pandas, NumPy, and Torch-backed inference code are imported
inside :func:`run_prediction_unit`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class RQ4FoldPooledPredictionError(RuntimeError):
    """Raised when a prediction unit or its artifacts drift."""


PREDICTION_KIND = "rq4_fold_pooled"
MODELS = {"film_cnn", "pure_cnn"}
EXPECTED_SPLITS = {"train", "validation", "test"}
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_EVALUATOR: Any | None = None


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
        raise RQ4FoldPooledPredictionError(
            "Prediction payload is not finite canonical JSON"
        ) from exc


def payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def noise_bank_profile_sha256(
    *, seed: int, fold: str, sample_ids: Sequence[str], draws: int = 64
) -> str:
    """Hash the arm-independent stable-key noise universe for one fold."""

    identifiers = [str(value).strip() for value in sample_ids]
    if (
        int(draws) != 64
        or not identifiers
        or any(not value for value in identifiers)
        or len(identifiers) != len(set(identifiers))
    ):
        raise RQ4FoldPooledPredictionError(
            "RQ4 fold-pooled noise bank requires 64 draws and unique sample IDs"
        )
    return payload_sha256(
        {
            "schema_version": 1,
            "kind": "rq4_fold_pooled_shared_mc_noise_bank_v1",
            "method": "stable_noise_for_keys_v1",
            "alignment_tolerance_minutes": 30,
            "forecast_horizon_minutes": 5,
            "seed": int(seed),
            "checkpoint_fold": str(fold),
            "sample_ids": sorted(identifiers),
            "draws": 64,
            "noise_dim": 32,
        }
    )


def _require_sha(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA_RE.fullmatch(digest) is None:
        raise RQ4FoldPooledPredictionError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_file(
    value: object,
    *,
    label: str,
    sha256: object | None = None,
    size_bytes: object | None = None,
) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    if not path.is_file():
        raise RQ4FoldPooledPredictionError(f"{label} is missing: {path}")
    if size_bytes is not None and path.stat().st_size != int(size_bytes):
        raise RQ4FoldPooledPredictionError(f"{label} size drift: {path}")
    if sha256 is not None and sha256_file(path) != _require_sha(sha256, f"{label} SHA"):
        raise RQ4FoldPooledPredictionError(f"{label} SHA drift: {path}")
    return path


def _require_inside(path: Path, root: Path, label: str) -> None:
    if path != root and root not in path.parents:
        raise RQ4FoldPooledPredictionError(f"{label} escaped artifact root: {path}")


def _validate_device(logical_device: int) -> None:
    if int(logical_device) != 0:
        raise RQ4FoldPooledPredictionError(
            "Prediction workers must use isolated logical cuda:0"
        )
    declared = os.environ.get("RQ3_LOGICAL_CUDA_DEVICE")
    if declared is not None and declared != "0":
        raise RQ4FoldPooledPredictionError("RQ3_LOGICAL_CUDA_DEVICE must equal zero")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) != 1:
            raise RQ4FoldPooledPredictionError(
                "Prediction child must see exactly one physical GPU"
            )


def _validate_unit(unit: Mapping[str, Any], logical_device: int) -> dict[str, Any]:
    payload = dict(unit)
    required = {
        "prediction_unit_id",
        "prediction_kind",
        "model_name",
        "source_arm",
        "seed",
        "fold",
        "checkpoint_path",
        "checkpoint_sha256",
        "panel_path",
        "panel_sha256",
        "panel_row_count",
        "panel_pair_count",
        "panel_session_count",
        "artifact_root",
        "cell_output_dir",
        "noise_bank_profile_sha256",
        "mc_samples",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RQ4FoldPooledPredictionError(
            f"Prediction unit is missing fields: {missing}"
        )
    if str(payload["prediction_kind"]) != PREDICTION_KIND:
        raise RQ4FoldPooledPredictionError("Unexpected prediction_kind")
    if str(payload["model_name"]) not in MODELS:
        raise RQ4FoldPooledPredictionError("Unexpected RQ1 model")
    if int(payload["mc_samples"]) != 64:
        raise RQ4FoldPooledPredictionError("Formal prediction requires MC=64")
    if (
        not str(payload["prediction_unit_id"]).strip()
        or not str(payload["fold"]).strip()
    ):
        raise RQ4FoldPooledPredictionError("Unit ID and fold must be non-empty")
    int(payload["seed"])
    _require_sha(payload["noise_bank_profile_sha256"], "noise-bank profile")
    _require_file(
        payload["checkpoint_path"],
        label="Generator checkpoint",
        sha256=payload["checkpoint_sha256"],
        size_bytes=payload.get("checkpoint_size_bytes"),
    )
    _require_file(
        payload["panel_path"],
        label="Fold-pooled panel",
        sha256=payload["panel_sha256"],
    )
    if payload.get("source_qa_path"):
        _require_file(
            payload["source_qa_path"],
            label="Source QA",
            sha256=payload.get("source_qa_sha256"),
        )
    artifact_root = Path(str(payload["artifact_root"])).expanduser().resolve()
    if not artifact_root.is_dir():
        raise RQ4FoldPooledPredictionError(f"Artifact root is missing: {artifact_root}")
    cell_dir = Path(str(payload["cell_output_dir"])).expanduser().resolve()
    _require_inside(cell_dir, artifact_root, "Cell output directory")
    _validate_device(logical_device)
    return payload


def _artifact_paths(unit: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(str(unit["cell_output_dir"])).expanduser().resolve()
    return (
        root / "predicted_surfaces.csv.gz",
        root / "pair_metrics.csv.gz",
        root / "prediction_manifest.json",
    )


def _bundle_state(paths: Sequence[Path]) -> str:
    exists = [path.is_file() for path in paths]
    if all(exists):
        return "complete"
    if any(exists):
        return "partial"
    return "absent"


def _atomic_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.gz")
    try:
        frame.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest_payload(
    unit: Mapping[str, Any], prediction_path: Path, metrics_path: Path
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "rq4_fold_pooled_prediction_bundle_v1",
        "prediction_unit_id": str(unit["prediction_unit_id"]),
        "model": str(unit["model_name"]),
        "source_arm": str(unit["source_arm"]),
        "seed": int(unit["seed"]),
        "checkpoint_fold": str(unit["fold"]),
        "checkpoint_path": str(Path(str(unit["checkpoint_path"])).resolve()),
        "checkpoint_sha256": str(unit["checkpoint_sha256"]),
        "panel_path": str(Path(str(unit["panel_path"])).resolve()),
        "panel_sha256": str(unit["panel_sha256"]),
        "panel_row_count": int(unit["panel_row_count"]),
        "panel_pair_count": int(unit["panel_pair_count"]),
        "panel_session_count": int(unit["panel_session_count"]),
        "prediction_mc_samples": 64,
        "noise_bank_profile_sha256": str(unit["noise_bank_profile_sha256"]),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_row_count": int(unit["panel_row_count"]),
        "pair_metrics_path": str(metrics_path.resolve()),
        "pair_metrics_sha256": sha256_file(metrics_path),
        "pair_metrics_row_count": int(unit["panel_pair_count"]),
    }
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RQ4FoldPooledPredictionError(
            f"Invalid prediction manifest: {path}"
        ) from exc
    observed = str(payload.pop("payload_sha256", ""))
    if observed != payload_sha256(payload):
        raise RQ4FoldPooledPredictionError("Prediction manifest payload SHA drift")
    payload["payload_sha256"] = observed
    return payload


def _validate_complete_bundle(
    unit: Mapping[str, Any],
    prediction_path: Path,
    metrics_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    payload = _read_manifest(manifest_path)
    expected = _manifest_payload(unit, prediction_path, metrics_path)
    keys = {
        "kind",
        "prediction_unit_id",
        "model",
        "source_arm",
        "seed",
        "checkpoint_fold",
        "checkpoint_sha256",
        "panel_sha256",
        "panel_row_count",
        "panel_pair_count",
        "panel_session_count",
        "prediction_mc_samples",
        "noise_bank_profile_sha256",
        "prediction_sha256",
        "prediction_row_count",
        "pair_metrics_sha256",
        "pair_metrics_row_count",
    }
    if any(payload.get(key) != expected.get(key) for key in keys):
        raise RQ4FoldPooledPredictionError("Prediction bundle lineage drift")
    predictions = pd.read_csv(prediction_path, low_memory=False)
    metrics = pd.read_csv(metrics_path, low_memory=False)
    if (
        len(predictions) != int(unit["panel_row_count"])
        or len(metrics) != int(unit["panel_pair_count"])
        or predictions["sample_id"].astype(str).duplicated().any()
        or metrics["pair_id"].astype(str).duplicated().any()
    ):
        raise RQ4FoldPooledPredictionError("Prediction bundle coverage drift")
    for column in ("target_mae", "persistence_mae"):
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or bool((values < 0.0).any()):
            raise RQ4FoldPooledPredictionError(
                f"Prediction bundle contains invalid {column}"
            )
    return payload


def _get_evaluator(unit: Mapping[str, Any]) -> Any:
    global _EVALUATOR
    if _EVALUATOR is None:
        from scripts.rq3.news_first_vol_comparison_analysis import TrainedRunEvaluator

        _EVALUATOR = TrainedRunEvaluator(
            mc_samples=64,
            sample_batch_size=int(unit.get("sample_batch_size", 32)),
            draw_batch_size=int(unit.get("draw_batch_size", 64)),
            device="cuda:0",
        )
    return _EVALUATOR


def _run_prediction(unit: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import pandas as pd
    import torch

    from scripts.rq3 import news_first_vol_comparison_analysis as comparison
    from wgan_option.utils.reproducibility import seed_everything

    seed_everything(int(unit["seed"]))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    panel_path = Path(str(unit["panel_path"])).resolve()
    panel = pd.read_csv(panel_path, low_memory=False)
    if "scheduled_event" not in panel.columns and "is_scheduled_event" in panel.columns:
        panel["scheduled_event"] = panel["is_scheduled_event"]
    if "market_jump" not in panel.columns and "is_market_jump" in panel.columns:
        panel["market_jump"] = panel["is_market_jump"]
    required = {
        "sample_id",
        "pair_id",
        "session_id",
        "source_split",
        "checkpoint_fold",
        "event_regime",
        "scheduled_event",
        "market_jump",
        "source_5m_membership",
        "current_surface_flat",
        "target_surface_flat",
        "lp_embedding",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RQ4FoldPooledPredictionError(
            f"Fold-pooled panel is missing columns: {missing}"
        )
    if (
        len(panel) != int(unit["panel_row_count"])
        or panel["pair_id"].astype(str).nunique() != int(unit["panel_pair_count"])
        or panel["session_id"].astype(str).nunique() != int(unit["panel_session_count"])
        or panel["sample_id"].astype(str).duplicated().any()
        or panel["pair_id"].astype(str).duplicated().any()
        or set(panel["source_split"].astype(str)) != EXPECTED_SPLITS
        or set(panel["checkpoint_fold"].astype(str)) != {str(unit["fold"])}
    ):
        raise RQ4FoldPooledPredictionError("Fold-pooled panel coverage drift")
    expected_noise = noise_bank_profile_sha256(
        seed=int(unit["seed"]),
        fold=str(unit["fold"]),
        sample_ids=panel["sample_id"].astype(str).tolist(),
        draws=64,
    )
    if expected_noise != str(unit["noise_bank_profile_sha256"]):
        raise RQ4FoldPooledPredictionError("Frozen noise-bank profile drift")

    run = comparison.RunSpec(
        run_id=str(unit["prediction_unit_id"]),
        run_dir=Path(
            str(unit.get("source_root") or Path(unit["checkpoint_path"]).parent)
        ),
        model="wgan",
        tolerance_minutes=30,
        seed=int(unit["seed"]),
        checkpoint_path=Path(str(unit["checkpoint_path"])),
        text_ablation_mode="real_text",
        support_mask_mode="raw_joint",
        generator_current_input_mode="current_support_masked",
        metadata={
            "checkpoint_fold": str(unit["fold"]),
            "source_arm": str(unit["source_arm"]),
            "rq4_model": str(unit["model_name"]),
        },
    )
    panel_name = f"{unit['fold']}_rq4_fold_pooled_t30"
    predictions = _get_evaluator(unit)(run, panel_name, panel)
    samples, general_exclusions, _metric_exclusions = comparison.compute_sample_metrics(
        run,
        panel_name,
        panel,
        predictions,
        evaluate_embedded_atm_skew=False,
    )
    if not general_exclusions.empty or len(samples) != len(panel):
        counts = (
            general_exclusions.get("exclusion_code", pd.Series(dtype=str))
            .astype(str)
            .value_counts()
            .to_dict()
        )
        raise RQ4FoldPooledPredictionError(
            f"Prediction coverage failed: samples={len(samples)}, exclusions={counts}"
        )
    prediction_export = comparison._prediction_export_frame(
        run, panel_name, panel, predictions, samples
    )
    comparison._enforce_formal_run_coverage(
        run,
        panel_name,
        panel,
        samples,
        general_exclusions,
        prediction_export,
    )

    lineage_columns = [
        "sample_id",
        "source_split",
        "checkpoint_fold",
        "event_regime",
        "scheduled_event",
        "market_jump",
        "scheduled_event_ids",
        "market_jump_tiers",
        "source_5m_membership",
        "effective_origin_utc",
    ]
    lineage_columns = [column for column in lineage_columns if column in panel.columns]
    lineage = panel[lineage_columns].copy()
    prediction_export = prediction_export.merge(
        lineage, on="sample_id", how="left", validate="one_to_one"
    )
    prediction_export["model"] = str(unit["model_name"])
    prediction_export["effective_text_use"] = bool(unit["model_name"] == "film_cnn")
    prediction_export["source_arm"] = str(unit["source_arm"])
    prediction_export["checkpoint_fold"] = str(unit["fold"])
    prediction_export["checkpoint_sha256"] = str(unit["checkpoint_sha256"])
    prediction_export["panel_sha256"] = str(unit["panel_sha256"])
    prediction_export["noise_bank_profile_sha256"] = str(
        unit["noise_bank_profile_sha256"]
    )

    metrics = samples.copy()
    metrics["target_mae"] = pd.to_numeric(metrics["model_mae"], errors="raise")
    metrics["persistence_mae"] = pd.to_numeric(
        metrics["persistence_mae"], errors="raise"
    )
    metric_columns = [
        "sample_id",
        "pair_id",
        "session_id",
        "target_mae",
        "persistence_mae",
        "supported_cell_count",
        "supported_cell_fraction",
        "support_method",
        "support_grid_fingerprint",
    ]
    metrics = metrics[metric_columns].merge(
        lineage, on="sample_id", how="left", validate="one_to_one"
    )
    metrics["model"] = str(unit["model_name"])
    metrics["effective_text_use"] = bool(unit["model_name"] == "film_cnn")
    metrics["source_arm"] = str(unit["source_arm"])
    metrics["seed"] = int(unit["seed"])
    metrics["checkpoint_fold"] = str(unit["fold"])
    metrics["checkpoint_sha256"] = str(unit["checkpoint_sha256"])
    metrics["panel_sha256"] = str(unit["panel_sha256"])
    metrics["noise_bank_profile_sha256"] = str(unit["noise_bank_profile_sha256"])
    metrics["prediction_mc_samples"] = 64
    for column in ("target_mae", "persistence_mae", "supported_cell_count"):
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RQ4FoldPooledPredictionError(f"Non-finite metric: {column}")
    if bool((pd.to_numeric(metrics["supported_cell_count"]) <= 0).any()):
        raise RQ4FoldPooledPredictionError("Raw-joint support is empty")

    prediction_path, metrics_path, manifest_path = _artifact_paths(unit)
    prediction_export = prediction_export.sort_values("sample_id", kind="stable")
    metrics = metrics.sort_values("pair_id", kind="stable")
    _atomic_csv(prediction_export, prediction_path)
    _atomic_csv(metrics, metrics_path)
    _atomic_json(manifest_path, _manifest_payload(unit, prediction_path, metrics_path))
    _validate_complete_bundle(unit, prediction_path, metrics_path, manifest_path)
    return prediction_path, metrics_path, manifest_path


def _artifact_record(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_prediction_unit(
    *, unit: Mapping[str, Any], logical_device: int, resume: bool
) -> dict[str, Any]:
    """Evaluate one RQ1 checkpoint on one RQ4 fold-pooled panel."""

    frozen = _validate_unit(unit, logical_device)
    paths = _artifact_paths(frozen)
    state = _bundle_state(paths)
    if state == "partial":
        raise RQ4FoldPooledPredictionError(
            f"Partial prediction bundle requires audit: {frozen['prediction_unit_id']}"
        )
    if state == "complete":
        if not resume:
            raise RQ4FoldPooledPredictionError(
                f"Existing prediction bundle requires resume=True: {frozen['prediction_unit_id']}"
            )
        manifest = _validate_complete_bundle(frozen, *paths)
    else:
        paths = _run_prediction(frozen)
        manifest = _validate_complete_bundle(frozen, *paths)
    prediction_path, metrics_path, manifest_path = paths
    return {
        "noise_bank_profile_sha256": str(frozen["noise_bank_profile_sha256"]),
        "artifacts": [
            _artifact_record("predicted_surfaces", prediction_path),
            _artifact_record("pair_metrics", metrics_path),
            _artifact_record("prediction_manifest", manifest_path),
        ],
        "metadata": {
            "prediction_kind": PREDICTION_KIND,
            "model": str(frozen["model_name"]),
            "seed": int(frozen["seed"]),
            "checkpoint_fold": str(frozen["fold"]),
            "pair_count": int(frozen["panel_pair_count"]),
            "session_count": int(frozen["panel_session_count"]),
            "prediction_rows": int(manifest["prediction_row_count"]),
            "logical_cuda_device": 0,
        },
    }


prediction_worker = run_prediction_unit


__all__ = [
    "PREDICTION_KIND",
    "RQ4FoldPooledPredictionError",
    "noise_bank_profile_sha256",
    "payload_sha256",
    "prediction_worker",
    "run_prediction_unit",
    "sha256_file",
]
