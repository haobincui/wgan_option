"""Spawn-safe prediction callbacks for the Pure-CNN -> FiLM experiment.

This module deliberately imports only the Python standard library at module
import time.  The parallel scheduler imports :func:`run_prediction_unit` only
after its child initializer has restricted ``CUDA_VISIBLE_DEVICES``.  Pandas,
Torch, the model profiles, and the production evaluator are all imported
lazily inside the callback.

The callback supports the three frozen prediction cell types used by the
experiment:

* ``standard_test``;
* ``matched_checkpoint_intervention`` (zero/wrong text); and
* ``validation_trajectory``.

Every cell is fail-closed.  A complete bundle may be adopted only with
``resume=True`` and only after all artifact hashes and scientific lineage are
revalidated.  A partial bundle is never overwritten automatically.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence


class TextEffectPredictionWorkerError(RuntimeError):
    """Raised when a prediction cell violates its frozen contract."""


_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SUPPORTED_KINDS = {
    "standard_test",
    "matched_checkpoint_intervention",
    "validation_trajectory",
}
_PURE_PARENT_ARM = "pure_cnn_parent"
_PURE_CONTINUATION_ARM = "pure_cnn_continue_no_text"
_MATCHED_ARM = "film_lp_matched"
_VALIDATION_PANEL_CACHE: dict[tuple[str, str, str], Any] = {}
_CANONICAL_VALIDATION_CACHE: dict[str, Mapping[str, Mapping[str, Any]]] = {}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TextEffectPredictionWorkerError(
            "Prediction-cell payload is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA_RE.fullmatch(digest) is None:
        raise TextEffectPredictionWorkerError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_file(
    path_value: object,
    *,
    sha256: object | None = None,
    size_bytes: object | None = None,
    label: str,
) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file():
        raise TextEffectPredictionWorkerError(f"{label} is missing: {path}")
    if size_bytes is not None and int(size_bytes) != path.stat().st_size:
        raise TextEffectPredictionWorkerError(f"{label} size drift: {path}")
    if sha256 is not None:
        digest = _require_sha(sha256, f"{label} SHA")
        if sha256_file(path) != digest:
            raise TextEffectPredictionWorkerError(f"{label} SHA drift: {path}")
    return path


def _experiment_root(unit: Mapping[str, Any]) -> Path:
    for key in ("experiment_root", "output_dir", "artifact_root"):
        raw = str(unit.get(key) or "").strip()
        if raw:
            root = Path(raw).expanduser().resolve()
            if not root.is_dir():
                raise TextEffectPredictionWorkerError(
                    f"Prediction experiment root is missing: {root}"
                )
            return root
    raise TextEffectPredictionWorkerError(
        "Prediction unit requires experiment_root (or output_dir/artifact_root)"
    )


def _validate_callback_device(logical_device: int) -> None:
    if int(logical_device) != 0:
        raise TextEffectPredictionWorkerError(
            "Prediction workers must address the isolated GPU as logical cuda:0"
        )
    declared = os.environ.get("RQ3_LOGICAL_CUDA_DEVICE")
    if declared is not None and declared != "0":
        raise TextEffectPredictionWorkerError(
            "RQ3_LOGICAL_CUDA_DEVICE must be zero inside a prediction worker"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [value.strip() for value in visible.split(",") if value.strip()]
        if len(devices) != 1:
            raise TextEffectPredictionWorkerError(
                "Prediction child must see exactly one physical CUDA device"
            )


def _validate_unit(unit: Mapping[str, Any], logical_device: int) -> dict[str, Any]:
    payload = dict(unit)
    required = {
        "prediction_unit_id",
        "prediction_kind",
        "seed",
        "fold",
        "noise_bank_profile_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise TextEffectPredictionWorkerError(
            f"Prediction unit is missing fields: {missing}"
        )
    if str(payload["prediction_kind"]) not in _SUPPORTED_KINDS:
        raise TextEffectPredictionWorkerError(
            f"Unsupported prediction kind: {payload['prediction_kind']}"
        )
    if not str(payload["prediction_unit_id"]).strip():
        raise TextEffectPredictionWorkerError("prediction_unit_id must be non-empty")
    int(payload["seed"])
    if not str(payload["fold"]).strip():
        raise TextEffectPredictionWorkerError("fold must be non-empty")
    _require_sha(payload["noise_bank_profile_sha256"], "noise-bank profile")
    _require_file(
        payload["checkpoint_path"],
        sha256=payload["checkpoint_sha256"],
        size_bytes=payload.get("checkpoint_size_bytes"),
        label="Generator checkpoint",
    )
    kind = str(payload["prediction_kind"])
    if kind == "standard_test":
        kind_required = {"job_id", "arm"}
    elif kind == "matched_checkpoint_intervention":
        kind_required = {"job_id", "arm", "input_condition"}
    else:
        kind_required = {
            "job_id",
            "parent_job_id",
            "arm",
            "checkpoint_label",
            "epoch",
            "parent_checkpoint_path",
            "parent_checkpoint_sha256",
        }
    missing_kind = sorted(kind_required - set(payload))
    if missing_kind:
        raise TextEffectPredictionWorkerError(
            f"{kind} prediction unit is missing fields: {missing_kind}"
        )
    if kind == "validation_trajectory":
        _require_file(
            payload["parent_checkpoint_path"],
            sha256=payload["parent_checkpoint_sha256"],
            label="Trajectory parent checkpoint",
        )
    _validate_callback_device(logical_device)
    _experiment_root(payload)
    return payload


def _stage_description(arm: str) -> tuple[str, str, bool]:
    if arm == _PURE_PARENT_ARM:
        return "backbones", "backbones", False
    if arm == _PURE_CONTINUATION_ARM:
        return "pure_continuation", "pure_continuation", True
    if arm.startswith("film_"):
        return "film_continuations", "film_continuations", True
    raise TextEffectPredictionWorkerError(f"Unknown experiment arm: {arm}")


@contextmanager
def _stage_runtime(
    main: Any, *, main_root: Path, arm: str
) -> Iterator[tuple[Any, Any, Any]]:
    """Install the branch-local profile and expose (direct, core, stage root)."""

    stage_key, stage_name, use_graft = _stage_description(arm)
    stage_root = main._stage_roots(main_root)[stage_key]
    if not stage_root.is_dir():
        raise TextEffectPredictionWorkerError(
            f"Prediction stage root is missing: {stage_root}"
        )
    with ExitStack() as stack:
        if arm in {_PURE_PARENT_ARM, _PURE_CONTINUATION_ARM}:
            stack.enter_context(
                main._pure_stage_profile(
                    main_root=main_root,
                    stage_name=stage_name,
                    arm=arm,
                    use_graft=use_graft,
                )
            )
            stack.enter_context(main.pure._runtime_profile())
        else:
            stack.enter_context(
                main._film_stage_profile(main_root=main_root, stage_name=stage_name)
            )
            stack.enter_context(main.film_text.film_text_profile())
            stack.enter_context(main.film_text.multiseed.multiseed_profile())
        yield main.direct, main.core, stage_root


def _find_internal_job(
    direct: Any,
    stage_root: Path,
    *,
    seed: int,
    fold: str,
    arm: str,
) -> dict[str, Any]:
    registry = direct.read_registry(stage_root)
    matches = [
        dict(job)
        for job in registry.get("jobs", [])
        if int(job["seed"]) == int(seed)
        and str(job["fold"]) == str(fold)
        and str(job["arm"]) == str(arm)
    ]
    if len(matches) != 1:
        raise TextEffectPredictionWorkerError(
            f"Expected one internal job for seed={seed}, fold={fold}, arm={arm}"
        )
    return matches[0]


def _checkpoint_from_unit(
    unit: Mapping[str, Any],
    *,
    direct: Any,
    stage_root: Path,
    internal_job: Mapping[str, Any],
) -> dict[str, str]:
    checkpoints = direct._checkpoint_map(stage_root)
    frozen = dict(checkpoints[str(internal_job["job_id"])])
    path = str(Path(str(unit["checkpoint_path"])).expanduser().resolve())
    digest = _require_sha(unit["checkpoint_sha256"], "checkpoint")
    if str(Path(frozen["path"]).resolve()) != path or str(frozen["sha256"]) != digest:
        raise TextEffectPredictionWorkerError(
            f"Prediction checkpoint disagrees with stage allowlist: {unit['prediction_unit_id']}"
        )
    return {"path": path, "sha256": digest, "role": str(frozen["role"])}


def _expected_test_pairs(main: Any, fold: str) -> int:
    counts = main._expected_fold_counts()
    if str(fold) not in counts:
        raise TextEffectPredictionWorkerError(f"Unknown rolling fold: {fold}")
    return int(counts[str(fold)][4])


def _verify_declared_noise_profile(
    unit: Mapping[str, Any], sample_ids: Sequence[object]
) -> str:
    from scripts.rq3.news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction import (
        noise_bank_profile_sha256,
    )

    expected = noise_bank_profile_sha256(
        split=str(
            unit.get("split")
            or (
                "validation"
                if unit["prediction_kind"] == "validation_trajectory"
                else "test"
            )
        ),
        seed=int(unit["seed"]),
        fold=str(unit["fold"]),
        sample_ids=[str(value) for value in sample_ids],
        draws=int(
            unit.get("mc_samples")
            or (16 if unit["prediction_kind"] == "validation_trajectory" else 64)
        ),
        tolerance_minutes=int(unit.get("tolerance_minutes", 5)),
    )
    observed = _require_sha(unit["noise_bank_profile_sha256"], "noise-bank profile")
    if observed != expected:
        raise TextEffectPredictionWorkerError(
            f"Frozen noise-bank profile drift: {unit['prediction_unit_id']}"
        )
    return observed


def _artifact_record(role: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TextEffectPredictionWorkerError(f"Prediction artifact is missing: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _select_artifacts(
    unit: Mapping[str, Any], available: Mapping[str, Path]
) -> list[dict[str, Any]]:
    raw = unit.get("expected_artifact_roles")
    if raw is None or raw == "":
        roles = tuple(available)
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [value.strip() for value in raw.split(",") if value.strip()]
        roles = tuple(map(str, parsed))
    else:
        roles = tuple(map(str, raw))
    if not roles or len(set(roles)) != len(roles):
        raise TextEffectPredictionWorkerError(
            "expected_artifact_roles must be non-empty and unique"
        )
    missing = sorted(set(roles) - set(available))
    if missing:
        raise TextEffectPredictionWorkerError(
            f"Unknown requested prediction artifact roles: {missing}"
        )
    return [_artifact_record(role, available[role]) for role in sorted(roles)]


def _existing_bundle_state(paths: Sequence[Path]) -> str:
    exists = [path.is_file() for path in paths]
    if all(exists):
        return "complete"
    if any(exists):
        return "partial"
    return "absent"


def _write_core_pair_evidence(
    direct: Any,
    core: Any,
    manifest_path: Path,
    evidence: Any,
) -> tuple[dict[str, Any], Path]:
    evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
    core._write_dataframe_csv(evidence_path, evidence)
    payload = core.read_json(manifest_path)
    payload.pop("payload_sha256", None)
    payload.update(
        pair_metrics_path=str(evidence_path.resolve()),
        pair_metrics_sha256=sha256_file(evidence_path),
        pair_metrics_row_count=int(len(evidence)),
    )
    payload["payload_sha256"] = direct.payload_sha256(payload)
    core.write_json(manifest_path, payload)
    return payload, evidence_path


def _run_core_test_cell(
    unit: Mapping[str, Any], *, resume: bool, intervention: bool
) -> tuple[dict[str, Path], dict[str, Any]]:
    # Delayed import is required for CUDA isolation under multiprocessing spawn.
    from scripts.rq3 import (
        news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as main,
    )

    main_root = _experiment_root(unit)
    source_arm = _MATCHED_ARM if intervention else str(unit["arm"])
    with _stage_runtime(main, main_root=main_root, arm=source_arm) as (
        direct,
        core,
        stage_root,
    ):
        internal_job = _find_internal_job(
            direct,
            stage_root,
            seed=int(unit["seed"]),
            fold=str(unit["fold"]),
            arm=source_arm,
        )
        checkpoint = _checkpoint_from_unit(
            unit,
            direct=direct,
            stage_root=stage_root,
            internal_job=internal_job,
        )
        job = dict(internal_job)
        job["gpu_id"] = 0
        if intervention:
            condition = str(unit.get("input_condition", ""))
            intervention_arm = {
                "zero_input": "film_lp_matched__zero_input",
                "wrong_input": "film_lp_matched__wrong_input",
            }.get(condition)
            if intervention_arm is None:
                raise TextEffectPredictionWorkerError(
                    f"Unknown matched-checkpoint intervention: {condition!r}"
                )
            declared_overlay = str(unit.get("input_overlay_arm") or "")
            accepted = {
                "zero_input": {"film_zero_text", intervention_arm},
                "wrong_input": {"film_lp_independent_wrong", intervention_arm},
            }[condition]
            if declared_overlay and declared_overlay not in accepted:
                raise TextEffectPredictionWorkerError(
                    f"Intervention overlay alias drift: {declared_overlay}"
                )
            job["arm"] = intervention_arm

        expected_pairs = _expected_test_pairs(main, str(unit["fold"]))
        # Entering direct_profile binds the shared evaluator's paths, model
        # profile, overlay semantics, and inference determinism contract.
        with direct.direct_profile():
            panel = core._panel_with_overlay(stage_root, job)
            declared_noise = _verify_declared_noise_profile(
                unit, panel["sample_id"].astype(str).tolist()
            )
            prediction_path = core._prediction_path(stage_root, job)
            manifest_path = core._prediction_job_manifest_path(stage_root, job)
            evidence_path = manifest_path.with_suffix(".pair_metrics.csv")
            state = _existing_bundle_state(
                (prediction_path, manifest_path, evidence_path)
            )
            if state == "partial":
                raise TextEffectPredictionWorkerError(
                    f"Partial prediction bundle requires audit: {unit['prediction_unit_id']}"
                )
            if state == "complete":
                if not resume:
                    raise TextEffectPredictionWorkerError(
                        f"Existing prediction bundle requires resume=True: {unit['prediction_unit_id']}"
                    )
                payload = core._validate_prediction_job(
                    stage_root,
                    job,
                    checkpoint,
                    expected_pairs=expected_pairs,
                )
            else:
                core._configure_prediction_determinism(int(unit["seed"]))
                payload, evidence = core._evaluate_prediction_job(
                    stage_root,
                    job,
                    checkpoint,
                    expected_pairs=expected_pairs,
                )
                payload, evidence_path = _write_core_pair_evidence(
                    direct, core, manifest_path, evidence
                )
                payload = core._validate_prediction_job(
                    stage_root,
                    job,
                    checkpoint,
                    expected_pairs=expected_pairs,
                )
            if int(payload.get("pair_metrics_row_count", -1)) != expected_pairs:
                raise TextEffectPredictionWorkerError(
                    f"Prediction pair coverage drift: {unit['prediction_unit_id']}"
                )
            available = {
                "prediction": prediction_path,
                "core_prediction_manifest": manifest_path,
                "pair_metrics": evidence_path,
            }
            metadata = {
                "stage_root": str(stage_root),
                "internal_job_id": str(internal_job["job_id"]),
                "source_arm": source_arm,
                "evaluated_arm": str(job["arm"]),
                "pair_count": expected_pairs,
                "core_noise_bank_profile_sha256": str(
                    payload["noise_bank_profile_sha256"]
                ),
                "declared_shared_noise_bank_profile_sha256": declared_noise,
            }
            return available, metadata


def _validation_panel(main: Any, unit: Mapping[str, Any], main_root: Path) -> Any:
    import pandas as pd

    explicit = str(unit.get("validation_panel_path") or "").strip()
    if explicit:
        path = _require_file(
            explicit,
            sha256=unit.get("validation_panel_sha256"),
            label="Frozen validation panel",
        )
        panel = pd.read_csv(path, low_memory=False)
    else:
        root_key = str(main_root)
        canonical = _CANONICAL_VALIDATION_CACHE.get(root_key)
        if canonical is None:
            config = main._load_frozen_config(main_root / "resolved_config.yaml")
            canonical = main._canonical_validation_rows(config, main_root)
            _CANONICAL_VALIDATION_CACHE[root_key] = canonical
        cache_key = (root_key, str(unit["fold"]), str(unit["arm"]))
        panel = _VALIDATION_PANEL_CACHE.get(cache_key)
        if panel is None:
            panel = main._validation_panel(
                main_root,
                canonical,
                fold=str(unit["fold"]),
                arm=str(unit["arm"]),
            )
            _VALIDATION_PANEL_CACHE[cache_key] = panel
        panel = panel.copy()
    required = {"pair_id", "session_id", "sample_id", "lp_embedding"}
    if panel.empty or not required.issubset(panel.columns):
        raise TextEffectPredictionWorkerError(
            f"Validation panel contract drift: {unit['prediction_unit_id']}"
        )
    if panel["pair_id"].astype(str).duplicated().any():
        raise TextEffectPredictionWorkerError("Validation panel duplicates pair IDs")
    return panel


def _trajectory_output_paths(
    unit: Mapping[str, Any], main_root: Path
) -> tuple[Path, Path]:
    explicit = str(unit.get("pair_metrics_path") or "").strip()
    if explicit:
        pair_path = Path(explicit).expanduser().resolve()
        if pair_path != main_root and main_root not in pair_path.parents:
            raise TextEffectPredictionWorkerError(
                f"Validation trajectory output escaped experiment root: {pair_path}"
            )
    else:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(unit["prediction_unit_id"])).strip(
            "._"
        )
        suffix = hashlib.sha256(
            str(unit["prediction_unit_id"]).encode("utf-8")
        ).hexdigest()[:12]
        pair_path = (
            main_root
            / "evaluation/validation_trajectory_cells"
            / f"{safe[:120]}__{suffix}.csv.gz"
        )
    manifest_path = Path(str(pair_path) + ".manifest.json")
    return pair_path, manifest_path


def _validate_trajectory_bundle(
    unit: Mapping[str, Any], pair_path: Path, manifest_path: Path, expected_rows: int
) -> dict[str, Any]:
    import pandas as pd

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextEffectPredictionWorkerError(
            f"Invalid trajectory manifest: {manifest_path}"
        ) from exc
    unsigned = dict(payload)
    observed_payload_sha = str(unsigned.pop("payload_sha256", ""))
    if observed_payload_sha != _canonical_sha256(unsigned):
        raise TextEffectPredictionWorkerError("Trajectory manifest payload SHA drift")
    if (
        payload.get("kind") != "pure_cnn_film_text_effect_validation_trajectory_cell_v1"
        or payload.get("prediction_unit_id") != str(unit["prediction_unit_id"])
        or payload.get("checkpoint_sha256") != str(unit["checkpoint_sha256"])
        or payload.get("noise_bank_profile_sha256")
        != str(unit["noise_bank_profile_sha256"])
        or payload.get("pair_metrics_path") != str(pair_path.resolve())
        or payload.get("pair_metrics_sha256") != sha256_file(pair_path)
        or int(payload.get("pair_metrics_row_count", -1)) != int(expected_rows)
    ):
        raise TextEffectPredictionWorkerError("Trajectory manifest lineage drift")
    frame = pd.read_csv(pair_path)
    required = {
        "prediction_unit_id",
        "pair_id",
        "target_mae",
        "checkpoint_sha256",
        "noise_bank_profile_sha256",
    }
    if (
        len(frame) != int(expected_rows)
        or not required.issubset(frame.columns)
        or frame["pair_id"].astype(str).duplicated().any()
        or set(frame["prediction_unit_id"].astype(str))
        != {str(unit["prediction_unit_id"])}
        or set(frame["checkpoint_sha256"].astype(str))
        != {str(unit["checkpoint_sha256"])}
        or set(frame["noise_bank_profile_sha256"].astype(str))
        != {str(unit["noise_bank_profile_sha256"])}
    ):
        raise TextEffectPredictionWorkerError("Trajectory pair evidence drift")
    return payload


def _run_validation_trajectory(
    unit: Mapping[str, Any], *, resume: bool
) -> tuple[dict[str, Path], dict[str, Any]]:
    # Delayed import is required for CUDA isolation under multiprocessing spawn.
    from scripts.rq3 import (
        news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as main,
    )

    main_root = _experiment_root(unit)
    arm = str(unit["arm"])
    pair_path, manifest_path = _trajectory_output_paths(unit, main_root)
    panel = _validation_panel(main, unit, main_root)
    expected_rows = len(panel)
    declared_noise = _verify_declared_noise_profile(
        unit, panel["sample_id"].astype(str).tolist()
    )
    state = _existing_bundle_state((pair_path, manifest_path))
    if state == "partial":
        raise TextEffectPredictionWorkerError(
            f"Partial trajectory bundle requires audit: {unit['prediction_unit_id']}"
        )
    if state == "complete":
        if not resume:
            raise TextEffectPredictionWorkerError(
                f"Existing trajectory bundle requires resume=True: {unit['prediction_unit_id']}"
            )
        _validate_trajectory_bundle(unit, pair_path, manifest_path, expected_rows)
    else:
        with _stage_runtime(main, main_root=main_root, arm=arm) as (
            direct,
            _core,
            stage_root,
        ):
            runtime_unit = dict(unit)
            runtime_unit["stage_root"] = str(stage_root)
            internal_job = _find_internal_job(
                direct,
                stage_root,
                seed=int(unit["seed"]),
                fold=str(unit["fold"]),
                arm=arm,
            )
            expected_logical_job = main.continuation_job_id(
                int(unit["seed"]), str(unit["fold"]), arm
            )
            expected_parent_job = main.parent_job_id(
                int(unit["seed"]), str(unit["fold"])
            )
            if (
                str(unit["job_id"]) != expected_logical_job
                or str(unit["parent_job_id"]) != expected_parent_job
            ):
                raise TextEffectPredictionWorkerError(
                    f"Trajectory logical parent/job lineage drift: {unit['prediction_unit_id']}"
                )
            label = str(unit["checkpoint_label"])
            if label == "epoch_0":
                role = "generator_initial_epoch0"
                expected_epoch = 0
            elif label == "best":
                role = "generator_best_learned"
                expected_epoch = int(unit["epoch"])
            else:
                match = re.fullmatch(r"epoch_(1|5|10|20|30)", label)
                if match is None:
                    raise TextEffectPredictionWorkerError(
                        f"Unsupported trajectory checkpoint label: {label}"
                    )
                expected_epoch = int(match.group(1))
                role = f"generator_validation_epoch_{expected_epoch:04d}"
            if int(unit["epoch"]) != expected_epoch:
                raise TextEffectPredictionWorkerError(
                    f"Trajectory epoch/label drift: {unit['prediction_unit_id']}"
                )
            status = main._read_json(
                direct._status_path(stage_root, str(internal_job["job_id"]))
            )
            artifact = direct._artifact(status, role)
            if str(Path(str(artifact["path"])).resolve()) != str(
                Path(str(unit["checkpoint_path"])).resolve()
            ) or str(artifact["sha256"]) != str(unit["checkpoint_sha256"]):
                raise TextEffectPredictionWorkerError(
                    f"Trajectory checkpoint is not a frozen stage artifact: {unit['prediction_unit_id']}"
                )
            frame = main._evaluate_validation_unit(runtime_unit, panel, gpu_id=0)
        if len(frame) != expected_rows:
            raise TextEffectPredictionWorkerError(
                f"Validation trajectory coverage drift: {unit['prediction_unit_id']}"
            )
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        main.core._write_dataframe_csv(pair_path, frame, gzip=True)
        payload = {
            "schema_version": 1,
            "kind": "pure_cnn_film_text_effect_validation_trajectory_cell_v1",
            "prediction_unit_id": str(unit["prediction_unit_id"]),
            "job_id": str(unit["job_id"]),
            "parent_job_id": str(unit["parent_job_id"]),
            "seed": int(unit["seed"]),
            "fold": str(unit["fold"]),
            "arm": arm,
            "checkpoint_label": str(unit["checkpoint_label"]),
            "epoch": int(unit["epoch"]),
            "checkpoint_path": str(Path(unit["checkpoint_path"]).resolve()),
            "checkpoint_sha256": str(unit["checkpoint_sha256"]),
            "noise_bank_profile_sha256": declared_noise,
            "pair_metrics_path": str(pair_path.resolve()),
            "pair_metrics_sha256": sha256_file(pair_path),
            "pair_metrics_row_count": expected_rows,
            "logical_cuda_device": 0,
        }
        payload["payload_sha256"] = _canonical_sha256(payload)
        main._write_json(manifest_path, payload)
        _validate_trajectory_bundle(unit, pair_path, manifest_path, expected_rows)
    return {
        "pair_metrics": pair_path,
        "trajectory_manifest": manifest_path,
    }, {
        "stage_root": str(main._stage_roots(main_root)[_stage_description(arm)[0]]),
        "pair_count": expected_rows,
        "checkpoint_label": str(unit["checkpoint_label"]),
        "declared_shared_noise_bank_profile_sha256": declared_noise,
    }


def _run_standard_test(
    unit: Mapping[str, Any], *, resume: bool
) -> tuple[dict[str, Path], dict[str, Any]]:
    return _run_core_test_cell(unit, resume=resume, intervention=False)


def _run_matched_intervention(
    unit: Mapping[str, Any], *, resume: bool
) -> tuple[dict[str, Path], dict[str, Any]]:
    return _run_core_test_cell(unit, resume=resume, intervention=True)


def run_prediction_unit(
    *, unit: Mapping[str, Any], logical_device: int, resume: bool
) -> dict[str, Any]:
    """Run one spawn-isolated prediction cell and return SHA-bound artifacts.

    This is the importable callback passed to
    ``run_parallel_prediction_units(..., worker_entrypoint=...)``.
    """

    frozen = _validate_unit(unit, logical_device)
    kind = str(frozen["prediction_kind"])
    if kind == "standard_test":
        available, metadata = _run_standard_test(frozen, resume=bool(resume))
    elif kind == "matched_checkpoint_intervention":
        available, metadata = _run_matched_intervention(frozen, resume=bool(resume))
    else:
        available, metadata = _run_validation_trajectory(frozen, resume=bool(resume))
    return {
        "noise_bank_profile_sha256": str(frozen["noise_bank_profile_sha256"]),
        "artifacts": _select_artifacts(frozen, available),
        "metadata": {
            "prediction_kind": kind,
            "logical_cuda_device": 0,
            **metadata,
        },
    }


# Short alias useful for declarative worker manifests.
prediction_worker = run_prediction_unit


__all__ = [
    "TextEffectPredictionWorkerError",
    "prediction_worker",
    "run_prediction_unit",
    "sha256_file",
]
