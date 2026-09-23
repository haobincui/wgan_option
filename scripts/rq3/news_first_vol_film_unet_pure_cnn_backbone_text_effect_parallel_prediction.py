"""Safe multi-process prediction scheduling for the backbone text experiment.

The helper is intentionally evaluator-agnostic.  A caller supplies an
importable ``module:function`` worker entrypoint which wraps the existing core
evaluator.  Each spawned child sees exactly one physical GPU through
``CUDA_VISIBLE_DEVICES`` and must use logical CUDA device zero.  The parent
owns scheduling, per-cell SHA manifests, fail-stop behaviour, and resume
validation; it never changes the frozen Monte-Carlo noise profile.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import threading
from typing import Any

from scripts.rq3.news_first_vol_experiment_supervisor_helpers import (
    SupervisorLock,
    append_stage_journal,
    atomic_write_json,
)


class ParallelPredictionError(RuntimeError):
    """Raised when scheduling or prediction artifacts violate the contract."""


PREDICTION_KINDS = {
    "standard_test",
    "matched_checkpoint_intervention",
    "validation_trajectory",
    "rq4_fold_pooled",
}
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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
        raise ParallelPredictionError(
            "Prediction contract is not canonical JSON"
        ) from exc


def payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    unsigned["payload_sha256"] = payload_sha256(unsigned)
    return unsigned


def _read_signed(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParallelPredictionError(f"Invalid prediction manifest: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ParallelPredictionError(f"Prediction manifest is not a mapping: {path}")
    payload = dict(raw)
    observed = str(payload.pop("payload_sha256", ""))
    if observed != payload_sha256(payload):
        raise ParallelPredictionError(f"Prediction manifest payload SHA drift: {path}")
    payload["payload_sha256"] = observed
    return payload


def _require_sha(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA_RE.fullmatch(digest) is None:
        raise ParallelPredictionError(f"{label} must be a lowercase SHA-256")
    return digest


def _safe_unit_id(value: object) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ParallelPredictionError("prediction_unit_id must be non-empty")
    safe = _SAFE_ID_RE.sub("_", identifier).strip("._")
    suffix = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:100]}__{suffix}"


def _unit_contract(unit: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "prediction_unit_id",
        "prediction_kind",
        "seed",
        "fold",
        "noise_bank_profile_sha256",
    }
    missing = sorted(required - set(unit))
    if missing:
        raise ParallelPredictionError(f"Prediction unit is missing fields: {missing}")
    kind = str(unit["prediction_kind"])
    if kind not in PREDICTION_KINDS:
        raise ParallelPredictionError(f"Unsupported prediction_kind: {kind}")
    _require_sha(unit["noise_bank_profile_sha256"], "noise-bank profile")
    # Runtime attempt paths and device aliases may not redefine the scientific
    # cell.  Everything else, including checkpoint and noise lineage, is bound.
    excluded = {
        "physical_gpu_id",
        "logical_device",
        "resume",
        "attempt",
        "worker_pid",
    }
    scientific = {
        key: value for key, value in dict(unit).items() if key not in excluded
    }
    payload = {
        "schema_version": 1,
        "kind": "backbone_text_effect_prediction_unit_contract_v1",
        "scientific_unit": scientific,
        "logical_cuda_device": 0,
    }
    payload["unit_contract_sha256"] = payload_sha256(payload)
    return payload


def assign_units_to_physical_gpus(
    units: Sequence[Mapping[str, Any]],
    *,
    gpu_ids: Sequence[int] = (0, 1),
    group_fields: Sequence[str] = ("seed", "fold"),
) -> list[dict[str, Any]]:
    """Assign stable groups round-robin so one group never crosses GPUs."""

    physical = tuple(map(int, gpu_ids))
    if not physical or len(set(physical)) != len(physical):
        raise ParallelPredictionError("gpu_ids must be non-empty and unique")
    rows = [dict(unit) for unit in units]
    if not rows:
        raise ParallelPredictionError("Prediction unit list is empty")
    seen: set[str] = set()
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        identifier = str(row.get("prediction_unit_id", "")).strip()
        if not identifier or identifier in seen:
            raise ParallelPredictionError("Prediction unit IDs are empty or duplicated")
        seen.add(identifier)
        missing = [field for field in group_fields if field not in row]
        if missing:
            raise ParallelPredictionError(f"GPU group fields are missing: {missing}")
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    assignments = {
        key: physical[index % len(physical)] for index, key in enumerate(sorted(groups))
    }
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        gpu = assignments[key]
        for row in sorted(
            groups[key], key=lambda value: str(value["prediction_unit_id"])
        ):
            declared = row.get("physical_gpu_id")
            if declared is not None and int(declared) != gpu:
                raise ParallelPredictionError(
                    f"Frozen physical GPU mapping drift: {row['prediction_unit_id']}"
                )
            row["physical_gpu_id"] = gpu
            row["logical_device"] = 0
            output.append(row)
    return output


def validate_shared_noise_lineage(units: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Require one frozen profile for every declared noise-bank namespace."""

    profiles: dict[str, str] = {}
    for unit in units:
        namespace = str(unit.get("noise_bank_namespace", "")).strip()
        if not namespace:
            raise ParallelPredictionError("Every unit requires noise_bank_namespace")
        digest = _require_sha(unit.get("noise_bank_profile_sha256"), "noise profile")
        previous = profiles.setdefault(namespace, digest)
        if previous != digest:
            raise ParallelPredictionError(
                f"MC noise lineage drift within namespace: {namespace}"
            )
    return dict(sorted(profiles.items()))


def prediction_cell_manifest_path(
    control_dir: str | Path, prediction_unit_id: object
) -> Path:
    return (
        Path(control_dir).resolve()
        / "prediction_cells"
        / f"{_safe_unit_id(prediction_unit_id)}.json"
    )


def _executor_contract(worker_entrypoint: str, start_method: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", worker_entrypoint):
        raise ParallelPredictionError(
            "worker_entrypoint must be an importable module:function"
        )
    if start_method not in {"spawn", "forkserver"}:
        raise ParallelPredictionError("CUDA prediction requires spawn or forkserver")
    payload = {
        "schema_version": 1,
        "kind": "backbone_text_effect_prediction_executor_contract_v1",
        "worker_entrypoint": worker_entrypoint,
        "multiprocessing_start_method": start_method,
        "child_cuda_visible_devices": "one_physical_gpu",
        "child_logical_cuda_device": 0,
        "mc_noise_policy": "caller_frozen_profile_pass_through_unchanged",
    }
    payload["executor_contract_sha256"] = payload_sha256(payload)
    return payload


def _artifact_rows(
    result: Mapping[str, Any], *, artifact_root: Path
) -> list[dict[str, Any]]:
    raw_rows = result.get("artifacts")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ParallelPredictionError("Worker result artifacts must be a sequence")
    rows: list[dict[str, Any]] = []
    roles: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ParallelPredictionError("Worker artifact must be a mapping")
        role = str(raw.get("role", "")).strip()
        path = Path(str(raw.get("path", ""))).expanduser().resolve()
        if not role or role in roles:
            raise ParallelPredictionError(
                "Worker artifact roles are empty or duplicated"
            )
        roles.add(role)
        if path != artifact_root and artifact_root not in path.parents:
            raise ParallelPredictionError(f"Prediction artifact escaped root: {path}")
        if not path.is_file():
            raise ParallelPredictionError(f"Prediction artifact is missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        declared_size = int(raw.get("size_bytes", -1))
        declared_sha = _require_sha(raw.get("sha256"), f"{role} artifact SHA")
        if size != declared_size or digest != declared_sha:
            raise ParallelPredictionError(f"Prediction artifact SHA/size drift: {path}")
        rows.append(
            {
                "role": role,
                "path": str(path),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if not rows:
        raise ParallelPredictionError("Worker returned no prediction artifacts")
    return sorted(rows, key=lambda row: row["role"])


def _validate_cell_manifest(
    path: Path,
    *,
    unit: Mapping[str, Any],
    executor: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    payload = _read_signed(path)
    contract = _unit_contract(unit)
    if (
        payload.get("kind") != "backbone_text_effect_prediction_cell_manifest_v1"
        or payload.get("prediction_unit_id") != str(unit["prediction_unit_id"])
        or payload.get("unit_contract_sha256") != contract["unit_contract_sha256"]
        or payload.get("executor_contract_sha256")
        != executor["executor_contract_sha256"]
        or int(payload.get("logical_device", -1)) != 0
        or str(payload.get("noise_bank_profile_sha256"))
        != str(unit["noise_bank_profile_sha256"])
    ):
        raise ParallelPredictionError(
            f"Prediction cell manifest contract drift: {path}"
        )
    _artifact_rows(payload, artifact_root=artifact_root)
    return payload


def _worker_initializer(physical_gpu_id: int) -> None:
    # This runs before the evaluator module (and therefore torch) is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    os.environ["RQ3_LOGICAL_CUDA_DEVICE"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _load_entrypoint(value: str) -> Any:
    module_name, function_name = value.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ParallelPredictionError(f"Worker entrypoint is not callable: {value}")
    return function


def _worker_dispatch(task: Mapping[str, Any]) -> dict[str, Any]:
    physical = int(task["physical_gpu_id"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != str(physical):
        raise ParallelPredictionError(
            f"Child GPU isolation drift: physical={physical}, visible={visible!r}"
        )
    if os.environ.get("RQ3_LOGICAL_CUDA_DEVICE") != "0":
        raise ParallelPredictionError("Child logical CUDA device must be zero")
    function = _load_entrypoint(str(task["worker_entrypoint"]))
    result = function(
        unit=dict(task["unit"]),
        logical_device=0,
        resume=bool(task["resume"]),
    )
    if not isinstance(result, Mapping):
        raise ParallelPredictionError("Prediction worker must return a mapping")
    payload = dict(result)
    payload["prediction_unit_id"] = str(task["unit"]["prediction_unit_id"])
    payload["physical_gpu_id"] = physical
    payload["logical_device"] = 0
    payload["cuda_visible_devices"] = visible
    payload["worker_pid"] = os.getpid()
    return payload


def _publish_worker_result(
    result: Mapping[str, Any],
    *,
    unit: Mapping[str, Any],
    executor: Mapping[str, Any],
    control_dir: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    identifier = str(unit["prediction_unit_id"])
    if result.get("prediction_unit_id") != identifier:
        raise ParallelPredictionError(f"Worker returned the wrong cell: {identifier}")
    if int(result.get("logical_device", -1)) != 0:
        raise ParallelPredictionError(
            f"Worker used a nonzero logical device: {identifier}"
        )
    physical = int(unit["physical_gpu_id"])
    if int(result.get("physical_gpu_id", -1)) != physical or str(
        result.get("cuda_visible_devices")
    ) != str(physical):
        raise ParallelPredictionError(
            f"Worker physical GPU lineage drift: {identifier}"
        )
    noise = _require_sha(
        result.get("noise_bank_profile_sha256"), "worker noise profile"
    )
    if noise != str(unit["noise_bank_profile_sha256"]):
        raise ParallelPredictionError(
            f"Worker changed the MC noise lineage: {identifier}"
        )
    artifacts = _artifact_rows(result, artifact_root=artifact_root)
    expected_roles = tuple(map(str, unit.get("expected_artifact_roles", ())))
    if expected_roles and {row["role"] for row in artifacts} != set(expected_roles):
        raise ParallelPredictionError(
            f"Worker artifact role universe drift: {identifier}"
        )
    contract = _unit_contract(unit)
    payload = _signed(
        {
            "schema_version": 1,
            "kind": "backbone_text_effect_prediction_cell_manifest_v1",
            "prediction_unit_id": identifier,
            "prediction_kind": str(unit["prediction_kind"]),
            "unit_contract_sha256": contract["unit_contract_sha256"],
            "executor_contract_sha256": executor["executor_contract_sha256"],
            "physical_gpu_id": physical,
            "logical_device": 0,
            "cuda_visible_devices": str(physical),
            "worker_pid": int(result["worker_pid"]),
            "noise_bank_profile_sha256": noise,
            "artifacts": artifacts,
            "metadata": dict(result.get("metadata") or {}),
        }
    )
    path = prediction_cell_manifest_path(control_dir, identifier)
    atomic_write_json(path, payload)
    return _validate_cell_manifest(
        path, unit=unit, executor=executor, artifact_root=artifact_root
    )


def _workers_by_gpu(
    workers_per_gpu: int | Mapping[int, int], gpu_ids: Sequence[int]
) -> dict[int, int]:
    if isinstance(workers_per_gpu, Mapping):
        result = {int(gpu): int(workers_per_gpu.get(int(gpu), 0)) for gpu in gpu_ids}
    else:
        result = {int(gpu): int(workers_per_gpu) for gpu in gpu_ids}
    if any(value <= 0 for value in result.values()):
        raise ParallelPredictionError("Every GPU requires at least one worker")
    return result


def _run_gpu_queue(
    *,
    physical_gpu_id: int,
    units: Sequence[Mapping[str, Any]],
    workers: int,
    worker_entrypoint: str,
    resume: bool,
    stop_event: threading.Event,
    on_result: Any | None = None,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    pending_units = deque(dict(unit) for unit in units)
    results: list[dict[str, Any]] = []
    executor = ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(int(physical_gpu_id),),
    )
    futures: dict[Any, dict[str, Any]] = {}
    try:
        while pending_units and len(futures) < workers and not stop_event.is_set():
            unit = pending_units.popleft()
            task = {
                "physical_gpu_id": int(physical_gpu_id),
                "worker_entrypoint": worker_entrypoint,
                "unit": unit,
                "resume": bool(resume),
            }
            futures[executor.submit(_worker_dispatch, task)] = unit
        while futures:
            completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in completed:
                unit = futures.pop(future)
                try:
                    result = dict(future.result())
                    if on_result is not None:
                        on_result(result)
                    results.append(result)
                except BaseException:
                    stop_event.set()
                    for remaining in futures:
                        remaining.cancel()
                    raise
                if pending_units and not stop_event.is_set():
                    next_unit = pending_units.popleft()
                    task = {
                        "physical_gpu_id": int(physical_gpu_id),
                        "worker_entrypoint": worker_entrypoint,
                        "unit": next_unit,
                        "resume": bool(resume),
                    }
                    futures[executor.submit(_worker_dispatch, task)] = next_unit
        return results
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _validate_execution_manifest(
    path: Path,
    *,
    units: Sequence[Mapping[str, Any]],
    executor: Mapping[str, Any],
    control_dir: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    payload = _read_signed(path)
    expected_ids = {str(unit["prediction_unit_id"]) for unit in units}
    if (
        payload.get("kind") != "backbone_text_effect_parallel_prediction_manifest_v1"
        or payload.get("executor_contract_sha256")
        != executor["executor_contract_sha256"]
        or set(payload.get("prediction_unit_ids") or ()) != expected_ids
    ):
        raise ParallelPredictionError("Parallel prediction execution manifest drift")
    unit_by_id = {str(unit["prediction_unit_id"]): unit for unit in units}
    cell_rows = payload.get("cell_manifests")
    if not isinstance(cell_rows, list) or len(cell_rows) != len(units):
        raise ParallelPredictionError("Parallel prediction cell manifest count drift")
    for row in cell_rows:
        identifier = str(row["prediction_unit_id"])
        cell_path = prediction_cell_manifest_path(control_dir, identifier)
        if (
            identifier not in unit_by_id
            or not cell_path.is_file()
            or sha256_file(cell_path) != str(row["sha256"])
        ):
            raise ParallelPredictionError("Parallel prediction cell binding drift")
        _validate_cell_manifest(
            cell_path,
            unit=unit_by_id[identifier],
            executor=executor,
            artifact_root=artifact_root,
        )
    return payload


def run_parallel_prediction_units(
    units: Sequence[Mapping[str, Any]],
    *,
    worker_entrypoint: str,
    control_dir: str | Path,
    artifact_root: str | Path,
    gpu_ids: Sequence[int] = (0, 1),
    workers_per_gpu: int | Mapping[int, int] = 1,
    resume: bool = False,
    experiment_kind: str = "film_unet_pure_cnn_backbone_text_effect_10seed",
) -> dict[str, Any]:
    """Run or safely resume all prediction cells using isolated spawned workers."""

    control = Path(control_dir).resolve()
    artifact_root_path = Path(artifact_root).resolve()
    if not artifact_root_path.is_dir():
        raise ParallelPredictionError(f"Artifact root is missing: {artifact_root_path}")
    assigned = assign_units_to_physical_gpus(units, gpu_ids=gpu_ids)
    profiles = validate_shared_noise_lineage(assigned)
    executor_contract = _executor_contract(worker_entrypoint, "spawn")
    worker_counts = _workers_by_gpu(workers_per_gpu, tuple(map(int, gpu_ids)))
    execution_path = control / "parallel_prediction_execution_manifest.json"
    if execution_path.is_file():
        return _validate_execution_manifest(
            execution_path,
            units=assigned,
            executor=executor_contract,
            control_dir=control,
            artifact_root=artifact_root_path,
        )

    with SupervisorLock(control, name="prediction"):
        append_stage_journal(
            control,
            experiment_kind=str(experiment_kind),
            stage="parallel_prediction",
            status="running",
            details={"prediction_units": len(assigned)},
            name="prediction_journal.json",
        )
        manifests: dict[str, dict[str, Any]] = {}
        remaining: list[dict[str, Any]] = []
        for unit in assigned:
            path = prediction_cell_manifest_path(control, unit["prediction_unit_id"])
            if path.is_file():
                if not resume:
                    raise ParallelPredictionError(
                        "Partial prediction manifests require resume=True"
                    )
                manifests[str(unit["prediction_unit_id"])] = _validate_cell_manifest(
                    path,
                    unit=unit,
                    executor=executor_contract,
                    artifact_root=artifact_root_path,
                )
            else:
                remaining.append(unit)
        grouped = {
            gpu: [unit for unit in remaining if int(unit["physical_gpu_id"]) == gpu]
            for gpu in map(int, gpu_ids)
        }
        stop_event = threading.Event()
        thread_errors: list[BaseException] = []
        manifest_lock = threading.Lock()
        unit_by_id = {str(unit["prediction_unit_id"]): unit for unit in assigned}

        def publish_result(result: Mapping[str, Any]) -> None:
            identifier = str(result["prediction_unit_id"])
            published = _publish_worker_result(
                result,
                unit=unit_by_id[identifier],
                executor=executor_contract,
                control_dir=control,
                artifact_root=artifact_root_path,
            )
            with manifest_lock:
                manifests[identifier] = published

        def launch_gpu(gpu: int) -> None:
            try:
                _run_gpu_queue(
                    physical_gpu_id=gpu,
                    units=grouped[gpu],
                    workers=worker_counts[gpu],
                    worker_entrypoint=worker_entrypoint,
                    resume=resume,
                    stop_event=stop_event,
                    on_result=publish_result,
                )
            except BaseException as exc:
                stop_event.set()
                thread_errors.append(exc)

        threads = [
            threading.Thread(target=launch_gpu, args=(gpu,), name=f"predict-gpu-{gpu}")
            for gpu in map(int, gpu_ids)
            if grouped[gpu]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if thread_errors:
            error = thread_errors[0]
            append_stage_journal(
                control,
                experiment_kind=str(experiment_kind),
                stage="parallel_prediction",
                status="failed",
                details={"error": f"{type(error).__name__}: {error}"},
                name="prediction_journal.json",
            )
            raise ParallelPredictionError(
                "Parallel prediction stopped after worker failure"
            ) from error
        if set(manifests) != set(unit_by_id):
            raise ParallelPredictionError(
                "Parallel prediction did not complete every cell"
            )
        cells = []
        for identifier in sorted(manifests):
            path = prediction_cell_manifest_path(control, identifier)
            cells.append(
                {
                    "prediction_unit_id": identifier,
                    "relative_path": path.relative_to(control).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
        payload = _signed(
            {
                "schema_version": 1,
                "kind": "backbone_text_effect_parallel_prediction_manifest_v1",
                "executor_contract_sha256": executor_contract[
                    "executor_contract_sha256"
                ],
                "prediction_unit_ids": sorted(unit_by_id),
                "prediction_unit_count": len(unit_by_id),
                "physical_gpu_ids": list(map(int, gpu_ids)),
                "workers_per_gpu": worker_counts,
                "shared_noise_profiles": profiles,
                "cell_manifests": cells,
            }
        )
        atomic_write_json(execution_path, payload)
        append_stage_journal(
            control,
            experiment_kind=str(experiment_kind),
            stage="parallel_prediction",
            status="completed",
            details={"prediction_units": len(assigned)},
            name="prediction_journal.json",
        )
        return _validate_execution_manifest(
            execution_path,
            units=assigned,
            executor=executor_contract,
            control_dir=control,
            artifact_root=artifact_root_path,
        )


__all__ = [
    "PREDICTION_KINDS",
    "ParallelPredictionError",
    "assign_units_to_physical_gpus",
    "payload_sha256",
    "prediction_cell_manifest_path",
    "run_parallel_prediction_units",
    "sha256_file",
    "validate_shared_noise_lineage",
]
