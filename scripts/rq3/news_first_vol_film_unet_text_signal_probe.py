"""Single-seed FiLM text-signal mechanism probe.

The orchestration in this module is intentionally narrower than the rolling
RQ1--RQ3 pipelines.  It opens only the f2 training and validation partitions,
binds a frozen no-text Generator/Critic pair, and runs a 2^4 text-adapter
screen.  Model, data preparation, analysis, and reporting live in the lazily
imported ``news_first_vol_film_unet_text_signal_probe_core`` module.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import fcntl
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/rq3/news_first_vol_film_unet_text_signal_seed42_f2_probe.yaml"
DEFAULT_OUTPUT_DIR = (
    "outputs/experiments/rq3_news_first_vol_film_unet_text_signal_seed42_f2_5m_v1"
)
EXPERIMENT_KIND = "film_unet_text_signal_seed42_f2_5m_probe_v1"
INTERPRETATION = "single_seed_mechanism_validation"
CORE_MODULE = "scripts.rq3.news_first_vol_film_unet_text_signal_probe_core"
FOLD_ID = "f2_2023q2"
SEED = 42
FACTOR_ORDER = ("A", "B", "C", "D")
SCREEN_FACTOR_CODES = tuple(f"{value:04b}" for value in range(16))
SCREEN_EPOCHS = 60
CONFIRMATION_EPOCHS = 240
EXPECTED_COUNTS = {
    "train_pairs": 526,
    "train_sessions": 133,
    "validation_pairs": 110,
    "validation_sessions": 34,
}
BASELINE_GENERATOR_SHA256 = (
    "c30ae7d30a317a4d4cffa205597345bea38086f6f01b52137f2ea07fd44f8fee"
)
BASELINE_CRITIC_SHA256 = (
    "6807d97f847389a262dc23365075ff757eca718b9969213db79b69ffcae29c6c"
)
ACTION_NAMES = (
    "prepare",
    "canary",
    "launch-screen",
    "analyze-screen",
    "launch-confirmation",
    "report",
    "qa",
    "status",
    "run-pipeline",
)
REGISTRY_KIND = "film_unet_text_signal_probe_task_registry_v1"
JOB_SPEC_KIND = "film_unet_text_signal_probe_job_spec_v1"
RESULT_KIND = "film_unet_text_signal_probe_job_result_v1"
CANARY_KIND = "film_unet_text_signal_probe_canary_result_v1"
INPUT_MANIFEST_RELATIVE_PATH = Path("inputs/probe_input_manifest.json")

_HELD_LOCKS: dict[Path, tuple[int, int]] = {}


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def _write_json(path: Path, payload: object) -> Path:
    return _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _signed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return {**unsigned, "payload_sha256": _payload_sha256(unsigned)}


def _verify_signed_payload(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    observed = str(unsigned.pop("payload_sha256", ""))
    if not observed or observed != _payload_sha256(unsigned):
        raise ValueError(f"{label} payload SHA drift")


def _file_row(role: str, path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _verify_file_row(row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != int(row.get("size_bytes", -1)):
        raise ValueError(f"{label} size drift: {path}")
    if _sha256_file(path) != str(row.get("sha256", "")):
        raise ValueError(f"{label} SHA drift: {path}")
    return path


def _load_core() -> Any:
    return importlib.import_module(CORE_MODULE)


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = _resolve_path(config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = _mapping(raw, "probe config")
    config["source_config_path"] = str(source)
    config["source_config_sha256"] = _sha256_file(source)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config.get("experiment"), "experiment")
    data = _mapping(config.get("data"), "data")
    baseline = _mapping(config.get("baseline"), "baseline")
    model = _mapping(config.get("model"), "model")
    factors = _mapping(config.get("factors"), "factors")
    training = _mapping(config.get("training"), "training")
    analysis = _mapping(config.get("analysis"), "analysis")
    runtime = _mapping(config.get("runtime"), "runtime")

    if int(experiment.get("schema_version", -1)) != 1:
        raise ValueError("experiment.schema_version must be 1")
    if experiment.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError("Experiment kind drift")
    if experiment.get("interpretation") != INTERPRETATION:
        raise ValueError("Interpretation drift")
    if experiment.get("core_module") != CORE_MODULE:
        raise ValueError("Core module drift")

    if int(data.get("tolerance_minutes", -1)) != 5:
        raise ValueError("Only the 5m alignment is permitted")
    if data.get("fold_id") != FOLD_ID:
        raise ValueError(f"fold_id must be {FOLD_ID}")
    split = _mapping(data.get("split"), "data.split")
    if set(split) != {
        "train_end_utc",
        "validation_start_utc",
        "validation_end_utc",
    }:
        raise ValueError("The probe split may contain train/validation bounds only")
    if split != {
        "train_end_utc": "2023-01-01T00:00:00Z",
        "validation_start_utc": "2023-01-01T00:00:00Z",
        "validation_end_utc": "2023-04-01T00:00:00Z",
    }:
        raise ValueError("Frozen f2 train/validation bounds drift")
    counts = {
        key: int(value)
        for key, value in _mapping(
            data.get("expected_counts"), "data.expected_counts"
        ).items()
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Pair/session counts drift: {counts}")
    if tuple(map(int, data.get("maturity_days_grid", ()))) != (
        1,
        2,
        3,
        6,
        7,
        8,
        9,
        10,
        14,
        15,
        17,
        21,
        26,
        30,
        35,
        38,
    ):
        raise ValueError("Exact-TTM axis drift")
    if len(tuple(data.get("strike_grid", ()))) != 16:
        raise ValueError("Strike axis must have 16 coordinates")
    source_rows = list(data.get("source_files") or [])
    if len(source_rows) != 6:
        raise ValueError("Exactly six data-source bindings are required")
    roles = [str(_mapping(row, "data source").get("role")) for row in source_rows]
    if len(set(roles)) != len(roles):
        raise ValueError("Data-source roles must be unique")
    for row in source_rows:
        value = _mapping(row, "data source")
        if len(str(value.get("sha256", ""))) != 64:
            raise ValueError(f"Missing data SHA for {value.get('role')}")

    generator = _mapping(baseline.get("generator_checkpoint"), "baseline Generator")
    critic = _mapping(baseline.get("critic_checkpoint"), "baseline Critic")
    if generator.get("sha256") != BASELINE_GENERATOR_SHA256:
        raise ValueError("Frozen Generator SHA drift")
    if critic.get("sha256") != BASELINE_CRITIC_SHA256:
        raise ValueError("Frozen Critic SHA drift")
    if int(generator.get("size_bytes", -1)) != 3_359_378:
        raise ValueError("Frozen Generator size drift")
    if int(critic.get("size_bytes", -1)) != 2_958_626:
        raise ValueError("Frozen Critic size drift")

    if model.get("generator_conditioning_mode") != "film_unet_mask_coords_v1":
        raise ValueError("Generator mode drift")
    if model.get("critic_conditioning_mode") != "lp_disabled_same_shape_v1":
        raise ValueError("The mechanism probe requires a NoLP Critic")
    expected_parameters = {
        "frozen_generator_parameters": 827_745,
        "frozen_critic_parameters": 729_157,
        "expected_spatial_r2_generator_parameters": 1_058_913,
        "expected_text328_global_generator_parameters": 1_058_345,
    }
    for key, expected in expected_parameters.items():
        if int(model.get(key, -1)) != expected:
            raise ValueError(f"model.{key} must be {expected}")

    if tuple(factors.get("order", ())) != FACTOR_ORDER:
        raise ValueError("Factor order must be A/B/C/D")
    for letter in FACTOR_ORDER:
        levels = _mapping(factors.get(letter), f"factors.{letter}")
        if not {"0", "1"}.issubset(levels):
            raise ValueError(f"factors.{letter} must define levels 0 and 1")

    required_training = {
        "seed": SEED,
        "batch_size": 16,
        "discriminator_steps": 5,
        "validation_mc_samples": 16,
        "screen_epochs": SCREEN_EPOCHS,
        "confirmation_epochs": CONFIRMATION_EPOCHS,
        "early_stopping": False,
        "scheduler": False,
        "checkpoint_selection": "final_epoch",
        "state_schema": "text_adapter_probe_state_v1",
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} drift")
    if analysis.get("evaluation_partition") != "validation":
        raise ValueError("The probe may evaluate validation only")
    if int(analysis.get("bootstrap_replicates", -1)) != 10_000:
        raise ValueError("Analysis requires 10,000 bootstrap replicates")
    if tuple(map(int, runtime.get("gpu_ids", ()))) != (0, 1):
        raise ValueError("runtime.gpu_ids must be [0, 1]")
    if tuple(map(int, runtime.get("workers_per_gpu_candidates", ()))) != (8, 4):
        raise ValueError("Concurrency candidates must be 8 then 4 workers/GPU")
    if tuple(map(str, runtime.get("canary_factor_codes", ()))) != ("0000", "1111"):
        raise ValueError("Canary must exercise 0000 and 1111")


def _control_dir(root: Path) -> Path:
    return root.with_name(root.name + "_control")


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    control = _control_dir(root)
    control.mkdir(parents=True, exist_ok=True)
    path = (control / "pipeline.lock").resolve()
    held = _HELD_LOCKS.get(path)
    if held is not None:
        descriptor, depth = held
        _HELD_LOCKS[path] = (descriptor, depth + 1)
        try:
            yield
        finally:
            descriptor, current_depth = _HELD_LOCKS[path]
            _HELD_LOCKS[path] = (descriptor, current_depth - 1)
        return
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"Another probe supervisor owns {path}") from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
    _HELD_LOCKS[path] = (descriptor, 1)
    _atomic_write_text(control / "active.pid", f"{os.getpid()}\n")
    try:
        yield
    finally:
        current_descriptor, depth = _HELD_LOCKS[path]
        if depth != 1:
            raise RuntimeError("Unbalanced re-entrant pipeline lock")
        _HELD_LOCKS.pop(path, None)
        fcntl.flock(current_descriptor, fcntl.LOCK_UN)
        os.close(current_descriptor)
        (control / "active.pid").unlink(missing_ok=True)


def _journal(root: Path, stage: str, status: str, **details: Any) -> Path:
    path = _control_dir(root) / "stage_journal.json"
    history: list[dict[str, Any]] = []
    if path.is_file():
        current = _read_json(path)
        history = list(current.get("history") or [])
    history.append(
        {
            "stage": stage,
            "status": status,
            "pid": os.getpid(),
            "at_utc": _utc_now(),
            **details,
        }
    )
    return _write_json(path, {"history": history, "latest": history[-1]})


def _manifest_path(root: Path, name: str) -> Path:
    return root / "contracts" / f"{name}.json"


def _write_manifest(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    payload = _signed_payload(
        {
            "schema_version": 1,
            "kind": f"film_unet_text_signal_probe_{name}_v1",
            "rows": [dict(row) for row in rows],
        }
    )
    return _write_json(_manifest_path(root, name), payload)


def _verify_manifest(path: Path, label: str) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed_payload(payload, label)
    rows = list(payload.get("rows") or [])
    if not rows:
        raise ValueError(f"{label} has no rows")
    for row in rows:
        _verify_file_row(_mapping(row, f"{label} row"), label)
    return payload


def _source_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    runtime = _mapping(config["runtime"], "runtime")
    files: set[Path] = set()
    for root_value in runtime.get("source_hash_roots", ()):
        source_root = _resolve_path(str(root_value))
        files.update(path.resolve() for path in source_root.rglob("*.py"))
    files.update(
        _resolve_path(str(value)) for value in runtime.get("source_hash_files", ())
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Probe source files missing: {missing}")
    return [
        _file_row(
            f"source:{path.relative_to(REPO_ROOT).as_posix()}",
            path,
        )
        for path in sorted(files, key=lambda value: value.as_posix())
    ]


def _data_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in config["data"]["source_files"]:
        source = _mapping(raw, "data source")
        row = _file_row(str(source["role"]), _resolve_path(str(source["path"])))
        if row["sha256"] != str(source["sha256"]):
            raise ValueError(f"Frozen data SHA drift: {row['path']}")
        rows.append(row)
    return rows


def _baseline_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, key in (
        ("baseline_generator", "generator_checkpoint"),
        ("baseline_critic", "critic_checkpoint"),
    ):
        frozen = _mapping(config["baseline"][key], role)
        row = _file_row(role, _resolve_path(str(frozen["path"])))
        if row["sha256"] != str(frozen["sha256"]):
            raise ValueError(f"{role} SHA drift")
        if row["size_bytes"] != int(frozen["size_bytes"]):
            raise ValueError(f"{role} size drift")
        rows.append(row)
    return rows


def _verify_input_manifest(root: Path) -> dict[str, Any]:
    path = root / INPUT_MANIFEST_RELATIVE_PATH
    payload = _read_json(path)
    if tuple(payload.get("partitions") or ()) != ("train", "validation"):
        raise ValueError("Probe input manifest may contain train/validation only")
    if payload.get("fold_id") != FOLD_ID:
        raise ValueError("Probe input fold drift")
    observed_counts = {
        key: int(value)
        for key, value in _mapping(payload.get("counts"), "input counts").items()
    }
    if observed_counts != EXPECTED_COUNTS:
        raise ValueError(f"Prepared input counts drift: {observed_counts}")
    artifacts = list(payload.get("artifacts") or [])
    if not artifacts:
        raise ValueError("Probe input manifest must bind its materialized artifacts")
    for artifact in artifacts:
        _verify_file_row(_mapping(artifact, "input artifact"), "input artifact")
    return payload


def _registry_path(root: Path) -> Path:
    return root / "registry/task_registry.json"


def _write_registry(root: Path, payload: Mapping[str, Any]) -> Path:
    value = dict(payload)
    value["updated_at_utc"] = _utc_now()
    return _write_json(_registry_path(root), _signed_payload(value))


def _read_registry(root: Path) -> dict[str, Any]:
    registry = _read_json(_registry_path(root))
    _verify_signed_payload(registry, "task registry")
    if registry.get("kind") != REGISTRY_KIND:
        raise ValueError("Task-registry kind drift")
    return registry


def _factor_levels(config: Mapping[str, Any], code: str) -> dict[str, dict[str, Any]]:
    if code not in SCREEN_FACTOR_CODES:
        raise ValueError(f"Invalid factor code: {code}")
    return {
        letter: deepcopy(_mapping(config["factors"][letter][bit], f"factor {letter}"))
        for letter, bit in zip(FACTOR_ORDER, code, strict=True)
    }


def build_screen_designs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    designs: list[dict[str, Any]] = []
    gpu_ids = tuple(map(int, config["runtime"]["gpu_ids"]))
    for index, code in enumerate(SCREEN_FACTOR_CODES):
        designs.append(
            {
                "job_id": f"screen_05m_{FOLD_ID}_seed_{SEED}_abcd_{code}",
                "stage": "screen",
                "seed": SEED,
                "fold_id": FOLD_ID,
                "tolerance_minutes": 5,
                "factor_code": code,
                "factors": _factor_levels(config, code),
                "text_assignment": "matched",
                "physical_gpu_id": gpu_ids[index % len(gpu_ids)],
                "max_epochs": SCREEN_EPOCHS,
            }
        )
    counts = {
        gpu: sum(int(row["physical_gpu_id"]) == gpu for row in designs)
        for gpu in gpu_ids
    }
    if len(designs) != 16 or len({row["job_id"] for row in designs}) != 16:
        raise AssertionError("Screen must contain 16 unique jobs")
    if counts != {gpu: 8 for gpu in gpu_ids}:
        raise AssertionError(f"Screen GPU assignment is not 8/8: {counts}")
    return designs


def build_confirmation_designs(
    config: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not bool(selection.get("confirmation_eligible")):
        return []
    winner = str(selection.get("winner_factor_code", ""))
    if winner not in SCREEN_FACTOR_CODES:
        raise ValueError("Eligible screen selection lacks a valid winner")
    gpu_ids = tuple(map(int, config["runtime"]["gpu_ids"]))
    definitions: list[tuple[str, str, str, int]] = [
        ("baseline_0000_matched", "0000", "matched", 128),
        ("winner_matched", winner, "matched", 128),
        ("winner_independent_shuffle", winner, "independent_shuffle", 128),
    ]
    if winner[3] == "1":
        definitions.append(
            ("winner_text328_global_capacity_control", winner[:3] + "0", "matched", 328)
        )
    designs: list[dict[str, Any]] = []
    for index, (role, code, assignment, text_dimension) in enumerate(definitions):
        designs.append(
            {
                "job_id": f"confirmation_05m_{FOLD_ID}_seed_{SEED}_{role}",
                "stage": "confirmation",
                "confirmation_role": role,
                "seed": SEED,
                "fold_id": FOLD_ID,
                "tolerance_minutes": 5,
                "factor_code": code,
                "winner_factor_code": winner,
                "factors": _factor_levels(config, code),
                "text_assignment": assignment,
                "text_output_dimension": text_dimension,
                "capacity_matched_control": text_dimension == 328,
                "physical_gpu_id": gpu_ids[index % len(gpu_ids)],
                "max_epochs": CONFIRMATION_EPOCHS,
            }
        )
    if len(designs) not in (3, 4) or len({row["job_id"] for row in designs}) != len(
        designs
    ):
        raise AssertionError("Confirmation matrix must contain 3 or 4 unique jobs")
    return designs


def _contract_bindings(root: Path) -> dict[str, Any]:
    paths = {
        "source_manifest": _manifest_path(root, "source_hashes"),
        "data_manifest": _manifest_path(root, "data_hashes"),
        "baseline_manifest": _manifest_path(root, "baseline_hashes"),
        "config_manifest": _manifest_path(root, "config_hashes"),
        "input_manifest": root / INPUT_MANIFEST_RELATIVE_PATH,
    }
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _checkpoint_contract(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = deepcopy(_mapping(config["baseline"][key], f"baseline.{key}"))
    value["path"] = str(_resolve_path(str(value["path"])))
    return value


def _write_job_spec(path: Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    unsigned = dict(payload)
    unsigned.pop("job_spec_sha256", None)
    digest = _payload_sha256(unsigned)
    target = _write_json(path, {**unsigned, "job_spec_sha256": digest})
    return target, digest


def _verify_job_spec(
    path: str | Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    payload = _read_json(path)
    observed = str(payload.pop("job_spec_sha256", ""))
    calculated = _payload_sha256(payload)
    if not observed or observed != calculated:
        raise ValueError(f"Job-spec payload drift: {path}")
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"Job-spec registry binding drift: {path}")
    return {**payload, "job_spec_sha256": observed}


def _materialize_job_specs(
    root: Path,
    config: Mapping[str, Any],
    designs: Sequence[Mapping[str, Any]],
    *,
    auxiliary_weight: float,
) -> list[dict[str, Any]]:
    bindings = _contract_bindings(root)
    records: list[dict[str, Any]] = []
    for design in designs:
        stage = str(design["stage"])
        job_id = str(design["job_id"])
        output_dir = (root / "runs" / stage / job_id).resolve()
        spec_path = root / "registry/job_specs" / stage / f"{job_id}.json"
        b_enabled = str(design["factor_code"])[1] == "1"
        payload = {
            "schema_version": 1,
            "kind": JOB_SPEC_KIND,
            **dict(design),
            "output_dir": str(output_dir),
            "evaluation_partition": "validation",
            "allowed_partitions": ["train", "validation"],
            "core_module": CORE_MODULE,
            "baseline_generator": _checkpoint_contract(config, "generator_checkpoint"),
            "baseline_critic": _checkpoint_contract(config, "critic_checkpoint"),
            "contract_bindings": bindings,
            "input_manifest_path": bindings["input_manifest"]["path"],
            "input_manifest_sha256": bindings["input_manifest"]["sha256"],
            "calibrate_auxiliary_weight": False,
            "alignment_auxiliary_weight": auxiliary_weight if b_enabled else 0.0,
            "training_contract": deepcopy(config["training"]),
            "model_contract": deepcopy(config["model"]),
            "analysis_contract": deepcopy(config["analysis"]),
        }
        path, digest = _write_job_spec(spec_path, payload)
        records.append(
            {
                **dict(design),
                "job_spec_path": str(path.resolve()),
                "job_spec_sha256": digest,
                "output_dir": str(output_dir),
            }
        )
    return records


def _validate_contracts(root: Path, config: Mapping[str, Any]) -> None:
    frozen_config = _read_json(root / "contracts/resolved_config.json")
    if frozen_config != dict(config):
        raise ValueError("Frozen resolved config drift")
    for name in ("source_hashes", "data_hashes", "baseline_hashes", "config_hashes"):
        _verify_manifest(_manifest_path(root, name), name)
    _verify_input_manifest(root)
    registry = _read_registry(root)
    if registry.get("contract_bindings") != _contract_bindings(root):
        raise ValueError("Task-registry contract bindings drift")


def prepare(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _journal(root, "prepare", "running")
        if _registry_path(root).is_file():
            if not resume:
                raise FileExistsError(f"Probe root already prepared: {root}")
            _validate_contracts(root, config)
            _journal(root, "prepare", "completed", resumed=True)
            return root
        if root.exists() and any(root.iterdir()) and not resume:
            raise FileExistsError(f"Non-empty probe root requires --resume: {root}")
        for relative in (
            "contracts",
            "inputs",
            "registry/job_specs/screen",
            "registry/job_specs/confirmation",
            "registry/job_status",
            "runs/screen",
            "runs/confirmation",
            "analysis",
            "report",
            "qa",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

        _write_json(root / "contracts/resolved_config.json", config)
        _write_manifest(root, "source_hashes", _source_rows(config))
        _write_manifest(root, "data_hashes", _data_rows(config))
        _write_manifest(root, "baseline_hashes", _baseline_rows(config))
        config_rows = [
            _file_row("source_config", config["source_config_path"]),
            _file_row("resolved_config", root / "contracts/resolved_config.json"),
        ]
        _write_manifest(root, "config_hashes", config_rows)

        core = _load_core()
        core.prepare_probe_inputs(root, config)
        _verify_input_manifest(root)
        designs = build_screen_designs(config)
        registry = {
            "schema_version": 1,
            "kind": REGISTRY_KIND,
            "experiment_kind": EXPERIMENT_KIND,
            "interpretation": INTERPRETATION,
            "status": "prepared",
            "root": str(root),
            "source_config_path": config["source_config_path"],
            "source_config_sha256": config["source_config_sha256"],
            "contract_bindings": _contract_bindings(root),
            "screen_jobs": designs,
            "screen_job_specs_frozen": False,
            "confirmation_jobs": [],
            "confirmation_status": "not_selected",
            "partitions_materialized": ["train", "validation"],
            "test_loader_count": 0,
            "prediction_count": 0,
            "created_at_utc": _utc_now(),
        }
        _write_registry(root, registry)
        _validate_contracts(root, config)
        _journal(root, "prepare", "completed", screen_jobs=16)
        return root


def _status_path(root: Path, stage: str, job_id: str) -> Path:
    if stage == "canary":
        return root / "job_status" / f"{job_id}.json"
    return root / "registry/job_status" / f"{job_id}.json"


def _verify_result(spec: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    result = _read_json(result_path)
    if int(result.get("schema_version", -1)) != 1:
        raise ValueError(f"Job result schema drift: {result_path}")
    if result.get("kind") != RESULT_KIND:
        raise ValueError(f"Job result kind drift: {result_path}")
    if result.get("status") != "completed":
        raise ValueError(f"Job result is not completed: {result_path}")
    if result.get("job_id") != spec.get("job_id"):
        raise ValueError(f"Job/result ID mismatch: {result_path}")
    if result.get("job_spec_sha256") != spec.get("job_spec_sha256"):
        raise ValueError(f"Job/result spec SHA mismatch: {result_path}")
    expected_fields = {
        "epochs": int(spec["max_epochs"]),
        "factor_code": str(spec["factor_code"]),
        "assignment": str(spec["text_assignment"]),
        "input_manifest_sha256": str(spec["input_manifest_sha256"]),
        "allowed_partitions": ["train", "validation"],
        "test_loader_count": 0,
        "prediction_count": 0,
    }
    for key, expected in expected_fields.items():
        if result.get(key) != expected:
            raise ValueError(
                f"Job result {key} drift: {result.get(key)!r} != {expected!r}"
            )
    artifacts = list(result.get("artifacts") or [])
    if not artifacts:
        raise ValueError(f"Job result has no real artifacts: {result_path}")
    for artifact in artifacts:
        _verify_file_row(_mapping(artifact, "job artifact"), "job artifact")
    return result


def _completed_valid(record: Mapping[str, Any], root: Path) -> bool:
    status_path = _status_path(root, str(record["stage"]), str(record["job_id"]))
    if not status_path.is_file():
        return False
    try:
        status = _read_json(status_path)
        if status.get("status") != "completed" or status.get(
            "job_spec_sha256"
        ) != record.get("job_spec_sha256"):
            return False
        spec = _verify_job_spec(record["job_spec_path"], record["job_spec_sha256"])
        result_path = Path(str(status["result_path"])).resolve()
        if _sha256_file(result_path) != str(status["result_sha256"]):
            return False
        _verify_result(spec, Path(str(record["output_dir"])))
        return True
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _job_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise RuntimeError(f"Duplicate live job: {path}") from exc
    return descriptor


def _worker_entry(
    spec_path_value: str,
    output_dir_value: str,
    status_path_value: str,
    physical_gpu_id: int,
    cpu_threads: int,
) -> None:
    spec_path = Path(spec_path_value).resolve()
    output_dir = Path(output_dir_value).resolve()
    status_path = Path(status_path_value).resolve()
    descriptor = _job_lock(status_path.with_suffix(".lock"))
    try:
        spec = _verify_job_spec(spec_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        running = {
            "status": "running",
            "job_id": spec["job_id"],
            "job_spec_path": str(spec_path),
            "job_spec_sha256": spec["job_spec_sha256"],
            "physical_gpu_id": physical_gpu_id,
            "pid": os.getpid(),
            "started_at_utc": _utc_now(),
        }
        _write_json(status_path, running)
        log_path = output_dir / "worker.log"
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                with redirect_stdout(log_handle), redirect_stderr(log_handle):
                    core = _load_core()
                    core.run_probe_job(spec_path, output_dir)
            result = _verify_result(spec, output_dir)
            result_path = output_dir / "result.json"
            _write_json(
                status_path,
                {
                    **running,
                    "status": "completed",
                    "completed_at_utc": _utc_now(),
                    "result_kind": result.get("kind", RESULT_KIND),
                    "result_path": str(result_path.resolve()),
                    "result_sha256": _sha256_file(result_path),
                    "worker_log_path": str(log_path.resolve()),
                    "worker_log_sha256": _sha256_file(log_path),
                },
            )
        except BaseException as exc:
            with log_path.open("a", encoding="utf-8") as log_handle:
                traceback.print_exc(file=log_handle)
            _write_json(
                status_path,
                {
                    **running,
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "worker_log_path": str(log_path.resolve()),
                    "worker_log_sha256": _sha256_file(log_path),
                },
            )
            raise
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _host_ram_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0])
    return 1.0 - values["MemAvailable"] / values["MemTotal"]


def _gpu_memory_snapshot(config: Mapping[str, Any]) -> dict[int, float]:
    executable = str(config["runtime"].get("nvidia_smi_executable", "nvidia-smi"))
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    requested = set(map(int, config["runtime"]["gpu_ids"]))
    values: dict[int, float] = {}
    for line in completed.stdout.splitlines():
        index_text, memory_text = (value.strip() for value in line.split(","))
        index = int(index_text)
        if index in requested:
            values[index] = float(memory_text) / 1024.0
    if set(values) != requested:
        raise RuntimeError(f"nvidia-smi omitted requested GPUs: {values}")
    return values


def _launch_records(
    records: Sequence[Mapping[str, Any]],
    root: Path,
    config: Mapping[str, Any],
    *,
    workers_per_gpu: int,
    resume: bool,
) -> dict[str, Any]:
    gpu_ids = tuple(map(int, config["runtime"]["gpu_ids"]))
    stages = {str(row["stage"]) for row in records}
    if len(stages) != 1:
        raise ValueError(f"Resource launch requires one stage, got {sorted(stages)}")
    resource_path = root / f"resource_summary_{next(iter(stages))}.json"
    pending = [dict(row) for row in records]
    if resume:
        pending = [row for row in pending if not _completed_valid(row, root)]
    elif any(
        _status_path(root, str(row["stage"]), str(row["job_id"])).exists()
        for row in pending
    ):
        raise FileExistsError("Existing job status requires --resume")
    if not pending:
        if not resource_path.is_file():
            raise ValueError("Completed jobs lack their frozen resource summary")
        frozen = _read_json(resource_path)
        return {
            "workers_per_gpu": int(frozen["workers_per_gpu"]),
            "peak_host_ram_fraction": float(frozen["peak_host_ram_fraction"]),
            "peak_gpu_memory_gib": dict(frozen["peak_gpu_memory_gib"]),
            "completed_jobs": sum(_completed_valid(row, root) for row in records),
        }
    active: dict[int, tuple[multiprocessing.Process, dict[str, Any]]] = {}
    active_per_gpu = {gpu: 0 for gpu in gpu_ids}
    context = multiprocessing.get_context("spawn")
    peak_ram = _host_ram_fraction()
    peak_gpu = _gpu_memory_snapshot(config)
    resource_rows: list[dict[str, Any]] = []
    last_sample = 0.0
    poll = float(config["runtime"]["poll_interval_seconds"])
    sample_interval = float(config["runtime"]["resource_sample_interval_seconds"])
    try:
        while pending or active:
            for gpu in gpu_ids:
                while active_per_gpu[gpu] < workers_per_gpu:
                    index = next(
                        (
                            idx
                            for idx, record in enumerate(pending)
                            if int(record["physical_gpu_id"]) == gpu
                        ),
                        None,
                    )
                    if index is None:
                        break
                    record = pending.pop(index)
                    status_path = _status_path(
                        root, str(record["stage"]), str(record["job_id"])
                    )
                    process = context.Process(
                        target=_worker_entry,
                        args=(
                            str(record["job_spec_path"]),
                            str(record["output_dir"]),
                            str(status_path),
                            gpu,
                            int(config["runtime"]["cpu_threads_per_job"]),
                        ),
                        name=str(record["job_id"]),
                    )
                    process.start()
                    active[process.pid or -1] = (process, record)
                    active_per_gpu[gpu] += 1

            now = time.monotonic()
            if now - last_sample >= sample_interval:
                ram = _host_ram_fraction()
                gpu_values = _gpu_memory_snapshot(config)
                peak_ram = max(peak_ram, ram)
                peak_gpu = {
                    gpu: max(peak_gpu.get(gpu, 0.0), gpu_values[gpu]) for gpu in gpu_ids
                }
                resource_rows.append(
                    {
                        "at_utc": _utc_now(),
                        "host_ram_fraction": ram,
                        "gpu_memory_gib": gpu_values,
                        "active_jobs": len(active),
                    }
                )
                last_sample = now

            failures: list[str] = []
            for pid, (process, record) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                active.pop(pid)
                gpu = int(record["physical_gpu_id"])
                active_per_gpu[gpu] -= 1
                if process.exitcode != 0 or not _completed_valid(record, root):
                    status_path = _status_path(
                        root, str(record["stage"]), str(record["job_id"])
                    )
                    detail = _read_json(status_path) if status_path.is_file() else {}
                    failures.append(
                        f"{record['job_id']}: exit={process.exitcode}: "
                        f"{detail.get('error', 'invalid completion artifacts')}"
                    )
            if failures:
                for process, _record in active.values():
                    if process.is_alive():
                        os.kill(process.pid, signal.SIGTERM)
                for process, _record in active.values():
                    process.join(timeout=15)
                raise RuntimeError("Probe worker failure: " + "; ".join(failures))
            if pending or active:
                time.sleep(poll)
    finally:
        _write_json(
            resource_path,
            {
                "workers_per_gpu": workers_per_gpu,
                "peak_host_ram_fraction": peak_ram,
                "peak_gpu_memory_gib": peak_gpu,
                "samples": resource_rows,
            },
        )
    return {
        "workers_per_gpu": workers_per_gpu,
        "peak_host_ram_fraction": peak_ram,
        "peak_gpu_memory_gib": peak_gpu,
        "completed_jobs": sum(_completed_valid(row, root) for row in records),
    }


def _canary_result_path(root: Path) -> Path:
    return _control_dir(root) / "canary_result.json"


def _verify_canary_result(
    formal_root: Path, config: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    result = _read_json(path)
    _verify_signed_payload(result, "canary result")
    if result.get("kind") != CANARY_KIND or result.get("status") != "passed":
        raise ValueError("Existing canary result is not a passing result")
    workers = int(result.get("selected_workers_per_gpu", -1))
    if workers not in tuple(map(int, config["runtime"]["workers_per_gpu_candidates"])):
        raise ValueError("Canary selected an unconfigured concurrency")
    if result.get("contract_bindings") != _contract_bindings(formal_root):
        raise ValueError("Canary contract binding drift")
    canary_root = Path(str(result.get("canary_root", ""))).resolve()
    records = list(result.get("jobs") or [])
    expected_count = workers * len(tuple(config["runtime"]["gpu_ids"]))
    if len(records) != expected_count:
        raise ValueError("Canary job universe drift")
    if any(not _completed_valid(row, canary_root) for row in records):
        raise ValueError("Canary completion artifacts drift")
    weight = float(result.get("alignment_auxiliary_weight", math.nan))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("Canary auxiliary-weight binding drift")
    return result


def _canary_records(
    formal_root: Path,
    config: Mapping[str, Any],
    workers_per_gpu: int,
) -> tuple[Path, list[dict[str, Any]]]:
    canary_root = _control_dir(formal_root) / "canary" / f"workers_{workers_per_gpu}"
    canary_root.mkdir(parents=True, exist_ok=True)
    bindings = _contract_bindings(formal_root)
    records: list[dict[str, Any]] = []
    for gpu in map(int, config["runtime"]["gpu_ids"]):
        for replica in range(workers_per_gpu):
            code = ("0000", "1111")[replica % 2]
            job_id = f"canary_gpu{gpu}_rep{replica:02d}_{code}"
            output_dir = (canary_root / "runs" / job_id).resolve()
            payload = {
                "schema_version": 1,
                "kind": JOB_SPEC_KIND,
                "job_id": job_id,
                "stage": "canary",
                "seed": SEED,
                "fold_id": FOLD_ID,
                "tolerance_minutes": 5,
                "factor_code": code,
                "factors": _factor_levels(config, code),
                "text_assignment": "matched",
                "physical_gpu_id": gpu,
                "max_epochs": 1,
                "output_dir": str(output_dir),
                "evaluation_partition": "validation",
                "allowed_partitions": ["train", "validation"],
                "core_module": CORE_MODULE,
                "baseline_generator": _checkpoint_contract(
                    config, "generator_checkpoint"
                ),
                "baseline_critic": _checkpoint_contract(config, "critic_checkpoint"),
                "contract_bindings": bindings,
                "input_manifest_path": bindings["input_manifest"]["path"],
                "input_manifest_sha256": bindings["input_manifest"]["sha256"],
                "calibrate_auxiliary_weight": code == "1111",
                "alignment_auxiliary_weight": None if code == "1111" else 0.0,
                "training_contract": deepcopy(config["training"]),
                "model_contract": deepcopy(config["model"]),
                "analysis_contract": deepcopy(config["analysis"]),
            }
            path, digest = _write_job_spec(
                canary_root / "job_specs" / f"{job_id}.json", payload
            )
            records.append(
                {
                    "job_id": job_id,
                    "stage": "canary",
                    "factor_code": code,
                    "physical_gpu_id": gpu,
                    "job_spec_path": str(path.resolve()),
                    "job_spec_sha256": digest,
                    "output_dir": str(output_dir),
                }
            )
    return canary_root, records


def _capacity_related(root: Path, exc: BaseException) -> bool:
    fragments = [str(exc).lower()]
    for path in root.rglob("*.json"):
        if path.name.endswith("status.json") or path.parent.name == "job_status":
            fragments.append(path.read_text(encoding="utf-8", errors="replace").lower())
    text = "\n".join(fragments)
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "gpu-memory gate",
            "host-ram gate",
        )
    )


def _freeze_screen_specs(root: Path, config: Mapping[str, Any], weight: float) -> None:
    registry = _read_registry(root)
    if registry.get("screen_job_specs_frozen"):
        for record in registry["screen_jobs"]:
            _verify_job_spec(record["job_spec_path"], record["job_spec_sha256"])
        if not math.isclose(
            float(registry["alignment_auxiliary_weight"]),
            weight,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("Frozen screen auxiliary weight drift")
        return
    records = _materialize_job_specs(
        root,
        config,
        registry["screen_jobs"],
        auxiliary_weight=weight,
    )
    if len(records) != 16:
        raise AssertionError("Exactly 16 screen job specs must be frozen")
    registry.update(
        status="canary_passed",
        screen_job_specs_frozen=True,
        screen_jobs=records,
        alignment_auxiliary_weight=weight,
    )
    _write_registry(root, registry)


def canary(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _validate_contracts(root, config)
        _journal(root, "canary", "running")
        result_path = _canary_result_path(root)
        if result_path.is_file():
            result = _verify_canary_result(root, config, result_path)
            _freeze_screen_specs(
                root, config, float(result["alignment_auxiliary_weight"])
            )
            registry = _read_registry(root)
            registry["formal_workers_per_gpu"] = int(result["selected_workers_per_gpu"])
            registry["canary_result_path"] = str(result_path.resolve())
            registry["canary_result_sha256"] = _sha256_file(result_path)
            _write_registry(root, registry)
            _journal(root, "canary", "completed", resumed=True)
            return result_path

        errors: list[str] = []
        candidates = tuple(map(int, config["runtime"]["workers_per_gpu_candidates"]))
        for index, workers in enumerate(candidates):
            canary_root, records = _canary_records(root, config, workers)
            try:
                resources = _launch_records(
                    records,
                    canary_root,
                    config,
                    workers_per_gpu=workers,
                    resume=resume,
                )
                peak_gpu = max(
                    float(value) for value in resources["peak_gpu_memory_gib"].values()
                )
                peak_ram = float(resources["peak_host_ram_fraction"])
                if peak_gpu >= float(config["runtime"]["maximum_peak_gpu_memory_gib"]):
                    raise RuntimeError("GPU-memory gate failed")
                if peak_ram >= float(config["runtime"]["maximum_host_ram_fraction"]):
                    raise RuntimeError("Host-RAM gate failed")
                calibration_rows: list[tuple[dict[str, Any], float]] = []
                for record in records:
                    if record["factor_code"] != "1111":
                        continue
                    spec = _verify_job_spec(
                        record["job_spec_path"], record["job_spec_sha256"]
                    )
                    result = _verify_result(spec, Path(str(record["output_dir"])))
                    value = float(result["calibrated_auxiliary_weight"])
                    if not math.isfinite(value) or value <= 0.0:
                        raise ValueError("Canary produced an invalid auxiliary weight")
                    calibration_rows.append((record, value))
                canonical = [
                    value
                    for record, value in calibration_rows
                    if int(record["physical_gpu_id"]) == 0
                    and str(record["job_id"]).endswith("rep01_1111")
                ]
                if len(canonical) != 1:
                    raise ValueError("Canonical GPU0 1111 calibration is missing")
                weight = canonical[0]
                if any(
                    not math.isclose(value, weight, rel_tol=1e-6, abs_tol=1e-12)
                    for _record, value in calibration_rows
                ):
                    raise ValueError("Canary auxiliary-weight replicas disagree")
                payload = _signed_payload(
                    {
                        "schema_version": 1,
                        "kind": CANARY_KIND,
                        "status": "passed",
                        "selected_workers_per_gpu": workers,
                        "alignment_auxiliary_weight": weight,
                        "calibration_relative_tolerance": 1e-6,
                        "canary_root": str(canary_root.resolve()),
                        "canary_job_count": len(records),
                        "jobs": records,
                        "contract_bindings": _contract_bindings(root),
                        "resources": resources,
                        "completed_at_utc": _utc_now(),
                    }
                )
                _write_json(result_path, payload)
                _verify_canary_result(root, config, result_path)
                _freeze_screen_specs(root, config, weight)
                registry = _read_registry(root)
                registry["formal_workers_per_gpu"] = workers
                registry["canary_result_path"] = str(result_path.resolve())
                registry["canary_result_sha256"] = _sha256_file(result_path)
                _write_registry(root, registry)
                _journal(root, "canary", "completed", workers_per_gpu=workers)
                return result_path
            except BaseException as exc:
                errors.append(f"{workers}/GPU: {type(exc).__name__}: {exc}")
                if index + 1 < len(candidates) and _capacity_related(canary_root, exc):
                    continue
                raise RuntimeError("Canary failed: " + "; ".join(errors)) from exc
        raise RuntimeError("All canary concurrency candidates failed")


def launch_screen(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _validate_contracts(root, config)
        registry = _read_registry(root)
        if not registry.get("screen_job_specs_frozen"):
            raise RuntimeError("Run canary before launch-screen")
        workers = int(registry.get("formal_workers_per_gpu", 0))
        if workers not in (8, 4):
            raise ValueError("Formal concurrency was not frozen by canary")
        _journal(root, "launch-screen", "running", workers_per_gpu=workers)
        result = _launch_records(
            registry["screen_jobs"],
            root,
            config,
            workers_per_gpu=workers,
            resume=resume,
        )
        if int(result["completed_jobs"]) != 16:
            raise RuntimeError("Screen did not complete all 16 jobs")
        registry = _read_registry(root)
        registry.update(status="screen_completed", screen_completed_at_utc=_utc_now())
        _write_registry(root, registry)
        _journal(root, "launch-screen", "completed", completed_jobs=16)
        return _registry_path(root)


def _selection_path(root: Path) -> Path:
    return root / "analysis/screen_selection.json"


def _verify_selection(path: Path) -> dict[str, Any]:
    selection = _read_json(path)
    if selection.get("status") != "completed":
        raise ValueError("Screen selection is not completed")
    eligible = bool(selection.get("confirmation_eligible"))
    winner = selection.get("winner_factor_code")
    if eligible and str(winner) not in SCREEN_FACTOR_CODES:
        raise ValueError("Eligible selection must contain a valid winner factor code")
    if not eligible and winner not in (None, ""):
        raise ValueError("Ineligible selection must not name a winner")
    return selection


def _load_frozen_selection(root: Path) -> dict[str, Any]:
    registry = _read_registry(root)
    path = _selection_path(root)
    if Path(str(registry.get("screen_selection_path", ""))).resolve() != path.resolve():
        raise ValueError("Frozen screen-selection path is missing or drifted")
    if not path.is_file() or _sha256_file(path) != registry.get(
        "screen_selection_sha256"
    ):
        raise ValueError("Frozen screen-selection SHA drift")
    binding_path = Path(
        str(registry.get("screen_selection_binding_path", ""))
    ).resolve()
    if not binding_path.is_file() or _sha256_file(binding_path) != registry.get(
        "screen_selection_binding_sha256"
    ):
        raise ValueError("Frozen screen-selection binding SHA drift")
    binding = _read_json(binding_path)
    _verify_signed_payload(binding, "screen-selection binding")
    if binding.get("path") != str(path.resolve()) or binding.get(
        "sha256"
    ) != _sha256_file(path):
        raise ValueError("Screen-selection binding content drift")
    return _verify_selection(path)


def analyze_screen(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    del resume
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _validate_contracts(root, config)
        registry = _read_registry(root)
        incomplete = [
            row["job_id"]
            for row in registry["screen_jobs"]
            if not _completed_valid(row, root)
        ]
        if incomplete:
            raise RuntimeError(
                f"Screen analysis requires 16 completed jobs: {incomplete}"
            )
        _journal(root, "analyze-screen", "running")
        path = _selection_path(root)
        if registry.get("screen_selection_path"):
            if Path(str(registry["screen_selection_path"])).resolve() != path.resolve():
                raise ValueError("Frozen screen-selection path drift")
            if _sha256_file(path) != registry.get("screen_selection_sha256"):
                raise ValueError("Frozen screen-selection SHA drift")
            binding_path = Path(
                str(registry["screen_selection_binding_path"])
            ).resolve()
            if _sha256_file(binding_path) != registry.get(
                "screen_selection_binding_sha256"
            ):
                raise ValueError("Frozen screen-selection binding drift")
            _verify_signed_payload(_read_json(binding_path), "screen-selection binding")
            selection = _verify_selection(path)
            _journal(root, "analyze-screen", "completed", resumed=True)
            return path
        if not path.is_file():
            core = _load_core()
            core.analyze_probe_screen(root)
        selection = _verify_selection(path)
        selection_binding = _signed_payload(
            {
                "schema_version": 1,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "confirmation_eligible": bool(selection["confirmation_eligible"]),
                "winner_factor_code": selection.get("winner_factor_code"),
            }
        )
        binding_path = _write_json(
            root / "contracts/screen_selection_binding.json", selection_binding
        )
        registry.update(
            status="screen_analyzed",
            screen_selection_path=str(path.resolve()),
            screen_selection_sha256=_sha256_file(path),
            screen_selection_binding_path=str(binding_path.resolve()),
            screen_selection_binding_sha256=_sha256_file(binding_path),
            confirmation_eligible=bool(selection["confirmation_eligible"]),
            winner_factor_code=selection.get("winner_factor_code"),
        )
        if not bool(selection["confirmation_eligible"]):
            registry["confirmation_status"] = "skipped_no_eligible_winner"
        _write_registry(root, registry)
        _journal(
            root,
            "analyze-screen",
            "completed",
            confirmation_eligible=bool(selection["confirmation_eligible"]),
            winner_factor_code=selection.get("winner_factor_code"),
        )
        return path


def _freeze_confirmation_specs(
    root: Path, config: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    registry = _read_registry(root)
    existing = list(registry.get("confirmation_jobs") or [])
    if existing:
        for record in existing:
            _verify_job_spec(record["job_spec_path"], record["job_spec_sha256"])
        return [dict(row) for row in existing]
    designs = build_confirmation_designs(config, selection)
    if not designs:
        registry["confirmation_status"] = "skipped_no_eligible_winner"
        _write_registry(root, registry)
        return []
    weight = float(registry["alignment_auxiliary_weight"])
    records = _materialize_job_specs(root, config, designs, auxiliary_weight=weight)
    registry.update(
        confirmation_jobs=records,
        confirmation_status="specs_frozen",
        confirmation_specs_frozen_at_utc=_utc_now(),
    )
    _write_registry(root, registry)
    return records


def launch_confirmation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _validate_contracts(root, config)
        selection = _load_frozen_selection(root)
        records = _freeze_confirmation_specs(root, config, selection)
        if not records:
            _journal(
                root, "launch-confirmation", "skipped", reason="no_eligible_winner"
            )
            return _registry_path(root)
        registry = _read_registry(root)
        workers = int(registry["formal_workers_per_gpu"])
        _journal(
            root,
            "launch-confirmation",
            "running",
            jobs=len(records),
            workers_per_gpu=workers,
        )
        result = _launch_records(
            records,
            root,
            config,
            workers_per_gpu=workers,
            resume=resume,
        )
        if int(result["completed_jobs"]) != len(records):
            raise RuntimeError("Confirmation did not complete its frozen job universe")
        registry = _read_registry(root)
        registry.update(
            status="confirmation_completed",
            confirmation_status="completed",
            confirmation_completed_at_utc=_utc_now(),
        )
        _write_registry(root, registry)
        _journal(root, "launch-confirmation", "completed", completed_jobs=len(records))
        return _registry_path(root)


def report(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _validate_contracts(root, config)
        registry = _read_registry(root)
        _load_frozen_selection(root)
        if registry.get("report_path"):
            reported = Path(str(registry["report_path"])).resolve()
            if not reported.is_file() or _sha256_file(reported) != registry.get(
                "report_sha256"
            ):
                raise ValueError("Frozen report binding drift")
            if not resume:
                raise FileExistsError("Report already exists; use --resume")
            _journal(root, "report", "completed", resumed=True)
            return reported
        if registry.get("confirmation_eligible"):
            incomplete = [
                row["job_id"]
                for row in registry.get("confirmation_jobs", [])
                if not _completed_valid(row, root)
            ]
            if incomplete:
                raise RuntimeError(
                    f"Report requires completed confirmation: {incomplete}"
                )
        _journal(root, "report", "running")
        core = _load_core()
        reported = Path(core.write_probe_report(root)).resolve()
        if not reported.is_file() or reported.stat().st_size == 0:
            raise ValueError("Report core did not produce a non-empty report")
        registry = _read_registry(root)
        registry.update(
            status="reported",
            report_path=str(reported),
            report_sha256=_sha256_file(reported),
        )
        _write_registry(root, registry)
        _journal(root, "report", "completed", report_path=str(reported))
        return reported


def _output_hash_manifest(root: Path) -> Path:
    path = root / "qa/output_hashes.csv"
    files = [
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.resolve() != path.resolve()
    ]
    rows = [
        {
            "path": candidate.relative_to(root).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(files, key=lambda value: value.as_posix())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def _verify_output_hash_manifest(path: Path, root: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not frame:
        raise ValueError("Output hash manifest is empty")
    observed_paths: set[str] = set()
    for row in frame:
        relative = str(row.get("path", ""))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(
                "Output hash manifest escapes the experiment root"
            ) from exc
        if relative in observed_paths:
            raise ValueError(f"Duplicate output hash row: {relative}")
        observed_paths.add(relative)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError(f"Output artifact size drift: {candidate}")
        if _sha256_file(candidate) != str(row.get("sha256", "")):
            raise ValueError(f"Output artifact SHA drift: {candidate}")
    expected_paths = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.resolve() != path.resolve()
    }
    if observed_paths != expected_paths:
        raise ValueError(
            "Output hash manifest file universe drift: "
            f"missing={sorted(expected_paths - observed_paths)[:5]}, "
            f"extra={sorted(observed_paths - expected_paths)[:5]}"
        )


def qa(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    config = load_config(config_path)
    root = Path(output_dir).resolve()
    with _exclusive_lock(root):
        _journal(root, "qa", "running")
        _validate_contracts(root, config)
        registry = _read_registry(root)
        if registry.get("terminal_qa_path"):
            terminal_path = Path(str(registry["terminal_qa_path"])).resolve()
            if not terminal_path.is_file() or _sha256_file(
                terminal_path
            ) != registry.get("terminal_qa_sha256"):
                raise ValueError("Frozen terminal-QA binding drift")
            _verify_signed_payload(_read_json(terminal_path), "terminal QA")
            if not resume:
                raise FileExistsError("Terminal QA already exists; use --resume")
            hash_path = root / "qa/output_hashes.csv"
            _verify_output_hash_manifest(hash_path, root)
            return terminal_path
        screen_jobs = list(registry.get("screen_jobs") or [])
        if len(screen_jobs) != 16 or len({row["job_id"] for row in screen_jobs}) != 16:
            raise ValueError("Terminal QA requires exactly 16 unique screen jobs")
        if any(not _completed_valid(row, root) for row in screen_jobs):
            raise RuntimeError("Terminal QA found incomplete or drifted screen jobs")
        selection = _load_frozen_selection(root)
        confirmation = list(registry.get("confirmation_jobs") or [])
        if bool(selection["confirmation_eligible"]):
            expected = 4 if str(selection["winner_factor_code"])[3] == "1" else 3
            if len(confirmation) != expected:
                raise ValueError("Conditional confirmation job count drift")
            if any(not _completed_valid(row, root) for row in confirmation):
                raise RuntimeError("Terminal QA found incomplete confirmation jobs")
        elif confirmation:
            raise ValueError("Ineligible selection unexpectedly has confirmation jobs")
        if (
            registry.get("test_loader_count") != 0
            or registry.get("prediction_count") != 0
        ):
            raise ValueError("The mechanism probe opened test data or predictions")
        forbidden = [
            path
            for path in root.rglob("*")
            if path.name.lower() in {"test", "tests", "predictions", "evaluation"}
        ]
        if forbidden:
            raise ValueError(f"Forbidden test/prediction artifacts: {forbidden}")
        report_path = Path(str(registry.get("report_path", ""))).resolve()
        if not report_path.is_file() or _sha256_file(report_path) != registry.get(
            "report_sha256"
        ):
            raise ValueError("Terminal report binding drift")
        terminal = _signed_payload(
            {
                "schema_version": 1,
                "kind": "film_unet_text_signal_probe_terminal_qa_v1",
                "status": "passed",
                "screen_jobs": 16,
                "confirmation_jobs": len(confirmation),
                "test_loader_count": 0,
                "prediction_count": 0,
                "output_hash_manifest_path": str(
                    (root / "qa/output_hashes.csv").resolve()
                ),
                "output_hash_manifest_excludes_itself": True,
                "completed_at_utc": _utc_now(),
            }
        )
        terminal_path = _write_json(root / "qa/terminal_qa.json", terminal)
        registry = _read_registry(root)
        registry.update(
            status="completed",
            terminal_qa_path=str(terminal_path.resolve()),
            terminal_qa_sha256=_sha256_file(terminal_path),
        )
        _write_registry(root, registry)
        _journal(root, "qa", "completed")
        hashes = _output_hash_manifest(root)
        _verify_output_hash_manifest(hashes, root)
        return terminal_path


def status(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    if not _registry_path(root).is_file():
        return {
            "root": str(root),
            "status": "not_prepared",
            "screen_completed": 0,
            "screen_total": 16,
            "confirmation_completed": 0,
            "confirmation_total": 0,
        }
    registry = _read_registry(root)
    screen = list(registry.get("screen_jobs") or [])
    confirmation = list(registry.get("confirmation_jobs") or [])
    return {
        "root": str(root),
        "status": registry.get("status"),
        "formal_workers_per_gpu": registry.get("formal_workers_per_gpu"),
        "screen_completed": sum(_completed_valid(row, root) for row in screen),
        "screen_total": len(screen),
        "confirmation_eligible": registry.get("confirmation_eligible"),
        "winner_factor_code": registry.get("winner_factor_code"),
        "confirmation_status": registry.get("confirmation_status"),
        "confirmation_completed": sum(
            _completed_valid(row, root) for row in confirmation
        ),
        "confirmation_total": len(confirmation),
        "test_loader_count": registry.get("test_loader_count", 0),
        "prediction_count": registry.get("prediction_count", 0),
    }


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    resume: bool = False,
) -> Path:
    root = Path(output_dir).resolve()
    if _registry_path(root).is_file():
        registry = _read_registry(root)
        if registry.get("status") == "completed":
            if not resume:
                raise FileExistsError("Completed probe root requires --resume")
            config = load_config(config_path)
            _validate_contracts(root, config)
            terminal = Path(str(registry["terminal_qa_path"])).resolve()
            if _sha256_file(terminal) != registry.get("terminal_qa_sha256"):
                raise ValueError("Completed terminal-QA binding drift")
            _verify_signed_payload(_read_json(terminal), "terminal QA")
            _verify_output_hash_manifest(root / "qa/output_hashes.csv", root)
            return terminal
    with _exclusive_lock(root):
        control = _control_dir(root)
        _atomic_write_text(control / "pipeline.pid", f"{os.getpid()}\n")
        try:
            prepare(config_path, output_dir, resume=resume)
            canary(config_path, output_dir, resume=resume)
            launch_screen(config_path, output_dir, resume=resume)
            analyze_screen(config_path, output_dir, resume=resume)
            launch_confirmation(config_path, output_dir, resume=resume)
            report(config_path, output_dir, resume=resume)
            _journal(root, "run-pipeline", "completed")
            result = qa(config_path, output_dir, resume=resume)
            return result
        except BaseException as exc:
            _journal(
                root,
                "run-pipeline",
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            (control / "pipeline.pid").unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTION_NAMES)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "status":
        print(json.dumps(status(args.output_dir), indent=2, sort_keys=True))
        return 0
    handlers = {
        "prepare": prepare,
        "canary": canary,
        "launch-screen": launch_screen,
        "analyze-screen": analyze_screen,
        "launch-confirmation": launch_confirmation,
        "report": report,
        "qa": qa,
        "run-pipeline": run_pipeline,
    }
    result = handlers[args.action](
        args.config,
        args.output_dir,
        resume=args.resume,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_NAMES",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "SCREEN_FACTOR_CODES",
    "analyze_screen",
    "build_confirmation_designs",
    "build_screen_designs",
    "canary",
    "launch_confirmation",
    "launch_screen",
    "load_config",
    "main",
    "prepare",
    "qa",
    "report",
    "run_pipeline",
    "status",
    "validate_config",
]
