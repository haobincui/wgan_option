"""Reproducible dual-GPU orchestration for the news-first vol comparison.

The module deliberately stays outside ``src/``: it coordinates the existing
WGAN and deterministic-regression trainers without changing their public CLI.
Each worker is an isolated process with one visible physical GPU, while the
launcher owns wave ordering, fail-closed cleanup, status files, and resource
telemetry.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from wgan_option.utils.text_ablation import (
    REAL_TEXT,
    TEXT_ABLATION_MODES,
    TEXT_SHUFFLE,
    fixed_text_shuffle,
    normalize_text_ablation_mode,
    text_information_path,
)


ROOT_KEY = "news_first_vol_training"
REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCES = (5, 10, 15, 30)
MODEL_FAMILIES = ("wgan", "regression")
CAPACITY_PROFILE_SHAPE_FIELDS = (
    "gen_base_channels",
    "gen_text_hidden_dim",
    "gen_text_out_dim",
    "gen_hidden_dim",
    "disc_base_channels",
    "disc_text_hidden_dim",
    "disc_hidden_dim",
)
CAPACITY_PROFILE_METADATA_FIELDS = (
    "expected_regression_parameters",
    "expected_wgan_parameters",
)
CAPACITY_PROFILE_NAMES = ("micro", "tiny", "small", "medium", "large", "legacy")
FROZEN_CAPACITY_PROFILES: dict[str, dict[str, int]] = {
    "micro": {
        "gen_base_channels": 1,
        "gen_text_hidden_dim": 8,
        "gen_text_out_dim": 4,
        "gen_hidden_dim": 32,
        "disc_base_channels": 1,
        "disc_text_hidden_dim": 4,
        "disc_hidden_dim": 24,
        "expected_regression_parameters": 20_070,
        "expected_wgan_parameters": 25_850,
    },
    "tiny": {
        "gen_base_channels": 2,
        "gen_text_hidden_dim": 16,
        "gen_text_out_dim": 8,
        "gen_hidden_dim": 64,
        "disc_base_channels": 2,
        "disc_text_hidden_dim": 8,
        "disc_hidden_dim": 48,
        "expected_regression_parameters": 46_528,
        "expected_wgan_parameters": 59_227,
    },
    "small": {
        "gen_base_channels": 4,
        "gen_text_hidden_dim": 32,
        "gen_text_out_dim": 16,
        "gen_hidden_dim": 128,
        "disc_base_channels": 4,
        "disc_text_hidden_dim": 16,
        "disc_hidden_dim": 96,
        "expected_regression_parameters": 119_376,
        "expected_wgan_parameters": 149_333,
    },
    "medium": {
        "gen_base_channels": 8,
        "gen_text_hidden_dim": 64,
        "gen_text_out_dim": 32,
        "gen_hidden_dim": 256,
        "disc_base_channels": 8,
        "disc_text_hidden_dim": 32,
        "disc_hidden_dim": 192,
        "expected_regression_parameters": 344_800,
        "expected_wgan_parameters": 422_953,
    },
    "large": {
        "gen_base_channels": 12,
        "gen_text_hidden_dim": 96,
        "gen_text_out_dim": 48,
        "gen_hidden_dim": 384,
        "disc_base_channels": 12,
        "disc_text_hidden_dim": 48,
        "disc_hidden_dim": 288,
        "expected_regression_parameters": 676_528,
        "expected_wgan_parameters": 821_117,
    },
    "legacy": {
        "gen_base_channels": 32,
        "gen_text_hidden_dim": 256,
        "gen_text_out_dim": 128,
        "gen_hidden_dim": 1024,
        "disc_base_channels": 32,
        "disc_text_hidden_dim": 128,
        "disc_hidden_dim": 786,
        "expected_regression_parameters": 3_929_728,
        "expected_wgan_parameters": 4_691_653,
    },
}
CAPACITY_STAGES = (
    "regression_screen",
    "regression_confirm",
    "wgan_screen",
    "wgan_confirm",
)
CAPACITY_SCREEN_TOLERANCES = (5, 30)
CAPACITY_SCREEN_TEXT_MODES = ("current_only", REAL_TEXT)
LEGACY_RESIDUAL_OUTPUT_MODE = "legacy_softplus"
SUPPORTED_RESIDUAL_OUTPUT_MODES = {
    LEGACY_RESIDUAL_OUTPUT_MODE,
    "identity_softplus_residual",
}
RAW_JOINT_SUPPORT_METHOD = "raw_bracket_intersection_v1"
TERMINAL_SUCCESS = {"completed", "dry_run_passed"}
EXPECTED_SPLIT_COUNTS = {
    5: {"train_rows": 1234, "train_pairs": 952, "train_sessions": 222},
    10: {"train_rows": 1506, "train_pairs": 1115, "train_sessions": 234},
    15: {"train_rows": 1748, "train_pairs": 1230, "train_sessions": 242},
    30: {"train_rows": 2263, "train_pairs": 1452, "train_sessions": 254},
}
EXPECTED_COMMON_VALIDATION = {"rows": 176, "pairs": 161, "sessions": 37}
EXPECTED_COMMON_TEST = {"rows": 200, "pairs": 170, "sessions": 49}
EXPECTED_RAW_JOINT_SPLIT_COUNTS = {
    5: {"train_rows": 936, "train_pairs": 720, "train_sessions": 210},
    10: {"train_rows": 1123, "train_pairs": 836, "train_sessions": 219},
    15: {"train_rows": 1273, "train_pairs": 911, "train_sessions": 225},
    30: {"train_rows": 1591, "train_pairs": 1050, "train_sessions": 237},
}
EXPECTED_RAW_JOINT_COMMON_VALIDATION = {"rows": 133, "pairs": 123, "sessions": 33}
EXPECTED_RAW_JOINT_COMMON_TEST = {"rows": 152, "pairs": 130, "sessions": 45}
EXPECTED_RAW_JOINT_BROAD_TEST = {"rows": 245, "pairs": 182, "sessions": 51}
DIRECT_CODE_PATHS = (
    "scripts/rq3/main.py",
    "scripts/rq3/news_first_vol_training.py",
    "scripts/rq3/news_first_vol_comparison_analysis.py",
    "scripts/rq3/news_first_vol_training_report.py",
    "scripts/rq3/news_first_vol_capacity_analysis.py",
    "scripts/rq3/news_first_vol_capacity_report.py",
    "src/film_wgan/support.py",
    "src/wgan_option/merge_support.py",
    "src/trainer.py",
    "src/wgan_option/config.py",
    "src/wgan_option/trainer.py",
    "src/wgan_option/models/gan_model.py",
    "src/wgan_option/models/generator.py",
    "src/wgan_option/models/discriminator.py",
    "src/wgan_option/models/common.py",
    "src/wgan_option/models/vol_regressor.py",
    "src/wgan_option/train_vol_xlsx.py",
    "src/wgan_option/train_vol_regression_xlsx.py",
    "src/wgan_option/utils/merged_xlsx.py",
    "src/wgan_option/utils/merged_xlsx_dataloaders.py",
    "src/wgan_option/utils/merged_xlsx_parsing.py",
    "src/wgan_option/utils/news_first_dataloaders.py",
    "src/wgan_option/utils/merged_xlsx_samples.py",
    "src/wgan_option/utils/merged_xlsx_types.py",
    "src/wgan_option/utils/inference_helpers.py",
    "src/wgan_option/utils/reproducibility.py",
    "src/wgan_option/utils/training_artifacts.py",
    "src/wgan_option/utils/text_ablation.py",
    "src/wgan_option/utils/vol_forecast_metrics.py",
    "src/wgan_option/utils/weighted_training.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    return _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    return _atomic_write_text(
        path,
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False),
    )


def _write_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)
    return target


def _write_csv_gzip(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Path:
    """Write a deterministic gzip CSV suitable for lineage hashing."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(fields),
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, target)
    return target


def _configured_text_ablation_modes(datasets: Mapping[str, Any]) -> tuple[str, ...]:
    raw_modes = datasets.get("text_ablation_modes", (REAL_TEXT,))
    if isinstance(raw_modes, str):
        raw_modes = [raw_modes]
    modes = tuple(normalize_text_ablation_mode(value) for value in raw_modes)
    if not modes:
        raise ValueError("datasets.text_ablation_modes must not be empty")
    if len(set(modes)) != len(modes):
        raise ValueError("datasets.text_ablation_modes must be unique")
    return modes


def _support_mask_mode(datasets: Mapping[str, Any]) -> str:
    mode = str(datasets.get("support_mask_mode", "none") or "none").strip().lower()
    if mode not in {"none", "raw_joint"}:
        raise ValueError("datasets.support_mask_mode must be one of none/raw_joint")
    return mode


def _capacity_sweep_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = config.get("capacity_sweep")
    if raw is None:
        return None
    sweep = _require_mapping(raw, "capacity_sweep")
    return sweep if bool(sweep.get("enabled", False)) else None


def _capacity_profile_payload(name: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    normalized_name = str(name).strip().lower()
    return {
        "schema_version": 1,
        "capacity_profile": normalized_name,
        "shape": {
            field: int(profile[field]) for field in CAPACITY_PROFILE_SHAPE_FIELDS
        },
    }


def _capacity_profile_sha256(name: str, profile: Mapping[str, Any]) -> str:
    return _payload_sha256(_capacity_profile_payload(name, profile))


def _capacity_profiles(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    sweep = _capacity_sweep_config(config)
    if sweep is None:
        return {}
    raw_profiles = _require_mapping(sweep.get("profiles"), "capacity_sweep.profiles")
    observed_names = tuple(str(name).strip().lower() for name in raw_profiles)
    if observed_names != CAPACITY_PROFILE_NAMES:
        raise ValueError(
            "capacity_sweep.profiles must preserve the frozen order and names: "
            f"{list(CAPACITY_PROFILE_NAMES)}"
        )
    profiles: dict[str, dict[str, int]] = {}
    allowed = set(CAPACITY_PROFILE_SHAPE_FIELDS + CAPACITY_PROFILE_METADATA_FIELDS)
    for name, raw_profile in raw_profiles.items():
        normalized_name = str(name).strip().lower()
        profile = _require_mapping(raw_profile, f"capacity_sweep.profiles.{name}")
        unknown = sorted(set(profile) - allowed)
        missing = sorted(allowed - set(profile))
        if unknown or missing:
            raise ValueError(
                f"capacity profile {normalized_name} has invalid fields; "
                f"missing={missing}, unknown={unknown}"
            )
        normalized = {field: int(profile[field]) for field in allowed}
        expected = FROZEN_CAPACITY_PROFILES[normalized_name]
        if normalized != expected:
            raise ValueError(
                f"capacity profile {normalized_name} differs from the frozen shape/counts: "
                f"expected={expected}, observed={normalized}"
            )
        if any(normalized[field] <= 0 for field in allowed):
            raise ValueError(
                f"capacity profile {normalized_name} fields must be positive"
            )
        profiles[normalized_name] = normalized
    return profiles


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a YAML mapping")
    return dict(value)


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = _resolve_repo_path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _require_mapping(payload, "training config")
    if ROOT_KEY not in root:
        raise ValueError(f"Missing top-level '{ROOT_KEY}' in {path}")
    config = _require_mapping(root[ROOT_KEY], ROOT_KEY)
    config["_source_config_path"] = str(path)
    return config


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    datasets = _require_mapping(config.get("datasets"), "datasets")
    split = _require_mapping(config.get("split"), "split")
    runtime = _require_mapping(config.get("runtime"), "runtime")
    waves = list(config.get("waves") or [])
    models = _require_mapping(config.get("models"), "models")
    capacity_sweep = _capacity_sweep_config(config)
    capacity_enabled = capacity_sweep is not None

    tolerances = tuple(int(value) for value in datasets.get("tolerances_minutes", ()))
    if tolerances != TOLERANCES:
        raise ValueError(f"tolerances_minutes is frozen to {list(TOLERANCES)}")
    if int(datasets.get("common_evaluation_tolerance_minutes", -1)) != 5:
        raise ValueError("common_evaluation_tolerance_minutes is frozen to 5")
    if str(datasets.get("sheet_name", "")) != "gan_input_ready":
        raise ValueError("sheet_name is frozen to gan_input_ready")
    if str(datasets.get("text_embedding_mode", "")).lower() != "lp":
        raise ValueError("text_embedding_mode is frozen to lp")
    if int(datasets.get("seed", -1)) != 42:
        raise ValueError("seed is frozen to 42")
    text_modes = _configured_text_ablation_modes(datasets)
    if int(datasets.get("text_shuffle_seed", datasets.get("seed", -1))) != int(
        datasets["seed"]
    ):
        raise ValueError("text_shuffle_seed must equal the frozen training seed")
    support_mode = _support_mask_mode(datasets)
    if len(text_modes) > 1 and tuple(text_modes) != TEXT_ABLATION_MODES:
        raise ValueError(
            "The formal text ablation is frozen to real_text/current_only/text_shuffle"
        )
    if len(text_modes) > 1 and support_mode != "raw_joint":
        raise ValueError(
            "The formal text ablation requires support_mask_mode=raw_joint"
        )
    if capacity_enabled:
        if tuple(text_modes) != TEXT_ABLATION_MODES:
            raise ValueError(
                "The capacity sweep is frozen to real_text/current_only/text_shuffle"
            )
        if support_mode != "raw_joint":
            raise ValueError("The capacity sweep requires support_mask_mode=raw_joint")
        _capacity_profiles(config)
    if str(split.get("train_end_utc")) != "2023-07-01T00:00:00Z":
        raise ValueError("train_end_utc is frozen to 2023-07-01T00:00:00Z")
    if str(split.get("validation_end_utc")) != "2023-10-01T00:00:00Z":
        raise ValueError("validation_end_utc is frozen to 2023-10-01T00:00:00Z")
    if int(runtime.get("slots_per_gpu", -1)) != 2:
        raise ValueError("slots_per_gpu is frozen to 2")
    if int(runtime.get("cpu_threads_per_job", -1)) != 8:
        raise ValueError("cpu_threads_per_job is frozen to 8")
    gpu_ids = tuple(int(value) for value in runtime.get("gpu_ids", ()))
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("runtime.gpu_ids must contain exactly two distinct GPUs")
    if capacity_enabled:
        if waves:
            raise ValueError(
                "capacity_sweep dynamically materializes waves; remove the static waves list"
            )
        selection = _require_mapping(
            capacity_sweep.get("selection"), "capacity_sweep.selection"
        )
        if int(selection.get("shortlist_size", -1)) != 2:
            raise ValueError("capacity_sweep.selection.shortlist_size is frozen to 2")
        if float(selection.get("minimum_mean_improvement_fraction", -1.0)) != 0.005:
            raise ValueError(
                "capacity_sweep.selection.minimum_mean_improvement_fraction is frozen to 0.005"
            )
        if int(selection.get("bootstrap_clusters", -1)) != 10_000:
            raise ValueError(
                "capacity_sweep.selection.bootstrap_clusters is frozen to 10000"
            )
        if str(selection.get("selection_rule", "")) != "one_standard_error_smallest":
            raise ValueError(
                "capacity_sweep.selection.selection_rule is frozen to "
                "one_standard_error_smallest"
            )
    else:
        expected_waves = tuple(
            (family, mode) for family in MODEL_FAMILIES for mode in text_modes
        )
        observed_waves = tuple(
            (
                str(_require_mapping(item, "wave").get("model")),
                normalize_text_ablation_mode(
                    _require_mapping(item, "wave").get(
                        "text_ablation_mode",
                        text_modes[0] if len(text_modes) == 1 else "",
                    )
                ),
            )
            for item in waves
        )
        if observed_waves != expected_waves:
            raise ValueError(
                "waves must cover model x text_ablation_mode in frozen order: "
                f"expected={expected_waves}, observed={observed_waves}"
            )
    residual_output_modes: dict[str, str] = {}
    expected_patience = (
        {"wgan": 16, "regression": 12}
        if capacity_enabled
        else {"wgan": 12, "regression": 10}
    )
    for family in MODEL_FAMILIES:
        patience = expected_patience[family]
        model = _require_mapping(models.get(family), f"models.{family}")
        training = _require_mapping(model.get("training"), f"models.{family}.training")
        residual_output_mode = (
            str(training.get("residual_output_mode", LEGACY_RESIDUAL_OUTPUT_MODE))
            .strip()
            .lower()
        )
        if residual_output_mode not in SUPPORTED_RESIDUAL_OUTPUT_MODES:
            raise ValueError(
                f"{family} residual_output_mode must be one of "
                f"{sorted(SUPPORTED_RESIDUAL_OUTPUT_MODES)}"
            )
        residual_output_modes[family] = residual_output_mode
        if int(training.get("num_epochs", -1)) != 100:
            raise ValueError(f"{family} num_epochs is frozen to 100")
        if int(training.get("early_stopping_patience", -1)) != patience:
            raise ValueError(
                f"{family} early_stopping_patience is frozen to {patience}"
            )
        if str(training.get("best_checkpoint_metric")) != "val_hybrid_score":
            raise ValueError(
                f"{family} best_checkpoint_metric is frozen to val_hybrid_score"
            )
        if not bool(training.get("use_early_stopping")):
            raise ValueError(f"{family} early stopping must remain enabled")
        if int(training.get("save_every", 0)) <= int(training["num_epochs"]):
            raise ValueError(
                f"{family} save_every must exceed num_epochs (best/final only)"
            )
        if len(text_modes) > 1:
            if residual_output_mode != "identity_softplus_residual":
                raise ValueError(
                    "The formal masked text ablation requires identity_softplus_residual"
                )
            if not bool(training.get("evaluate_initial_checkpoint", False)):
                raise ValueError(
                    "The formal masked text ablation requires evaluate_initial_checkpoint=true"
                )
        if capacity_enabled:
            if int(training.get("early_stopping_min_epochs", -1)) != 16:
                raise ValueError(f"{family} early_stopping_min_epochs is frozen to 16")
            if int(training.get("reduce_lr_patience", -1)) != 3:
                raise ValueError(f"{family} reduce_lr_patience is frozen to 3")
            if float(training.get("learning_rate", -1.0)) != 1.0e-4:
                raise ValueError(f"{family} learning_rate is frozen to 1e-4")
            if float(training.get("reduce_lr_factor", -1.0)) != 0.5:
                raise ValueError(f"{family} reduce_lr_factor is frozen to 0.5")
            if float(training.get("reduce_lr_min_lr", -1.0)) != 1.0e-5:
                raise ValueError(f"{family} reduce_lr_min_lr is frozen to 1e-5")
    if len(set(residual_output_modes.values())) != 1:
        raise ValueError(
            "WGAN and regression must use the same residual_output_mode for a "
            f"controlled comparison; got {residual_output_modes}"
        )


def _resolved_config(config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path)
    _validate_frozen_config(config)
    resolved = deepcopy(config)
    resolved.pop("_source_config_path", None)
    datasets = resolved["datasets"]
    datasets["text_ablation_modes"] = list(_configured_text_ablation_modes(datasets))
    datasets["text_shuffle_seed"] = int(
        datasets.get("text_shuffle_seed", datasets["seed"])
    )
    datasets["support_mask_mode"] = _support_mask_mode(datasets)
    resolved["datasets"]["root"] = str(_resolve_repo_path(resolved["datasets"]["root"]))
    python_value = str(resolved["runtime"].get("python_executable", sys.executable))
    resolved["runtime"]["python_executable"] = str(_resolve_repo_path(python_value))
    resolved["source_config_path"] = str(_resolve_repo_path(config_path))
    return resolved


def _workbook_path(dataset_root: Path, tolerance: int, template: str) -> Path:
    return dataset_root / template.format(
        tolerance=int(tolerance), tolerance02=f"{int(tolerance):02d}"
    )


def _declared_dataset_hashes(dataset_root: Path) -> dict[str, str]:
    manifest = dataset_root / "dataset_output_sha256.txt"
    if not manifest.exists():
        return {}
    result: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            result[fields[1].strip()] = fields[0].strip().lower()
    return result


def _source_rows(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    datasets = _require_mapping(resolved["datasets"], "datasets")
    dataset_root = Path(datasets["root"])
    template = str(
        datasets.get("workbook_template", "tolerance_{tolerance02}m/merged_vol.xlsx")
    )
    declared = _declared_dataset_hashes(dataset_root)
    declaration_exists = (dataset_root / "dataset_output_sha256.txt").is_file()

    def verified_dataset_digest(path: Path) -> str:
        relative = path.relative_to(dataset_root).as_posix()
        actual = _sha256_file(path)
        if declaration_exists:
            expected = declared.get(relative)
            if expected is None:
                raise ValueError(
                    f"Dataset hash declaration is missing a required source: {relative}"
                )
            if actual.lower() != expected.lower():
                raise ValueError(
                    "Dataset source hash mismatch: "
                    f"{relative}; declared={expected}; actual={actual}"
                )
        return actual

    rows: list[dict[str, Any]] = []
    for tolerance in TOLERANCES:
        workbook = _workbook_path(dataset_root, tolerance, template)
        if not workbook.is_file():
            raise FileNotFoundError(f"Missing tolerance workbook: {workbook}")
        digest = verified_dataset_digest(workbook)
        rows.append(
            {
                "source_role": f"training_workbook_{tolerance:02d}m",
                "path": str(workbook),
                "sha256": digest,
                "size_bytes": workbook.stat().st_size,
            }
        )
        if _support_mask_mode(datasets) == "raw_joint":
            support_audit = (
                dataset_root
                / f"tolerance_{int(tolerance):02d}m"
                / "surface_support_audit.csv.gz"
            )
            if not support_audit.is_file():
                raise FileNotFoundError(
                    f"Missing raw-joint support audit: {support_audit}"
                )
            rows.append(
                {
                    "source_role": f"support_audit_{tolerance:02d}m",
                    "path": str(support_audit),
                    "sha256": verified_dataset_digest(support_audit),
                    "size_bytes": support_audit.stat().st_size,
                }
            )
    summary = dataset_root / "dataset_summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"Missing dataset validation summary: {summary}")
    rows.append(
        {
            "source_role": "dataset_summary",
            "path": str(summary),
            "sha256": verified_dataset_digest(summary),
            "size_bytes": summary.stat().st_size,
        }
    )
    source_config = Path(str(resolved["source_config_path"]))
    rows.append(
        {
            "source_role": "orchestration_config",
            "path": str(source_config),
            "sha256": _sha256_file(source_config),
            "size_bytes": source_config.stat().st_size,
        }
    )
    return rows


def _code_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in DIRECT_CODE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment code is missing: {path}")
        rows.append(
            {
                "path": str(path.resolve()),
                "relative_path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def _git_snapshot() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit_value = commit.stdout.strip() if commit.returncode == 0 else ""
    status_lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "git_commit": commit_value,
        "git_commit_status": "ok" if commit.returncode == 0 else "unavailable",
        "git_dirty": bool(status_lines),
        "git_status_porcelain": status_lines,
        "git_status_error": status.stderr.strip() if status.returncode != 0 else "",
        "commit_scope_note": (
            "git_commit identifies the base commit only; dirty/untracked working-tree "
            "content is captured independently by code_hashes.csv and is not claimed "
            "to be contained in that commit."
        ),
    }


def _validate_dataset_summary(dataset_root: Path) -> None:
    with (dataset_root / "dataset_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_tolerance = {int(row["tolerance_minutes"]): row for row in rows}
    if set(by_tolerance) != set(TOLERANCES):
        raise ValueError(
            "dataset_summary.csv must contain exactly 5/10/15/30-minute rows"
        )
    failed = [
        value
        for value, row in by_tolerance.items()
        if str(row.get("status", "")).lower() != "pass"
    ]
    if failed:
        raise ValueError(f"Dataset validation did not pass for tolerances: {failed}")


def _read_split_keys(workbook: Path, sheet_name: str):
    import pandas as pd

    frame = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        usecols=[
            "sample_id",
            "news_row_id",
            "effective_origin_utc",
            "pair_id",
            "session_id",
        ],
        dtype=str,
        engine="openpyxl",
    )
    frame["effective_origin_utc"] = pd.to_datetime(
        frame["effective_origin_utc"], errors="coerce", utc=True
    )
    if frame["effective_origin_utc"].isna().any():
        raise ValueError(f"Invalid effective_origin_utc in {workbook}")
    for column in ("pair_id", "session_id"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Missing {column} in {workbook}")
    frame["sample_id"] = frame["sample_id"].fillna("").astype(str).str.strip()
    if frame["sample_id"].eq("").any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"sample_id must be non-empty and unique in {workbook}")
    return frame


def _split_counts(frame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "pairs": int(frame["pair_id"].nunique()),
        "sessions": int(frame["session_id"].nunique()),
    }


def _read_supported_pair_ids(dataset_root: Path, tolerance: int) -> set[str]:
    """Read the generated pair-grain raw-support audit and fail closed."""

    import pandas as pd

    path = (
        dataset_root
        / f"tolerance_{int(tolerance):02d}m"
        / "surface_support_audit.csv.gz"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"support_mask_mode=raw_joint requires the generated support audit: {path}"
        )
    frame = pd.read_csv(
        path,
        usecols=[
            "pair_id",
            "surface_training_eligible",
            "joint_zero_support",
            "joint_strict_support_cell_count",
            "support_method",
            "grid_fingerprint",
        ],
        low_memory=False,
    )
    if frame.empty or frame["pair_id"].astype(str).duplicated().any():
        raise ValueError(f"Support audit must contain one row per unique pair: {path}")
    methods = set(frame["support_method"].dropna().astype(str).str.strip())
    if methods != {RAW_JOINT_SUPPORT_METHOD}:
        raise ValueError(f"Support audit method mismatch in {path}: {sorted(methods)}")
    fingerprints = set(frame["grid_fingerprint"].dropna().astype(str).str.strip())
    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError(
            f"Support audit must contain one non-empty grid fingerprint: {path}"
        )

    def truthy(values):
        return values.map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )

    eligible = truthy(frame["surface_training_eligible"])
    zero = truthy(frame["joint_zero_support"])
    counts = pd.to_numeric(frame["joint_strict_support_cell_count"], errors="coerce")
    inconsistent = eligible & ((~zero & counts.le(0)) | (zero & counts.gt(0)))
    if inconsistent.any():
        raise ValueError(f"Support audit zero/count fields are inconsistent: {path}")
    supported = frame.loc[eligible & ~zero & counts.gt(0), "pair_id"].astype(str)
    return set(supported)


def _support_audit_grid_fingerprint(dataset_root: Path, tolerance: int) -> str:
    import pandas as pd

    path = (
        dataset_root
        / f"tolerance_{int(tolerance):02d}m"
        / "surface_support_audit.csv.gz"
    )
    frame = pd.read_csv(path, usecols=["grid_fingerprint"], low_memory=False)
    values = set(frame["grid_fingerprint"].dropna().astype(str).str.strip())
    if len(values) != 1 or "" in values:
        raise ValueError(f"Invalid support-audit grid fingerprint: {path}")
    return next(iter(values))


def _apply_support_eligibility(
    frame,
    *,
    dataset_root: Path,
    tolerance: int,
    support_mode: str,
):
    if support_mode == "none":
        return frame.copy()
    supported_pairs = _read_supported_pair_ids(dataset_root, tolerance)
    output = frame[frame["pair_id"].astype(str).isin(supported_pairs)].copy()
    if output.empty:
        raise ValueError(
            f"{tolerance}m contains no samples with positive raw joint support"
        )
    return output


def _write_text_ablation_audit(
    experiment_root: Path,
    *,
    split_frames: Mapping[str, tuple[int, Any]],
    modes: Sequence[str],
    seed: int,
    support_mode: str,
) -> tuple[Path, Path | None]:
    """Persist split-scoped treatment definitions and fixed shuffle donors."""

    summary_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for split_name, (tolerance, frame) in split_frames.items():
        ordered = frame.sort_values("sample_id", kind="stable").reset_index(drop=True)
        keys = ordered["sample_id"].astype(str).tolist()
        universe_sha256 = _payload_sha256(sorted(keys))
        for mode in modes:
            normalized = normalize_text_ablation_mode(mode)
            mapping_sha256 = ""
            fixed_points: int | str = ""
            if normalized == TEXT_SHUFFLE:
                mapping = fixed_text_shuffle(
                    keys,
                    seed=int(seed),
                    namespace=split_name,
                    pair_ids=ordered["pair_id"].astype(str).tolist(),
                    session_ids=ordered["session_id"].astype(str).tolist(),
                )
                mapping_sha256 = mapping.mapping_sha256
                fixed_points = mapping.fixed_point_count
                donors = mapping.donor_by_receiver()
                lookup = ordered.set_index("sample_id", drop=False)
                for receiver in mapping.receiver_keys:
                    donor = donors[receiver]
                    receiver_row = lookup.loc[receiver]
                    donor_row = lookup.loc[donor]
                    mapping_rows.append(
                        {
                            "split": split_name,
                            "tolerance_minutes": int(tolerance),
                            "receiver_sample_id": receiver,
                            "receiver_news_row_id": receiver_row["news_row_id"],
                            "receiver_pair_id": receiver_row["pair_id"],
                            "receiver_session_id": receiver_row["session_id"],
                            "donor_sample_id": donor,
                            "donor_news_row_id": donor_row["news_row_id"],
                            "donor_pair_id": donor_row["pair_id"],
                            "donor_session_id": donor_row["session_id"],
                            "text_shuffle_seed": int(seed),
                            "text_shuffle_method": mapping.method,
                            "mapping_sha256": mapping.mapping_sha256,
                            "universe_sha256": mapping.universe_sha256,
                            "same_pair": bool(
                                receiver_row["pair_id"] == donor_row["pair_id"]
                            ),
                            "same_session": bool(
                                receiver_row["session_id"] == donor_row["session_id"]
                            ),
                        }
                    )
                if set(mapping.receiver_keys) != set(mapping.donor_keys):
                    raise RuntimeError(
                        f"Text shuffle donor population drifted in {split_name}"
                    )
                if mapping.fixed_point_count or mapping.same_pair_count:
                    raise RuntimeError(
                        f"Text shuffle donor constraints failed in {split_name}"
                    )
            summary_rows.append(
                {
                    "text_ablation_mode": normalized,
                    "text_information_path": text_information_path(normalized),
                    "split": split_name,
                    "tolerance_minutes": int(tolerance),
                    "sample_count": int(len(ordered)),
                    "pair_count": int(ordered["pair_id"].nunique()),
                    "session_count": int(ordered["session_id"].nunique()),
                    "text_shuffle_seed": int(seed),
                    "text_shuffle_method": (
                        mapping.method if normalized == TEXT_SHUFFLE else "none"
                    ),
                    "mapping_sha256": mapping_sha256,
                    "universe_sha256": universe_sha256,
                    "fixed_point_count": fixed_points,
                    "same_pair_count": (
                        mapping.same_pair_count if normalized == TEXT_SHUFFLE else ""
                    ),
                    "same_session_count": (
                        mapping.same_session_count if normalized == TEXT_SHUFFLE else ""
                    ),
                    "support_mask_mode": support_mode,
                    "cross_split_donors_allowed": False,
                    "status": "pass",
                }
            )
    summary_path = _write_csv(
        experiment_root / "text_ablation_manifest.csv",
        summary_rows,
        tuple(summary_rows[0]),
    )
    mapping_path: Path | None = None
    if mapping_rows:
        mapping_path = _write_csv_gzip(
            experiment_root / "text_shuffle_mapping.csv.gz",
            mapping_rows,
            tuple(mapping_rows[0]),
        )
    return summary_path, mapping_path


def _build_split_manifest(resolved: Mapping[str, Any], experiment_root: Path) -> Path:
    """Write and hard-validate the fixed train/common-validation/test design."""

    import pandas as pd

    datasets = _require_mapping(resolved["datasets"], "datasets")
    split = _require_mapping(resolved["split"], "split")
    dataset_root = Path(datasets["root"])
    template = str(
        datasets.get("workbook_template", "tolerance_{tolerance02}m/merged_vol.xlsx")
    )
    sheet = str(datasets["sheet_name"])
    train_end = pd.Timestamp(split["train_end_utc"])
    validation_end = pd.Timestamp(split["validation_end_utc"])
    common_tolerance = int(datasets["common_evaluation_tolerance_minutes"])
    support_mode = _support_mask_mode(datasets)
    text_modes = _configured_text_ablation_modes(datasets)
    if support_mode == "raw_joint":
        fingerprints = {
            _support_audit_grid_fingerprint(dataset_root, tolerance)
            for tolerance in TOLERANCES
        }
        if len(fingerprints) != 1:
            raise ValueError(
                "All masked-ablation tolerances must share one support grid fingerprint: "
                f"{sorted(fingerprints)}"
            )
    common_path = _workbook_path(dataset_root, common_tolerance, template)
    common_raw = _read_split_keys(common_path, sheet)
    common = _apply_support_eligibility(
        common_raw,
        dataset_root=dataset_root,
        tolerance=common_tolerance,
        support_mode=support_mode,
    )
    validation = common[
        (common["effective_origin_utc"] >= train_end)
        & (common["effective_origin_utc"] < validation_end)
    ]
    test = common[common["effective_origin_utc"] >= validation_end]
    validation_counts = _split_counts(validation)
    test_counts = _split_counts(test)
    expected_validation = (
        EXPECTED_RAW_JOINT_COMMON_VALIDATION
        if support_mode == "raw_joint"
        else EXPECTED_COMMON_VALIDATION
    )
    expected_test = (
        EXPECTED_RAW_JOINT_COMMON_TEST
        if support_mode == "raw_joint"
        else EXPECTED_COMMON_TEST
    )
    if validation_counts != expected_validation:
        raise ValueError(
            f"Common validation count drift: {validation_counts} != {expected_validation}"
        )
    if test_counts != expected_test:
        raise ValueError(f"Common test count drift: {test_counts} != {expected_test}")

    rows: list[dict[str, Any]] = []
    validation_pairs = set(validation["pair_id"])
    validation_sessions = set(validation["session_id"])
    test_pairs = set(test["pair_id"])
    test_sessions = set(test["session_id"])
    if validation_pairs & test_pairs or validation_sessions & test_sessions:
        raise ValueError("Common validation/test pair or session leakage detected")
    split_frames: dict[str, tuple[int, Any]] = {
        "common_validation_05m": (common_tolerance, validation.copy()),
        "common_test_core_05m": (common_tolerance, test.copy()),
    }
    broad_test = None
    for tolerance in TOLERANCES:
        training_path = _workbook_path(dataset_root, tolerance, template)
        source_raw = (
            common_raw
            if tolerance == common_tolerance
            else _read_split_keys(training_path, sheet)
        )
        source = (
            common
            if tolerance == common_tolerance
            else _apply_support_eligibility(
                source_raw,
                dataset_root=dataset_root,
                tolerance=tolerance,
                support_mode=support_mode,
            )
        )
        train = source[source["effective_origin_utc"] < train_end]
        train_counts = _split_counts(train)
        expected = (
            EXPECTED_RAW_JOINT_SPLIT_COUNTS[tolerance]
            if support_mode == "raw_joint"
            else EXPECTED_SPLIT_COUNTS[tolerance]
        )
        observed = {f"train_{name}": value for name, value in train_counts.items()}
        if observed != expected:
            raise ValueError(
                f"{tolerance}m training count drift: {observed} != {expected}"
            )
        train_pairs = set(train["pair_id"])
        train_sessions = set(train["session_id"])
        overlaps = {
            "train_validation_pair_overlap": len(train_pairs & validation_pairs),
            "train_test_pair_overlap": len(train_pairs & test_pairs),
            "validation_test_pair_overlap": len(validation_pairs & test_pairs),
            "train_validation_session_overlap": len(
                train_sessions & validation_sessions
            ),
            "train_test_session_overlap": len(train_sessions & test_sessions),
            "validation_test_session_overlap": len(validation_sessions & test_sessions),
        }
        if any(overlaps.values()):
            raise ValueError(f"{tolerance}m cross-split leakage: {overlaps}")
        split_frames[f"train_{tolerance:02d}m"] = (tolerance, train.copy())
        if tolerance == 30:
            broad_test = source[source["effective_origin_utc"] >= validation_end].copy()
            if support_mode == "raw_joint":
                broad_counts = _split_counts(broad_test)
                if broad_counts != EXPECTED_RAW_JOINT_BROAD_TEST:
                    raise ValueError(
                        "30m broad-test count drift: "
                        f"{broad_counts} != {EXPECTED_RAW_JOINT_BROAD_TEST}"
                    )
            split_frames["broad_test_30m"] = (tolerance, broad_test)
        rows.append(
            {
                "tolerance_minutes": tolerance,
                "training_workbook": str(training_path),
                "common_evaluation_workbook": str(common_path),
                "train_end_utc": str(split["train_end_utc"]),
                "validation_end_utc": str(split["validation_end_utc"]),
                "support_mask_mode": support_mode,
                "source_rows_before_support_filter": int(len(source_raw)),
                "source_rows_after_support_filter": int(len(source)),
                "excluded_zero_joint_support_rows": int(len(source_raw) - len(source)),
                **expected,
                "validation_rows": validation_counts["rows"],
                "validation_pairs": validation_counts["pairs"],
                "validation_sessions": validation_counts["sessions"],
                "test_rows": test_counts["rows"],
                "test_pairs": test_counts["pairs"],
                "test_sessions": test_counts["sessions"],
                "broad_test_rows": (
                    int(len(broad_test))
                    if tolerance == 30 and broad_test is not None
                    else ""
                ),
                "broad_test_pairs": (
                    int(broad_test["pair_id"].nunique())
                    if tolerance == 30 and broad_test is not None
                    else ""
                ),
                "broad_test_sessions": (
                    int(broad_test["session_id"].nunique())
                    if tolerance == 30 and broad_test is not None
                    else ""
                ),
                **overlaps,
                "status": "pass",
            }
        )
    fields = tuple(rows[0])
    manifest_path = _write_csv(experiment_root / "split_manifest.csv", rows, fields)
    _write_text_ablation_audit(
        experiment_root,
        split_frames=split_frames,
        modes=text_modes,
        seed=int(datasets.get("text_shuffle_seed", datasets["seed"])),
        support_mode=support_mode,
    )
    return manifest_path


def _slot_assignments(resolved: Mapping[str, Any]) -> dict[int, tuple[int, int, int]]:
    runtime = _require_mapping(resolved["runtime"], "runtime")
    gpu_ids = [int(value) for value in runtime["gpu_ids"]]
    slots_per_gpu = int(runtime["slots_per_gpu"])
    slots = [(gpu, slot) for gpu in gpu_ids for slot in range(slots_per_gpu)]
    order = [
        int(value) for value in runtime.get("tolerance_slot_order", (5, 30, 10, 15))
    ]
    if sorted(order) != list(TOLERANCES) or len(slots) != len(TOLERANCES):
        raise ValueError(
            "tolerance_slot_order and GPU slots must cover four tolerance jobs exactly once"
        )
    numa = {
        int(key): int(value)
        for key, value in dict(runtime.get("gpu_numa_nodes") or {}).items()
    }
    missing = sorted(set(gpu_ids) - set(numa))
    if missing:
        raise ValueError(f"Missing NUMA mapping for GPUs: {missing}")
    return {
        tolerance: (gpu, slot, numa[gpu])
        for tolerance, (gpu, slot) in zip(order, slots)
    }


def _capacity_worker_slots(
    resolved: Mapping[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    runtime = _require_mapping(resolved["runtime"], "runtime")
    gpu_ids = [int(value) for value in runtime["gpu_ids"]]
    slots_per_gpu = int(runtime["slots_per_gpu"])
    numa = {
        int(key): int(value)
        for key, value in dict(runtime.get("gpu_numa_nodes") or {}).items()
    }
    missing = sorted(set(gpu_ids) - set(numa))
    if missing:
        raise ValueError(f"Missing NUMA mapping for GPUs: {missing}")
    slots = tuple(
        (gpu_id, gpu_slot, numa[gpu_id])
        for gpu_id in gpu_ids
        for gpu_slot in range(slots_per_gpu)
    )
    if len(slots) != 4:
        raise ValueError("The capacity sweep requires exactly four GPU worker slots")
    return slots


def _training_payload(
    resolved: Mapping[str, Any],
    *,
    family: str,
    tolerance: int,
    text_ablation_mode: str,
    multi_mode: bool,
    experiment_root: Path,
    capacity_profile: str | None = None,
) -> dict[str, Any]:
    datasets = _require_mapping(resolved["datasets"], "datasets")
    split = _require_mapping(resolved["split"], "split")
    model = _require_mapping(
        _require_mapping(resolved["models"], "models")[family], f"models.{family}"
    )
    payload = deepcopy(_require_mapping(model["training"], f"models.{family}.training"))
    dataset_root = Path(datasets["root"])
    template = str(
        datasets.get("workbook_template", "tolerance_{tolerance02}m/merged_vol.xlsx")
    )
    common_tolerance = int(datasets["common_evaluation_tolerance_minutes"])
    mode = normalize_text_ablation_mode(text_ablation_mode)
    run_root = experiment_root / "runs" / family
    profile_sha256 = ""
    if capacity_profile is not None:
        profile_name = str(capacity_profile).strip().lower()
        profiles = _capacity_profiles(resolved)
        if profile_name not in profiles:
            raise ValueError(f"Unknown capacity profile: {profile_name}")
        profile = profiles[profile_name]
        payload.update(
            {field: int(profile[field]) for field in CAPACITY_PROFILE_SHAPE_FIELDS}
        )
        profile_sha256 = _capacity_profile_sha256(profile_name, profile)
        payload["news_first_capacity_profile"] = profile_name
        payload["news_first_capacity_profile_sha256"] = profile_sha256
        run_root = run_root / profile_name / mode
    elif multi_mode:
        run_root = run_root / mode
    payload.update(
        {
            "data_path": str(_workbook_path(dataset_root, tolerance, template)),
            "sheet_name": str(datasets["sheet_name"]),
            "text_embedding_mode": str(datasets["text_embedding_mode"]),
            "seed": int(datasets["seed"]),
            "support_mask_mode": _support_mask_mode(datasets),
            "news_first_dataset_tolerance_minutes": int(tolerance),
            "news_first_text_ablation_mode": mode,
            "news_first_text_information_path": text_information_path(mode),
            "news_first_text_shuffle_seed": int(
                datasets.get("text_shuffle_seed", datasets["seed"])
            ),
            "cuda": True,
            "num_workers": 0,
            "news_first_common_eval_data_path": str(
                _workbook_path(dataset_root, common_tolerance, template)
            ),
            "news_first_train_end_utc": str(split["train_end_utc"]),
            "news_first_validation_end_utc": str(split["validation_end_utc"]),
            "validation_mc_samples": int(split.get("validation_mc_samples", 1)),
            "output_root": str(run_root / f"tolerance_{tolerance:02d}m"),
        }
    )
    return payload


def _job_status_path(experiment_root: Path, job_id: str) -> Path:
    return experiment_root / "registry" / "jobs" / f"{job_id}.status.json"


def _load_registry(experiment_root: str | Path) -> dict[str, Any]:
    return _read_json(Path(experiment_root) / "registry" / "jobs.json")


def _load_job(experiment_root: Path, job_id: str) -> dict[str, Any]:
    registry = _load_registry(experiment_root)
    for job in registry["jobs"]:
        if job["job_id"] == job_id:
            return dict(job)
    raise KeyError(f"Unknown job_id: {job_id}")


def _maximum_registered_attempt(experiment_root: Path) -> int:
    registry_path = experiment_root / "registry" / "jobs.json"
    if not registry_path.is_file():
        return 0
    attempts = []
    for job in _read_json(registry_path).get("jobs", []):
        path = _job_status_path(experiment_root, str(job["job_id"]))
        if path.is_file():
            attempts.append(int(_read_json(path).get("attempt", 0)))
    return max(attempts, default=0)


def _csv_hash_map(path: Path, key: str = "relative_path") -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row[key]): str(row["sha256"]) for row in csv.DictReader(handle)}


def refresh_experiment_lineage(
    config_path: str | Path,
    experiment_root: str | Path,
) -> Path:
    """Backfill/refresh code and git lineage without touching any job state.

    Before the first worker attempt, ``code_hashes.csv`` may be safely updated.
    Once an attempt exists, its prepared snapshot is preserved; if the working
    tree changed, a separate ``code_hashes_current.csv`` records the difference.
    """

    root = Path(experiment_root).resolve(strict=False)
    expected_hash_path = root / "registry" / "resolved_config.sha256"
    if not expected_hash_path.is_file():
        raise ValueError(f"Not a prepared experiment root: {root}")
    resolved = _resolved_config(config_path)
    resolved_hash = _payload_sha256(resolved)
    if expected_hash_path.read_text(encoding="utf-8").strip() != resolved_hash:
        raise ValueError(
            "Prepared experiment config hash differs from the requested config"
        )

    attempts = _maximum_registered_attempt(root)
    code_rows = _code_rows()
    code_path = root / "code_hashes.csv"
    current_map = {str(row["relative_path"]): str(row["sha256"]) for row in code_rows}
    prepared_map = _csv_hash_map(code_path)
    code_changed_since_snapshot = bool(prepared_map and prepared_map != current_map)
    current_code_path: Path | None = None
    if not prepared_map or attempts == 0:
        _write_csv(
            code_path,
            code_rows,
            ("relative_path", "path", "size_bytes", "sha256"),
        )
        prepared_map = current_map
        code_changed_since_snapshot = False
    elif code_changed_since_snapshot:
        current_code_path = root / "code_hashes_current.csv"
        _write_csv(
            current_code_path,
            code_rows,
            ("relative_path", "path", "size_bytes", "sha256"),
        )

    source_path = root / "source_hashes.csv"
    if not source_path.is_file():
        source_rows = _source_rows(resolved)
        _write_csv(
            source_path,
            source_rows,
            ("source_role", "path", "size_bytes", "sha256"),
        )

    manifest_path = root / "run_manifest.json"
    previous = _read_json(manifest_path) if manifest_path.is_file() else {}
    current_git = _git_snapshot()
    initial_git = previous.get("initial_git_snapshot") or current_git
    first_capture_attempt = int(previous.get("first_capture_max_job_attempt", attempts))
    manifest = {
        "schema_version": 1,
        "experiment_root": str(root),
        "resolved_config_sha256": resolved_hash,
        "initial_git_snapshot": initial_git,
        "current_git_snapshot": current_git,
        "first_capture_max_job_attempt": first_capture_attempt,
        "lineage_capture_timing": (
            "before_first_worker_attempt"
            if first_capture_attempt == 0
            else "backfilled_after_worker_attempts_started"
        ),
        "lineage_limitation": (
            ""
            if first_capture_attempt == 0
            else "Code/git lineage was first captured after at least one worker attempt; "
            "the hashes must not be represented as independently proving the exact code "
            "loaded by earlier attempts."
        ),
        "prepared_code_hashes_path": str(code_path),
        "prepared_code_hashes_sha256": _sha256_file(code_path),
        "current_code_hashes_path": str(current_code_path or code_path),
        "current_code_hashes_sha256": _sha256_file(current_code_path or code_path),
        "code_changed_since_prepared_snapshot": code_changed_since_snapshot,
        "source_hashes_path": str(source_path),
        "source_hashes_sha256": _sha256_file(source_path),
        "config_hashes_path": str(root / "config_hashes.csv"),
        "resource_telemetry": {
            "telemetry_scope": "assigned_gpu_wave_aggregate",
            "concurrent_slots_on_gpu": 2,
            "interpretation_note": (
                "GPU memory and utilization are whole-device samples shared by the two "
                "concurrent jobs assigned to that GPU in the wave; they are not per-process "
                "or task-exclusive measurements. gpu_hours is derived independently from "
                "each job's wall-clock runtime."
            ),
        },
        "updated_at_utc": _utc_now(),
    }
    _write_json(manifest_path, manifest)
    return manifest_path


TASK_FIELDS = (
    "job_id",
    "wave",
    "model_family",
    "text_ablation_mode",
    "text_information_path",
    "support_mask_mode",
    "tolerance_minutes",
    "gpu_id",
    "gpu_slot",
    "capacity_profile",
    "capacity_profile_sha256",
    "capacity_stage",
    "selection_sha256",
    "selection_chain_sha256",
    "job_spec_sha256",
    "expected_regression_parameters",
    "expected_wgan_parameters",
    "numa_node",
    "status",
    "attempt",
    "pid",
    "config_sha256",
    "dataset_sha256",
    "run_dir",
    "log_path",
    "exit_code",
    "started_at_utc",
    "completed_at_utc",
    "error",
)


def _refresh_registry_exports(experiment_root: Path) -> Path:
    registry = _load_registry(experiment_root)
    rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for job in registry["jobs"]:
        status_path = _job_status_path(experiment_root, job["job_id"])
        status = (
            _read_json(status_path) if status_path.exists() else {"status": "missing"}
        )
        rows.append({**job, **status})
        for artifact in status.get("artifacts", []):
            output_rows.append({"job_id": job["job_id"], **artifact})
    _write_csv(experiment_root / "task_registry.csv", rows, TASK_FIELDS)
    stable_outputs = [
        experiment_root / "task_registry.csv",
        experiment_root / "split_manifest.csv",
        experiment_root / "resource_usage.csv",
        experiment_root / "resource_summary.csv",
        experiment_root / "config_hashes.csv",
        experiment_root / "source_hashes.csv",
        experiment_root / "code_hashes.csv",
        experiment_root / "code_hashes_current.csv",
        experiment_root / "run_manifest.json",
        experiment_root / "text_ablation_manifest.csv",
        experiment_root / "text_shuffle_mapping.csv.gz",
        experiment_root / "capacity_profile_manifest.csv",
        experiment_root / "capacity_stage_status.json",
        experiment_root / "capacity_selection.json",
        experiment_root / "capacity_selection_audit.csv",
        experiment_root / "capacity_comparisons.csv",
        experiment_root / "registry" / "jobs.json",
        experiment_root / "registry" / "experiment_status.json",
        experiment_root / "registry" / "resolved_config.sha256",
    ]
    job_status_dir = experiment_root / "registry" / "jobs"
    if job_status_dir.is_dir():
        stable_outputs.extend(
            path for path in job_status_dir.glob("*.status.json") if path.is_file()
        )
    selection_dir = experiment_root / "registry" / "selections"
    if selection_dir.is_dir():
        stable_outputs.extend(
            path for path in selection_dir.glob("*.json") if path.is_file()
        )
    for directory_name in ("analysis", "report"):
        directory = experiment_root / directory_name
        if directory.is_dir():
            stable_outputs.extend(
                path for path in directory.rglob("*") if path.is_file()
            )
    seen_paths = {str(row.get("path", "")) for row in output_rows}
    for path in stable_outputs:
        if not path.is_file() or str(path) in seen_paths:
            continue
        output_rows.append(
            {
                "job_id": "",
                "artifact_role": f"experiment:{path.relative_to(experiment_root).as_posix()}",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_csv(
        experiment_root / "output_hashes.csv",
        output_rows,
        ("job_id", "artifact_role", "path", "size_bytes", "sha256"),
    )
    return experiment_root / "task_registry.csv"


def prepare_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    """Validate inputs and materialize the immutable model/tolerance/mode registry."""

    experiment_root = _resolve_repo_path(output_dir)
    resolved = _resolved_config(config_path)
    resolved_hash = _payload_sha256(resolved)
    existing_hash_path = experiment_root / "registry" / "resolved_config.sha256"
    if experiment_root.exists() and any(experiment_root.iterdir()):
        if not reuse:
            raise FileExistsError(
                f"Experiment root already exists; use --resume/--reuse: {experiment_root}"
            )
        if not existing_hash_path.is_file():
            raise ValueError(
                f"Existing directory is not a prepared experiment: {experiment_root}"
            )
        existing_hash = existing_hash_path.read_text(encoding="utf-8").strip()
        if existing_hash != resolved_hash:
            raise ValueError(
                "Prepared experiment config hash differs from the requested config"
            )
        if not (experiment_root / "split_manifest.csv").is_file():
            _build_split_manifest(resolved, experiment_root)
        refresh_experiment_lineage(config_path, experiment_root)
        _refresh_registry_exports(experiment_root)
        return experiment_root

    dataset_root = Path(resolved["datasets"]["root"])
    _validate_dataset_summary(dataset_root)
    source_rows = _source_rows(resolved)
    source_hash_by_path = {row["path"]: row["sha256"] for row in source_rows}
    assignments = _slot_assignments(resolved)
    text_modes = _configured_text_ablation_modes(resolved["datasets"])
    multi_mode = len(text_modes) > 1
    experiment_root.mkdir(parents=True, exist_ok=True)
    for relative in ("registry/jobs", "configs/jobs", "logs", "resources", "runs"):
        (experiment_root / relative).mkdir(parents=True, exist_ok=True)
    _build_split_manifest(resolved, experiment_root)

    jobs: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for wave_index, wave in enumerate(resolved["waves"], start=1):
        family = str(wave["model"])
        mode = normalize_text_ablation_mode(
            wave.get("text_ablation_mode", text_modes[0])
        )
        for tolerance in TOLERANCES:
            gpu_id, gpu_slot, numa_node = assignments[tolerance]
            job_id = (
                f"{family}_{mode}_{tolerance:02d}m"
                if multi_mode
                else f"{family}_{tolerance:02d}m"
            )
            training = _training_payload(
                resolved,
                family=family,
                tolerance=tolerance,
                text_ablation_mode=mode,
                multi_mode=multi_mode,
                experiment_root=experiment_root,
            )
            training_path = experiment_root / "configs" / "jobs" / f"{job_id}.yaml"
            _write_yaml(training_path, training)
            training_hash = _sha256_file(training_path)
            job = {
                "job_id": job_id,
                "wave": wave_index,
                "model_family": family,
                "text_ablation_mode": mode,
                "text_information_path": text_information_path(mode),
                "support_mask_mode": _support_mask_mode(resolved["datasets"]),
                "trainer_command": str(resolved["models"][family]["trainer_command"]),
                "tolerance_minutes": tolerance,
                "gpu_id": gpu_id,
                "gpu_slot": gpu_slot,
                "numa_node": numa_node,
                "training_config_path": str(training_path),
                "config_sha256": training_hash,
                "dataset_path": training["data_path"],
                "dataset_sha256": source_hash_by_path[training["data_path"]],
                "output_root": training["output_root"],
            }
            job["job_spec_sha256"] = _payload_sha256(job)
            jobs.append(job)
            config_rows.append(
                {
                    "config_role": job_id,
                    "path": str(training_path),
                    "sha256": training_hash,
                }
            )
            _write_json(
                _job_status_path(experiment_root, job_id),
                {
                    "job_id": job_id,
                    "status": "prepared",
                    "attempt": 0,
                    "config_sha256": training_hash,
                    "updated_at_utc": _utc_now(),
                },
            )

    registry = {
        "schema_version": 2 if multi_mode else 1,
        "experiment_root": str(experiment_root),
        "resolved_config_sha256": resolved_hash,
        "created_at_utc": _utc_now(),
        "jobs": jobs,
    }
    _write_yaml(experiment_root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(existing_hash_path, resolved_hash + "\n")
    _write_json(experiment_root / "registry" / "jobs.json", registry)
    _write_json(
        experiment_root / "registry" / "experiment_status.json",
        {"status": "prepared", "current_wave": 0, "updated_at_utc": _utc_now()},
    )
    _write_csv(
        experiment_root / "source_hashes.csv",
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
    )
    resolved_path = experiment_root / "resolved_config.yaml"
    config_rows.insert(
        0,
        {
            "config_role": "resolved_orchestration",
            "path": str(resolved_path),
            "sha256": _sha256_file(resolved_path),
        },
    )
    _write_csv(
        experiment_root / "config_hashes.csv",
        config_rows,
        ("config_role", "path", "sha256"),
    )
    refresh_experiment_lineage(config_path, experiment_root)
    _refresh_registry_exports(experiment_root)
    return experiment_root


def _capacity_job_id(
    family: str,
    profile: str,
    mode: str,
    tolerance: int,
) -> str:
    return (
        f"capacity_{str(family).strip().lower()}_"
        f"{str(profile).strip().lower()}_{normalize_text_ablation_mode(mode)}_"
        f"{int(tolerance):02d}m"
    )


def _capacity_stage_specs(
    stage: str,
    profiles: Sequence[str],
) -> list[dict[str, Any]]:
    normalized_stage = str(stage).strip().lower()
    if normalized_stage not in CAPACITY_STAGES:
        raise ValueError(f"Unknown capacity stage: {stage}")
    normalized_profiles = tuple(str(value).strip().lower() for value in profiles)
    if len(set(normalized_profiles)) != len(normalized_profiles):
        raise ValueError(f"Capacity stage profiles must be unique: {profiles}")
    invalid = sorted(set(normalized_profiles) - set(CAPACITY_PROFILE_NAMES))
    if invalid:
        raise ValueError(f"Unknown capacity profiles: {invalid}")

    if normalized_stage == "regression_screen":
        families = ("regression",)
        tolerances = CAPACITY_SCREEN_TOLERANCES
        modes = CAPACITY_SCREEN_TEXT_MODES
    elif normalized_stage == "regression_confirm":
        families = ("regression",)
        tolerances = TOLERANCES
        modes = TEXT_ABLATION_MODES
    elif normalized_stage == "wgan_screen":
        families = ("wgan",)
        tolerances = CAPACITY_SCREEN_TOLERANCES
        modes = ("current_only",)
    else:
        families = ("wgan",)
        tolerances = TOLERANCES
        modes = TEXT_ABLATION_MODES
    return [
        {
            "model_family": family,
            "capacity_profile": profile,
            "text_ablation_mode": mode,
            "tolerance_minutes": int(tolerance),
            "capacity_stage": normalized_stage,
        }
        for profile in normalized_profiles
        for family in families
        for mode in modes
        for tolerance in tolerances
    ]


def _write_capacity_profile_manifest(
    experiment_root: Path,
    profiles: Mapping[str, Mapping[str, Any]],
) -> Path:
    rows: list[dict[str, Any]] = []
    for name in CAPACITY_PROFILE_NAMES:
        profile = profiles[name]
        row: dict[str, Any] = {
            "capacity_profile": name,
            "capacity_profile_sha256": _capacity_profile_sha256(name, profile),
            **{field: int(profile[field]) for field in CAPACITY_PROFILE_SHAPE_FIELDS},
            **{
                field: int(profile[field]) for field in CAPACITY_PROFILE_METADATA_FIELDS
            },
        }
        for tolerance in TOLERANCES:
            train_pairs = EXPECTED_RAW_JOINT_SPLIT_COUNTS[tolerance]["train_pairs"]
            row[f"regression_parameters_per_train_pair_{tolerance:02d}m"] = float(
                profile["expected_regression_parameters"]
            ) / float(train_pairs)
            row[f"wgan_parameters_per_train_pair_{tolerance:02d}m"] = float(
                profile["expected_wgan_parameters"]
            ) / float(train_pairs)
        rows.append(row)
    return _write_csv(
        experiment_root / "capacity_profile_manifest.csv",
        rows,
        tuple(rows[0]),
    )


def _capacity_selection_snapshot_path(experiment_root: Path, stage: str) -> Path:
    return experiment_root / "registry" / "selections" / f"{stage}.json"


def _capacity_selection_hashes(experiment_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for stage in CAPACITY_STAGES:
        path = _capacity_selection_snapshot_path(experiment_root, stage)
        if path.is_file():
            hashes[stage] = _sha256_file(path)
    return hashes


def _refresh_capacity_config_hashes(experiment_root: Path) -> Path:
    registry = _load_registry(experiment_root)
    resolved_path = experiment_root / "resolved_config.yaml"
    rows = [
        {
            "config_role": "resolved_orchestration",
            "path": str(resolved_path),
            "sha256": _sha256_file(resolved_path),
        }
    ]
    rows.extend(
        {
            "config_role": str(job["job_id"]),
            "path": str(job["training_config_path"]),
            "sha256": str(job["config_sha256"]),
        }
        for job in registry["jobs"]
    )
    return _write_csv(
        experiment_root / "config_hashes.csv",
        rows,
        ("config_role", "path", "sha256"),
    )


def _append_capacity_stage_jobs(
    experiment_root: Path,
    resolved: Mapping[str, Any],
    *,
    stage: str,
    profiles: Sequence[str],
    selection_sha256: str = "",
) -> tuple[list[str], list[str]]:
    """Append a capacity stage without ever rewriting an existing job spec."""

    specs = _capacity_stage_specs(stage, profiles)
    registry = _load_registry(experiment_root)
    existing = {str(job["job_id"]): dict(job) for job in registry["jobs"]}
    profile_values = _capacity_profiles(resolved)
    source_hashes: dict[str, str] = {}
    with (experiment_root / "source_hashes.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            source_hashes[str(row["path"])] = str(row["sha256"])
    slots = _capacity_worker_slots(resolved)
    next_wave = max((int(job["wave"]) for job in existing.values()), default=0) + 1
    required_ids: list[str] = []
    missing_specs: list[dict[str, Any]] = []
    for spec in specs:
        job_id = _capacity_job_id(
            spec["model_family"],
            spec["capacity_profile"],
            spec["text_ablation_mode"],
            spec["tolerance_minutes"],
        )
        required_ids.append(job_id)
        if job_id in existing:
            prior = existing[job_id]
            immutable = {
                "model_family": spec["model_family"],
                "capacity_profile": spec["capacity_profile"],
                "text_ablation_mode": spec["text_ablation_mode"],
                "tolerance_minutes": spec["tolerance_minutes"],
            }
            mismatches = {
                key: (prior.get(key), value)
                for key, value in immutable.items()
                if prior.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"Existing capacity job conflicts with requested stage: "
                    f"{job_id}: {mismatches}"
                )
            continue
        missing_specs.append(spec)

    selection_chain = _capacity_selection_hashes(experiment_root)
    selection_chain_sha256 = _payload_sha256(selection_chain) if selection_chain else ""
    new_ids: list[str] = []
    new_jobs: list[dict[str, Any]] = []
    for index, spec in enumerate(missing_specs):
        wave = next_wave + (index // len(slots))
        gpu_id, gpu_slot, numa_node = slots[index % len(slots)]
        family = str(spec["model_family"])
        profile_name = str(spec["capacity_profile"])
        mode = normalize_text_ablation_mode(spec["text_ablation_mode"])
        tolerance = int(spec["tolerance_minutes"])
        job_id = _capacity_job_id(family, profile_name, mode, tolerance)
        profile = profile_values[profile_name]
        profile_sha256 = _capacity_profile_sha256(profile_name, profile)
        training = _training_payload(
            resolved,
            family=family,
            tolerance=tolerance,
            text_ablation_mode=mode,
            multi_mode=True,
            experiment_root=experiment_root,
            capacity_profile=profile_name,
        )
        training_path = experiment_root / "configs" / "jobs" / f"{job_id}.yaml"
        _write_yaml(training_path, training)
        training_hash = _sha256_file(training_path)
        dataset_path = str(training["data_path"])
        if dataset_path not in source_hashes:
            raise ValueError(f"Dataset source hash is unavailable for {dataset_path}")
        job = {
            "job_id": job_id,
            "wave": wave,
            "model_family": family,
            "text_ablation_mode": mode,
            "text_information_path": text_information_path(mode),
            "support_mask_mode": _support_mask_mode(resolved["datasets"]),
            "trainer_command": str(resolved["models"][family]["trainer_command"]),
            "tolerance_minutes": tolerance,
            "gpu_id": gpu_id,
            "gpu_slot": gpu_slot,
            "numa_node": numa_node,
            "training_config_path": str(training_path),
            "config_sha256": training_hash,
            "dataset_path": dataset_path,
            "dataset_sha256": source_hashes[dataset_path],
            "output_root": training["output_root"],
            "capacity_profile": profile_name,
            "capacity_profile_sha256": profile_sha256,
            "capacity_stage": str(stage),
            "selection_sha256": str(selection_sha256),
            "selection_chain_sha256": selection_chain_sha256,
            "expected_regression_parameters": int(
                profile["expected_regression_parameters"]
            ),
            "expected_wgan_parameters": int(profile["expected_wgan_parameters"]),
        }
        job["job_spec_sha256"] = _payload_sha256(job)
        new_jobs.append(job)
        new_ids.append(job_id)
        _write_json(
            _job_status_path(experiment_root, job_id),
            {
                "job_id": job_id,
                "status": "prepared",
                "attempt": 0,
                "config_sha256": training_hash,
                "capacity_profile": profile_name,
                "capacity_profile_sha256": profile_sha256,
                "capacity_stage": str(stage),
                "updated_at_utc": _utc_now(),
            },
        )
    if new_jobs:
        registry["jobs"].extend(new_jobs)
        registry["updated_at_utc"] = _utc_now()
        _write_json(experiment_root / "registry" / "jobs.json", registry)

    status_path = experiment_root / "capacity_stage_status.json"
    capacity_status = (
        _read_json(status_path)
        if status_path.is_file()
        else {
            "schema_version": 1,
            "status": "prepared",
            "current_stage": str(stage),
            "stages": {},
        }
    )
    stages = dict(capacity_status.get("stages") or {})
    previous_stage = dict(stages.get(stage) or {})
    stages[stage] = {
        **previous_stage,
        "status": previous_stage.get("status", "prepared"),
        "profiles": list(dict.fromkeys(str(value) for value in profiles)),
        "required_job_ids": required_ids,
        "new_job_ids": new_ids or list(previous_stage.get("new_job_ids") or []),
        "selection_sha256": str(selection_sha256),
        "updated_at_utc": _utc_now(),
    }
    capacity_status.update(
        {
            "status": capacity_status.get("status", "prepared"),
            "current_stage": str(stage),
            "stages": stages,
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(status_path, capacity_status)
    _refresh_capacity_config_hashes(experiment_root)
    _refresh_registry_exports(experiment_root)
    return required_ids, new_ids


def prepare_capacity_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    reuse: bool = False,
) -> Path:
    """Prepare only the 24-job Regression screen; later stages append jobs."""

    experiment_root = _resolve_repo_path(output_dir)
    resolved = _resolved_config(config_path)
    if _capacity_sweep_config(resolved) is None:
        raise ValueError(
            "The capacity-sweep command requires capacity_sweep.enabled=true"
        )
    resolved_hash = _payload_sha256(resolved)
    existing_hash_path = experiment_root / "registry" / "resolved_config.sha256"
    if experiment_root.exists() and any(experiment_root.iterdir()):
        if not reuse:
            raise FileExistsError(
                f"Experiment root already exists; use --resume/--reuse: {experiment_root}"
            )
        if not existing_hash_path.is_file():
            raise ValueError(
                f"Existing directory is not a prepared experiment: {experiment_root}"
            )
        if existing_hash_path.read_text(encoding="utf-8").strip() != resolved_hash:
            raise ValueError(
                "Prepared experiment config hash differs from the requested config"
            )
        refresh_experiment_lineage(config_path, experiment_root)
        _refresh_registry_exports(experiment_root)
        return experiment_root

    dataset_root = Path(resolved["datasets"]["root"])
    _validate_dataset_summary(dataset_root)
    source_rows = _source_rows(resolved)
    experiment_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "registry/jobs",
        "registry/selections",
        "configs/jobs",
        "logs",
        "resources",
        "runs",
    ):
        (experiment_root / relative).mkdir(parents=True, exist_ok=True)
    _build_split_manifest(resolved, experiment_root)
    _write_yaml(experiment_root / "resolved_config.yaml", {ROOT_KEY: resolved})
    _atomic_write_text(existing_hash_path, resolved_hash + "\n")
    _write_json(
        experiment_root / "registry" / "jobs.json",
        {
            "schema_version": 3,
            "experiment_kind": "capacity_sweep",
            "experiment_root": str(experiment_root),
            "resolved_config_sha256": resolved_hash,
            "created_at_utc": _utc_now(),
            "jobs": [],
        },
    )
    _write_json(
        experiment_root / "registry" / "experiment_status.json",
        {"status": "prepared", "current_wave": 0, "updated_at_utc": _utc_now()},
    )
    _write_csv(
        experiment_root / "source_hashes.csv",
        source_rows,
        ("source_role", "path", "size_bytes", "sha256"),
    )
    _write_capacity_profile_manifest(experiment_root, _capacity_profiles(resolved))
    _write_json(
        experiment_root / "capacity_stage_status.json",
        {
            "schema_version": 1,
            "status": "prepared",
            "current_stage": "regression_screen",
            "stages": {},
            "updated_at_utc": _utc_now(),
        },
    )
    _append_capacity_stage_jobs(
        experiment_root,
        resolved,
        stage="regression_screen",
        profiles=CAPACITY_PROFILE_NAMES,
    )
    refresh_experiment_lineage(config_path, experiment_root)
    _refresh_registry_exports(experiment_root)
    return experiment_root


def _pid_is_live(pid: Any) -> bool:
    try:
        numeric = int(pid)
        if numeric <= 0:
            return False
        os.kill(numeric, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def _artifact_rows(job: Mapping[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = run_dir / "checkpoints"
    metrics = run_dir / "metrics"
    if job["model_family"] == "wgan":
        required = {
            "generator_best": checkpoints / "generator_best.pt",
            "discriminator_best": checkpoints / "discriminator_best.pt",
            "generator_final": checkpoints / "generator.pt",
            "discriminator_final": checkpoints / "discriminator.pt",
        }
        if job.get("capacity_profile"):
            required.update(
                {
                    "generator_initial_epoch0": checkpoints
                    / "generator_initial_epoch0.pt",
                    "discriminator_initial_epoch0": checkpoints
                    / "discriminator_initial_epoch0.pt",
                    "generator_best_learned": checkpoints / "generator_best_learned.pt",
                    "discriminator_best_learned": checkpoints
                    / "discriminator_best_learned.pt",
                }
            )
    else:
        required = {
            "regressor_best": checkpoints / "vol_regressor_best.pt",
            "regressor_final": checkpoints / "vol_regressor.pt",
        }
        if job.get("capacity_profile"):
            required.update(
                {
                    "regressor_initial_epoch0": checkpoints
                    / "vol_regressor_initial_epoch0.pt",
                    "regressor_best_learned": checkpoints
                    / "vol_regressor_best_learned.pt",
                }
            )
    required.update(
        {
            "training_metrics_csv": metrics / "training_metrics.csv",
            "training_metrics_json": metrics / "training_metrics.json",
            "best_checkpoint": metrics / "best_checkpoint.json",
            "resolved_training_config": metrics / "training_resolved_config.yaml",
            "run_log": run_dir / "run.log",
        }
    )
    if job.get("capacity_profile"):
        required["initial_checkpoint"] = metrics / "initial_checkpoint.json"
        required["best_learned_checkpoint"] = metrics / "best_learned_checkpoint.json"
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Training completed without required artifacts: {missing}")
    allowed_initial = {
        "generator_initial_epoch0.pt",
        "discriminator_initial_epoch0.pt",
        "vol_regressor_initial_epoch0.pt",
    }
    epoch_files = sorted(
        path
        for path in checkpoints.glob("*_epoch_*.pt")
        if path.name not in allowed_initial
    )
    if epoch_files:
        raise RuntimeError(
            f"Best/final-only policy violated by epoch checkpoints: {epoch_files}"
        )
    return [
        {
            "artifact_role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for role, path in required.items()
    ]


def _execute_training_job(
    job: Mapping[str, Any], *, dry_run: bool
) -> tuple[Path, list[dict[str, Any]]]:
    # Delayed imports are intentional: CUDA visibility and thread limits must
    # be fixed by the launcher before torch is imported.
    from wgan_option.config import load_config

    config = load_config(str(job["training_config_path"]))
    if job["model_family"] == "wgan":
        from wgan_option.train_vol_xlsx import VolSurfaceXlsxTrainer

        trainer = VolSurfaceXlsxTrainer(
            config, config_path=str(job["training_config_path"])
        )
    elif job["model_family"] == "regression":
        from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer

        trainer = VolSurfaceRegressionTrainer(
            config, config_path=str(job["training_config_path"])
        )
    else:
        raise ValueError(f"Unsupported model family: {job['model_family']}")
    result = trainer.dry_run() if dry_run else trainer.train()
    if result is None:
        raise RuntimeError("Trainer did not return a run directory")
    run_dir = Path(result).resolve(strict=False)
    return run_dir, [] if dry_run else _artifact_rows(job, run_dir)


def _validate_capacity_job_lineage(
    experiment_root: Path,
    job: Mapping[str, Any],
) -> None:
    profile_name = str(job.get("capacity_profile", "")).strip().lower()
    if not profile_name:
        return
    resolved = yaml.safe_load(
        (experiment_root / "resolved_config.yaml").read_text(encoding="utf-8")
    )[ROOT_KEY]
    profiles = _capacity_profiles(resolved)
    if profile_name not in profiles:
        raise ValueError(f"Capacity job references an unknown profile: {profile_name}")
    profile = profiles[profile_name]
    expected_profile_sha256 = _capacity_profile_sha256(profile_name, profile)
    if str(job.get("capacity_profile_sha256", "")) != expected_profile_sha256:
        raise ValueError(f"Capacity profile hash mismatch for {job['job_id']}")
    training = yaml.safe_load(
        Path(str(job["training_config_path"])).read_text(encoding="utf-8")
    )
    if str(training.get("news_first_capacity_profile", "")) != profile_name:
        raise ValueError(f"Training profile metadata mismatch for {job['job_id']}")
    if (
        str(training.get("news_first_capacity_profile_sha256", ""))
        != expected_profile_sha256
    ):
        raise ValueError(f"Training profile hash mismatch for {job['job_id']}")
    for field in CAPACITY_PROFILE_SHAPE_FIELDS:
        if int(training.get(field, -1)) != int(profile[field]):
            raise ValueError(
                f"Training profile shape mismatch for {job['job_id']}: {field}"
            )

    parent_by_stage = {
        "regression_screen": (),
        "regression_confirm": ("regression_screen",),
        "wgan_screen": ("regression_screen", "regression_confirm"),
        "wgan_confirm": (
            "regression_screen",
            "regression_confirm",
            "wgan_screen",
        ),
    }
    stage = str(job.get("capacity_stage", ""))
    if stage not in parent_by_stage:
        raise ValueError(f"Capacity job has an unknown stage: {stage}")
    selection_hashes: dict[str, str] = {}
    for parent_stage in parent_by_stage[stage]:
        selection_path = _capacity_selection_snapshot_path(
            experiment_root, parent_stage
        )
        if not selection_path.is_file():
            raise ValueError(
                f"Capacity job is missing frozen parent selection: {parent_stage}"
            )
        selection_hashes[parent_stage] = _sha256_file(selection_path)
    expected_parent_sha256 = (
        selection_hashes[parent_by_stage[stage][-1]] if parent_by_stage[stage] else ""
    )
    if str(job.get("selection_sha256", "")) != expected_parent_sha256:
        raise ValueError(f"Capacity selection hash mismatch for {job['job_id']}")
    expected_chain_sha256 = (
        _payload_sha256(selection_hashes) if selection_hashes else ""
    )
    if str(job.get("selection_chain_sha256", "")) != expected_chain_sha256:
        raise ValueError(f"Capacity selection-chain hash mismatch for {job['job_id']}")


def run_worker(
    experiment_root: str | Path,
    job_id: str,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    """Run one registry job in the current isolated CUDA process."""

    root = _resolve_repo_path(experiment_root)
    job = _load_job(root, job_id)
    if _sha256_file(job["training_config_path"]) != job["config_sha256"]:
        raise ValueError(f"Training config hash mismatch for {job_id}")
    actual_dataset_sha256 = _sha256_file(job["dataset_path"])
    if actual_dataset_sha256.lower() != str(job["dataset_sha256"]).lower():
        raise ValueError(
            f"Dataset hash mismatch for {job_id}: "
            f"expected={job['dataset_sha256']}; actual={actual_dataset_sha256}"
        )
    _validate_capacity_job_lineage(root, job)
    status_path = _job_status_path(root, job_id)
    previous = _read_json(status_path)
    if previous.get("status") == "completed" and not dry_run:
        if resume and _completed_job_is_valid(root, job, previous):
            return Path(previous["run_dir"])
        if not resume:
            raise RuntimeError(f"Job already completed: {job_id}")
    if previous.get("status") == "running" and _pid_is_live(previous.get("pid")):
        raise RuntimeError(
            f"Job is already running with pid={previous.get('pid')}: {job_id}"
        )

    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    assigned = str(job["gpu_id"])
    if visible and visible.split(",")[0].strip() != assigned:
        raise RuntimeError(
            f"Worker GPU mismatch: assigned={assigned}, CUDA_VISIBLE_DEVICES={visible}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned
    orchestration = yaml.safe_load(
        (root / "resolved_config.yaml").read_text(encoding="utf-8")
    )[ROOT_KEY]
    threads = int(orchestration["runtime"]["cpu_threads_per_job"])
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)

    attempt = int(previous.get("attempt", 0)) + 1
    running = {
        "job_id": job_id,
        "status": "running",
        "attempt": attempt,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "gpu_id": job["gpu_id"],
        "config_sha256": job["config_sha256"],
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "dry_run": bool(dry_run),
        "log_path": str(os.environ.get("NEWS_FIRST_JOB_LOG_PATH", "")),
    }
    _write_json(status_path, running)
    try:
        run_dir, artifacts = _execute_training_job(job, dry_run=dry_run)
        finished = {
            **running,
            "status": "dry_run_passed" if dry_run else "completed",
            "run_dir": str(run_dir),
            "artifacts": artifacts,
            "exit_code": 0,
            "completed_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
        }
        _write_json(status_path, finished)
        return run_dir
    except BaseException as exc:
        _write_json(
            status_path,
            {
                **running,
                "status": "failed",
                "exit_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
            },
        )
        raise


def build_worker_command(
    config_path: str | Path,
    experiment_root: str | Path,
    job: Mapping[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> list[str]:
    resolved = yaml.safe_load(
        (Path(experiment_root) / "resolved_config.yaml").read_text(encoding="utf-8")
    )[ROOT_KEY]
    runtime = resolved["runtime"]
    command: list[str] = []
    if bool(runtime.get("use_numa_binding", True)):
        executable = str(runtime.get("numactl_executable", "numactl"))
        command.extend(
            [
                executable,
                f"--cpunodebind={int(job['numa_node'])}",
                f"--membind={int(job['numa_node'])}",
            ]
        )
    orchestration_command = (
        "train-news-first-vol-capacity-sweep"
        if _capacity_sweep_config(resolved) is not None
        else "train-news-first-vol-comparison"
    )
    command.extend(
        [
            str(runtime["python_executable"]),
            str(REPO_ROOT / "scripts" / "rq3" / "main.py"),
            orchestration_command,
            "worker",
            "--config",
            str(_resolve_repo_path(config_path)),
            "--output-dir",
            str(Path(experiment_root).resolve(strict=False)),
            "--job-id",
            str(job["job_id"]),
        ]
    )
    if dry_run:
        command.append("--worker-dry-run")
    if resume:
        command.append("--resume")
    return command


RESOURCE_FIELDS = (
    "timestamp_utc",
    "wave",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "utilization_gpu_pct",
    "utilization_memory_pct",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
    "sample_status",
    "error",
)


def _query_gpu_resources(executable: str, wave: int) -> list[dict[str, Any]]:
    query = (
        "timestamp,index,uuid,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,temperature.gpu"
    )
    completed = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return [
            {
                "timestamp_utc": _utc_now(),
                "wave": wave,
                "sample_status": "error",
                "error": completed.stderr.strip() or f"exit={completed.returncode}",
            }
        ]
    rows: list[dict[str, Any]] = []
    for fields in csv.reader(completed.stdout.splitlines()):
        if len(fields) != 10:
            continue
        rows.append(
            {
                "timestamp_utc": _utc_now(),
                "wave": wave,
                "gpu_index": fields[1].strip(),
                "gpu_uuid": fields[2].strip(),
                "gpu_name": fields[3].strip(),
                "utilization_gpu_pct": fields[4].strip(),
                "utilization_memory_pct": fields[5].strip(),
                "memory_used_mib": fields[6].strip(),
                "memory_total_mib": fields[7].strip(),
                "power_draw_w": fields[8].strip(),
                "temperature_c": fields[9].strip(),
                "sample_status": "ok",
                "error": "",
            }
        )
    return rows


class _ResourceMonitor:
    def __init__(
        self, output_path: Path, *, executable: str, interval_seconds: float, wave: int
    ):
        self.output_path = output_path
        self.executable = executable
        self.interval_seconds = max(0.2, float(interval_seconds))
        self.wave = int(wave)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"gpu-monitor-wave-{wave}", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds + 2.0))

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.output_path.exists() or self.output_path.stat().st_size == 0
        )
        with self.output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(RESOURCE_FIELDS), extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            while True:
                writer.writerows(_query_gpu_resources(self.executable, self.wave))
                handle.flush()
                if self.stop_event.wait(self.interval_seconds):
                    break


RESOURCE_SUMMARY_FIELDS = (
    "job_id",
    "model",
    "text_ablation_mode",
    "support_mask_mode",
    "capacity_profile",
    "capacity_profile_sha256",
    "capacity_stage",
    "tolerance_minutes",
    "wave",
    "gpu_id",
    "gpu_slot",
    "telemetry_scope",
    "concurrent_slots_on_gpu",
    "runtime_minutes",
    "gpu_hours",
    "peak_memory_mib",
    "mean_utilization_gpu_pct",
    "peak_utilization_gpu_pct",
    "sample_count",
    "dry_run",
    "status",
)


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _write_resource_summary(experiment_root: Path) -> Path:
    raw_path = experiment_root / "resource_usage.csv"
    samples: list[dict[str, Any]] = []
    if raw_path.is_file():
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            samples = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for job in _load_registry(experiment_root)["jobs"]:
        status = _read_json(_job_status_path(experiment_root, job["job_id"]))
        if status.get("status") not in TERMINAL_SUCCESS:
            continue
        started = _parse_utc(status.get("started_at_utc"))
        completed = _parse_utc(status.get("completed_at_utc"))
        runtime_minutes = (
            max(0.0, (completed - started).total_seconds() / 60.0)
            if started is not None and completed is not None
            else 0.0
        )
        selected = []
        for sample in samples:
            sample_time = _parse_utc(sample.get("timestamp_utc"))
            if str(sample.get("sample_status")) != "ok":
                continue
            if int(sample.get("wave", -1)) != int(job["wave"]):
                continue
            if int(sample.get("gpu_index", -1)) != int(job["gpu_id"]):
                continue
            if (
                started is not None
                and sample_time is not None
                and sample_time < started
            ):
                continue
            if (
                completed is not None
                and sample_time is not None
                and sample_time > completed
            ):
                continue
            selected.append(sample)
        memory = [float(sample["memory_used_mib"]) for sample in selected]
        utilization = [float(sample["utilization_gpu_pct"]) for sample in selected]
        rows.append(
            {
                "job_id": job["job_id"],
                "model": job["model_family"],
                "text_ablation_mode": job.get("text_ablation_mode", REAL_TEXT),
                "support_mask_mode": job.get("support_mask_mode", "none"),
                "capacity_profile": job.get("capacity_profile", "default"),
                "capacity_profile_sha256": job.get("capacity_profile_sha256", ""),
                "capacity_stage": job.get("capacity_stage", ""),
                "tolerance_minutes": job["tolerance_minutes"],
                "wave": job["wave"],
                "gpu_id": job["gpu_id"],
                "gpu_slot": job["gpu_slot"],
                "telemetry_scope": "assigned_gpu_wave_aggregate",
                "concurrent_slots_on_gpu": 2,
                "runtime_minutes": runtime_minutes,
                "gpu_hours": runtime_minutes / 60.0,
                "peak_memory_mib": max(memory) if memory else "",
                "mean_utilization_gpu_pct": sum(utilization) / len(utilization)
                if utilization
                else "",
                "peak_utilization_gpu_pct": max(utilization) if utilization else "",
                "sample_count": len(selected),
                "dry_run": bool(status.get("dry_run", False)),
                "status": status.get("status", ""),
            }
        )
    return _write_csv(
        experiment_root / "resource_summary.csv",
        rows,
        RESOURCE_SUMMARY_FIELDS,
    )


def _terminate_processes(processes: Iterable[subprocess.Popen[Any]]) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in active:
        process.terminate()
    deadline = time.monotonic() + 10.0
    for process in active:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def _completed_job_is_valid(
    root: Path, job: Mapping[str, Any], status: Mapping[str, Any]
) -> bool:
    if (
        status.get("status") != "completed"
        or status.get("config_sha256") != job["config_sha256"]
    ):
        return False
    artifacts = list(status.get("artifacts") or [])
    if not artifacts:
        return False
    for artifact in artifacts:
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or _sha256_file(path) != artifact.get("sha256"):
            return False
    return True


def _jobs_for_wave(
    root: Path, wave: int, *, resume: bool, dry_run: bool
) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in _load_registry(root)["jobs"]
        if int(job["wave"]) == int(wave)
    ]
    selected: list[dict[str, Any]] = []
    for job in jobs:
        status = _read_json(_job_status_path(root, job["job_id"]))
        if not dry_run and status.get("status") == "completed":
            if _completed_job_is_valid(root, job, status):
                if resume:
                    continue
                raise RuntimeError(
                    f"Job already completed; rerun launcher with --resume: {job['job_id']}"
                )
            if not resume:
                raise RuntimeError(
                    f"Completed job has invalid artifacts and requires --resume: {job['job_id']}"
                )
        if status.get("status") == "running" and _pid_is_live(status.get("pid")):
            raise RuntimeError(
                f"Refusing duplicate live job: {job['job_id']} pid={status.get('pid')}"
            )
        if status.get("status") == "running" and not dry_run and not resume:
            raise RuntimeError(f"Interrupted job requires --resume: {job['job_id']}")
        if status.get("status") == "failed" and not resume and not dry_run:
            raise RuntimeError(f"Failed job requires --resume: {job['job_id']}")
        selected.append(job)
    return selected


def _validate_wave_completion(
    root: Path,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
) -> None:
    failures: list[str] = []
    for job in jobs:
        status = _read_json(_job_status_path(root, str(job["job_id"])))
        if dry_run:
            valid = status.get("status") == "dry_run_passed"
        else:
            valid = _completed_job_is_valid(root, job, status)
        if not valid:
            failures.append(f"{job['job_id']}={status.get('status', 'missing')}")
    if failures:
        raise RuntimeError(
            f"Wave workers did not produce valid terminal states: {failures}"
        )


def _run_wave(
    root: Path,
    config_path: str | Path,
    wave: int,
    jobs: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool,
    resume: bool,
) -> None:
    if not jobs:
        return
    resolved = yaml.safe_load(
        (root / "resolved_config.yaml").read_text(encoding="utf-8")
    )[ROOT_KEY]
    runtime = resolved["runtime"]
    monitor = _ResourceMonitor(
        root / "resource_usage.csv",
        executable=str(runtime.get("nvidia_smi_executable", "nvidia-smi")),
        interval_seconds=float(runtime.get("resource_sample_interval_seconds", 5)),
        wave=wave,
    )
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        monitor.start()
        for job in jobs:
            previous = _read_json(_job_status_path(root, job["job_id"]))
            attempt = int(previous.get("attempt", 0)) + 1
            log_path = root / "logs" / f"{job['job_id']}.attempt_{attempt:02d}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
            env["NEWS_FIRST_JOB_LOG_PATH"] = str(log_path)
            env["PYTHONPATH"] = os.pathsep.join(
                (str(REPO_ROOT / "src"), str(REPO_ROOT))
            )
            threads = str(int(runtime["cpu_threads_per_job"]))
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = threads
            command = build_worker_command(
                config_path,
                root,
                job,
                dry_run=dry_run,
                resume=resume,
            )
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
        pending = set(range(len(processes)))
        while pending:
            for index in tuple(pending):
                code = processes[index].poll()
                if code is None:
                    continue
                pending.remove(index)
                if code != 0:
                    _terminate_processes(processes)
                    raise RuntimeError(
                        f"Wave {wave} job {jobs[index]['job_id']} exited with code {code}"
                    )
            if pending:
                time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        _terminate_processes(processes)
        raise
    finally:
        monitor.stop()
        for handle in handles:
            handle.close()


def launch_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    postprocess_hook: Callable[[Path], Any] | None = None,
) -> Path:
    """Launch every configured fail-closed wave with four isolated GPU jobs."""

    root = prepare_experiment(config_path, output_dir, reuse=True)
    effective_postprocess = postprocess_hook
    if not dry_run and effective_postprocess is None:
        effective_postprocess = run_default_postprocess
    status_path = root / "registry" / "experiment_status.json"
    _write_json(
        status_path,
        {
            "status": "dry_running" if dry_run else "running",
            "current_wave": 0,
            "updated_at_utc": _utc_now(),
        },
    )
    try:
        wave_numbers = sorted(
            {int(job["wave"]) for job in _load_registry(root)["jobs"]}
        )
        for wave in wave_numbers:
            jobs = _jobs_for_wave(root, wave, resume=resume, dry_run=dry_run)
            _write_json(
                status_path,
                {
                    "status": "dry_running" if dry_run else "running",
                    "current_wave": wave,
                    "job_ids": [job["job_id"] for job in jobs],
                    "updated_at_utc": _utc_now(),
                },
            )
            _run_wave(root, config_path, wave, jobs, dry_run=dry_run, resume=resume)
            _validate_wave_completion(root, jobs, dry_run=dry_run)
            _write_resource_summary(root)
            _refresh_registry_exports(root)
        if not dry_run and effective_postprocess is not None:
            # Capture any code drift during long-running training before the
            # analysis/report modules are imported and executed.
            refresh_experiment_lineage(config_path, root)
            _refresh_registry_exports(root)
            effective_postprocess(root)
        _write_json(
            status_path,
            {
                "status": "dry_run_passed" if dry_run else "completed",
                "current_wave": max(wave_numbers, default=0),
                "completed_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
            },
        )
    except BaseException as exc:
        _write_json(
            status_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at_utc": _utc_now(),
            },
        )
        refresh_experiment_lineage(config_path, root)
        _refresh_registry_exports(root)
        raise
    _write_resource_summary(root)
    refresh_experiment_lineage(config_path, root)
    _refresh_registry_exports(root)
    return root


def _write_capacity_stage_state(
    experiment_root: Path,
    *,
    stage: str,
    stage_status: str,
    overall_status: str | None = None,
    **details: Any,
) -> Path:
    path = experiment_root / "capacity_stage_status.json"
    payload = _read_json(path)
    stages = dict(payload.get("stages") or {})
    stage_payload = dict(stages.get(stage) or {})
    stage_payload.update(
        {
            "status": str(stage_status),
            "updated_at_utc": _utc_now(),
            **details,
        }
    )
    stages[stage] = stage_payload
    payload.update(
        {
            "status": str(overall_status or payload.get("status", "running")),
            "current_stage": str(stage),
            "stages": stages,
            "updated_at_utc": _utc_now(),
        }
    )
    return _write_json(path, payload)


def _capacity_stage_entry(experiment_root: Path, stage: str) -> dict[str, Any]:
    payload = _read_json(experiment_root / "capacity_stage_status.json")
    stages = _require_mapping(payload.get("stages"), "capacity stage status.stages")
    if stage not in stages:
        raise ValueError(f"Capacity stage has not been materialized: {stage}")
    return _require_mapping(stages[stage], f"capacity stage {stage}")


def _run_capacity_registered_stage(
    experiment_root: Path,
    config_path: str | Path,
    stage: str,
    *,
    resume: bool,
    dry_run: bool,
) -> None:
    entry = _capacity_stage_entry(experiment_root, stage)
    required_ids = [str(value) for value in entry.get("required_job_ids", ())]
    new_ids = set(str(value) for value in entry.get("new_job_ids", ()))
    registry = _load_registry(experiment_root)
    by_id = {str(job["job_id"]): dict(job) for job in registry["jobs"]}
    missing = sorted(set(required_ids) - set(by_id))
    if missing:
        raise ValueError(f"Capacity stage registry is missing jobs: {missing}")

    _write_capacity_stage_state(
        experiment_root,
        stage=stage,
        stage_status="dry_running" if dry_run else "running",
        overall_status="dry_running" if dry_run else "running",
    )
    wave_numbers = sorted({int(by_id[job_id]["wave"]) for job_id in new_ids})
    for wave in wave_numbers:
        selected_ids = {
            job_id for job_id in new_ids if int(by_id[job_id]["wave"]) == wave
        }
        jobs = [
            job
            for job in _jobs_for_wave(
                experiment_root,
                wave,
                resume=resume,
                dry_run=dry_run,
            )
            if str(job["job_id"]) in selected_ids
        ]
        _run_wave(
            experiment_root,
            config_path,
            wave,
            jobs,
            dry_run=dry_run,
            resume=resume,
        )
        _validate_wave_completion(experiment_root, jobs, dry_run=dry_run)
        _write_resource_summary(experiment_root)
        _refresh_registry_exports(experiment_root)

    failures: list[str] = []
    for job_id in required_ids:
        job = by_id[job_id]
        status = _read_json(_job_status_path(experiment_root, job_id))
        valid = (
            status.get("status") == "dry_run_passed"
            if dry_run
            else _completed_job_is_valid(experiment_root, job, status)
        )
        if not valid:
            failures.append(f"{job_id}={status.get('status', 'missing')}")
    if failures:
        raise RuntimeError(
            f"Capacity stage did not complete all required jobs: {failures}"
        )
    _write_capacity_stage_state(
        experiment_root,
        stage=stage,
        stage_status="dry_run_passed" if dry_run else "completed",
        overall_status="dry_run_passed" if dry_run else "running",
        completed_at_utc=_utc_now(),
    )


def _freeze_capacity_selection(
    experiment_root: Path,
    stage: str,
    *,
    selection_hook: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    path = _capacity_selection_snapshot_path(experiment_root, stage)
    if path.is_file():
        return _require_mapping(
            _read_json(path), f"capacity selection {stage}"
        ), _sha256_file(path)
    if selection_hook is None:
        from scripts.rq3.news_first_vol_capacity_analysis import (
            evaluate_capacity_stage,
        )

        selection_hook = evaluate_capacity_stage
    result = _require_mapping(
        selection_hook(experiment_root, stage),
        f"capacity selection result for {stage}",
    )
    payload = {"schema_version": 1, "stage": stage, **result}
    _write_json(path, payload)
    selection_sha256 = _sha256_file(path)
    _write_capacity_stage_state(
        experiment_root,
        stage=stage,
        stage_status="selected",
        selection_path=str(path),
        selection_sha256=selection_sha256,
    )
    return payload, selection_sha256


def _selection_profiles(
    selection: Mapping[str, Any],
    *,
    expected_count: int,
) -> tuple[str, ...]:
    raw = selection.get("selected_profiles", selection.get("shortlisted_profiles"))
    if isinstance(raw, str):
        raw = [raw]
    profiles = tuple(str(value).strip().lower() for value in (raw or ()))
    if len(profiles) != expected_count or len(set(profiles)) != expected_count:
        raise ValueError(
            f"Capacity selection must contain {expected_count} unique profiles; got {profiles}"
        )
    invalid = sorted(set(profiles) - set(CAPACITY_PROFILE_NAMES))
    if invalid:
        raise ValueError(f"Capacity selection contains unknown profiles: {invalid}")
    return profiles


def _selection_gate_and_winner(
    selection: Mapping[str, Any],
) -> tuple[bool, str]:
    if "gate_passed" not in selection:
        raise ValueError("Capacity gate selection is missing gate_passed")
    passed = bool(selection["gate_passed"])
    winner = (
        str(selection.get("winner_profile", selection.get("selected_profile", "")))
        .strip()
        .lower()
    )
    if passed and winner not in CAPACITY_PROFILE_NAMES:
        raise ValueError(f"Capacity gate winner is invalid: {winner!r}")
    if not passed:
        winner = ""
    return passed, winner


def _mark_capacity_terminal(
    experiment_root: Path,
    *,
    stage: str,
    status: str,
) -> Path:
    _write_capacity_stage_state(
        experiment_root,
        stage=stage,
        stage_status=status,
        overall_status=status,
        completed_at_utc=_utc_now(),
    )
    return _write_json(
        experiment_root / "registry" / "experiment_status.json",
        {
            "status": status,
            "current_stage": stage,
            "completed_at_utc": _utc_now(),
            "updated_at_utc": _utc_now(),
        },
    )


def run_capacity_postprocess(experiment_root: str | Path) -> Path:
    """Run Q4 only after every capacity choice has been frozen from Q3."""

    root = Path(experiment_root).resolve(strict=False)
    stage_status = _read_json(root / "capacity_stage_status.json")
    selection_document = _read_json(root / "capacity_selection.json")
    document_stages = _require_mapping(
        selection_document.get("stages"), "capacity_selection.stages"
    )
    status_stages = _require_mapping(
        stage_status.get("stages"), "capacity_stage_status.stages"
    )
    required_stages = {"regression_screen", "regression_confirm"}
    current_stage = str(stage_status.get("current_stage", "")).strip().lower()
    if current_stage in {"wgan_screen", "wgan_confirm"} or any(
        stage in document_stages or stage in status_stages
        for stage in ("wgan_screen", "wgan_confirm")
    ):
        required_stages.add("wgan_screen")
    wgan_screen = document_stages.get("wgan_screen")
    if isinstance(wgan_screen, Mapping) and bool(wgan_screen.get("gate_passed")):
        required_stages.add("wgan_confirm")
    for stage, status_entry in status_stages.items():
        if stage not in CAPACITY_STAGES or not isinstance(status_entry, Mapping):
            continue
        if str(status_entry.get("selection_sha256", "")).strip():
            required_stages.add(stage)

    existing_stages = {
        stage
        for stage in CAPACITY_STAGES
        if _capacity_selection_snapshot_path(root, stage).is_file()
    }
    for stage in sorted(required_stages | existing_stages):
        snapshot_path = _capacity_selection_snapshot_path(root, stage)
        if not snapshot_path.is_file():
            raise ValueError(
                f"Missing required frozen capacity selection snapshot: {stage}"
            )
        snapshot = _require_mapping(
            _read_json(snapshot_path), f"capacity selection snapshot {stage}"
        )
        expected_sha256 = str(
            _require_mapping(status_stages.get(stage), f"capacity stage {stage}").get(
                "selection_sha256", ""
            )
        )
        if not expected_sha256 or _sha256_file(snapshot_path) != expected_sha256:
            raise ValueError(f"Frozen capacity selection hash mismatch for {stage}")
        snapshot_result = {
            key: value for key, value in snapshot.items() if key != "stage"
        }
        document_result = _require_mapping(
            document_stages.get(stage), f"capacity_selection.stages.{stage}"
        )
        if _canonical_json(snapshot_result) != _canonical_json(document_result):
            raise ValueError(
                f"Cumulative capacity selection disagrees with frozen snapshot: {stage}"
            )
    from scripts.rq3.news_first_vol_capacity_analysis import run_final_q4_analysis
    from scripts.rq3.news_first_vol_capacity_report import render_capacity_report

    run_final_q4_analysis(root, bootstrap_iterations=10_000)
    return Path(render_capacity_report(root))


def _render_capacity_stage_only_report(
    experiment_root: Path,
    config_path: str | Path,
    *,
    stage: str,
    status: str,
) -> Path:
    """Finalize a Regression-gate stop without importing or reading Q4."""

    _write_resource_summary(experiment_root)
    _mark_capacity_terminal(experiment_root, stage=stage, status=status)
    refresh_experiment_lineage(config_path, experiment_root)
    _refresh_registry_exports(experiment_root)
    from scripts.rq3.news_first_vol_capacity_report import render_capacity_report

    report_path = Path(render_capacity_report(experiment_root))
    refresh_experiment_lineage(config_path, experiment_root)
    _refresh_registry_exports(experiment_root)
    return report_path


def launch_capacity_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    selection_hook: Callable[[Path, str], Mapping[str, Any]] | None = None,
    postprocess_hook: Callable[[Path], Any] | None = None,
) -> Path:
    """Run the append-only Regression-first capacity state machine."""

    root = prepare_capacity_experiment(config_path, output_dir, reuse=True)
    resolved = yaml.safe_load(
        (root / "resolved_config.yaml").read_text(encoding="utf-8")
    )[ROOT_KEY]
    experiment_status_path = root / "registry" / "experiment_status.json"
    previous_status = _read_json(experiment_status_path)
    if (
        previous_status.get("status") in {"failed", "running"}
        and not resume
        and not dry_run
    ):
        raise RuntimeError("Interrupted capacity experiment requires --resume")
    _write_json(
        experiment_status_path,
        {
            "status": "dry_running" if dry_run else "running",
            "current_stage": "regression_screen",
            "updated_at_utc": _utc_now(),
        },
    )
    try:
        _run_capacity_registered_stage(
            root,
            config_path,
            "regression_screen",
            resume=resume,
            dry_run=dry_run,
        )
        if dry_run:
            _mark_capacity_terminal(
                root,
                stage="regression_screen",
                status="dry_run_passed",
            )
            _write_resource_summary(root)
            refresh_experiment_lineage(config_path, root)
            _refresh_registry_exports(root)
            return root

        screen_selection, screen_sha256 = _freeze_capacity_selection(
            root,
            "regression_screen",
            selection_hook=selection_hook,
        )
        if "gate_passed" not in screen_selection:
            raise ValueError("Regression screen selection is missing gate_passed")
        if not bool(screen_selection["gate_passed"]):
            _render_capacity_stage_only_report(
                root,
                config_path,
                stage="regression_screen",
                status="completed_no_learned_capacity",
            )
            return root
        shortlisted = _selection_profiles(screen_selection, expected_count=2)
        _append_capacity_stage_jobs(
            root,
            resolved,
            stage="regression_confirm",
            profiles=shortlisted,
            selection_sha256=screen_sha256,
        )
        _run_capacity_registered_stage(
            root,
            config_path,
            "regression_confirm",
            resume=resume,
            dry_run=False,
        )
        regression_selection, regression_sha256 = _freeze_capacity_selection(
            root,
            "regression_confirm",
            selection_hook=selection_hook,
        )
        regression_passed, _regression_winner = _selection_gate_and_winner(
            regression_selection
        )
        if not regression_passed:
            _render_capacity_stage_only_report(
                root,
                config_path,
                stage="regression_confirm",
                status="completed_no_learned_capacity",
            )
            return root

        wgan_profiles = tuple(dict.fromkeys((*shortlisted, "legacy")))
        _append_capacity_stage_jobs(
            root,
            resolved,
            stage="wgan_screen",
            profiles=wgan_profiles,
            selection_sha256=regression_sha256,
        )
        _run_capacity_registered_stage(
            root,
            config_path,
            "wgan_screen",
            resume=resume,
            dry_run=False,
        )
        wgan_selection, wgan_sha256 = _freeze_capacity_selection(
            root,
            "wgan_screen",
            selection_hook=selection_hook,
        )
        wgan_passed, wgan_winner = _selection_gate_and_winner(wgan_selection)
        if not wgan_passed:
            # Regression has already passed its Q3 learnability gate and its
            # capacity is frozen.  Q4 may therefore evaluate that winner even
            # though WGAN is correctly reported as gate_failed/not_run.
            refresh_experiment_lineage(config_path, root)
            _refresh_registry_exports(root)
            effective_postprocess = postprocess_hook or run_capacity_postprocess
            effective_postprocess(root)
            _mark_capacity_terminal(
                root,
                stage="wgan_screen",
                status="completed_regression_only",
            )
            _write_resource_summary(root)
            refresh_experiment_lineage(config_path, root)
            _refresh_registry_exports(root)
            return root

        _append_capacity_stage_jobs(
            root,
            resolved,
            stage="wgan_confirm",
            profiles=(wgan_winner,),
            selection_sha256=wgan_sha256,
        )
        _run_capacity_registered_stage(
            root,
            config_path,
            "wgan_confirm",
            resume=resume,
            dry_run=False,
        )
        wgan_confirm_selection, _wgan_confirm_sha256 = _freeze_capacity_selection(
            root,
            "wgan_confirm",
            selection_hook=selection_hook,
        )
        wgan_confirm_passed, _confirmed_wgan_winner = _selection_gate_and_winner(
            wgan_confirm_selection
        )
        refresh_experiment_lineage(config_path, root)
        _refresh_registry_exports(root)
        effective_postprocess = postprocess_hook or run_capacity_postprocess
        effective_postprocess(root)
        _mark_capacity_terminal(
            root,
            stage="wgan_confirm",
            status=(
                "completed" if wgan_confirm_passed else "completed_regression_only"
            ),
        )
    except BaseException as exc:
        _write_json(
            experiment_status_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at_utc": _utc_now(),
            },
        )
        capacity_status_path = root / "capacity_stage_status.json"
        if capacity_status_path.is_file():
            capacity_status = _read_json(capacity_status_path)
            capacity_status.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at_utc": _utc_now(),
                }
            )
            _write_json(capacity_status_path, capacity_status)
        refresh_experiment_lineage(config_path, root)
        _refresh_registry_exports(root)
        raise
    _write_resource_summary(root)
    refresh_experiment_lineage(config_path, root)
    _refresh_registry_exports(root)
    return root


def dry_run_capacity_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> Path:
    return launch_capacity_experiment(
        config_path,
        output_dir,
        resume=resume,
        dry_run=True,
    )


def dry_run_experiment(
    config_path: str | Path, output_dir: str | Path, *, resume: bool = False
) -> Path:
    return launch_experiment(config_path, output_dir, resume=resume, dry_run=True)


def run_default_postprocess(experiment_root: str | Path) -> Path:
    """Run formal fixed-test analysis and render its portable HTML report."""

    root = Path(experiment_root).resolve(strict=False)
    status_path = root / "registry" / "postprocess_status.json"
    _write_json(status_path, {"status": "running", "started_at_utc": _utc_now()})
    try:
        from scripts.rq3.news_first_vol_comparison_analysis import (
            run_experiment_analysis,
        )
        from scripts.rq3.news_first_vol_training_report import render_portable_report

        analysis_output = run_experiment_analysis(root)
        report_output = render_portable_report(root)
        _write_json(
            status_path,
            {
                "status": "completed",
                "analysis_output": str(analysis_output),
                "report_output": str(report_output),
                "completed_at_utc": _utc_now(),
            },
        )
        return Path(report_output)
    except BaseException as exc:
        _write_json(
            status_path,
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at_utc": _utc_now(),
            },
        )
        raise


def run_news_first_vol_training(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "launch",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
    postprocess_hook: Callable[[Path], Any] | None = None,
) -> Path:
    action = str(action).strip().lower()
    if action == "prepare":
        return prepare_experiment(config_path, output_dir, reuse=bool(reuse or resume))
    if action == "dry-run":
        return dry_run_experiment(config_path, output_dir, resume=resume)
    if action == "worker":
        if not job_id:
            raise ValueError("--job-id is required for the worker action")
        return run_worker(output_dir, job_id, dry_run=worker_dry_run, resume=resume)
    if action == "launch":
        return launch_experiment(
            config_path,
            output_dir,
            resume=resume,
            dry_run=False,
            postprocess_hook=postprocess_hook or run_default_postprocess,
        )
    raise ValueError(f"Unsupported action: {action}")


def run_news_first_vol_capacity_sweep(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    action: str = "launch",
    job_id: str = "",
    resume: bool = False,
    reuse: bool = False,
    worker_dry_run: bool = False,
    selection_hook: Callable[[Path, str], Mapping[str, Any]] | None = None,
    postprocess_hook: Callable[[Path], Any] | None = None,
) -> Path:
    action = str(action).strip().lower()
    if action == "prepare":
        return prepare_capacity_experiment(
            config_path,
            output_dir,
            reuse=bool(reuse or resume),
        )
    if action == "dry-run":
        return dry_run_capacity_experiment(
            config_path,
            output_dir,
            resume=resume,
        )
    if action == "worker":
        if not job_id:
            raise ValueError("--job-id is required for the worker action")
        return run_worker(output_dir, job_id, dry_run=worker_dry_run, resume=resume)
    if action == "launch":
        return launch_capacity_experiment(
            config_path,
            output_dir,
            resume=resume,
            selection_hook=selection_hook,
            postprocess_hook=postprocess_hook,
        )
    raise ValueError(f"Unsupported action: {action}")


__all__ = [
    "build_worker_command",
    "dry_run_capacity_experiment",
    "dry_run_experiment",
    "launch_capacity_experiment",
    "launch_experiment",
    "prepare_capacity_experiment",
    "prepare_experiment",
    "refresh_experiment_lineage",
    "run_capacity_postprocess",
    "run_default_postprocess",
    "run_news_first_vol_capacity_sweep",
    "run_news_first_vol_training",
    "run_worker",
]
