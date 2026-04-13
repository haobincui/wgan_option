"""Shared helpers for trainer-specific training and generate-result directories."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from utils.output_paths import find_latest_run_dir, prepare_run_dir, resolve_output_root, utc_timestamp
from wgan_option.config import Config, derive_training_output_paths

TRAINING_RESOLVED_CONFIG_NAME = "training_resolved_config.yaml"
GENERATE_RESOLVED_CONFIG_NAME = "generate_resolved_config.yaml"
TRAINING_RUN_LOG_NAME = "run.log"

_DEFAULT_CONFIG = Config()
_DIRECTORY_FIELDS = ("models_path", "outputs_path", "samples_path", "metrics_path")
_AUTO_TRAINING_ROOT = Path("outputs/training")


def _normalize_for_compare(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _directory_root_and_tail(path_value: str) -> tuple[Path, Path]:
    path = Path(path_value)
    return path.parent, Path(path.name)


def _normalization_root_and_tail(path_value: str) -> tuple[Path, Path]:
    path = Path(path_value)
    parent = path.parent
    if parent.name:
        return parent.parent, Path(parent.name) / path.name
    return Path("."), Path(path.name)


def _root_candidates(
    config: Config,
    *,
    include_normalization_stats: bool,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    all_candidates: list[tuple[str, Path]] = []
    explicit_candidates: list[tuple[str, Path]] = []

    for field_name in _DIRECTORY_FIELDS:
        raw_value = str(getattr(config, field_name)).strip()
        if not raw_value:
            continue
        root, _ = _directory_root_and_tail(raw_value)
        all_candidates.append((field_name, root))
        if raw_value != str(getattr(_DEFAULT_CONFIG, field_name)):
            explicit_candidates.append((field_name, root))

    if include_normalization_stats:
        raw_value = str(config.normalization_stats_path).strip()
        if raw_value:
            root, _ = _normalization_root_and_tail(raw_value)
            all_candidates.append(("normalization_stats_path", root))
            if raw_value != str(_DEFAULT_CONFIG.normalization_stats_path):
                explicit_candidates.append(("normalization_stats_path", root))

    return all_candidates, explicit_candidates


def _validate_shared_root(root_candidates: Iterable[tuple[str, Path]]) -> Path:
    candidates = list(root_candidates)
    if not candidates:
        raise ValueError("Unable to determine a training output root because no output paths were configured.")

    shared_root = candidates[0][1]
    normalized_shared_root = _normalize_for_compare(shared_root)
    mismatched = [
        f"{field_name}={candidate_root}"
        for field_name, candidate_root in candidates[1:]
        if _normalize_for_compare(candidate_root) != normalized_shared_root
    ]
    if mismatched:
        described = ", ".join(f"{field_name}={candidate_root}" for field_name, candidate_root in candidates)
        raise ValueError(
            "Training output paths must share one common root before timestamp injection. "
            f"Got: {described}"
        )
    return shared_root


def infer_training_output_root(
    config: Config,
    *,
    trainer_id: str,
    fallback_family: str = "default",
    legacy_suffix: str = "all",
) -> Path:
    output_root = str(config.output_root).strip()
    if output_root:
        return Path(output_root)
    if str(config.data_path).strip():
        return resolve_output_root(
            "",
            base_root=_AUTO_TRAINING_ROOT / trainer_id,
            data_path=config.data_path,
            legacy_suffix=legacy_suffix,
            fallback_family=fallback_family,
        )
    all_candidates, explicit_candidates = _root_candidates(config, include_normalization_stats=False)
    return _validate_shared_root(explicit_candidates or all_candidates)


def training_run_timestamp(run_dir: Path | None) -> str:
    if run_dir is not None and str(run_dir.name).strip():
        return run_dir.name
    return utc_timestamp()


def training_run_config_path(config: Config, run_dir: Path | None) -> Path:
    del run_dir
    return Path(config.metrics_path) / TRAINING_RESOLVED_CONFIG_NAME


def training_run_log_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / TRAINING_RUN_LOG_NAME


def generate_result_config_path(generate_dir: str | Path) -> Path:
    return Path(generate_dir) / GENERATE_RESOLVED_CONFIG_NAME


def checkpoint_filename(checkpoint_path: str | Path) -> str:
    stem = Path(checkpoint_path).stem.strip()
    if not stem:
        raise ValueError(f"Could not derive checkpoint filename from path: {checkpoint_path}")
    return stem


def checkpoint_named_dir(base_dir: str | Path, checkpoint_path: str | Path) -> Path:
    target = Path(base_dir) / checkpoint_filename(checkpoint_path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def generate_result_dir(run_dir: str | Path, output_dir: str | Path = "") -> Path:
    run_path = Path(run_dir)
    requested = str(output_dir).strip()
    if not requested:
        target = run_path / "generate_result"
    else:
        candidate = Path(requested)
        target = candidate if candidate.is_absolute() else run_path / candidate
    normalized_run_path = run_path.resolve(strict=False)
    normalized_target = target.resolve(strict=False)
    if not normalized_target.is_relative_to(normalized_run_path):
        raise ValueError(
            f"generate_result output_dir must stay under the training run directory. "
            f"Got output_dir={target} run_dir={run_path}"
        )
    normalized_target.mkdir(parents=True, exist_ok=True)
    return normalized_target


def infer_run_dir_from_checkpoint(checkpoint_path: str | Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return None


def resolve_existing_run_dir(
    *,
    output_root: str | Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    inferred = infer_run_dir_from_checkpoint(checkpoint_path)
    if inferred is not None:
        return inferred
    return find_latest_run_dir(output_root)


def prepare_timestamped_training_config(
    config: Config,
    *,
    trainer_id: str,
    include_normalization_stats: bool = False,
    fallback_family: str = "default",
    legacy_suffix: str = "all",
) -> tuple[Config, Path]:
    """Rewrite training artifact paths under a fresh trainer-specific run directory."""

    timestamp = utc_timestamp()
    base_root = str(config.output_root).strip()
    if base_root:
        run_dir = prepare_run_dir(base_root, timestamp=timestamp, create=False)
        updates = derive_training_output_paths(str(run_dir))
        resolved_config = replace(config, output_root=str(base_root), **updates)
        return resolved_config, run_dir

    if str(config.data_path).strip():
        base_root_path = infer_training_output_root(
            config,
            trainer_id=trainer_id,
            fallback_family=fallback_family,
            legacy_suffix=legacy_suffix,
        )
        run_dir = prepare_run_dir(base_root_path, timestamp=timestamp, create=False)
        updates = derive_training_output_paths(str(run_dir))
        resolved_config = replace(config, output_root=str(base_root_path), **updates)
        return resolved_config, run_dir

    all_candidates, explicit_candidates = _root_candidates(
        config,
        include_normalization_stats=include_normalization_stats,
    )
    if not explicit_candidates and not all_candidates:
        raise ValueError(
            "Unable to determine a training output root. "
            "Provide `output_root` or use a `data_path` under data/processed/<family>/<run_ts>/."
        )
    inferred_root = _validate_shared_root(explicit_candidates or all_candidates)
    run_dir = inferred_root / timestamp

    updates = {}
    for field_name in _DIRECTORY_FIELDS:
        raw_value = str(getattr(config, field_name)).strip()
        if not raw_value:
            continue
        _, relative_tail = _directory_root_and_tail(raw_value)
        updates[field_name] = str(run_dir / relative_tail)

    if include_normalization_stats and str(config.normalization_stats_path).strip():
        _, relative_tail = _normalization_root_and_tail(config.normalization_stats_path)
        updates["normalization_stats_path"] = str(run_dir / relative_tail)

    resolved_config = replace(config, output_root=str(inferred_root), **updates)
    return resolved_config, run_dir
