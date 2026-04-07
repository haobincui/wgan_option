"""Helpers for timestamping merged-xlsx training artifact directories."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from wgan_option.config import Config, derive_training_output_paths

_DEFAULT_CONFIG = Config()
_DIRECTORY_FIELDS = ("models_path", "outputs_path", "samples_path", "metrics_path")


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


def _create_training_output_directories(
    config: Config,
    *,
    include_normalization_stats: bool,
) -> None:
    for field_name in _DIRECTORY_FIELDS:
        raw_value = str(getattr(config, field_name)).strip()
        if raw_value:
            Path(raw_value).mkdir(parents=True, exist_ok=True)

    if include_normalization_stats and str(config.normalization_stats_path).strip():
        Path(config.normalization_stats_path).parent.mkdir(parents=True, exist_ok=True)


def prepare_timestamped_training_config(
    config: Config,
    *,
    include_normalization_stats: bool = False,
) -> tuple[Config, Path]:
    """Rewrite training artifact paths under a fresh <root>/<run_ts>/ directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = str(config.output_root).strip()

    if output_root:
        run_dir = Path(output_root) / timestamp
        updates = derive_training_output_paths(str(run_dir))
        resolved_config = replace(config, **updates)
        _create_training_output_directories(
            resolved_config,
            include_normalization_stats=include_normalization_stats,
        )
        return resolved_config, run_dir

    all_candidates, explicit_candidates = _root_candidates(
        config,
        include_normalization_stats=include_normalization_stats,
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

    resolved_config = replace(config, **updates)
    _create_training_output_directories(
        resolved_config,
        include_normalization_stats=include_normalization_stats,
    )
    return resolved_config, run_dir
