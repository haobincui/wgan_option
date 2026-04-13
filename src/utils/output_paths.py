"""Shared helpers for dataset-aware output roots and timestamped run directories."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_TIMESTAMP_PATTERN = re.compile(r"\d{8}_\d{6}$")


def utc_timestamp() -> str:
    """Return the canonical UTC timestamp used by run directories."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def infer_dataset_family(
    data_path: str | Path,
    *,
    legacy_suffix: str | None = None,
    fallback: str = "default",
) -> str:
    """Infer the dataset family from a processed workbook path.

    When the path contains a ``data/processed/<family>/...`` segment, the
    ``<family>`` component is preferred.  Otherwise the function falls back to
    the workbook's parent/grandparent directory names.
    """

    raw_value = str(data_path).strip()
    if not raw_value:
        return fallback

    path = Path(raw_value)
    parts = path.parts
    processed_indexes = [idx for idx, part in enumerate(parts) if part == "processed"]
    if processed_indexes:
        processed_idx = processed_indexes[-1]
        if processed_idx + 1 < len(parts):
            family = str(parts[processed_idx + 1]).strip()
            if family:
                if legacy_suffix and "-" not in family:
                    return f"{family}-{legacy_suffix}"
                return family

    if path.parent.name and path.parent.name not in {"", "."}:
        grandparent = str(path.parent.parent.name).strip()
        if grandparent:
            return grandparent
        return str(path.parent.name).strip() or fallback

    return fallback


def default_output_root(
    base_root: str | Path,
    data_path: str | Path,
    *,
    legacy_suffix: str | None = None,
    fallback_family: str = "default",
) -> Path:
    """Build the default output root for one dataset family."""

    family = infer_dataset_family(
        data_path,
        legacy_suffix=legacy_suffix,
        fallback=fallback_family,
    )
    return Path(base_root) / family


def resolve_output_root(
    output_root: str | Path,
    *,
    base_root: str | Path,
    data_path: str | Path,
    legacy_suffix: str | None = None,
    fallback_family: str = "default",
) -> Path:
    """Return the explicit output root or the dataset-aware default."""

    raw_value = str(output_root).strip()
    if raw_value:
        return Path(raw_value)
    return default_output_root(
        base_root,
        data_path,
        legacy_suffix=legacy_suffix,
        fallback_family=fallback_family,
    )


def prepare_run_dir(
    output_root: str | Path,
    *,
    timestamp: str | None = None,
    create: bool = True,
) -> Path:
    """Return a timestamped run directory under ``output_root``."""

    run_dir = Path(output_root) / (timestamp or utc_timestamp())
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_latest_run_dir(root: str | Path) -> Path:
    """Return the most recent timestamped subdirectory under ``root``."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Run root directory does not exist: {root}")

    run_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and _TIMESTAMP_PATTERN.fullmatch(path.name)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped run directories found under {root}")
    return run_dirs[-1]


def find_best_checkpoint(run_dir: str | Path, filename: str = "") -> Path:
    """Locate the best checkpoint inside ``<run_dir>/checkpoints``."""

    checkpoints_dir = Path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory does not exist: {checkpoints_dir}")

    if filename:
        candidate = checkpoints_dir / filename
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    for candidate_name in ("volgan_best.pt", "generator_best.pt"):
        candidate = checkpoints_dir / candidate_name
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No best checkpoint found in {checkpoints_dir}")
