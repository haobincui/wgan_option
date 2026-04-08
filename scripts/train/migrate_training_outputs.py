"""Migration helper for historical merged-xlsx training directories."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

LEGACY_DIR_PATTERN = re.compile(r"^(vol_xlsx|svi_xlsx)_(\d{8}-\d+)$")
RUN_CONFIG_PATTERN = re.compile(r"^run_config_(\d{8}_\d{6})\.ya?ml$")
FALLBACK_SUFFIX_PATTERN = re.compile(r"^(\d{8})-(\d+)$")
RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")
TRAINING_ROOT_NAME = "training"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate historical merged-xlsx training outputs into outputs/training/{vol_xlsx|svi_xlsx}/<run_ts>/..."
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root outputs directory that contains historical vol_xlsx / svi_xlsx training folders.",
    )
    return parser


def _infer_run_timestamp(source_dir: Path) -> str:
    metrics_dir = source_dir / "metrics"
    run_configs = sorted(metrics_dir.glob("run_config_*.yaml")) + sorted(metrics_dir.glob("run_config_*.yml"))
    if len(run_configs) == 1:
        match = RUN_CONFIG_PATTERN.match(run_configs[0].name)
        if not match:
            raise ValueError(f"Could not parse run timestamp from {run_configs[0]}")
        return match.group(1)
    if len(run_configs) > 1:
        raise ValueError(f"Found multiple run_config files under {metrics_dir}, cannot infer a unique run timestamp.")

    if RUN_DIR_PATTERN.match(source_dir.name):
        return source_dir.name

    legacy_match = LEGACY_DIR_PATTERN.match(source_dir.name)
    if not legacy_match:
        raise ValueError(f"Source directory name does not match an expected training-run pattern: {source_dir.name}")
    suffix_match = FALLBACK_SUFFIX_PATTERN.match(legacy_match.group(2))
    if not suffix_match:
        raise ValueError(f"Legacy directory suffix does not match the expected pattern: {legacy_match.group(2)}")
    return f"{suffix_match.group(1)}_{int(suffix_match.group(2)):06d}"


def _legacy_training_directories(outputs_root: Path) -> List[tuple[str, Path]]:
    legacy_dirs: List[tuple[str, Path]] = []
    for candidate in sorted(outputs_root.iterdir()) if outputs_root.exists() else []:
        if not candidate.is_dir():
            continue

        legacy_match = LEGACY_DIR_PATTERN.match(candidate.name)
        if legacy_match:
            legacy_dirs.append((legacy_match.group(1), candidate))
            continue

        if candidate.name not in {"vol_xlsx", "svi_xlsx"}:
            continue

        for run_dir in sorted(path for path in candidate.iterdir() if path.is_dir() and RUN_DIR_PATTERN.match(path.name)):
            legacy_dirs.append((candidate.name, run_dir))
    return legacy_dirs


def _cleanup_empty_training_family_dirs(source_dir: Path, outputs_root: Path) -> None:
    family_dir = source_dir.parent
    if family_dir == outputs_root:
        return
    if family_dir.parent != outputs_root:
        return
    if family_dir.name not in {"vol_xlsx", "svi_xlsx"}:
        return
    if any(family_dir.iterdir()):
        return
    family_dir.rmdir()


def migrate_training_outputs(outputs_root: str | Path) -> Tuple[List[Path], List[str]]:
    """Move historical training runs into outputs/training/{vol_xlsx|svi_xlsx}/<run_ts>/."""

    root = Path(outputs_root)
    migrated: List[Path] = []
    errors: List[str] = []

    for family, source_dir in _legacy_training_directories(root):
        try:
            run_ts = _infer_run_timestamp(source_dir)
            destination = root / TRAINING_ROOT_NAME / family / run_ts
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_dir), str(destination))
            (destination / "samples").mkdir(parents=True, exist_ok=True)
            _cleanup_empty_training_family_dirs(source_dir, root)
            migrated.append(destination)
        except Exception as exc:  # pragma: no cover - exercised via returned errors
            errors.append(f"{source_dir}: {exc}")

    return migrated, errors


def main(argv: Iterable[str] | None = None) -> List[Path]:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    migrated, errors = migrate_training_outputs(args.outputs_root)
    for destination in migrated:
        print(f"Migrated: {destination}")
    for error in errors:
        print(f"Skipped: {error}", file=sys.stderr)

    if errors:
        raise SystemExit(1)
    return migrated


if __name__ == "__main__":
    main()
