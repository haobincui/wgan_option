"""One-time migration helper for historical merged-xlsx training directories."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

LEGACY_DIR_PATTERN = re.compile(r"^(vol_xlsx|svi_xlsx)_(\d{8}-\d+)$")
RUN_CONFIG_PATTERN = re.compile(r"^run_config_(\d{8}_\d{6})\.ya?ml$")
FALLBACK_SUFFIX_PATTERN = re.compile(r"^(\d{8})-(\d+)$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate historical merged-xlsx training outputs into root/<run_ts>/...")
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root outputs directory that contains legacy vol_xlsx_* / svi_xlsx_* folders.",
    )
    return parser


def _infer_run_timestamp(legacy_dir: Path) -> str:
    metrics_dir = legacy_dir / "metrics"
    run_configs = sorted(metrics_dir.glob("run_config_*.yaml")) + sorted(metrics_dir.glob("run_config_*.yml"))
    if len(run_configs) == 1:
        match = RUN_CONFIG_PATTERN.match(run_configs[0].name)
        if not match:
            raise ValueError(f"Could not parse run timestamp from {run_configs[0]}")
        return match.group(1)
    if len(run_configs) > 1:
        raise ValueError(f"Found multiple run_config files under {metrics_dir}, cannot infer a unique run timestamp.")

    legacy_match = LEGACY_DIR_PATTERN.match(legacy_dir.name)
    if not legacy_match:
        raise ValueError(f"Legacy directory name does not match the expected pattern: {legacy_dir.name}")
    suffix_match = FALLBACK_SUFFIX_PATTERN.match(legacy_match.group(2))
    if not suffix_match:
        raise ValueError(f"Legacy directory suffix does not match the expected pattern: {legacy_match.group(2)}")
    return f"{suffix_match.group(1)}_{int(suffix_match.group(2)):06d}"


def _legacy_training_directories(outputs_root: Path) -> List[Path]:
    legacy_dirs: List[Path] = []
    for candidate in sorted(outputs_root.iterdir()) if outputs_root.exists() else []:
        if candidate.is_dir() and LEGACY_DIR_PATTERN.match(candidate.name):
            legacy_dirs.append(candidate)
    return legacy_dirs


def migrate_training_outputs(outputs_root: str | Path) -> Tuple[List[Path], List[str]]:
    """Move historical training runs into outputs/{vol_xlsx|svi_xlsx}/<run_ts>/."""

    root = Path(outputs_root)
    migrated: List[Path] = []
    errors: List[str] = []

    for legacy_dir in _legacy_training_directories(root):
        match = LEGACY_DIR_PATTERN.match(legacy_dir.name)
        assert match is not None
        family = match.group(1)
        try:
            run_ts = _infer_run_timestamp(legacy_dir)
            destination = root / family / run_ts
            if destination.exists():
                raise FileExistsError(f"Destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_dir), str(destination))
            (destination / "samples").mkdir(parents=True, exist_ok=True)
            migrated.append(destination)
        except Exception as exc:  # pragma: no cover - exercised via returned errors
            errors.append(f"{legacy_dir}: {exc}")

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
