"""Shared runtime helpers for result-generation and error-analysis jobs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from wgan_option.surface_grid import build_surface_grids_from_config
from wgan_option.utils.merged_xlsx import OrderedSplitSelection

VALID_SELECTION_MODES = {"sample_id", "row_index", "first_n", "all"}
VALID_SPLITS = {"train", "val", "all"}
RUN_DIR_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def add_shared_sample_selection_args(
    parser: argparse.ArgumentParser,
    *,
    default_config_path: str,
) -> None:
    """Add the shared CLI arguments common to result-generation and error-analysis."""

    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help=f"Path to YAML config file (default: {default_config_path})",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field from CLI. Repeat this arg for multiple overrides.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output root directory.")
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default=None, help="Which chronological split to use.")
    parser.add_argument(
        "--selection-mode",
        choices=sorted(VALID_SELECTION_MODES),
        default=None,
        help="How to choose samples from the selected split.",
    )
    parser.add_argument("--sample-id", type=str, default=None, help="Sample id when --selection-mode=sample_id.")
    parser.add_argument("--row-index", type=int, default=None, help="0-based row index inside the selected split.")
    parser.add_argument("--limit", type=int, default=None, help="Sample count when --selection-mode=first_n.")


def apply_shared_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> None:
    """Apply shared CLI overrides into the config override dict."""

    if args.checkpoint is not None:
        overrides["checkpoint_path"] = args.checkpoint
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.split is not None:
        overrides["split"] = args.split
    if args.selection_mode is not None:
        overrides["selection_mode"] = args.selection_mode
    if args.sample_id is not None:
        overrides["sample_id"] = args.sample_id
    if args.row_index is not None:
        overrides["row_index"] = args.row_index
    if args.limit is not None:
        overrides["limit"] = args.limit


def prepare_run_output_dir(output_root: str | Path) -> Path:
    """Create a timestamped output directory for one runtime job."""

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_sample_filename(sample_id: str) -> str:
    """Turn a sample id into a filename-safe stem."""

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(sample_id))
    return cleaned or "sample"


def write_summary_csv(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Write summary rows to CSV using the union of observed keys."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
    return output


def save_payload_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Save a JSON payload for one runtime artifact."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
    return output


write_json = save_payload_json


def compute_surface_metrics(generated_surface: np.ndarray, real_surface: np.ndarray) -> Dict[str, float]:
    """Compute simple surface comparison metrics."""

    diff = np.asarray(generated_surface, dtype=np.float32) - np.asarray(real_surface, dtype=np.float32)
    abs_diff = np.abs(diff)
    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(abs_diff.max()),
    }


def resolve_checkpoint_path(
    config: Any,
    *,
    artifact_key: str,
    fallback_filenames: Sequence[str],
) -> Path:
    """Resolve a checkpoint path using explicit config, best-checkpoint metadata, then defaults."""

    candidates: List[Path] = []

    if str(config.checkpoint_path).strip():
        candidates.append(Path(config.checkpoint_path))

    for metrics_dir in _artifact_directory_candidates(str(config.metrics_path).strip(), leaf_name="metrics"):
        best_path = metrics_dir / "best_checkpoint.json"
        if best_path.exists():
            with best_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            artifact_path = payload.get("artifacts", {}).get(artifact_key)
            if artifact_path:
                candidates.append(Path(artifact_path))

    for model_root in _artifact_directory_candidates(str(config.models_path).strip(), leaf_name="checkpoints"):
        for filename in fallback_filenames:
            candidates.append(model_root / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve checkpoint for artifact '{artifact_key}'. Tried: "
        f"{[str(path) for path in candidates]}"
    )


def _artifact_directory_candidates(base_path: str, *, leaf_name: str) -> List[Path]:
    """Return direct and latest-run artifact directories for metrics/checkpoints lookup."""

    if not base_path:
        return []

    base = Path(base_path)
    directories: List[Path] = []
    seen: set[str] = set()

    def add_directory(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            directories.append(path)

    add_directory(base)
    if base.name != leaf_name:
        add_directory(base / leaf_name)

    container_roots = [base]
    if base.name == leaf_name:
        container_roots.append(base.parent)

    for container_root in container_roots:
        if not container_root.exists() or not container_root.is_dir():
            continue
        run_dirs = sorted(
            [
                child
                for child in container_root.iterdir()
                if child.is_dir() and RUN_DIR_PATTERN.match(child.name)
            ],
            reverse=True,
        )
        for run_dir in run_dirs:
            add_directory(run_dir / leaf_name)

    return directories


def select_samples_from_split(
    split_selection: OrderedSplitSelection,
    config: Any,
) -> List[Any]:
    """Apply sample-id / row-index / first-n / all selection inside one split."""

    candidates = list(split_selection.selected_items)
    if not candidates:
        raise ValueError(f"No samples are available in split '{split_selection.effective_split}'.")

    selection_mode = str(config.selection_mode).strip().lower()
    if selection_mode == "all":
        return candidates
    if selection_mode == "first_n":
        return candidates[: int(config.limit)]
    if selection_mode == "row_index":
        row_index = int(config.row_index)
        if row_index >= len(candidates):
            raise IndexError(
                f"row_index={row_index} is out of range for split '{split_selection.effective_split}' "
                f"with {len(candidates)} sample(s)."
            )
        return [candidates[row_index]]
    if selection_mode == "sample_id":
        sample_id = str(config.sample_id)
        matched = [sample for sample in candidates if str(getattr(sample, "sample_id", "")) == sample_id]
        if not matched:
            raise ValueError(f"sample_id='{sample_id}' was not found in split '{split_selection.effective_split}'.")
        return matched
    raise ValueError(f"Unsupported selection_mode: {config.selection_mode}")


def build_surface_grids(config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed grid used to reconstruct SVI into vol surfaces."""

    return build_surface_grids_from_config(config, dtype=np.float32)


def split_metadata(split_selection: OrderedSplitSelection) -> Dict[str, Any]:
    """Serialize split-selection metadata for payloads and summaries."""

    return {
        "requested_split": split_selection.requested_split,
        "effective_split": split_selection.effective_split,
        "fallback_to_all": bool(split_selection.fallback_to_all),
        "train_count": len(split_selection.train_items),
        "val_count": len(split_selection.val_items),
        "total_count": len(split_selection.all_items),
    }
