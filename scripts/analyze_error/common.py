"""Shared CLI and output helpers for analyze-error entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.generate_result.common import (  # noqa: E402
    VALID_SELECTION_MODES,
    VALID_SPLITS,
    build_surface_grids,
    prepare_run_output_dir,
    resolve_checkpoint_path,
    select_samples_from_split,
    split_metadata,
    write_summary_csv,
)
from wgan_option.analysis_config import (  # noqa: E402
    AnalysisConfig,
    analysis_config_to_dict,
    load_analysis_config,
    parse_analysis_overrides,
    save_analysis_config_yaml,
)


def build_analysis_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    """Create a standard analysis parser."""

    parser = argparse.ArgumentParser(description=description)
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
        help="Print resolved config before analysis.",
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
    parser.add_argument("--bootstrap-samples", type=int, default=None, help="Number of bootstrap replicates.")
    parser.add_argument("--confidence-level", type=float, default=None, help="Bootstrap confidence level in (0, 1).")
    parser.add_argument("--bootstrap-seed", type=int, default=None, help="Random seed for bootstrap resampling.")
    parser.add_argument(
        "--no-bootstrap-distribution",
        action="store_true",
        help="Disable bootstrap_distribution.csv output.",
    )
    return parser


def resolve_analysis_config(
    *,
    argv: Optional[Iterable[str]],
    description: str,
    default_config_path: str,
) -> AnalysisConfig:
    """Parse CLI flags and return a validated analysis config."""

    parser = build_analysis_arg_parser(description=description, default_config_path=default_config_path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = parse_analysis_overrides(args.set)
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
    if args.bootstrap_samples is not None:
        overrides["bootstrap_samples"] = args.bootstrap_samples
    if args.confidence_level is not None:
        overrides["confidence_level"] = args.confidence_level
    if args.bootstrap_seed is not None:
        overrides["bootstrap_seed"] = args.bootstrap_seed
    if args.no_bootstrap_distribution:
        overrides["save_bootstrap_distribution"] = False

    config = load_analysis_config(config_path=args.config, overrides=overrides)
    validate_analysis_config(config)

    if args.print_config:
        print(yaml.safe_dump(analysis_config_to_dict(config), sort_keys=False, allow_unicode=False))

    return config


def validate_analysis_config(config: AnalysisConfig) -> None:
    """Fail fast on invalid high-level analysis config values."""

    if config.split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got: {config.split}")
    if config.selection_mode not in VALID_SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {sorted(VALID_SELECTION_MODES)}, got: {config.selection_mode}"
        )
    if config.selection_mode == "sample_id" and not str(config.sample_id).strip():
        raise ValueError("sample_id must be provided when selection_mode=sample_id.")
    if config.selection_mode == "row_index" and int(config.row_index) < 0:
        raise ValueError("row_index must be >= 0 when selection_mode=row_index.")
    if config.selection_mode == "first_n" and int(config.limit) <= 0:
        raise ValueError("limit must be > 0 when selection_mode=first_n.")
    if int(config.histogram_bins) <= 0:
        raise ValueError("histogram_bins must be > 0.")
    if int(config.bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be > 0.")
    if not (0.0 < float(config.confidence_level) < 1.0):
        raise ValueError("confidence_level must be between 0 and 1.")


def write_resolved_config(config: AnalysisConfig, run_dir: str | Path) -> Path:
    """Persist the resolved runtime config into the run directory."""

    return save_analysis_config_yaml(config, Path(run_dir) / "resolved_config.yaml")


def compute_error_metrics(error_surface: np.ndarray) -> Dict[str, float | int]:
    """Compute signed-error and absolute-error metrics from a 2D error surface."""

    error = np.asarray(error_surface, dtype=np.float32)
    if error.ndim != 2:
        raise ValueError(f"Expected a 2D error surface, got shape {error.shape}")
    squared_error = error * error
    abs_error = np.abs(error)
    return {
        "mse": float(np.mean(squared_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mae": float(np.mean(abs_error)),
        "max_abs": float(np.max(abs_error)),
        "error_mean": float(np.mean(error)),
        "error_std": float(np.std(error)),
        "error_min": float(np.min(error)),
        "error_max": float(np.max(error)),
        "surface_rows": int(error.shape[0]),
        "surface_cols": int(error.shape[1]),
    }


def write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """Persist a JSON payload."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
    return output


def write_bootstrap_outputs(
    *,
    summary: Mapping[str, Any],
    distribution: Sequence[float],
    run_dir: str | Path,
    save_distribution: bool,
) -> None:
    """Persist bootstrap outputs under one analysis run directory."""

    run_path = Path(run_dir)
    write_summary_csv([summary], run_path / "bootstrap_summary.csv")
    write_json(summary, run_path / "bootstrap_summary.json")

    if save_distribution:
        rows = [
            {"replicate_index": int(index), "mean_mse": float(value)}
            for index, value in enumerate(distribution)
        ]
        write_summary_csv(rows, run_path / "bootstrap_distribution.csv")
