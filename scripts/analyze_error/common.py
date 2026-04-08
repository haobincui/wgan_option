"""Shared CLI and output helpers for analyze-error entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

import scripts._path_setup  # noqa: F401

from scripts.generate_result.common import (  # noqa: E402
    VALID_SELECTION_MODES,
    VALID_SPLITS,
    add_shared_sample_selection_args,
    apply_shared_overrides,
    build_surface_grids,
    prepare_run_output_dir,
    resolve_checkpoint_path,
    select_samples_from_split,
    split_metadata,
    write_json,
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
    add_shared_sample_selection_args(parser, default_config_path=default_config_path)
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
    apply_shared_overrides(args, overrides)
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
