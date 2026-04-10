"""Result-generation CLI helpers plus compatibility re-exports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import yaml

import scripts._path_setup  # noqa: F401

from wgan_option.postprocess_runtime import (  # noqa: E402
    VALID_SELECTION_MODES,
    VALID_SPLITS,
    add_shared_sample_selection_args,
    apply_shared_overrides,
    build_surface_grids,
    compute_surface_metrics,
    prepare_run_output_dir,
    resolve_checkpoint_path,
    safe_sample_filename,
    save_payload_json,
    select_samples_from_split,
    split_metadata,
    write_json,
    write_summary_csv,
)
from wgan_option.result_config import (  # noqa: E402
    DEFAULT_SVI_RESULT_CONFIG_PATH,
    DEFAULT_VOL_RESULT_CONFIG_PATH,
    GenerateResultConfig,
    generate_result_config_to_dict,
    load_generate_result_config,
    parse_generate_result_overrides,
    save_generate_result_config_yaml,
)


def build_result_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    """Create a standard result-generation parser."""

    parser = argparse.ArgumentParser(description=description)
    add_shared_sample_selection_args(parser, default_config_path=default_config_path)
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG plot generation.")
    parser.add_argument("--no-json", action="store_true", help="Disable per-sample JSON payload output.")
    return parser


def resolve_result_config(
    *,
    argv: Optional[Iterable[str]],
    description: str,
    default_config_path: str,
) -> GenerateResultConfig:
    """Parse CLI flags and return a validated result-generation config."""

    parser = build_result_arg_parser(description=description, default_config_path=default_config_path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = parse_generate_result_overrides(args.set)
    apply_shared_overrides(args, overrides)
    if args.no_plot:
        overrides["save_plots"] = False
    if args.no_json:
        overrides["save_json"] = False

    config = load_generate_result_config(config_path=args.config, overrides=overrides)
    validate_result_config(config)

    if args.print_config:
        print(yaml.safe_dump(generate_result_config_to_dict(config), sort_keys=False, allow_unicode=False))

    return config


def validate_result_config(config: GenerateResultConfig) -> None:
    """Fail fast on invalid high-level generation config values."""

    if config.split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got: {config.split}")
    if config.selection_mode not in VALID_SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {sorted(VALID_SELECTION_MODES)}, got: {config.selection_mode}"
        )
    if str(config.plot_style).strip().lower() != "heatmap_diff":
        raise ValueError(f"Only plot_style=heatmap_diff is currently supported, got: {config.plot_style}")
    if config.selection_mode == "sample_id" and not str(config.sample_id).strip():
        raise ValueError("sample_id must be provided when selection_mode=sample_id.")
    if config.selection_mode == "row_index" and int(config.row_index) < 0:
        raise ValueError("row_index must be >= 0 when selection_mode=row_index.")
    if config.selection_mode == "first_n" and int(config.limit) <= 0:
        raise ValueError("limit must be > 0 when selection_mode=first_n.")


def write_resolved_config(config: GenerateResultConfig, run_dir: str | Path) -> Path:
    """Persist the resolved runtime config into the run directory."""

    return save_generate_result_config_yaml(config, Path(run_dir) / "resolved_config.yaml")


DEFAULT_RESULT_CONFIG_PATHS = {
    "vol": DEFAULT_VOL_RESULT_CONFIG_PATH,
    "svi": DEFAULT_SVI_RESULT_CONFIG_PATH,
}
