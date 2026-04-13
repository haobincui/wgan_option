"""Unified CLI router for trainer-led generate-result jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Callable, Iterable

import yaml

from utils.generate_result_runtime import build_result_arg_parser
from utils.postprocess_runtime import apply_shared_overrides
from utils.result_config import generate_result_config_to_dict, parse_generate_result_overrides
from wgan_option.config import load_config
from wgan_option.train_svi_xlsx import SviXlsxTrainer
from wgan_option.train_vol_regression_xlsx import VolSurfaceRegressionTrainer
from wgan_option.train_vol_xlsx import VolSurfaceXlsxTrainer
from wgan_option.visualization.surface_plot import plot_surface_payload

HELP_TEXT = dedent(
    """
    Usage:
      python scripts/generate_result/main.py vol --config configs/wgan/train_vol_xlsx.yaml
      python scripts/generate_result/main.py vol-regression --config configs/wgan/train_vol_regression_xlsx.yaml
      python scripts/generate_result/main.py svi --config configs/wgan/train_svi_xlsx.yaml
      python scripts/generate_result/main.py plot --input-json <payload.json> [--output <file.png>]

    Notes:
      The config file is the same merged training YAML used for training.
    """
).strip()


def _run_trainer_generate(
    *,
    argv: Iterable[str] | None,
    description: str,
    default_config_path: str,
    trainer_cls: type,
) -> Path:
    parser = build_result_arg_parser(description=description, default_config_path=default_config_path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = parse_generate_result_overrides(args.set)
    apply_shared_overrides(args, overrides)
    if args.no_plot:
        overrides["save_plots"] = False
    if args.no_json:
        overrides["save_json"] = False

    config = load_config(args.config)
    trainer = trainer_cls(config, config_path=args.config)
    if args.print_config:
        resolved_config, _ = trainer._prepare_generate_result(overrides=overrides, config_path=args.config)
        print(yaml.safe_dump(generate_result_config_to_dict(resolved_config), sort_keys=False, allow_unicode=False))
    return trainer.generate_result(overrides=overrides, config_path=args.config)


def _plot_command(argv: Iterable[str] | None) -> Path:
    parser = argparse.ArgumentParser(description="Render a surface comparison payload JSON into a PNG.")
    parser.add_argument("--input-json", required=True, help="Path to a generate_result payload JSON file.")
    parser.add_argument("--output", default=None, help="Optional PNG output path.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")
    return plot_surface_payload(payload, output_path)


COMMANDS: dict[str, Callable[[Iterable[str] | None], Path | None]] = {
    "vol": lambda argv: _run_trainer_generate(
        argv=argv,
        description="Generate future vol surfaces from a trained WGAN checkpoint.",
        default_config_path="configs/wgan/train_vol_xlsx.yaml",
        trainer_cls=VolSurfaceXlsxTrainer,
    ),
    "vol-regression": lambda argv: _run_trainer_generate(
        argv=argv,
        description="Generate future vol surfaces from a deterministic regression checkpoint.",
        default_config_path="configs/wgan/train_vol_regression_xlsx.yaml",
        trainer_cls=VolSurfaceRegressionTrainer,
    ),
    "svi": lambda argv: _run_trainer_generate(
        argv=argv,
        description="Generate future SVI params, reconstruct surfaces, and compare them.",
        default_config_path="configs/wgan/train_svi_xlsx.yaml",
        trainer_cls=SviXlsxTrainer,
    ),
    "plot": _plot_command,
}


def main(argv: list[str] | None = None) -> Path | None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if not argv_list or argv_list[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return None

    subcommand = argv_list[0]
    if subcommand not in COMMANDS:
        raise SystemExit(f"Unsupported subcommand: {subcommand}\n\n{HELP_TEXT}")
    return COMMANDS[subcommand](argv_list[1:])
