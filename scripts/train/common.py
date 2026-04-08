"""Shared CLI helpers for training entrypoints."""

from __future__ import annotations

import argparse
from typing import Iterable, Optional, Type

import yaml

import scripts._path_setup  # noqa: F401

from wgan_option.config import config_to_dict, load_config, parse_cli_overrides  # noqa: E402

DEFAULT_VOL_CONFIG_PATH = "configs/wgan/train_vol_xlsx.yaml"
DEFAULT_SVI_CONFIG_PATH = "configs/wgan/train_svi_xlsx.yaml"


def build_train_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    """Create a standard training CLI parser."""

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
        "--dry-run",
        action="store_true",
        help="Load config, build dataloader/model, and exit without training.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config before run.",
    )
    return parser


def run_training_cli(
    *,
    argv: Optional[Iterable[str]],
    description: str,
    default_config_path: str,
    trainer_cls: Type,
) -> None:
    """Parse config flags and run the requested trainer."""

    parser = build_train_arg_parser(description=description, default_config_path=default_config_path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    overrides = parse_cli_overrides(args.set)
    config = load_config(config_path=args.config, overrides=overrides)

    if args.print_config:
        print(yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=False))

    trainer = trainer_cls(config)
    if args.dry_run:
        trainer.dry_run()
    else:
        trainer.start_train()
