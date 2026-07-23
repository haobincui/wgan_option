"""Shared CLI helpers for standalone trainer-led model packages."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from typing import Any

import yaml


def build_train_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    """Create the standard training parser for trainer-led standalone models."""

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
        help="Override one training config field from CLI. Repeat this arg for multiple overrides.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load config, build data/model state, and exit.")
    parser.add_argument("--train-only", action="store_true", help="Run training only and skip generate_result.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Skip training and run generate_result from the latest/existing training run.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print resolved training config.")
    return parser


def run_training_cli(
    *,
    argv: Iterable[str] | None,
    description: str,
    default_config_path: str,
    load_config: Callable[[str, dict[str, Any] | None], Any],
    parse_overrides: Callable[[Iterable[str]], dict[str, Any]],
    config_to_dict: Callable[[Any], dict[str, Any]],
    trainer_cls: type,
) -> Any:
    """Parse training flags and execute one trainer lifecycle."""

    parser = build_train_arg_parser(description=description, default_config_path=default_config_path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    overrides = parse_overrides(args.set)
    config = load_config(args.config, overrides=overrides)

    if args.print_config:
        print(yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=False))

    if args.train_only and args.generate_only:
        raise ValueError("--train-only and --generate-only cannot be used together.")

    trainer = trainer_cls(config, config_path=args.config)
    if args.dry_run:
        return trainer.dry_run()
    if args.generate_only:
        return trainer.generate_result(config_path=args.config)
    if args.train_only:
        return trainer.train()
    return trainer.run_pipeline(config_path=args.config)


def build_generate_arg_parser(*, description: str, default_config_path: str) -> argparse.ArgumentParser:
    """Create the standard generate-result parser for standalone models."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help=f"Path to training YAML config file (default: {default_config_path})",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one generate_result config field from CLI. Repeat this arg for multiple overrides.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path override.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override generate_result subdirectory.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default=None,
        help="Which split to sample.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["all", "head", "tail"],
        default=None,
        help="How to choose samples from the selected split.",
    )
    parser.add_argument(
        "--selection-count",
        type=int,
        default=None,
        help="Sample count for selection_mode=head/tail.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG plot generation.")
    parser.add_argument("--no-json", action="store_true", help="Disable per-sample JSON payload output.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved generate_result config.")
    return parser


def run_generate_cli(
    *,
    argv: Iterable[str] | None,
    description: str,
    default_config_path: str,
    load_train_config: Callable[[str, dict[str, Any] | None], Any],
    parse_overrides: Callable[[Iterable[str]], dict[str, Any]],
    config_to_dict: Callable[[Any], dict[str, Any]],
    trainer_cls: type,
) -> Any:
    """Parse generate-result flags and call the trainer-led generate flow."""

    parser = build_generate_arg_parser(description=description, default_config_path=default_config_path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = parse_overrides(args.set)
    if args.checkpoint is not None:
        overrides["checkpoint_path"] = args.checkpoint
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.split is not None:
        overrides["split"] = args.split
    if args.selection_mode is not None:
        overrides["selection_mode"] = args.selection_mode
    if args.selection_count is not None:
        overrides["selection_count"] = args.selection_count
    if args.no_plot:
        overrides["save_plots"] = False
    if args.no_json:
        overrides["save_json"] = False

    trainer = trainer_cls(load_train_config(args.config), config_path=args.config)
    if args.print_config:
        resolved_config, _ = trainer._prepare_generate_result(overrides=overrides, config_path=args.config)
        print(yaml.safe_dump(config_to_dict(resolved_config), sort_keys=False, allow_unicode=False))
    return trainer.generate_result(overrides=overrides, config_path=args.config)
