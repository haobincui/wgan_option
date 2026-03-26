"""CLI entrypoint for WGAN training."""

import yaml

from wgan_option.config import (
    config_to_dict,
    load_config,
    parse_cli_overrides,
)
from wgan_option.cli import parse_train_args
from wgan_option.train import WGANTrainer


def main():
    """Parse CLI args, resolve config, and launch training/dry-run."""
    args = parse_train_args()
    overrides = parse_cli_overrides(args.set)
    config = load_config(config_path=args.config, overrides=overrides)

    if args.print_config:
        print(yaml.safe_dump(config_to_dict(config), sort_keys=False, allow_unicode=False))

    trainer = WGANTrainer(config)
    if args.dry_run:
        trainer.dry_run()
    else:
        trainer.start_train()


if __name__ == "__main__":
    main()
