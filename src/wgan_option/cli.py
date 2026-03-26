"""CLI argument definitions for training entrypoint."""

import argparse

from wgan_option.config import DEFAULT_CONFIG_PATH


def parse_train_args():
    """Return parsed arguments for `scripts/train.py`."""
    parser = argparse.ArgumentParser(
        description="Train conditional WGAN for bond option vol-surface forecasting."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG_PATH})",
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
    return parser.parse_args()
