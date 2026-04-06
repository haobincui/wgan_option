"""Unified CLI for merged xlsx training entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.train.train_svi import main as svi_train_main  # noqa: E402
from scripts.train.train_vol import main as vol_train_main  # noqa: E402

HELP_TEXT = dedent(
    """
    Usage:
      python scripts/train/main.py vol-xlsx [args...]
      python scripts/train/main.py svi-xlsx [args...]

    Subcommands:
      vol-xlsx   Train WGAN-GP on merged vol-surface xlsx data.
      svi-xlsx   Train supervised SVI regressor on merged SVI xlsx data.

    Notes:
      Pass --help after a subcommand to see that job's detailed arguments.
      Aliases `train-vol` and `train-svi` are also accepted.
    """
).strip()

COMMANDS = {
    "vol-xlsx": vol_train_main,
    "train-vol": vol_train_main,
    "svi-xlsx": svi_train_main,
    "train-svi": svi_train_main,
}


def main(argv=None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if not argv_list or argv_list[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return

    subcommand = argv_list[0]
    if subcommand not in COMMANDS:
        raise SystemExit(f"Unsupported subcommand: {subcommand}\n\n{HELP_TEXT}")

    COMMANDS[subcommand](argv_list[1:])


if __name__ == "__main__":
    main()
