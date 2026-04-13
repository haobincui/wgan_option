"""Unified CLI for surface-error analysis entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.analyze_error.analyze_svi import main as svi_analyze_main  # noqa: E402
from scripts.analyze_error.analyze_vol import main as vol_analyze_main  # noqa: E402

HELP_TEXT = dedent(
    """
    Usage:
      python scripts/analyze_error/main.py vol [args...]
      python scripts/analyze_error/main.py svi [args...]

    Subcommands:
      vol    Analyze target-vs-generated vol surface errors.
      svi    Analyze target-vs-generated SVI-reconstructed surface errors.

    Notes:
      Pass --help after a subcommand to see that job's detailed arguments.
    """
).strip()

COMMANDS = {
    "vol": vol_analyze_main,
    "svi": svi_analyze_main,
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
