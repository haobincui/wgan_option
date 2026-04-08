"""Unified CLI for result-generation entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.generate_result.generate_svi import main as svi_generate_main  # noqa: E402
from scripts.generate_result.generate_vol import main as vol_generate_main  # noqa: E402
from scripts.generate_result.plot_surface import main as plot_surface_main  # noqa: E402

HELP_TEXT = dedent(
    """
    Usage:
      python scripts/generate_result/main.py vol [args...]
      python scripts/generate_result/main.py svi [args...]
      python scripts/generate_result/main.py plot [args...]

    Subcommands:
      vol    Generate future vol surfaces from a trained WGAN checkpoint.
      svi    Generate future SVI parameters, reconstruct surfaces, and compare them.
      plot   Render a saved payload JSON into a comparison PNG.

    Notes:
      Pass --help after a subcommand to see that job's detailed arguments.
    """
).strip()

COMMANDS = {
    "vol": vol_generate_main,
    "svi": svi_generate_main,
    "plot": plot_surface_main,
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
