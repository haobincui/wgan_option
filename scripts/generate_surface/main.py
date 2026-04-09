"""Unified CLI for daily and minute surface generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401
from scripts.generate_surface.common.config_utils import (
    load_surface_builder_root,
    resolve_config_path as _resolve_config_path,
)

from scripts.generate_surface.daily_surface import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    main as daily_surface_main,
)
from scripts.generate_surface.dispatch import ensure_cuda_available, extract_device_arg  # noqa: E402
from scripts.generate_surface.backend.surface_cpu.all import main as cpu_minute_main  # noqa: E402
from scripts.generate_surface.backend.surface_cpu.window import (  # noqa: E402
    main as cpu_window_main,
)
from scripts.generate_surface.backend.surface_cpu.excel import (  # noqa: E402
    main as cpu_excel_main,
)
from scripts.generate_surface.backend.surface_gpu.all import main as gpu_minute_main  # noqa: E402
from scripts.generate_surface.backend.surface_gpu.window import (  # noqa: E402
    main as gpu_window_main,
)
from scripts.generate_surface.backend.surface_gpu.excel import (  # noqa: E402
    main as gpu_excel_main,
)

HELP_TEXT = dedent(
    """
    Usage:
      python scripts/generate_surface/main.py daily-surface [args...]
      python scripts/generate_surface/main.py minute-svi [--device {cpu,gpu}] [--model {svi,sabr,cubic,raw}] [args...]
      python scripts/generate_surface/main.py minute-svi-window [--device {cpu,gpu}] [--model {svi,sabr,cubic,raw}] [args...]
      python scripts/generate_surface/main.py minute-svi-excel [--device {cpu,gpu}] [--model {svi,sabr,cubic,raw}] [args...]
      python scripts/generate_surface/main.py [--config CONFIG] [--device {cpu,gpu}] [--model {svi,sabr,cubic,raw}] [args...]

    Subcommands:
      daily-surface      Build daily surface tensors.
      minute-svi         Build minute surfaces for all eligible minutes.
      minute-svi-window  Build minute surfaces around target datetime windows.
      minute-svi-excel   Build minute surfaces from Excel PD/ET target timestamps.

    Notes:
      When no subcommand is provided, the CLI reads `surface_builder.job` from the config file.
      minute-* commands default to --device cpu.
      minute-* commands also support --model {svi,sabr,cubic,raw}; default is svi.
      If --device gpu is selected and CUDA is unavailable, the command exits with an error.
      Pass --help after a subcommand to see that job's detailed arguments.
    """
).strip()

MINUTE_COMMANDS = {
    "minute-svi": {"cpu": cpu_minute_main, "gpu": gpu_minute_main},
    "minute-svi-window": {"cpu": cpu_window_main, "gpu": gpu_window_main},
    "minute-svi-excel": {"cpu": cpu_excel_main, "gpu": gpu_excel_main},
}
ALL_COMMANDS = {"daily-surface", *MINUTE_COMMANDS.keys()}


def _extract_config_path(argv_list: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parsed, _ = parser.parse_known_args(argv_list)
    return _resolve_config_path(str(parsed.config))


def _load_job_from_config(argv_list: list[str]) -> str:
    config_path = _extract_config_path(argv_list)
    config_path, config_root = load_surface_builder_root(str(config_path))

    job = str(config_root.get("job", "")).strip()
    if not job:
        raise ValueError(
            f"Missing `surface_builder.job` in config file: {config_path}. "
            f"Expected one of: {sorted(ALL_COMMANDS)}"
        )
    if job not in ALL_COMMANDS:
        raise ValueError(
            f"Unsupported `surface_builder.job={job}` in config file: {config_path}. "
            f"Expected one of: {sorted(ALL_COMMANDS)}"
        )
    return job


def main(argv=None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if argv_list and argv_list[0] in {"-h", "--help"}:
        print(HELP_TEXT)
        return

    if argv_list and argv_list[0] in ALL_COMMANDS:
        subcommand = argv_list[0]
        remaining = argv_list[1:]
    else:
        subcommand = _load_job_from_config(argv_list)
        remaining = argv_list

    if subcommand == "daily-surface":
        _, forwarded_argv = extract_device_arg(remaining, default="cpu")
        daily_surface_main(forwarded_argv)
        return

    device, forwarded_argv = extract_device_arg(remaining, default="cpu")
    if device == "gpu":
        ensure_cuda_available(device)
    MINUTE_COMMANDS[subcommand][device](forwarded_argv)


if __name__ == "__main__":
    main()
