"""Unified CLI for minute surface generation."""

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
from scripts.generate_surface.dispatch import ensure_cuda_available, extract_device_arg  # noqa: E402
from scripts.generate_surface.data_helperd.all import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_RANGE,
    SUPPORTED_DATA_RANGES,
)
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
      python scripts/generate_surface/main.py generate_surface [--device {cpu,gpu}] [--model {svi,sabr,cubic,raw}] [--data_range {all,window,excel}] [args...]
      python scripts/generate_surface/main.py [--config CONFIG] [--device {cpu,gpu}] [--model {svi,sabr,cubic,raw}] [--data_range {all,window,excel}] [args...]

    Subcommands:
      generate_surface   Build minute surfaces for the requested model + data_range.

    Notes:
      When no subcommand is provided, the CLI reads `surface_builder.job` from the config file.
      generate_surface defaults to --device cpu.
      generate_surface also supports --model {svi,sabr,cubic,raw} and --data_range {all,window,excel}.
      If --device gpu is selected and CUDA is unavailable, the command exits with an error.
      Pass --help after a subcommand to see that job's detailed arguments.
    """
).strip()

MINUTE_COMMANDS = {
    "all": {"cpu": cpu_minute_main, "gpu": gpu_minute_main},
    "window": {"cpu": cpu_window_main, "gpu": gpu_window_main},
    "excel": {"cpu": cpu_excel_main, "gpu": gpu_excel_main},
}
ALL_COMMANDS = {"generate_surface"}


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


def _extract_data_range(argv_list: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data_range", "--data-range", dest="data_range", type=str, default=None)
    parsed, _ = parser.parse_known_args(argv_list)
    if parsed.data_range:
        return str(parsed.data_range).strip().lower()

    config_path = _resolve_config_path(str(parsed.config))
    _, config_root = load_surface_builder_root(str(config_path))
    section = config_root.get("generate_surface")
    if not isinstance(section, dict):
        raise ValueError(
            f"Missing `surface_builder.generate_surface` mapping in config file: {config_path}"
        )
    return str(section.get("data_range", DEFAULT_DATA_RANGE)).strip().lower()


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

    device, forwarded_argv = extract_device_arg(remaining, default="cpu")
    if device == "gpu":
        ensure_cuda_available(device)
    data_range = _extract_data_range(remaining)
    if data_range not in SUPPORTED_DATA_RANGES:
        raise ValueError(
            f"Unsupported data_range `{data_range}`. Expected one of: {sorted(SUPPORTED_DATA_RANGES)}"
        )
    MINUTE_COMMANDS[data_range][device](forwarded_argv)


if __name__ == "__main__":
    main()
