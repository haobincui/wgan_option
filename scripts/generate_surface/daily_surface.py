"""Daily surface generation job used by the unified surface CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401

from scripts.generate_surface.common.config_utils import (  # noqa: E402
    build_config_scope,
    load_surface_builder_section,
    parse_bool_from_config,
    resolve_config_path,
    resolve_config_variables,
)
from quantlib.vol_surface.builder import OptionSurfaceBuilder, OptionSurfaceBuilderConfig  # noqa: E402

DEFAULT_CONFIG_PATH = "configs/surface_builder/default.yaml"
SUPPORTED_DAILY_SURFACE_CONFIG_KEYS = {
    "input_glob",
    "output_dir",
    "strike_bins",
    "maturity_bins",
    "moneyness_min",
    "moneyness_max",
    "maturity_min_days",
    "maturity_max_days",
    "save_daily_csv",
    "npz_name",
}


_resolve_config_path = resolve_config_path
_parse_bool_from_config = parse_bool_from_config


def _load_daily_surface_config(config_path_value: str) -> Dict[str, Any]:
    config_path, config_root, defaults = load_surface_builder_section(
        config_path_value,
        section_key="daily_surface",
        supported_keys=SUPPORTED_DAILY_SURFACE_CONFIG_KEYS,
        merge_root_supported_defaults=True,
    )

    shared_glob = config_root.get("option_data_glob")
    if "input_glob" not in defaults and isinstance(shared_glob, str):
        defaults["input_glob"] = shared_glob

    return resolve_config_variables(defaults, extra_scope=build_config_scope(config_root))


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="YAML config path for surface_builder settings.",
    )
    pre_args, _ = pre_parser.parse_known_args(argv_list)

    if "-h" in argv_list or "--help" in argv_list:
        config_defaults: Dict[str, Any] = {}
    else:
        config_defaults = _load_daily_surface_config(pre_args.config)

    default_save_daily_csv = _parse_bool_from_config(
        config_defaults.get("save_daily_csv", True),
        key="save_daily_csv",
    )

    parser = argparse.ArgumentParser(
        description="Generate daily volatility-like surfaces from raw option data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="YAML config path with a required `surface_builder.daily_surface` mapping.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=str(config_defaults.get("input_glob", "data/raw/option_data/*.csv.gz")),
        help="Glob pattern for raw option trade files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config_defaults.get("output_dir", "outputs/vol_surface")),
        help="Output directory for generated surface files.",
    )
    parser.add_argument(
        "--strike-bins",
        type=int,
        default=int(config_defaults.get("strike_bins", 16)),
        help="Number of strike buckets in the generated daily grid.",
    )
    parser.add_argument(
        "--maturity-bins",
        type=int,
        default=int(config_defaults.get("maturity_bins", 16)),
        help="Number of maturity buckets in the generated daily grid.",
    )
    parser.add_argument(
        "--moneyness-min",
        type=float,
        default=float(config_defaults.get("moneyness_min", 0.7)),
        help="Minimum moneyness included in the generated grid.",
    )
    parser.add_argument(
        "--moneyness-max",
        type=float,
        default=float(config_defaults.get("moneyness_max", 1.3)),
        help="Maximum moneyness included in the generated grid.",
    )
    parser.add_argument(
        "--maturity-min-days",
        type=int,
        default=int(config_defaults.get("maturity_min_days", 7)),
        help="Minimum maturity days included in the generated grid.",
    )
    parser.add_argument(
        "--maturity-max-days",
        type=int,
        default=int(config_defaults.get("maturity_max_days", 365)),
        help="Maximum maturity days included in the generated grid.",
    )
    parser.add_argument(
        "--save-daily-csv",
        dest="save_daily_csv",
        action="store_true",
        default=default_save_daily_csv,
        help="If set, save per-day CSV snapshots alongside the compressed NPZ output.",
    )
    parser.add_argument(
        "--no-daily-csv",
        dest="save_daily_csv",
        action="store_false",
        help="Disable saving per-day CSV snapshots and only write the NPZ output.",
    )
    parser.add_argument(
        "--npz-name",
        type=str,
        default=str(config_defaults.get("npz_name", "vol_surface_stack.npz")),
        help="Filename for the compressed surface stack NPZ.",
    )

    args = parser.parse_args(argv_list)
    args.config = str(_resolve_config_path(args.config))
    return args


def run(args) -> Path:
    config = OptionSurfaceBuilderConfig(
        option_data_glob=args.input_glob,
        strike_bins=args.strike_bins,
        maturity_bins=args.maturity_bins,
        moneyness_min=args.moneyness_min,
        moneyness_max=args.moneyness_max,
        maturity_min_days=args.maturity_min_days,
        maturity_max_days=args.maturity_max_days,
    )
    builder = OptionSurfaceBuilder(config)
    return builder.save(
        output_dir=args.output_dir,
        save_daily_csv=bool(args.save_daily_csv),
        npz_name=str(args.npz_name),
    )


def main(argv=None) -> None:
    args = _parse_args(argv)
    out_file = run(args)
    print(f"Surface file generated: {out_file}")


if __name__ == "__main__":
    main()
