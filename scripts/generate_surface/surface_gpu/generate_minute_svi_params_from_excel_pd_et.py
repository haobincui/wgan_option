"""GPU Excel-driven minute-SVI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[3]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

import scripts._path_setup  # noqa: F401
from scripts.generate_surface.common.minute_svi_excel_common import (  # noqa: E402
    _excel_time_fraction_to_hms,
    _load_target_datetimes_from_excel,
    _normalize_date_value,
    _normalize_time_value,
    _parse_args,
    _to_utc_string,
    run_excel_job,
)
from scripts.generate_surface.surface_gpu.generate_minute_svi_params import _process_minute  # noqa: E402


def run(args):
    return run_excel_job(args, _process_minute)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
