"""GPU datetime-window minute-SVI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT_DIR = Path(__file__).resolve().parents[3]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from scripts.generate_surface.common.minute_svi_window_common import (  # noqa: E402
        _build_target_window_map,
        _collect_target_datetime_tokens,
        _extract_target_surfaces,
        _parse_args,
        _split_datetime_tokens,
        _to_utc_minute_ts,
        generate_surfaces_for_datetime_windows,
        run_window_job,
    )
    from scripts.generate_surface.surface_gpu.generate_minute_svi_params import _process_minute  # noqa: E402
else:
    from ..common.minute_svi_window_common import (  # noqa: E402
        _build_target_window_map,
        _collect_target_datetime_tokens,
        _extract_target_surfaces,
        _parse_args,
        _split_datetime_tokens,
        _to_utc_minute_ts,
        generate_surfaces_for_datetime_windows,
        run_window_job,
    )
    from .generate_minute_svi_params import _process_minute  # noqa: E402


def run(args):
    return run_window_job(args, _process_minute)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
