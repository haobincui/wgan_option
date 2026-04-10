"""CPU datetime-window surface generation entrypoint."""

from __future__ import annotations

from wgan_option.surface_generation.data_helperd.window import (  # noqa: E402
    _build_target_window_map,
    _collect_target_datetime_tokens,
    _extract_target_surfaces,
    _parse_args,
    _split_datetime_tokens,
    _to_utc_minute_ts,
    generate_surfaces_for_datetime_windows,
    run_window_job,
)
from wgan_option.surface_generation.backend.surface_cpu.all import _process_minute  # noqa: E402
from wgan_option.surface_generation.data_helperd.all import resolve_parallel_calibration_workers  # noqa: E402


def run(args):
    parallel_workers = resolve_parallel_calibration_workers(
        args,
        device="cpu",
        data_range="window",
    )
    return run_window_job(
        args,
        _process_minute,
        parallel_calibration_workers=parallel_workers,
    )


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
