"""GPU Excel-driven surface generation entrypoint."""

from __future__ import annotations

from wgan_option.surface_generation.data_helperd.excel import (  # noqa: E402
    _excel_time_fraction_to_hms,
    _load_target_datetimes_from_excel,
    _normalize_date_value,
    _normalize_time_value,
    _parse_args,
    _to_utc_string,
    run_excel_job,
)
from wgan_option.surface_generation.backend.surface_gpu.all import _process_minute  # noqa: E402
from wgan_option.surface_generation.data_helperd.all import resolve_parallel_calibration_workers  # noqa: E402


def run(args):
    resolve_parallel_calibration_workers(
        args,
        device="gpu",
        data_range="excel",
    )
    return run_excel_job(args, _process_minute)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
