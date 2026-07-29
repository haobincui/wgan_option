"""CPU all-minute surface generation entrypoint."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from wgan_option.surface_generation.data_helperd.all import (  # noqa: E402
    PRECALIB_CSV_HEADERS,
    ContractMeta,
    MinuteOptionCandidate,
    MinuteTradeRow,
    _build_rows_for_minute,
    _collect_minute_spot,
    _finalize_minute_surface,
    _get_file_target_future_month_code,
    _infer_file_date_range,
    _log_cli_arguments,
    _make_expiry_dt_utc,
    _make_spot_cache_key,
    _option_type_name,
    _parse_args,
    _parse_data_date,
    _prepare_option_candidates,
    _resolve_config_path,
    resolve_parallel_calibration_workers,
    _tau_years_from_trade_to_expiry,
    _to_json_native,
    _to_utc_minute_string,
    _to_utc_timestamp,
    run_minute_svi_job,
)

from quantlib.calculation.analytics.models.analytical.equity.formula import (  # noqa: E402
    black_scholes_implied_vol,
)
from wgan_option.market.black76 import black76_implied_vol  # noqa: E402


def _compute_implied_vols_cpu(candidates: List[MinuteOptionCandidate]) -> List[Optional[float]]:
    implied_vols: List[Optional[float]] = []
    for candidate in candidates:
        try:
            if candidate.pricing_model == "black76":
                implied_vol = black76_implied_vol(
                    price=candidate.price,
                    futures_price=candidate.spot,
                    strike=candidate.strike,
                    tau=candidate.tau,
                    discount_factor=candidate.discount_factor,
                    option_type=candidate.meta.option_type,
                )
            else:
                implied_vol = black_scholes_implied_vol(
                        price=candidate.price,
                        strike=candidate.strike,
                        option_type=candidate.meta.option_type,
                        spot=candidate.spot,
                        tau=candidate.tau,
                        r=0.0,
                        q=0.0,
                    )
            implied_vols.append(float(implied_vol))
        except Exception:
            implied_vols.append(None)
    return implied_vols


def _process_minute(
    minute_ts: pd.Timestamp,
    rows: List[MinuteTradeRow],
    days_in_year: int,
    min_strikes_per_expiry: int,
    min_expiries_per_minute: int,
    max_precalib_iv: float,
    vol_daycount,
    calendar,
    target_future_month_code: Optional[str],
    last_spot_by_key: Dict[Tuple[str, str], float],
    results: Dict[str, Dict[str, Any]],
    stats: Dict[str, int],
    precalib_writer: Optional[csv.DictWriter] = None,
    tau_anchor_ts: Optional[pd.Timestamp] = None,
    count_stat_key: str = "total_minutes",
    surface_model: str = "svi",
    pricing_context=None,
) -> None:
    del days_in_year
    stats[count_stat_key] += 1

    corrected_black76 = (
        pricing_context is not None
        and pricing_context.pricing_model == "black76"
    )
    if corrected_black76:
        minute_spot = {}
        option_rows = [
            row for row in rows if row.meta.contract_type == "option"
        ]
        stats["future_rows"] += sum(
            row.meta.contract_type == "future" for row in rows
        )
        stats["option_rows"] += len(option_rows)
    else:
        minute_spot, option_rows = _collect_minute_spot(
            rows=rows,
            target_future_month_code=target_future_month_code,
            last_spot_by_key=last_spot_by_key,
            stats=stats,
        )
    if not option_rows:
        return

    rejected_audit_rows: List[Dict[str, Any]] = []
    valuation_date, candidates = _prepare_option_candidates(
        minute_ts=minute_ts,
        option_rows=option_rows,
        minute_spot=minute_spot,
        last_spot_by_key=last_spot_by_key,
        target_future_month_code=target_future_month_code,
        vol_daycount=vol_daycount,
        calendar=calendar,
        stats=stats,
        tau_anchor_ts=tau_anchor_ts,
        all_rows=rows,
        pricing_context=pricing_context,
        rejected_audit_rows=rejected_audit_rows,
    )
    implied_vols = _compute_implied_vols_cpu(candidates)
    _finalize_minute_surface(
        minute_ts=minute_ts,
        valuation_date=valuation_date,
        candidates=candidates,
        implied_vols=implied_vols,
        min_strikes_per_expiry=min_strikes_per_expiry,
        min_expiries_per_minute=min_expiries_per_minute,
        max_precalib_iv=max_precalib_iv,
        vol_daycount=vol_daycount,
        results=results,
        stats=stats,
        precalib_writer=precalib_writer,
        surface_model=surface_model,
        iv_aggregation_mode=(
            pricing_context.iv_aggregation_mode
            if pricing_context is not None
            else "volume_weighted_mean"
        ),
        rejected_audit_rows=rejected_audit_rows,
        option_filter_mode=(
            pricing_context.option_filter_mode
            if pricing_context is not None
            else "none"
        ),
        max_itm_moneyness_distance=(
            pricing_context.max_itm_moneyness_distance
            if pricing_context is not None
            else 0.05
        ),
    )


def run(args):
    resolve_parallel_calibration_workers(
        args,
        device="cpu",
        data_range="all",
    )
    return run_minute_svi_job(args, _process_minute)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
