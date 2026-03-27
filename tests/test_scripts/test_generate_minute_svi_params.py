import csv
import io
import sys
import unittest
from collections import defaultdict
from datetime import date, time as dt_time, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantlib.calculation.analytics.position.instruments.features import OptionType  # noqa: E402
from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402
from scripts.generate_surface.common.minute_svi_common import DEFAULT_EXPIRATION_TIME_UTC  # noqa: E402
from scripts.generate_surface.surface_cpu.generate_minute_svi_params import (  # noqa: E402
    ContractMeta,
    MinuteOptionCandidate,
    MinuteTradeRow,
    PRECALIB_CSV_HEADERS,
    _build_rows_for_minute,
    _collect_minute_spot,
    _finalize_minute_surface,
    _get_file_target_future_month_code,
    _infer_file_date_range,
    _make_expiry_dt_utc,
    _tau_years_from_trade_to_expiry,
)


class TestGenerateMinuteSviParams(unittest.TestCase):
    def _vol_daycount(self) -> DayCountBusN:
        calendar = usd_calendar()
        return DayCountBusN(name="BUS250USD", calendar=calendar, days_in_year=250)

    def test_infer_file_date_range_and_target_month(self):
        path = "data/raw/option_data/0#TY+/0#TY+_2025-11-14_2025-11-15.csv.gz"
        self.assertEqual(
            _infer_file_date_range(path),
            (date(2025, 11, 14), date(2025, 11, 15)),
        )
        self.assertEqual(_get_file_target_future_month_code(path), "Z")

    def test_collect_minute_spot_uses_target_future_month_only(self):
        minute = pd.Timestamp("2025-11-14T18:03:00Z")
        rows = [
            MinuteTradeRow(
                trade_ts=minute,
                meta=ContractMeta(
                    contract_type="future",
                    underlying="TY",
                    maturity_month_code="Z",
                ),
                price=112.60,
                volume=10,
            ),
            MinuteTradeRow(
                trade_ts=minute,
                meta=ContractMeta(
                    contract_type="future",
                    underlying="TY",
                    maturity_month_code="H",
                ),
                price=112.50,
                volume=10,
            ),
            MinuteTradeRow(
                trade_ts=minute,
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=110.0,
                ),
                price=0.328125,
                volume=5,
            ),
        ]

        stats = defaultdict(int)
        last_spot_by_key = {}
        minute_spot, option_rows = _collect_minute_spot(
            rows=rows,
            target_future_month_code="Z",
            last_spot_by_key=last_spot_by_key,
            stats=stats,
        )

        self.assertEqual(minute_spot, {("TY", "Z"): 112.60})
        self.assertEqual(last_spot_by_key, {("TY", "Z"): 112.60})
        self.assertEqual(len(option_rows), 1)
        self.assertEqual(stats["future_rows"], 2)
        self.assertEqual(stats["skip_future_non_target_month"], 1)
        self.assertEqual(stats["option_rows"], 1)

    def test_tau_years_from_trade_to_expiry_uses_exact_timestamp(self):
        vol_daycount = self._vol_daycount()
        trade_ts = pd.Timestamp("2025-11-14T18:03:33.390377620Z")
        expiration_time_utc = dt_time(20, 0, 0, tzinfo=timezone.utc)
        expiry_dt_utc = _make_expiry_dt_utc(date(2026, 3, 31), expiration_time_utc)

        actual = _tau_years_from_trade_to_expiry(trade_ts, expiry_dt_utc, vol_daycount)
        base_tau = vol_daycount(trade_ts.date(), expiry_dt_utc.date())
        quote_fraction = (
            trade_ts - trade_ts.normalize()
        ).total_seconds() / 86400.0
        expiry_ts = pd.Timestamp(expiry_dt_utc)
        expiry_fraction = (
            expiry_ts - expiry_ts.normalize()
        ).total_seconds() / 86400.0
        expected = base_tau + (expiry_fraction - quote_fraction) / vol_daycount.days_in_year

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertGreater(actual, 0.36)

    def test_tau_years_from_trade_to_expiry_supports_same_day_positive_fraction(self):
        vol_daycount = self._vol_daycount()
        trade_ts = pd.Timestamp("2025-11-14T09:15:00Z")
        expiry_dt_utc = pd.Timestamp("2025-11-14T20:00:00Z").to_pydatetime(warn=False)

        actual = _tau_years_from_trade_to_expiry(trade_ts, expiry_dt_utc, vol_daycount)
        expected = ((20 * 3600) - (9 * 3600 + 15 * 60)) / 86400.0 / vol_daycount.days_in_year

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertGreater(actual, 0.0)

    def test_tau_years_from_trade_to_expiry_counts_business_days_and_intraday_fraction(self):
        vol_daycount = self._vol_daycount()
        trade_ts = pd.Timestamp("2025-11-14T18:03:33Z")
        expiry_dt_utc = pd.Timestamp("2025-11-17T20:00:00Z").to_pydatetime(warn=False)

        actual = _tau_years_from_trade_to_expiry(trade_ts, expiry_dt_utc, vol_daycount)
        expected = 1.0 / vol_daycount.days_in_year + (
            ((20 * 3600) - (18 * 3600 + 3 * 60 + 33)) / 86400.0 / vol_daycount.days_in_year
        )

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(actual, 1.0 / 250.0 + 6987.0 / 86400.0 / 250.0, places=12)

    def test_build_rows_for_minute_uses_configured_expiry_time_utc(self):
        minute_df = pd.DataFrame(
            {
                "ric": ["TY110O26"],
                "trade_dt": [pd.Timestamp("2025-11-14T18:03:00Z")],
                "price": [0.328125],
                "volume": [5.0],
            }
        )
        stats = defaultdict(int)
        rows = _build_rows_for_minute(
            minute_df=minute_df,
            expiry_inference_date=date(2025, 11, 14),
            expiration_time_utc=dt_time.fromisoformat(DEFAULT_EXPIRATION_TIME_UTC).replace(
                tzinfo=timezone.utc
            ),
            calendar=usd_calendar(),
            contract_cache={},
            stats=stats,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].meta.contract_id, "TY110O26")
        self.assertEqual(rows[0].meta.expiry_date, date(2026, 3, 31))
        self.assertEqual(
            rows[0].meta.expiry_dt_utc,
            pd.Timestamp("2026-03-31T20:00:00Z").to_pydatetime(warn=False),
        )

    def test_finalize_minute_surface_writes_price_and_spot_to_precalib_csv(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        expiry_date = date(2026, 3, 31)
        candidates = [
            MinuteOptionCandidate(
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=110.0,
                    option_type=OptionType.PUT,
                    expiry_date=expiry_date,
                    contract_id="TY110O26",
                ),
                price=0.25,
                weight=1.0,
                strike=110.0,
                spot=112.5934,
                tau=0.376,
                business_days=94,
            ),
            MinuteOptionCandidate(
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=110.0,
                    option_type=OptionType.PUT,
                    expiry_date=expiry_date,
                    contract_id="TY110O26",
                ),
                price=0.50,
                weight=3.0,
                strike=110.0,
                spot=112.5934,
                tau=0.376,
                business_days=94,
            ),
        ]

        precalib_buffer = io.StringIO()
        precalib_writer = csv.DictWriter(precalib_buffer, fieldnames=PRECALIB_CSV_HEADERS)
        precalib_writer.writeheader()

        calendar = usd_calendar()
        vol_daycount = DayCountBusN(name="BUS250USD", calendar=calendar, days_in_year=250)
        stats = defaultdict(int)

        _finalize_minute_surface(
            minute_ts=minute_ts,
            valuation_date=minute_ts.date(),
            candidates=candidates,
            implied_vols=[0.20, 0.30],
            min_strikes_per_expiry=1,
            min_expiries_per_minute=1,
            vol_daycount=vol_daycount,
            results={},
            stats=stats,
            precalib_writer=precalib_writer,
        )

        rows = list(csv.DictReader(io.StringIO(precalib_buffer.getvalue())))
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertIn("contract_id", row)
        self.assertIn("price", row)
        self.assertIn("spot", row)
        self.assertEqual(row["contract_id"], "TY110O26")
        self.assertAlmostEqual(float(row["price"]), 0.4375, places=10)
        self.assertAlmostEqual(float(row["spot"]), 112.5934, places=10)
        self.assertAlmostEqual(float(row["weight_sum"]), 4.0, places=10)
        self.assertEqual(stats["precalib_rows_written"], 1)
        self.assertEqual(stats["precalib_minutes_written"], 1)


if __name__ == "__main__":
    unittest.main()
