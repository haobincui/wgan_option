import sys
import unittest
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.generate_surface.surface_cpu.generate_minute_svi_params import (  # noqa: E402
    ContractMeta,
    MinuteTradeRow,
    _collect_minute_spot,
    _get_file_target_future_month_code,
    _infer_file_date_range,
    _make_expiry_dt_utc,
    _tau_years_from_trade_to_expiry,
)


class TestGenerateMinuteSviParams(unittest.TestCase):
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
        trade_ts = pd.Timestamp("2025-11-14T18:03:33.390377620Z")
        expiry_dt_utc = _make_expiry_dt_utc(date(2026, 3, 31))

        actual = _tau_years_from_trade_to_expiry(trade_ts, expiry_dt_utc)
        expected = (pd.Timestamp(expiry_dt_utc) - trade_ts).total_seconds() / 86400 / 365

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertGreater(actual, 0.37)


if __name__ == "__main__":
    unittest.main()
