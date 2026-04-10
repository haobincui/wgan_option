import csv
import io
import sys
import unittest
from collections import defaultdict
from datetime import date, time as dt_time, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantlib.calculation.analytics.position.instruments.features import OptionType  # noqa: E402
from quantlib.calculation.analytics.models.analytical.equity.formula import (  # noqa: E402
    black_scholes_implied_vol,
)
from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402
from scripts.generate_surface.data_helperd.all import (  # noqa: E402
    DEFAULT_EXPIRATION_TIME_UTC,
    DEFAULT_MAX_PRECALIB_IV,
    _get_ty_option_underlying_future_month_code,
    _prepare_option_candidates,
)
from scripts.generate_surface.backend.surface_cpu.all import (  # noqa: E402
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
        return DayCountBusN("BUS250USD", calendar, 250)

    def _make_candidate(
        self,
        *,
        contract_id: str,
        strike: float,
        option_type,
        price: float,
        weight: float,
        spot: float = 112.5934,
        trade_ts: str = "2025-11-14T18:03:00Z",
        business_days: int = 94,
    ) -> MinuteOptionCandidate:
        expiry_date = date(2026, 3, 31)
        return MinuteOptionCandidate(
            meta=ContractMeta(
                contract_type="option",
                underlying="TY",
                strike=strike,
                option_type=option_type,
                expiry_date=expiry_date,
                contract_id=contract_id,
            ),
            price=price,
            weight=weight,
            strike=strike,
            spot=spot,
            tau=0.376,
            business_days=business_days,
            trade_ts=pd.Timestamp(trade_ts),
        )

    def test_infer_file_date_range_and_target_month(self):
        path = "data/raw/option_data/0#TY+/0#TY+_2025-11-14_2025-11-15.csv.gz"
        self.assertEqual(
            _infer_file_date_range(path),
            (date(2025, 11, 14), date(2025, 11, 15)),
        )
        self.assertEqual(_get_file_target_future_month_code(path), "Z")

    def test_ty_option_underlying_future_month_mapping_covers_all_months(self):
        expected = {
            "A": "H",
            "M": "H",
            "B": "H",
            "N": "H",
            "C": "H",
            "O": "H",
            "D": "M",
            "P": "M",
            "E": "M",
            "Q": "M",
            "F": "M",
            "R": "M",
            "G": "U",
            "S": "U",
            "H": "U",
            "T": "U",
            "I": "U",
            "U": "U",
            "J": "Z",
            "V": "Z",
            "K": "Z",
            "W": "Z",
            "L": "Z",
            "X": "Z",
        }

        for option_month_code, future_month_code in expected.items():
            with self.subTest(option_month_code=option_month_code):
                self.assertEqual(
                    _get_ty_option_underlying_future_month_code(option_month_code=option_month_code),
                    future_month_code,
                )

        self.assertEqual(
            _get_ty_option_underlying_future_month_code(expiry_date=date(2022, 6, 30)),
            "M",
        )

    def test_collect_minute_spot_keeps_non_ty_fallback_filter(self):
        minute = pd.Timestamp("2025-11-14T18:03:00Z")
        rows = [
            MinuteTradeRow(
                trade_ts=minute,
                meta=ContractMeta(
                    contract_type="future",
                    underlying="FV",
                    maturity_month_code="Z",
                ),
                price=112.60,
                volume=10,
            ),
            MinuteTradeRow(
                trade_ts=minute,
                meta=ContractMeta(
                    contract_type="future",
                    underlying="FV",
                    maturity_month_code="H",
                ),
                price=112.50,
                volume=10,
            ),
            MinuteTradeRow(
                trade_ts=minute,
                meta=ContractMeta(
                    contract_type="option",
                    underlying="FV",
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

        self.assertEqual(minute_spot, {("FV", "Z"): 112.60})
        self.assertEqual(last_spot_by_key, {("FV", "Z"): 112.60})
        self.assertEqual(len(option_rows), 1)
        self.assertEqual(stats["future_rows"], 2)
        self.assertEqual(stats["skip_future_non_target_month"], 1)
        self.assertEqual(stats["option_rows"], 1)

    def test_prepare_option_candidates_uses_ty_option_expiry_month_mapping(self):
        minute_ts = pd.Timestamp("2022-02-08T15:02:00Z")
        option_rows = [
            MinuteTradeRow(
                trade_ts=pd.Timestamp("2022-02-08T15:02:19.094316781Z"),
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    maturity_month_code="F",
                    strike=119.5,
                    option_type=OptionType.CALL,
                    expiry_date=date(2022, 6, 30),
                    expiry_dt_utc=pd.Timestamp("2022-06-30T20:00:00Z").to_pydatetime(warn=False),
                    contract_id="TY1195F2",
                ),
                price=6.984375,
                volume=1.0,
            )
        ]

        _, candidates = _prepare_option_candidates(
            minute_ts=minute_ts,
            option_rows=option_rows,
            minute_spot={
                ("TY", "H"): 126.52317731522707,
                ("TY", "M"): 126.390625,
            },
            last_spot_by_key={},
            target_future_month_code="H",
            vol_daycount=self._vol_daycount(),
            calendar=usd_calendar(),
            stats=defaultdict(int),
        )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].spot, 126.390625, places=12)

    def test_ty1195f2_uses_tym2_even_when_file_fallback_month_is_h(self):
        minute_ts = pd.Timestamp("2022-02-08T15:02:00Z")
        rows = [
            MinuteTradeRow(
                trade_ts=minute_ts,
                meta=ContractMeta(
                    contract_type="future",
                    underlying="TY",
                    maturity_month_code="H",
                    contract_id="TYH2",
                ),
                price=126.52317731522707,
                volume=10.0,
            ),
            MinuteTradeRow(
                trade_ts=minute_ts,
                meta=ContractMeta(
                    contract_type="future",
                    underlying="TY",
                    maturity_month_code="M",
                    contract_id="TYM2",
                ),
                price=126.390625,
                volume=13.0,
            ),
            MinuteTradeRow(
                trade_ts=pd.Timestamp("2022-02-08T15:02:19.094316781Z"),
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    maturity_month_code="F",
                    strike=119.5,
                    option_type=OptionType.CALL,
                    expiry_date=date(2022, 6, 30),
                    expiry_dt_utc=pd.Timestamp("2022-06-30T20:00:00Z").to_pydatetime(warn=False),
                    contract_id="TY1195F2",
                ),
                price=6.984375,
                volume=1.0,
            ),
        ]

        minute_spot, option_rows = _collect_minute_spot(
            rows=rows,
            target_future_month_code="H",
            last_spot_by_key={},
            stats=defaultdict(int),
        )
        _, candidates = _prepare_option_candidates(
            minute_ts=minute_ts,
            option_rows=option_rows,
            minute_spot=minute_spot,
            last_spot_by_key={},
            target_future_month_code="H",
            vol_daycount=self._vol_daycount(),
            calendar=usd_calendar(),
            stats=defaultdict(int),
        )

        self.assertEqual(minute_spot[("TY", "H")], 126.52317731522707)
        self.assertEqual(minute_spot[("TY", "M")], 126.390625)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].spot, 126.390625, places=12)

    def test_ty1195f2_with_corrected_underlying_future_has_finite_implied_vol(self):
        implied_vol = black_scholes_implied_vol(
            price=6.984375,
            strike=119.5,
            option_type=OptionType.CALL,
            spot=126.390625,
            tau=99.0 / 250.0,
            r=0.0,
            q=0.0,
        )

        self.assertTrue(implied_vol > 0.0)
        self.assertLess(implied_vol, 3.0)
        self.assertAlmostEqual(implied_vol, 0.054929327695134575, places=12)

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

    def test_finalize_minute_surface_writes_raw_trade_rows_to_precalib_csv_and_aggregates_for_calibration(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.50,
                weight=3.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]

        precalib_buffer = io.StringIO()
        precalib_writer = csv.DictWriter(precalib_buffer, fieldnames=PRECALIB_CSV_HEADERS)
        precalib_writer.writeheader()

        calendar = usd_calendar()
        vol_daycount = DayCountBusN("BUS250USD", calendar, 250)
        stats = defaultdict(int)
        results = {}

        with patch("scripts.generate_surface.model.svi.SviCalibrationQuasiExplicit") as calibration_cls:
            calibration_cls.return_value.params = {
                "a": [0.01],
                "b": [0.02],
                "rho": [0.0],
                "m": [0.0],
                "sigma": [0.1],
                "business_days": [94],
            }

            _finalize_minute_surface(
                minute_ts=minute_ts,
                valuation_date=minute_ts.date(),
                candidates=candidates,
                implied_vols=[0.20, 0.30],
                min_strikes_per_expiry=1,
                min_expiries_per_minute=1,
                max_precalib_iv=DEFAULT_MAX_PRECALIB_IV,
                vol_daycount=vol_daycount,
                results=results,
                stats=stats,
                precalib_writer=precalib_writer,
            )

        rows = list(csv.DictReader(io.StringIO(precalib_buffer.getvalue())))
        self.assertEqual(len(rows), 2)

        self.assertNotIn("snapshot_time_utc", rows[0])
        self.assertNotIn("weight_sum", rows[0])
        self.assertIn("trade_datetime_utc", rows[0])
        self.assertIn("calibration_datetime_utc", rows[0])
        self.assertIn("weight", rows[0])

        self.assertEqual(
            [row["trade_datetime_utc"] for row in rows],
            ["2025-11-14T18:03:11Z", "2025-11-14T18:03:47Z"],
        )
        self.assertEqual(
            [row["calibration_datetime_utc"] for row in rows],
            ["2025-11-14T18:03:00Z", "2025-11-14T18:03:00Z"],
        )
        self.assertEqual([row["contract_id"] for row in rows], ["TY110O26", "TY110O26"])
        self.assertAlmostEqual(float(rows[0]["price"]), 0.25, places=10)
        self.assertAlmostEqual(float(rows[1]["price"]), 0.50, places=10)
        self.assertAlmostEqual(float(rows[0]["spot"]), 112.5934, places=10)
        self.assertEqual([row["passes_precalib_filter"] for row in rows], ["true", "true"])
        self.assertEqual([row["filter_reason"] for row in rows], ["", ""])
        self.assertEqual([float(row["weight"]) for row in rows], [1.0, 3.0])
        self.assertIn("2025-11-14T18:03:00Z", results)
        self.assertEqual(results["2025-11-14T18:03:00Z"]["surface_model"], "svi")
        self.assertEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"],
            calibration_cls.return_value.params,
        )
        self.assertEqual(len(calibration_cls.call_args.kwargs["vols"]), 1)
        self.assertEqual(len(calibration_cls.call_args.kwargs["vols"][0]), 1)
        self.assertAlmostEqual(calibration_cls.call_args.kwargs["vols"][0][0], 0.275, places=12)
        self.assertEqual(len(calibration_cls.call_args.kwargs["percent_strikes"]), 1)
        self.assertEqual(len(calibration_cls.call_args.kwargs["percent_strikes"][0]), 1)
        self.assertAlmostEqual(
            calibration_cls.call_args.kwargs["percent_strikes"][0][0],
            110.0 / 112.5934,
            places=12,
        )
        self.assertEqual(stats["precalib_rows_written"], 2)
        self.assertEqual(stats["precalib_minutes_written"], 1)

    def test_finalize_minute_surface_threads_calibration_diagnostics_into_stats(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.50,
                weight=3.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]

        stats = defaultdict(int)
        results = {}

        def _fake_calibration(**kwargs):
            kwargs["stats"]["qls_boundary_retry_slices"] += 1
            kwargs["stats"]["qls_fallback_attempt_slices"] += 1
            kwargs["stats"]["qls_fallback_success_slices"] += 1

            class _CalibrationResult:
                params = {
                    "a": [0.01],
                    "b": [0.02],
                    "rho": [0.1],
                    "m": [0.0],
                    "sigma": [0.2],
                    "business_days": [94],
                }

            return _CalibrationResult()

        with patch(
            "scripts.generate_surface.model.svi.SviCalibrationQuasiExplicit",
            side_effect=_fake_calibration,
        ) as calibration_cls:
            _finalize_minute_surface(
                minute_ts=minute_ts,
                valuation_date=minute_ts.date(),
                candidates=candidates,
                implied_vols=[0.20, 0.30],
                min_strikes_per_expiry=1,
                min_expiries_per_minute=1,
                max_precalib_iv=DEFAULT_MAX_PRECALIB_IV,
                vol_daycount=self._vol_daycount(),
                results=results,
                stats=stats,
                precalib_writer=None,
            )

        self.assertIn("2025-11-14T18:03:00Z", results)
        self.assertEqual(stats["calibrated_minutes"], 1)
        self.assertEqual(stats["qls_boundary_retry_slices"], 1)
        self.assertEqual(stats["qls_fallback_attempt_slices"], 1)
        self.assertEqual(stats["qls_fallback_success_slices"], 1)
        self.assertIs(calibration_cls.call_args.kwargs["stats"], stats)
        self.assertEqual(results["2025-11-14T18:03:00Z"]["surface_model"], "svi")

    def test_finalize_minute_surface_flags_filtered_points_and_skips_calibration_if_strikes_drop_below_minimum(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY115C26",
                strike=115.0,
                option_type=OptionType.CALL,
                price=0.50,
                weight=1.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]

        precalib_buffer = io.StringIO()
        precalib_writer = csv.DictWriter(precalib_buffer, fieldnames=PRECALIB_CSV_HEADERS)
        precalib_writer.writeheader()

        stats = defaultdict(int)
        results = {}

        _finalize_minute_surface(
            minute_ts=minute_ts,
            valuation_date=minute_ts.date(),
            candidates=candidates,
            implied_vols=[0.20, 4.00],
            min_strikes_per_expiry=2,
            min_expiries_per_minute=1,
            max_precalib_iv=DEFAULT_MAX_PRECALIB_IV,
            vol_daycount=self._vol_daycount(),
            results=results,
            stats=stats,
            precalib_writer=precalib_writer,
        )

        rows = list(csv.DictReader(io.StringIO(precalib_buffer.getvalue())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(results, {})
        self.assertEqual(stats["skip_precalib_iv_above_cap"], 1)
        self.assertEqual(stats["skip_sample_insufficient"], 1)
        self.assertEqual(stats["precalib_rows_written"], 2)
        self.assertEqual(stats["precalib_minutes_written"], 1)

        rows_by_strike = {float(row["strike"]): row for row in rows}
        self.assertEqual(rows_by_strike[110.0]["trade_datetime_utc"], "2025-11-14T18:03:11Z")
        self.assertEqual(rows_by_strike[110.0]["calibration_datetime_utc"], "2025-11-14T18:03:00Z")
        self.assertEqual(rows_by_strike[110.0]["passes_precalib_filter"], "true")
        self.assertEqual(rows_by_strike[110.0]["filter_reason"], "")
        self.assertEqual(rows_by_strike[115.0]["passes_precalib_filter"], "false")
        self.assertEqual(rows_by_strike[115.0]["filter_reason"], "implied_vol_above_cap")

    def test_finalize_minute_surface_disables_iv_cap_when_threshold_is_nonpositive(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY115C26",
                strike=115.0,
                option_type=OptionType.CALL,
                price=0.50,
                weight=1.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]

        precalib_buffer = io.StringIO()
        precalib_writer = csv.DictWriter(precalib_buffer, fieldnames=PRECALIB_CSV_HEADERS)
        precalib_writer.writeheader()

        stats = defaultdict(int)
        results = {}

        with patch("scripts.generate_surface.model.svi.SviCalibrationQuasiExplicit") as calibration_cls:
            calibration_cls.return_value.params = {
                "a": [0.01],
                "b": [0.02],
                "rho": [0.0],
                "m": [0.0],
                "sigma": [0.1],
                "business_days": [94],
            }

            _finalize_minute_surface(
                minute_ts=minute_ts,
                valuation_date=minute_ts.date(),
                candidates=candidates,
                implied_vols=[0.20, 4.00],
                min_strikes_per_expiry=2,
                min_expiries_per_minute=1,
                max_precalib_iv=0.0,
                vol_daycount=self._vol_daycount(),
                results=results,
                stats=stats,
                precalib_writer=precalib_writer,
            )

        rows = list(csv.DictReader(io.StringIO(precalib_buffer.getvalue())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["skip_precalib_iv_above_cap"], 0)
        self.assertIn("2025-11-14T18:03:00Z", results)
        self.assertEqual(
            calibration_cls.call_args.kwargs["vols"],
            [[0.20, 4.00]],
        )
        self.assertEqual(
            calibration_cls.call_args.kwargs["percent_strikes"],
            [[110.0 / 112.5934, 115.0 / 112.5934]],
        )
        self.assertEqual(
            [row["trade_datetime_utc"] for row in rows],
            ["2025-11-14T18:03:11Z", "2025-11-14T18:03:47Z"],
        )
        self.assertEqual(
            [row["calibration_datetime_utc"] for row in rows],
            ["2025-11-14T18:03:00Z", "2025-11-14T18:03:00Z"],
        )
        self.assertEqual([row["passes_precalib_filter"] for row in rows], ["true", "true"])
        self.assertEqual([row["filter_reason"] for row in rows], ["", ""])

    def test_finalize_minute_surface_dispatches_to_sabr(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.50,
                weight=3.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]
        stats = defaultdict(int)
        results = {}

        with patch("scripts.generate_surface.model.sabr.SabrCalibrationHagan") as calibration_cls:
            calibration_cls.return_value.params = {
                "alpha": [0.2],
                "beta": [1.0],
                "rho": [0.0],
                "nu": [0.1],
                "business_days": [94],
            }
            _finalize_minute_surface(
                minute_ts=minute_ts,
                valuation_date=minute_ts.date(),
                candidates=candidates,
                implied_vols=[0.20, 0.30],
                min_strikes_per_expiry=1,
                min_expiries_per_minute=1,
                max_precalib_iv=DEFAULT_MAX_PRECALIB_IV,
                vol_daycount=self._vol_daycount(),
                results=results,
                stats=stats,
                surface_model="sabr",
            )

        self.assertEqual(results["2025-11-14T18:03:00Z"]["surface_model"], "sabr")
        self.assertEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"],
            calibration_cls.return_value.params,
        )
        self.assertAlmostEqual(calibration_cls.call_args.kwargs["vols"][0][0], 0.275, places=12)

    def test_finalize_minute_surface_dispatches_to_cubic_and_serializes_implied_vols(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.50,
                weight=3.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]
        stats = defaultdict(int)
        results = {}

        with patch("scripts.generate_surface.model.cubic.CubicSplineVolSurfaceBuilder") as builder_cls:
            builder_cls.return_value.get_vol_surface.return_value = object()
            _finalize_minute_surface(
                minute_ts=minute_ts,
                valuation_date=minute_ts.date(),
                candidates=candidates,
                implied_vols=[0.20, 0.30],
                min_strikes_per_expiry=1,
                min_expiries_per_minute=1,
                max_precalib_iv=DEFAULT_MAX_PRECALIB_IV,
                vol_daycount=self._vol_daycount(),
                results=results,
                stats=stats,
                surface_model="cubic",
            )

        self.assertEqual(results["2025-11-14T18:03:00Z"]["surface_model"], "cubic")
        self.assertEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"]["business_days"],
            [94],
        )
        self.assertAlmostEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"]["percent_strikes"][0][0],
            110.0 / 112.5934,
            places=12,
        )
        self.assertAlmostEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"]["implied_vols"][0][0],
            0.275,
            places=12,
        )
        self.assertAlmostEqual(builder_cls.call_args.kwargs["vols"][0][0], 0.275, places=12)

    def test_finalize_minute_surface_dispatches_to_raw_and_serializes_implied_vols(self):
        minute_ts = pd.Timestamp("2025-11-14T18:03:00Z")
        candidates = [
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.25,
                weight=1.0,
                trade_ts="2025-11-14T18:03:11Z",
            ),
            self._make_candidate(
                contract_id="TY110O26",
                strike=110.0,
                option_type=OptionType.PUT,
                price=0.50,
                weight=3.0,
                trade_ts="2025-11-14T18:03:47Z",
            ),
        ]
        stats = defaultdict(int)
        results = {}

        _finalize_minute_surface(
            minute_ts=minute_ts,
            valuation_date=minute_ts.date(),
            candidates=candidates,
            implied_vols=[0.20, 0.30],
            min_strikes_per_expiry=1,
            min_expiries_per_minute=1,
            max_precalib_iv=DEFAULT_MAX_PRECALIB_IV,
            vol_daycount=self._vol_daycount(),
            results=results,
            stats=stats,
            surface_model="raw",
        )

        self.assertEqual(results["2025-11-14T18:03:00Z"]["surface_model"], "raw")
        self.assertEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"]["business_days"],
            [94],
        )
        self.assertAlmostEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"]["percent_strikes"][0][0],
            110.0 / 112.5934,
            places=12,
        )
        self.assertAlmostEqual(
            results["2025-11-14T18:03:00Z"]["surface_params"]["implied_vols"][0][0],
            0.275,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
