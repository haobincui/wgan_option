import csv
import json
import logging
import sys
import tempfile
import unittest
from argparse import Namespace
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock, patch

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from quantlib.calculation.analytics.position.instruments.features import OptionType  # noqa: E402
from quantlib.calendar.daycount import DayCountBusN  # noqa: E402
from quantlib.calendar.holidays import usd_calendar  # noqa: E402
from scripts.generate_surface.data_helperd.all import (  # noqa: E402
    PRECALIB_CSV_HEADERS,
    ContractMeta,
    MinuteTradeRow,
    _make_expiry_dt_utc,
    _prepare_option_candidates,
    _tau_years_from_trade_to_expiry,
)
from scripts.generate_surface.data_helperd import excel as excel_common  # noqa: E402
from scripts.generate_surface.data_helperd import window as window_common  # noqa: E402
from scripts.generate_surface.backend.surface_cpu.all import (  # noqa: E402
    _process_minute as cpu_process_minute,
)


class TestGenerateSurfaceWindowLogic(unittest.TestCase):
    TARGET_TS = pd.Timestamp("2026-03-09T14:35:00Z")
    FORWARD_TS = pd.Timestamp("2026-03-09T14:40:00Z")
    EXPIRY_DATE = date(2026, 3, 31)
    EXPIRY_DT = pd.Timestamp("2026-03-31T20:00:00Z").to_pydatetime(warn=False)

    def _make_args(self, tmpdir: str, csv_path: Path) -> Namespace:
        return Namespace(
            config="tests/window-logic",
            input_glob=str(csv_path),
            output_json=str(Path(tmpdir) / "window.json"),
            log_file=str(Path(tmpdir) / "window.log"),
            model="svi",
            data_range="window",
            run_ts="test-run",
            output_dir=str(Path(tmpdir) / "run"),
            resolved_config_path=str(Path(tmpdir) / "run" / "surface-resolved_config.yaml"),
            data_date="2026-03-09",
            expiration_time_utc="20:00:00",
            days_in_year=250,
            min_strikes_per_expiry=1,
            min_expiries_per_minute=1,
            max_precalib_iv=3.0,
            max_files=0,
            max_minutes=0,
            chunk_size=1000,
            save_precalib_csv=False,
            precalib_csv=str(Path(tmpdir) / "window_precalib.csv"),
        )

    def _write_trade_csv(self, tmpdir: str, rows: List[Dict[str, object]]) -> Path:
        csv_path = Path(tmpdir) / "window_input.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def _fake_build_rows_for_minute(
        self,
        minute_df: pd.DataFrame,
        expiry_inference_date,
        expiration_time_utc,
        calendar,
        contract_cache,
        stats,
    ) -> List[MinuteTradeRow]:
        del expiry_inference_date, expiration_time_utc, calendar, contract_cache, stats
        rows: List[MinuteTradeRow] = []
        for rec in minute_df.itertuples(index=False):
            ric = str(rec.ric)
            if ric.startswith("FUT"):
                meta = ContractMeta(
                    contract_type="future",
                    underlying="TY",
                    maturity_month_code="H",
                    contract_id=ric,
                )
            else:
                strike = 110.0
                if "111" in ric:
                    strike = 111.0
                elif "112" in ric:
                    strike = 112.0
                elif "113" in ric:
                    strike = 113.0
                meta = ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=strike,
                    option_type=OptionType.PUT if "P" in ric else OptionType.CALL,
                    expiry_date=self.EXPIRY_DATE,
                    expiry_dt_utc=self.EXPIRY_DT,
                    contract_id=ric,
                )
            rows.append(
                MinuteTradeRow(
                    trade_ts=rec.trade_dt,
                    meta=meta,
                    price=float(rec.price),
                    volume=float(rec.volume),
                )
            )
        return rows

    def _make_fake_process(self, recorded_calls: Dict[str, Dict[str, object]], target_ts: pd.Timestamp):
        def _fake_process(
            minute_ts,
            rows,
            days_in_year,
            min_strikes_per_expiry,
            min_expiries_per_minute,
            max_precalib_iv,
            vol_daycount,
            calendar,
            target_future_month_code,
            last_spot_by_key,
            results,
            stats,
            precalib_writer=None,
            tau_anchor_ts=None,
            count_stat_key="total_minutes",
            surface_model="svi",
        ):
            del (
                days_in_year,
                min_strikes_per_expiry,
                min_expiries_per_minute,
                max_precalib_iv,
                vol_daycount,
                calendar,
                target_future_month_code,
                precalib_writer,
                surface_model,
            )
            key = window_common._to_utc_minute_string(minute_ts)
            recorded_calls[key] = {
                "row_count": len(rows),
                "unique_row_minutes": sorted(
                    {
                        window_common._to_utc_minute_string(pd.Timestamp(row.trade_ts).floor("min"))
                        for row in rows
                    }
                ),
                "last_spot_by_key": dict(last_spot_by_key),
                "tau_anchor_ts": window_common._to_utc_minute_string(tau_anchor_ts),
                "count_stat_key": count_stat_key,
            }
            stats[count_stat_key] += 1
            results[key] = {
                "row_count": len(rows),
                "unique_row_minutes": recorded_calls[key]["unique_row_minutes"],
            }

        return _fake_process

    def _vol_daycount(self) -> DayCountBusN:
        calendar = usd_calendar()
        return DayCountBusN("BUS250USD", calendar, 250)

    def test_build_target_window_map_uses_strict_forward_backward_ranges(self):
        window_map = window_common._build_target_window_map([self.TARGET_TS], window_minutes=5)
        target_key = "2026-03-09T14:35:00Z"

        self.assertEqual(
            [window_common._to_utc_minute_string(ts) for ts in window_map[target_key]["backward"]["minutes"]],
            [
                "2026-03-09T14:30:00Z",
                "2026-03-09T14:31:00Z",
                "2026-03-09T14:32:00Z",
                "2026-03-09T14:33:00Z",
                "2026-03-09T14:34:00Z",
                "2026-03-09T14:35:00Z",
            ],
        )
        self.assertEqual(
            [window_common._to_utc_minute_string(ts) for ts in window_map[target_key]["forward"]["minutes"]],
            [
                "2026-03-09T14:35:00Z",
                "2026-03-09T14:36:00Z",
                "2026-03-09T14:37:00Z",
                "2026-03-09T14:38:00Z",
                "2026-03-09T14:39:00Z",
                "2026-03-09T14:40:00Z",
            ],
        )
        self.assertEqual(
            window_common._to_utc_minute_string(window_map[target_key]["backward"]["anchor_ts"]),
            "2026-03-09T14:35:00Z",
        )
        self.assertEqual(
            window_common._to_utc_minute_string(window_map[target_key]["forward"]["anchor_ts"]),
            "2026-03-09T14:40:00Z",
        )

    def test_generate_surfaces_for_datetime_windows_aggregates_each_side_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._write_trade_csv(
                tmpdir,
                [
                    {"#RIC": "FUT_PRIOR", "Date-Time": "2026-03-09T14:30:00Z", "Price": 100.0, "Volume": 1.0},
                    {"#RIC": "OPTP110", "Date-Time": "2026-03-09T14:34:00Z", "Price": 0.25, "Volume": 1.0},
                    {"#RIC": "OPTP111", "Date-Time": "2026-03-09T14:35:00Z", "Price": 0.35, "Volume": 1.0},
                    {"#RIC": "FUT_WIN", "Date-Time": "2026-03-09T14:36:00Z", "Price": 101.0, "Volume": 1.0},
                    {"#RIC": "OPTC112", "Date-Time": "2026-03-09T14:36:00Z", "Price": 0.45, "Volume": 1.0},
                    {"#RIC": "OPTC113", "Date-Time": "2026-03-09T14:40:00Z", "Price": 0.55, "Volume": 1.0},
                ],
            )
            args = self._make_args(tmpdir, csv_path)
            recorded_calls: Dict[str, Dict[str, object]] = {}
            fake_process = self._make_fake_process(recorded_calls, self.TARGET_TS)

            with patch.object(
                window_common,
                "_build_rows_for_minute",
                side_effect=self._fake_build_rows_for_minute,
            ):
                surfaces = window_common.generate_surfaces_for_datetime_windows(
                    args=args,
                    target_datetimes=[self.TARGET_TS],
                    process_minute_fn=fake_process,
                    window_minutes=5,
                )
                logging.shutdown()

            backward_key = "2026-03-09T14:35:00Z"
            forward_key = "2026-03-09T14:40:00Z"
            target_key = "2026-03-09T14:35:00Z"

            self.assertEqual(set(recorded_calls.keys()), {backward_key, forward_key})
            self.assertEqual(
                recorded_calls[backward_key]["unique_row_minutes"],
                ["2026-03-09T14:30:00Z", "2026-03-09T14:34:00Z", "2026-03-09T14:35:00Z"],
            )
            self.assertEqual(
                recorded_calls[forward_key]["unique_row_minutes"],
                ["2026-03-09T14:35:00Z", "2026-03-09T14:36:00Z", "2026-03-09T14:40:00Z"],
            )
            self.assertEqual(recorded_calls[backward_key]["row_count"], 3)
            self.assertEqual(recorded_calls[forward_key]["row_count"], 4)
            self.assertEqual(recorded_calls[backward_key]["last_spot_by_key"], {("TY", "H"): 100.0})
            self.assertEqual(recorded_calls[forward_key]["last_spot_by_key"], {("TY", "H"): 101.0})
            self.assertEqual(recorded_calls[backward_key]["tau_anchor_ts"], backward_key)
            self.assertEqual(recorded_calls[forward_key]["tau_anchor_ts"], forward_key)
            self.assertEqual(recorded_calls[backward_key]["count_stat_key"], "window_surface_attempts")
            self.assertEqual(recorded_calls[forward_key]["count_stat_key"], "window_surface_attempts")

            self.assertEqual(surfaces[target_key]["backward"]["snapshot_time_utc"], backward_key)
            self.assertEqual(surfaces[target_key]["forward"]["snapshot_time_utc"], forward_key)
            self.assertEqual(
                surfaces[target_key]["backward"]["surface_model"],
                "svi",
            )
            self.assertEqual(
                surfaces[target_key]["forward"]["surface_model"],
                "svi",
            )
            self.assertEqual(
                surfaces[target_key]["backward"]["surface_params"],
                {
                    "row_count": 3,
                    "unique_row_minutes": [
                        "2026-03-09T14:30:00Z",
                        "2026-03-09T14:34:00Z",
                        "2026-03-09T14:35:00Z",
                    ],
                },
            )
            self.assertEqual(
                surfaces[target_key]["forward"]["surface_params"],
                {
                    "row_count": 4,
                    "unique_row_minutes": [
                        "2026-03-09T14:35:00Z",
                        "2026-03-09T14:36:00Z",
                        "2026-03-09T14:40:00Z",
                    ],
                },
            )
            resolved_config_path = Path(args.output_dir) / "surface-resolved_config.yaml"
            self.assertTrue(resolved_config_path.exists())
            resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
            self.assertEqual(resolved_config["surface_builder"]["generate_surface"]["model"], "svi")
            self.assertEqual(resolved_config["surface_builder"]["generate_surface"]["data_range"], "window")
            self.assertEqual(resolved_config["surface_builder"]["generate_surface"]["run_ts"], "test-run")
            self.assertEqual(
                resolved_config["surface_builder"]["generate_surface"]["output_json"],
                str(Path(tmpdir) / "window.json"),
            )

            written = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
            self.assertEqual(written, surfaces)

    def test_window_precalib_csv_uses_excel_compatible_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._write_trade_csv(
                tmpdir,
                [
                    {"#RIC": "FUT_PRIOR", "Date-Time": "2026-03-09T14:30:00Z", "Price": 112.5, "Volume": 1.0},
                    {"#RIC": "OPTP110", "Date-Time": "2026-03-09T14:34:00Z", "Price": 0.25, "Volume": 1.0},
                    {"#RIC": "OPTP111", "Date-Time": "2026-03-09T14:35:00Z", "Price": 0.35, "Volume": 1.0},
                    {"#RIC": "FUT_WIN", "Date-Time": "2026-03-09T14:36:00Z", "Price": 112.7, "Volume": 1.0},
                    {"#RIC": "OPTC112", "Date-Time": "2026-03-09T14:36:00Z", "Price": 0.45, "Volume": 1.0},
                    {"#RIC": "OPTC113", "Date-Time": "2026-03-09T14:40:00Z", "Price": 0.55, "Volume": 1.0},
                ],
            )
            args = self._make_args(tmpdir, csv_path)
            args.save_precalib_csv = True

            with patch.object(
                window_common,
                "_build_rows_for_minute",
                side_effect=self._fake_build_rows_for_minute,
            ):
                surfaces = window_common.generate_surfaces_for_datetime_windows(
                    args=args,
                    target_datetimes=[self.TARGET_TS],
                    process_minute_fn=cpu_process_minute,
                    window_minutes=5,
                )
                logging.shutdown()

            precalib_path = Path(args.precalib_csv)
            with precalib_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, PRECALIB_CSV_HEADERS)
            self.assertTrue(rows)
            self.assertNotIn("snapshot_time_utc", rows[0])
            self.assertIn("calibration_datetime_utc", rows[0])
            self.assertEqual(
                sorted({row["calibration_datetime_utc"] for row in rows}),
                ["2026-03-09T14:35:00Z", "2026-03-09T14:40:00Z"],
            )
            self.assertEqual(
                surfaces["2026-03-09T14:35:00Z"]["backward"]["snapshot_time_utc"],
                "2026-03-09T14:35:00Z",
            )
            self.assertEqual(
                surfaces["2026-03-09T14:35:00Z"]["forward"]["snapshot_time_utc"],
                "2026-03-09T14:40:00Z",
            )

    def test_generate_surfaces_for_datetime_windows_keeps_null_side_when_calibration_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self._write_trade_csv(
                tmpdir,
                [
                    {"#RIC": "FUT_PRIOR", "Date-Time": "2026-03-09T14:30:00Z", "Price": 100.0, "Volume": 1.0},
                    {"#RIC": "OPTP110", "Date-Time": "2026-03-09T14:34:00Z", "Price": 0.25, "Volume": 1.0},
                    {"#RIC": "OPTC112", "Date-Time": "2026-03-09T14:36:00Z", "Price": 0.45, "Volume": 1.0},
                ],
            )
            args = self._make_args(tmpdir, csv_path)

            def fake_process(
                minute_ts,
                rows,
                days_in_year,
                min_strikes_per_expiry,
                min_expiries_per_minute,
                max_precalib_iv,
                vol_daycount,
                calendar,
                target_future_month_code,
                last_spot_by_key,
                results,
                stats,
                precalib_writer=None,
                tau_anchor_ts=None,
                count_stat_key="total_minutes",
                surface_model="svi",
            ):
                del (
                    rows,
                    days_in_year,
                    min_strikes_per_expiry,
                    min_expiries_per_minute,
                    max_precalib_iv,
                    vol_daycount,
                    calendar,
                    target_future_month_code,
                    last_spot_by_key,
                    precalib_writer,
                    tau_anchor_ts,
                    surface_model,
                )
                stats[count_stat_key] += 1
                key = window_common._to_utc_minute_string(minute_ts)
                if key == "2026-03-09T14:35:00Z":
                    results[key] = {"ok": True}

            with patch.object(
                window_common,
                "_build_rows_for_minute",
                side_effect=self._fake_build_rows_for_minute,
            ):
                surfaces = window_common.generate_surfaces_for_datetime_windows(
                    args=args,
                    target_datetimes=[self.TARGET_TS],
                    process_minute_fn=fake_process,
                    window_minutes=5,
                )
                logging.shutdown()

            target_key = "2026-03-09T14:35:00Z"
            self.assertEqual(surfaces[target_key]["backward"]["surface_params"], {"ok": True})
            self.assertEqual(surfaces[target_key]["backward"]["surface_model"], "svi")
            self.assertIsNone(surfaces[target_key]["forward"]["surface_params"])

    def test_prepare_option_candidates_uses_window_anchor_for_tau(self):
        anchor_ts = pd.Timestamp("2026-03-09T14:40:00Z")
        option_rows = [
            MinuteTradeRow(
                trade_ts=pd.Timestamp("2026-03-09T14:36:15Z"),
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=110.0,
                    option_type=OptionType.PUT,
                    expiry_date=self.EXPIRY_DATE,
                    expiry_dt_utc=self.EXPIRY_DT,
                    contract_id="TY110P26",
                ),
                price=0.25,
                volume=1.0,
            ),
            MinuteTradeRow(
                trade_ts=pd.Timestamp("2026-03-09T14:39:45Z"),
                meta=ContractMeta(
                    contract_type="option",
                    underlying="TY",
                    strike=111.0,
                    option_type=OptionType.CALL,
                    expiry_date=self.EXPIRY_DATE,
                    expiry_dt_utc=self.EXPIRY_DT,
                    contract_id="TY111C26",
                ),
                price=0.35,
                volume=2.0,
            ),
        ]

        calendar = usd_calendar()
        vol_daycount = self._vol_daycount()
        stats = defaultdict(int)

        _, anchored = _prepare_option_candidates(
            minute_ts=anchor_ts,
            option_rows=option_rows,
            minute_spot={("TY", "H"): 112.5},
            last_spot_by_key={},
            target_future_month_code="H",
            vol_daycount=vol_daycount,
            calendar=calendar,
            stats=stats,
            tau_anchor_ts=anchor_ts,
        )
        _, unanchored = _prepare_option_candidates(
            minute_ts=anchor_ts,
            option_rows=option_rows,
            minute_spot={("TY", "H"): 112.5},
            last_spot_by_key={},
            target_future_month_code="H",
            vol_daycount=vol_daycount,
            calendar=calendar,
            stats=defaultdict(int),
        )

        expected_tau = _tau_years_from_trade_to_expiry(anchor_ts, self.EXPIRY_DT, vol_daycount)
        self.assertEqual(len(anchored), 2)
        self.assertEqual(len(unanchored), 2)
        self.assertTrue(all(candidate.tau == expected_tau for candidate in anchored))
        self.assertNotEqual(anchored[0].tau, unanchored[0].tau)
        self.assertEqual(anchored[0].business_days, anchored[1].business_days)

    def test_prepare_option_candidates_uses_ty_underlying_future_month_in_window_mode(self):
        anchor_ts = pd.Timestamp("2022-02-08T15:02:00Z")
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
            minute_ts=anchor_ts,
            option_rows=option_rows,
            minute_spot={
                ("TY", "H"): 126.52635324112379,
                ("TY", "M"): 126.390625,
            },
            last_spot_by_key={},
            target_future_month_code="H",
            vol_daycount=self._vol_daycount(),
            calendar=usd_calendar(),
            stats=defaultdict(int),
            tau_anchor_ts=anchor_ts,
        )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].spot, 126.390625, places=12)

    def test_filter_files_for_target_windows_keeps_only_overlapping_month_files(self):
        window_map = window_common._build_target_window_map(
            [pd.Timestamp("2022-01-27T09:24:00Z")],
            window_minutes=5,
        )

        files = [
            "data/raw/option_data/0#TY+/0#TY+_2021-12-01_2021-12-31.csv.gz",
            "data/raw/option_data/0#TY+/0#TY+_2022-01-01_2022-01-31.csv.gz",
            "data/raw/option_data/0#TY+/0#TY+_2022-02-01_2022-02-28.csv.gz",
            "data/raw/option_data/window_input.csv",
        ]

        filtered = window_common._filter_files_for_target_windows(files, window_map)

        self.assertEqual(
            filtered,
            [
                "data/raw/option_data/0#TY+/0#TY+_2022-01-01_2022-01-31.csv.gz",
                "data/raw/option_data/window_input.csv",
            ],
        )

    def test_run_excel_job_preserves_forward_backward_structure(self):
        sample_output = {
            "2026-03-09T14:35:00Z": {
                "backward": {
                    "snapshot_time_utc": "2026-03-09T14:35:00Z",
                    "surface_model": "svi",
                    "surface_params": {"ok": True},
                },
                "forward": {
                    "snapshot_time_utc": "2026-03-09T14:40:00Z",
                    "surface_model": "svi",
                    "surface_params": None,
                },
            }
        }
        args = Namespace(
            target_xlsx="ignored.xlsx",
            sheet_name="Sheet1",
            date_column="PD",
            time_column="ET",
            source_timezone="America/New_York",
            max_target_datetimes=0,
            window_minutes=5,
        )

        with patch.object(
            excel_common,
            "_load_target_datetimes_from_excel",
            return_value=([self.TARGET_TS], {"excel_rows_total": 1, "parsed_rows": 1, "invalid_rows": 0, "deduped_targets": 1, "final_targets_used": 1}),
        ), patch.object(
            excel_common,
            "generate_surfaces_for_datetime_windows",
            return_value=sample_output,
        ) as mock_generate, patch.object(excel_common.logger, "info"):
            result = excel_common.run_excel_job(args, process_minute_fn=Mock())

        self.assertEqual(result, sample_output)
        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
