import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.merge_file import merge_svi  # noqa: E402


def _model_iv(total_var: float, business_days: int) -> float:
    return math.sqrt(total_var / (business_days / merge_svi.DAYS_IN_YEAR))


class TestMergeSvi(unittest.TestCase):
    def _write_fixture_files(self, tmpdir: str):
        xlsx_path = Path(tmpdir) / "news_input.xlsx"
        result_dir = Path(tmpdir) / "result"
        result_dir.mkdir()
        csv_path = result_dir / "minute_svi_precalib_points.csv"
        json_path = result_dir / "minute_svi_params.json"

        news_df = pd.DataFrame(
            [
                {
                    "SourceFile": "a.txt",
                    "ArticleID": 1,
                    "HD": "headline a",
                    "LP": "lead a",
                    "PD": "2022-12-30",
                    "ET": "08:28:00",
                    "HD_embedding": "[0.1, 0.2]",
                    "LP_embedding": "[0.3, 0.4]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
                {
                    "SourceFile": "b.txt",
                    "ArticleID": 2,
                    "HD": "headline b",
                    "LP": "lead b",
                    "PD": "2022-12-30",
                    "ET": "08:39:00",
                    "HD_embedding": "[0.5, 0.6]",
                    "LP_embedding": "[0.7, 0.8]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
                {
                    "SourceFile": "bad.txt",
                    "ArticleID": 3,
                    "HD": "headline bad",
                    "LP": "lead bad",
                    "PD": "not-a-date",
                    "ET": "not-a-time",
                    "HD_embedding": "[0.9, 1.0]",
                    "LP_embedding": "[1.1, 1.2]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
            ]
        )
        news_df.to_excel(xlsx_path, index=False, engine="openpyxl")

        total_var_30 = 0.001
        iv_30 = _model_iv(total_var_30, 30)
        total_var_40 = 0.001
        iv_40 = _model_iv(total_var_40, 40)

        csv_df = pd.DataFrame(
            [
                {
                    "trade_datetime_utc": "2022-12-30T13:28:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:28:00Z",
                    "business_days": 30,
                    "maturity_date": "2023-02-01",
                    "contract_id": "OPT1",
                    "option_type": "P",
                    "strike": 110.0,
                    "price": 0.25,
                    "spot": 110.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_30,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 2.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:28:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:28:00Z",
                    "business_days": 35,
                    "maturity_date": "2023-02-06",
                    "contract_id": "OPT2",
                    "option_type": "C",
                    "strike": 110.0,
                    "price": 0.35,
                    "spot": 110.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_30,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:28:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:28:00Z",
                    "business_days": 30,
                    "maturity_date": "2023-02-01",
                    "contract_id": "OPT3",
                    "option_type": "P",
                    "strike": 112.0,
                    "price": 0.45,
                    "spot": 112.0,
                    "percent_strike": 1.02,
                    "implied_vol": 0.30,
                    "passes_precalib_filter": False,
                    "filter_reason": "cap",
                    "weight": 3.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:39:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:39:00Z",
                    "business_days": 40,
                    "maturity_date": "2023-02-11",
                    "contract_id": "OPT4",
                    "option_type": "P",
                    "strike": 109.0,
                    "price": 0.20,
                    "spot": 109.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_40,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:44:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:44:00Z",
                    "business_days": 50,
                    "maturity_date": "2023-02-21",
                    "contract_id": "OPT5",
                    "option_type": "C",
                    "strike": 111.0,
                    "price": 0.22,
                    "spot": 111.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_40,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
            ]
        )
        csv_df.to_csv(csv_path, index=False)

        json_payload = {
            "2022-12-30T13:28:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:28:00Z",
                    "svi_params": {
                        "a": [0.0005],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.05],
                        "business_days": [30],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:33:00Z",
                    "svi_params": None,
                },
            },
            "2022-12-30T13:33:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:33:00Z",
                    "svi_params": {
                        "a": [0.0007],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.03],
                        "business_days": [45],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:38:00Z",
                    "svi_params": {
                        "a": [0.0007],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.03],
                        "business_days": [45],
                    },
                },
            },
            "2022-12-30T13:39:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:39:00Z",
                    "svi_params": {
                        "a": [0.002],
                        "b": [0.0],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.0],
                        "business_days": [40],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:44:00Z",
                    "svi_params": {
                        "a": [0.0006, 0.0012],
                        "b": [0.01, 0.01],
                        "rho": [0.0, 0.0],
                        "m": [0.0, 0.0],
                        "sigma": [0.04, 0.03],
                        "business_days": [40, 60],
                    },
                },
            },
        }
        json_path.write_text(json.dumps(json_payload), encoding="utf-8")
        return xlsx_path, result_dir

    def test_build_workbook_frames_creates_two_direction_rows_with_expected_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)

            workbook = merge_svi.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_direction_audit"]

            self.assertEqual(len(audit), 6)
            self.assertEqual(audit["news_row_id"].value_counts().to_dict(), {1: 2, 2: 2, 3: 2})

            usable = audit[(audit["news_row_id"] == 1) & (audit["direction"] == "backward")].iloc[0]
            self.assertEqual(usable["matched_snapshot_time_utc"], "2022-12-30T13:28:00Z")
            self.assertEqual(usable["json_target_timestamp_utc"], "2022-12-30T13:28:00Z")
            self.assertEqual(usable["raw_point_count"], 3)
            self.assertEqual(usable["raw_point_pass_count"], 2)
            self.assertEqual(usable["raw_point_fail_count"], 1)
            self.assertAlmostEqual(float(usable["exact_slice_point_ratio"]), 0.5, places=8)
            self.assertAlmostEqual(float(usable["nearest_slice_gap_days_mean"]), 2.5, places=8)
            self.assertAlmostEqual(float(usable["nearest_slice_gap_days_max"]), 5.0, places=8)
            self.assertAlmostEqual(float(usable["weighted_iv_rmse"]), 0.0, places=8)
            self.assertEqual(int(usable["svi_boundary_flag"]), 0)
            self.assertEqual(int(usable["svi_placeholder_flag"]), 0)
            self.assertEqual(usable["fit_quality_label"], "usable")
            self.assertEqual(int(usable["training_candidate_flag"]), 1)

            strict_forward = audit[(audit["news_row_id"] == 1) & (audit["direction"] == "forward")].iloc[0]
            self.assertEqual(strict_forward["matched_snapshot_time_utc"], "2022-12-30T13:33:00Z")
            self.assertFalse(bool(strict_forward["has_svi_params"]))
            self.assertEqual(strict_forward["fit_quality_label"], "no_svi")
            self.assertEqual(strict_forward["exclude_reason"], "no_svi")

            placeholder = audit[(audit["news_row_id"] == 2) & (audit["direction"] == "backward")].iloc[0]
            self.assertEqual(int(placeholder["svi_placeholder_flag"]), 1)
            self.assertEqual(int(placeholder["svi_boundary_flag"]), 1)
            self.assertEqual(placeholder["fit_quality_label"], "placeholder")
            self.assertEqual(placeholder["exclude_reason"], "placeholder")

            poor = audit[(audit["news_row_id"] == 2) & (audit["direction"] == "forward")].iloc[0]
            self.assertTrue(bool(poor["has_svi_params"]))
            self.assertEqual(int(poor["svi_placeholder_flag"]), 0)
            self.assertEqual(poor["fit_quality_label"], "poor")
            self.assertEqual(poor["exclude_reason"], "low_exact_slice_ratio")
            self.assertAlmostEqual(float(poor["exact_slice_point_ratio"]), 0.0, places=8)

            invalid = audit[(audit["news_row_id"] == 3) & (audit["direction"] == "backward")].iloc[0]
            self.assertEqual(invalid["matched_snapshot_time_utc"], "")
            self.assertEqual(invalid["fit_quality_label"], "no_svi")

    def test_slice_detail_uses_expected_assignment_and_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)

            workbook = merge_svi.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            slice_df = workbook["svi_slice_detail"]

            self.assertEqual(len(slice_df), 4)

            usable_slice = slice_df[
                (slice_df["news_row_id"] == 1)
                & (slice_df["direction"] == "backward")
                & (slice_df["slice_index"] == 0)
            ].iloc[0]
            self.assertEqual(int(usable_slice["raw_point_count_on_slice"]), 3)
            self.assertEqual(int(usable_slice["raw_point_pass_count_on_slice"]), 2)
            self.assertAlmostEqual(float(usable_slice["weighted_iv_rmse_slice"]), 0.0, places=8)
            self.assertEqual(int(usable_slice["boundary_flag_slice"]), 0)
            self.assertEqual(int(usable_slice["placeholder_flag_slice"]), 0)

            placeholder_slice = slice_df[
                (slice_df["news_row_id"] == 2)
                & (slice_df["direction"] == "backward")
            ].iloc[0]
            self.assertEqual(int(placeholder_slice["boundary_flag_slice"]), 1)
            self.assertEqual(int(placeholder_slice["placeholder_flag_slice"]), 1)

            tie_break_low = slice_df[
                (slice_df["news_row_id"] == 2)
                & (slice_df["direction"] == "forward")
                & (slice_df["slice_index"] == 0)
            ].iloc[0]
            tie_break_high = slice_df[
                (slice_df["news_row_id"] == 2)
                & (slice_df["direction"] == "forward")
                & (slice_df["slice_index"] == 1)
            ].iloc[0]
            self.assertEqual(int(tie_break_low["raw_point_pass_count_on_slice"]), 1)
            self.assertEqual(int(tie_break_high["raw_point_pass_count_on_slice"]), 0)

    def test_main_writes_expected_workbook_and_filters_gan_input_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)
            output_path = result_dir / "merged_svi.xlsx"

            with patch.object(merge_svi, "DEFAULT_NEWS_XLSX_PATH", xlsx_path):
                written_path = merge_svi.main(["--input-dir", str(result_dir)])

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())

            workbook = pd.ExcelFile(output_path, engine="openpyxl")
            self.assertEqual(
                workbook.sheet_names,
                ["news_direction_audit", "svi_slice_detail", "gan_input_ready"],
            )

            gan_ready = pd.read_excel(output_path, sheet_name="gan_input_ready", engine="openpyxl")
            self.assertEqual(len(gan_ready), 1)
            self.assertEqual(gan_ready.loc[0, "direction"], "backward")
            self.assertEqual(gan_ready.loc[0, "matched_snapshot_time_utc"], "2022-12-30T13:28:00Z")
            self.assertEqual(int(gan_ready.loc[0, "training_candidate_flag"]), 1)


if __name__ == "__main__":
    unittest.main()
