import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.merge_file import merge_vol  # noqa: E402


def _model_iv(total_var: float, business_days: int) -> float:
    return math.sqrt(total_var / (business_days / merge_vol.DAYS_IN_YEAR))


class TestMergeVol(unittest.TestCase):
    def test_parse_args_accepts_explicit_grid_contract(self):
        args = merge_vol._parse_args(
            [
                "--input-dir",
                "data/processed/example",
                "--news-xlsx",
                "data/raw/news.xlsx",
                "--strike-bins",
                "8",
                "--maturity-bins",
                "10",
                "--moneyness-min",
                "0.8",
                "--moneyness-max",
                "1.2",
                "--maturity-min-days",
                "14",
                "--maturity-max-days",
                "180",
            ]
        )

        self.assertEqual(args.news_xlsx, "data/raw/news.xlsx")
        self.assertEqual(args.strike_bins, 8)
        self.assertEqual(args.maturity_bins, 10)
        self.assertAlmostEqual(args.moneyness_min, 0.8)
        self.assertAlmostEqual(args.moneyness_max, 1.2)
        self.assertEqual(args.maturity_min_days, 14)
        self.assertEqual(args.maturity_max_days, 180)

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
                    "SourceFile": "c.txt",
                    "ArticleID": 3,
                    "HD": "headline c",
                    "LP": "lead c",
                    "PD": "2022-12-30",
                    "ET": "08:50:00",
                    "HD_embedding": "[0.9, 1.0]",
                    "LP_embedding": "[1.1, 1.2]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
            ]
        )
        news_df.to_excel(xlsx_path, index=False, engine="openpyxl")

        total_var_30 = 0.001
        total_var_40 = 0.001
        total_var_20 = 0.002
        total_var_25 = 0.0012
        iv_30 = _model_iv(total_var_30, 30)
        iv_40 = _model_iv(total_var_40, 40)
        iv_20 = _model_iv(total_var_20, 20)
        iv_25 = _model_iv(total_var_25, 25)

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
                    "trade_datetime_utc": "2022-12-30T13:33:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:33:00Z",
                    "business_days": 40,
                    "maturity_date": "2023-02-11",
                    "contract_id": "OPT3",
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
                    "trade_datetime_utc": "2022-12-30T13:39:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:39:00Z",
                    "business_days": 20,
                    "maturity_date": "2023-01-30",
                    "contract_id": "OPT4",
                    "option_type": "P",
                    "strike": 108.0,
                    "price": 0.18,
                    "spot": 108.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_20,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:44:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:44:00Z",
                    "business_days": 25,
                    "maturity_date": "2023-02-06",
                    "contract_id": "OPT5",
                    "option_type": "C",
                    "strike": 109.0,
                    "price": 0.21,
                    "spot": 109.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_25,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:50:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:50:00Z",
                    "business_days": 20,
                    "maturity_date": "2023-01-30",
                    "contract_id": "OPT6",
                    "option_type": "P",
                    "strike": 107.0,
                    "price": 0.19,
                    "spot": 107.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_20,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:55:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:55:00Z",
                    "business_days": 25,
                    "maturity_date": "2023-02-06",
                    "contract_id": "OPT7",
                    "option_type": "C",
                    "strike": 107.0,
                    "price": 0.22,
                    "spot": 107.0,
                    "percent_strike": 1.0,
                    "implied_vol": iv_25,
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
                    "svi_params": {
                        "a": [0.0007],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.03],
                        "business_days": [40],
                    },
                },
            },
            "2022-12-30T13:39:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:39:00Z",
                    "svi_params": {
                        "a": [0.002],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.03],
                        "business_days": [20],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:44:00Z",
                    "svi_params": None,
                },
            },
            "2022-12-30T13:50:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:50:00Z",
                    "svi_params": {
                        "a": [0.002],
                        "b": [0.0],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.0],
                        "business_days": [20],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:55:00Z",
                    "svi_params": {
                        "a": [0.0008],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.04],
                        "business_days": [25],
                    },
                },
            },
        }
        json_path.write_text(json.dumps(json_payload), encoding="utf-8")
        return xlsx_path, result_dir

    def _write_model_payload_fixture_files(
        self,
        tmpdir: str,
        *,
        surface_model: str,
        surface_params: Dict[str, object],
        implied_vol: float,
    ):
        xlsx_path = Path(tmpdir) / "news_input.xlsx"
        result_dir = Path(tmpdir) / "result"
        result_dir.mkdir()
        csv_path = result_dir / f"surface-{surface_model}-all-precalib-points.csv"
        json_path = result_dir / f"surface-{surface_model}-all.json"

        pd.DataFrame(
            [
                {
                    "SourceFile": "model.txt",
                    "ArticleID": 1,
                    "HD": "headline",
                    "LP": "lead",
                    "PD": "2022-12-30",
                    "ET": "08:28:00",
                    "HD_embedding": "[0.1, 0.2]",
                    "LP_embedding": "[0.3, 0.4]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                }
            ]
        ).to_excel(xlsx_path, index=False, engine="openpyxl")

        pd.DataFrame(
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
                    "implied_vol": implied_vol,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 2.0,
                },
                {
                    "trade_datetime_utc": "2022-12-30T13:33:00Z",
                    "calibration_datetime_utc": "2022-12-30T13:33:00Z",
                    "business_days": 30,
                    "maturity_date": "2023-02-01",
                    "contract_id": "OPT2",
                    "option_type": "C",
                    "strike": 110.0,
                    "price": 0.30,
                    "spot": 110.0,
                    "percent_strike": 1.0,
                    "implied_vol": implied_vol,
                    "passes_precalib_filter": True,
                    "filter_reason": "",
                    "weight": 1.0,
                },
            ]
        ).to_csv(csv_path, index=False)

        json_payload = {
            "2022-12-30T13:28:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:28:00Z",
                    "surface_model": surface_model,
                    "surface_params": surface_params,
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:33:00Z",
                    "surface_model": surface_model,
                    "surface_params": surface_params,
                },
            }
        }
        json_path.write_text(json.dumps(json_payload), encoding="utf-8")
        return xlsx_path, result_dir

    def test_build_workbook_frames_creates_pair_rows_with_expected_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)

            workbook = merge_vol.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_surface_pair_audit"]

            self.assertEqual(len(audit), 3)

            usable = audit[audit["news_row_id"] == 1].iloc[0]
            self.assertEqual(usable["current_snapshot_time_utc"], "2022-12-30T13:28:00Z")
            self.assertEqual(usable["target_snapshot_time_utc"], "2022-12-30T13:33:00Z")
            self.assertTrue(bool(usable["current_has_svi"]))
            self.assertTrue(bool(usable["target_has_svi"]))
            self.assertTrue(bool(usable["current_has_surface"]))
            self.assertTrue(bool(usable["target_has_surface"]))
            self.assertEqual(int(usable["current_surface_slice_count"]), 1)
            self.assertEqual(int(usable["target_surface_slice_count"]), 1)
            self.assertEqual(usable["pair_quality_label"], "usable")
            self.assertEqual(int(usable["training_candidate_flag"]), 1)
            self.assertEqual(usable["surface_shape"], "[16, 16]")
            self.assertEqual(len(json.loads(usable["current_surface_flat"])), 256)
            self.assertEqual(len(json.loads(usable["target_surface_flat"])), 256)

            missing_target = audit[audit["news_row_id"] == 2].iloc[0]
            self.assertTrue(bool(missing_target["current_has_svi"]))
            self.assertFalse(bool(missing_target["target_has_svi"]))
            self.assertEqual(missing_target["pair_quality_label"], "no_target_svi")
            self.assertEqual(missing_target["exclude_reason"], "no_target_svi")

            current_placeholder = audit[audit["news_row_id"] == 3].iloc[0]
            self.assertEqual(int(current_placeholder["current_placeholder_flag"]), 1)
            self.assertEqual(current_placeholder["pair_quality_label"], "current_placeholder")
            self.assertEqual(current_placeholder["exclude_reason"], "current_placeholder")

    def test_surface_side_detail_tracks_side_quality_and_surface_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)

            workbook = merge_vol.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            side_df = workbook["surface_side_detail"]

            self.assertEqual(len(side_df), 6)

            current_back = side_df[(side_df["news_row_id"] == 1) & (side_df["side"] == "current_back")].iloc[0]
            self.assertEqual(current_back["source_direction"], "backward")
            self.assertTrue(bool(current_back["has_surface"]))
            self.assertEqual(int(current_back["surface_slice_count"]), 1)
            self.assertEqual(current_back["side_quality_label"], "usable")
            self.assertEqual(int(current_back["raw_point_pass_count"]), 2)
            self.assertAlmostEqual(float(current_back["exact_slice_point_ratio"]), 0.5, places=8)
            self.assertGreater(float(current_back["surface_max"]), float(current_back["surface_min"]))
            self.assertEqual(len(json.loads(current_back["surface_flat"])), 256)

            target_missing = side_df[(side_df["news_row_id"] == 2) & (side_df["side"] == "target_forward")].iloc[0]
            self.assertFalse(bool(target_missing["has_svi"]))
            self.assertEqual(target_missing["surface_flat"], "")
            self.assertEqual(target_missing["side_quality_label"], "no_svi")

            placeholder = side_df[(side_df["news_row_id"] == 3) & (side_df["side"] == "current_back")].iloc[0]
            self.assertEqual(int(placeholder["placeholder_flag"]), 1)
            self.assertEqual(int(placeholder["boundary_flag"]), 1)
            self.assertEqual(placeholder["side_quality_label"], "placeholder")

    def test_main_writes_expected_workbook_and_filters_gan_input_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)
            output_path = result_dir / "merged_vol.xlsx"

            with patch.object(merge_vol, "DEFAULT_NEWS_XLSX_PATH", xlsx_path):
                written_path = merge_vol.main(
                    [
                        "--input-dir",
                        str(result_dir),
                        "--source-timezone",
                        "America/New_York",
                    ]
                )

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())

            workbook = pd.ExcelFile(output_path, engine="openpyxl")
            self.assertEqual(
                workbook.sheet_names,
                ["news_surface_pair_audit", "surface_side_detail", "gan_input_ready"],
            )

            gan_ready = pd.read_excel(output_path, sheet_name="gan_input_ready", engine="openpyxl")
            self.assertEqual(len(gan_ready), 1)
            self.assertEqual(gan_ready.loc[0, "sample_id"], "news_1")
            self.assertEqual(gan_ready.loc[0, "current_snapshot_time_utc"], "2022-12-30T13:28:00Z")
            self.assertEqual(gan_ready.loc[0, "target_snapshot_time_utc"], "2022-12-30T13:33:00Z")
            self.assertEqual(int(gan_ready.loc[0, "training_candidate_flag"]), 1)

    def test_alignment_csv_replaces_exact_news_snapshot_keys_and_is_audited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interpolated_atm_iv = math.sqrt((0.24 ** 2 + 0.26 ** 2) / 2.0)
            xlsx_path, result_dir = self._write_model_payload_fixture_files(
                tmpdir,
                surface_model="raw",
                surface_params={
                    "business_days": [30],
                    "percent_strikes": [[0.9, 1.1]],
                    "implied_vols": [[0.24, 0.26]],
                },
                implied_vol=interpolated_atm_iv,
            )
            json_path = result_dir / "surface-raw-all.json"
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            sides = next(iter(payload.values()))
            sides["backward"]["snapshot_time_utc"] = "2022-12-30T13:40:00Z"
            sides["forward"]["snapshot_time_utc"] = "2022-12-30T13:45:00Z"
            json_path.write_text(
                json.dumps({"2022-12-30T13:40:00Z": sides}),
                encoding="utf-8",
            )
            csv_path = result_dir / "surface-raw-all-precalib-points.csv"
            points = pd.read_csv(csv_path)
            points.loc[0, "calibration_datetime_utc"] = "2022-12-30T13:40:00Z"
            points.loc[1, "calibration_datetime_utc"] = "2022-12-30T13:45:00Z"
            points.to_csv(csv_path, index=False)

            alignment_path = Path(tmpdir) / "alignment.csv"
            pd.DataFrame(
                [
                    {
                        "news_row_id": 1,
                        "has_match": 1,
                        "news_available_time_utc": "2022-12-30T13:28:00Z",
                        "effective_origin_utc": "2022-12-30T13:40:00Z",
                        "target_anchor_utc": "2022-12-30T13:45:00Z",
                        "origin_shift_minutes": 12,
                        "alignment_type": "intraday_shift",
                        "matching_rank": 1,
                        "collision_count": 1,
                        "news_cluster_id": "surface_pair_20221230T1340Z",
                        "current_window_start_utc": "2022-12-30T13:35:00Z",
                        "current_window_end_utc": "2022-12-30T13:40:00Z",
                        "target_window_start_utc": "2022-12-30T13:40:00Z",
                        "target_window_end_utc": "2022-12-30T13:45:00Z",
                        "original_news_quarter": "2022Q4",
                        "effective_origin_quarter": "2022Q4",
                    }
                ]
            ).to_csv(alignment_path, index=False)

            workbook = merge_vol.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
                alignment_csv_path=alignment_path,
            )
            audit = workbook["news_surface_pair_audit"].iloc[0]
            gan = workbook["gan_input_ready"].iloc[0]
            self.assertEqual(
                audit["current_snapshot_time_utc"],
                "2022-12-30T13:40:00Z",
            )
            self.assertEqual(
                audit["target_snapshot_time_utc"],
                "2022-12-30T13:45:00Z",
            )
            self.assertEqual(audit["alignment_type"], "intraday_shift")
            self.assertEqual(int(audit["origin_shift_minutes"]), 12)
            self.assertEqual(gan["news_alignment_mode"], "forward_valid_pair")
            self.assertEqual(
                gan["effective_origin_utc"],
                "2022-12-30T13:40:00Z",
            )

    def test_build_workbook_frames_supports_model_aware_sabr_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_model_payload_fixture_files(
                tmpdir,
                surface_model="sabr",
                surface_params={
                    "alpha": [0.2],
                    "beta": [1.0],
                    "rho": [0.0],
                    "nu": [0.1],
                    "business_days": [30],
                },
                implied_vol=0.2,
            )

            workbook = merge_vol.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_surface_pair_audit"]
            side = workbook["surface_side_detail"]

            self.assertEqual(audit.loc[0, "surface_model"], "sabr")
            self.assertEqual(audit.loc[0, "pair_quality_label"], "usable")
            self.assertEqual(int(audit.loc[0, "training_candidate_flag"]), 1)
            self.assertEqual(side.loc[0, "surface_model"], "sabr")
            self.assertTrue(bool(side.loc[0, "has_svi"]))
            self.assertTrue(bool(side.loc[0, "surface_param_json"]))

    def test_build_workbook_frames_supports_model_aware_cubic_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_model_payload_fixture_files(
                tmpdir,
                surface_model="cubic",
                surface_params={
                    "business_days": [30],
                    "percent_strikes": [[0.9, 1.0, 1.1, 1.2]],
                    "implied_vols": [[0.25, 0.25, 0.25, 0.25]],
                },
                implied_vol=0.25,
            )

            workbook = merge_vol.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_surface_pair_audit"]
            side = workbook["surface_side_detail"]

            self.assertEqual(audit.loc[0, "surface_model"], "cubic")
            self.assertEqual(audit.loc[0, "pair_quality_label"], "usable")
            self.assertEqual(side.loc[0, "surface_model"], "cubic")
            self.assertTrue(bool(side.loc[0, "surface_flat"]))

    def test_build_workbook_frames_supports_model_aware_raw_payload_with_interpolated_surface(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interpolated_atm_iv = math.sqrt((0.24 ** 2 + 0.26 ** 2) / 2.0)
            xlsx_path, result_dir = self._write_model_payload_fixture_files(
                tmpdir,
                surface_model="raw",
                surface_params={
                    "business_days": [30],
                    "percent_strikes": [[0.9, 1.1]],
                    "implied_vols": [[0.24, 0.26]],
                },
                implied_vol=interpolated_atm_iv,
            )

            workbook = merge_vol.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_surface_pair_audit"]
            side = workbook["surface_side_detail"]

            self.assertEqual(audit.loc[0, "surface_model"], "raw")
            self.assertEqual(audit.loc[0, "pair_quality_label"], "usable")
            self.assertTrue(bool(audit.loc[0, "current_has_surface"]))
            self.assertTrue(bool(audit.loc[0, "target_has_surface"]))
            self.assertEqual(side.loc[0, "surface_model"], "raw")
            self.assertTrue(bool(side.loc[0, "has_surface"]))

            surface_flat = json.loads(side.loc[0, "surface_flat"])
            self.assertEqual(len(surface_flat), 256)
            self.assertTrue(all(math.isfinite(value) for value in surface_flat))
            self.assertGreater(min(surface_flat), 0.0)


if __name__ == "__main__":
    unittest.main()
