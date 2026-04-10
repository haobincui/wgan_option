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

from scripts.merge_file import merge_params  # noqa: E402


def _model_iv(total_var: float, business_days: int) -> float:
    return math.sqrt(total_var / (business_days / merge_params.DAYS_IN_YEAR))


class TestMergeParams(unittest.TestCase):
    def test_parse_args_accepts_explicit_news_workbook(self):
        args = merge_params._parse_args(
            [
                "--input-dir",
                "data/processed/example",
                "--news-xlsx",
                "data/raw/news.xlsx",
                "--source-timezone",
                "UTC",
                "--offset-minutes",
                "15",
            ]
        )

        self.assertEqual(args.input_dir, "data/processed/example")
        self.assertEqual(args.news_xlsx, "data/raw/news.xlsx")
        self.assertEqual(args.source_timezone, "UTC")
        self.assertEqual(args.offset_minutes, 15)

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

    def _write_fixture_files(self, tmpdir: str):
        xlsx_path = Path(tmpdir) / "news_input.xlsx"
        result_dir = Path(tmpdir) / "result"
        result_dir.mkdir()
        csv_path = result_dir / "minute_svi_precalib_points.csv"
        json_path = result_dir / "minute_svi_params.json"

        pd.DataFrame(
            [
                {
                    "SourceFile": "svi.txt",
                    "ArticleID": 1,
                    "HD": "headline svi",
                    "LP": "lead svi",
                    "PD": "2022-12-30",
                    "ET": "08:28:00",
                    "HD_embedding": "[0.1, 0.2]",
                    "LP_embedding": "[0.3, 0.4]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
                {
                    "SourceFile": "sabr.txt",
                    "ArticleID": 2,
                    "HD": "headline sabr",
                    "LP": "lead sabr",
                    "PD": "2022-12-30",
                    "ET": "08:39:00",
                    "HD_embedding": "[0.5, 0.6]",
                    "LP_embedding": "[0.7, 0.8]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
                {
                    "SourceFile": "cubic.txt",
                    "ArticleID": 3,
                    "HD": "headline cubic",
                    "LP": "lead cubic",
                    "PD": "2022-12-30",
                    "ET": "08:50:00",
                    "HD_embedding": "[0.9, 1.0]",
                    "LP_embedding": "[1.1, 1.2]",
                    "HD_dim": 2,
                    "LP_dim": 2,
                },
            ]
        ).to_excel(xlsx_path, index=False, engine="openpyxl")

        svi_iv = _model_iv(0.001, 30)
        csv_rows = [
            {
                "trade_datetime_utc": "2022-12-30T13:28:00Z",
                "calibration_datetime_utc": "2022-12-30T13:28:00Z",
                "business_days": 30,
                "maturity_date": "2023-02-01",
                "contract_id": "SVI_B",
                "option_type": "P",
                "strike": 110.0,
                "price": 0.25,
                "spot": 110.0,
                "percent_strike": 1.0,
                "implied_vol": svi_iv,
                "passes_precalib_filter": True,
                "filter_reason": "",
                "weight": 1.0,
            },
            {
                "trade_datetime_utc": "2022-12-30T13:33:00Z",
                "calibration_datetime_utc": "2022-12-30T13:33:00Z",
                "business_days": 30,
                "maturity_date": "2023-02-01",
                "contract_id": "SVI_F",
                "option_type": "C",
                "strike": 110.0,
                "price": 0.30,
                "spot": 110.0,
                "percent_strike": 1.0,
                "implied_vol": svi_iv,
                "passes_precalib_filter": True,
                "filter_reason": "",
                "weight": 1.0,
            },
            {
                "trade_datetime_utc": "2022-12-30T13:39:00Z",
                "calibration_datetime_utc": "2022-12-30T13:39:00Z",
                "business_days": 30,
                "maturity_date": "2023-02-01",
                "contract_id": "SABR_B",
                "option_type": "P",
                "strike": 110.0,
                "price": 0.20,
                "spot": 110.0,
                "percent_strike": 1.0,
                "implied_vol": 0.2,
                "passes_precalib_filter": True,
                "filter_reason": "",
                "weight": 1.0,
            },
            {
                "trade_datetime_utc": "2022-12-30T13:44:00Z",
                "calibration_datetime_utc": "2022-12-30T13:44:00Z",
                "business_days": 30,
                "maturity_date": "2023-02-01",
                "contract_id": "SABR_F",
                "option_type": "C",
                "strike": 110.0,
                "price": 0.21,
                "spot": 110.0,
                "percent_strike": 1.0,
                "implied_vol": 0.2,
                "passes_precalib_filter": True,
                "filter_reason": "",
                "weight": 1.0,
            },
            {
                "trade_datetime_utc": "2022-12-30T13:50:00Z",
                "calibration_datetime_utc": "2022-12-30T13:50:00Z",
                "business_days": 30,
                "maturity_date": "2023-02-01",
                "contract_id": "CUBIC_B",
                "option_type": "P",
                "strike": 110.0,
                "price": 0.22,
                "spot": 110.0,
                "percent_strike": 1.0,
                "implied_vol": 0.25,
                "passes_precalib_filter": True,
                "filter_reason": "",
                "weight": 1.0,
            },
            {
                "trade_datetime_utc": "2022-12-30T13:55:00Z",
                "calibration_datetime_utc": "2022-12-30T13:55:00Z",
                "business_days": 30,
                "maturity_date": "2023-02-01",
                "contract_id": "CUBIC_F",
                "option_type": "C",
                "strike": 110.0,
                "price": 0.23,
                "spot": 110.0,
                "percent_strike": 1.0,
                "implied_vol": 0.25,
                "passes_precalib_filter": True,
                "filter_reason": "",
                "weight": 1.0,
            },
        ]
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

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
                        "a": [0.0005],
                        "b": [0.01],
                        "rho": [0.0],
                        "m": [0.0],
                        "sigma": [0.05],
                        "business_days": [30],
                    },
                },
            },
            "2022-12-30T13:39:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:39:00Z",
                    "surface_model": "sabr",
                    "surface_params": {
                        "alpha": [0.2],
                        "beta": [1.0],
                        "rho": [0.0],
                        "nu": [0.1],
                        "business_days": [30],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:44:00Z",
                    "surface_model": "sabr",
                    "surface_params": {
                        "alpha": [0.2],
                        "beta": [1.0],
                        "rho": [0.0],
                        "nu": [0.1],
                        "business_days": [30],
                    },
                },
            },
            "2022-12-30T13:50:00Z": {
                "backward": {
                    "snapshot_time_utc": "2022-12-30T13:50:00Z",
                    "surface_model": "cubic",
                    "surface_params": {
                        "business_days": [30],
                        "percent_strikes": [[0.9, 1.0, 1.1, 1.2]],
                        "implied_vols": [[0.25, 0.25, 0.25, 0.25]],
                    },
                },
                "forward": {
                    "snapshot_time_utc": "2022-12-30T13:55:00Z",
                    "surface_model": "cubic",
                    "surface_params": {
                        "business_days": [30],
                        "percent_strikes": [[0.9, 1.0, 1.1, 1.2]],
                        "implied_vols": [[0.25, 0.25, 0.25, 0.25]],
                    },
                },
            },
        }
        json_path.write_text(json.dumps(json_payload), encoding="utf-8")
        return xlsx_path, result_dir

    def test_build_workbook_frames_emits_model_neutral_audit_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)

            workbook = merge_params.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_direction_audit"]
            slice_df = workbook["surface_slice_detail"]

            self.assertEqual(len(audit), 6)
            self.assertEqual(set(audit["surface_model"]), {"svi", "sabr", "cubic"})
            self.assertTrue((audit["training_candidate_flag"] == 1).all())
            self.assertTrue((audit["fit_quality_label"] == "usable").all())
            self.assertEqual(len(slice_df), 6)
            self.assertEqual(set(slice_df["surface_model"]), {"svi", "sabr", "cubic"})
            self.assertTrue(slice_df["slice_param_json"].astype(str).str.len().gt(0).all())

    def test_main_writes_expected_workbook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_fixture_files(tmpdir)
            output_path = result_dir / "merged_params.xlsx"

            with patch.object(merge_params, "DEFAULT_NEWS_XLSX_PATH", xlsx_path):
                written_path = merge_params.main(["--input-dir", str(result_dir)])

            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())

            workbook = pd.ExcelFile(output_path, engine="openpyxl")
            self.assertEqual(
                workbook.sheet_names,
                ["news_direction_audit", "surface_slice_detail", "gan_input_ready"],
            )

            gan_ready = pd.read_excel(output_path, sheet_name="gan_input_ready", engine="openpyxl")
            self.assertEqual(len(gan_ready), 6)
            self.assertEqual(set(gan_ready["surface_model"]), {"svi", "sabr", "cubic"})

    def test_build_workbook_frames_supports_model_aware_raw_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path, result_dir = self._write_model_payload_fixture_files(
                tmpdir,
                surface_model="raw",
                surface_params={
                    "business_days": [30],
                    "percent_strikes": [[0.9, 1.1]],
                    "implied_vols": [[0.24, 0.26]],
                },
                implied_vol=0.25,
            )

            workbook = merge_params.build_workbook_frames(
                result_dir,
                news_xlsx_path=xlsx_path,
                source_timezone="America/New_York",
                offset_minutes=5,
            )
            audit = workbook["news_direction_audit"]
            slice_df = workbook["surface_slice_detail"]

            self.assertEqual(set(audit["surface_model"]), {"raw"})
            self.assertEqual(set(slice_df["surface_model"]), {"raw"})
            self.assertEqual(audit.loc[0, "fit_quality_label"], "usable")
            self.assertIn("percent_strikes", slice_df.loc[0, "slice_param_json"])


if __name__ == "__main__":
    unittest.main()
