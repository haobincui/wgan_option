import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.rq3.event_study import (  # noqa: E402
    analyze_results,
    analyze_workbook,
    build_workbook_sample_metrics,
    label_rows_by_event,
    load_events,
    quality_audit,
    summarize_groups,
    write_event_template,
)
from scripts.rq3.main import main as rq3_main  # noqa: E402


def _serial(values):
    return json.dumps(values)


def _workbook_rows():
    strike_grid = [0.9, 1.0]
    maturity_grid = [30.0, 7.0]
    return pd.DataFrame(
        [
            {
                "sample_id": "news_1",
                "news_timestamp_utc": "2022-11-02T10:00:00Z",
                "current_surface_flat": _serial([0.10, 0.20, 0.30, 0.40]),
                "target_surface_flat": _serial([0.11, 0.22, 0.33, 0.44]),
                "strike_grid": _serial(strike_grid),
                "maturity_days_grid": _serial(maturity_grid),
                "pair_quality_label": "usable",
                "current_weighted_iv_rmse": 0.01,
                "target_weighted_iv_rmse": 0.02,
                "training_candidate_flag": 1,
            },
            {
                "sample_id": "news_2",
                "news_timestamp_utc": "2022-11-02T10:10:00Z",
                "current_surface_flat": _serial([0.20, 0.30, 0.40, 0.50]),
                "target_surface_flat": _serial([0.21, 0.34, 0.43, 0.70]),
                "strike_grid": _serial(strike_grid),
                "maturity_days_grid": _serial(maturity_grid),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
            {
                "sample_id": "news_3",
                "news_timestamp_utc": "2022-11-02T10:20:00Z",
                "current_surface_flat": _serial([0.30, 0.40, 0.50, 0.60]),
                "target_surface_flat": _serial([0.40, 0.50, 0.70, 0.90]),
                "strike_grid": _serial(strike_grid),
                "maturity_days_grid": _serial(maturity_grid),
                "pair_quality_label": "poor_fit",
                "training_candidate_flag": 1,
            },
            {
                "sample_id": "news_4",
                "news_timestamp_utc": "2022-11-02T10:40:00Z",
                "current_surface_flat": _serial([0.40, 0.50, 0.60, 0.70]),
                "target_surface_flat": _serial([0.45, 0.55, 0.65, 0.75]),
                "strike_grid": _serial(strike_grid),
                "maturity_days_grid": _serial(maturity_grid),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
            {
                "sample_id": "news_5",
                "news_timestamp_utc": "2022-11-02T11:00:00Z",
                "current_surface_flat": _serial([0.50, 0.60, 0.70, 0.80]),
                "target_surface_flat": _serial([0.55, 0.66, 0.77, 0.88]),
                "strike_grid": _serial(strike_grid),
                "maturity_days_grid": _serial(maturity_grid),
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            },
        ]
    )


def _events_frame(tmpdir: str) -> Path:
    events_path = Path(tmpdir) / "events.csv"
    pd.DataFrame(
        [
            {
                "event_id": "fomc-test",
                "event_time_utc": "2022-11-02T10:15:00Z",
                "event_name": "Synthetic FOMC",
                "event_type": "FOMC",
            }
        ]
    ).to_csv(events_path, index=False)
    return events_path


class TestRQ3EventStudy(unittest.TestCase):
    def test_event_window_labeling_and_vol_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = load_events(_events_frame(tmpdir))
            metrics = build_workbook_sample_metrics(_workbook_rows(), split="all", train_ratio=0.8)
            labeled = label_rows_by_event(metrics, events, window_minutes=10)

            self.assertEqual(labeled["event_group"].tolist(), ["quiet", "announcement", "announcement", "quiet", "quiet"])
            event_rows = labeled[labeled["is_announcement_window"]]
            self.assertEqual(event_rows["event_id"].tolist(), ["fomc-test", "fomc-test"])

            news_2 = labeled[labeled["sample_id"] == "news_2"].iloc[0]
            self.assertAlmostEqual(news_2["surface_jump_mae"], (0.01 + 0.04 + 0.03 + 0.20) / 4.0, places=6)
            self.assertAlmostEqual(news_2["atm_short_current_vol"], 0.50, places=6)
            self.assertAlmostEqual(news_2["atm_short_target_vol"], 0.70, places=6)
            self.assertAlmostEqual(news_2["atm_short_abs_jump"], 0.20, places=6)

            group_summary = summarize_groups(labeled)
            self.assertEqual(set(group_summary["event_group"].tolist()), {"announcement", "quiet"})
            announcement_summary = group_summary[group_summary["event_group"] == "announcement"].iloc[0]
            self.assertEqual(int(announcement_summary["sample_count"]), 2)

            audit = quality_audit(labeled)
            poor_fit_event = audit[
                (audit["event_group"] == "announcement") & (audit["pair_quality_label"] == "poor_fit")
            ].iloc[0]
            self.assertEqual(int(poor_fit_event["sample_count"]), 1)

    def test_chronological_val_split_uses_holdout_tail(self):
        metrics = build_workbook_sample_metrics(_workbook_rows(), split="val", train_ratio=0.8)

        self.assertEqual(metrics["sample_id"].tolist(), ["news_5"])
        self.assertEqual(metrics["split"].tolist(), ["val"])

    def test_analyze_workbook_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook_path = Path(tmpdir) / "merged_vol.xlsx"
            output_dir = Path(tmpdir) / "rq3"
            _workbook_rows().to_excel(workbook_path, sheet_name="gan_input_ready", index=False)

            result_dir = analyze_workbook(
                merged_vol_path=workbook_path,
                events_csv=_events_frame(tmpdir),
                output_dir=output_dir,
                split="all",
                train_ratio=0.8,
                window_minutes=10,
                save_plots=False,
            )

            self.assertEqual(result_dir, output_dir)
            self.assertTrue((output_dir / "rq3_labeled_samples.csv").exists())
            self.assertTrue((output_dir / "rq3_group_summary.csv").exists())
            self.assertTrue((output_dir / "rq3_event_summary.csv").exists())
            self.assertTrue((output_dir / "rq3_quality_audit.csv").exists())
            self.assertTrue((output_dir / "rq3_run_manifest.json").exists())
            labeled = pd.read_csv(output_dir / "rq3_labeled_samples.csv")
            self.assertEqual(int(labeled["is_announcement_window"].sum()), 2)

    def test_result_summary_group_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "summary.csv"
            output_dir = Path(tmpdir) / "rq3_result"
            pd.DataFrame(
                [
                    {
                        "sample_id": "news_1",
                        "news_timestamp_utc": "2022-11-02T10:10:00Z",
                        "mae": 0.4,
                        "current_mae": 0.5,
                        "mae_gap_vs_current": -0.1,
                        "win_flag_vs_current": 1.0,
                    },
                    {
                        "sample_id": "news_2",
                        "news_timestamp_utc": "2022-11-02T11:00:00Z",
                        "mae": 0.8,
                        "current_mae": 0.7,
                        "mae_gap_vs_current": 0.1,
                        "win_flag_vs_current": 0.0,
                    },
                ]
            ).to_csv(result_path, index=False)

            analyze_results(
                result_specs=[f"text={result_path}"],
                events_csv=_events_frame(tmpdir),
                output_dir=output_dir,
                window_minutes=10,
            )

            summary = pd.read_csv(output_dir / "rq3_result_group_metrics.csv")
            event_summary = summary[summary["event_group"] == "announcement"].iloc[0]
            quiet_summary = summary[summary["event_group"] == "quiet"].iloc[0]
            self.assertAlmostEqual(event_summary["mae_gap_vs_current_mean"], -0.1)
            self.assertAlmostEqual(quiet_summary["mae_gap_vs_current_mean"], 0.1)

    def test_missing_event_time_column_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad_events.csv"
            pd.DataFrame([{"event_id": "bad"}]).to_csv(bad_path, index=False)

            with self.assertRaisesRegex(ValueError, "event_time_utc"):
                load_events(bad_path)

    def test_write_event_template_and_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "template.csv"

            written = rq3_main(["write-event-template", "--output", str(output)])

            self.assertEqual(written, output)
            template = pd.read_csv(output)
            self.assertEqual(list(template.columns), ["event_id", "event_time_utc", "event_name", "event_type"])
            self.assertEqual(len(template), 0)


if __name__ == "__main__":
    unittest.main()
