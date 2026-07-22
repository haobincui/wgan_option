import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from film_wgan.config import FilmWGANSampleConfig  # noqa: E402
from film_wgan.data import load_film_wgan_samples  # noqa: E402
from scripts.rq3.news_quiet import (  # noqa: E402
    NEWS_EVENT_GROUP,
    QUIET_GROUP,
    analyze_news_quiet_results,
    analyze_news_quiet_workbook,
    build_news_quiet_workbook_from_window,
    build_news_quiet_workbook,
    prepare_news_quiet_targets,
)


def _serial(values):
    return json.dumps(values)


def _raw_surface(vol_shift=0.0):
    return {
        "surface_model": "raw",
        "surface_params": {
            "business_days": [7, 30],
            "percent_strikes": [[0.9, 1.0], [0.9, 1.0]],
            "implied_vols": [
                [0.20 + vol_shift, 0.21 + vol_shift],
                [0.22 + vol_shift, 0.23 + vol_shift],
            ],
        },
    }


def _source_workbook(path: Path):
    frame = pd.DataFrame(
        [
            {
                "sample_id": "news_1",
                "news_timestamp_utc": "2022-01-01T05:00:00Z",
                "current_snapshot_time_utc": "2022-01-01T05:00:00Z",
                "target_snapshot_time_utc": "2022-01-01T05:05:00Z",
                "surface_model": "raw",
                "hd_embedding": _serial([0.1, 0.2, 0.3]),
                "lp_embedding": _serial([0.4, 0.5]),
                "hd_dim": 3,
                "lp_dim": 2,
                "bow_embedding": _serial([0.6, 0.7, 0.8, 0.9]),
                "bow_dim": 4,
                "sentiment_embedding": _serial([1.0, 2.0, 3.0]),
                "sentiment_dim": 3,
                "strike_grid": _serial([0.9, 1.0]),
                "maturity_days_grid": _serial([7.0, 30.0]),
                "surface_shape": _serial([2, 2]),
                "current_surface_flat": _serial([0.20, 0.21, 0.22, 0.23]),
                "target_surface_flat": _serial([0.21, 0.22, 0.23, 0.24]),
                "current_weighted_iv_rmse": 0.01,
                "target_weighted_iv_rmse": 0.01,
                "pair_quality_label": "usable",
                "training_candidate_flag": 1,
            }
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="gan_input_ready", index=False)


def _news_xlsx(path: Path):
    pd.DataFrame(
        [
            {
                "SourceFile": "test.xlsx",
                "ArticleID": "a1",
                "PD": "2022-01-01",
                "ET": "00:00:00",
                "HD": "headline",
                "LP": "lead paragraph",
                "HD_embedding": _serial([0.1, 0.2, 0.3]),
                "LP_embedding": _serial([0.4, 0.5]),
                "HD_dim": 3,
                "LP_dim": 2,
            }
        ]
    ).to_excel(path, index=False)


def _surface_all_json(path: Path):
    payload = {
        "2022-01-01T05:00:00Z": _raw_surface(0.00),
        "2022-01-01T05:05:00Z": _raw_surface(0.01),
        "2022-01-01T07:00:00Z": _raw_surface(0.02),
        "2022-01-01T07:05:00Z": _raw_surface(0.03),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _surface_window_json(path: Path):
    payload = {
        "2022-01-01T07:00:00Z": {
            "backward": {"snapshot_time_utc": "2022-01-01T07:00:00Z", **_raw_surface(0.02)},
            "forward": {"snapshot_time_utc": "2022-01-01T07:05:00Z", **_raw_surface(0.03)},
        },
        "2022-01-01T08:00:00Z": {
            "backward": {"snapshot_time_utc": "2022-01-01T08:00:00Z", **_raw_surface(0.04)},
            "forward": {"snapshot_time_utc": "2022-01-01T08:05:00Z", **_raw_surface(0.05)},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestRQ3NewsQuiet(unittest.TestCase):
    def test_prepare_news_quiet_targets_samples_buffered_grid_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "merged_vol_rq2_text.xlsx"
            news = tmp / "news.xlsx"
            output = tmp / "targets"
            _source_workbook(source)
            _news_xlsx(news)
            frame = pd.read_excel(source, sheet_name="gan_input_ready")
            later = frame.iloc[0].copy()
            later["sample_id"] = "news_2"
            later["news_timestamp_utc"] = "2022-01-01T08:00:00Z"
            later["current_snapshot_time_utc"] = "2022-01-01T08:00:00Z"
            later["target_snapshot_time_utc"] = "2022-01-01T08:05:00Z"
            frame = pd.concat([frame, pd.DataFrame([later])], ignore_index=True)
            with pd.ExcelWriter(source, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="gan_input_ready", index=False)

            prepare_news_quiet_targets(
                source_merged_vol=source,
                news_xlsx=news,
                output_dir=output,
                candidate_count=3,
                sample_seed=7,
                quiet_buffer_minutes=60,
            )

            targets = [line.strip() for line in (output / "quiet_targets.txt").read_text().splitlines() if line.strip()]
            self.assertEqual(len(targets), 3)
            self.assertEqual(targets, sorted(targets))
            for value in targets:
                stamp = pd.Timestamp(value)
                self.assertEqual(stamp.minute % 5, 0)
                self.assertGreaterEqual((stamp - pd.Timestamp("2022-01-01T05:00:00Z")).total_seconds() / 60, 60)
            audit = pd.read_csv(output / "quiet_target_audit.csv")
            self.assertEqual(int(audit["selected_for_generation"].sum()), 3)

    def test_build_news_quiet_workbook_selects_quiet_and_zeroes_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "merged_vol_rq2_text.xlsx"
            news = tmp / "news.xlsx"
            surface = tmp / "surface-raw-all.json"
            output = tmp / "news_quiet.xlsx"
            _source_workbook(source)
            _news_xlsx(news)
            _surface_all_json(surface)

            build_news_quiet_workbook(
                source_merged_vol=source,
                surface_all_json=surface,
                news_xlsx=news,
                output_workbook=output,
                quiet_buffer_minutes=60,
            )

            result = pd.read_excel(output, sheet_name="gan_input_ready")
            self.assertEqual(set(result["event_group"]), {NEWS_EVENT_GROUP, QUIET_GROUP})
            quiet = result[result["event_group"] == QUIET_GROUP].iloc[0]
            self.assertEqual(int(quiet["has_news"]), 0)
            self.assertEqual(json.loads(quiet["lp_embedding"]), [0.0, 0.0])
            self.assertEqual(json.loads(quiet["bow_embedding"]), [0.0, 0.0, 0.0, 0.0])
            audit = pd.read_excel(output, sheet_name="news_quiet_audit")
            selected = audit[audit["selected"] == 1]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected.iloc[0]["candidate_timestamp_utc"], "2022-01-01T07:00:00Z")

    def test_build_news_quiet_workbook_from_window_selects_quiet_and_zeroes_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "merged_vol_rq2_text.xlsx"
            surface = tmp / "surface-raw-window.json"
            output = tmp / "news_quiet.xlsx"
            _source_workbook(source)
            _surface_window_json(surface)

            build_news_quiet_workbook_from_window(
                source_merged_vol=source,
                window_surface_json=surface,
                output_workbook=output,
                quiet_max_samples=1,
                quiet_sample_seed=11,
            )

            result = pd.read_excel(output, sheet_name="gan_input_ready")
            self.assertEqual(set(result["event_group"]), {NEWS_EVENT_GROUP, QUIET_GROUP})
            quiet = result[result["event_group"] == QUIET_GROUP].iloc[0]
            self.assertEqual(int(quiet["has_news"]), 0)
            self.assertEqual(json.loads(quiet["lp_embedding"]), [0.0, 0.0])
            self.assertEqual(json.loads(quiet["bow_embedding"]), [0.0, 0.0, 0.0, 0.0])
            audit = pd.read_excel(output, sheet_name="news_quiet_audit")
            self.assertEqual(int(audit["selected"].sum()), 1)

    def test_build_news_quiet_workbook_from_window_fails_when_valid_quiet_is_too_small(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "merged_vol_rq2_text.xlsx"
            surface = tmp / "surface-raw-window.json"
            output = tmp / "news_quiet.xlsx"
            _source_workbook(source)
            _surface_window_json(surface)

            with self.assertRaisesRegex(ValueError, "quiet_max_samples"):
                build_news_quiet_workbook_from_window(
                    source_merged_vol=source,
                    window_surface_json=surface,
                    output_workbook=output,
                    quiet_max_samples=3,
                )

    def test_window_payload_is_rejected_for_quiet_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "merged_vol_rq2_text.xlsx"
            news = tmp / "news.xlsx"
            surface = tmp / "surface-raw-excel.json"
            _source_workbook(source)
            _news_xlsx(news)
            surface.write_text(
                json.dumps({"2022-01-01T05:00:00Z": {"backward": _raw_surface(), "forward": _raw_surface()}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "data_range=all"):
                build_news_quiet_workbook(
                    source_merged_vol=source,
                    surface_all_json=surface,
                    news_xlsx=news,
                    output_workbook=tmp / "out.xlsx",
                )

    def test_news_quiet_workbook_analysis_outputs_group_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "merged_vol_rq2_text.xlsx"
            news = tmp / "news.xlsx"
            surface = tmp / "surface-raw-all.json"
            workbook = tmp / "news_quiet.xlsx"
            output = tmp / "analysis"
            _source_workbook(source)
            _news_xlsx(news)
            _surface_all_json(surface)
            build_news_quiet_workbook(
                source_merged_vol=source,
                surface_all_json=surface,
                news_xlsx=news,
                output_workbook=workbook,
            )

            analyze_news_quiet_workbook(workbook_path=workbook, output_dir=output)

            self.assertTrue((output / "rq3_news_quiet_group_summary.csv").exists())
            self.assertTrue((output / "rq3_news_quiet_tests.csv").exists())
            summary = pd.read_csv(output / "rq3_news_quiet_group_summary.csv")
            self.assertEqual(set(summary["event_group"]), {NEWS_EVENT_GROUP, QUIET_GROUP})

    def test_news_quiet_result_did_direction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            text = tmp / "text.csv"
            baseline = tmp / "no_text.csv"
            output = tmp / "result"
            rows = [
                {"sample_id": "n1", "global_index": 0, "news_timestamp_utc": "2022-01-01T05:00:00Z", "event_group": NEWS_EVENT_GROUP},
                {"sample_id": "n2", "global_index": 1, "news_timestamp_utc": "2022-01-01T05:05:00Z", "event_group": NEWS_EVENT_GROUP},
                {"sample_id": "q1", "global_index": 2, "news_timestamp_utc": "2022-01-01T07:00:00Z", "event_group": QUIET_GROUP},
                {"sample_id": "q2", "global_index": 3, "news_timestamp_utc": "2022-01-01T07:05:00Z", "event_group": QUIET_GROUP},
            ]
            text_rows = []
            base_rows = []
            for idx, row in enumerate(rows):
                text_mae = 0.10 if row["event_group"] == NEWS_EVENT_GROUP else 0.20
                base_mae = 0.15 if row["event_group"] == NEWS_EVENT_GROUP else 0.21
                common = {**row, "short_atm_weighted_mae": text_mae, "atm_short_pure_mae": text_mae}
                text_rows.append({**common, "mae": text_mae})
                common_base = {**row, "short_atm_weighted_mae": base_mae, "atm_short_pure_mae": base_mae}
                base_rows.append({**common_base, "mae": base_mae})
            pd.DataFrame(text_rows).to_csv(text, index=False)
            pd.DataFrame(base_rows).to_csv(baseline, index=False)

            analyze_news_quiet_results(
                result_specs=[f"text_seed={text}", f"no_text_seed={baseline}"],
                output_dir=output,
                text_label="text_seed",
            )

            did = pd.read_csv(output / "rq3_news_quiet_text_advantage_did.csv")
            surface = did[did["metric"] == "surface_mae"].iloc[0]
            self.assertGreater(surface["did_mean"], 0.0)
            pairwise = pd.read_csv(output / "rq3_news_quiet_text_vs_baselines.csv")
            event_surface = pairwise[
                (pairwise["event_group"] == NEWS_EVENT_GROUP) & (pairwise["metric"] == "surface_mae")
            ].iloc[0]
            self.assertGreater(event_surface["baseline_minus_text_mean"], 0.0)

    def test_film_wgan_loader_preserves_news_quiet_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook = Path(tmpdir) / "merged_vol.xlsx"
            _source_workbook(workbook)
            frame = pd.read_excel(workbook, sheet_name="gan_input_ready")
            frame["event_group"] = NEWS_EVENT_GROUP
            frame["has_news"] = 1
            frame["news_cluster_id"] = "news_1"
            frame["quiet_buffer_minutes"] = 60
            frame["quiet_grid_minutes"] = 5
            with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
                frame.to_excel(writer, sheet_name="gan_input_ready", index=False)

            samples = load_film_wgan_samples(FilmWGANSampleConfig(data_path=str(workbook), text_embedding_mode="lp"))

            self.assertEqual(samples[0].metadata["event_group"], NEWS_EVENT_GROUP)
            self.assertEqual(int(samples[0].metadata["has_news"]), 1)
            self.assertEqual(samples[0].metadata["news_cluster_id"], "news_1")


if __name__ == "__main__":
    unittest.main()
