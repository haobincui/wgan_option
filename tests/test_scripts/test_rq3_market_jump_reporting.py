import json
import re
import sys
import tempfile
import unittest
from html import unescape
from pathlib import Path

import nbformat
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.rq3.market_jump_reporting import (  # noqa: E402
    build_companion_notebook,
    generate_figures,
    render_notebook_html,
)


class TestRQ3MarketJumpReporting(unittest.TestCase):
    def _write_frame(self, root: Path, name: str, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(root / name, index=False, compression="gzip" if name.endswith(".gz") else None)

    def _build_archive(self, root: Path, *, include_bridges: bool = True) -> None:
        root.mkdir(parents=True, exist_ok=True)
        quality_rows = [
            {"check": "reconstructed_slice_count", "actual": 4, "expected": 4, "status": "pass"},
            {"check": "reconstructed_surface_point_count", "actual": 12, "expected": 12, "status": "pass"},
        ]
        self._write_frame(root, "data_quality_summary.csv", quality_rows)
        (root / "validation_summary.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "counts": {
                        "reconstructed_slice_count": 4,
                        "reconstructed_surface_point_count": 12,
                    },
                    "candidate_pair_count": 2,
                    "candidate_episode_count": 2,
                    "official_bridge_rows": 1 if include_bridges else 0,
                    "news_bridge_rows": 1 if include_bridges else 0,
                    "causal_claim_supported": False,
                }
            ),
            encoding="utf-8",
        )
        (root / "field_dictionary.json").write_text(
            json.dumps({"delta_atm_iv": "Target minus current common-strike IV."}),
            encoding="utf-8",
        )

        atm_rows = [
            {
                "slice_id": f"slice_{index}",
                "anchor_time_utc": f"2022-01-03T14:{minute:02d}:00Z",
                "maturity_date": "2022-03-25",
                "atm_quality": quality,
                "abs_moneyness_distance": distance,
                "atm_iv_decimal": 0.08 + index * 0.001,
            }
            for index, (minute, quality, distance) in enumerate(
                [(0, "A", 0.002), (5, "A", 0.008), (10, "B", 0.015), (15, "C", 0.03), (20, "outside_5pct", 0.06)]
            )
        ]
        self._write_frame(root, "atm_observations.csv.gz", atm_rows)

        point_rows = []
        for anchor, shift in (
            ("2022-01-03T14:00:00Z", 0.0),
            ("2022-01-03T14:05:00Z", 0.006),
            ("2022-02-10T15:00:00Z", 0.0),
            ("2022-02-10T15:05:00Z", -0.004),
        ):
            for strike, q, iv in ((126.0, 0.98, 0.091), (128.0, 0.995, 0.082), (130.0, 1.01, 0.078)):
                point_rows.append(
                    {
                        "anchor_time_utc": anchor,
                        "anchor_time_london": anchor,
                        "maturity_date": "2022-03-25",
                        "underlying_contract_id": "TYH2",
                        "strike": strike,
                        "strike_over_forward": q,
                        "log_moneyness": 0.0,
                        "implied_vol": iv + shift,
                        "implied_vol_pct": (iv + shift) * 100,
                        "weight_sum": 10,
                    }
                )
        self._write_frame(root, "all_surface_points.csv.gz", point_rows)

        metric_rows = [
            {
                "slice_pair_id": "slice_pair_1",
                "pair_id": "pair_1",
                "origin_time_utc": "2022-01-03T14:00:00Z",
                "target_time_utc": "2022-01-03T14:05:00Z",
                "maturity_date": "2022-03-25",
                "underlying_contract_id": "TYH2",
                "pair_atm_strike": 128.0,
                "pair_atm_quality": "A",
                "delta_atm_iv": 0.006,
                "abs_delta_atm_iv": 0.006,
                "delta_atm_iv_skew_secant": 0.12,
                "abs_delta_atm_iv_skew_secant": 0.12,
                "rolling_jump_skew_60m": 0.7,
                "delta_rolling_jump_skew_60m": 0.3,
            },
            {
                "slice_pair_id": "slice_pair_2",
                "pair_id": "pair_2",
                "origin_time_utc": "2022-02-10T15:00:00Z",
                "target_time_utc": "2022-02-10T15:05:00Z",
                "maturity_date": "2022-03-25",
                "underlying_contract_id": "TYH2",
                "pair_atm_strike": 128.0,
                "pair_atm_quality": "B",
                "delta_atm_iv": -0.004,
                "abs_delta_atm_iv": 0.004,
                "delta_atm_iv_skew_secant": -0.08,
                "abs_delta_atm_iv_skew_secant": 0.08,
                "rolling_jump_skew_60m": -0.4,
                "delta_rolling_jump_skew_60m": -0.8,
            },
        ]
        self._write_frame(root, "pair_slice_metrics.csv.gz", metric_rows)

        ranking_rows = [
            {
                "slice_pair_id": "slice_pair_1",
                "pair_id": "pair_1",
                "origin_time_utc": "2022-01-03T14:00:00Z",
                "maturity_bucket": "22_63bd",
                "metric_name": "atm_iv_jump",
                "anomaly_tier": "high",
                "anomaly_tier_order": 3,
                "abs_robust_z": 7.2,
                "abs_empirical_percentile": 1.0,
                "abs_metric_change": 0.006,
            },
            {
                "slice_pair_id": "slice_pair_1",
                "pair_id": "pair_1",
                "origin_time_utc": "2022-01-03T14:00:00Z",
                "maturity_bucket": "22_63bd",
                "metric_name": "smile_skew_jump",
                "anomaly_tier": "primary",
                "anomaly_tier_order": 2,
                "abs_robust_z": 5.4,
                "abs_empirical_percentile": 0.998,
                "abs_metric_change": 0.12,
            },
            {
                "slice_pair_id": "slice_pair_2",
                "pair_id": "pair_2",
                "origin_time_utc": "2022-02-10T15:00:00Z",
                "maturity_bucket": "22_63bd",
                "metric_name": "rolling_jump_skew_change",
                "anomaly_tier": "broad",
                "anomaly_tier_order": 1,
                "abs_robust_z": 4.3,
                "abs_empirical_percentile": 0.992,
                "metric_change": -0.8,
                "abs_metric_change": 0.8,
            },
        ]
        self._write_frame(root, "metric_rankings.csv.gz", ranking_rows)

        pair_rows = [
            {
                "pair_id": "pair_1",
                "origin_time_utc": "2022-01-03T14:00:00Z",
                "target_time_utc": "2022-01-03T14:05:00Z",
                "session_id": "2022-01-03",
                "anomaly_tier": "high",
                "anomaly_tier_order": 3,
                "max_abs_robust_z": 7.2,
                "max_abs_empirical_percentile": 1.0,
                "peak_metric_name": "atm_iv_jump",
                "peak_maturity_date": "2022-03-25",
            },
            {
                "pair_id": "pair_2",
                "origin_time_utc": "2022-02-10T15:00:00Z",
                "target_time_utc": "2022-02-10T15:05:00Z",
                "session_id": "2022-02-10",
                "anomaly_tier": "broad",
                "anomaly_tier_order": 1,
                "max_abs_robust_z": 4.3,
                "max_abs_empirical_percentile": 0.992,
                "peak_metric_name": "rolling_jump_skew_change",
                "peak_maturity_date": "2022-03-25",
            },
        ]
        self._write_frame(root, "market_pair_rankings.csv.gz", pair_rows)
        self._write_frame(root, "candidate_pairs.csv", pair_rows)

        episode_rows = [
            {
                "episode_id": "episode_1",
                "episode_rank": 1,
                "episode_start_utc": "2022-01-03T14:00:00Z",
                "episode_end_utc": "2022-01-03T14:05:00Z",
                "anomaly_tier": "high",
                "peak_pair_id": "pair_1",
                "peak_metric_name": "atm_iv_jump",
                "max_abs_robust_z": 7.2,
                "market_pair_count": 1,
            },
            {
                "episode_id": "episode_2",
                "episode_rank": 2,
                "episode_start_utc": "2022-02-10T15:00:00Z",
                "episode_end_utc": "2022-02-10T15:05:00Z",
                "anomaly_tier": "broad",
                "peak_pair_id": "pair_2",
                "peak_metric_name": "rolling_jump_skew_change",
                "max_abs_robust_z": 4.3,
                "market_pair_count": 1,
            },
        ]
        self._write_frame(root, "candidate_episodes.csv", episode_rows)
        self._write_frame(
            root,
            "episode_pair_members.csv",
            [
                {"episode_id": "episode_1", "pair_id": "pair_1"},
                {"episode_id": "episode_2", "pair_id": "pair_2"},
            ],
        )

        if include_bridges:
            self._write_frame(
                root,
                "episode_official_event_bridge.csv",
                [
                    {
                        "episode_id": "episode_1",
                        "window_relation": "impact",
                        "event_id": "cpi_1",
                        "event_name": "CPI",
                        "event_type": "CPI",
                        "release_time_utc": "2022-01-03T14:00:00Z",
                    }
                ],
            )
            self._write_frame(
                root,
                "episode_news_bridge.csv",
                [
                    {
                        "episode_id": "episode_2",
                        "window_relation": "post_reporting",
                        "news_available_time_utc": "2022-02-10T15:06:00Z",
                        "headline": "Rates move after release",
                        "article_id": "A1",
                        "alignment_match_method": "exact",
                    }
                ],
            )

    def test_generate_figures_writes_png_svg_manifest_and_case_smiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_archive(root)

            outputs = generate_figures(root, max_case_events=2)

            self.assertGreaterEqual(len(outputs), 16)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertEqual({path.suffix for path in outputs}, {".png", ".svg"})
            manifest = json.loads((root / "figures" / "chart_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["formal_inputs_only"])
            self.assertIn("no red-green semantics", manifest["palette_policy"])
            self.assertEqual(len([chart for chart in manifest["charts"] if chart["id"].startswith("case_smile_")]), 2)
            rolling = next(chart for chart in manifest["charts"] if chart["id"] == "rolling_skew_diagnostic")
            self.assertEqual(rolling["unit"], "dimensionless Fisher-Pearson skewness")
            self.assertEqual(rolling["flagged_rows"], 1)
            self.assertTrue((root / "figures" / "06_rolling_skew_diagnostic.svg").is_file())
            self.assertTrue((root / "figures" / "07_metric_change_distributions.svg").is_file())

    def test_missing_optional_evidence_bridges_still_yields_coverage_figure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_archive(root, include_bridges=False)

            generate_figures(root, max_case_events=0)

            self.assertTrue((root / "figures" / "05_episode_evidence_coverage.png").is_file())
            manifest = json.loads((root / "figures" / "chart_manifest.json").read_text(encoding="utf-8"))
            evidence = next(chart for chart in manifest["charts"] if chart["id"] == "episode_evidence_coverage")
            self.assertEqual(evidence["counts"], {"Both": 0, "Official only": 0, "Factiva only": 0, "Neither": 2})

    def test_notebook_executes_and_html_embeds_figures_without_remote_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_archive(root)
            generate_figures(root, max_case_events=1)

            notebook_path = build_companion_notebook(root)
            notebook = nbformat.read(notebook_path, as_version=4)
            headings = "\n".join(
                cell.source for cell in notebook.cells if cell.cell_type == "markdown"
            )
            expected_order = ["## tl;dr", "## Context & Methods", "## Data", "## Results", "## Takeaways"]
            offsets = [headings.index(heading) for heading in expected_order]
            self.assertEqual(offsets, sorted(offsets))
            code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
            self.assertNotIn("market_surface_index.sqlite", code)
            self.assertNotIn("market_jump_detection", code)

            html_path = render_notebook_html(notebook_path, root / "report.html", timeout_seconds=120)

            executed = nbformat.read(notebook_path, as_version=4)
            self.assertTrue(any(cell.get("outputs") for cell in executed.cells if cell.cell_type == "code"))
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("<title>TY 期权近似 ATM Vol", html)
            self.assertIn("TY 期权近似 ATM Vol", unescape(html))
            self.assertIn("data:image/png;base64", html)
            self.assertIsNone(re.search(r'''(?:src|href)=["']https?://''', html, flags=re.IGNORECASE))

    def test_incomplete_archive_fails_with_named_formal_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(FileNotFoundError, "data_quality_summary.csv"):
                generate_figures(Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
