from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.rq3 import news_first_vol_f4_film_pure_capacity_3seed_analysis as analysis


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _parameter_counts() -> dict[str, dict[str, dict[str, int]]]:
    payload: dict[str, dict[str, dict[str, int]]] = {
        architecture: {} for architecture in analysis.ARCHITECTURES
    }
    for index, capacity_id in enumerate(analysis.CAPACITY_IDS, start=1):
        critic = 1_000 * index
        pure_generator = 2_000 * index
        film_generator = 4_000 * index
        payload["film_cnn"][capacity_id] = {
            "generator_parameters": film_generator,
            "critic_parameters": critic,
            "total_wgan_parameters": film_generator + critic,
        }
        payload["pure_cnn"][capacity_id] = {
            "generator_parameters": pure_generator,
            "critic_parameters": critic,
            "total_wgan_parameters": pure_generator + critic,
        }
    return payload


def _pair_metrics(*, include_parameters: bool = True) -> pd.DataFrame:
    multipliers = {
        "c08": 1.10,
        "c12": 1.05,
        "c16": 1.01,
        "c24": 0.99,
        "c32": 1.00,
        "c48": 0.98,
    }
    references = {"film_cnn": 0.0020, "pure_cnn": 0.0025}
    parameters = _parameter_counts()
    rows: list[dict[str, object]] = []
    for architecture in analysis.ARCHITECTURES:
        for capacity_id in analysis.CAPACITY_IDS:
            for seed in analysis.SEEDS:
                for pair_index in range(analysis.PAIR_COUNT):
                    row: dict[str, object] = {
                        "architecture": architecture,
                        "capacity_id": capacity_id,
                        "seed": seed,
                        "pair_id": f"pair-{pair_index:03d}",
                        "session_id": f"session-{pair_index % 45:02d}",
                        "target_mae": references[architecture]
                        * multipliers[capacity_id],
                        "noise_bank_profile_sha256": _digest(f"noise-{seed}"),
                        "fold": analysis.FOLD_ID,
                        "tolerance_minutes": 5,
                        "prediction_mc_samples": 64,
                    }
                    if include_parameters:
                        row.update(parameters[architecture][capacity_id])
                    rows.append(row)
    return pd.DataFrame(rows)


class F4FilmPureCapacityAnalysisTests(unittest.TestCase):
    def test_summary_uses_architecture_specific_unrounded_c32_reference(self) -> None:
        summary = analysis.summarize_pair_metrics(_pair_metrics())

        self.assertEqual(summary["capacity_id"].tolist(), list(analysis.CAPACITY_IDS))
        self.assertEqual(
            summary["is_current_capacity"].tolist(),
            [False, False, False, False, True, False],
        )
        expected = {
            "c08": -10.0,
            "c12": -5.0,
            "c16": -1.0,
            "c24": 1.0,
            "c32": 0.0,
            "c48": 2.0,
        }
        for row in summary.itertuples(index=False):
            self.assertAlmostEqual(
                float(row.film_cnn_improvement_vs_c32_pct),
                expected[row.capacity_id],
                places=12,
            )
            self.assertAlmostEqual(
                float(row.pure_cnn_improvement_vs_c32_pct),
                expected[row.capacity_id],
                places=12,
            )
            self.assertEqual(int(row.film_cnn_seed_pair_rows), 429)
            self.assertEqual(int(row.pure_cnn_seed_pair_rows), 429)

        c32 = summary.loc[summary["capacity_id"].eq("c32")].iloc[0]
        self.assertAlmostEqual(float(c32["film_cnn_observed_mae"]), 0.0020)
        self.assertAlmostEqual(float(c32["pure_cnn_observed_mae"]), 0.0025)

    def test_explicit_parameter_contract_is_supported(self) -> None:
        summary = analysis.summarize_pair_metrics(
            _pair_metrics(include_parameters=False),
            parameter_counts=_parameter_counts(),
        )
        c48 = summary.loc[summary["capacity_id"].eq("c48")].iloc[0]
        self.assertEqual(int(c48["film_cnn_total_wgan_parameters"]), 30_000)
        self.assertEqual(int(c48["pure_cnn_total_wgan_parameters"]), 18_000)

    def test_latex_is_complete_single_panel_and_uses_frozen_formatting(self) -> None:
        latex = analysis.render_latex_table(
            analysis.summarize_pair_metrics(_pair_metrics())
        )

        self.assertTrue(latex.startswith("\\begin{table}[htbp]"))
        self.assertTrue(latex.endswith("\\end{table}\n"))
        self.assertIn(
            rf"\label{{{analysis.TABLE_LABEL}}}",
            latex,
        )
        self.assertIn(
            r"\shortstack[l]{\textbf{c32 (current}\\\textbf{capacity; reference)}}",
            latex,
        )
        self.assertIn(
            "matches spatial-network and Critic widths, not total",
            latex,
        )
        self.assertIn(
            "comparisons should use the MAE columns",
            latex,
        )
        self.assertIn("2023Q4 is the held-out test period", latex)
        self.assertIn("0.0020000000", latex)
        self.assertIn("0.0025000000", latex)
        self.assertIn("+2.0000", latex)
        self.assertIn("-10.0000", latex)
        self.assertIn("0.0000", latex)
        self.assertNotIn("Panel A", latex)
        self.assertNotIn("Panel B", latex)
        self.assertNotIn("Persistence", latex)
        self.assertNotIn("confidence interval", latex.lower())
        self.assertNotIn("p-value", latex.lower())
        self.assertNotIn("significance", latex.lower())
        self.assertNotIn("PENDING", latex)

    def test_postprocess_writes_deterministic_auditable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "frozen_input.csv.gz"
            _pair_metrics().to_csv(source, index=False, compression="gzip")

            paths = analysis.postprocess_experiment(root, source)
            expected = {
                "pair_metrics": root / "analysis" / analysis.PAIR_METRICS_NAME,
                "summary_csv": root / "analysis" / analysis.SUMMARY_CSV_NAME,
                "summary_json": root / "analysis" / analysis.SUMMARY_JSON_NAME,
                "latex_table": root / "analysis" / analysis.TABLE_TEX_NAME,
            }
            self.assertEqual(dict(paths), expected)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertEqual(
                len(pd.read_csv(paths["pair_metrics"])), analysis.EXPECTED_ROWS
            )
            self.assertEqual(len(pd.read_csv(paths["summary_csv"])), 6)

            payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], analysis.ANALYSIS_KIND)
            self.assertEqual(payload["pair_metric_rows"], 5_148)
            self.assertEqual(payload["pair_count"], 143)
            self.assertEqual(payload["session_count"], 45)
            self.assertEqual(payload["current_capacity_id"], "c32")
            self.assertEqual(len(payload["rows"]), 6)
            self.assertEqual(
                set(payload["artifacts"]),
                {
                    analysis.PAIR_METRICS_NAME,
                    analysis.SUMMARY_CSV_NAME,
                    analysis.TABLE_TEX_NAME,
                },
            )

            first_hashes = {
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in paths.items()
            }
            second_paths = analysis.postprocess_experiment(root, source)
            second_hashes = {
                key: hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in second_paths.items()
            }
            self.assertEqual(first_hashes, second_hashes)

    def test_rejects_incomplete_pair_matrix(self) -> None:
        frame = _pair_metrics().iloc[:-1].copy()
        with self.assertRaisesRegex(analysis.F4CapacityAnalysisError, "5,148"):
            analysis.summarize_pair_metrics(frame)

    def test_rejects_pair_panel_drift(self) -> None:
        frame = _pair_metrics()
        mask = (
            frame["architecture"].eq("film_cnn")
            & frame["capacity_id"].eq("c08")
            & frame["seed"].eq(42)
            & frame["pair_id"].eq("pair-000")
        )
        frame.loc[mask, "pair_id"] = "pair-rogue"
        with self.assertRaisesRegex(
            analysis.F4CapacityAnalysisError,
            "pair/session panel|Pair-panel drift",
        ):
            analysis.summarize_pair_metrics(frame)

    def test_rejects_mc64_noise_bank_drift(self) -> None:
        frame = _pair_metrics()
        mask = (
            frame["architecture"].eq("film_cnn")
            & frame["capacity_id"].eq("c08")
            & frame["seed"].eq(42)
        )
        frame.loc[mask, "noise_bank_profile_sha256"] = _digest("wrong-noise")
        with self.assertRaisesRegex(
            analysis.F4CapacityAnalysisError,
            "share one MC64 bank",
        ):
            analysis.summarize_pair_metrics(frame)

    def test_rejects_parameter_contract_drift(self) -> None:
        frame = _pair_metrics()
        frame.loc[
            frame["architecture"].eq("film_cnn") & frame["capacity_id"].eq("c08"),
            "total_wgan_parameters",
        ] += 1
        with self.assertRaisesRegex(analysis.F4CapacityAnalysisError, r"G\+D"):
            analysis.summarize_pair_metrics(frame)


if __name__ == "__main__":
    unittest.main()
