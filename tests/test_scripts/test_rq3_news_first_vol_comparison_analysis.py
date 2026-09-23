from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.rq3.news_first_vol_comparison_analysis import (
    MAE_NUMERICAL_TIE_TOLERANCE,
    ComparisonAnalysisError,
    GridInterpolationError,
    RunSpec,
    TrainedRunEvaluator,
    _experiment_run_specs,
    _load_panel_source,
    aggregate_pair_metrics,
    build_bootstrap_comparisons,
    build_model_comparison,
    build_text_ablation_comparisons,
    cme_session_cluster_bootstrap,
    compute_maturity_metrics,
    compute_sample_metrics,
    discover_run_spec,
    holm_adjust,
    run_analysis,
    select_primary_tolerance_bootstrap,
    surface_arbitrage_violations,
    surface_diagnostic_metrics,
    strict_grid_interpolate,
)


def _serial(values) -> str:
    return json.dumps(np.asarray(values, dtype=float).reshape(-1).tolist())


def _panel_row(
    sample_id: str,
    pair_id: str,
    session_id: str,
    *,
    current: float = 0.20,
    target: float = 0.22,
    tolerance: int = 5,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "news_row_id": int(sample_id.split("_")[-1]),
        "pair_id": pair_id,
        "session_id": session_id,
        "effective_origin_utc": "2023-11-01T14:00:00Z",
        "current_surface_flat": _serial([current] * 4),
        "target_surface_flat": _serial([target] * 4),
        "surface_shape": "[2, 2]",
        "strike_grid": "[0.9, 1.1]",
        "maturity_days_grid": "[10, 20]",
        "hd_embedding": "[0.1, 0.2]",
        "lp_embedding": "[0.3, 0.4]",
        "sample_weight": 1.0,
        "first_included_tolerance_minutes": tolerance,
        "origin_shift_minutes": 0,
        "atm_ab_training_eligible": True,
        "dataset_tolerance_minutes": tolerance,
    }


def _run(tolerance: int, *, model: str = "wgan", run_id: str | None = None) -> RunSpec:
    return RunSpec(
        run_id=run_id or f"{model}_{tolerance:02d}m",
        run_dir=Path(f"/tmp/{model}_{tolerance:02d}m"),
        model=model,
        tolerance_minutes=tolerance,
        seed=42,
        checkpoint_path=Path(f"/tmp/{model}_{tolerance:02d}m/best.pt"),
    )


def _frozen_panel(panel_name: str) -> pd.DataFrame:
    if panel_name == "core":
        row_count, pair_count, session_count, tolerance = 200, 170, 49, 5
    else:
        row_count, pair_count, session_count, tolerance = 361, 263, 55, 30
    rows = []
    for index in range(row_count):
        pair_index = index if index < pair_count else index - pair_count
        pair_id = f"{panel_name}_pair_{pair_index}"
        session_id = f"{panel_name}_session_{pair_index % session_count}"
        row = _panel_row(
            f"news_{100000 + index}",
            pair_id,
            session_id,
            tolerance=tolerance,
        )
        row["strike_grid"] = "[0.98, 1.02]"
        row["dataset_tolerance_minutes"] = tolerance
        rows.append(row)
    outside = dict(rows[0])
    outside["sample_id"] = f"outside_{panel_name}"
    outside["news_row_id"] = 999999
    outside["effective_origin_utc"] = "2023-09-30T23:59:00Z"
    return pd.DataFrame(rows + [outside])


class TestNewsFirstVolComparisonAnalysis(unittest.TestCase):
    def test_numerical_ties_are_not_counted_as_wins_and_bootstrap_to_zero(self):
        panel = pd.DataFrame([_panel_row("news_1", "pair_1", "session_1")])
        predictions = pd.DataFrame(
            {
                "sample_id": ["news_1"],
                "predicted_surface_flat": [_serial([0.200000005] * 4)],
            }
        )
        samples, exclusions, _ = compute_sample_metrics(
            _run(5), "core", panel, predictions
        )
        self.assertTrue(exclusions.empty)
        self.assertLess(abs(float(samples.loc[0, "mae_gap"])), MAE_NUMERICAL_TIE_TOLERANCE)
        self.assertEqual(float(samples.loc[0, "mae_tie"]), 1.0)
        self.assertEqual(float(samples.loc[0, "win"]), 0.0)
        pairs = aggregate_pair_metrics(samples)
        overall = pairs[pairs["stratum_type"].eq("overall")].iloc[0]
        self.assertEqual(float(overall["mae_tie"]), 1.0)
        self.assertEqual(float(overall["win"]), 0.0)
        summary = build_model_comparison(pairs)
        overall_summary = summary[summary["stratum_type"].eq("overall")].iloc[0]
        self.assertEqual(float(overall_summary["tie_rate"]), 1.0)
        self.assertEqual(float(overall_summary["win"]), 0.0)

        comparison_rows = []
        for tolerance, model_mae in ((5, 0.02), (10, 0.020000005)):
            comparison_rows.append(
                {
                    "run_id": f"wgan_{tolerance}",
                    "model": "wgan",
                    "seed": 42,
                    "tolerance_minutes": tolerance,
                    "panel": "core",
                    "stratum_type": "overall",
                    "stratum_value": "all",
                    "pair_id": "pair_1",
                    "session_id": "session_1",
                    "model_mae": model_mae,
                }
            )
        bootstrap = build_bootstrap_comparisons(
            pd.DataFrame(comparison_rows),
            focal_tolerances=[10],
            metrics=["model_mae"],
            bootstrap_iterations=20,
        )
        self.assertEqual(float(bootstrap.loc[0, "mean_diff"]), 0.0)
        self.assertEqual(
            float(bootstrap.loc[0, "numerical_tie_tolerance"]),
            MAE_NUMERICAL_TIE_TOLERANCE,
        )

    def test_raw_joint_mask_drives_surface_metrics_and_auxiliary_exclusions(self):
        row = _panel_row("news_1", "pair_1", "session_1")
        params = json.dumps(
            {
                "business_days": [10, 20],
                "percent_strikes": [[0.89, 0.91], [0.89, 0.91]],
                "implied_vols": [[0.2, 0.2], [0.2, 0.2]],
            }
        )
        row["surface_model"] = "raw"
        row["current_surface_param_json"] = params
        row["target_surface_param_json"] = params
        row["current_surface_flat"] = _serial([0.20, 0.20, 0.20, 0.20])
        row["target_surface_flat"] = _serial([0.22, 0.70, 0.22, 0.70])
        predictions = pd.DataFrame(
            {
                "sample_id": ["news_1"],
                "predicted_surface_flat": [_serial([0.21, 0.10, 0.21, 0.10])],
            }
        )
        run = RunSpec(
            run_id="masked",
            run_dir=Path("/tmp/masked"),
            model="regression",
            tolerance_minutes=5,
            seed=42,
            checkpoint_path=Path("/tmp/masked/best.pt"),
            support_mask_mode="raw_joint",
        )
        metrics, general, exclusions = compute_sample_metrics(
            run, "core", pd.DataFrame([row]), predictions
        )
        self.assertTrue(general.empty)
        self.assertEqual(int(metrics.loc[0, "supported_cell_count"]), 2)
        self.assertAlmostEqual(float(metrics.loc[0, "model_mae"]), 0.01)
        self.assertAlmostEqual(float(metrics.loc[0, "persistence_mae"]), 0.02)
        self.assertIn("short_atm", set(exclusions["metric_family"]))
        with self.assertRaisesRegex(GridInterpolationError, "outside raw joint support"):
            strict_grid_interpolate(
                np.asarray([[0.2, 0.3], [0.2, 0.3]]),
                [0.9, 1.1],
                [10, 20],
                moneyness=1.0,
                maturity_days=15,
                support_mask=np.asarray([[True, False], [True, False]]),
            )

    def test_strict_grid_interpolation_never_extrapolates(self):
        surface = np.asarray([[1.0, 1.2], [1.1, 1.3]])
        value = strict_grid_interpolate(
            surface,
            [0.9, 1.1],
            [10.0, 20.0],
            moneyness=1.0,
            maturity_days=15.0,
        )
        self.assertAlmostEqual(value, 1.15)
        self.assertAlmostEqual(
            strict_grid_interpolate(
                surface,
                [0.9, 1.1],
                [10.0, 20.0],
                moneyness=0.9,
                maturity_days=10.0,
            ),
            1.0,
        )
        with self.assertRaises(GridInterpolationError) as caught:
            strict_grid_interpolate(
                surface,
                [0.9, 1.1],
                [10.0, 20.0],
                moneyness=1.1001,
                maturity_days=15.0,
            )
        self.assertEqual(caught.exception.code, "moneyness_outside_grid")

    def test_sample_metrics_use_persistence_and_do_not_assume_q_one(self):
        panel = pd.DataFrame([_panel_row("news_1", "pair_1", "session_1")])
        predictions = pd.DataFrame(
            {
                "sample_id": ["news_1"],
                "predicted_surface_flat": [_serial([0.21] * 4)],
            }
        )
        metrics, general, atm_skew = compute_sample_metrics(
            _run(5), "core", panel, predictions
        )
        self.assertTrue(general.empty)
        self.assertAlmostEqual(float(metrics.loc[0, "model_mae"]), 0.01)
        self.assertAlmostEqual(float(metrics.loc[0, "persistence_mae"]), 0.02)
        self.assertAlmostEqual(float(metrics.loc[0, "skill"]), 0.5)
        self.assertEqual(set(atm_skew["exclusion_code"]), {
            "missing_atm_coordinates",
            "missing_skew_coordinates",
        })
        self.assertIn("q=1 is deliberately not assumed", " ".join(atm_skew["detail"]))

    def test_actual_maturity_atm_and_skew_are_interpolated_inside_grid(self):
        panel_row = _panel_row("news_1", "pair_1", "session_1")
        panel_row["current_surface_flat"] = _serial([[0.20, 0.30], [0.30, 0.40]])
        panel_row["target_surface_flat"] = _serial([[0.22, 0.32], [0.32, 0.42]])
        panel = pd.DataFrame([panel_row])
        predictions = pd.DataFrame(
            {
                "sample_id": ["news_1"],
                "predicted_surface_flat": [panel_row["target_surface_flat"]],
            }
        )
        target_skew = 0.10 / (math.log(1.1) - math.log(0.9))
        bridge = pd.DataFrame(
            [
                {
                    "sample_id": "news_1",
                    "news_row_id": 1,
                    "pair_id": "pair_1",
                    "session_id": "session_1",
                    "slice_pair_id": "slice_1",
                    "maturity_date": "2023-12-15",
                    "underlying_contract_id": "TYZ3",
                    "origin_business_days": 15,
                    "target_business_days": 15,
                    "pair_maturity_count": 1,
                    "metric_status": "ok",
                    "pair_atm_quality": "A",
                    "pair_atm_strike": 100.0,
                    "target_atm_q": 1.0,
                    "current_atm_iv": 0.29,
                    "target_atm_iv": 0.32,
                    "skew_status": "ok",
                    "skew_quality": "A",
                    "skew_left_strike": 90.0,
                    "skew_right_strike": 110.0,
                    "current_atm_iv_skew_secant": target_skew - 0.05,
                    "target_atm_iv_skew_secant": target_skew,
                }
            ]
        )
        maturity, exclusions = compute_maturity_metrics(
            _run(5), "core", panel, predictions, bridge
        )
        self.assertTrue(exclusions.empty)
        self.assertEqual(maturity.loc[0, "atm_metric_status"], "ok")
        self.assertAlmostEqual(float(maturity.loc[0, "predicted_atm_iv"]), 0.32)
        self.assertAlmostEqual(float(maturity.loc[0, "atm_model_abs_error"]), 0.0)
        self.assertEqual(maturity.loc[0, "skew_metric_status"], "ok")
        self.assertAlmostEqual(float(maturity.loc[0, "predicted_skew"]), target_skew)
        self.assertAlmostEqual(float(maturity.loc[0, "skew_model_abs_error"]), 0.0)

        bridge.loc[0, "target_atm_q"] = 1.2
        _, exclusions = compute_maturity_metrics(_run(5), "core", panel, predictions, bridge)
        self.assertIn("moneyness_outside_grid", set(exclusions["exclusion_code"]))

    def test_pair_aggregation_is_pair_balanced_not_article_balanced(self):
        rows = []
        for sample, pair, weight, model_mae in (
            ("a", "p1", 0.5, 0.1),
            ("b", "p1", 0.5, 0.3),
            ("c", "p2", 1.0, 0.4),
        ):
            rows.append(
                {
                    "run_id": "wgan_05m",
                    "model": "wgan",
                    "seed": 42,
                    "tolerance_minutes": 5,
                    "panel": "core",
                    "sample_id": sample,
                    "news_row_id": sample,
                    "pair_id": pair,
                    "session_id": f"s_{pair}",
                    "first_included_tolerance_minutes": 5,
                    "origin_shift_minutes": 0,
                    "atm_ab_flag": "atm_ab",
                    "sample_weight": weight,
                    "model_mae": model_mae,
                    "model_rmse": model_mae,
                    "model_max_abs": model_mae,
                    "persistence_mae": 0.5,
                    "persistence_rmse": 0.5,
                    "persistence_max_abs": 0.5,
                    "mae_gap": model_mae - 0.5,
                    "skill": 1.0 - model_mae / 0.5,
                    "win": 1.0,
                    "atm_model_abs_error": np.nan,
                    "atm_persistence_abs_error": np.nan,
                    "atm_error_gap": np.nan,
                    "skew_model_abs_error": np.nan,
                    "skew_persistence_abs_error": np.nan,
                    "skew_error_gap": np.nan,
                }
            )
        pairs = aggregate_pair_metrics(pd.DataFrame(rows))
        overall = pairs[pairs["stratum_type"].eq("overall")].set_index("pair_id")
        self.assertAlmostEqual(float(overall.loc["p1", "model_mae"]), 0.2)
        comparison = build_model_comparison(pairs)
        summary = comparison[comparison["stratum_type"].eq("overall")].iloc[0]
        self.assertAlmostEqual(float(summary["mae"]), 0.3)
        self.assertEqual(int(summary["pair_count"]), 2)

    def test_cluster_bootstrap_is_deterministic_and_holm_is_monotone(self):
        exact_tie = cme_session_cluster_bootstrap(
            [0.0], ["only_session"], iterations=10, seed=7
        )
        self.assertEqual(exact_tie["status"], "numerical_tie")
        self.assertEqual(exact_tie["mean_diff"], 0.0)
        self.assertEqual(exact_tie["ci_95_lower"], 0.0)
        self.assertEqual(exact_tie["ci_95_upper"], 0.0)
        self.assertEqual(exact_tie["p_two_sided"], 1.0)
        first = cme_session_cluster_bootstrap(
            [-0.1, -0.2, -0.3, -0.4],
            ["s1", "s1", "s2", "s3"],
            iterations=500,
            seed=7,
        )
        second = cme_session_cluster_bootstrap(
            [-0.1, -0.2, -0.3, -0.4],
            ["s1", "s1", "s2", "s3"],
            iterations=500,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_diff"], -0.25)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_wgan_tolerance_comparison_matches_pairs_and_holm_adjusts_three_tests(self):
        rows = []
        for tolerance, error in ((5, 0.20), (10, 0.15), (15, 0.12), (30, 0.10)):
            for index in range(4):
                rows.append(
                    {
                        "run_id": f"wgan_{tolerance}",
                        "model": "wgan",
                        "seed": 42,
                        "tolerance_minutes": tolerance,
                        "panel": "core",
                        "stratum_type": "overall",
                        "stratum_value": "all",
                        "pair_id": f"p{index}",
                        "session_id": f"s{index // 2}",
                        "model_mae": error,
                        "model_rmse": error,
                        "model_max_abs": error,
                        "persistence_mae": 0.25,
                        "persistence_rmse": 0.25,
                        "persistence_max_abs": 0.25,
                        "mae_gap": error - 0.25,
                        "skill": 1.0 - error / 0.25,
                        "win": 1.0,
                        "atm_model_abs_error": np.nan,
                        "atm_persistence_abs_error": np.nan,
                        "atm_error_gap": np.nan,
                        "skew_model_abs_error": np.nan,
                        "skew_persistence_abs_error": np.nan,
                        "skew_error_gap": np.nan,
                    }
                )
        result = build_bootstrap_comparisons(
            pd.DataFrame(rows),
            metrics=["model_mae"],
            bootstrap_iterations=500,
            bootstrap_seed=11,
        )
        self.assertEqual(len(result), 3)
        self.assertTrue((result["mean_diff"] < 0.0).all())
        self.assertTrue(result["negative_means_focal_better"].all())
        self.assertTrue(result["p_holm"].notna().all())
        self.assertTrue((result["pair_count"] == 4).all())

    def test_text_ablation_builds_one_global_16_test_holm_family(self):
        rows = []
        values = {"real_text": 0.10, "current_only": 0.20, "text_shuffle": 0.15}
        for model in ("wgan", "regression"):
            for tolerance in (5, 10, 15, 30):
                for mode, value in values.items():
                    for pair_index in range(4):
                        rows.append(
                            {
                                "run_id": f"{model}_{mode}_{tolerance}",
                                "model": model,
                                "text_ablation_mode": mode,
                                "support_mask_mode": "raw_joint",
                                "seed": 42,
                                "tolerance_minutes": tolerance,
                                "panel": "core",
                                "stratum_type": "overall",
                                "stratum_value": "all",
                                "pair_id": f"pair_{pair_index}",
                                "session_id": f"session_{pair_index % 2}",
                                "model_mae": value + pair_index * 0.001,
                            }
                        )
        result = build_text_ablation_comparisons(
            pd.DataFrame(rows), bootstrap_iterations=100, bootstrap_seed=42
        )
        self.assertEqual(len(result), 16)
        self.assertEqual(
            set(result["holm_family"]),
            {"global|core|model_mae|real_vs_controls"},
        )
        self.assertTrue((result["mean_diff"] < 0).all())
        self.assertTrue(result["negative_means_real_text_better"].all())
        self.assertTrue((result["p_holm"] >= result["p_two_sided"]).all())
        tie_rows = []
        tie_values = {
            "real_text": 0.10,
            "current_only": 0.10 + 0.5 * MAE_NUMERICAL_TIE_TOLERANCE,
            "text_shuffle": 0.10 - 0.5 * MAE_NUMERICAL_TIE_TOLERANCE,
        }
        for model in ("wgan", "regression"):
            for tolerance in (5, 10, 15, 30):
                for mode, value in tie_values.items():
                    for pair_index in range(2):
                        tie_rows.append(
                            {
                                "run_id": f"{model}_{mode}_{tolerance}",
                                "model": model,
                                "text_ablation_mode": mode,
                                "support_mask_mode": "raw_joint",
                                "seed": 42,
                                "tolerance_minutes": tolerance,
                                "panel": "core",
                                "stratum_type": "overall",
                                "stratum_value": "all",
                                "pair_id": f"pair_{pair_index}",
                                "session_id": f"session_{pair_index}",
                                "model_mae": value,
                            }
                        )
        ties = build_text_ablation_comparisons(
            pd.DataFrame(tie_rows), bootstrap_iterations=20, bootstrap_seed=42
        )
        self.assertTrue(ties["mean_diff"].eq(0.0).all())
        self.assertTrue(ties["p_two_sided"].eq(1.0).all())
        self.assertTrue(
            ties["numerical_tie_tolerance"].eq(
                MAE_NUMERICAL_TIE_TOLERANCE
            ).all()
        )
        candidate = pd.DataFrame(
            [
                {
                    "model": "wgan",
                    "text_ablation_mode": mode,
                    "focal_tolerance_minutes": tolerance,
                    "panel": "core",
                    "metric": "model_mae",
                    "stratum_type": "overall",
                    "stratum_value": "all",
                }
                for mode in ("real_text", "current_only", "text_shuffle")
                for tolerance in (10, 15, 30)
            ]
        )
        primary = select_primary_tolerance_bootstrap(candidate)
        self.assertEqual(len(primary), 3)
        self.assertEqual(set(primary["text_ablation_mode"]), {"real_text"})

    def test_discover_run_spec_uses_parent_tolerance_and_regressor_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "regression" / "tolerance_10m" / "stamp"
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "checkpoints").mkdir()
            (run_dir / "metrics" / "training_resolved_config.yaml").write_text(
                "seed: 202\ntext_embedding_mode: lp\n",
                encoding="utf-8",
            )
            (run_dir / "checkpoints" / "vol_regressor_best.pt").write_bytes(b"checkpoint")
            spec = discover_run_spec(run_dir)
            self.assertEqual(spec.model, "regression")
            self.assertEqual(spec.tolerance_minutes, 10)
            self.assertEqual(spec.seed, 202)
            self.assertEqual(spec.checkpoint_path.name, "vol_regressor_best.pt")

    def test_frozen_q4_panel_counts_are_enforced_for_core_and_broad(self):
        for panel_name, expected in (
            ("core", (200, 170, 49)),
            ("broad", (361, 263, 55)),
        ):
            panel, lineage, bridge = _load_panel_source(
                _frozen_panel(panel_name),
                sheet_name="gan_input_ready",
                panel_name=panel_name,
                freeze_test_window=True,
                enforce_expected_counts=True,
            )
            self.assertEqual(
                (len(panel), panel["pair_id"].nunique(), panel["session_id"].nunique()),
                expected,
            )
            self.assertTrue(lineage["expected_counts_verified"])
            self.assertTrue(bridge.empty)

    def test_experiment_specs_require_eight_completed_best_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry = root / "registry"
            (registry / "jobs").mkdir(parents=True)
            jobs = []
            for model in ("wgan", "regression"):
                for tolerance in (5, 10, 15, 30):
                    job_id = f"{model}_{tolerance:02d}m"
                    jobs.append(
                        {
                            "job_id": job_id,
                            "model_family": model,
                            "tolerance_minutes": tolerance,
                        }
                    )
                    run_dir = root / "runs" / model / f"tolerance_{tolerance:02d}m" / "stamp"
                    (run_dir / "checkpoints").mkdir(parents=True)
                    (run_dir / "metrics").mkdir()
                    checkpoint = (
                        "generator_best.pt" if model == "wgan" else "vol_regressor_best.pt"
                    )
                    (run_dir / "checkpoints" / checkpoint).write_bytes(b"best")
                    (run_dir / "metrics" / "training_resolved_config.yaml").write_text(
                        "seed: 42\ntext_embedding_mode: lp\n", encoding="utf-8"
                    )
                    (registry / "jobs" / f"{job_id}.status.json").write_text(
                        json.dumps({"status": "completed", "run_dir": str(run_dir)}),
                        encoding="utf-8",
                    )
            (registry / "jobs.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            specs = _experiment_run_specs(root)
            self.assertEqual(len(specs), 8)
            self.assertEqual(
                {spec.checkpoint_path.name for spec in specs},
                {"generator_best.pt", "vol_regressor_best.pt"},
            )

    def test_formal_analysis_fails_on_incomplete_prediction_keys(self):
        core = _frozen_panel("core")
        broad = _frozen_panel("broad")

        def incomplete_evaluator(run, panel_name, frame):
            del run, panel_name
            frame = frame.iloc[:-1]
            return pd.DataFrame(
                {
                    "sample_id": frame["sample_id"],
                    "predicted_surface_flat": frame["target_surface_flat"],
                    "prediction_status": "ok",
                    "prediction_mc_samples": 64,
                    "prediction_fallback": False,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(ComparisonAnalysisError):
            run_analysis(
                [_run(5)],
                core,
                broad,
                incomplete_evaluator,
                tmpdir,
                expected_run_count=1,
                bootstrap_iterations=10,
                freeze_test_window=True,
                enforce_expected_panel_counts=True,
            )

    def test_production_wgan_evaluator_uses_stable_sample_keys(self):
        import torch

        if not hasattr(torch, "cuda"):
            self.skipTest("full PyTorch runtime is unavailable in this interpreter")

        from wgan_option.config import Config
        from wgan_option.models.generator import Generator

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "generator_best.pt"
            config = Config(
                cuda=False,
                text_embedding_mode="lp",
                channels=1,
                embedding_dim=2,
                noise_dim=3,
                gen_base_channels=2,
                gen_res_blocks=0,
                gen_text_hidden_dim=4,
                gen_text_out_dim=3,
                gen_hidden_dim=8,
            )
            model = Generator(
                channels=1,
                embedding_dim=2,
                noise_dim=3,
                surface_height=2,
                surface_width=2,
                base_channels=2,
                res_blocks=0,
                text_hidden_dim=4,
                text_out_dim=3,
                hidden_dim=8,
            )
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(config), "embedding_dim": 2},
                checkpoint,
            )
            run = RunSpec("wgan_05m", Path(tmpdir), "wgan", 5, 42, checkpoint)
            panel = pd.DataFrame(
                [
                    _panel_row("news_1", "p1", "s1"),
                    _panel_row("news_2", "p2", "s2"),
                ]
            )
            evaluator = TrainedRunEvaluator(
                mc_samples=4,
                sample_batch_size=1,
                draw_batch_size=2,
                device="cpu",
            )
            first = evaluator(run, "core", panel).set_index("sample_id")
            second = evaluator(run, "core", panel.iloc[::-1].reset_index(drop=True)).set_index("sample_id")
            for sample_id in ("news_1", "news_2"):
                np.testing.assert_allclose(
                    first.loc[sample_id, "predicted_surface_flat"],
                    second.loc[sample_id, "predicted_surface_flat"],
                )
                self.assertEqual(len(first.loc[sample_id, "predicted_surface_flat"]), 4)
                self.assertTrue(np.isfinite(first.loc[sample_id, "predicted_surface_flat"]).all())
            self.assertTrue((first["prediction_mc_samples"] == 4).all())
            self.assertFalse(first["prediction_fallback"].any())

    def test_short_atm_and_trainer_constraint_diagnostics(self):
        strikes = np.asarray([0.98, 1.02, 1.06])
        maturities = np.asarray([7.0, 30.0, 90.0])
        target = np.full((3, 3), 0.20)
        current = target + 0.03
        predicted = target + 0.01
        metrics = surface_diagnostic_metrics(
            current, target, predicted, strikes, maturities
        )
        self.assertEqual(metrics["short_atm_cell_count"], 4)
        self.assertAlmostEqual(metrics["short_atm_model_mae"], 0.01)
        self.assertAlmostEqual(metrics["short_atm_persistence_mae"], 0.03)
        self.assertAlmostEqual(metrics["short_atm_mae_gap"], -0.02)
        violations = surface_arbitrage_violations(
            np.asarray([[0.40, 0.40, 0.40], [0.10, 0.10, 0.10], [0.10, 0.10, 0.10]]),
            strikes,
            maturities,
        )
        self.assertGreater(violations["calendar_violation_count"], 0)
        self.assertEqual(violations["calendar_constraint_count"], 6)
        with self.assertRaises(ComparisonAnalysisError):
            surface_arbitrage_violations(target, [0.98, 1.01, 1.06], maturities)

    def test_narrow_grid_short_atm_count_is_derived_as_160(self):
        strikes = np.linspace(0.97, 1.03, 16)
        maturities = np.asarray(
            [7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 30, 32, 34, 36, 38],
            dtype=float,
        )
        surface = np.full((16, 16), 0.20)

        metrics = surface_diagnostic_metrics(
            surface,
            surface,
            surface,
            strikes,
            maturities,
        )

        self.assertEqual(metrics["short_atm_cell_count"], 160)

    def test_run_analysis_writes_stable_outputs_with_injected_evaluator(self):
        panel = pd.DataFrame(
            [
                _panel_row("news_1", "p1", "s1"),
                _panel_row("news_2", "p2", "s2"),
            ]
        )
        panel["strike_grid"] = "[0.98, 1.02]"

        def evaluator(run, panel_name, frame):
            del panel_name
            error = {5: 0.015, 10: 0.010, 15: 0.008, 30: 0.005}[run.tolerance_minutes]
            return pd.DataFrame(
                {
                    "sample_id": frame["sample_id"],
                    "predicted_surface_flat": [
                        _serial(_serial_to_array(value) - error)
                        for value in frame["target_surface_flat"]
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = run_analysis(
                [_run(value) for value in (5, 10, 15, 30)],
                panel,
                panel,
                evaluator,
                tmpdir,
                expected_run_count=4,
                bootstrap_iterations=200,
                freeze_test_window=False,
                enforce_expected_panel_counts=False,
            )
            expected = {
                "sample_metrics.csv.gz",
                "maturity_metrics.csv.gz",
                "pair_metrics.csv.gz",
                "model_comparison.csv",
                "stratified_metrics.csv",
                "training_curves.csv",
                "checkpoint_summary.csv",
                "bootstrap_comparisons.csv",
                "bootstrap_sensitivity.csv",
                "evaluation_exclusions.csv.gz",
                "atm_skew_exclusions.csv.gz",
                "evaluation_coverage.csv",
                "analysis_manifest.json",
                "output_schema.json",
                "analysis_config.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            samples = pd.read_csv(output / "sample_metrics.csv.gz")
            self.assertEqual(len(samples), 16)
            self.assertTrue((samples["surface_diagnostic_status"] == "ok").all())
            self.assertEqual(set(samples["short_atm_cell_count"]), {4})
            self.assertTrue(
                np.isfinite(
                    samples[
                        [
                            "short_atm_model_mae",
                            "predicted_calendar_violation_rate",
                            "predicted_butterfly_violation_rate",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            )
            bootstrap = pd.read_csv(output / "bootstrap_comparisons.csv")
            surface = bootstrap[
                bootstrap["metric"].eq("model_mae")
                & bootstrap["stratum_type"].eq("overall")
            ]
            self.assertEqual(len(surface), 3)
            self.assertEqual(set(surface["panel"]), {"core"})
            self.assertEqual(set(surface["focal_tolerance_minutes"]), {10, 15, 30})
            self.assertTrue((surface["mean_diff"] < 0.0).all())
            sensitivity = pd.read_csv(output / "bootstrap_sensitivity.csv")
            broad = sensitivity[
                sensitivity["metric"].eq("model_mae")
                & sensitivity["stratum_type"].eq("overall")
                & sensitivity["panel"].eq("broad")
            ]
            self.assertEqual(len(broad), 3)
            prediction_paths = sorted((output / "predictions").glob("*.csv.gz"))
            self.assertEqual(len(prediction_paths), 8)
            prediction = pd.read_csv(prediction_paths[0])
            self.assertEqual(len(prediction), 2)
            self.assertFalse(prediction["prediction_fallback"].any())
            manifest = json.loads((output / "analysis_manifest.json").read_text())
            self.assertEqual(manifest["row_counts"]["prediction_rows"], 16)
            self.assertIn("core=4", manifest["methodology"]["short_atm"])
            self.assertNotIn("six cells", manifest["methodology"]["short_atm"])
            analysis_config = json.loads((output / "analysis_config.json").read_text())
            self.assertEqual(
                analysis_config["numerical_tie_policy"]["mae_gap_tolerance_iv"],
                MAE_NUMERICAL_TIE_TOLERANCE,
            )
            self.assertEqual(
                analysis_config["short_atm"]["observed_cell_counts_by_panel"],
                {"broad": [4], "core": [4]},
            )
            for relative, expected_hash in manifest["output_sha256"].items():
                import hashlib

                self.assertEqual(
                    hashlib.sha256((output / relative).read_bytes()).hexdigest(),
                    expected_hash,
                )


def _serial_to_array(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


if __name__ == "__main__":
    unittest.main()
