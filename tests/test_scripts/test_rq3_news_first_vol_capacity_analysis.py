from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.rq3.news_first_vol_capacity_analysis import (
    CapacityAnalysisError,
    PROFILE_PARAMETER_COUNTS,
    evaluate_capacity_stage,
    one_standard_error_selection,
    run_final_q4_analysis,
    summarize_profile_scores,
)
from scripts.rq3 import news_first_vol_capacity_report as capacity_report_module
from scripts.rq3.news_first_vol_capacity_report import render_capacity_report
from scripts.rq3.news_first_vol_training import FROZEN_CAPACITY_PROFILES
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator
from wgan_option.models.vol_regressor import VolSurfaceRegressor


class CapacityStatisticsTests(unittest.TestCase):
    def test_frozen_profile_counts_are_strictly_increasing(self):
        for family in ("regression", "wgan"):
            counts = [
                PROFILE_PARAMETER_COUNTS[name][family]
                for name in ("micro", "tiny", "small", "medium", "large", "legacy")
            ]
            self.assertEqual(counts, sorted(counts))
            self.assertEqual(len(counts), len(set(counts)))
        self.assertEqual(PROFILE_PARAMETER_COUNTS["micro"]["regression"], 20_070)
        self.assertEqual(PROFILE_PARAMETER_COUNTS["legacy"]["wgan"], 4_691_653)

    def test_profile_counts_match_real_model_instantiation(self):
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters())

        for name, profile in FROZEN_CAPACITY_PROFILES.items():
            shared = {
                "channels": 1,
                "embedding_dim": 1024,
                "surface_height": 16,
                "surface_width": 16,
                "base_channels": profile["gen_base_channels"],
                "res_blocks": 0,
                "text_hidden_dim": profile["gen_text_hidden_dim"],
                "text_out_dim": profile["gen_text_out_dim"],
                "hidden_dim": profile["gen_hidden_dim"],
                "residual_output_mode": "identity_softplus_residual",
            }
            regression = VolSurfaceRegressor(**shared)
            generator = Generator(noise_dim=32, **shared)
            critic = Discriminator(
                channels=1,
                embedding_dim=1024,
                surface_height=16,
                surface_width=16,
                base_channels=profile["disc_base_channels"],
                res_blocks=0,
                text_hidden_dim=profile["disc_text_hidden_dim"],
                hidden_dim=profile["disc_hidden_dim"],
            )
            self.assertEqual(
                count(regression), PROFILE_PARAMETER_COUNTS[name]["regression"]
            )
            self.assertEqual(
                count(generator) + count(critic),
                PROFILE_PARAMETER_COUNTS[name]["wgan"],
            )

    def test_score_is_equal_tolerance_mean_log_ratio_and_gate_is_fail_closed(self):
        rows = []
        for profile, ratios in {"micro": (0.99, 0.99), "tiny": (0.98, 1.001)}.items():
            for tolerance, ratio in zip((5, 30), ratios):
                rows.append(
                    {
                        "model_family": "regression",
                        "capacity_profile": profile,
                        "text_ablation_mode": "real_text",
                        "tolerance_minutes": tolerance,
                        "parameter_count": PROFILE_PARAMETER_COUNTS[profile][
                            "regression"
                        ],
                        "log_mae_ratio": math.log(ratio),
                        "mae_ratio": ratio,
                    }
                )
        result = summarize_profile_scores(
            pd.DataFrame(rows),
            model_family="regression",
            text_ablation_mode="real_text",
            tolerances=(5, 30),
        ).set_index("capacity_profile")
        self.assertAlmostEqual(
            float(result.loc["micro", "mean_log_mae_ratio"]), math.log(0.99)
        )
        self.assertTrue(bool(result.loc["micro", "profile_gate_passed"]))
        self.assertFalse(bool(result.loc["tiny", "profile_gate_passed"]))

    def test_one_standard_error_rule_selects_smallest_eligible_profile(self):
        rows = []
        values = {
            "micro": {"s1": 0.95, "s2": 0.95},
            "tiny": {"s1": 0.88, "s2": 1.00},
        }
        for profile, by_session in values.items():
            for tolerance in (5, 30):
                for index, (session, model_mae) in enumerate(by_session.items()):
                    rows.append(
                        {
                            "capacity_profile": profile,
                            "model": "regression",
                            "text_ablation_mode": "real_text",
                            "tolerance_minutes": tolerance,
                            "pair_id": f"p{index}",
                            "session_id": session,
                            "model_mae": model_mae,
                            "persistence_mae": 1.0,
                        }
                    )
        result = one_standard_error_selection(
            pd.DataFrame(rows),
            profiles=("micro", "tiny"),
            tolerances=(5, 30),
            model_family="regression",
            text_ablation_mode="real_text",
            iterations=200,
            seed=7,
        )
        self.assertEqual(result["best_score_profile"], "tiny")
        self.assertIn("micro", result["one_se_eligible_profiles"])
        self.assertEqual(result["winner_profile"], "micro")


class CapacityFixtureMixin:
    def _write_job(
        self,
        root: Path,
        *,
        job_id: str,
        stage: str,
        model: str,
        profile: str,
        mode: str,
        tolerance: int,
        ratio: float,
    ) -> dict:
        run_dir = root / "runs" / job_id
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        metadata = {
            "best_epoch": 2,
            "best_metric": ratio,
            "metrics": {"val_recon": ratio, "val_current_recon": 1.0},
            "artifacts": {},
        }
        (metrics / "best_learned_checkpoint.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (metrics / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "best_epoch": 0,
                    "best_metric": 1.0,
                    "selection_scope": "baseline_inclusive",
                    "metrics": {"val_recon": 1.0},
                }
            ),
            encoding="utf-8",
        )
        train_metric = "g_recon" if model == "wgan" else "train_recon"
        (metrics / "training_metrics.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": 2,
                        train_metric: max(1.0e-6, ratio - 0.01),
                        "val_recon": ratio,
                    }
                ]
            ),
            encoding="utf-8",
        )
        config_path = root / "configs" / f"{job_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "news_first_capacity_profile": profile,
                    "news_first_capacity_profile_sha256": f"sha-{profile}",
                    "support_mask_mode": "raw_joint",
                    "seed": 42,
                }
            ),
            encoding="utf-8",
        )
        status_dir = root / "registry" / "jobs"
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / f"{job_id}.status.json").write_text(
            json.dumps({"status": "completed", "run_dir": str(run_dir)}),
            encoding="utf-8",
        )
        return {
            "job_id": job_id,
            "capacity_stage": stage,
            "model_family": model,
            "capacity_profile": profile,
            "capacity_profile_sha256": f"sha-{profile}",
            "text_ablation_mode": mode,
            "tolerance_minutes": tolerance,
            "training_config_path": str(config_path),
        }

    def _write_registry(self, root: Path, jobs: list[dict]) -> None:
        registry = root / "registry"
        registry.mkdir(parents=True, exist_ok=True)
        (registry / "jobs.json").write_text(
            json.dumps({"jobs": jobs}), encoding="utf-8"
        )
        pd.DataFrame(
            [
                {
                    "job_id": job["job_id"],
                    "model": job["model_family"],
                    "capacity_profile": job["capacity_profile"],
                    "runtime_minutes": 12.5,
                    "peak_memory_mib": 2048.0,
                }
                for job in jobs
            ]
        ).to_csv(root / "resource_summary.csv", index=False)


class CapacityStageTests(CapacityFixtureMixin, unittest.TestCase):
    def test_regression_screen_selects_top2_without_opening_any_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = []
            profile_ratio = {
                "micro": 0.98,
                "tiny": 0.985,
                "small": 0.99,
                "medium": 0.995,
                "large": 1.0,
                "legacy": 1.01,
            }
            for profile, ratio in profile_ratio.items():
                for mode in ("current_only", "real_text"):
                    for tolerance in (5, 30):
                        jobs.append(
                            self._write_job(
                                root,
                                job_id=f"regression_screen_{profile}_{mode}_{tolerance}",
                                stage="regression_screen",
                                model="regression",
                                profile=profile,
                                mode=mode,
                                tolerance=tolerance,
                                ratio=0.99 if mode == "current_only" else ratio,
                            )
                        )
            self._write_registry(root, jobs)
            with patch(
                "pandas.read_excel", side_effect=AssertionError("Q4/panel read")
            ):
                result = evaluate_capacity_stage(root, "regression_screen")
            self.assertEqual(result["selected_profiles"], ["micro", "tiny"])
            self.assertTrue(result["gate_passed"])
            self.assertFalse(result["q4_used_for_selection"])
            self.assertTrue((root / "capacity_selection.json").is_file())
            audit = pd.read_csv(root / "capacity_selection_audit.csv")
            self.assertEqual(int(audit.loc[0, "q4_rows_passed_to_evaluator"]), 0)
            comparisons = pd.read_csv(root / "capacity_comparisons.csv")
            required = {
                "best_learned_train_recon",
                "best_learned_val_recon",
                "best_learned_train_val_gap",
                "baseline_inclusive_best_epoch",
                "baseline_inclusive_epoch0_selected",
                "parameter_log10",
                "runtime_minutes",
                "peak_memory_mib",
            }
            self.assertTrue(required.issubset(comparisons.columns))
            self.assertTrue(
                comparisons["baseline_inclusive_epoch0_selected"].astype(bool).all()
            )
            self.assertTrue(
                (
                    comparisons["best_learned_val_recon"]
                    - comparisons["best_learned_train_recon"]
                    - comparisons["best_learned_train_val_gap"]
                )
                .abs()
                .lt(1.0e-12)
                .all()
            )

    def test_failed_gate_records_ranked_candidates_but_no_formal_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = []
            ratios = {
                "micro": 1.02,
                "tiny": 1.015,
                "small": 1.01,
                "medium": 1.005,
                "large": 0.999,
                "legacy": 1.001,
            }
            for profile, ratio in ratios.items():
                for mode in ("current_only", "real_text"):
                    for tolerance in (5, 30):
                        jobs.append(
                            self._write_job(
                                root,
                                job_id=f"failed_{profile}_{mode}_{tolerance}",
                                stage="regression_screen",
                                model="regression",
                                profile=profile,
                                mode=mode,
                                tolerance=tolerance,
                                ratio=ratio,
                            )
                        )
            self._write_registry(root, jobs)
            result = evaluate_capacity_stage(root, "regression_screen")
            self.assertFalse(result["gate_passed"])
            self.assertEqual(result["winner_profile"], "")
            self.assertEqual(result["capacity_profile"], "")
            self.assertIsNone(result["winner_parameter_count"])
            self.assertEqual(result["selected_profiles"], [])
            self.assertEqual(result["ranked_leader_profile"], "large")
            self.assertEqual(result["ranked_top2_profiles"], ["large", "legacy"])
            comparisons = pd.read_csv(root / "capacity_comparisons.csv")
            self.assertFalse(comparisons["selected_top2"].astype(bool).any())
            ranked = set(
                comparisons.loc[
                    comparisons["ranked_top2"].astype(bool), "capacity_profile"
                ].astype(str)
            )
            self.assertEqual(ranked, {"large", "legacy"})

    def test_regression_confirm_uses_real_text_and_one_se_not_shuffle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "capacity_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "q4_used_for_selection": False,
                        "stages": {
                            "regression_screen": {
                                "gate_passed": True,
                                "selected_profiles": ["micro", "tiny"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            jobs = []
            for profile in ("micro", "tiny"):
                for mode in ("current_only", "real_text", "text_shuffle"):
                    for tolerance in (5, 10, 15, 30):
                        ratio = {
                            "current_only": 0.99,
                            "real_text": 0.98 if profile == "micro" else 0.97,
                            # A deliberately awful shuffle must remain diagnostic.
                            "text_shuffle": 2.0,
                        }[mode]
                        stage = (
                            "regression_screen"
                            if mode in {"current_only", "real_text"}
                            and tolerance in {5, 30}
                            else "regression_confirm"
                        )
                        jobs.append(
                            self._write_job(
                                root,
                                job_id=f"regression_{profile}_{mode}_{tolerance}",
                                stage=stage,
                                model="regression",
                                profile=profile,
                                mode=mode,
                                tolerance=tolerance,
                                ratio=ratio,
                            )
                        )
            self._write_registry(root, jobs)
            pairs = []
            for profile in ("micro", "tiny"):
                by_session = (
                    {"s1": 0.98, "s2": 0.98}
                    if profile == "micro"
                    else {"s1": 0.90, "s2": 1.04}
                )
                for tolerance in (5, 10, 15, 30):
                    for index, (session, value) in enumerate(by_session.items()):
                        pairs.append(
                            {
                                "capacity_profile": profile,
                                "model": "regression",
                                "text_ablation_mode": "real_text",
                                "tolerance_minutes": tolerance,
                                "pair_id": f"p{index}",
                                "session_id": session,
                                "model_mae": value,
                                "persistence_mae": 1.0,
                            }
                        )
            result = evaluate_capacity_stage(
                root,
                "regression_confirm",
                q3_pair_metrics=pd.DataFrame(pairs),
                bootstrap_iterations=200,
                bootstrap_seed=9,
            )
            self.assertEqual(result["selection_text_ablation_mode"], "real_text")
            self.assertEqual(result["winner_profile"], "micro")
            self.assertTrue(result["gate_passed"])
            self.assertNotIn(
                "text_shuffle",
                json.dumps(result["one_standard_error_selection"]),
            )

    def test_final_q4_refuses_to_read_panel_when_regression_gate_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "registry").mkdir(parents=True)
            (root / "registry" / "jobs.json").write_text(
                json.dumps({"jobs": []}), encoding="utf-8"
            )
            (root / "capacity_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "q4_used_for_selection": False,
                        "stages": {
                            "regression_screen": {"gate_passed": True},
                            "regression_confirm": {"gate_passed": False},
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("pandas.read_excel", side_effect=AssertionError("Q4 read")):
                with self.assertRaisesRegex(
                    CapacityAnalysisError, "Q4 must remain unread"
                ):
                    run_final_q4_analysis(root, bootstrap_iterations=10)


class CapacityReportTests(unittest.TestCase):
    def test_stage_only_report_for_valid_no_learned_capacity_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "registry").mkdir(parents=True)
            (root / "registry" / "experiment_status.json").write_text(
                json.dumps(
                    {
                        "status": "completed_no_learned_capacity",
                        "current_stage": "regression_screen",
                        "completed_at_utc": "2026-08-19T19:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            (root / "capacity_stage_status.json").write_text(
                json.dumps(
                    {
                        "status": "completed_no_learned_capacity",
                        "current_stage": "regression_screen",
                        "stages": {
                            "regression_screen": {
                                "status": "completed_no_learned_capacity",
                                "completed_at_utc": "2026-08-19T19:00:00Z",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "capacity_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "q4_used_for_selection": False,
                        "stages": {
                            "regression_screen": {
                                "gate_passed": False,
                                "selected_profiles": [],
                                "winner_profile": "",
                                "capacity_profile": "",
                                "winner_parameter_count": None,
                                "ranked_leader_profile": "large",
                                "ranked_leader_parameter_count": 676_528,
                                "ranked_top2_profiles": ["large", "medium"],
                                "selection_text_ablation_mode": "real_text",
                                "selection_tolerances": [5, 30],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            comparison_rows = []
            for profile, parameters, ratio in (
                ("medium", 344_800, 1.004),
                ("large", 676_528, 0.999),
            ):
                for tolerance in (5, 30):
                    comparison_rows.append(
                        {
                            "selection_stage": "regression_screen",
                            "model_family": "regression",
                            "capacity_profile": profile,
                            "text_ablation_mode": "real_text",
                            "tolerance_minutes": tolerance,
                            "parameter_count": parameters,
                            "parameter_log10": math.log10(parameters),
                            "mae_ratio": ratio,
                            "best_learned_epoch": 17,
                            "best_learned_train_recon": 0.020,
                            "best_learned_val_recon": 0.024,
                            "best_learned_train_val_gap": 0.004,
                            "baseline_inclusive_best_epoch": 0,
                            "baseline_inclusive_epoch0_selected": True,
                            "runtime_minutes": 8.0,
                            "peak_memory_mib": 1024.0,
                            "profile_gate_passed": False,
                        }
                    )
            pd.DataFrame(comparison_rows).to_csv(
                root / "capacity_comparisons.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "capacity_profile": "medium",
                        "expected_regression_parameters": 344_800,
                        "expected_wgan_parameters": 422_953,
                        "capacity_profile_sha256": "m" * 64,
                    },
                    {
                        "capacity_profile": "large",
                        "expected_regression_parameters": 676_528,
                        "expected_wgan_parameters": 821_117,
                        "capacity_profile_sha256": "l" * 64,
                    },
                ]
            ).to_csv(root / "capacity_profile_manifest.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "job_id": "reg-large-05",
                        "model": "regression",
                        "capacity_stage": "regression_screen",
                        "capacity_profile": "large",
                        "text_ablation_mode": "real_text",
                        "tolerance_minutes": 5,
                        "runtime_minutes": 8.0,
                        "gpu_hours": 8.0 / 60.0,
                        "peak_memory_mib": 1024.0,
                        "mean_utilization_gpu_pct": 70.0,
                        "status": "completed",
                    }
                ]
            ).to_csv(root / "resource_summary.csv", index=False)

            with patch.object(
                capacity_report_module,
                "_read_json",
                wraps=capacity_report_module._read_json,
            ) as read_json:
                output = render_capacity_report(root)
            html = output.read_text(encoding="utf-8")
            self.assertIn("正式容量赢家：无", html)
            self.assertIn("WGAN：未运行", html)
            self.assertIn("Q4：未读取、未评估", html)
            self.assertIn("large 只是最接近", html)
            self.assertIn("<svg", html)
            self.assertIn("Val−Train gap", html)
            self.assertNotIn("https://", html)
            self.assertTrue(
                all(
                    "final_q4" not in str(call.args[0])
                    for call in read_json.call_args_list
                )
            )

    def test_missing_q4_is_fail_closed_for_other_terminal_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "registry").mkdir(parents=True)
            (root / "registry" / "experiment_status.json").write_text(
                json.dumps({"status": "completed", "current_stage": "wgan_confirm"}),
                encoding="utf-8",
            )
            (root / "capacity_stage_status.json").write_text(
                json.dumps({"status": "completed", "stages": {}}),
                encoding="utf-8",
            )
            (root / "capacity_selection.json").write_text(
                json.dumps({"q4_used_for_selection": False, "stages": {}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Final Q4 artifacts are absent"):
                render_capacity_report(root)

    def test_partial_q4_artifacts_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "analysis").mkdir(parents=True)
            (root / "capacity_selection.json").write_text(
                json.dumps({"q4_used_for_selection": False, "stages": {}}),
                encoding="utf-8",
            )
            pd.DataFrame([{"model": "regression"}]).to_csv(
                root / "analysis" / "final_q4_model_comparison.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "Partial final_q4"):
                render_capacity_report(root)

    def test_report_supports_regression_only_when_wgan_gate_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "analysis"
            analysis.mkdir(parents=True)
            (root / "capacity_selection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "q4_used_for_selection": False,
                        "stages": {
                            "regression_confirm": {
                                "gate_passed": True,
                                "winner_profile": "micro",
                                "winner_parameter_count": 20_070,
                            },
                            "wgan_screen": {
                                "gate_passed": False,
                                "winner_profile": "tiny",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (analysis / "final_q4_validation_summary.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "regression_capacity_profile": "micro",
                        "wgan_capacity_profile": "",
                        "wgan_status": "gate_failed",
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "selection_stage": "regression_confirm",
                        "model_family": "regression",
                        "capacity_profile": "micro",
                    }
                ]
            ).to_csv(root / "capacity_comparisons.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model": "regression",
                        "capacity_profile": "micro",
                        "stratum_type": "overall",
                        "mae": 0.9,
                        "persistence_mae": 1.0,
                    }
                ]
            ).to_csv(analysis / "final_q4_model_comparison.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "model": "regression",
                        "capacity_profile": "micro",
                        "mean_diff": -0.1,
                        "p_holm": 0.04,
                    }
                ]
            ).to_csv(analysis / "final_q4_persistence_bootstrap.csv", index=False)
            pd.DataFrame(
                columns=["model", "capacity_profile", "mean_diff", "p_holm"]
            ).to_csv(analysis / "final_q4_text_ablation_bootstrap.csv", index=False)
            pd.DataFrame([{"model": "regression", "capacity_profile": "micro"}]).to_csv(
                analysis / "final_q4_pair_metrics.csv.gz",
                index=False,
                compression="gzip",
            )
            output = render_capacity_report(root)
            html = output.read_text(encoding="utf-8")
            self.assertIn("WGAN容量gate未通过", html)
            self.assertIn("micro", html)
            self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
