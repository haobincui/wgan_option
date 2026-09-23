from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from scripts.rq3.news_first_vol_lr_analysis import (
    FROZEN_LR_PROFILES,
    LearningRateAnalysisError,
    evaluate_lr_stage,
    one_standard_error_lr_selection,
    summarize_lr_scores,
)
from scripts.rq3.news_first_vol_lr_report import render_lr_sweep_report
from scripts.rq3.news_first_vol_lr_sweep import (
    FROZEN_SCHEDULER_MIN_LRS,
    _lr_profile_sha256,
)


class LearningRateFixtureMixin:
    def _write_resolved_config(
        self,
        root: Path,
        *,
        selection_rule: str = "one_standard_error_lower_lr",
        selection_mode: str = "current_only",
    ) -> None:
        payload = {
            "news_first_vol_training": {
                "lr_sweep": {
                    "enabled": True,
                    "capacity_profile": "large",
                    "profiles": FROZEN_LR_PROFILES,
                    "screen_tolerances_minutes": [5, 30],
                    "confirm_tolerances_minutes": [10, 15],
                    "text_ablation_modes": ["current_only", "real_text"],
                    "selection": {
                        "selection_panel": "common_validation_05m",
                        "screen_selection_mode": selection_mode,
                        "selection_rule": selection_rule,
                        "minimum_mean_improvement_fraction": 0.005,
                        "bootstrap_clusters": 10_000,
                    },
                }
            }
        }
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )

    def _write_job(
        self,
        root: Path,
        *,
        profile: str,
        mode: str,
        tolerance: int,
        ratio: float,
        stage: str,
    ) -> dict:
        job_id = f"lr_regression_large_{profile}_{mode}_{tolerance:02d}m"
        profile_hash = _lr_profile_sha256(profile)
        run_dir = root / "runs" / job_id
        metrics = run_dir / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        learned = {
            "best_epoch": 4,
            "best_metric": ratio,
            "selection_scope": "trained_epochs_only",
            "lr_profile": profile,
            "lr_profile_sha256": profile_hash,
            "initial_learning_rate": FROZEN_LR_PROFILES[profile],
            "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile],
            "metrics": {"val_recon": ratio, "val_current_recon": 1.0},
            "artifacts": {},
        }
        (metrics / "best_learned_checkpoint.json").write_text(
            json.dumps(learned), encoding="utf-8"
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
        (metrics / "training_metrics.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": 4,
                        "train_recon": max(1.0e-6, ratio - 0.02),
                        "val_recon": ratio,
                    }
                ]
            ),
            encoding="utf-8",
        )
        config = root / "configs" / "jobs" / f"{job_id}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            yaml.safe_dump(
                {
                    "news_first_lr_profile": profile,
                    "news_first_lr_profile_sha256": profile_hash,
                    "news_first_capacity_profile": "large",
                    "learning_rate": FROZEN_LR_PROFILES[profile],
                    "reduce_lr_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile],
                    "support_mask_mode": "raw_joint",
                    "news_first_train_end_utc": "2023-07-01T00:00:00Z",
                    "news_first_validation_end_utc": "2023-10-01T00:00:00Z",
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
            "lr_stage": stage,
            "model_family": "regression",
            "capacity_profile": "large",
            "capacity_profile_sha256": "capacity-large-sha",
            "lr_profile": profile,
            "lr_profile_sha256": profile_hash,
            "initial_learning_rate": FROZEN_LR_PROFILES[profile],
            "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile],
            "text_ablation_mode": mode,
            "tolerance_minutes": tolerance,
            "training_config_path": str(config),
        }

    def _write_registry(self, root: Path, jobs: list[dict]) -> None:
        registry = root / "registry"
        registry.mkdir(parents=True, exist_ok=True)
        (registry / "jobs.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_kind": "learning_rate_sweep",
                    "jobs": jobs,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "job_id": job["job_id"],
                    "runtime_minutes": 9.5,
                    "peak_memory_mib": 1536.0,
                    "mean_utilization_gpu_pct": 72.0,
                }
                for job in jobs
            ]
        ).to_csv(root / "resource_summary.csv", index=False)

    def _write_canonical_manifest(self, root: Path) -> None:
        pd.DataFrame(
            [
                {
                    "lr_profile": profile,
                    "lr_profile_sha256": _lr_profile_sha256(profile),
                    "initial_learning_rate": learning_rate,
                    "scheduler_min_lr": FROZEN_SCHEDULER_MIN_LRS[profile],
                    "capacity_profile": "large",
                    "gen_base_channels": 12,
                    "canonical_sentinel": "preserve-me",
                }
                for profile, learning_rate in FROZEN_LR_PROFILES.items()
            ]
        ).to_csv(root / "lr_profile_manifest.csv", index=False)

    def _screen_jobs(
        self,
        root: Path,
        *,
        current_ratios: dict[str, float],
        real_text_ratio: float = 1.5,
    ) -> list[dict]:
        jobs = []
        for profile in FROZEN_LR_PROFILES:
            for mode in ("current_only", "real_text"):
                for tolerance in (5, 30):
                    jobs.append(
                        self._write_job(
                            root,
                            profile=profile,
                            mode=mode,
                            tolerance=tolerance,
                            ratio=(
                                current_ratios[profile]
                                if mode == "current_only"
                                else real_text_ratio
                            ),
                            stage="lr_screen",
                        )
                    )
        return jobs

    def _pair_metrics(
        self,
        profiles: tuple[str, ...],
        tolerances: tuple[int, ...],
        ratios: dict[str, float],
    ) -> pd.DataFrame:
        rows = []
        for profile in profiles:
            for tolerance in tolerances:
                for index, session in enumerate(("s1", "s2", "s3")):
                    rows.append(
                        {
                            "lr_profile": profile,
                            "model": "regression",
                            "text_ablation_mode": "current_only",
                            "tolerance_minutes": tolerance,
                            "pair_id": f"p{index}",
                            "session_id": session,
                            "model_mae": ratios[profile],
                            "persistence_mae": 1.0,
                        }
                    )
        return pd.DataFrame(rows)


class LearningRateStatisticsTests(unittest.TestCase):
    def test_score_uses_equal_tolerance_log_ratio_and_strict_gate(self):
        rows = []
        for profile, ratios in {
            "lr_1e_06": (0.99, 0.99),
            "lr_3e_06": (0.98, 1.001),
        }.items():
            for tolerance, ratio in zip((5, 30), ratios):
                rows.append(
                    {
                        "lr_profile": profile,
                        "text_ablation_mode": "current_only",
                        "tolerance_minutes": tolerance,
                        "mae_ratio": ratio,
                        "log_mae_ratio": math.log(ratio),
                    }
                )
        result = summarize_lr_scores(
            pd.DataFrame(rows),
            profiles=("lr_1e_06", "lr_3e_06"),
            tolerances=(5, 30),
        ).set_index("lr_profile")
        self.assertAlmostEqual(
            float(result.loc["lr_1e_06", "mean_log_mae_ratio"]), math.log(0.99)
        )
        self.assertTrue(bool(result.loc["lr_1e_06", "lr_gate_passed"]))
        self.assertFalse(bool(result.loc["lr_3e_06", "lr_gate_passed"]))

    def test_one_se_rule_chooses_lower_lr_among_gate_passing_candidates(self):
        rows = []
        values = {
            "lr_1e_06": {"s1": 0.97, "s2": 0.97},
            "lr_3e_06": {"s1": 0.90, "s2": 1.02},
        }
        for profile, by_session in values.items():
            for tolerance in (5, 30):
                for index, (session, model_mae) in enumerate(by_session.items()):
                    rows.append(
                        {
                            "lr_profile": profile,
                            "model": "regression",
                            "text_ablation_mode": "current_only",
                            "tolerance_minutes": tolerance,
                            "pair_id": f"p{index}",
                            "session_id": session,
                            "model_mae": model_mae,
                            "persistence_mae": 1.0,
                        }
                    )
        result = one_standard_error_lr_selection(
            pd.DataFrame(rows),
            profiles=("lr_1e_06", "lr_3e_06"),
            tolerances=(5, 30),
            gate_passed_profiles=("lr_1e_06", "lr_3e_06"),
            iterations=500,
            seed=7,
        )
        self.assertEqual(result["best_gate_passing_lr_profile"], "lr_3e_06")
        self.assertIn("lr_1e_06", result["one_se_eligible_lr_profiles"])
        self.assertEqual(result["selected_lr_profile"], "lr_1e_06")


class LearningRateStageTests(LearningRateFixtureMixin, unittest.TestCase):
    def test_screen_uses_current_only_and_freezes_provisional_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_resolved_config(root)
            ratios = {
                "lr_1e_06": 0.99,
                "lr_3e_06": 0.985,
                "lr_1e_05": 0.98,
                "lr_3e_05": 0.981,
                "lr_1e_04": 1.01,
            }
            jobs = self._screen_jobs(root, current_ratios=ratios, real_text_ratio=1.8)
            self._write_registry(root, jobs)
            self._write_canonical_manifest(root)
            result = evaluate_lr_stage(
                root,
                "lr_screen",
                q3_pair_metrics=self._pair_metrics(
                    tuple(FROZEN_LR_PROFILES), (5, 30), ratios
                ),
                bootstrap_iterations=300,
                bootstrap_seed=11,
            )
            self.assertTrue(result["gate_passed"])
            self.assertEqual(result["selected_lr_profile"], "lr_1e_05")
            self.assertEqual(result["winner_lr_profile"], "")
            self.assertEqual(result["lr_profile"], "")
            self.assertTrue(result["selection_is_provisional"])
            self.assertFalse(result["q4_used_for_selection"])
            self.assertEqual(result["selection_text_ablation_mode"], "current_only")
            self.assertEqual(
                result["lr_profile_sha256"], _lr_profile_sha256("lr_1e_05")
            )
            comparisons = pd.read_csv(root / "lr_comparisons.csv")
            self.assertEqual(len(comparisons), 20)
            self.assertEqual(int(comparisons["used_for_selection"].sum()), 10)
            self.assertTrue((root / "lr_profile_manifest.csv").is_file())
            manifest = pd.read_csv(root / "lr_profile_manifest.csv")
            self.assertIn("canonical_sentinel", manifest.columns)
            self.assertEqual(set(manifest["canonical_sentinel"]), {"preserve-me"})
            audit = pd.read_csv(root / "lr_selection_audit.csv")
            self.assertEqual(int(audit.loc[0, "q4_rows_passed_to_evaluator"]), 0)
            self.assertIn(
                "diagnostic_only",
                str(result["real_text_role"]),
            )

    def test_gate_failure_clears_selection_and_keeps_ranked_leader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_resolved_config(root)
            ratios = {
                profile: 1.01 + index * 0.001
                for index, profile in enumerate(FROZEN_LR_PROFILES)
            }
            jobs = self._screen_jobs(root, current_ratios=ratios)
            self._write_registry(root, jobs)
            result = evaluate_lr_stage(
                root,
                "lr_screen",
                q3_pair_metrics=self._pair_metrics(
                    tuple(FROZEN_LR_PROFILES), (5, 30), ratios
                ),
                bootstrap_iterations=100,
            )
            self.assertFalse(result["gate_passed"])
            self.assertEqual(result["selected_lr_profile"], "")
            self.assertEqual(result["selected_lr_profiles"], [])
            self.assertEqual(result["winner_lr_profile"], "")
            self.assertEqual(result["lr_profile_sha256"], "")
            self.assertEqual(result["ranked_leader_lr_profile"], "lr_1e_06")
            (root / "registry" / "experiment_status.json").write_text(
                json.dumps({"status": "completed_no_learned_lr", "q4_accessed": False}),
                encoding="utf-8",
            )
            html = render_lr_sweep_report(root).read_text(encoding="utf-8")
            self.assertIn("没有学习率赢家", html)
            self.assertIn("仅是排名诊断", html)

    def test_confirm_regates_selected_lr_on_all_four_tolerances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_resolved_config(root)
            screen_ratios = {
                "lr_1e_06": 0.99,
                "lr_3e_06": 0.985,
                "lr_1e_05": 0.98,
                "lr_3e_05": 1.002,
                "lr_1e_04": 1.01,
            }
            jobs = self._screen_jobs(root, current_ratios=screen_ratios)
            self._write_registry(root, jobs)
            screen = evaluate_lr_stage(
                root,
                "lr_screen",
                q3_pair_metrics=self._pair_metrics(
                    tuple(FROZEN_LR_PROFILES), (5, 30), screen_ratios
                ),
                bootstrap_iterations=100,
            )
            selected = screen["selected_lr_profile"]
            self.assertEqual(selected, "lr_1e_05")
            for mode in ("current_only", "real_text"):
                for tolerance in (10, 15):
                    jobs.append(
                        self._write_job(
                            root,
                            profile=selected,
                            mode=mode,
                            tolerance=tolerance,
                            ratio=0.98 if mode == "current_only" else 1.4,
                            stage="lr_confirm",
                        )
                    )
            self._write_registry(root, jobs)
            confirm_ratios = {selected: 0.98}
            result = evaluate_lr_stage(
                root,
                "lr_confirm",
                q3_pair_metrics=self._pair_metrics(
                    (selected,), (5, 10, 15, 30), confirm_ratios
                ),
                bootstrap_iterations=100,
            )
            self.assertTrue(result["gate_passed"])
            self.assertFalse(result["selection_is_provisional"])
            self.assertEqual(result["winner_lr_profile"], selected)
            self.assertEqual(result["lr_profile"], selected)
            self.assertEqual(result["selection_tolerances"], [5, 10, 15, 30])
            selection = json.loads((root / "lr_selection.json").read_text())
            self.assertEqual(set(selection["stages"]), {"lr_screen", "lr_confirm"})

    def test_selection_contract_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_resolved_config(root, selection_mode="real_text")
            (root / "registry").mkdir()
            (root / "registry" / "jobs.json").write_text(
                json.dumps({"jobs": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                LearningRateAnalysisError, "screen_selection_mode"
            ):
                evaluate_lr_stage(root, "lr_screen", bootstrap_iterations=10)


class LearningRateReportTests(LearningRateFixtureMixin, unittest.TestCase):
    def test_stage_only_report_is_self_contained_and_declares_no_q4(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._write_resolved_config(root)
            ratios = {
                "lr_1e_06": 0.99,
                "lr_3e_06": 0.985,
                "lr_1e_05": 0.98,
                "lr_3e_05": 0.981,
                "lr_1e_04": 1.01,
            }
            jobs = self._screen_jobs(root, current_ratios=ratios)
            self._write_registry(root, jobs)
            evaluate_lr_stage(
                root,
                "lr_screen",
                q3_pair_metrics=self._pair_metrics(
                    tuple(FROZEN_LR_PROFILES), (5, 30), ratios
                ),
                bootstrap_iterations=100,
            )
            (root / "registry" / "experiment_status.json").write_text(
                json.dumps(
                    {
                        "status": "completed_q3_only",
                        "q4_accessed": False,
                    }
                ),
                encoding="utf-8",
            )
            output = render_lr_sweep_report(root)
            html = output.read_text(encoding="utf-8")
            self.assertIn("Q4 未生成预测、未评估、未参与选择", html)
            self.assertIn("Real-text 仅作诊断", html)
            self.assertIn("Val−Train gap", html)
            self.assertIn("<svg", html)
            self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()
