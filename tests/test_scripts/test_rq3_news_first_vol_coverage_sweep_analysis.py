from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq3.news_first_vol_capacity_analysis import PROFILE_PARAMETER_COUNTS
from scripts.rq3.news_first_vol_coverage_sweep_analysis import (
    COMBINATION_KEY,
    FIXED_LEARNING_RATE,
    LEARNING_RATES,
    MAE_NUMERICAL_TIE_TOLERANCE,
    PROFILES,
    SEEDS,
    STAGE_1,
    STAGE_2,
    STAGE_3,
    STAGE_4,
    STAGE_CONTRACTS,
    STAGE_ORDER,
    CoverageSweepAnalysisError,
    build_seed_run_metrics,
    build_stage_bootstrap,
    build_stage_scores,
    compose_stage_evidence,
    run_coverage_sweep_analysis,
    summarize_across_seeds,
    validate_new_job_matrix,
    validate_stage_pair_metrics,
)
from scripts.rq3.news_first_vol_coverage_sweep_report import (
    CoverageSweepReportError,
    render_coverage_sweep_report,
)


PAIR_COUNT = 6
SESSION_COUNT = 3


def _ratio(
    model: str,
    profile: str,
    rate: float,
    mode: str,
    tolerance: int,
    seed: int,
) -> float:
    capacity = {
        "micro": -0.00015,
        "tiny": -0.00022,
        "small": -0.00030,
        "medium": -0.00042,
        "large": -0.00035,
        "legacy": -0.00008,
    }[profile]
    lr = -0.00012 * (1.0 - abs(np.log10(rate / 7.5e-7)))
    text = {"current_only": 0.0, "real_text": -0.00004, "text_shuffle": 0.00006}[mode]
    model_offset = 0.00010 if model == "wgan" else 0.0
    seed_offset = {42: -0.000015, 202: 0.0, 404: 0.000015}[seed]
    tolerance_offset = (tolerance - 15) * 1.0e-6
    return 1.0 + capacity + lr + text + model_offset + seed_offset + tolerance_offset


def _metrics_for_keys(
    keys: set[tuple[object, ...]], *, evidence_source: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, profile, rate, tolerance, mode, seed in sorted(keys):
        for pair_index in range(PAIR_COUNT):
            persistence = 0.0012 + pair_index * 1.0e-5
            session_index = pair_index % SESSION_COUNT
            session_effect = (session_index - 1) * 2.0e-5
            ratio = (
                _ratio(
                    str(model),
                    str(profile),
                    float(rate),
                    str(mode),
                    int(tolerance),
                    int(seed),
                )
                + session_effect
            )
            rows.append(
                {
                    "model_family": model,
                    "capacity_profile": profile,
                    "parameter_count": PROFILE_PARAMETER_COUNTS[str(profile)][
                        str(model)
                    ],
                    "initial_learning_rate": rate,
                    "lr_profile": f"lr_{rate}",
                    "seed": seed,
                    "tolerance_minutes": tolerance,
                    "text_ablation_mode": mode,
                    "pair_id": f"pair_{pair_index:02d}",
                    "session_id": f"session_{session_index:02d}",
                    "model_mae": persistence * ratio,
                    "persistence_mae": persistence,
                    "support_mask_mode": "raw_joint",
                    "panel": "common_validation_05m",
                    "evidence_source": evidence_source,
                }
            )
    return pd.DataFrame(rows)


def _new_keys(stage_id: str) -> set[tuple[object, ...]]:
    contract = STAGE_CONTRACTS[stage_id]
    modes = ("text_shuffle",) if stage_id == STAGE_2 else contract.text_modes
    profiles = (
        tuple(profile for profile in PROFILES if profile != "large")
        if stage_id == STAGE_4
        else contract.profiles
    )
    rates = (
        tuple(rate for rate in LEARNING_RATES if rate != FIXED_LEARNING_RATE)
        if stage_id == STAGE_4
        else contract.learning_rates
    )
    return {
        (model, profile, rate, tolerance, mode, seed)
        for model in contract.model_families
        for profile in profiles
        for rate in rates
        for tolerance in contract.tolerances
        for mode in modes
        for seed in SEEDS
    }


def _new_stage_metrics() -> dict[str, pd.DataFrame]:
    return {
        stage_id: _metrics_for_keys(
            _new_keys(stage_id), evidence_source=f"new_{stage_id}"
        )
        for stage_id in STAGE_ORDER
    }


def _fixed_reference() -> pd.DataFrame:
    keys = {
        ("regression", profile, FIXED_LEARNING_RATE, tolerance, mode, seed)
        for profile in PROFILES
        for tolerance in (5, 30)
        for mode in ("current_only", "real_text")
        for seed in SEEDS
    }
    return _metrics_for_keys(keys, evidence_source="fixed_reference")


def _local_reference() -> pd.DataFrame:
    keys = {
        ("regression", "large", rate, tolerance, mode, seed)
        for rate in LEARNING_RATES
        for tolerance in (5, 30)
        for mode in ("current_only", "real_text")
        for seed in SEEDS
    }
    return _metrics_for_keys(keys, evidence_source="local_reference")


class CoverageSweepMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.new = _new_stage_metrics()
        cls.fixed = _fixed_reference()
        cls.local = _local_reference()

    def _evidence(self, stage_id: str) -> pd.DataFrame:
        frame = compose_stage_evidence(
            stage_id,
            self.new,
            fixed_reference=self.fixed,
            local_lr_reference=self.local,
        )
        return validate_stage_pair_metrics(
            stage_id,
            frame,
            expected_pair_count=PAIR_COUNT,
            expected_session_count=SESSION_COUNT,
        )

    def test_combination_key_and_exact_stage_grids(self) -> None:
        self.assertEqual(
            COMBINATION_KEY,
            (
                "model_family",
                "capacity_profile",
                "initial_learning_rate",
                "tolerance_minutes",
                "text_ablation_mode",
                "seed",
            ),
        )
        expected_cells = {STAGE_1: 72, STAGE_2: 216, STAGE_3: 72, STAGE_4: 360}
        for stage_id, count in expected_cells.items():
            frame = self._evidence(stage_id)
            self.assertEqual(
                frame[list(COMBINATION_KEY)].drop_duplicates().shape[0], count
            )
            self.assertEqual(len(frame), count * PAIR_COUNT)
        self.assertEqual(
            set(self._evidence(STAGE_2)["evidence_source"]),
            {
                "fixed_reference",
                f"new_{STAGE_1}",
                f"new_{STAGE_2}",
            },
        )

    def test_each_stage_has_three_seed_summary_and_cluster_bootstrap(self) -> None:
        for stage_id in STAGE_ORDER:
            evidence = self._evidence(stage_id)
            seed_runs = build_seed_run_metrics(evidence)
            across = summarize_across_seeds(seed_runs)
            self.assertTrue((across["seed_count"] == 3).all())
            scores = build_stage_scores(stage_id, seed_runs)
            bootstrap = build_stage_bootstrap(
                stage_id, evidence, scores, iterations=8, random_seed=700
            )
            self.assertTrue(
                bootstrap["resampling_method"]
                .eq("seed_then_paired_CME_session_cluster")
                .all()
            )
            self.assertTrue(np.isfinite(bootstrap["ci_95_lower"]).all())
        stage4 = build_stage_bootstrap(
            STAGE_4,
            self._evidence(STAGE_4),
            build_stage_scores(
                STAGE_4, build_seed_run_metrics(self._evidence(STAGE_4))
            ),
            iterations=6,
            random_seed=701,
        )
        self.assertEqual(
            len(
                stage4[
                    stage4["contrast_family"].eq(
                        "capacity_by_learning_rate_interaction"
                    )
                ]
            ),
            5 * 4 * 2 * 3,
        )

    def test_pair_win_rate_uses_frozen_numerical_tie_tolerance(self) -> None:
        rows = []
        persistence = 0.001
        for pair_id, gap in enumerate((-2.0e-8, -0.5e-8, 0.0, 2.0e-8)):
            rows.append(
                {
                    "stage_id": STAGE_1,
                    "model_family": "regression",
                    "capacity_profile": "micro",
                    "initial_learning_rate": FIXED_LEARNING_RATE,
                    "tolerance_minutes": 10,
                    "text_ablation_mode": "current_only",
                    "seed": 42,
                    "parameter_count": PROFILE_PARAMETER_COUNTS["micro"]["regression"],
                    "pair_id": f"pair_{pair_id}",
                    "session_id": "session_0",
                    "model_mae": persistence + gap,
                    "persistence_mae": persistence,
                }
            )
        summary = build_seed_run_metrics(pd.DataFrame(rows))
        self.assertEqual(MAE_NUMERICAL_TIE_TOLERANCE, 1.0e-8)
        self.assertAlmostEqual(float(summary.iloc[0]["pair_win_rate"]), 0.25)

        # When the upstream pair aggregator supplied its canonical indicator,
        # retain it instead of reclassifying a serialized boundary value.
        with_win = pd.DataFrame(rows)
        with_win["win"] = (1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(
            float(build_seed_run_metrics(with_win).iloc[0]["pair_win_rate"]),
            0.25,
        )

    def test_missing_seed_and_q4_panel_fail_closed(self) -> None:
        evidence = self._evidence(STAGE_1)
        first = evidence.iloc[0]
        broken = evidence[
            ~(
                evidence["model_family"].eq(first["model_family"])
                & evidence["capacity_profile"].eq(first["capacity_profile"])
                & np.isclose(
                    evidence["initial_learning_rate"],
                    float(first["initial_learning_rate"]),
                )
                & evidence["tolerance_minutes"].eq(first["tolerance_minutes"])
                & evidence["text_ablation_mode"].eq(first["text_ablation_mode"])
                & evidence["seed"].eq(first["seed"])
            )
        ]
        with self.assertRaises(CoverageSweepAnalysisError):
            validate_stage_pair_metrics(
                STAGE_1,
                broken,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
            )
        q4 = evidence.copy()
        q4["panel"] = "common_test_q4"
        with self.assertRaises(CoverageSweepAnalysisError):
            validate_stage_pair_metrics(
                STAGE_1,
                q4,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
            )

    def test_new_job_registry_contract(self) -> None:
        jobs = []
        for index, key in enumerate(sorted(_new_keys(STAGE_4))):
            model, profile, rate, tolerance, mode, seed = key
            jobs.append(
                {
                    "job_id": f"job_{index}",
                    "status": "completed",
                    "stage_id": STAGE_4,
                    "model_family": model,
                    "capacity_profile": profile,
                    "initial_learning_rate": rate,
                    "tolerance_minutes": tolerance,
                    "text_ablation_mode": mode,
                    "seed": seed,
                }
            )
        self.assertEqual(len(validate_new_job_matrix(STAGE_4, jobs)), 240)
        jobs[0]["seed"] = 999
        with self.assertRaises(CoverageSweepAnalysisError):
            validate_new_job_matrix(STAGE_4, jobs)


class CoverageSweepArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.new = _new_stage_metrics()
        cls.fixed = _fixed_reference()
        cls.local = _local_reference()

    def test_full_injected_analysis_freezes_candidate_without_q4_and_renders(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            validation_path = run_coverage_sweep_analysis(
                root,
                stage_pair_metrics=self.new,
                fixed_reference_pair_metrics=self.fixed,
                local_lr_reference_pair_metrics=self.local,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=6,
                bootstrap_seed=702,
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(validation["all_four_stages_complete"])
            self.assertFalse(validation["minimum_improvement_gate_applied"])
            self.assertFalse(validation["q4_used_for_selection"])
            self.assertFalse(validation["q4_predictions_generated"])
            self.assertEqual(
                validation["numerical_tie_policy"]["mae_gap_tolerance_iv"],
                1.0e-8,
            )
            final = json.loads(
                (root / "coverage_final_candidate.json").read_text(encoding="utf-8")
            )
            self.assertTrue(final["candidate_frozen"])
            self.assertFalse(final["q4_evaluation_permitted_by_this_artifact"])
            report = render_coverage_sweep_report(root)
            html = report.read_text(encoding="utf-8")
            self.assertIn("四阶段Q3覆盖测试已完成", html)
            self.assertIn("capacity×LR", html)
            self.assertIn("Q4保持锁定", html)
            self.assertIn("5/10/15/30分钟", html)
            self.assertIn("MAE gap &lt; -1e-8 IV", html)
            self.assertIn("<svg", html)
            self.assertNotIn("https://", html)
            self.assertNotIn("http://", html)

    def test_report_distinguishes_process_hours_from_device_gpu_hours(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_coverage_sweep_analysis(
                root,
                stage_pair_metrics=self.new,
                fixed_reference_pair_metrics=self.fixed,
                local_lr_reference_pair_metrics=self.local,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=2,
                bootstrap_seed=709,
            )
            pd.DataFrame(
                [
                    {
                        "job_id": "a",
                        "stage_id": STAGE_1,
                        "global_wave": 1,
                        "gpu_id": 0,
                        "gpu_hours": 1.0,
                        "peak_memory_mib": 100.0,
                        "mean_utilization_gpu_pct": 75.0,
                    },
                    {
                        "job_id": "b",
                        "stage_id": STAGE_1,
                        "global_wave": 1,
                        "gpu_id": 1,
                        "gpu_hours": 1.0,
                        "peak_memory_mib": 110.0,
                        "mean_utilization_gpu_pct": 80.0,
                    },
                ]
            ).to_csv(root / "resource_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "job_id": "a",
                        "stage_id": STAGE_1,
                        "global_wave": 1,
                        "gpu_id": 0,
                        "started_at_utc": "2026-08-20T00:00:00Z",
                        "completed_at_utc": "2026-08-20T01:00:00Z",
                    },
                    {
                        "job_id": "b",
                        "stage_id": STAGE_1,
                        "global_wave": 1,
                        "gpu_id": 1,
                        "started_at_utc": "2026-08-20T00:00:00Z",
                        "completed_at_utc": "2026-08-20T01:00:00Z",
                    },
                ]
            ).to_csv(root / "task_registry.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "timestamp_utc": "2026-08-20T00:30:00Z",
                        "wave": 1,
                        "sample_status": "ok",
                        "utilization_gpu_pct": 80.0,
                        "memory_used_mib": 120.0,
                    }
                ]
            ).to_csv(root / "resource_usage.csv", index=False)
            html = render_coverage_sweep_report(root).read_text(encoding="utf-8")
            self.assertIn("process-hours", html)
            self.assertIn("device GPU-hours", html)
            self.assertNotIn(">GPU-hours<", html)

    def test_full_analysis_can_be_recomputed_from_hashed_pair_caches(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_coverage_sweep_analysis(
                root,
                stage_pair_metrics=self.new,
                fixed_reference_pair_metrics=self.fixed,
                local_lr_reference_pair_metrics=self.local,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=2,
                bootstrap_seed=707,
            )
            validation_path = run_coverage_sweep_analysis(
                root,
                fixed_reference_pair_metrics=self.fixed,
                local_lr_reference_pair_metrics=self.local,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=2,
                bootstrap_seed=707,
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertTrue(validation["all_four_stages_complete"])
            self.assertEqual(validation["analysed_stages"], list(STAGE_ORDER))
            self.assertTrue(
                all(
                    validation["panel_lineage"][stage]["reused_hashed_q3_pair_cache"]
                    for stage in STAGE_ORDER
                )
            )
            self.assertTrue((root / "coverage_final_candidate.json").is_file())

    def test_partial_analysis_does_not_freeze_final_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_coverage_sweep_analysis(
                root,
                stages=(STAGE_1,),
                stage_pair_metrics={STAGE_1: self.new[STAGE_1]},
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=4,
                bootstrap_seed=703,
            )
            self.assertFalse((root / "coverage_final_candidate.json").exists())
            report = render_coverage_sweep_report(root)
            self.assertIn(
                "已完成 1/4 个Q3阶段",
                report.read_text(encoding="utf-8"),
            )

    def test_stage_two_local_invocation_reuses_hashed_stage_one_pair_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_coverage_sweep_analysis(
                root,
                stages=(STAGE_1,),
                stage_pair_metrics={STAGE_1: self.new[STAGE_1]},
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=4,
                bootstrap_seed=705,
            )
            validation_path = run_coverage_sweep_analysis(
                root,
                stages=(STAGE_2,),
                stage_pair_metrics={STAGE_2: self.new[STAGE_2]},
                fixed_reference_pair_metrics=self.fixed,
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=4,
                bootstrap_seed=706,
            )
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            self.assertEqual(validation["analysed_stages"], [STAGE_2])
            self.assertEqual(
                validation["stage_validations"][STAGE_2]["evidence_job_count"], 216
            )

    def test_report_rejects_hashed_analysis_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            run_coverage_sweep_analysis(
                root,
                stages=(STAGE_1,),
                stage_pair_metrics={STAGE_1: self.new[STAGE_1]},
                expected_pair_count=PAIR_COUNT,
                expected_session_count=SESSION_COUNT,
                bootstrap_iterations=4,
                bootstrap_seed=704,
            )
            scores = root / "analysis" / "coverage_q3_scores.csv"
            scores.write_text(
                scores.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaises(CoverageSweepReportError):
                render_coverage_sweep_report(root)


if __name__ == "__main__":
    unittest.main()
