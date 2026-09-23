from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.rq3.news_first_vol_current_input_analysis import (
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    EXPECTED_PAIR_ROWS,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    PERSISTENCE_MATCH_ATOL,
    CurrentInputAnalysisError,
    _reference_job_id,
    _sha256_file,
    _validation_lineage,
    build_cell_summary,
    build_combined_summary,
    build_cross_seed_summary,
    build_current_input_bootstrap,
    build_paired_pair_metrics,
    build_persistence_bootstrap,
    freeze_full_current_reference_manifest,
    load_full_current_reference_evidence,
    run_current_input_analysis,
    validate_mode_pair_metrics,
)
from scripts.rq3.news_first_vol_current_input_report import (
    CurrentInputReportError,
    render_current_input_report,
)
from wgan_option.models.common import generator_current_input_fingerprint


SEEDS = (42, 202, 404)
TOLERANCES = (5, 30)


def _pair_metrics(mode: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for tolerance in TOLERANCES:
            for index in range(123):
                persistence = 0.001 + index * 1.0e-7
                full = (
                    persistence
                    + 3.0e-6
                    + (seed % 10) * 1.0e-8
                    + (tolerance - 5) * 1.0e-9
                    + (index % 5) * 1.0e-8
                )
                masked = full - 2.0e-6 + (index % 3 - 1) * 1.0e-8
                run_id = (
                    _reference_job_id(seed, tolerance)
                    if mode == FULL_CURRENT_GENERATOR_INPUT_MODE
                    else f"masked_seed_{seed:03d}_{tolerance:02d}m"
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "model": "wgan",
                        "text_ablation_mode": "real_text",
                        "support_mask_mode": "raw_joint",
                        "generator_current_input_mode": mode,
                        "generator_current_input_fingerprint": (
                            generator_current_input_fingerprint(mode)
                        ),
                        "seed": seed,
                        "tolerance_minutes": tolerance,
                        "panel": "common_validation_05m",
                        "stratum_type": "overall",
                        "stratum_value": "all",
                        "pair_id": f"pair_{index:03d}",
                        "session_id": f"cme_session_{index % 33:02d}",
                        "model_mae": (
                            masked
                            if mode == CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                            else full
                        ),
                        "persistence_mae": persistence,
                        "selection_metadata_sha256": "fixture-selection",
                        "generator_checkpoint_sha256": "fixture-checkpoint",
                    }
                )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_reference_root(root: Path) -> Path:
    registry_dir = root / "registry"
    status_dir = registry_dir / "jobs"
    status_dir.mkdir(parents=True)
    pair_path = (
        root
        / "analysis"
        / "stages"
        / "stage_3_wgan_low_lr_capacity"
        / "q3_pair_metrics.csv.gz"
    )
    pair_path.parent.mkdir(parents=True)
    _pair_metrics(FULL_CURRENT_GENERATOR_INPUT_MODE).drop(
        columns=[
            "generator_current_input_mode",
            "generator_current_input_fingerprint",
        ]
    ).to_csv(pair_path, index=False, compression="gzip")
    jobs = []
    for seed in SEEDS:
        for tolerance in TOLERANCES:
            job_id = _reference_job_id(seed, tolerance)
            run_dir = root / "runs" / job_id
            metrics_dir = run_dir / "metrics"
            checkpoint_dir = run_dir / "checkpoints"
            metrics_dir.mkdir(parents=True)
            checkpoint_dir.mkdir(parents=True)
            config_path = metrics_dir / "training_resolved_config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "support_mask_mode": "raw_joint",
                        "seed": seed,
                        "news_first_dataset_tolerance_minutes": tolerance,
                    }
                ),
                encoding="utf-8",
            )
            metadata_path = metrics_dir / "best_learned_checkpoint.json"
            _write_json(
                metadata_path,
                {
                    "best_epoch": 1,
                    "best_learned_epoch_ge_1": 1,
                    "selection_scope": "trained_epochs_only",
                    "artifacts": {},
                },
            )
            checkpoint_path = checkpoint_dir / "generator_best_learned.pt"
            torch.save(
                {"config": {"support_mask_mode": "raw_joint"}, "state_dict": {}},
                checkpoint_path,
            )
            artifacts = []
            for role, path in (
                ("generator_best_learned", checkpoint_path),
                ("best_learned_checkpoint", metadata_path),
                ("resolved_training_config", config_path),
            ):
                artifacts.append(
                    {
                        "artifact_role": role,
                        "path": str(path.resolve()),
                        "sha256": _sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            config_sha = _sha256_file(config_path)
            jobs.append(
                {
                    "job_id": job_id,
                    "stage_id": "stage_3_wgan_low_lr_capacity",
                    "model_family": "wgan",
                    "capacity_profile": "small",
                    "lr_profile": "lr_5e_07",
                    "initial_learning_rate": 5.0e-7,
                    "seed": seed,
                    "text_ablation_mode": "real_text",
                    "tolerance_minutes": tolerance,
                    "support_mask_mode": "raw_joint",
                    "config_sha256": config_sha,
                }
            )
            _write_json(
                status_dir / f"{job_id}.status.json",
                {
                    "job_id": job_id,
                    "status": "completed",
                    "config_sha256": config_sha,
                    "run_dir": str(run_dir.resolve()),
                    "artifacts": artifacts,
                },
            )
    _write_json(
        registry_dir / "jobs.json",
        {
            "schema_version": 1,
            "experiment_kind": "coverage_completion_sweep",
            "jobs": jobs,
        },
    )
    return root


class TestCurrentInputAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.masked = _pair_metrics(CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE)
        self.full = _pair_metrics(FULL_CURRENT_GENERATOR_INPUT_MODE)

    def test_exact_mode_key_pairing_and_summaries(self):
        masked = validate_mode_pair_metrics(
            self.masked,
            expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        )
        full = validate_mode_pair_metrics(
            self.full,
            expected_mode=FULL_CURRENT_GENERATOR_INPUT_MODE,
        )
        self.assertEqual(len(masked), EXPECTED_PAIR_ROWS)
        self.assertEqual(len(full), EXPECTED_PAIR_ROWS)
        paired = build_paired_pair_metrics(masked, full)
        self.assertEqual(len(paired), EXPECTED_PAIR_ROWS)
        self.assertTrue(
            np.allclose(
                paired["masked_minus_full_current"],
                -2.0e-6
                + np.tile(
                    np.asarray([(index % 3 - 1) * 1.0e-8 for index in range(123)]),
                    6,
                ),
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        cell = build_cell_summary(paired)
        self.assertEqual(len(cell), 6)
        cross_seed = build_cross_seed_summary(cell)
        combined = build_combined_summary(cell)
        for frame in (paired, cell, cross_seed, combined):
            self.assertEqual(
                set(frame["generator_current_input_mode"]),
                {CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE},
            )
            self.assertEqual(
                set(frame["reference_generator_current_input_mode"]),
                {FULL_CURRENT_GENERATOR_INPUT_MODE},
            )
        self.assertEqual(len(cross_seed), 2)
        self.assertEqual(len(combined), 1)
        bad = self.masked.copy()
        bad.loc[0, "generator_current_input_mode"] = FULL_CURRENT_GENERATOR_INPUT_MODE
        with self.assertRaisesRegex(CurrentInputAnalysisError, "mode drifted"):
            validate_mode_pair_metrics(
                bad,
                expected_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            )

    def test_seed_then_session_bootstrap_and_holm_families(self):
        paired = build_paired_pair_metrics(self.masked, self.full)
        comparisons = build_current_input_bootstrap(paired)
        self.assertEqual(len(comparisons), 3)
        primary = comparisons[comparisons["analysis_role"].eq("primary")]
        secondary = comparisons[comparisons["analysis_role"].eq("secondary_tolerance")]
        self.assertEqual(len(primary), 1)
        self.assertEqual(set(secondary["tolerance_scope"]), {"05m", "30m"})
        self.assertTrue(
            comparisons["resampling_method"]
            .eq("seed_then_paired_CME_session_cluster")
            .all()
        )
        self.assertTrue(comparisons["bootstrap_iterations"].eq(10_000).all())
        self.assertTrue(comparisons["seed_count"].eq(3).all())
        self.assertTrue(comparisons["session_count_per_seed"].eq(33).all())
        self.assertEqual(
            set(comparisons["generator_current_input_mode"]),
            {CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE},
        )
        self.assertTrue(
            (
                secondary["holm_adjusted_p"].to_numpy(dtype=float)
                >= secondary["p_two_sided"].to_numpy(dtype=float)
            ).all()
        )
        persistence = build_persistence_bootstrap(paired)
        self.assertEqual(len(persistence), 6)
        self.assertEqual(
            set(persistence["generator_current_input_mode"]),
            {
                CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
                FULL_CURRENT_GENERATOR_INPUT_MODE,
            },
        )

    def test_validation_lineage_tolerates_roundtrip_noise_but_not_real_drift(self):
        roundtripped = self.masked.copy()
        roundtripped["persistence_mae"] = (
            roundtripped["persistence_mae"].astype(float) + 5.0e-16
        )
        lineage = _validation_lineage(roundtripped, self.full)
        self.assertLessEqual(
            lineage["max_abs_persistence_diff_vs_canonical"],
            PERSISTENCE_MATCH_ATOL,
        )
        self.assertEqual(lineage["observed_persistence_vector_sha256_count"], 2)
        self.assertEqual(
            {row["persistence_vector_sha256"] for row in lineage["cells"]},
            {lineage["persistence_vector_sha256"]},
        )
        self.assertEqual(
            len(
                {row["observed_persistence_vector_sha256"] for row in lineage["cells"]}
            ),
            2,
        )

        drifted = self.masked.copy()
        drifted.loc[0, "persistence_mae"] += 2.0 * PERSISTENCE_MATCH_ATOL
        with self.assertRaisesRegex(
            CurrentInputAnalysisError, "persistence differs by more than"
        ):
            _validation_lineage(drifted, self.full)

        wrong_panel = self.masked.copy()
        wrong_panel.loc[0, "session_id"] = "different_CME_session"
        with self.assertRaisesRegex(
            CurrentInputAnalysisError, "pair/session panel differs"
        ):
            _validation_lineage(wrong_panel, self.full)

    def test_reference_manifest_is_frozen_and_root_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            reference = _build_reference_root(temp / "reference")
            experiment = temp / "masked"
            experiment.mkdir()
            manifest = freeze_full_current_reference_manifest(
                experiment,
                reference_root=reference,
            )
            self.assertTrue(manifest.is_file())
            metrics, selection, lineage = load_full_current_reference_evidence(
                experiment,
                reference_root=reference,
            )
            self.assertEqual(len(metrics), EXPECTED_PAIR_ROWS)
            self.assertEqual(len(selection), 6)
            self.assertEqual(
                set(metrics["generator_current_input_mode"]),
                {FULL_CURRENT_GENERATOR_INPUT_MODE},
            )
            self.assertEqual(lineage["reference_root"], str(reference.resolve()))
            different = temp / "different-reference"
            different.mkdir()
            with self.assertRaisesRegex(
                CurrentInputAnalysisError, "differs from requested root"
            ):
                load_full_current_reference_evidence(
                    experiment,
                    reference_root=different,
                )
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CurrentInputAnalysisError, "companion hash mismatch"
            ):
                load_full_current_reference_evidence(
                    experiment,
                    reference_root=reference,
                )

    def test_formal_analysis_requires_manifest_before_job_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(
                CurrentInputAnalysisError, "requires the frozen"
            ):
                run_current_input_analysis(root, reference_root=root / "reference")

    def test_end_to_end_fixture_outputs_q3_only_self_contained_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = run_current_input_analysis(
                root,
                masked_pair_metrics=self.masked,
                full_current_pair_metrics=self.full,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["bootstrap_iterations"], 10_000)
            self.assertEqual(summary["paired_rows"], EXPECTED_PAIR_ROWS)
            self.assertFalse(summary["mask_channel_added"])
            self.assertEqual(
                summary["residual_anchor"],
                "original_unmasked_full_current_surface",
            )
            self.assertFalse(summary["q4_predictions_generated"])
            self.assertFalse(summary["q4_evaluated"])
            self.assertEqual(
                summary["reference_evidence_design"],
                "historical_immutable_coverage_experiment",
            )
            self.assertFalse(summary["reference_retrained_concurrently"])
            report = render_current_input_report(root)
            rendered = report.read_text(encoding="utf-8")
            self.assertIn("current_surface × current_support_mask", rendered)
            self.assertIn("Residual/persistence anchor", rendered)
            self.assertIn("没有增加 mask channel", rendered)
            self.assertIn("没有生成或评价 Q4", rendered)
            self.assertIn("不是本轮与 masked 模型同期重训", rendered)
            self.assertIn("进入 evaluator 的 Q4 rows 为 0", rendered)
            self.assertNotIn("<script src=", rendered)
            cell_path = (
                root
                / "analysis"
                / "current_input_ablation"
                / "current_input_cell_summary.csv"
            )
            cell_path.write_text(
                cell_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(CurrentInputReportError, "artifact hash drift"):
                render_current_input_report(root)


if __name__ == "__main__":
    unittest.main()
