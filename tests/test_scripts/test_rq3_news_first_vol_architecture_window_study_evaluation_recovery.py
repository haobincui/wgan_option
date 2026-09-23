from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_architecture_window_study as study
from scripts.rq3 import news_first_vol_architecture_window_study_analysis as analysis
from scripts.rq3 import (
    news_first_vol_architecture_window_study_evaluation_recovery as recovery,
)


def _valid_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_news_current_overlap_minutes": [0.0, 2.0],
            "target_mae": [0.01, 0.02],
            "persistence_mae": [0.02, 0.04],
            "persistence_skill": [0.5, 0.5],
            "predicted_calendar_constraint_count": [0.0, 4.0],
            "target_calendar_constraint_count": [0.0, 4.0],
            "predicted_calendar_violation_rate": [np.nan, 0.25],
            "target_calendar_violation_rate": [np.nan, 0.5],
            "calendar_violation_rate_gap": [np.nan, -0.25],
            "predicted_butterfly_constraint_count": [0.0, 5.0],
            "target_butterfly_constraint_count": [0.0, 5.0],
            "predicted_butterfly_violation_rate": [np.nan, 0.4],
            "target_butterfly_violation_rate": [np.nan, 0.2],
            "butterfly_violation_rate_gap": [np.nan, 0.2],
        }
    )


class ArchitectureWindowEvaluationRecoveryTests(unittest.TestCase):
    def test_conditional_contract_accepts_zero_denominator_nan(self) -> None:
        recovery.validate_pair_metric_contract(_valid_metrics())

    def test_conditional_contract_rejects_nan_when_constraints_exist(self) -> None:
        frame = _valid_metrics()
        frame.loc[1, "predicted_calendar_violation_rate"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            recovery.validate_pair_metric_contract(frame)

    def test_conditional_contract_rejects_finite_rate_when_constraints_absent(
        self,
    ) -> None:
        frame = _valid_metrics()
        frame.loc[0, "predicted_calendar_violation_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "must be NaN"):
            recovery.validate_pair_metric_contract(frame)

    def test_conditional_contract_rejects_invalid_counts(self) -> None:
        for value, message in (
            (np.nan, "missing/non-finite"),
            (-1.0, "non-negative"),
            (1.5, "integer-valued"),
        ):
            with self.subTest(value=value):
                frame = _valid_metrics()
                frame.loc[1, "predicted_calendar_constraint_count"] = value
                frame.loc[1, "target_calendar_constraint_count"] = value
                with self.assertRaisesRegex(ValueError, message):
                    recovery.validate_pair_metric_contract(frame)

    def test_conditional_contract_rejects_count_mismatch(self) -> None:
        frame = _valid_metrics()
        frame.loc[1, "target_butterfly_constraint_count"] = 4.0
        with self.assertRaisesRegex(ValueError, "constraint counts differ"):
            recovery.validate_pair_metric_contract(frame)

    def test_conditional_contract_rejects_rate_domain_and_gap_drift(self) -> None:
        outside = _valid_metrics()
        outside.loc[1, "target_calendar_violation_rate"] = 1.1
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            recovery.validate_pair_metric_contract(outside)

        inconsistent = _valid_metrics()
        inconsistent.loc[1, "butterfly_violation_rate_gap"] = 0.3
        with self.assertRaisesRegex(ValueError, "gap is inconsistent"):
            recovery.validate_pair_metric_contract(inconsistent)

    def test_conditional_contract_keeps_core_metrics_fail_closed(self) -> None:
        frame = _valid_metrics()
        frame.loc[0, "target_mae"] = np.nan
        with self.assertRaisesRegex(ValueError, "target_mae.*non-finite"):
            recovery.validate_pair_metric_contract(frame)

    def test_recovery_context_restores_frozen_hooks(self) -> None:
        original_evaluator = study._evaluate_prediction_cell
        original_metrics = analysis._metrics
        original_required = analysis.PAIR_METRIC_COLUMNS
        original_finite = analysis.FINITE_COLUMNS
        with mock.patch.object(recovery, "_frozen_source_attestation", return_value={}):
            with recovery.corrected_evaluation_contract(Path("/unused")):
                self.assertIs(
                    study._evaluate_prediction_cell, recovery._evaluate_prediction_cell
                )
                self.assertIs(analysis._metrics, recovery._analysis_metrics)
                self.assertTrue(
                    recovery.CONSTRAINT_COUNT_COLUMNS.issubset(
                        analysis.PAIR_METRIC_COLUMNS
                    )
                )
                self.assertNotIn(
                    "predicted_calendar_violation_rate", analysis.FINITE_COLUMNS
                )
        self.assertIs(study._evaluate_prediction_cell, original_evaluator)
        self.assertIs(analysis._metrics, original_metrics)
        self.assertIs(analysis.PAIR_METRIC_COLUMNS, original_required)
        self.assertIs(analysis.FINITE_COLUMNS, original_finite)

    def test_frozen_source_attestation_rejects_binding_drift(self) -> None:
        bindings = []
        for relative, expected in recovery.FROZEN_SOURCE_BINDINGS.items():
            bindings.append(
                {
                    "path": str((study.REPO_ROOT / relative).resolve()),
                    **expected,
                }
            )
        registry = {
            "study_kind": study.ARCHITECTURE_KIND,
            "source_config_sha256": "config-sha",
            "source_code_bindings": bindings,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / next(iter(recovery.FORMAL_CONFIGS))
            with (
                mock.patch.object(study, "_load_registry", return_value=registry),
                mock.patch.object(study, "_verify_prepare_lineage"),
            ):
                attestation = recovery._frozen_source_attestation(root)
                self.assertEqual(len(attestation["frozen_sources"]), 3)
                drifted = {
                    **registry,
                    "source_code_bindings": [dict(row) for row in bindings],
                }
                drifted["source_code_bindings"][0]["sha256"] = "0" * 64
                with (
                    mock.patch.object(study, "_load_registry", return_value=drifted),
                    self.assertRaisesRegex(ValueError, "Frozen source drift"),
                ):
                    recovery._frozen_source_attestation(root)

    def test_terminal_resume_only_refreezes_external_control_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / next(iter(recovery.FORMAL_CONFIGS))
            root.mkdir()
            (root / "output_sha256.txt").write_text("snapshot\n", encoding="utf-8")
            registry = {
                "terminal_complete": True,
                "source_config_sha256": "config-sha",
            }
            with (
                mock.patch.object(
                    study,
                    "load_config",
                    return_value={"source_config_sha256": "config-sha"},
                ),
                mock.patch.object(study, "SupervisorLock", return_value=nullcontext()),
                mock.patch.object(study, "_load_registry", return_value=registry),
                mock.patch.object(
                    recovery, "_verify_terminal_bundles"
                ) as verify_bundles,
                mock.patch.object(study, "_verify_output_sha_manifest") as verify_root,
                mock.patch.object(
                    study, "_write_control_output_sha_manifest"
                ) as write_control,
                mock.patch.object(
                    study, "_verify_control_output_sha_manifest"
                ) as verify_control,
                mock.patch.object(study, "_write_output_sha_manifest") as write_root,
            ):
                self.assertEqual(recovery.run_evaluation_pipeline(root), root)
            verify_bundles.assert_called_once_with(root, registry)
            verify_root.assert_called_once_with(root)
            write_control.assert_called_once()
            verify_control.assert_called_once()
            write_root.assert_not_called()

    def test_recovery_completes_manifest_before_qa_without_post_qa_root_write(
        self,
    ) -> None:
        root = Path("/tmp") / next(iter(recovery.FORMAL_CONFIGS))
        registry = {
            "terminal_complete": False,
            "evaluation_frozen": True,
            "source_config_sha256": "config-sha",
            "study_kind": study.ARCHITECTURE_KIND,
            "jobs": [],
        }
        events: list[str] = []

        def record_manifest(
            _root: Path, *, status: str, details: object = None
        ) -> Path:
            del details
            events.append(f"manifest:{status}")
            return _root / recovery.RECOVERY_PATH

        def run_stage(_root: Path, name: str, operation: object) -> object:
            events.append(f"stage:{name}")
            return operation()

        def record_qa(_root: Path) -> Path:
            events.append("qa-called")
            return _root / "qa.json"

        with (
            mock.patch.object(
                study,
                "load_config",
                return_value={"source_config_sha256": "config-sha"},
            ),
            mock.patch.object(study, "SupervisorLock", return_value=nullcontext()),
            mock.patch.object(study, "_load_registry", return_value=registry),
            mock.patch.object(study, "_all_complete", return_value=True),
            mock.patch.object(study, "_verify_evaluation_bundle"),
            mock.patch.object(
                recovery, "_write_recovery_manifest", side_effect=record_manifest
            ),
            mock.patch.object(
                recovery, "corrected_evaluation_contract", return_value=nullcontext()
            ),
            mock.patch.object(recovery, "_stage", side_effect=run_stage),
            mock.patch.object(study, "predict", return_value=root / "predictions.csv"),
            mock.patch.object(study, "analyze", return_value=root / "analysis.json"),
            mock.patch.object(study, "bootstrap", return_value=root / "bootstrap.json"),
            mock.patch.object(study, "report", return_value=root / "report.json"),
            mock.patch.object(study, "qa", side_effect=record_qa),
            mock.patch.object(study, "append_stage_journal"),
            mock.patch.object(study, "_append_resource_snapshot"),
            mock.patch.object(
                study, "_write_control_output_sha_manifest"
            ) as write_control,
            mock.patch.object(study, "_write_output_sha_manifest") as write_root,
        ):
            self.assertEqual(recovery.run_evaluation_pipeline(root), root)
        self.assertLess(events.index("manifest:completed"), events.index("qa-called"))
        write_control.assert_called_once()
        write_root.assert_not_called()


if __name__ == "__main__":
    unittest.main()
