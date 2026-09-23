"""Observed point estimates must keep folds/seeds equally weighted."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq123 import chapter3_point_estimate_audit as audit


def panel(dataset: str) -> pd.DataFrame:
    rows = []
    for arm in audit.ARMS[dataset]:
        for seed_index, seed in enumerate(audit.SEEDS):
            for fold_index, fold in enumerate(audit.FOLDS):
                for pair in range(fold_index + 1):
                    rows.append({
                        "arm": arm, "seed": seed, "fold": fold,
                        "pair_id": f"{fold}-{pair}", "session_id": f"{fold}-session",
                        "effective_origin_utc": f"2023-01-{pair + 1:02}T12:00:00Z",
                        "target_mae": 1.0 + fold_index * 2 + seed_index / 10 + pair,
                        "persistence_mae": 10.0 + fold_index,
                        "tolerance_minutes": 5,
                        "checkpoint_sha256": f"checkpoint-{seed}-{fold}",
                        "prediction_sha256": f"prediction-{seed}-{fold}",
                        "noise_bank_profile_sha256": "noise",
                        "inference_determinism_contract_sha256": "contract",
                    })
    return pd.DataFrame(rows)


class ObservedPointAuditTests(unittest.TestCase):
    def test_unequal_fold_sizes_keep_equal_fold_and_seed_weights(self) -> None:
        frame = audit.validate_panel(panel("rq12"), audit.ARMS["rq12"])
        summaries = audit.summarise_panel(frame, "rq12")
        result = summaries["overall_summary"].set_index("arm").loc["lp_matched"]
        # Within-fold pair means: 1, 3.5, 6, 8.5; mean seed effect .45.
        self.assertAlmostEqual(result.observed_mean_mae, 5.2)
        self.assertNotAlmostEqual(result.observed_mean_mae,
                                  frame.loc[frame.arm.eq("lp_matched"), "target_mae"].mean())
        self.assertAlmostEqual(result.improvement_vs_persistence_percent, 100 * (1 - 5.2 / 11.5))
        ratio = summaries["contrasts"].query(
            "fold == 'overall' and focal == 'lp_matched' and reference == 'persistence'").iloc[0]
        cells = summaries["cell_summary"].query("arm == 'lp_matched'")
        expected = np.log(cells.observed_mean_mae / cells.persistence_mean_mae).mean()
        self.assertAlmostEqual(ratio.mean_cell_log_mae_ratio, expected)

    def test_bootstrap_summary_columns_cannot_change_points(self) -> None:
        frame = panel("rq12")
        expected = audit.summarise_panel(frame, "rq12")
        frame["bootstrap_mean_mae"] = 999.0
        frame["bootstrap_draw"] = -999.0
        actual = audit.summarise_panel(frame, "rq12")
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], actual[name])
        with self.assertRaisesRegex(ValueError, "Missing pair-level columns"):
            audit.validate_panel(pd.DataFrame({"bootstrap_mean_mae": [1.0]}), audit.ARMS["rq12"])

    def test_missing_cell_and_duplicate_rows_are_rejected(self) -> None:
        frame = panel("rq12")
        missing = frame.loc[~(frame.arm.eq("lp_matched") & frame.seed.eq(audit.SEEDS[0])
                              & frame.fold.eq(audit.FOLDS[0]))]
        with self.assertRaisesRegex(ValueError, "Missing arm/seed/fold cell"):
            audit.validate_panel(missing, audit.ARMS["rq12"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.validate_panel(pd.concat([frame, frame.iloc[[0]]]), audit.ARMS["rq12"])

    def test_missing_pair_in_one_seed_is_rejected(self) -> None:
        frame = panel("rq12")
        index = frame.index[frame.fold.eq(audit.FOLDS[-1])][0]
        with self.assertRaisesRegex(ValueError, "Unpaired market observations"):
            audit.validate_panel(frame.drop(index), audit.ARMS["rq12"])

    def test_matched_checkpoint_and_predictions_must_be_identical(self) -> None:
        standard, intervention = panel("rq3_standard"), panel("rq3_intervention")
        self.assertTrue(audit.matched_identity(standard, intervention)["passed"])
        intervention.loc[0, "checkpoint_sha256"] = "different-checkpoint"
        with self.assertRaisesRegex(ValueError, "identity failed"):
            audit.matched_identity(standard, intervention)

    def test_archive_records_hashes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for dataset in audit.DEFAULT_SOURCES:
                sources[dataset] = root / f"{dataset}.csv"
                panel(dataset).to_csv(sources[dataset], index=False)
            original = {name: audit.sha256_file(path) for name, path in sources.items()}
            output = root / "audit"
            audit.run_audit(sources, output)
            self.assertTrue((output / "audit_manifest.json").is_file())
            for name, path in sources.items():
                self.assertEqual(original[name], audit.sha256_file(path))
            with self.assertRaises(FileExistsError):
                audit.run_audit(sources, output)


if __name__ == "__main__":
    unittest.main()
