"""Contracts for the pure-CNN-parent FiLM text-effect prediction stage."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction as prediction,
)


ZERO_SHA = "0" * 64


def _jobs(
    *,
    seeds: tuple[int, ...] = prediction.CANONICAL_SEEDS,
    folds: tuple[str, ...] = prediction.CANONICAL_FOLDS,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for seed in seeds:
        for fold in folds:
            parent_id = f"parent__{fold}__seed_{seed}"
            result.append(
                {
                    "job_id": parent_id,
                    "seed": seed,
                    "fold": fold,
                    "arm": prediction.PARENT_ARM,
                }
            )
            for arm in prediction.BRANCH_ARMS:
                result.append(
                    {
                        "job_id": f"branch__{fold}__seed_{seed}__{arm}",
                        "parent_job_id": parent_id,
                        "seed": seed,
                        "fold": fold,
                        "arm": arm,
                    }
                )
    return result


def _allowlist(jobs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job in jobs:
        for role in ("generator_best_learned", "discriminator_best_learned"):
            rows.append(
                {
                    "job_id": job["job_id"],
                    "checkpoint_role": role,
                    "checkpoint_path": f"/not/read/{job['job_id']}/{role}.pt",
                    "checkpoint_sha256": ZERO_SHA,
                    "size_bytes": 1,
                }
            )
    return pd.DataFrame(rows)


def _inventory(jobs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job in jobs:
        if job["arm"] == prediction.PARENT_ARM:
            continue
        for label in prediction.TRAJECTORY_LABELS:
            epoch = 17 if label == "best" else int(label.split("_")[1])
            rows.append(
                {
                    "job_id": job["job_id"],
                    "checkpoint_label": label,
                    "epoch": epoch,
                    "checkpoint_path": f"/not/read/{job['job_id']}/{label}.pt",
                    "checkpoint_sha256": ZERO_SHA,
                    "size_bytes": 1,
                }
            )
    return pd.DataFrame(rows)


class TextEffectPredictionTests(unittest.TestCase):
    def test_full_prediction_universes_are_exactly_280_and_80(self) -> None:
        jobs = _jobs()
        allowlist = _allowlist(jobs)
        standard = prediction.plan_standard_prediction_units(
            jobs, allowlist, verify_checkpoint_artifacts=False
        )
        interventions = prediction.plan_intervention_prediction_units(
            jobs, allowlist, verify_checkpoint_artifacts=False
        )
        self.assertEqual(len(standard), 280)
        self.assertEqual(len(interventions), 80)
        self.assertFalse(standard["prediction_unit_id"].duplicated().any())
        self.assertFalse(interventions["prediction_unit_id"].duplicated().any())
        self.assertEqual(
            set(interventions["input_condition"]), {"zero_input", "wrong_input"}
        )
        matched_sha = standard.loc[
            standard["arm"].eq("film_lp_matched"),
            ["seed", "fold", "checkpoint_sha256"],
        ]
        intervention_sha = interventions[
            ["seed", "fold", "checkpoint_sha256"]
        ].drop_duplicates()
        self.assertEqual(
            set(matched_sha.itertuples(index=False, name=None)),
            set(intervention_sha.itertuples(index=False, name=None)),
        )

    def test_trajectory_is_sparse_seven_snapshots_per_branch(self) -> None:
        seeds = (11, 22)
        folds = ("fold_a", "fold_b")
        jobs = _jobs(seeds=seeds, folds=folds)
        units = prediction.plan_validation_trajectory_units(
            jobs,
            _inventory(jobs),
            _allowlist(jobs),
            seeds=seeds,
            folds=folds,
            verify_checkpoint_artifacts=False,
        )
        self.assertEqual(len(units), 2 * 2 * 6 * 7)
        counts = units.groupby("job_id")["checkpoint_label"].nunique()
        self.assertTrue(counts.eq(7).all())
        self.assertEqual(
            set(units["checkpoint_label"]), set(prediction.TRAJECTORY_LABELS)
        )
        self.assertTrue(units["mc_samples"].eq(16).all())
        self.assertFalse((units["split"] == "test").any())

        broken = _inventory(jobs)
        broken = broken.drop(broken.index[broken["checkpoint_label"].eq("epoch_5")][0])
        with self.assertRaisesRegex(prediction.TextEffectPredictionError, "universe"):
            prediction.plan_validation_trajectory_units(
                jobs,
                broken,
                _allowlist(jobs),
                seeds=seeds,
                folds=folds,
                verify_checkpoint_artifacts=False,
            )

    def test_noise_bank_is_shared_by_arm_condition_and_snapshot(self) -> None:
        seeds = (11,)
        folds = ("fold_a",)
        jobs = _jobs(seeds=seeds, folds=folds)
        standard = prediction.plan_standard_prediction_units(
            jobs,
            _allowlist(jobs),
            seeds=seeds,
            folds=folds,
            verify_checkpoint_artifacts=False,
        )
        interventions = prediction.plan_intervention_prediction_units(
            jobs,
            _allowlist(jobs),
            seeds=seeds,
            folds=folds,
            verify_checkpoint_artifacts=False,
        )
        combined = pd.concat([standard, interventions], ignore_index=True)
        profiled = prediction.attach_noise_bank_profiles(
            combined,
            sample_ids_by_split_fold={("test", "fold_a"): ["pair::one", "pair::two"]},
        )
        self.assertEqual(profiled["noise_bank_profile_sha256"].nunique(), 1)
        drifted = profiled.copy()
        drifted.loc[drifted.index[-1], "noise_bank_profile_sha256"] = "f" * 64
        with self.assertRaisesRegex(prediction.TextEffectPredictionError, "share"):
            prediction.validate_shared_noise_banks(drifted)

    def test_wrong_text_mapping_is_bijective_cross_session_and_independent(
        self,
    ) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": [f"p{index}" for index in range(6)],
                "session_id": ["a", "a", "b", "b", "c", "c"],
            }
        )
        first = prediction.deterministic_wrong_text_mapping(
            pairs,
            master_seed=91,
            namespace="test/fold_a",
        )
        training = dict(
            first[["pair_id", "donor_pair_id"]].itertuples(index=False, name=None)
        )
        second = prediction.deterministic_wrong_text_mapping(
            pairs,
            master_seed=92,
            namespace="test/fold_a/intervention",
            forbidden_mapping=training,
        )
        repeated = prediction.deterministic_wrong_text_mapping(
            pairs,
            master_seed=92,
            namespace="test/fold_a/intervention",
            forbidden_mapping=training,
        )
        pd.testing.assert_frame_equal(second, repeated, check_exact=True)
        self.assertEqual(set(second["donor_pair_id"]), set(pairs["pair_id"]))
        self.assertFalse(second["pair_id"].eq(second["donor_pair_id"]).any())
        self.assertFalse(second["session_id"].eq(second["donor_session_id"]).any())
        self.assertTrue(
            all(
                donor != training[receiver]
                for receiver, donor in second[["pair_id", "donor_pair_id"]].itertuples(
                    index=False, name=None
                )
            )
        )

    def test_intervention_overlays_preserve_receiver_and_move_donor_embedding(
        self,
    ) -> None:
        matched = pd.DataFrame(
            {
                "pair_id": ["p0", "p1", "p2", "p3"],
                "session_id": ["a", "a", "b", "b"],
                "embedding": [
                    np.asarray(
                        [index, index + 1, index + 2, index + 3], dtype=np.float32
                    )
                    for index in range(4)
                ],
            }
        )
        overlays = prediction.build_intervention_overlay_frames(
            matched,
            master_seed=7,
            namespace="fold_a",
            embedding_dimension=4,
        )
        self.assertEqual(set(overlays), {"zero_input", "wrong_input", "mapping"})
        self.assertTrue(
            all(
                np.count_nonzero(value) == 0
                for value in overlays["zero_input"]["embedding"]
            )
        )
        source = dict(
            matched[["pair_id", "embedding"]].itertuples(index=False, name=None)
        )
        for row in overlays["wrong_input"].itertuples(index=False):
            self.assertTrue(np.array_equal(row.embedding, source[row.donor_pair_id]))

    def test_test_gate_rejects_before_freeze_and_validates_hashes_after(self) -> None:
        jobs = _jobs(seeds=(11,), folds=("fold_a",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows: list[dict[str, object]] = []
            for job in jobs:
                for role in ("generator_best_learned", "discriminator_best_learned"):
                    checkpoint = root / f"{job['job_id']}__{role}.pt"
                    checkpoint.write_bytes(f"{job['job_id']}::{role}".encode())
                    rows.append(
                        {
                            "job_id": job["job_id"],
                            "checkpoint_role": role,
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": prediction.sha256_file(checkpoint),
                            "size_bytes": checkpoint.stat().st_size,
                        }
                    )
            allowlist = root / "allowlist.csv"
            pd.DataFrame(rows).to_csv(allowlist, index=False)
            registry = {
                "evaluation_frozen": False,
                "jobs": jobs,
                "checkpoint_allowlist_path": str(allowlist),
                "checkpoint_allowlist_sha256": prediction.sha256_file(allowlist),
            }
            with self.assertRaisesRegex(prediction.TextEffectPredictionError, "frozen"):
                prediction.require_frozen_test_access(
                    registry, expected_training_jobs=len(jobs)
                )
            registry["evaluation_frozen"] = True
            frozen = prediction.require_frozen_test_access(
                registry, expected_training_jobs=len(jobs)
            )
            self.assertEqual(len(frozen), len(jobs) * 2)
            allowlist.write_text(allowlist.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                prediction.TextEffectPredictionError, "SHA drift"
            ):
                prediction.require_frozen_test_access(
                    registry, expected_training_jobs=len(jobs)
                )

    def test_analysis_panels_use_matched_standard_once(self) -> None:
        rows: list[dict[str, object]] = []
        for arm, value in (
            ("film_lp_matched", 0.8),
            ("film_zero_text", 1.0),
            ("film_lp_shuffle", 0.9),
        ):
            rows.append(
                {
                    "arm": arm,
                    "seed": 11,
                    "fold": "fold_a",
                    "pair_id": "p0",
                    "session_id": "s0",
                    "target_mae": value,
                    "persistence_mae": 1.1,
                }
            )
        standard = pd.DataFrame(rows)
        normalized = prediction.standard_pair_metrics_for_analysis(standard)
        self.assertEqual(
            set(normalized["arm"]), {"matched", "film_zero_text", "film_lp_shuffle"}
        )
        counterfactual = pd.DataFrame(
            [
                {
                    "input_condition": condition,
                    "seed": 11,
                    "fold": "fold_a",
                    "pair_id": "p0",
                    "session_id": "s0",
                    "target_mae": value,
                }
                for condition, value in (("zero_input", 1.0), ("wrong_input", 0.95))
            ]
        )
        intervention = prediction.compose_intervention_analysis_panel(
            standard, counterfactual
        )
        self.assertEqual(
            set(intervention["input_condition"]),
            {"matched_input", "zero_input", "wrong_input"},
        )
        self.assertEqual(len(intervention), 3)

    def test_self_hashed_prediction_manifest_detects_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for name in (
                "checkpoint",
                "panel",
                "overlay",
                "prediction",
                "pair_metrics",
            ):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode())
                artifacts[name] = path
            unit = {
                "prediction_unit_id": "standard::job",
                "prediction_kind": "standard_test",
                "job_id": "job",
                "seed": 11,
                "fold": "fold_a",
                "arm": "film_lp_matched",
                "checkpoint_path": str(artifacts["checkpoint"]),
                "checkpoint_sha256": prediction.sha256_file(artifacts["checkpoint"]),
                "mc_samples": 64,
                "split": "test",
            }
            profile = prediction.noise_bank_profile_sha256(
                split="test",
                seed=11,
                fold="fold_a",
                sample_ids=["pair::p0"],
                draws=64,
            )
            manifest = prediction.build_prediction_manifest_record(
                unit,
                panel_path=artifacts["panel"],
                overlay_path=artifacts["overlay"],
                prediction_path=artifacts["prediction"],
                pair_metrics_path=artifacts["pair_metrics"],
                row_count=1,
                noise_bank_profile_sha256_value=profile,
            )
            units = pd.DataFrame([{**unit, "noise_bank_profile_sha256": profile}])
            validated = prediction.validate_prediction_manifests([manifest], units)
            self.assertEqual(len(validated), 1)
            artifacts["prediction"].write_bytes(b"drift")
            with self.assertRaisesRegex(prediction.TextEffectPredictionError, "drift"):
                prediction.validate_prediction_manifests([manifest], units)

    def test_noise_profile_rejects_wrong_mc_and_duplicate_samples(self) -> None:
        with self.assertRaisesRegex(prediction.TextEffectPredictionError, "MC=16"):
            prediction.noise_bank_profile_sha256(
                split="validation",
                seed=11,
                fold="fold_a",
                sample_ids=["one"],
                draws=64,
            )
        with self.assertRaisesRegex(prediction.TextEffectPredictionError, "unique"):
            prediction.noise_bank_profile_sha256(
                split="test",
                seed=11,
                fold="fold_a",
                sample_ids=["one", "one"],
                draws=64,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
