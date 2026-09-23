"""Contracts for the single-seed direct-training FiLM U-Net experiment."""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as experiment


EXPECTED_ARMS = (
    "lp_matched",
    "lp_shuffle",
    "no_text",
    "bow",
    "sentiment",
)
EXPECTED_FOLDS = (
    "f1_2023q1",
    "f2_2023q2",
    "f3_2023q3",
    "f4_2023q4",
)
FORBIDDEN_LINEAGE_KEYS = {
    "parent_checkpoint",
    "parent_job_id",
    "parent_state_allowlist_path",
    "parent_state_allowlist_sha256",
    "continuation_checkpoint",
    "branch_recipe",
    "branch_recipe_path",
    "branch_recipe_sha256",
    "resume_checkpoint",
    "generator_checkpoint",
    "discriminator_checkpoint",
    "refit_mode",
    "lr_replay",
}


def _synthetic_pair_metrics() -> pd.DataFrame:
    """Return paired evidence whose focal/reference MAE ratio is always 1/2."""

    rows: list[dict[str, object]] = []
    for fold_index, fold in enumerate(EXPECTED_FOLDS):
        for pair_index in range(4):
            pair_id = f"{fold}-pair-{pair_index}"
            session_id = f"{fold}-session-{pair_index // 2}"
            focal_mae = 1.0 + 0.1 * fold_index + 0.01 * pair_index
            common = {
                "fold": fold,
                "pair_id": pair_id,
                "session_id": session_id,
                "effective_origin_utc": (
                    f"2023-0{fold_index + 1}-01T00:{pair_index:02d}:00Z"
                ),
                "persistence_mae": 3.0,
            }
            rows.append({**common, "arm": "lp_matched", "target_mae": focal_mae})
            rows.append({**common, "arm": "no_text", "target_mae": 2.0 * focal_mae})
    return pd.DataFrame(rows)


class DirectFiveArmConfigAndMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        cls.specs = experiment.planned_specs(cls.config)

    def test_constants_and_frozen_config_contract(self) -> None:
        self.assertEqual(experiment.DIRECT_STAGE, "direct_arms")
        self.assertEqual(tuple(experiment.DIRECT_ARMS), EXPECTED_ARMS)
        self.assertEqual(tuple(experiment.FOLDS), EXPECTED_FOLDS)
        self.assertEqual(experiment.SEED, 42)
        self.assertEqual(experiment.EXPECTED_TRAINING_JOBS, 20)
        self.assertEqual(experiment.EXPECTED_PREDICTION_CELLS, 20)
        self.assertEqual(experiment.EXPECTED_PAIR_METRIC_ROWS, 2_500)
        self.assertEqual(
            experiment.INTERPRETATION,
            "retrospective_rolling_development_single_seed_descriptive",
        )

        self.assertEqual(
            self.config["experiment"]["experiment_kind"],
            experiment.EXPERIMENT_KIND,
        )
        self.assertEqual(
            self.config["experiment"]["interpretation"],
            experiment.INTERPRETATION,
        )
        self.assertEqual(self.config["matrix"]["seeds"], [42])
        self.assertEqual(tuple(self.config["matrix"]["direct_arms"]), EXPECTED_ARMS)
        self.assertEqual(self.config["data"]["tolerances_minutes"], [5])
        self.assertEqual(
            self.config["training"]["protocol"], "independent_random_init_v1"
        )
        self.assertEqual(self.config["training"]["num_epochs"], 240)
        self.assertEqual(self.config["training"]["early_stopping_min_epochs"], 30)
        self.assertEqual(self.config["training"]["early_stopping_patience"], 20)
        self.assertEqual(self.config["training"]["validation_mc_samples"], 16)
        self.assertEqual(self.config["training"]["prediction_mc_samples"], 64)
        self.assertEqual(
            {
                "generator": self.config["model"]["expected_generator_parameters"],
                "critic": self.config["model"]["expected_critic_parameters"],
                "total": self.config["model"]["expected_total_parameters"],
            },
            {"generator": 827_745, "critic": 729_157, "total": 1_556_902},
        )

    def test_config_validation_rejects_drift_and_parent_lineage_schema(self) -> None:
        experiment.validate_config(copy.deepcopy(self.config))

        epochs = copy.deepcopy(self.config)
        epochs["training"]["num_epochs"] = 239
        with self.assertRaisesRegex(ValueError, "num_epochs|240"):
            experiment.validate_config(epochs)

        arms = copy.deepcopy(self.config)
        arms["matrix"]["direct_arms"] = [*EXPECTED_ARMS[:-1], "unknown"]
        with self.assertRaisesRegex(ValueError, "arm"):
            experiment.validate_config(arms)

        lineage = copy.deepcopy(self.config)
        lineage["training"]["protocol"] = "parent_then_continuation_v1"
        with self.assertRaisesRegex(ValueError, "protocol|independent"):
            experiment.validate_config(lineage)

        for key in (
            "parent_arms",
            "continuation_arms",
            "branch_recipes",
            "parent_state_allowlist",
            "lr_replay",
        ):
            unsupported = copy.deepcopy(self.config)
            unsupported["matrix"][key] = []
            with self.subTest(unsupported_lineage_key=key):
                with self.assertRaisesRegex(
                    ValueError, "unsupported.*lineage|parent|continuation"
                ):
                    experiment.validate_config(unsupported)

    def test_exactly_twenty_unique_direct_specs_with_ten_ten_gpu_balance(self) -> None:
        self.assertEqual(len(self.specs), 20)
        self.assertEqual(len({row["job_id"] for row in self.specs}), 20)
        self.assertEqual({row["stage"] for row in self.specs}, {"direct_arms"})
        self.assertEqual({int(row["seed"]) for row in self.specs}, {42})
        self.assertEqual({str(row["arm"]) for row in self.specs}, set(EXPECTED_ARMS))
        self.assertEqual({str(row["fold"]) for row in self.specs}, set(EXPECTED_FOLDS))
        self.assertEqual(
            Counter(int(row["gpu_id"]) for row in self.specs), {0: 10, 1: 10}
        )
        self.assertEqual(
            {
                fold: {int(row["gpu_id"]) for row in self.specs if row["fold"] == fold}
                for fold in EXPECTED_FOLDS
            },
            {
                "f1_2023q1": {0},
                "f2_2023q2": {1},
                "f3_2023q3": {0},
                "f4_2023q4": {1},
            },
        )
        experiment.validate_gpu_balance(self.specs)

    def test_all_five_arms_share_epoch_zero_generator_and_critic_states(self) -> None:
        for fold in EXPECTED_FOLDS:
            rows = [row for row in self.specs if row["fold"] == fold]
            self.assertEqual(len(rows), 5)
            generator_hashes = {
                str(row["initial_generator_state_sha256"]) for row in rows
            }
            critic_hashes = {str(row["initial_critic_state_sha256"]) for row in rows}
            self.assertEqual(len(generator_hashes), 1)
            self.assertEqual(len(critic_hashes), 1)
            generator_hash = next(iter(generator_hashes))
            critic_hash = next(iter(critic_hashes))
            self.assertRegex(generator_hash, r"^[0-9a-f]{64}$")
            self.assertRegex(critic_hash, r"^[0-9a-f]{64}$")
            self.assertNotEqual(generator_hash, critic_hash)

    def test_planned_epoch_zero_hashes_match_the_trainer_initialization_order(
        self,
    ) -> None:
        from wgan_option.config import Config
        from wgan_option.models.gan_model import WGAN_GP

        model = self.config["model"]
        training = self.config["training"]
        trainer_config = Config(
            cuda=False,
            seed=42,
            support_mask_mode="raw_joint",
            channels=int(model["channels"]),
            embedding_dim=int(model["embedding_dim"]),
            noise_dim=int(model["noise_dim"]),
            generator_noise_mode=str(model["generator_noise_mode"]),
            generator_current_input_mode=str(model["generator_current_input_mode"]),
            generator_conditioning_mode=str(model["generator_conditioning_mode"]),
            critic_conditioning_mode=str(model["critic_conditioning_mode"]),
            critic_normalization_mode=str(model["critic_normalization_mode"]),
            gen_base_channels=int(model["gen_base_channels"]),
            gen_res_blocks=int(model["gen_res_blocks"]),
            gen_text_hidden_dim=int(model["gen_text_hidden_dim"]),
            gen_text_out_dim=int(model["gen_text_out_dim"]),
            gen_hidden_dim=int(model["gen_hidden_dim"]),
            disc_base_channels=int(model["disc_base_channels"]),
            disc_res_blocks=int(model["disc_res_blocks"]),
            disc_text_hidden_dim=int(model["disc_text_hidden_dim"]),
            disc_hidden_dim=int(model["disc_hidden_dim"]),
            residual_output_mode=str(model["residual_output_mode"]),
            generator_learning_rate=float(training["generator_learning_rate"]),
            discriminator_learning_rate=float(training["discriminator_learning_rate"]),
            news_first_full_training_state_mode="none",
        )
        trainer = WGAN_GP(
            trainer_config,
            strike_grid=np.asarray(
                self.config["data"]["strike_grid"], dtype=np.float32
            ),
            maturity_grid_days=np.asarray(
                self.config["data"]["maturity_days_grid"], dtype=np.float32
            ),
            embedding_dim=1024,
        )
        actual = {
            "initial_generator_state_sha256": experiment._state_dict_sha256(trainer.G),
            "initial_critic_state_sha256": experiment._state_dict_sha256(trainer.D),
        }
        self.assertEqual(actual, experiment._initial_state_hashes(self.config))
        self.assertEqual(
            actual["initial_generator_state_sha256"],
            self.specs[0]["initial_generator_state_sha256"],
        )
        self.assertEqual(
            actual["initial_critic_state_sha256"],
            self.specs[0]["initial_critic_state_sha256"],
        )

    def test_direct_profile_restores_every_shared_core_global_after_failure(
        self,
    ) -> None:
        names = (
            "DEFAULT_CONFIG",
            "DEFAULT_OUTPUT_DIR",
            "EXPERIMENT_KIND",
            "INTERPRETATION",
            "SEEDS",
            "TOLERANCES",
            "FOLDS",
            "PARENT_ARM",
            "CONTINUATION_ARM",
            "TEXT_ARMS_5M",
            "TEXT_ARMS_30M",
            "ALL_ARMS",
            "PARENT_STAGE",
            "CONTINUATION_STAGE",
            "BRANCH_STAGE",
            "STAGES",
            "EXPECTED_STAGE_COUNTS",
            "EXPECTED_TRAINING_JOBS",
            "EXPECTED_PREDICTION_CELLS",
            "WORKER_MODULE",
            "SOURCE_CODE_RELATIVE_PATHS",
            "INFERENCE_DETERMINISM_KIND",
        )
        before = {name: getattr(experiment.core, name) for name in names}
        with self.assertRaisesRegex(RuntimeError, "profile fixture"):
            with experiment.direct_profile():
                self.assertEqual(experiment.core.STAGES, ("direct_arms",))
                self.assertEqual(experiment.core.SEEDS, (42,))
                self.assertEqual(experiment.core.ALL_ARMS, EXPECTED_ARMS)
                self.assertEqual(
                    experiment.core.GENERATOR_MODE, "film_unet_mask_coords_v1"
                )
                self.assertEqual(
                    experiment.core.CRITIC_MODE, "lp_disabled_same_shape_v1"
                )
                raise RuntimeError("profile fixture")
        for name, expected in before.items():
            self.assertIs(getattr(experiment.core, name), expected)

    def test_arm_overlay_modes_are_explicit_and_complete(self) -> None:
        self.assertEqual(
            {arm: experiment._overlay_mode(arm) for arm in EXPECTED_ARMS},
            {
                "lp_matched": "lp_mean_l2",
                "lp_shuffle": "lp_shuffle",
                "no_text": "current_only",
                "bow": "bow1024",
                "sentiment": "sentiment_pad1024",
            },
        )
        with self.assertRaises((KeyError, ValueError)):
            experiment._overlay_mode("unknown")

    def test_training_payload_is_direct_and_never_materializes_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for spec in self.specs:
                payload = experiment._training_payload(self.config, root, spec)
                self.assertEqual(
                    payload["generator_conditioning_mode"],
                    "film_unet_mask_coords_v1",
                )
                self.assertEqual(
                    payload["critic_conditioning_mode"],
                    "lp_disabled_same_shape_v1",
                )
                self.assertEqual(payload["num_epochs"], 240)
                self.assertEqual(payload["early_stopping_min_epochs"], 30)
                self.assertEqual(payload["early_stopping_patience"], 20)
                self.assertEqual(payload["validation_mc_samples"], 16)
                self.assertTrue(payload["news_first_materialize_validation_loader"])
                self.assertFalse(payload["news_first_materialize_test_loader"])
                self.assertEqual(
                    payload["news_first_pair_text_overlay_mode"],
                    experiment._overlay_mode(str(spec["arm"])),
                )
                self.assertEqual(
                    payload["news_first_full_training_state_mode"],
                    "save_dynamic_v1",
                )
                self.assertEqual(payload["news_first_refit_mode"], "none")
                self.assertTrue(FORBIDDEN_LINEAGE_KEYS.isdisjoint(payload))

    def test_build_job_strips_planning_only_fields_and_has_no_state_parent(
        self,
    ) -> None:
        spec = {**self.specs[0], "gpu_slot": 0, "wave": 1}
        base_job = {
            "stage": "direct_arms",
            "fold": spec["fold"],
            "arm": spec["arm"],
            "seed": 42,
            "job_id": spec["job_id"],
            "parent_job_id": "",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                experiment.core,
                "build_job",
                return_value=(base_job, {"unused": True}),
            ) as build_job,
        ):
            job = experiment._build_job(
                self.config,
                Path(directory),
                spec,
                slots_per_gpu=10,
                num_epochs=240,
            )
        passed_spec = build_job.call_args.args[2]
        self.assertNotIn("job_id", passed_spec)
        self.assertNotIn("initial_generator_state_sha256", passed_spec)
        self.assertNotIn("initial_critic_state_sha256", passed_spec)
        self.assertEqual(build_job.call_args.kwargs["slots_per_gpu"], 10)
        self.assertEqual(build_job.call_args.kwargs["num_epochs_override"], 240)
        self.assertNotIn("parent_state", build_job.call_args.kwargs)
        self.assertNotIn("recipe", build_job.call_args.kwargs)
        self.assertEqual(job["parent_job_id"], "")
        self.assertEqual(job["parent_state_path"], "")
        self.assertEqual(job["parent_state_sha256"], "")
        self.assertEqual(job["continuation_job_id"], "")
        self.assertEqual(job["recipe_path"], "")
        self.assertEqual(job["recipe_sha256"], "")
        self.assertEqual(job["job_spec_sha256"], experiment.core._job_spec_sha(job))

    def test_no_parent_continuation_or_recipe_lineage_enters_specs(self) -> None:
        self.assertTrue(
            all(FORBIDDEN_LINEAGE_KEYS.isdisjoint(row) for row in self.specs)
        )
        self.assertEqual({row["stage"] for row in self.specs}, {"direct_arms"})
        serialized = json.dumps(self.specs, sort_keys=True)
        for forbidden in (
            '"stage": "parents"',
            '"stage": "continuations"',
            '"stage": "text_branches"',
            "branch_recipe",
            "parent_state_allowlist",
            "lr_replay",
        ):
            self.assertNotIn(forbidden, serialized)


class DirectFiveArmStatisticsAndSafetyTests(unittest.TestCase):
    def test_fold_session_bootstrap_is_paired_deterministic_and_fold_first(
        self,
    ) -> None:
        evidence = _synthetic_pair_metrics()
        kwargs = {
            "focal_arm": "lp_matched",
            "reference_arm": "no_text",
            "expected_folds": EXPECTED_FOLDS,
            "iterations": 256,
            "rng_seed": 1234,
        }
        first = experiment.fold_session_paired_bootstrap(evidence, **kwargs)
        second = experiment.fold_session_paired_bootstrap(evidence, **kwargs)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean_log_mae_ratio"], math.log(0.5), places=14)
        self.assertAlmostEqual(first["geometric_mae_ratio"], 0.5, places=14)
        self.assertEqual(first["focal_nonworse_fold_count"], 4)
        self.assertEqual(first["fold_count"], 4)
        self.assertEqual(first["pair_count"], 16)
        self.assertEqual(first["session_count"], 8)
        self.assertEqual(first["bootstrap_iterations"], 256)
        self.assertEqual(
            first["resampling_method"],
            "fold_then_paired_cme_session_cluster_recompute_fold_log_mae_ratio",
        )
        self.assertLess(first["ci_95_upper"], 0.0)

        incomplete = evidence.drop(
            evidence[
                evidence["arm"].eq("no_text")
                & evidence["pair_id"].eq("f1_2023q1-pair-0")
            ].index
        )
        with self.assertRaisesRegex(ValueError, "pair complete"):
            experiment.fold_session_paired_bootstrap(incomplete, **kwargs)

    def test_holm_adjustment_is_step_down_monotone_and_fail_closed(self) -> None:
        adjusted = experiment.holm_adjust({"comparison_b": 0.04, "comparison_a": 0.01})
        self.assertEqual(adjusted, {"comparison_a": 0.02, "comparison_b": 0.04})
        five = experiment.holm_adjust(
            {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.20}
        )
        self.assertEqual(five, {"a": 0.05, "b": 0.08, "c": 0.09, "d": 0.09, "e": 0.20})
        with self.assertRaises(ValueError):
            experiment.holm_adjust({"invalid": float("nan")})

    def test_predict_rejects_before_checkpoint_freeze_without_writing_test_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "frozen.*checkpoint|allowlist"):
                experiment.predict(root)
            self.assertFalse((root / "evaluation").exists())
            self.assertFalse((root / "predictions").exists())

            with (
                mock.patch.object(
                    experiment,
                    "read_registry",
                    return_value={"evaluation_frozen": False},
                ),
                mock.patch.object(experiment, "freeze_evaluation") as freeze,
                mock.patch.object(
                    experiment, "_materialize_test_inputs"
                ) as materialize,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "frozen.*checkpoint|allowlist"
                ):
                    experiment.predict(root)
                freeze.assert_not_called()
                materialize.assert_not_called()

    def test_frozen_file_verification_rejects_sha_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "checkpoint.pt"
            artifact.write_bytes(b"frozen checkpoint")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(experiment.sha256_file(artifact), digest)
            experiment._verify_frozen_file(artifact, digest)

            artifact.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA|hash|drift|tamper"):
                experiment._verify_frozen_file(artifact, digest)

    def test_terminal_complete_root_is_read_only(self) -> None:
        config = experiment.load_config(experiment.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            registry.mkdir(parents=True)
            (registry / "jobs.json").write_text(
                json.dumps({"terminal_complete": True, "jobs": []}),
                encoding="utf-8",
            )
            before = {
                path.relative_to(root): (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in root.rglob("*")
                if path.is_file()
            }
            with self.assertRaisesRegex(RuntimeError, "terminal|complete|read.only"):
                experiment.prepare(config, root, resume=True)
            after = {
                path.relative_to(root): (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_freeze_evaluation_rejects_an_unfinished_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises((FileNotFoundError, RuntimeError, ValueError)):
                experiment.freeze_evaluation(root)
            self.assertFalse(
                (root / "registry" / "evaluation_checkpoint_allowlist.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
