"""Focused lifecycle tests for the Pure-CNN-parent text-effect orchestrator."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed_prediction as prediction,
)
from wgan_option.config import Config
from wgan_option.models.common import IDENTITY_RESIDUAL_OUTPUT_MODE
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.utils.news_first_experiment_core import (
    FULL_TRAINING_STATE_PHASE,
    SAVE_DYNAMIC_FULL_TRAINING_STATE,
    write_full_training_state_contract,
)


class PureCnnBackboneTextEffectOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config()

    def _registry(self) -> dict[str, object]:
        jobs = experiment.planned_job_specs(self.config)
        return {
            "schema_version": 1,
            "kind": experiment.REGISTRY_KIND,
            "experiment_kind": experiment.EXPERIMENT_KIND,
            "interpretation": experiment.INTERPRETATION,
            "status": "prepared",
            "jobs": jobs,
            "jobs_sha256": experiment.direct.payload_sha256(jobs),
            "parents_frozen": False,
            "grafts_frozen": False,
            "continuations_prepared": False,
            "evaluation_frozen": False,
            "test_data_opened": False,
            "validation_trajectories_frozen": False,
            "standard_predictions_frozen": False,
            "interventions_frozen": False,
            "analysis_complete": False,
            "terminal_complete": False,
        }

    def test_profile_patcher_restores_and_removes_branch_local_attributes(
        self,
    ) -> None:
        module = SimpleNamespace(existing="before")
        with experiment._patched(
            module, {"existing": "during", "branch_local": "temporary"}
        ):
            self.assertEqual(module.existing, "during")
            self.assertEqual(module.branch_local, "temporary")
        self.assertEqual(module.existing, "before")
        self.assertFalse(hasattr(module, "branch_local"))
        self.assertFalse(hasattr(experiment.film_text, "validate_root"))
        shared_validate_root = experiment.film_text.multiseed.validate_root
        shared_digest = experiment.direct._checkpoint_state_sha256
        with experiment._film_stage_profile(
            main_root=Path("/not-used"), stage_name="film_continuations"
        ):
            self.assertTrue(callable(experiment.film_text.validate_root))
            self.assertIsNot(
                experiment.film_text.multiseed.validate_root,
                shared_validate_root,
            )
            self.assertIs(
                experiment.direct._checkpoint_state_sha256,
                experiment._canonical_checkpoint_state_sha256,
            )
        self.assertFalse(hasattr(experiment.film_text, "validate_root"))
        self.assertIs(
            experiment.film_text.multiseed.validate_root, shared_validate_root
        )
        self.assertIs(experiment.direct._checkpoint_state_sha256, shared_digest)

    def test_planned_grafts_are_common_within_seed_fold_not_across_folds(
        self,
    ) -> None:
        folds = experiment.FOLDS[:2]
        jobs = [
            {
                "seed": 42,
                "fold": fold,
                "arm": arm,
                "graft_state_path": f"{fold}.pt",
                "initial_generator_state_sha256": f"g-{fold}",
                "initial_critic_state_sha256": f"d-{fold}",
            }
            for fold in folds
            for arm in experiment.FILM_ARMS[:2]
        ]
        with (
            mock.patch.object(experiment, "SEEDS", (42,)),
            mock.patch.object(experiment, "FOLDS", folds),
        ):
            experiment._validate_planned_initial_state_fairness(
                jobs,
                use_graft=True,
            )

            within_block_drift = deepcopy(jobs)
            within_block_drift[0]["initial_generator_state_sha256"] = "drift"
            with self.assertRaisesRegex(ValueError, "within each seed/fold block"):
                experiment._validate_planned_initial_state_fairness(
                    within_block_drift,
                    use_graft=True,
                )

            cross_fold_reuse = deepcopy(jobs)
            for job in cross_fold_reuse:
                job["initial_generator_state_sha256"] = "same-g"
            with self.assertRaisesRegex(ValueError, "distinct across seed/fold blocks"):
                experiment._validate_planned_initial_state_fairness(
                    cross_fold_reuse,
                    use_graft=True,
                )

    def test_fresh_parent_initialization_is_common_by_seed_across_folds(self) -> None:
        seeds = (42, 202)
        folds = experiment.FOLDS[:2]
        jobs = [
            {
                "seed": seed,
                "fold": fold,
                "arm": experiment.PARENT_ARM,
                "initial_generator_state_sha256": f"g-{seed}",
                "initial_critic_state_sha256": f"d-{seed}",
            }
            for seed in seeds
            for fold in folds
        ]
        with (
            mock.patch.object(experiment, "SEEDS", seeds),
            mock.patch.object(experiment, "FOLDS", folds),
        ):
            experiment._validate_planned_initial_state_fairness(
                jobs,
                use_graft=False,
            )
            fold_drift = deepcopy(jobs)
            fold_drift[0]["initial_generator_state_sha256"] = "drift"
            with self.assertRaisesRegex(ValueError, "within each seed"):
                experiment._validate_planned_initial_state_fairness(
                    fold_drift,
                    use_graft=False,
                )

    def test_film_profile_routes_nested_validator_to_seed_fold_contract(self) -> None:
        root = Path("/not-used")
        sentinel = {"validated": True}
        original = experiment.film_text.multiseed.validate_root
        with mock.patch.object(
            experiment,
            "_validate_stage_root",
            return_value=sentinel,
        ) as validate_stage:
            with experiment._film_stage_profile(
                main_root=root,
                stage_name="film_continuations",
            ):
                observed = experiment.film_text.multiseed.validate_root(
                    root,
                    verify_large_inputs=False,
                )
        self.assertIs(observed, sentinel)
        validate_stage.assert_called_once_with(
            root,
            stage_name="film_continuations",
            arms=experiment.FILM_ARMS,
            generator_mode=experiment.FILM_MODE,
            expected_jobs=200,
            use_graft=True,
            verify_large_inputs=False,
        )
        self.assertIs(experiment.film_text.multiseed.validate_root, original)

    def test_training_and_prediction_counts_are_exact(self) -> None:
        jobs = experiment.planned_job_specs(self.config)
        self.assertEqual(len(jobs), 280)
        self.assertEqual(sum(row["stage"] == "backbone" for row in jobs), 40)
        self.assertEqual(sum(row["stage"] == "continuation" for row in jobs), 240)
        self.assertEqual(experiment.EXPECTED_STANDARD_PREDICTIONS, 280)
        self.assertEqual(experiment.EXPECTED_INTERVENTION_PREDICTIONS, 80)
        self.assertEqual(
            {gpu: sum(row["gpu_id"] == gpu for row in jobs) for gpu in (0, 1)},
            {0: 140, 1: 140},
        )

    def test_registry_state_rejects_test_and_analysis_phase_shortcuts(self) -> None:
        registry = self._registry()
        experiment._validate_registry_state(self.config, registry)

        opened = deepcopy(registry)
        opened["test_data_opened"] = True
        opened["jobs_sha256"] = experiment.direct.payload_sha256(opened["jobs"])
        with self.assertRaisesRegex(ValueError, "requires evaluation_frozen"):
            experiment._validate_registry_state(self.config, opened)

        analysis = deepcopy(registry)
        analysis["analysis_complete"] = True
        with self.assertRaisesRegex(ValueError, "three evidence layers"):
            experiment._validate_registry_state(self.config, analysis)

    def test_one_epoch_stage_filters_snapshots_and_evaluates_epoch_zero(self) -> None:
        stage_config = experiment._stage_config(
            self.config,
            stage_name="backbones",
            arms=(experiment.PARENT_ARM,),
            generator_mode=experiment.PURE_MODE,
            expected_counts={
                "training_jobs": 40,
                "pair_rows": 5_000,
                **experiment.PURE_COUNTS,
            },
            workers_per_gpu=1,
            output_root=Path("/not-used"),
            num_epochs=1,
        )
        self.assertEqual(stage_config["model"]["gen_text_hidden_dim"], 256)
        self.assertEqual(stage_config["model"]["gen_text_out_dim"], 128)
        with mock.patch.object(
            experiment.direct, "_source_paths", experiment._stage_source_paths
        ):
            self.assertTrue(experiment._stage_source_paths(stage_config))
        initial = {
            "initial_generator_state_sha256": "a" * 64,
            "initial_critic_state_sha256": "b" * 64,
        }
        with mock.patch.object(
            experiment,
            "_BASE_PURE_INITIAL_STATE_HASHES",
            side_effect=lambda _config, _seed: dict(initial),
        ):
            specs = experiment._planned_stage_specs(
                stage_config,
                arm_universe=(experiment.PARENT_ARM,),
                generator_mode=experiment.PURE_MODE,
                main_root=Path("/not-used"),
                use_graft=False,
            )
        self.assertEqual(len(specs), 40)
        self.assertTrue(all(row["validation_snapshot_epochs"] == (1,) for row in specs))
        payload = experiment._stage_training_payload(
            stage_config, Path("/not-used"), specs[0]
        )
        self.assertTrue(payload["evaluate_initial_checkpoint"])
        self.assertEqual(payload["news_first_validation_snapshot_epochs"], [1])

    def test_parent_allowlist_deeply_matches_full_state_to_best_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_root = root / "stage"
            training_path = root / "training.yaml"
            training_path.write_text("num_epochs: 1\n", encoding="utf-8")
            best_path = root / "best.json"
            best_path.write_text('{"best_epoch": 1}\n', encoding="utf-8")
            status_path = experiment.direct._status_path(stage_root, "internal-parent")
            status_path.parent.mkdir(parents=True)
            status_path.write_text('{"status": "completed"}\n', encoding="utf-8")
            job = {
                "job_id": "internal-parent",
                "seed": 42,
                "fold": experiment.FOLDS[0],
                "training_config_path": str(training_path),
                "validation_snapshot_epochs": (1,),
            }
            # Deliberately use non-lexicographic insertion order.  A model-state
            # digest is a function of names, dtypes, shapes, and bytes, not the
            # incidental order in which a serialized mapping exposes its keys.
            generator_state = {
                "surface_encoder.weight": torch.tensor([1.0]),
                "bottleneck_conv.weight": torch.tensor([3.0]),
            }
            critic_state = {
                "surface_encoder.weight": torch.tensor([2.0]),
                "classifier.weight": torch.tensor([4.0]),
            }
            generator_sha = experiment._tensor_state_sha256(generator_state)
            generator_path = root / "generator.pt"
            discriminator_path = root / "discriminator.pt"
            torch.save({"state_dict": generator_state}, generator_path)
            torch.save({"state_dict": critic_state}, discriminator_path)
            artifacts = {
                "best_learned_checkpoint": {"path": str(best_path), "sha256": "1"},
                "full_training_state": {"path": "full.pt", "sha256": "2"},
                "generator_best_learned": {
                    "path": str(generator_path),
                    "sha256": "3",
                },
                "discriminator_best_learned": {
                    "path": str(discriminator_path),
                    "sha256": "4",
                },
                "generator_validation_epoch_0001": {"path": "g1.pt"},
                "discriminator_validation_epoch_0001": {"path": "d1.pt"},
                "training_metrics_csv": {"path": "metrics.csv"},
            }
            full_state = {
                "completed_epoch": 1,
                "generator_state_dict": generator_state,
                "discriminator_state_dict": critic_state,
            }

            with (
                mock.patch.object(
                    experiment.direct,
                    "_artifact",
                    side_effect=lambda _status, role: artifacts[role],
                ),
                mock.patch.object(
                    experiment.core,
                    "_validated_full_state_payload",
                    return_value=full_state,
                ) as validate_full_state,
            ):
                row = experiment._parent_allowlist_row(stage_root, job)
            validate_full_state.assert_called_once()
            self.assertEqual(row["best_epoch"], row["full_state_completed_epoch"])
            self.assertEqual(row["full_state_generator_state_sha256"], generator_sha)

            with (
                mock.patch.object(
                    experiment.direct,
                    "_artifact",
                    side_effect=lambda _status, role: artifacts[role],
                ),
                mock.patch.object(
                    experiment.core,
                    "_validated_full_state_payload",
                    return_value=full_state,
                ),
                mock.patch.object(
                    experiment,
                    "_canonical_checkpoint_state_sha256",
                    return_value="f" * 64,
                ),
                self.assertRaisesRegex(ValueError, "differs from best checkpoint"),
            ):
                experiment._parent_allowlist_row(stage_root, job)

    def test_one_epoch_parent_full_state_matches_real_best_learned_checkpoints(
        self,
    ) -> None:
        """Exercise the real trainer save order at the smallest useful scale."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_state_path = root / "full_training_state_best_learned.pt"
            lineage_sha = hashlib.sha256(b"one-epoch-parent-lineage").hexdigest()
            lineage = {
                "fold_id": experiment.FOLDS[0],
                "seed": 42,
                "arm": experiment.PARENT_ARM,
                "model_contract_sha256": lineage_sha,
                "grid_sha256": lineage_sha,
                "training_config_payload_sha256": lineage_sha,
                "code_sha256": lineage_sha,
                "dataset_sha256": lineage_sha,
                "support_sha256": lineage_sha,
                "pair_universe_sha256": lineage_sha,
                "text_manifest_sha256": lineage_sha,
                "job_sha256": lineage_sha,
            }
            contract = write_full_training_state_contract(
                root / "full_state_contract.json",
                mode=SAVE_DYNAMIC_FULL_TRAINING_STATE,
                output_path=full_state_path,
                output_lineage=lineage,
            )
            config = Config(
                cuda=False,
                seed=42,
                channels=1,
                embedding_dim=4,
                noise_dim=2,
                support_mask_mode="raw_joint",
                generator_current_input_mode="current_support_masked",
                generator_conditioning_mode=experiment.PURE_MODE,
                critic_conditioning_mode=experiment.CRITIC_MODE,
                gen_base_channels=1,
                disc_base_channels=1,
                gen_res_blocks=0,
                disc_res_blocks=0,
                gen_text_hidden_dim=3,
                gen_text_out_dim=2,
                disc_text_hidden_dim=2,
                gen_hidden_dim=4,
                disc_hidden_dim=4,
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
                num_epochs=1,
                batch_size=1,
                discriminator_iter=1,
                learning_rate=1e-6,
                lr_scheduler_type="plateau",
                use_reduce_lr_on_plateau=True,
                reduce_lr_patience=1,
                reduce_lr_min_lr=1e-7,
                evaluate_initial_checkpoint=True,
                best_checkpoint_metric="val_recon",
                validation_mc_samples=1,
                use_early_stopping=False,
                use_calendar_constraint=False,
                use_butterfly_constraint=False,
                use_smooth_constraint=False,
                save_every=999,
                news_first_full_training_state_mode=(SAVE_DYNAMIC_FULL_TRAINING_STATE),
                news_first_full_training_state_contract_path=contract["contract_path"],
                news_first_full_training_state_contract_sha256=contract[
                    "contract_sha256"
                ],
                models_path=str(root / "checkpoints"),
                outputs_path=str(root / "checkpoints"),
                samples_path=str(root / "samples"),
                metrics_path=str(root / "metrics"),
            )
            model = WGAN_GP(
                config,
                strike_grid=np.linspace(0.97, 1.03, 16, dtype=np.float32),
                maturity_grid_days=np.asarray(
                    self.config["data"]["maturity_days_grid"],
                    dtype=np.float32,
                ),
                embedding_dim=4,
            )
            current = torch.full((1, 1, 16, 16), 0.20)
            target = torch.full((1, 1, 16, 16), 0.21)
            text = torch.zeros((1, 4))
            weight = torch.ones(1)
            stable_key = torch.tensor([101], dtype=torch.int64)
            support = torch.ones((1, 1, 16, 16))
            loader = DataLoader(
                TensorDataset(
                    current,
                    text,
                    target,
                    weight,
                    stable_key,
                    support,
                    support,
                ),
                batch_size=1,
                shuffle=True,
                generator=torch.Generator().manual_seed(777),
            )

            evaluation_count = 0

            def evaluate(_loader):
                nonlocal evaluation_count
                evaluation_count += 1
                metric = 2.0 if evaluation_count == 1 else 1.0
                return {
                    "val_recon": metric,
                    "val_current_recon": 2.0,
                    "val_baseline_gap": metric - 2.0,
                    "val_hybrid_score": metric,
                    "val_calendar": 0.0,
                    "val_butterfly": 0.0,
                    "val_delta_shrink": 0.0,
                }

            def discriminator_step(*_args, **_kwargs):
                with torch.no_grad():
                    next(model.D.parameters()).add_(0.125)
                return {"d_total": 0.0}

            def generator_step(*_args, **_kwargs):
                with torch.no_grad():
                    next(model.G.parameters()).add_(0.25)
                return {"g_total": 0.0}

            with (
                mock.patch.object(model, "_evaluate", side_effect=evaluate),
                mock.patch.object(
                    model,
                    "_discriminator_step",
                    side_effect=discriminator_step,
                ),
                mock.patch.object(
                    model,
                    "_generator_step",
                    side_effect=generator_step,
                ),
                mock.patch.object(
                    model,
                    "_step_plateau_scheduler",
                    wraps=model._step_plateau_scheduler,
                ) as scheduler_step,
                mock.patch.object(model, "_save_loss_curves"),
            ):
                model.train(loader, loader)

            full_state = torch.load(
                full_state_path,
                map_location="cpu",
                weights_only=False,
            )
            generator_checkpoint = torch.load(
                root / "checkpoints/generator_best_learned.pt",
                map_location="cpu",
                weights_only=False,
            )["state_dict"]
            discriminator_checkpoint = torch.load(
                root / "checkpoints/discriminator_best_learned.pt",
                map_location="cpu",
                weights_only=False,
            )["state_dict"]

            self.assertEqual(full_state["completed_epoch"], 1)
            self.assertEqual(full_state["save_phase"], FULL_TRAINING_STATE_PHASE)
            self.assertEqual(scheduler_step.call_count, 4)
            self.assertEqual(
                full_state["generator_scheduler_state_dict"]["last_epoch"], 2
            )
            self.assertEqual(
                full_state["discriminator_scheduler_state_dict"]["last_epoch"], 2
            )
            for full_key, checkpoint in (
                ("generator_state_dict", generator_checkpoint),
                ("discriminator_state_dict", discriminator_checkpoint),
            ):
                self.assertEqual(full_state[full_key].keys(), checkpoint.keys())
                for name, tensor in checkpoint.items():
                    torch.testing.assert_close(
                        full_state[full_key][name],
                        tensor,
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_grafted_epoch_zero_fairness_uses_canonical_checkpoint_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir()
            generator_state = {
                "surface_encoder.weight": torch.tensor([1.0]),
                "bottleneck_conv.weight": torch.tensor([2.0]),
            }
            critic_state = {
                "surface_encoder.weight": torch.tensor([3.0]),
                "classifier.weight": torch.tensor([4.0]),
            }
            generator_path = root / "generator_initial_epoch0.pt"
            critic_path = root / "discriminator_initial_epoch0.pt"
            torch.save({"state_dict": generator_state}, generator_path)
            torch.save({"state_dict": critic_state}, critic_path)
            canonical_g = experiment._tensor_state_sha256(generator_state)
            canonical_d = experiment._tensor_state_sha256(critic_state)
            self.assertNotEqual(
                canonical_g,
                experiment.direct._checkpoint_state_sha256(generator_path),
            )
            jobs = [
                {
                    "job_id": f"grafted-{arm}",
                    "seed": 42,
                    "fold": experiment.FOLDS[0],
                    "arm": arm,
                    "graft_state_path": str(root / "graft.pt"),
                    "graft_state_sha256": "a" * 64,
                    "initial_generator_state_sha256": canonical_g,
                    "initial_critic_state_sha256": canonical_d,
                }
                for arm in (
                    experiment.PURE_CONTINUATION_ARM,
                    *experiment.FILM_ARMS,
                )
            ]
            for job in jobs:
                status_path = experiment.direct._status_path(root, job["job_id"])
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.write_text('{"status":"completed"}\n', encoding="utf-8")

            def artifact(_status, role):
                return {
                    "path": str(
                        generator_path
                        if role == "generator_initial_epoch0"
                        else critic_path
                    )
                }

            with (
                mock.patch.object(experiment, "SEEDS", (42,)),
                mock.patch.object(experiment, "FOLDS", (experiment.FOLDS[0],)),
                mock.patch.object(experiment.direct, "_artifact", side_effect=artifact),
            ):
                path = experiment._stage_freeze_initial_state_fairness(root, jobs)
            payload = experiment._read_json(path)
            self.assertEqual(payload["job_count"], 6)
            self.assertEqual(
                {row["generator_initial_state_sha256"] for row in payload["rows"]},
                {canonical_g},
            )
            self.assertEqual(
                {row["critic_initial_state_sha256"] for row in payload["rows"]},
                {canonical_d},
            )

    def test_graft_resume_rejects_partial_pair(self) -> None:
        parent = {"parent_job_id": "parent", "seed": 42, "fold": experiment.FOLDS[0]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, _manifest = experiment._block_graft_paths(
                root, parent, experiment.FILM_MODE
            )
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"partial")
            with self.assertRaisesRegex(RuntimeError, "Partial graft"):
                experiment._resume_block_graft(
                    self.config,
                    root,
                    parent,
                    target_mode=experiment.FILM_MODE,
                )

    def test_formal_prepare_resume_materializes_a_missing_backbone_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    experiment,
                    "validate_partial_root",
                    return_value=self.config,
                ),
                mock.patch.object(
                    experiment,
                    "_read_registry",
                    return_value={"formal_workers_per_gpu": 6},
                ),
                mock.patch.object(experiment, "_prepare_backbone_stage") as prepare,
                mock.patch.object(
                    experiment, "validate_root", return_value=self.config
                ),
            ):
                result = experiment._prepare_formal(
                    self.config,
                    root,
                    workers=6,
                    resume=True,
                )
            self.assertEqual(result, root)
            prepare.assert_called_once_with(
                self.config,
                root,
                workers=6,
                num_epochs=240,
                resume=False,
            )

    def test_predict_test_gate_runs_before_any_stage_prediction(self) -> None:
        registry = {"evaluation_frozen": False, "jobs": []}
        with (
            mock.patch.object(experiment, "validate_root", return_value=self.config),
            mock.patch.object(experiment, "_read_registry", return_value=registry),
            mock.patch.object(experiment.pure, "predict") as pure_predict,
            self.assertRaisesRegex(prediction.TextEffectPredictionError, "frozen"),
        ):
            experiment.predict_test("/not-used")
        pure_predict.assert_not_called()

    def test_cli_exposes_worker_and_all_runtime_stage_actions(self) -> None:
        actions = experiment._parser()._subparsers
        del actions
        choices = next(
            action.choices
            for action in experiment._parser()._actions
            if getattr(action, "dest", "") == "action"
        )
        self.assertTrue(
            {
                "prepare",
                "worker",
                "launch-backbones",
                "freeze-backbones",
                "graft",
                "launch-continuations",
                "freeze-evaluation",
                "predict",
                "predict-interventions",
                "predict-validation-trajectories",
                "status",
                "validate",
            }.issubset(choices)
        )


if __name__ == "__main__":
    unittest.main()
