"""Integration contracts for the 10-seed Pure-CNN -> FiLM experiment.

These tests intentionally exercise the public experiment state machine rather
than the lower-level training implementation.  Their purpose is to keep the
280-job universe, the pre-test freeze boundary, and the three-layer text claim
from drifting while the long-running supervisor is assembled.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)


class TextEffectExperimentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config()

    def _prepared_registry(self) -> dict[str, object]:
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
            "standard_predictions_frozen": False,
            "interventions_frozen": False,
            "validation_trajectories_frozen": False,
            "analysis_complete": False,
            "terminal_complete": False,
        }

    @staticmethod
    def _bind_dummy_grafts(registry: dict[str, object]) -> None:
        jobs: list[dict[str, object]] = []
        for raw in registry["jobs"]:  # type: ignore[index]
            job = dict(raw)
            if job["stage"] == "continuation":
                job.update(
                    graft_state_path="/frozen/graft.pt",
                    graft_state_sha256="a" * 64,
                    graft_manifest_path="/frozen/graft.json",
                    graft_manifest_sha256="b" * 64,
                    initial_generator_state_sha256="c" * 64,
                    initial_critic_state_sha256="d" * 64,
                )
            jobs.append(job)
        registry["jobs"] = jobs
        registry["jobs_sha256"] = experiment.direct.payload_sha256(jobs)

    def test_config_and_job_universe_are_exact(self) -> None:
        experiment.validate_config(self.config, verify_source_evidence=False)
        jobs = experiment.planned_job_specs(self.config)
        self.assertEqual(len(jobs), 280)
        self.assertEqual(len({str(row["job_id"]) for row in jobs}), 280)
        self.assertEqual(
            Counter(row["stage"] for row in jobs), {"backbone": 40, "continuation": 240}
        )
        self.assertEqual(
            Counter(row["arm"] for row in jobs),
            {
                experiment.PARENT_ARM: 40,
                **{arm: 40 for arm in experiment.CONTINUATION_ARMS},
            },
        )
        self.assertEqual(
            Counter(int(row["gpu_id"]) for row in jobs if row["stage"] == "backbone"),
            {0: 20, 1: 20},
        )
        self.assertEqual(
            Counter(
                int(row["gpu_id"]) for row in jobs if row["stage"] == "continuation"
            ),
            {0: 120, 1: 120},
        )

        for seed in experiment.SEEDS:
            for fold in experiment.FOLDS:
                block = [
                    row
                    for row in jobs
                    if int(row["seed"]) == seed and str(row["fold"]) == fold
                ]
                self.assertEqual(len(block), 7)
                parent = next(row for row in block if row["stage"] == "backbone")
                continuations = [row for row in block if row["stage"] == "continuation"]
                self.assertEqual(
                    {str(row["arm"]) for row in continuations},
                    set(experiment.CONTINUATION_ARMS),
                )
                self.assertTrue(
                    all(
                        row["parent_job_id"] == parent["job_id"]
                        for row in continuations
                    )
                )
                self.assertEqual(
                    {int(row["gpu_id"]) for row in block}, {int(parent["gpu_id"])}
                )

    def test_executable_model_profiles_match_frozen_parameter_counts(self) -> None:
        pure_generator, pure_critic = experiment._instantiate_modules(
            self.config, generator_mode=experiment.PURE_MODE, seed=42
        )
        film_generator, film_critic = experiment._instantiate_modules(
            self.config, generator_mode=experiment.FILM_MODE, seed=42
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in pure_generator.parameters()),
            416_353,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in film_generator.parameters()),
            827_745,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in pure_critic.parameters()),
            729_157,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in film_critic.parameters()),
            729_157,
        )

    def test_config_fail_closes_if_test_freeze_policy_is_weakened(self) -> None:
        for key, unsafe_value in (
            ("freeze_all_checkpoints_before_test", False),
            ("test_loader_allowed_before_freeze", True),
        ):
            changed = deepcopy(self.config)
            changed["evaluation"][key] = unsafe_value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "freeze|test loader|test access"),
            ):
                experiment.validate_config(changed, verify_source_evidence=False)

    def test_config_freezes_text_effect_decision_rules(self) -> None:
        mutations = (
            (
                "intervention source",
                ("evaluation", "interventions", "source_checkpoint_arm"),
                "film_lp_shuffle",
            ),
            (
                "independent intervention mapping",
                (
                    "evaluation",
                    "interventions",
                    "mapping_must_differ_from_training_shuffle",
                ),
                False,
            ),
            (
                "seed consistency gate",
                ("analysis", "primary_support_gate", "minimum_nonworse_seeds"),
                6,
            ),
            (
                "fold consistency gate",
                ("analysis", "primary_support_gate", "minimum_nonworse_folds"),
                2,
            ),
        )
        for label, path, value in mutations:
            changed = deepcopy(self.config)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    ValueError, "intervention|mapping|seed|fold|support gate"
                ),
            ):
                experiment.validate_config(changed, verify_source_evidence=False)

    def test_public_state_machine_rejects_test_access_before_freeze(self) -> None:
        registry = self._prepared_registry()
        experiment._validate_registry_state(self.config, registry)
        registry["test_data_opened"] = True
        with self.assertRaisesRegex(
            ValueError, "test_data_opened|test data|evaluation"
        ):
            experiment._validate_registry_state(self.config, registry)

    def test_analysis_requires_all_three_evidence_layers(self) -> None:
        registry = self._prepared_registry()
        self._bind_dummy_grafts(registry)
        registry.update(
            parents_frozen=True,
            grafts_frozen=True,
            continuations_prepared=True,
            evaluation_frozen=True,
            checkpoint_allowlist_rows=560,
            test_data_opened=True,
            standard_predictions_frozen=True,
            standard_prediction_cells=280,
            standard_pair_metric_rows=35_000,
            analysis_complete=True,
        )
        # The standard frozen-test arm comparison alone cannot support the
        # planned text-effect claim: trajectory and same-checkpoint evidence
        # are both mandatory inputs to analysis.
        for missing_flag in (
            "interventions_frozen",
            "validation_trajectories_frozen",
        ):
            candidate = deepcopy(registry)
            candidate["interventions_frozen"] = True
            candidate["intervention_prediction_cells"] = 80
            candidate["intervention_pair_metric_rows"] = 10_000
            candidate["validation_trajectories_frozen"] = True
            candidate[missing_flag] = False
            with (
                self.subTest(missing_flag=missing_flag),
                self.assertRaisesRegex(
                    ValueError, "analysis_complete|Invalid registry phase"
                ),
            ):
                experiment._validate_registry_state(self.config, candidate)

    def test_predict_entry_does_not_mutate_or_dispatch_before_freeze(self) -> None:
        registry = self._prepared_registry()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            with (
                mock.patch.object(
                    experiment, "validate_root", return_value=self.config
                ),
                mock.patch.object(experiment, "_read_registry", return_value=registry),
                mock.patch.object(experiment, "_write_registry") as write_registry,
                mock.patch.object(experiment.pure, "predict") as pure_predict,
                mock.patch.object(experiment.film_text, "predict") as film_predict,
            ):
                with self.assertRaisesRegex(ValueError, "evaluation_frozen|frozen"):
                    experiment.predict_test(root)
            write_registry.assert_not_called()
            pure_predict.assert_not_called()
            film_predict.assert_not_called()
            self.assertFalse(root.exists())

    def test_graft_allowlist_enforces_identity_and_optimizer_reset(self) -> None:
        proof = {
            "generator_optimizer_state_empty": True,
            "discriminator_optimizer_state_empty": True,
            "generator_scheduler_fresh": True,
            "discriminator_scheduler_fresh": True,
            "apply_loads_optimizer_or_scheduler_state": False,
        }
        entries = [
            {
                "seed": seed,
                "fold": fold,
                "target_generator_mode": mode,
                "epoch0_max_abs": 0.0,
                "optimizer_reset_proof": proof,
            }
            for seed in experiment.SEEDS
            for fold in experiment.FOLDS
            for mode in (experiment.PURE_MODE, experiment.FILM_MODE)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowlist = root / "registry/graft_allowlist.json"

            def freeze(current_entries: list[dict[str, object]]) -> dict[str, str]:
                payload: dict[str, object] = {
                    "kind": experiment.GRAFT_ALLOWLIST_KIND,
                    "entry_count": 80,
                    "film_graft_count": 40,
                    "pure_restart_count": 40,
                    "entries": current_entries,
                }
                payload["payload_sha256"] = experiment.direct.payload_sha256(payload)
                experiment._write_json(allowlist, payload)
                return {
                    "graft_allowlist_path": str(allowlist),
                    "graft_allowlist_sha256": experiment.sha256_file(allowlist),
                }

            registry = freeze(entries)
            validated = experiment._validate_graft_allowlist(
                root, registry, verify_artifacts=False
            )
            self.assertEqual(len(validated), 80)

            bad_identity = deepcopy(entries)
            bad_identity[0]["epoch0_max_abs"] = 1.1e-7
            with self.assertRaisesRegex(ValueError, "epoch-0 equivalence"):
                experiment._validate_graft_allowlist(
                    root, freeze(bad_identity), verify_artifacts=False
                )

            bad_reset = deepcopy(entries)
            bad_reset[0]["optimizer_reset_proof"]["generator_optimizer_state_empty"] = (
                False
            )
            with self.assertRaisesRegex(ValueError, "optimizer/scheduler"):
                experiment._validate_graft_allowlist(
                    root, freeze(bad_reset), verify_artifacts=False
                )

    def test_cli_exposes_every_planned_lifecycle_action(self) -> None:
        parser = experiment._parser()
        action = next(item for item in parser._actions if item.dest == "action")
        required = {
            "prepare",
            "benchmark",
            "launch-backbones",
            "freeze-backbones",
            "graft",
            "launch-continuations",
            "freeze-evaluation",
            "predict-validation-trajectories",
            "predict",
            "predict-interventions",
            "analyze",
            "bootstrap",
            "report",
            "qa",
            "status",
            "run-pipeline",
        }
        self.assertTrue(required.issubset(set(action.choices)))

    def test_status_on_missing_root_is_read_only_and_reports_contract_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            payload = experiment.status(root)
            self.assertFalse(root.exists())
        self.assertEqual(payload["status"], "not_prepared")
        self.assertEqual(payload["expected_training_jobs"], 280)
        self.assertEqual(payload["expected_standard_predictions"], 280)
        self.assertEqual(payload["expected_intervention_predictions"], 80)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
