"""Contracts for the direct matched-LP FiLM learning-rate sweep."""

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_5arm_seed42 as shared_direct,
)
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42 as experiment,
)
from wgan_option.utils.news_first_experiment_core import (
    write_pair_text_overlay_manifest,
)


EXPECTED_ARMS = (
    "film_lr_2p5e7",
    "film_lr_5e7",
    "film_lr_1e6",
    "film_lr_2p5e6",
    "film_lr_5e6",
)
EXPECTED_RATES = {
    "film_lr_2p5e7": 2.5e-7,
    "film_lr_5e7": 5e-7,
    "film_lr_1e6": 1e-6,
    "film_lr_2p5e6": 2.5e-6,
    "film_lr_5e6": 5e-6,
}


class DirectMatchedFilmLrOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        cls.specs = experiment.planned_specs(cls.config)

    def test_frozen_matrix_optimizer_and_pure_reference_contract(self) -> None:
        self.assertEqual(experiment.DIRECT_ARMS, EXPECTED_ARMS)
        self.assertEqual(dict(experiment.FILM_LEARNING_RATES), EXPECTED_RATES)
        self.assertEqual(self.config["matrix"]["direct_arms"], list(EXPECTED_ARMS))
        self.assertEqual(
            self.config["matrix"]["arm_film_learning_rates"], EXPECTED_RATES
        )
        training = self.config["training"]
        self.assertEqual(
            training["generator_optimizer_profile"], "film_unet_split_lr_v1"
        )
        self.assertEqual(training["generator_learning_rate"], 5e-7)
        self.assertEqual(training["generator_text_learning_rate"], 2.5e-6)
        self.assertEqual(training["discriminator_learning_rate"], 5e-7)
        self.assertEqual(training["generator_text_min_learning_rate"], 2.5e-7)
        self.assertEqual(training["scheduler_min_lr"], 5e-8)
        self.assertEqual(training["group_scheduler_floor_ratio"], 0.1)
        self.assertEqual(training["num_epochs"], 240)
        reference = self.config["analysis"]["frozen_pure_cnn"]
        self.assertEqual(reference["arm"], "pure_cnn_no_text")
        self.assertFalse(reference["retrain"])
        self.assertRegex(reference["pair_metrics_sha256"], r"^[0-9a-f]{64}$")
        experiment.validate_config(copy.deepcopy(self.config))

    def test_twenty_jobs_are_balanced_and_differ_only_in_declared_film_lr(self) -> None:
        self.assertEqual(len(self.specs), 20)
        self.assertEqual(len({row["job_id"] for row in self.specs}), 20)
        self.assertEqual(Counter(row["gpu_id"] for row in self.specs), {0: 10, 1: 10})
        self.assertEqual({row["arm"] for row in self.specs}, set(EXPECTED_ARMS))
        self.assertEqual(
            {row["pair_text_overlay_mode"] for row in self.specs}, {"lp_mean_l2"}
        )
        self.assertEqual(
            {row["generator_optimizer_profile"] for row in self.specs},
            {"film_unet_split_lr_v1"},
        )
        self.assertEqual(
            {row["generator_text_learning_rate"] for row in self.specs}, {2.5e-6}
        )
        for row in self.specs:
            expected = EXPECTED_RATES[str(row["arm"])]
            self.assertEqual(row["generator_film_learning_rate"], expected)
            self.assertEqual(row["generator_film_min_learning_rate"], expected * 0.1)
            self.assertEqual(row["generator_text_min_learning_rate"], 2.5e-7)
        self.assertEqual(
            len({row["initial_generator_state_sha256"] for row in self.specs}), 1
        )
        self.assertEqual(
            len({row["initial_critic_state_sha256"] for row in self.specs}), 1
        )
        experiment.validate_gpu_balance(self.specs)

    def test_training_payload_has_split_lrs_and_never_opens_test(self) -> None:
        for spec in self.specs:
            payload = experiment._training_payload(self.config, Path("/unused"), spec)
            self.assertEqual(
                payload["generator_optimizer_profile"], "film_unet_split_lr_v1"
            )
            self.assertEqual(payload["generator_learning_rate"], 5e-7)
            self.assertEqual(payload["generator_text_learning_rate"], 2.5e-6)
            self.assertEqual(
                payload["generator_film_learning_rate"],
                EXPECTED_RATES[str(spec["arm"])],
            )
            self.assertEqual(
                payload["generator_film_min_learning_rate"],
                EXPECTED_RATES[str(spec["arm"])] * 0.1,
            )
            self.assertEqual(payload["news_first_pair_text_overlay_mode"], "lp_mean_l2")
            self.assertTrue(payload["news_first_materialize_validation_loader"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertEqual(payload["news_first_refit_mode"], "none")

    def test_profile_restores_shared_globals_and_functions_after_failure(self) -> None:
        names = (
            "DIRECT_ARMS",
            "NO_TEXT_ARM",
            "EXPECTED_TRAINING_JOBS",
            "WORKER_MODULE",
            "SOURCE_CODE_RELATIVE_PATHS",
            "load_config",
            "validate_config",
            "planned_specs",
            "_source_paths",
            "_overlay_mode",
            "_materialize_pair_text_overlays",
            "_training_payload",
            "_materialize_test_inputs",
        )
        before = {name: getattr(shared_direct, name) for name in names}
        with self.assertRaisesRegex(RuntimeError, "profile fixture"):
            with experiment.lr_sweep_profile():
                self.assertEqual(shared_direct.DIRECT_ARMS, EXPECTED_ARMS)
                self.assertIs(shared_direct.planned_specs, experiment.planned_specs)
                raise RuntimeError("profile fixture")
        for name, expected in before.items():
            self.assertIs(getattr(shared_direct, name), expected)

    def test_config_rejects_optimizer_lr_overlay_and_pure_hash_drift(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []
        optimizer = copy.deepcopy(self.config)
        optimizer["training"]["generator_optimizer_profile"] = "uniform_v1"
        cases.append(("optimizer", optimizer))
        text_lr = copy.deepcopy(self.config)
        text_lr["training"]["generator_text_learning_rate"] = 1e-6
        cases.append(("text LR", text_lr))
        film_lr = copy.deepcopy(self.config)
        film_lr["matrix"]["arm_film_learning_rates"]["film_lr_1e6"] = 9e-7
        cases.append(("FiLM", film_lr))
        overlay = copy.deepcopy(self.config)
        overlay["text_representations"]["film_lr_1e6"]["mode"] = "current_only"
        cases.append(("matched", overlay))
        pure = copy.deepcopy(self.config)
        pure["analysis"]["frozen_pure_cnn"]["pair_metrics_sha256"] = "0" * 64
        cases.append(("Pure", pure))
        for label, config in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    experiment.validate_config(config)

    def test_core_build_job_receives_overlay_and_group_lr_fields(self) -> None:
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
                shared_direct.core,
                "build_job",
                return_value=(base_job, {"unused": True}),
            ) as build_job,
            experiment.lr_sweep_profile(),
        ):
            shared_direct._build_job(
                self.config,
                Path(directory),
                spec,
                slots_per_gpu=10,
                num_epochs=240,
            )
        passed = build_job.call_args.args[2]
        self.assertEqual(passed["pair_text_overlay_mode"], "lp_mean_l2")
        self.assertEqual(passed["generator_optimizer_profile"], "film_unet_split_lr_v1")
        self.assertEqual(passed["generator_text_learning_rate"], 2.5e-6)
        self.assertEqual(passed["generator_film_learning_rate"], 2.5e-7)
        self.assertEqual(passed["generator_text_min_learning_rate"], 2.5e-7)
        self.assertEqual(passed["generator_film_min_learning_rate"], 2.5e-8)

    def test_all_development_lr_arms_copy_the_same_matched_lp_records(self) -> None:
        vector = np.zeros(1024, dtype=np.float32)
        vector[0] = 1.0

        def fake_builder(*, output_dir: Path, **_kwargs: object) -> list[Path]:
            paths: list[Path] = []
            for fold in shared_direct.FOLDS:
                path = Path(output_dir) / "tolerance_05m" / fold / "lp_matched.json"
                write_pair_text_overlay_manifest(
                    path,
                    mode="lp_mean_l2",
                    namespace=f"fixture/{fold}/lp",
                    records=[
                        {
                            "pair_id": f"{fold}-pair",
                            "session_id": f"{fold}-session",
                            "embedding": vector,
                        }
                    ],
                    transform={"method": "unique_article_lp_mean_l2_v1"},
                )
                paths.append(path)
            return paths * 6

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch(
                "wgan_option.utils.news_first_experiment_core."
                "build_pair_text_overlay_manifests",
                side_effect=fake_builder,
            ):
                manifest = experiment._materialize_pair_text_overlays(self.config, root)
            self.assertTrue(manifest.is_file())
            paths = sorted(
                (root / "inputs/pair_text_overlays").glob("tolerance_05m/*/*.json")
            )
            self.assertEqual(len(paths), 20)
            for fold in shared_direct.FOLDS:
                payloads = [
                    shared_direct.read_json(
                        shared_direct._overlay_path(root, fold, arm)
                    )
                    for arm in EXPECTED_ARMS
                ]
                self.assertTrue(
                    all(payload["mode"] == "lp_mean_l2" for payload in payloads)
                )
                self.assertTrue(
                    all(
                        payload["records"] == payloads[0]["records"]
                        for payload in payloads[1:]
                    )
                )

    def test_optimizer_contract_contains_four_complete_group_traces(self) -> None:
        job = self.specs[2]
        metrics = pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "g_lr_backbone": [5e-7, 5e-7, 2.5e-7],
                "g_lr_text_encoder": [2.5e-6, 2.5e-6, 1.25e-6],
                "g_lr_film_projection": [1e-6, 1e-6, 5e-7],
                "d_lr": [5e-7, 5e-7, 2.5e-7],
            }
        )
        contract = experiment._optimizer_contract(
            job=job, metrics=metrics, epochs_ran=2
        )
        self.assertEqual(
            set(contract["groups"]),
            {"backbone", "text_encoder", "film_projection", "critic"},
        )
        self.assertEqual(
            {
                name: group["parameter_count"]
                for name, group in contract["groups"].items()
            },
            {
                "backbone": 416_353,
                "text_encoder": 295_808,
                "film_projection": 115_584,
                "critic": 729_157,
            },
        )
        for group in contract["groups"].values():
            self.assertEqual([row["epoch"] for row in group["lr_trace"]], [0, 1, 2])

    def test_historical_direct_specs_remain_uniform_and_omit_new_keys(self) -> None:
        old = shared_direct.load_config(shared_direct.DEFAULT_CONFIG)
        specs = shared_direct.planned_specs(old)
        optional = {
            "pair_text_overlay_mode",
            "generator_optimizer_profile",
            "generator_text_learning_rate",
            "generator_film_learning_rate",
            "generator_text_min_learning_rate",
            "generator_film_min_learning_rate",
        }
        self.assertTrue(all(optional.isdisjoint(spec) for spec in specs))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
