"""Contracts for the direct matched-LP high FiLM-LR sweep."""

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_high_lr_seed42 as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_direct_matched_lr_seed42 as low_lr,
)


EXPECTED_ARMS = (
    "film_lr_5e6",
    "film_lr_1e5",
    "film_lr_2p5e5",
    "film_lr_5e5",
    "film_lr_1e4",
)
EXPECTED_RATES = {
    "film_lr_5e6": 5.0e-6,
    "film_lr_1e5": 1.0e-5,
    "film_lr_2p5e5": 2.5e-5,
    "film_lr_5e5": 5.0e-5,
    "film_lr_1e4": 1.0e-4,
}


class DirectMatchedHighFilmLrOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = experiment.load_config(experiment.DEFAULT_CONFIG)
        cls.specs = experiment.planned_specs(cls.config)

    def test_frozen_matrix_is_direct_matched_text_and_no_parent(self) -> None:
        self.assertEqual(experiment.DIRECT_ARMS, EXPECTED_ARMS)
        self.assertEqual(dict(experiment.FILM_LEARNING_RATES), EXPECTED_RATES)
        self.assertEqual(self.config["matrix"]["direct_arms"], list(EXPECTED_ARMS))
        self.assertEqual(
            self.config["matrix"]["arm_film_learning_rates"], EXPECTED_RATES
        )
        self.assertEqual(
            self.config["training"]["protocol"], "independent_random_init_v1"
        )
        self.assertEqual(self.config["training"]["num_epochs"], 240)
        self.assertEqual(self.config["analysis"]["bridge_arm"], "film_lr_5e6")
        self.assertEqual(self.config["analysis"]["stress_endpoint_arm"], "film_lr_1e4")
        self.assertFalse(self.config["analysis"]["test_based_lr_selection_permitted"])
        self.assertFalse(self.config["analysis"]["frozen_pure_cnn"]["retrain"])

    def test_twenty_jobs_are_balanced_and_share_initial_states(self) -> None:
        self.assertEqual(len(self.specs), 20)
        self.assertEqual(len({str(row["job_id"]) for row in self.specs}), 20)
        self.assertEqual(
            Counter(int(row["gpu_id"]) for row in self.specs), {0: 10, 1: 10}
        )
        self.assertEqual({str(row["arm"]) for row in self.specs}, set(EXPECTED_ARMS))
        self.assertEqual(
            {str(row["pair_text_overlay_mode"]) for row in self.specs},
            {"lp_mean_l2"},
        )
        self.assertEqual(
            len({str(row["initial_generator_state_sha256"]) for row in self.specs}),
            1,
        )
        self.assertEqual(
            len({str(row["initial_critic_state_sha256"]) for row in self.specs}),
            1,
        )
        for row in self.specs:
            arm = str(row["arm"])
            self.assertEqual(row["generator_film_learning_rate"], EXPECTED_RATES[arm])
            self.assertEqual(
                row["generator_film_min_learning_rate"], EXPECTED_RATES[arm] * 0.1
            )
            self.assertEqual(row["generator_text_learning_rate"], 2.5e-6)
            self.assertEqual(row["generator_text_min_learning_rate"], 2.5e-7)
        experiment.validate_gpu_balance(self.specs)

    def test_training_payload_changes_only_film_lr_group(self) -> None:
        for spec in self.specs:
            payload = experiment._training_payload(self.config, Path("/unused"), spec)
            arm = str(spec["arm"])
            self.assertEqual(
                payload["generator_optimizer_profile"], "film_unet_split_lr_v1"
            )
            self.assertEqual(payload["generator_learning_rate"], 5.0e-7)
            self.assertEqual(payload["generator_text_learning_rate"], 2.5e-6)
            self.assertEqual(payload["discriminator_learning_rate"], 5.0e-7)
            self.assertEqual(
                payload["generator_film_learning_rate"], EXPECTED_RATES[arm]
            )
            self.assertEqual(
                payload["generator_film_min_learning_rate"], EXPECTED_RATES[arm] * 0.1
            )
            self.assertEqual(payload["news_first_pair_text_overlay_mode"], "lp_mean_l2")
            self.assertTrue(payload["news_first_materialize_validation_loader"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertEqual(payload["news_first_refit_mode"], "none")

    def test_profile_patches_low_direct_and_core_then_restores_after_error(
        self,
    ) -> None:
        direct = low_lr.direct
        core = direct.core
        low_names = (
            "DEFAULT_CONFIG",
            "DEFAULT_OUTPUT_DIR",
            "EXPERIMENT_KIND",
            "DIRECT_ARMS",
            "FILM_LEARNING_RATES",
            "WORKER_MODULE",
            "SOURCE_CODE_RELATIVE_PATHS",
            "lr_analysis",
            "_overlay_mode",
            "benchmark",
            "prepare_experiment",
        )
        direct_names = (
            "DEFAULT_CONFIG",
            "DEFAULT_OUTPUT_DIR",
            "EXPERIMENT_KIND",
            "DIRECT_ARMS",
            "WORKER_MODULE",
            "SOURCE_CODE_RELATIVE_PATHS",
            "_overlay_mode",
        )
        low_before = {name: getattr(low_lr, name) for name in low_names}
        direct_before = {name: getattr(direct, name) for name in direct_names}
        core_before = core._overlay_mode
        with self.assertRaisesRegex(RuntimeError, "profile fixture"):
            with experiment.high_lr_profile():
                self.assertEqual(low_lr.DIRECT_ARMS, EXPECTED_ARMS)
                self.assertEqual(direct.DIRECT_ARMS, EXPECTED_ARMS)
                self.assertIs(low_lr._overlay_mode, experiment._overlay_mode)
                self.assertIs(direct._overlay_mode, experiment._overlay_mode)
                self.assertIs(core._overlay_mode, experiment._overlay_mode)
                self.assertEqual(core._overlay_mode("film_lr_1e4"), "lp_mean_l2")
                raise RuntimeError("profile fixture")
        for name, value in low_before.items():
            self.assertIs(getattr(low_lr, name), value)
        for name, value in direct_before.items():
            self.assertIs(getattr(direct, name), value)
        self.assertIs(core._overlay_mode, core_before)

    def test_config_rejects_rate_bridge_stress_and_selection_drift(self) -> None:
        cases = []
        rate = copy.deepcopy(self.config)
        rate["matrix"]["arm_film_learning_rates"]["film_lr_1e4"] = 9.0e-5
        cases.append(rate)
        bridge = copy.deepcopy(self.config)
        bridge["analysis"]["bridge_arm"] = "film_lr_1e5"
        cases.append(bridge)
        stress = copy.deepcopy(self.config)
        stress["analysis"]["stress_endpoint_arm"] = "film_lr_5e5"
        cases.append(stress)
        selection = copy.deepcopy(self.config)
        selection["analysis"]["test_based_lr_selection_permitted"] = True
        cases.append(selection)
        for config in cases:
            with self.subTest(config=config["analysis"]):
                with self.assertRaises(ValueError):
                    experiment.validate_config(config)

    def test_stress_canary_prepares_five_epochs_and_only_runs_four_1e4_jobs(
        self,
    ) -> None:
        jobs = [
            {
                "job_id": f"stress_{fold}",
                "fold": fold,
                "arm": experiment.STRESS_ARM,
                "wave": 0,
                "gpu_id": index % 2,
            }
            for index, fold in enumerate(low_lr.direct.FOLDS)
        ]
        jobs.extend(
            {
                "job_id": f"other_{index}",
                "fold": low_lr.direct.FOLDS[index % 4],
                "arm": experiment.BRIDGE_ARM,
                "wave": 0,
                "gpu_id": index % 2,
            }
            for index in range(16)
        )
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory) / "formal"
            canary_root = experiment._stress_root(formal)

            def fake_prepare(
                _config: object,
                root: Path,
                **kwargs: object,
            ) -> Path:
                self.assertEqual(
                    kwargs["workers_per_gpu"],
                    self.config["runtime"]["fallback_workers_per_gpu"],
                )
                self.assertEqual(kwargs["num_epochs"], 5)
                self.assertEqual(kwargs["root_mode"], "benchmark")
                (root / "registry").mkdir(parents=True)
                (root / "registry/task_registry.json").write_text(
                    "{}\n", encoding="utf-8"
                )
                return root

            with (
                experiment.high_lr_profile(),
                mock.patch.object(
                    experiment, "_BASE_PREPARE", side_effect=fake_prepare
                ),
                mock.patch.object(low_lr.direct, "validate_root", return_value=config),
                mock.patch.object(
                    low_lr.direct, "read_registry", return_value={"jobs": jobs}
                ),
                mock.patch.object(
                    low_lr.direct,
                    "read_json",
                    return_value={"status": "pending", "artifacts": []},
                ),
                mock.patch.object(
                    low_lr.direct, "_completed_valid", return_value=False
                ),
                mock.patch.object(low_lr.direct, "_run_wave", return_value=0.10) as run,
                mock.patch.object(
                    low_lr.direct,
                    "_resource_peaks",
                    return_value={
                        "peak_gpu_memory_mib": 1024.0,
                        "peak_gpu_memory_gib": 1.0,
                        "telemetry_rows": 4,
                        "telemetry_sha256": "a" * 64,
                    },
                ),
                mock.patch.object(
                    experiment,
                    "_verify_stress_result",
                    return_value={"status": "passed"},
                ),
            ):
                result = experiment._stress_canary_active(config, formal, resume=True)
            self.assertEqual(result, experiment._stress_result_path(formal))
            selected = run.call_args.args[1]
            self.assertEqual(len(selected), 4)
            self.assertEqual({row["arm"] for row in selected}, {experiment.STRESS_ARM})
            self.assertTrue(canary_root.is_dir())

    def test_stress_metrics_allow_epoch_zero_train_nan_but_reject_learned_inf(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "epoch": list(range(6)),
                "g_lr_film_projection": [1.0e-4] * 6,
                "val_hybrid_score": [0.2] * 6,
                "g_total": [float("nan"), 1.0, 0.9, 0.8, 0.7, 0.6],
            }
        )
        experiment._validate_stress_metrics(frame, job_id="fixture")
        frame.loc[3, "g_total"] = float("inf")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            experiment._validate_stress_metrics(frame, job_id="fixture")

    def test_formal_prepare_refuses_missing_stress_evidence(self) -> None:
        with (
            experiment.high_lr_profile(),
            mock.patch.object(
                experiment,
                "_verify_stress_result",
                side_effect=FileNotFoundError("missing stress result"),
            ),
            mock.patch.object(experiment, "_BASE_PREPARE_EXPERIMENT") as prepare,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing stress"):
                experiment._prepare_experiment_with_stress(
                    experiment.DEFAULT_CONFIG,
                    experiment.DEFAULT_OUTPUT_DIR,
                    resume=True,
                )
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
