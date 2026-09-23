from __future__ import annotations

from copy import deepcopy
import unittest

from scripts.rq3 import news_first_vol_film_unet_direct_5arm_seed42 as direct
from scripts.rq3 import (
    news_first_vol_film_unet_direct_text_ablation_lr2p5e5_seed42 as subject,
)


class DirectTextAblationOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = subject.load_config()

    def test_matrix_is_exactly_sixteen_balanced_jobs(self) -> None:
        specs = subject.planned_specs(self.config)
        self.assertEqual(len(specs), 16)
        self.assertEqual(len({row["job_id"] for row in specs}), 16)
        self.assertEqual({row["seed"] for row in specs}, {42})
        self.assertEqual({row["arm"] for row in specs}, set(subject.DIRECT_ARMS))
        self.assertEqual(
            {gpu: sum(row["gpu_id"] == gpu for row in specs) for gpu in (0, 1)},
            {0: 8, 1: 8},
        )
        self.assertEqual(
            len({row["initial_generator_state_sha256"] for row in specs}), 1
        )
        self.assertEqual(len({row["initial_critic_state_sha256"] for row in specs}), 1)

    def test_all_arms_use_frozen_split_lr_and_never_materialize_test(self) -> None:
        for spec in subject.planned_specs(self.config):
            payload = subject._training_payload(
                self.config, subject.direct.REPO_ROOT, spec
            )
            self.assertEqual(payload["generator_learning_rate"], 5.0e-7)
            self.assertEqual(payload["generator_text_learning_rate"], 2.5e-6)
            self.assertEqual(payload["generator_film_learning_rate"], 2.5e-5)
            self.assertEqual(payload["generator_film_min_learning_rate"], 2.5e-6)
            self.assertEqual(payload["discriminator_learning_rate"], 5.0e-7)
            self.assertEqual(payload["num_epochs"], 240)
            self.assertTrue(payload["news_first_materialize_validation_loader"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertEqual(
                payload["news_first_pair_text_overlay_mode"],
                direct._overlay_mode(spec["arm"]),
            )

    def test_frozen_matched_reference_is_verified_and_not_retrained(self) -> None:
        frozen = subject.ablation_analysis.verify_frozen_reference(self.config)
        self.assertEqual(frozen["reference"]["arm"], "film_lr_2p5e5")
        self.assertFalse(frozen["reference"]["retrain"])
        self.assertEqual(
            set(frozen["generator_checkpoint_sha256_by_fold"]), set(direct.FOLDS)
        )
        self.assertNotIn("lp_matched", subject.DIRECT_ARMS)

    def test_reference_hash_drift_is_rejected(self) -> None:
        config = deepcopy(self.config)
        config["analysis"]["frozen_matched_lp"]["pair_metrics_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pair_metrics drift"):
            subject.validate_config(config)

    def test_profile_restores_every_shared_symbol_after_exception(self) -> None:
        names = (
            "EXPERIMENT_KIND",
            "DIRECT_ARMS",
            "EXPECTED_TRAINING_JOBS",
            "load_config",
            "postprocess",
            "qa",
        )
        originals = {name: getattr(direct, name) for name in names}
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with subject.text_ablation_profile():
                self.assertEqual(direct.DIRECT_ARMS, subject.DIRECT_ARMS)
                self.assertEqual(direct.EXPECTED_TRAINING_JOBS, 16)
                raise RuntimeError("sentinel")
        for name, value in originals.items():
            self.assertIs(getattr(direct, name), value)


if __name__ == "__main__":
    unittest.main()
