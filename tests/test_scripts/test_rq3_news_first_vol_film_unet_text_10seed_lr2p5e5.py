"""Contracts for the ten-seed, four-arm direct FiLM-text experiment."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import unittest

from scripts.rq3 import news_first_vol_film_unet_text_10seed_lr2p5e5 as subject


class FilmTextTenSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = subject.load_config()
        cls.specs = subject.planned_specs(cls.config)

    def test_exact_matrix_and_gpu_balance(self) -> None:
        self.assertEqual(len(subject.SEEDS), 10)
        self.assertEqual(
            subject.DIRECT_ARMS,
            ("lp_matched", "lp_shuffle", "bow", "sentiment"),
        )
        self.assertEqual(len(self.specs), 160)
        self.assertEqual(len({str(row["job_id"]) for row in self.specs}), 160)
        self.assertEqual(
            Counter(str(row["arm"]) for row in self.specs),
            {arm: 40 for arm in subject.DIRECT_ARMS},
        )
        self.assertEqual(
            Counter(int(row["seed"]) for row in self.specs),
            {seed: 16 for seed in subject.SEEDS},
        )
        self.assertEqual(
            Counter(int(row["gpu_id"]) for row in self.specs), {0: 80, 1: 80}
        )
        subject.validate_gpu_balance(self.specs)

    def test_initial_states_are_common_within_seed_and_distinct_between_seeds(
        self,
    ) -> None:
        generator_hashes: list[str] = []
        critic_hashes: list[str] = []
        for seed in subject.SEEDS:
            rows = [row for row in self.specs if int(row["seed"]) == seed]
            generators = {str(row["initial_generator_state_sha256"]) for row in rows}
            critics = {str(row["initial_critic_state_sha256"]) for row in rows}
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(generators), 1)
            self.assertEqual(len(critics), 1)
            generator_hashes.extend(generators)
            critic_hashes.extend(critics)
        self.assertEqual(len(set(generator_hashes)), 10)
        self.assertEqual(len(set(critic_hashes)), 10)

    def test_every_job_is_direct_split_lr_and_test_closed(self) -> None:
        expected_modes = {
            "lp_matched": "lp_mean_l2",
            "lp_shuffle": "lp_shuffle",
            "bow": "bow1024",
            "sentiment": "sentiment_pad1024",
        }
        for spec in self.specs:
            payload = subject._training_payload(self.config, Path("/unused"), spec)
            self.assertEqual(payload["generator_learning_rate"], 5.0e-7)
            self.assertEqual(payload["generator_text_learning_rate"], 2.5e-6)
            self.assertEqual(payload["generator_film_learning_rate"], 2.5e-5)
            self.assertEqual(payload["discriminator_learning_rate"], 5.0e-7)
            self.assertEqual(payload["num_epochs"], 240)
            self.assertEqual(payload["seed"], int(spec["seed"]))
            self.assertEqual(
                payload["news_first_pair_text_overlay_mode"],
                expected_modes[str(spec["arm"])],
            )
            self.assertTrue(payload["news_first_materialize_validation_loader"])
            self.assertFalse(payload["news_first_materialize_test_loader"])
            self.assertEqual(payload["news_first_refit_mode"], "none")

    def test_parent_or_continuation_contract_is_rejected(self) -> None:
        for key in ("parent_arms", "continuation_arms", "branch_recipes"):
            config = deepcopy(self.config)
            config["matrix"][key] = []
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "prohibited parent/continuation"),
            ):
                subject.validate_config(config)

    def test_profile_restores_reused_multiseed_module(self) -> None:
        names = (
            "DIRECT_ARMS",
            "SEEDS",
            "EXPECTED_TRAINING_JOBS",
            "load_config",
            "postprocess",
        )
        before = {name: getattr(subject.multiseed, name) for name in names}
        with subject.film_text_profile():
            self.assertEqual(subject.multiseed.DIRECT_ARMS, subject.DIRECT_ARMS)
            self.assertEqual(subject.multiseed.SEEDS, subject.SEEDS)
            self.assertEqual(subject.multiseed.EXPECTED_TRAINING_JOBS, 160)
        for name, expected in before.items():
            self.assertIs(getattr(subject.multiseed, name), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
