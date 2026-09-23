from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import news_first_vol_film_lp_capacity_lr_seed as sweep


class FilmLpCapacityLrSeedSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(sweep.DEFAULT_CONFIG)
        cls.warmup_resolved = sweep.resolve_config(
            "configs/rq3/"
            "news_first_vol_film_lp_critic_capacity_3seed_lr5e7_warmup10.yaml"
        )

    def test_matrix_is_unique_and_gpu_balanced(self) -> None:
        specs = sweep.experiment_specs(self.resolved)
        self.assertEqual(len(specs), 54)
        self.assertEqual(len({spec["job_id"] for spec in specs}), 54)
        self.assertEqual(
            {gpu: sum(spec["gpu_id"] == gpu for spec in specs) for gpu in (0, 1)},
            {0: 27, 1: 27},
        )
        self.assertEqual({spec["seed"] for spec in specs}, {42, 202, 404})
        self.assertEqual(
            {spec["learning_rate"] for spec in specs},
            {5e-7, 2.5e-7, 1.25e-7},
        )

    def test_contract_changes_with_capacity_and_lr(self) -> None:
        contracts = {
            (capacity, lr): sweep._model_contract(self.resolved, capacity, lr)
            for capacity in sweep.CAPACITIES
            for lr in sweep.LEARNING_RATES
        }
        self.assertEqual(
            len({value["model_contract_sha256"] for value in contracts.values()}),
            18,
        )
        for capacity in sweep.CAPACITIES:
            self.assertEqual(
                len(
                    {
                        contracts[(capacity, lr)]["architecture_profile_sha256"]
                        for lr in sweep.LEARNING_RATES
                    }
                ),
                1,
            )
        legacy = contracts[("legacy", 2.5e-7)]
        self.assertEqual(
            legacy["model_contract_sha256"],
            "2d52f0e2b42f5e0392c5e837ee889c8469d4eef090ab763e381996bd413a787c",
        )

    def test_training_payload_freezes_q3_only_contract(self) -> None:
        spec = next(
            spec
            for spec in sweep.experiment_specs(self.resolved)
            if spec["capacity_profile"] == "micro"
            and spec["learning_rate"] == 1.25e-7
            and spec["seed"] == 404
        )
        payload = sweep._training_payload(self.resolved, Path("/tmp/pilot"), spec)
        self.assertEqual(payload["critic_conditioning_mode"], "lp_concat_v1")
        self.assertEqual(payload["generator_learning_rate"], 1.25e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 1.25e-7)
        self.assertEqual(payload["num_epochs"], 30)
        self.assertEqual(payload["validation_mc_samples"], 16)
        self.assertFalse(payload["news_first_materialize_test_loader"])
        self.assertEqual(
            payload["news_first_validation_end_utc"], "2023-10-01T00:00:00Z"
        )

    def test_prepare_is_idempotent_and_rejects_generated_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            registry = sweep.prepare(self.resolved, root, resume=False)
            self.assertEqual(registry["expected_job_count"], 54)
            self.assertEqual(len(list((root / "configs").glob("*.yaml"))), 54)
            resumed = sweep.prepare(self.resolved, root, resume=True)
            self.assertEqual(
                resumed["jobs_payload_sha256"], registry["jobs_payload_sha256"]
            )
            first_config = Path(registry["jobs"][0]["config_path"])
            first_config.write_text(
                first_config.read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "config drift"):
                sweep.prepare(self.resolved, root, resume=True)

    def test_warmup_matrix_and_payload_are_frozen(self) -> None:
        specs = sweep.experiment_specs(self.warmup_resolved)
        self.assertEqual(len(specs), 18)
        self.assertEqual(len({spec["job_id"] for spec in specs}), 18)
        self.assertEqual(
            {gpu: sum(spec["gpu_id"] == gpu for spec in specs) for gpu in (0, 1)},
            {0: 9, 1: 9},
        )
        self.assertEqual({spec["learning_rate"] for spec in specs}, {5e-7})
        self.assertEqual({spec["lr_warmup_epochs"] for spec in specs}, {10})
        self.assertEqual({spec["lr_warmup_start_factor"] for spec in specs}, {0.1})
        self.assertTrue(all("_warmup10_" in spec["job_id"] for spec in specs))

        legacy = next(
            spec
            for spec in specs
            if spec["capacity_profile"] == "legacy" and spec["seed"] == 404
        )
        payload = sweep._training_payload(
            self.warmup_resolved,
            Path("/tmp/warmup-pilot"),
            legacy,
        )
        self.assertEqual(payload["generator_learning_rate"], 5e-7)
        self.assertEqual(payload["discriminator_learning_rate"], 5e-7)
        self.assertEqual(payload["lr_warmup_epochs"], 10)
        self.assertEqual(payload["lr_warmup_start_factor"], 0.1)
        self.assertEqual(payload["reduce_lr_min_lr"], 5e-8)
        self.assertFalse(payload["news_first_materialize_test_loader"])
        contract = sweep._model_contract(
            self.warmup_resolved,
            "legacy",
            5e-7,
        )
        self.assertEqual(contract["lr_warmup_epochs"], 10)
        self.assertEqual(contract["lr_warmup_start_factor"], 0.1)

    def test_warmup_prepare_creates_exactly_18_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "warmup-experiment"
            registry = sweep.prepare(self.warmup_resolved, root, resume=False)
            self.assertEqual(registry["expected_job_count"], 18)
            self.assertEqual(
                registry["experiment_kind"],
                sweep.WARMUP_EXPERIMENT_KIND,
            )
            self.assertEqual(len(list((root / "configs").glob("*.yaml"))), 18)


if __name__ == "__main__":
    unittest.main()
