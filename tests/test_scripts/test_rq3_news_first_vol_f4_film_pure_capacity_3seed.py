from __future__ import annotations

import importlib.util
from itertools import product
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import ANY, Mock, patch

import pandas as pd
import yaml

from scripts.rq3.main import build_parser, main as rq3_main
from scripts.rq3 import news_first_vol_f4_film_pure_capacity_3seed_analysis as analysis


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rq3/news_first_vol_f4_film_pure_capacity_3seed.yaml"
COMMAND = "train-news-first-vol-f4-film-pure-capacity-3seed"
TORCH_RUNTIME_AVAILABLE = importlib.util.find_spec("torch.nn") is not None


def _runtime_modules():
    from scripts.rq3 import news_first_vol_f4_film_pure_capacity_3seed as runner
    from scripts.rq3 import (
        news_first_vol_f4_film_pure_capacity_3seed_evaluation as evaluation,
    )

    return runner, evaluation


class F4FilmPureCapacityCliContractTests(unittest.TestCase):
    def test_cli_exposes_the_simplified_f4_lifecycle(self) -> None:
        actions = (
            "benchmark",
            "prepare",
            "dry-run",
            "launch",
            "freeze-checkpoints",
            "evaluate-f4",
            "postprocess",
            "qa",
            "worker",
            "status",
        )
        parser = build_parser()
        for action in actions:
            with self.subTest(action=action):
                args = parser.parse_args([COMMAND, action])
                self.assertEqual(args.command, COMMAND)
                self.assertEqual(args.action, action)

        defaults = parser.parse_args([COMMAND])
        self.assertEqual(defaults.action, "prepare")
        self.assertEqual(defaults.config, str(CONFIG.relative_to(ROOT)))
        self.assertEqual(
            defaults.output_dir,
            "outputs/experiments/"
            "rq3_news_first_vol_f4_film_pure_capacity_3seed_exact_ttm_v1",
        )

    def test_cli_dispatches_all_orchestration_flags_lazily(self) -> None:
        module_name = "scripts.rq3.news_first_vol_f4_film_pure_capacity_3seed"
        fake_module = ModuleType(module_name)
        expected = ROOT / "outputs/test-f4-cli"
        runner = Mock(return_value=expected)
        fake_module.run_news_first_vol_f4_film_pure_capacity_3seed = runner

        with patch.dict(sys.modules, {module_name: fake_module}):
            observed = rq3_main(
                [
                    COMMAND,
                    "launch",
                    "--config",
                    str(CONFIG),
                    "--output-dir",
                    str(expected),
                    "--job-id",
                    "film_cnn_c48_seed404",
                    "--resume",
                    "--reuse",
                    "--worker-dry-run",
                ]
            )

        self.assertEqual(observed, expected)
        runner.assert_called_once_with(
            str(CONFIG),
            str(expected),
            action="launch",
            job_id="film_cnn_c48_seed404",
            resume=True,
            reuse=True,
            worker_dry_run=True,
        )

    def test_config_defines_exactly_the_36_independent_f4_jobs(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        matrix = payload["matrix"]
        architectures = tuple(matrix["architectures"])
        capacities = tuple(matrix["capacities"])
        seeds = tuple(int(seed) for seed in matrix["seeds"])

        self.assertEqual(architectures, ("film_cnn", "pure_cnn"))
        self.assertEqual(capacities, ("c08", "c12", "c16", "c24", "c32", "c48"))
        self.assertNotIn("c04", capacities)
        self.assertEqual(seeds, (42, 202, 404))
        self.assertEqual(matrix["fold"], "f4_2023q4")
        self.assertEqual(matrix["current_reference_capacity"], "c32")

        jobs = set(product(architectures, capacities, seeds))
        self.assertEqual(len(jobs), 36)
        self.assertEqual(matrix["expected_training_jobs"], len(jobs))
        self.assertEqual(matrix["expected_prediction_cells"], len(jobs))
        self.assertEqual(matrix["expected_pair_metric_rows"], 36 * 143)

    def test_c48_and_parameter_count_contracts_are_frozen(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        c32 = payload["profiles"]["c32"]
        c48 = payload["profiles"]["c48"]

        self.assertEqual(c48["gen_base_channels"], 48)
        self.assertEqual(c48["gen_hidden_dim"], 1536)
        self.assertEqual(c48["disc_base_channels"], 48)
        self.assertEqual(c48["disc_hidden_dim"], 1179)
        self.assertEqual(c48["gen_base_channels"], c32["gen_base_channels"] * 3 // 2)
        self.assertEqual(c48["gen_hidden_dim"], c32["gen_hidden_dim"] * 3 // 2)
        self.assertEqual(c48["disc_base_channels"], c32["disc_base_channels"] * 3 // 2)
        self.assertEqual(c48["disc_hidden_dim"], c32["disc_hidden_dim"] * 3 // 2)

        expected = payload["expected_parameter_counts"]
        self.assertEqual(expected["film_cnn"]["c32"]["total"], 1_556_902)
        self.assertEqual(expected["pure_cnn"]["c32"]["total"], 1_145_510)
        self.assertEqual(expected["film_cnn"]["c48"]["total"], 2_776_184)
        self.assertEqual(expected["pure_cnn"]["c48"]["total"], 2_307_000)
        for architecture in payload["matrix"]["architectures"]:
            for capacity in payload["matrix"]["capacities"]:
                counts = expected[architecture][capacity]
                self.assertEqual(
                    counts["total"], counts["generator"] + counts["critic"]
                )

    def test_fold_boundaries_counts_and_training_contract_are_single_f4(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        data = payload["data"]
        self.assertEqual(data["train_end_utc"], "2023-07-01T00:00:00Z")
        self.assertEqual(data["validation_end_utc"], "2023-10-01T00:00:00Z")
        self.assertEqual(data["test_end_utc"], "2024-01-01T00:00:00Z")
        self.assertEqual(
            data["expected_counts"],
            {
                "train": {"pairs": 748, "sessions": 203},
                "validation": {"pairs": 135, "sessions": 33},
                "test": {"pairs": 143, "sessions": 45},
            },
        )

        training = payload["training"]
        self.assertEqual(training["num_epochs"], 240)
        self.assertEqual(training["early_stopping_min_epochs"], 30)
        self.assertEqual(training["early_stopping_patience"], 20)
        self.assertEqual(training["validation_mc_samples"], 16)
        self.assertEqual(training["prediction_mc_samples"], 64)
        self.assertTrue(training["shared_prediction_noise_bank"])

    @unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "requires the training environment")
    def test_c32_regression_anchors_use_a_tight_relative_gate(self) -> None:
        runner, _ = _runtime_modules()
        observed = {
            "film_cnn": 0.0016690755742066528,
            "pure_cnn": 0.0016692537767823113,
        }
        expected_delta = {
            "film_cnn": 1.567737116904902e-08,
            "pure_cnn": -5.2598093115349687e-08,
        }
        self.assertEqual(runner.C32_REGRESSION_RELATIVE_TOLERANCE, 5e-5)
        for architecture in runner.ARCHITECTURES:
            anchor = runner.C32_REGRESSION_CONTROLS[architecture][
                "anchor_observed_mae"
            ]
            comparison = runner._c32_mae_regression(
                observed[architecture], anchor
            )
            self.assertAlmostEqual(
                comparison["delta_current_minus_anchor"],
                expected_delta[architecture],
                places=18,
            )
            self.assertTrue(comparison["within_relative_tolerance"])

            material_drift = runner._c32_mae_regression(
                anchor * (1.0 + 6e-5), anchor
            )
            self.assertFalse(material_drift["within_relative_tolerance"])

    @unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "requires the training environment")
    def test_c32_pair_panel_identity_is_order_invariant_and_count_locked(self) -> None:
        runner, _ = _runtime_modules()
        frame = pd.DataFrame(
            {
                "pair_id": [f"pair_{index:03d}" for index in range(143)],
                "session_id": [f"session_{index % 45:02d}" for index in range(143)],
            }
        )
        expected = runner._pair_panel_identity(frame, label="ordered")
        shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        self.assertEqual(
            runner._pair_panel_identity(shuffled, label="shuffled"), expected
        )
        with self.assertRaisesRegex(ValueError, "143 unique pairs in 45 sessions"):
            runner._pair_panel_identity(frame.iloc[:-1], label="incomplete")

    @unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "requires the training environment")
    def test_runner_dispatches_each_post_training_stage_after_completion(self) -> None:
        runner, _ = _runtime_modules()
        completed = {"counts": {"complete": runner.EXPECTED_JOB_COUNT}}
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            frozen = output_root / "registry/f4_checkpoint_allowlist.csv"
            evaluated = output_root / "evaluation/f4_pair_metrics.csv.gz"
            postprocessed = {"summary_csv": output_root / "analysis/summary.csv"}
            passed = {"status": "passed"}
            with (
                patch.object(runner, "status", return_value=completed),
                patch.object(runner, "freeze_checkpoints", return_value=frozen) as freeze,
                patch.object(runner, "evaluate_f4", return_value=evaluated) as evaluate,
                patch.object(runner, "postprocess", return_value=postprocessed) as process,
                patch.object(runner, "qa", return_value=passed) as qa,
                patch("builtins.print"),
            ):
                for action in (
                    "freeze-checkpoints",
                    "evaluate-f4",
                    "postprocess",
                    "qa",
                ):
                    observed = runner.run_news_first_vol_f4_film_pure_capacity_3seed(
                        CONFIG,
                        output_root,
                        action=action,
                        resume=True,
                    )
                    self.assertEqual(observed, output_root)

            freeze.assert_called_once_with(output_root)
            evaluate.assert_called_once_with(ANY, output_root, resume=True)
            process.assert_called_once_with(ANY, output_root)
            qa.assert_called_once_with(ANY, output_root)

    @unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "requires the training environment")
    def test_postprocess_binds_frozen_pair_metrics_and_parameter_contract(self) -> None:
        runner, _ = _runtime_modules()
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            expected = {"summary_csv": output_root / "analysis/summary.csv"}
            with patch.object(
                analysis, "postprocess_experiment", return_value=expected
            ) as process:
                observed = runner.postprocess(payload, output_root)

        self.assertEqual(observed, expected)
        process.assert_called_once_with(
            output_root,
            output_root / "evaluation/f4_pair_metrics.csv.gz",
            parameter_counts=payload["expected_parameter_counts"],
        )

    @unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "requires the training environment")
    def test_evaluation_wrappers_use_the_frozen_module_api(self) -> None:
        runner, evaluation = _runtime_modules()
        config = {"sentinel": True}
        output_root = ROOT / "outputs/test-f4-runner-wrappers"
        allowlist = output_root / "registry/f4_checkpoint_allowlist.csv"
        metrics = output_root / "evaluation/f4_pair_metrics.csv.gz"
        with (
            patch.object(
                evaluation, "freeze_checkpoints", return_value=allowlist
            ) as freeze,
            patch.object(evaluation, "evaluate_f4", return_value=metrics) as evaluate,
        ):
            self.assertEqual(runner.freeze_checkpoints(output_root), allowlist)
            self.assertEqual(
                runner.evaluate_f4(config, output_root, resume=True), metrics
            )

        freeze.assert_called_once_with(output_root)
        evaluate.assert_called_once_with(config, output_root, resume=True)

    @unittest.skipUnless(TORCH_RUNTIME_AVAILABLE, "requires the training environment")
    def test_post_training_stage_is_blocked_until_all_jobs_complete(self) -> None:
        runner, _ = _runtime_modules()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "status", return_value={"counts": {"complete": 35}}),
        ):
            with self.assertRaisesRegex(RuntimeError, "all 36 completed"):
                runner.run_news_first_vol_f4_film_pure_capacity_3seed(
                    CONFIG,
                    temporary,
                    action="freeze-checkpoints",
                )


if __name__ == "__main__":
    unittest.main()
