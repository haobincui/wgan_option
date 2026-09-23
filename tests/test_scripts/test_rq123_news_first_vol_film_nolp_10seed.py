from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from scripts.rq123 import news_first_vol_film_nolp_10seed as experiment


class UnifiedRq123OrchestratorTests(unittest.TestCase):
    def test_config_and_training_matrix_are_exact(self) -> None:
        config = experiment.load_config(experiment.DEFAULT_CONFIG)
        specs = experiment.planned_specs()

        self.assertEqual(len(specs), 400)
        self.assertEqual(
            Counter(row["stage"] for row in specs),
            {"parents": 80, "continuations": 80, "text_branches": 240},
        )
        experiment.validate_gpu_balance(specs)
        blocks = {
            (row["tolerance_minutes"], row["fold"], row["seed"]): row["gpu_id"]
            for row in specs
        }
        self.assertEqual(Counter(blocks.values()), {0: 40, 1: 40})
        self.assertEqual(
            experiment.model_contract(config)["total_parameters"], 4_749_445
        )
        architecture = experiment.architecture_profile_contract(config)
        self.assertEqual(
            architecture["architecture_profile_sha256"],
            experiment.EXPECTED_ARCHITECTURE_PROFILE_SHA256,
        )
        self.assertEqual(
            experiment.model_contract(config)["architecture_profile_sha256"],
            architecture["architecture_profile_sha256"],
        )
        self.assertTrue(
            {
                "scripts/rq3/news_first_vol_capacity_analysis.py",
                "scripts/rq3/news_first_vol_capacity_report.py",
                "scripts/rq3/news_first_vol_training_report.py",
            }.issubset(experiment.SOURCE_CODE_RELATIVE_PATHS)
        )

    def test_v1_config_is_immutable_and_v2_root_is_explicit(self) -> None:
        v1 = Path("configs/rq123/news_first_vol_film_nolp_legacy_10seed.yaml")
        self.assertEqual(
            experiment.sha256_file(v1),
            "38031f914e7e81aff58809ad61eb1e26dd3440c8c33abdeb22b9a7966ac2303e",
        )
        self.assertTrue(experiment.DEFAULT_CONFIG.endswith("_v2.yaml"))
        self.assertTrue(experiment.DEFAULT_OUTPUT_DIR.endswith("_v2"))
        self.assertTrue(experiment.EXPERIMENT_KIND.endswith("_v2"))

    def test_parent_epoch_cap_is_stage_specific_and_legacy_safe(self) -> None:
        legacy = experiment.load_config(
            "configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml"
        )
        legacy_training = legacy["training"]
        self.assertNotIn("parent_num_epochs", legacy_training)
        self.assertEqual(
            experiment._stage_num_epochs(legacy_training, experiment.PARENT_STAGE),
            100,
        )
        self.assertEqual(
            experiment._stage_num_epochs(
                legacy_training, experiment.CONTINUATION_STAGE
            ),
            100,
        )

        reduced = experiment.load_config(
            "configs/rq123/news_first_vol_film_nolp_legacy_10seed_parent15.yaml"
        )
        reduced_training = reduced["training"]
        self.assertEqual(
            experiment._stage_num_epochs(reduced_training, experiment.PARENT_STAGE),
            15,
        )
        self.assertEqual(
            experiment._stage_num_epochs(
                reduced_training, experiment.CONTINUATION_STAGE
            ),
            100,
        )
        self.assertEqual(
            experiment._stage_num_epochs(
                reduced_training,
                experiment.BRANCH_STAGE,
                recipe={"num_epochs": 3},
            ),
            3,
        )
        self.assertEqual(
            experiment._stage_num_epochs(reduced_training, experiment.BENCHMARK_STAGE),
            1,
        )
        self.assertEqual(
            experiment._stage_num_epochs(
                reduced_training,
                experiment.PARENT_STAGE,
                override=2,
            ),
            2,
        )
        self.assertEqual(
            experiment._stage_early_stopping_min_epochs(
                reduced_training, experiment.PARENT_STAGE
            ),
            15,
        )
        self.assertEqual(
            experiment._stage_early_stopping_min_epochs(
                reduced_training, experiment.CONTINUATION_STAGE
            ),
            30,
        )

    def test_parent_epoch_cap_validation_is_fail_closed(self) -> None:
        source = experiment.load_config(
            "configs/rq123/news_first_vol_film_nolp_legacy_10seed_parent15.yaml"
        )
        for field, value, message in (
            ("parent_num_epochs", 0, "parent_num_epochs"),
            ("parent_num_epochs", 101, "parent_num_epochs"),
            (
                "parent_early_stopping_min_epochs",
                16,
                "parent_early_stopping_min_epochs",
            ),
        ):
            candidate = {
                **source,
                "training": {**source["training"], field: value},
            }
            with self.assertRaisesRegex(ValueError, message):
                experiment.validate_config(candidate)

    def test_v2_config_rejects_v1_root_before_any_control_write(self) -> None:
        v1_root = experiment.resolve_path(
            "outputs/experiments/"
            "rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v1"
        )
        v1_control = v1_root.with_name(v1_root.name + "_control")
        watched = [
            Path("configs/rq123/news_first_vol_film_nolp_legacy_10seed.yaml"),
            v1_root / "registry/jobs.json",
            v1_control / "benchmark_result.json",
            v1_control / "pipeline.log",
        ]

        def snapshot(path: Path) -> tuple[bool, int, int, str]:
            if not path.is_file():
                return (False, 0, 0, "")
            return (
                True,
                path.stat().st_mtime_ns,
                path.stat().st_size,
                experiment.sha256_file(path),
            )

        before = {str(path): snapshot(path) for path in watched}
        with self.assertRaisesRegex(ValueError, "Output root differs"):
            experiment.run_benchmark(
                experiment.DEFAULT_CONFIG,
                v1_root,
                resume=True,
            )
        after = {str(path): snapshot(path) for path in watched}
        self.assertEqual(after, before)

    def test_trained_explicit_grid_checkpoint_loads_with_architecture_lineage(
        self,
    ) -> None:
        from wgan_option.config import Config
        from wgan_option.models.gan_model import WGAN_GP
        from wgan_option.utils.inference_helpers import load_vol_generator

        resolved = experiment.load_config(experiment.DEFAULT_CONFIG)
        architecture_sha = experiment.architecture_profile_contract(resolved)[
            "architecture_profile_sha256"
        ]
        model_sha = experiment.model_contract(resolved)["model_contract_sha256"]
        grid = experiment.grid_contract(resolved)
        model_values = resolved["model"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                cuda=False,
                channels=1,
                embedding_dim=1024,
                noise_dim=32,
                generator_noise_mode="gaussian",
                generator_current_input_mode="current_support_masked",
                generator_conditioning_mode=experiment.GENERATOR_MODE,
                critic_conditioning_mode=experiment.CRITIC_MODE,
                critic_normalization_mode="legacy_instance_norm_v1",
                residual_output_mode="identity_softplus_residual",
                support_mask_mode="raw_joint",
                gen_base_channels=int(model_values["gen_base_channels"]),
                gen_res_blocks=int(model_values["gen_res_blocks"]),
                gen_text_hidden_dim=int(model_values["gen_text_hidden_dim"]),
                gen_text_out_dim=int(model_values["gen_text_out_dim"]),
                gen_hidden_dim=int(model_values["gen_hidden_dim"]),
                disc_base_channels=int(model_values["disc_base_channels"]),
                disc_res_blocks=int(model_values["disc_res_blocks"]),
                disc_text_hidden_dim=int(model_values["disc_text_hidden_dim"]),
                disc_hidden_dim=int(model_values["disc_hidden_dim"]),
                discriminator_iter=1,
                use_calendar_constraint=False,
                use_butterfly_constraint=False,
                use_smooth_constraint=False,
                news_first_capacity_profile="legacy",
                news_first_capacity_profile_sha256=architecture_sha,
                news_first_architecture_profile_sha256=architecture_sha,
                news_first_model_contract_sha256=model_sha,
                news_first_surface_grid_profile=resolved["data"][
                    "surface_grid_profile"
                ],
                news_first_surface_grid_sha256=grid["grid_contract_sha256"],
                models_path=str(root / "checkpoints"),
                metrics_path=str(root / "metrics"),
            )
            model = WGAN_GP(
                config=config,
                strike_grid=np.asarray(grid["strike_grid"], dtype=np.float32),
                maturity_grid_days=np.asarray(
                    grid["maturity_days_grid"], dtype=np.float32
                ),
                embedding_dim=1024,
            )
            before = {
                key: value.detach().clone()
                for key, value in model.G.state_dict().items()
            }
            current = torch.full((1, 1, 16, 16), 0.2)
            text = torch.zeros((1, 1024))
            target = current + 0.01
            weights = torch.ones(1)
            support = torch.ones_like(current)
            metrics = model._generator_step(
                current,
                text,
                target,
                sample_weight=weights,
                support_mask=support,
                current_support_mask=support,
            )
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(
                any(
                    not torch.equal(before[key], value)
                    for key, value in model.G.state_dict().items()
                )
            )
            model._init_metrics_file()
            epoch_stats = {
                "val_recon": 0.1,
                "val_current_recon": 0.1,
                "val_baseline_gap": 0.0,
                "val_hybrid_score": 0.1,
            }
            model._save_best_learned_validation_checkpoint(
                epoch=1,
                monitor_metric="val_hybrid_score",
                current_metric=0.1,
                epoch_stats=epoch_stats,
            )
            checkpoint_path = root / "checkpoints/generator_best_learned.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(
                checkpoint["architecture_profile_sha256"], architecture_sha
            )
            self.assertEqual(checkpoint["model_contract_sha256"], model_sha)
            sample = SimpleNamespace(
                current_surface=np.zeros((1, 16, 16), dtype=np.float32),
                strike_grid=np.asarray(grid["strike_grid"], dtype=np.float32),
                maturity_grid_days=np.asarray(
                    grid["maturity_days_grid"], dtype=np.float32
                ),
            )
            loaded, loaded_config, embedding_dim = load_vol_generator(
                checkpoint_path, sample, torch.device("cpu")
            )
            self.assertEqual(embedding_dim, 1024)
            self.assertEqual(
                loaded_config.news_first_architecture_profile_sha256,
                architecture_sha,
            )
            with torch.no_grad():
                output = loaded(
                    current,
                    text,
                    noise=torch.zeros((1, 32)),
                    current_support_mask=support,
                )
            self.assertTrue(bool(torch.isfinite(output).all()))

    def test_benchmark_is_mixed_and_uses_only_valid_horizon_arms(self) -> None:
        specs = experiment.experiment_specs(experiment.BENCHMARK_STAGE)
        self.assertEqual(len(specs), 36)
        for gpu in (0, 1):
            selected = [row for row in specs if row["gpu_id"] == gpu]
            self.assertEqual(len(selected), 18)
            self.assertEqual(
                Counter(row["tolerance_minutes"] for row in selected), {5: 9, 30: 9}
            )
            observed = {
                tolerance: {
                    row["arm"]
                    for row in selected
                    if row["tolerance_minutes"] == tolerance
                }
                for tolerance in (5, 30)
            }
            self.assertEqual(
                observed[5],
                {
                    experiment.PARENT_ARM,
                    experiment.CONTINUATION_ARM,
                    *experiment.TEXT_ARMS_5M,
                },
            )
            self.assertEqual(
                observed[30],
                {
                    experiment.PARENT_ARM,
                    experiment.CONTINUATION_ARM,
                    *experiment.TEXT_ARMS_30M,
                },
            )

    def test_matrix_smoke_is_exact_one_to_one_formal_matrix(self) -> None:
        formal = experiment.planned_specs()
        smoke = experiment.matrix_smoke_specs()

        self.assertEqual(len(smoke), 400)
        self.assertTrue(
            all(row["stage"] == experiment.MATRIX_SMOKE_STAGE for row in smoke)
        )
        self.assertEqual(
            {
                (
                    row["formal_stage"],
                    row["tolerance_minutes"],
                    row["fold"],
                    row["seed"],
                    row["arm"],
                    row["gpu_id"],
                )
                for row in smoke
            },
            {
                (
                    row["stage"],
                    row["tolerance_minutes"],
                    row["fold"],
                    row["seed"],
                    row["arm"],
                    row["gpu_id"],
                )
                for row in formal
            },
        )
        experiment.validate_gpu_balance(smoke)

    def test_metrics_allow_epoch_zero_train_nans_but_require_learned_finite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_metrics.csv"
            frame = pd.DataFrame(
                {
                    "epoch": [0, 1],
                    "d_total": [float("nan"), 1.0],
                    "g_total": [float("nan"), 2.0],
                    "g_recon": [float("nan"), 0.1],
                    "gp": [float("nan"), 0.2],
                    "val_recon": [0.3, 0.25],
                    "g_lr": [5e-7, 5e-7],
                    "d_lr": [5e-7, 5e-7],
                }
            )
            frame.to_csv(path, index=False)
            result = experiment._validate_training_metrics(
                path, require_exactly_one_learned_epoch=True
            )
            self.assertEqual(result["maximum_epoch"], 1)
            frame.loc[1, "g_total"] = float("inf")
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Learned metric"):
                experiment._validate_training_metrics(
                    path, require_exactly_one_learned_epoch=True
                )

    def test_both_alignment_tolerances_keep_five_minute_forecast_horizon(self) -> None:
        valid = pd.DataFrame(
            {
                "current_snapshot_time_utc": ["2023-01-03T14:00:00Z"],
                "target_snapshot_time_utc": ["2023-01-03T14:05:00Z"],
            }
        )
        for tolerance in (5, 30):
            experiment._validate_five_minute_forecast_horizon(
                valid, tolerance_minutes=tolerance
            )
        invalid = valid.assign(target_snapshot_time_utc="2023-01-03T14:30:00Z")
        with self.assertRaisesRegex(ValueError, "forecast exactly 5 minutes"):
            experiment._validate_five_minute_forecast_horizon(
                invalid, tolerance_minutes=30
            )

    def test_checkpoint_roles_follow_dynamic_vs_frozen_protocol(self) -> None:
        self.assertEqual(
            experiment._selected_checkpoint_roles({"stage": experiment.PARENT_STAGE}),
            ("generator_best_learned", "discriminator_best_learned"),
        )
        self.assertEqual(
            experiment._selected_checkpoint_roles(
                {"stage": experiment.CONTINUATION_STAGE}
            ),
            ("generator_best_learned", "discriminator_best_learned"),
        )
        self.assertEqual(
            experiment._selected_checkpoint_roles({"stage": experiment.BRANCH_STAGE}),
            ("generator_final", "discriminator_final"),
        )

    def test_selected_only_checkpoint_pruning_keeps_best_learned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "job" / "timestamp"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            retained = {
                "generator_best_learned.pt",
                "discriminator_best_learned.pt",
            }
            removed = {
                "generator_initial_epoch0.pt",
                "discriminator_initial_epoch0.pt",
                "generator_best.pt",
                "discriminator_best.pt",
                "generator.pt",
                "discriminator.pt",
                "generator_epoch_0001.pt",
                "discriminator_epoch_0001.pt",
            }
            for name in retained | removed:
                (checkpoint_dir / name).write_bytes(name.encode("ascii"))
            experiment._prune_unselected_training_checkpoints(
                {"stage": experiment.PARENT_STAGE}, run_dir
            )
            self.assertEqual({path.name for path in checkpoint_dir.iterdir()}, retained)

    def test_selected_retention_artifact_manifest_stays_resume_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_root = Path(directory) / "parent_run"
            run_dir = job_root / "20260822_010203"
            checkpoint_dir = run_dir / "checkpoints"
            metrics_dir = run_dir / "metrics"
            checkpoint_dir.mkdir(parents=True)
            metrics_dir.mkdir(parents=True)
            for name in (
                "generator_best_learned.pt",
                "discriminator_best_learned.pt",
                "generator_initial_epoch0.pt",
                "discriminator_initial_epoch0.pt",
                "generator.pt",
                "discriminator.pt",
            ):
                (checkpoint_dir / name).write_bytes(name.encode("ascii"))
            for name in (
                "best_learned_checkpoint.json",
                "training_metrics.csv",
                "training_metrics.json",
                "training_resolved_config.yaml",
            ):
                (metrics_dir / name).write_text(name, encoding="utf-8")
            (run_dir / "run.log").write_text("ok", encoding="utf-8")
            overlay = Path(directory) / "overlay.json"
            overlay.write_text("{}", encoding="utf-8")
            state = checkpoint_dir / "full_training_state_best_learned.pt"
            state.write_bytes(b"state")
            contract = Path(directory) / "contract.json"
            experiment.write_json(contract, {"output_path": str(state)})
            job = {
                "stage": experiment.PARENT_STAGE,
                "job_id": "parent_fixture",
                "run_dir": str(job_root),
                "job_spec_sha256": "job-sha",
                "training_config_sha256": "config-sha",
                "overlay_path": str(overlay),
                "full_state_contract_path": str(contract),
            }
            artifacts = experiment._artifact_paths_for_job(job, run_dir)
            experiment._prune_unselected_training_checkpoints(job, run_dir)
            status = {
                "status": "completed",
                "job_id": "parent_fixture",
                "stage": experiment.PARENT_STAGE,
                "job_spec_sha256": "job-sha",
                "training_config_sha256": "config-sha",
                "run_dir": str(run_dir),
                "artifacts": artifacts,
            }
            self.assertTrue(experiment._completed_valid(job, status))
            self.assertFalse((checkpoint_dir / "generator.pt").exists())
            self.assertTrue((checkpoint_dir / "generator_best_learned.pt").is_file())

            branch_dir = Path(directory) / "branch" / "checkpoints"
            branch_dir.mkdir(parents=True)
            branch_final = branch_dir / "generator.pt"
            branch_final.write_bytes(b"branch")
            branch_best = branch_dir / "generator_best_learned.pt"
            branch_best.write_bytes(b"unselected")
            experiment._prune_unselected_training_checkpoints(
                {"stage": experiment.BRANCH_STAGE}, branch_dir.parent
            )
            self.assertTrue(branch_final.is_file())
            self.assertFalse(branch_best.exists())

            benchmark_dir = Path(directory) / "benchmark" / "checkpoints"
            benchmark_dir.mkdir(parents=True)
            initial = benchmark_dir / "generator_initial_epoch0.pt"
            final = benchmark_dir / "generator.pt"
            initial.write_bytes(b"initial")
            final.write_bytes(b"final")
            experiment._prune_unselected_training_checkpoints(
                {"stage": experiment.BENCHMARK_STAGE}, benchmark_dir.parent
            )
            self.assertTrue(initial.is_file())
            self.assertTrue(final.is_file())

    def test_state_update_check_detects_real_parameter_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / "initial.pt"
            final = Path(directory) / "final.pt"
            torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, initial)
            torch.save({"state_dict": {"weight": torch.tensor([2.0])}}, final)
            self.assertTrue(experiment._state_dicts_differ(initial, final))
            torch.save({"state_dict": {"weight": torch.tensor([1.0])}}, final)
            self.assertFalse(experiment._state_dicts_differ(initial, final))
            self.assertTrue(experiment._state_dicts_equal(initial, final))
            torch.save({"state_dict": {"weight": torch.tensor([3.0])}}, final)
            self.assertFalse(experiment._state_dicts_equal(initial, final))

    def test_test_materialization_is_strictly_after_checkpoint_freeze(self) -> None:
        order: list[str] = []
        with (
            mock.patch.object(
                experiment,
                "validate_root",
                return_value={"experiment": {}},
            ),
            mock.patch.object(
                experiment,
                "_freeze_checkpoint_allowlist",
                side_effect=lambda root: order.append("checkpoint") or root,
            ),
            mock.patch.object(
                experiment,
                "_materialize_test_inputs",
                side_effect=lambda config, root: order.append("test") or root,
            ),
        ):
            experiment.freeze_evaluation("/tmp/not-created-by-test")
        self.assertEqual(order, ["checkpoint", "test"])

    def test_deterministic_gzip_writer_has_stable_bytes(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv.gz"
            second = Path(directory) / "second.csv.gz"
            experiment._write_dataframe_csv(first, frame, gzip=True)
            experiment._write_dataframe_csv(second, frame, gzip=True)
            self.assertEqual(
                experiment.sha256_file(first), experiment.sha256_file(second)
            )
            pd.testing.assert_frame_equal(pd.read_csv(first), frame)

    def test_missing_test_text_never_becomes_a_nan_bow_token(self) -> None:
        self.assertEqual(experiment._canonical_article_text(float("nan")), "")
        vector = experiment._bow_vector_from_article_texts(
            [float("nan"), None, ""], ["nan", "real"]
        )
        self.assertEqual(float(vector.sum()), 0.0)

    def test_completed_pipeline_retry_uses_locked_read_only_postprocess(self) -> None:
        root = Path("/tmp/completed-rq123-fixture")
        lock = Path("/tmp/completed-rq123-fixture.lock")
        with (
            mock.patch.object(
                experiment,
                "load_config",
                return_value={"experiment": {"output_root": str(root)}},
            ),
            mock.patch.object(
                experiment, "read_registry", return_value={"status": "completed"}
            ),
            mock.patch.object(experiment, "postprocess") as postprocess,
            mock.patch.object(experiment, "_pipeline_lock", return_value=lock),
            mock.patch.object(experiment, "_release_pipeline_lock") as release,
        ):
            with mock.patch.object(Path, "is_dir", return_value=True):
                result = experiment.run_pipeline("unused.yaml", root, resume=True)
        self.assertEqual(result, root.resolve())
        postprocess.assert_called_once_with(root.resolve(), resume=True)
        release.assert_called_once_with(lock)

    def test_pipeline_flock_is_cross_process_exclusive_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            acquired = experiment._pipeline_lock(root)
            code = (
                "from pathlib import Path; "
                "from scripts.rq123 import news_first_vol_film_nolp_10seed as e; "
                f"r=Path({str(root)!r}); "
                "\ntry:\n e._pipeline_lock(r)\nexcept RuntimeError:\n print('blocked')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=experiment.REPO_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{experiment.REPO_ROOT / 'src'}:{experiment.REPO_ROOT}",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "blocked")
            with self.assertRaisesRegex(RuntimeError, "Live pipeline supervisor"):
                experiment._pipeline_lock(root)
            experiment._release_pipeline_lock(acquired)
            self.assertTrue(acquired.is_file())
            reacquired = experiment._pipeline_lock(root)
            self.assertEqual(
                int(
                    (root.with_name(root.name + "_control") / "pipeline.pid").read_text(
                        encoding="utf-8"
                    )
                ),
                os.getpid(),
            )
            experiment._release_pipeline_lock(reacquired)

    def test_job_flock_allows_only_one_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, descriptor = experiment._acquire_job_lock(root, "job")
            code = (
                "from pathlib import Path; "
                "from scripts.rq123 import news_first_vol_film_nolp_10seed as e; "
                f"r=Path({str(root)!r}); "
                "\ntry:\n e._acquire_job_lock(r, 'job')\n"
                "except RuntimeError:\n print('blocked')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=experiment.REPO_ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": (
                        f"{experiment.REPO_ROOT / 'src'}:{experiment.REPO_ROOT}"
                    ),
                },
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "blocked")
            experiment._release_file_lock(lock, descriptor)
            self.assertTrue(lock.is_file())
            second, second_descriptor = experiment._acquire_job_lock(root, "job")
            experiment._release_file_lock(second, second_descriptor)

    def test_pipeline_finally_releases_lock_but_keeps_pid_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            benchmark = Path(directory) / "benchmark_result.json"
            benchmark.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(
                    experiment,
                    "load_config",
                    return_value={"experiment": {"output_root": str(root)}},
                ),
                mock.patch.object(
                    experiment, "_benchmark_result_path", return_value=benchmark
                ),
                mock.patch.object(experiment, "run_recovery_canary"),
                mock.patch.object(
                    experiment,
                    "run_matrix_smoke",
                    side_effect=RuntimeError("smoke failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "smoke failed"):
                    experiment.run_pipeline("unused.yaml", root, resume=True)
            control = root.with_name(root.name + "_control")
            self.assertTrue((control / "pipeline.lock").is_file())
            self.assertTrue((control / "pipeline.pid").is_file())

    def test_pipeline_runs_matrix_smoke_before_formal_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            benchmark = Path(directory) / "benchmark_result.json"
            benchmark.write_text("{}", encoding="utf-8")
            lock = Path(directory) / "pipeline.lock"
            lock.write_text("fixture", encoding="utf-8")
            order: list[str] = []
            frozen_registry = {
                "status": "prepared",
                "parent_states_frozen": True,
                "branch_recipes_frozen": True,
                "evaluation_frozen": True,
                "predictions_frozen": True,
                "rq1_evaluated": True,
                "rq2_evaluated": True,
                "rq3_evaluated": True,
            }
            with (
                mock.patch.object(
                    experiment,
                    "load_config",
                    return_value={"experiment": {"output_root": str(root)}},
                ),
                mock.patch.object(experiment, "_pipeline_lock", return_value=lock),
                mock.patch.object(
                    experiment, "_benchmark_result_path", return_value=benchmark
                ),
                mock.patch.object(
                    experiment,
                    "run_recovery_canary",
                    side_effect=lambda *args, **kwargs: order.append("canary"),
                ),
                mock.patch.object(
                    experiment,
                    "run_matrix_smoke",
                    side_effect=lambda *args, **kwargs: order.append("smoke"),
                ),
                mock.patch.object(
                    experiment,
                    "prepare_experiment",
                    side_effect=lambda *args, **kwargs: order.append("prepare"),
                ),
                mock.patch.object(
                    experiment, "read_registry", return_value=frozen_registry
                ),
                mock.patch.object(experiment, "postprocess"),
            ):
                experiment.run_pipeline("unused.yaml", root, resume=True)
            self.assertEqual(order, ["canary", "smoke", "prepare"])
            self.assertTrue(lock.exists())

    def test_public_dry_run_before_prepare_executes_full_matrix_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "formal"
            with (
                mock.patch.object(
                    experiment,
                    "run_recovery_canary",
                    return_value=Path("canary.json"),
                ) as canary,
                mock.patch.object(
                    experiment, "run_matrix_smoke", return_value=Path("smoke.json")
                ) as smoke,
                mock.patch.object(
                    experiment,
                    "dry_run",
                    side_effect=AssertionError("formal dry-run is not prepared"),
                ),
            ):
                result = experiment.run_action(
                    "dry-run",
                    config_path="config.yaml",
                    output_dir=root,
                    resume=True,
                )
            self.assertEqual(result, Path("smoke.json"))
            canary.assert_called_once_with("config.yaml", root, resume=True)
            smoke.assert_called_once_with("config.yaml", root, resume=True)

    def test_public_action_set_keeps_matrix_smoke_behind_dry_run(self) -> None:
        parser = experiment._parser()
        action = next(item for item in parser._actions if item.dest == "action")
        self.assertNotIn("matrix-smoke", action.choices)
        self.assertIn("recovery-canary", action.choices)
        self.assertIn("dry-run", action.choices)
        with self.assertRaisesRegex(ValueError, "Unsupported action"):
            experiment.run_action(
                "matrix-smoke",
                config_path="unused.yaml",
                output_dir="unused",
            )

    def test_job_contract_rejects_config_contract_and_status_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            data_root = Path(directory) / "data"
            tolerance = 5
            dataset = data_root / "surface_05.xlsx"
            support = data_root / "tolerance_05m/surface_support_audit.csv.gz"
            dataset.parent.mkdir(parents=True)
            support.parent.mkdir(parents=True)
            dataset.write_bytes(b"dataset")
            support.write_bytes(b"support")
            spec = {
                "stage": experiment.PARENT_STAGE,
                "arm": experiment.PARENT_ARM,
                "tolerance_minutes": tolerance,
                "fold": experiment.FOLDS[0],
                "seed": experiment.SEEDS[0],
            }
            job_id = "fixture_job"
            config_path = root / f"configs/jobs/{job_id}.yaml"
            contract_path = root / f"configs/full_state_contracts/{job_id}.json"
            overlay = experiment._overlay_manifest_path(root, spec)
            loaded = experiment.load_config(experiment.DEFAULT_CONFIG)
            config = {
                **loaded,
                "data": {
                    **loaded["data"],
                    "root": str(data_root),
                    "workbook_template": "surface_{tolerance02}.xlsx",
                },
            }
            architecture_sha = experiment.architecture_profile_contract(config)[
                "architecture_profile_sha256"
            ]
            model_sha = experiment.model_contract(config)["model_contract_sha256"]
            grid_sha = experiment.grid_contract(config)["grid_contract_sha256"]
            training_payload = {
                "value": 1,
                "news_first_capacity_profile_sha256": architecture_sha,
                "news_first_architecture_profile_sha256": architecture_sha,
                "news_first_model_contract_sha256": model_sha,
                "news_first_surface_grid_sha256": grid_sha,
            }
            experiment.write_yaml(config_path, training_payload)
            experiment.write_json(contract_path, {"value": 2})
            experiment.write_json(overlay, {"value": 3})
            job = {
                **spec,
                "job_id": job_id,
                "architecture_profile_sha256": architecture_sha,
                "model_contract_sha256": model_sha,
                "grid_contract_sha256": grid_sha,
                "training_config_path": str(config_path.resolve()),
                "training_config_sha256": experiment.sha256_file(config_path),
                "full_state_contract_path": str(contract_path.resolve()),
                "full_state_contract_sha256": experiment.sha256_file(contract_path),
                "overlay_path": str(overlay.resolve()),
                "overlay_sha256": experiment.sha256_file(overlay),
                "dataset_path": str(dataset.resolve()),
                "dataset_sha256": experiment.sha256_file(dataset),
                "support_path": str(support.resolve()),
                "support_sha256": experiment.sha256_file(support),
                "run_dir": str(experiment._run_directory(root, spec)),
            }
            job["job_spec_sha256"] = experiment._job_spec_sha(job)
            experiment.write_json(
                experiment._status_path(root, job_id),
                experiment.initial_job_status(job),
            )
            experiment._validate_job_contract(root, config, job)
            config_path.write_text("value: tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training config drift"):
                experiment._validate_job_contract(root, config, job)
            experiment.write_yaml(config_path, training_payload)
            contract_path.write_text('{"value": "tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full-state contract drift"):
                experiment._validate_job_contract(root, config, job)
            experiment.write_json(contract_path, {"value": 2})
            status = experiment.read_json(experiment._status_path(root, job_id))
            status["stage"] = "tampered"
            experiment.write_json(experiment._status_path(root, job_id), status)
            with self.assertRaisesRegex(ValueError, "status lineage"):
                experiment._validate_job_contract(root, config, job)

    def test_full_state_deep_validation_rejects_best_epoch_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            lineage = {"job": "fixture"}
            experiment.write_json(contract, {"output_lineage": lineage})
            state_path = root / "state.pt"
            payload = {
                "schema_version": 1,
                "kind": experiment.FULL_STATE_KIND,
                "save_phase": experiment.FULL_STATE_PHASE,
                "completed_epoch": 3,
                "contract_sha256": experiment.sha256_file(contract),
                "lineage": lineage,
                "generator_state_dict": {},
                "discriminator_state_dict": {},
                "generator_optimizer_state_dict": {},
                "discriminator_optimizer_state_dict": {},
                "generator_scheduler_state_dict": {},
                "discriminator_scheduler_state_dict": {},
                "rng_state": {
                    "python": (),
                    "numpy": (),
                    "torch_cpu": torch.tensor([], dtype=torch.uint8),
                    "torch_cuda": [],
                    "loader_generator": torch.tensor([], dtype=torch.uint8),
                },
            }
            torch.save(payload, state_path)
            best = root / "best.json"
            experiment.write_json(best, {"best_epoch": 3})
            job = {
                "job_id": "fixture",
                "full_state_contract_path": str(contract),
                "full_state_contract_sha256": experiment.sha256_file(contract),
            }
            status = {
                "artifacts": [
                    experiment.manifest_row("full_training_state", state_path),
                    experiment.manifest_row("best_learned_checkpoint", best),
                ]
            }
            experiment._validated_full_state_payload(job, status)
            experiment.write_json(best, {"best_epoch": 2})
            status["artifacts"][1] = experiment.manifest_row(
                "best_learned_checkpoint", best
            )
            with self.assertRaisesRegex(ValueError, "contract/payload drift"):
                experiment._validated_full_state_payload(job, status)

    def test_selected_attempt_cleanup_preserves_state_and_selected_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_root = root / "job"
            (job_root / "checkpoints").mkdir(parents=True)
            (job_root / "checkpoints/state.pt").write_bytes(b"state")
            job = {
                "job_id": "fixture",
                "job_spec_sha256": "job-sha",
                "training_config_sha256": "config-sha",
                "run_dir": str(job_root),
            }
            old = job_root / "20260822_010101"
            selected = job_root / "20260822_020202"
            experiment._write_attempt_ownership(job, old)
            experiment._write_attempt_ownership(job, selected)
            experiment._cleanup_unselected_job_attempts(root, job, keep=selected)
            self.assertFalse(old.exists())
            self.assertTrue(selected.is_dir())
            self.assertTrue((job_root / "checkpoints/state.pt").is_file())

    def test_matrix_smoke_cleanup_rejects_foreign_same_name_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            formal = Path(directory) / "formal"
            foreign = experiment._matrix_smoke_root(formal, 12)
            foreign.mkdir()
            (foreign / "foreign.txt").write_text("do not delete", encoding="utf-8")
            with self.assertRaises((FileNotFoundError, RuntimeError, ValueError)):
                experiment._cleanup_matrix_smoke_root(formal, 12)
            self.assertTrue((foreign / "foreign.txt").is_file())

    def test_event_freeze_recovers_either_half_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            sources = []
            for index in range(2):
                path = Path(directory) / f"events_{index}.csv"
                offset = index * 84
                pd.DataFrame(
                    {
                        "event_id": [f"e{offset + row}" for row in range(84)],
                        "release_time_utc": pd.date_range(
                            "2023-01-01", periods=84, freq="h", tz="UTC"
                        ).astype(str),
                    }
                ).to_csv(path, index=False)
                sources.append(str(path))
            market_paths = []
            for name in ("pairs", "episodes", "summary"):
                path = Path(directory) / f"{name}.csv"
                path.write_text("value\n1\n", encoding="utf-8")
                market_paths.append(path)
            config = {
                "data": {
                    "scheduled_event_paths": sources,
                    "market_jump_candidate_pairs": str(market_paths[0]),
                    "market_jump_candidate_episodes": str(market_paths[1]),
                    "market_jump_validation_summary": str(market_paths[2]),
                }
            }
            manifest = experiment._materialize_frozen_event_sources(config, root)
            destination = root / "evaluation/frozen_scheduled_events.csv"
            frozen_sha = experiment.sha256_file(destination)
            manifest.unlink()
            experiment._materialize_frozen_event_sources(config, root)
            self.assertEqual(experiment.sha256_file(destination), frozen_sha)
            destination.unlink()
            experiment._materialize_frozen_event_sources(config, root)
            self.assertEqual(experiment.sha256_file(destination), frozen_sha)
            manifest.unlink()
            destination.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not reproducible"):
                experiment._materialize_frozen_event_sources(config, root)

    def test_frozen_test_manifest_recovers_missing_registry_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            roles = {
                *(
                    f"test_panel:{tolerance:02d}m:{fold}"
                    for tolerance in experiment.TOLERANCES
                    for fold in experiment.FOLDS
                ),
                *(
                    f"test_overlay:{tolerance:02d}m:{fold}:{arm}"
                    for tolerance in experiment.TOLERANCES
                    for fold in experiment.FOLDS
                    for arm in (
                        experiment.PARENT_ARM,
                        experiment.CONTINUATION_ARM,
                        *experiment._arms_for_tolerance(tolerance),
                    )
                ),
                "frozen_event_manifest",
                "frozen_scheduled_events",
            }
            for index, role in enumerate(sorted(roles)):
                artifact = root / "fixtures" / f"artifact_{index:02d}"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(role, encoding="utf-8")
                rows.append(experiment.manifest_row(role, artifact))
            manifest = experiment._manifest_csv(
                experiment._test_input_manifest_path(root), rows
            )
            experiment.write_registry(
                root,
                {
                    "experiment_kind": experiment.EXPERIMENT_KIND,
                    "jobs": [],
                    "evaluation_frozen": True,
                    "test_data_opened": False,
                },
            )
            with mock.patch.object(
                experiment, "_validate_test_inputs", return_value=manifest
            ):
                result = experiment._materialize_test_inputs({}, root)
            self.assertEqual(result, manifest)
            registry = experiment.read_registry(root)
            self.assertTrue(registry["test_data_opened"])
            self.assertEqual(
                registry["test_input_manifest_sha256"],
                experiment.sha256_file(manifest),
            )

    def test_prediction_validation_requires_per_cell_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {
                "job_id": "fixture",
                "tolerance_minutes": 5,
                "fold": experiment.FOLDS[0],
                "seed": experiment.SEEDS[0],
                "arm": "lp_matched",
            }
            prediction = experiment._prediction_path(root, job)
            prediction.parent.mkdir(parents=True)
            prediction.write_bytes(b"prediction")
            manifest = experiment._prediction_job_manifest_path(root, job)
            experiment.write_json(manifest, {"payload_sha256": "fixture"})
            with self.assertRaisesRegex(ValueError, "Partial prediction artifact"):
                experiment._validate_prediction_job(
                    root,
                    job,
                    {"path": str(root / "checkpoint.pt"), "sha256": "sha"},
                    expected_pairs=1,
                )

    def test_final_snapshot_binds_live_experiment_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = {
                "experiment_kind": experiment.EXPERIMENT_KIND,
                "jobs": [],
                "status": "postprocessing",
                "terminal_complete": False,
                "postprocessing_started_at_utc": "2026-08-22T00:00:00Z",
            }
            experiment.write_registry(root, live, updated_at_utc="2026-08-22T00:00:01Z")
            experiment.write_json(
                experiment._experiment_status_path(root),
                {
                    "status": "postprocessing",
                    "updated_at_utc": "2026-08-22T00:00:01Z",
                    "current_stage": "terminal",
                },
            )
            intended_registry = {
                **experiment.read_registry(root),
                "status": "completed",
                "terminal_complete": True,
                "completed_at_utc": "2026-08-22T00:00:02Z",
                "updated_at_utc": "2026-08-22T00:00:02Z",
            }
            intended_status = {
                "status": "completed",
                "updated_at_utc": "2026-08-22T00:00:02Z",
                "current_stage": "terminal",
            }
            snapshot = {
                "schema_version": 1,
                "experiment_kind": experiment.EXPERIMENT_KIND,
                "registry": intended_registry,
                "experiment_status": intended_status,
                "created_at_utc": "2026-08-22T00:00:02Z",
            }
            snapshot["payload_sha256"] = experiment.payload_sha256(snapshot)
            experiment.write_json(
                root / "registry/final_registry_snapshot.json", snapshot
            )
            experiment._validate_final_snapshot(root, allow_postprocessing=True)
            experiment.write_json(
                experiment._experiment_status_path(root), intended_status
            )
            experiment.write_registry(
                root,
                intended_registry,
                updated_at_utc=intended_registry["updated_at_utc"],
            )
            experiment._validate_final_snapshot(root)
            tampered = {**intended_status, "current_stage": "tampered"}
            experiment.write_json(experiment._experiment_status_path(root), tampered)
            with self.assertRaisesRegex(ValueError, "status snapshot"):
                experiment._validate_final_snapshot(root)

    def test_noise_bank_profile_is_shared_across_arms_but_not_seeds(self) -> None:
        samples = ["pair::b", "pair::a"]
        base = {
            "tolerance_minutes": 5,
            "fold": experiment.FOLDS[0],
            "seed": experiment.SEEDS[0],
            "arm": "lp_matched",
        }
        matched = experiment._noise_bank_profile_sha256(base, samples)
        shuffled = experiment._noise_bank_profile_sha256(
            {**base, "arm": "lp_shuffle"}, list(reversed(samples))
        )
        another_seed = experiment._noise_bank_profile_sha256(
            {**base, "seed": experiment.SEEDS[1]}, samples
        )
        self.assertEqual(matched, shuffled)
        self.assertNotEqual(matched, another_seed)

    def test_inference_determinism_contract_and_seed_are_replayable(self) -> None:
        first_contract = experiment._inference_determinism_contract()
        second_contract = experiment._inference_determinism_contract()
        self.assertEqual(first_contract, second_contract)
        self.assertEqual(first_contract["prediction_mc_samples"], 64)
        self.assertEqual(first_contract["noise_dim"], 32)
        experiment._configure_prediction_determinism(42)
        first = torch.randn(4)
        experiment._configure_prediction_determinism(42)
        second = torch.randn(4)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_run_wave_terminates_live_workers_on_interrupt(self) -> None:
        class FakeProcess:
            pid = 999999999

            def poll(self):
                return None

        class FakeMonitor:
            def start(self):
                return None

            def stop(self):
                return None

        job = {"job_id": "j", "gpu_id": 0}
        config = {
            "source_config_path": "fixture.yaml",
            "runtime": {
                "python_executable": sys.executable,
                "nvidia_smi_executable": "nvidia-smi",
                "resource_sample_interval_seconds": 1,
                "cpu_threads_per_job": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            experiment.write_json(experiment._status_path(root, "j"), {"attempt": 0})
            attestation = root / "attestation.json"
            experiment.write_json(attestation, {"payload_sha256": "sha"})
            with (
                mock.patch.object(experiment, "validate_root", return_value=config),
                mock.patch.object(
                    experiment, "_write_wave_attestation", return_value=attestation
                ),
                mock.patch(
                    "scripts.rq3.news_first_vol_training._ResourceMonitor",
                    return_value=FakeMonitor(),
                ),
                mock.patch.object(
                    experiment.subprocess, "Popen", return_value=FakeProcess()
                ),
                mock.patch.object(
                    experiment.time, "sleep", side_effect=KeyboardInterrupt
                ),
                mock.patch.object(experiment, "_terminate_processes") as terminate,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    experiment._run_wave(
                        root, [job], wave=1, dry_run=False, resume=True
                    )
            terminate.assert_called_once()

    def test_postprocess_interruption_never_commits_completed_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "analysis/manifest.json"
            analysis.parent.mkdir(parents=True)
            analysis.write_text("{}", encoding="utf-8")
            registry = {
                "experiment_kind": experiment.EXPERIMENT_KIND,
                "jobs": [],
                "status": "analysis_complete",
                "terminal_complete": False,
                "rq1_evaluated": True,
                "rq2_evaluated": True,
                "rq3_evaluated": True,
            }
            experiment.write_registry(root, registry)
            experiment.write_experiment_status(root, "analysis_complete")

            def render(**kwargs):
                output = Path(kwargs["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                (output / "report_manifest.json").write_text("{}", encoding="utf-8")

            def resource(target):
                path = target / "resource_summary.csv"
                path.write_text("gpu\n0\n", encoding="utf-8")
                return path

            original_write_registry = experiment.write_registry

            def fail_final(target, payload, **kwargs):
                if payload.get("status") == "completed":
                    raise RuntimeError("fault before completed commit")
                return original_write_registry(target, payload, **kwargs)

            common = (
                mock.patch.object(experiment, "validate_root"),
                mock.patch.object(
                    experiment, "_validate_analysis_bundle", return_value=analysis
                ),
                mock.patch(
                    "scripts.rq123.news_first_vol_film_nolp_10seed_report.render_unified_report",
                    side_effect=render,
                ),
                mock.patch.object(experiment, "_validate_report_manifest"),
                mock.patch.object(
                    experiment, "_resource_summary", side_effect=resource
                ),
                mock.patch.object(
                    experiment, "qa_experiment", return_value=root / "qa.json"
                ),
                mock.patch.object(
                    experiment,
                    "_validate_terminal_output_manifest",
                    side_effect=lambda *args, **kwargs: (
                        root / "output_hashes.csv",
                        1,
                    ),
                ),
            )
            with ExitStack() as stack:
                for context in common:
                    stack.enter_context(context)
                stack.enter_context(
                    mock.patch.object(
                        experiment, "write_registry", side_effect=fail_final
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "before completed"):
                    experiment.postprocess(root, resume=True)
            self.assertEqual(experiment.read_registry(root)["status"], "postprocessing")

            with (
                mock.patch.object(experiment, "validate_root"),
                mock.patch.object(
                    experiment, "_validate_analysis_bundle", return_value=analysis
                ),
                mock.patch.object(experiment, "_validate_report_manifest"),
                mock.patch.object(
                    experiment, "qa_experiment", return_value=root / "qa.json"
                ),
                mock.patch.object(
                    experiment,
                    "_validate_terminal_output_manifest",
                    side_effect=lambda *args, **kwargs: (
                        root / "output_hashes.csv",
                        1,
                    ),
                ),
            ):
                experiment.postprocess(root, resume=True)
            self.assertEqual(experiment.read_registry(root)["status"], "completed")

    def test_validate_root_rejects_registry_job_universe_sha_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = {
                "experiment_kind": experiment.EXPERIMENT_KIND,
                "jobs": [],
                "jobs_sha256": "tampered",
            }
            with (
                mock.patch.object(experiment, "read_manifest"),
                mock.patch.object(experiment, "read_registry", return_value=registry),
            ):
                with self.assertRaisesRegex(ValueError, "job-universe SHA"):
                    experiment.validate_root(root)


if __name__ == "__main__":
    unittest.main()
