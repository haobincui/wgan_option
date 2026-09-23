"""Regression contracts for the U-Net/NoLP unified RQ1--RQ3 profile."""

from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.rq123 import news_first_vol_film_nolp_10seed as core
from scripts.rq123 import news_first_vol_film_unet_nolp_10seed as experiment
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.generator import Generator
from wgan_option.utils.news_first_experiment_core import (
    load_full_training_state,
    save_full_training_state,
)


EXPECTED_PROFILE = {
    "DEFAULT_CONFIG": experiment.DEFAULT_CONFIG,
    "DEFAULT_OUTPUT_DIR": experiment.DEFAULT_OUTPUT_DIR,
    "EXPERIMENT_KIND": experiment.EXPERIMENT_KIND,
    "GENERATOR_MODE": "film_unet_mask_coords_v1",
    "CRITIC_MODE": "lp_disabled_same_shape_v1",
    "CAPACITY_PROFILE": "c32",
    "WORKER_MODULE": "scripts.rq123.news_first_vol_film_unet_nolp_10seed",
    "SOURCE_CODE_RELATIVE_PATHS": experiment.SOURCE_CODE_RELATIVE_PATHS,
    "EXPECTED_PARAMETER_COUNTS": {
        "generator": 827_745,
        "critic": 729_157,
        "total": 1_556_902,
    },
    "EXPECTED_ARCHITECTURE_PROFILE_SHA256": (
        "2b62f513e37aab49e447c75bcae25ef7d6c8e02fc9e0fe1fea6e7628ac7f3543"
    ),
}

EXPECTED_CONFIG_SHA256 = (
    "8d4a063b55a92585187ac1d363ecdcc10e203db3d9af4369d07ffae05da36319"
)
EXPECTED_MODEL_CONTRACT_SHA256 = (
    "bea434947dd950b73953f082841c2a98cc856c36abd9d34b3b5d833eb9b40d92"
)
LEGACY_DEFAULTS = {
    "DEFAULT_CONFIG": "configs/rq123/news_first_vol_film_nolp_legacy_10seed_v2.yaml",
    "DEFAULT_OUTPUT_DIR": (
        "outputs/experiments/"
        "rq123_news_first_vol_film_nolp_legacy_10seed_exact_ttm_rolling_v2"
    ),
    "EXPERIMENT_KIND": "rq123_news_first_vol_film_nolp_legacy_10seed_rolling_v2",
    "GENERATOR_MODE": "film_conv_bottleneck_concat_v1",
    "CRITIC_MODE": "lp_disabled_same_shape_v1",
    "CAPACITY_PROFILE": "legacy",
    "WORKER_MODULE": "scripts.rq123.news_first_vol_film_nolp_10seed",
}


def _lineage() -> dict[str, object]:
    digest = hashlib.sha256(b"unet-full-state-lineage").hexdigest()
    return {
        "fold_id": "f1_2023q1",
        "seed": 42,
        "arm": "parent_current_only",
        "model_contract_sha256": digest,
        "grid_sha256": digest,
        "training_config_payload_sha256": digest,
        "code_sha256": digest,
        "dataset_sha256": digest,
        "support_sha256": digest,
        "pair_universe_sha256": digest,
        "text_manifest_sha256": digest,
        "job_sha256": digest,
    }


def _models(config: dict[str, object]) -> tuple[Generator, Discriminator]:
    model = config["model"]
    data = config["data"]
    assert isinstance(model, dict)
    assert isinstance(data, dict)
    generator = Generator(
        channels=int(model["channels"]),
        embedding_dim=int(model["embedding_dim"]),
        noise_dim=int(model["noise_dim"]),
        surface_height=16,
        surface_width=16,
        base_channels=int(model["gen_base_channels"]),
        res_blocks=int(model["gen_res_blocks"]),
        text_hidden_dim=int(model["gen_text_hidden_dim"]),
        text_out_dim=int(model["gen_text_out_dim"]),
        hidden_dim=int(model["gen_hidden_dim"]),
        residual_output_mode=str(model["residual_output_mode"]),
        generator_noise_mode=str(model["generator_noise_mode"]),
        generator_current_input_mode=str(model["generator_current_input_mode"]),
        generator_conditioning_mode=str(model["generator_conditioning_mode"]),
        strike_grid=np.asarray(data["strike_grid"], dtype=np.float32),
        maturity_grid_days=np.asarray(data["maturity_days_grid"], dtype=np.float32),
    )
    discriminator = Discriminator(
        channels=int(model["channels"]),
        embedding_dim=int(model["embedding_dim"]),
        surface_height=16,
        surface_width=16,
        base_channels=int(model["disc_base_channels"]),
        res_blocks=int(model["disc_res_blocks"]),
        text_hidden_dim=int(model["disc_text_hidden_dim"]),
        hidden_dim=int(model["disc_hidden_dim"]),
        critic_normalization_mode=str(model["critic_normalization_mode"]),
        critic_conditioning_mode=str(model["critic_conditioning_mode"]),
    )
    return generator, discriminator


def _components(config: dict[str, object]):
    generator, discriminator = _models(config)
    training = config["training"]
    assert isinstance(training, dict)
    g_optimizer = Adam(
        generator.parameters(),
        lr=float(training["generator_learning_rate"]),
        betas=(float(training["beta_1"]), float(training["beta_2"])),
    )
    d_optimizer = Adam(
        discriminator.parameters(),
        lr=float(training["discriminator_learning_rate"]),
        betas=(float(training["beta_1"]), float(training["beta_2"])),
    )
    scheduler_kwargs = {
        "mode": "min",
        "factor": float(training["reduce_lr_factor"]),
        "patience": int(training["reduce_lr_patience"]),
        "min_lr": float(training["scheduler_min_lr"]),
    }
    return (
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        ReduceLROnPlateau(g_optimizer, **scheduler_kwargs),
        ReduceLROnPlateau(d_optimizer, **scheduler_kwargs),
    )


def _parameter_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scale: float,
) -> None:
    """Populate Adam state without coupling serialization QA to WGAN losses."""

    optimizer.zero_grad(set_to_none=True)
    terms = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            first = parameter.reshape(-1)[0]
            terms.append(first * scale + first.square() * 0.01)
    torch.stack(terms).sum().backward()
    optimizer.step()


def _assert_nested_equal(test: unittest.TestCase, left: object, right: object) -> None:
    test.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            _assert_nested_equal(test, first, second)
    else:
        test.assertEqual(left, right)


class TestUnetProfileIsolation(unittest.TestCase):
    def test_profile_enters_and_restores_all_core_globals_after_exception(self) -> None:
        originals = {
            name: getattr(core, name) for name in experiment._CORE_PROFILE_OVERRIDES
        }
        original_postprocess = core.postprocess
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with experiment.unet_nolp_profile():
                for name, expected in EXPECTED_PROFILE.items():
                    self.assertEqual(getattr(core, name), expected)
                self.assertIs(core.postprocess, experiment._profile_postprocess)
                raise RuntimeError("fixture")
        for name, original in originals.items():
            self.assertIs(getattr(core, name), original)
        self.assertIs(core.postprocess, original_postprocess)

    def test_run_action_dispatches_under_profile_and_then_restores(self) -> None:
        originals = {
            name: getattr(core, name) for name in experiment._CORE_PROFILE_OVERRIDES
        }
        original_postprocess = core.postprocess

        def observe(*_args, **_kwargs):
            profile = {
                name: copy.deepcopy(getattr(core, name))
                for name in experiment._CORE_PROFILE_OVERRIDES
            }
            profile["postprocess_is_profile_override"] = (
                core.postprocess is experiment._profile_postprocess
            )
            return profile

        with mock.patch.object(core, "run_action", side_effect=observe):
            observed = experiment.run_action("status")
        self.assertEqual(
            observed,
            {**EXPECTED_PROFILE, "postprocess_is_profile_override": True},
        )
        for name, original in originals.items():
            self.assertIs(getattr(core, name), original)
        self.assertIs(core.postprocess, original_postprocess)

    def test_branch_epoch_statistics_preserve_e_equals_one(self) -> None:
        statistics = experiment._epoch_statistics([17, 1, 240, 1, 30])
        self.assertEqual(
            statistics,
            {
                "cell_count": 5,
                "minimum": 1,
                "median": 17.0,
                "mean": 57.8,
                "maximum": 240,
                "e_equals_one_cells": 2,
                "frequency": {"1": 2, "17": 1, "30": 1, "240": 1},
            },
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            experiment._epoch_statistics([])

    def test_branch_epoch_report_covers_80_cells_and_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            manifest_rows = []
            index = 0
            for tolerance in core.TOLERANCES:
                for fold in core.FOLDS:
                    for seed in core.SEEDS:
                        continuation_id = f"continuation-{index:02d}"
                        epochs = index % 3 + 1
                        payload = {
                            "refit_mode": "frozen_epoch_lr_replay_v1",
                            "num_epochs": epochs,
                            "generator_lr_trace": [
                                {"epoch": epoch, "lr": 5.0e-7}
                                for epoch in range(1, epochs + 1)
                            ],
                            "discriminator_lr_trace": [
                                {"epoch": epoch, "lr": 2.5e-7}
                                for epoch in range(1, epochs + 1)
                            ],
                            "parent_state_sha256": f"{index + 1:064x}",
                        }
                        recipe_path = core.write_json(
                            root
                            / "registry/branch_recipes"
                            / f"{continuation_id}.json",
                            payload,
                        )
                        manifest_rows.append(
                            {
                                "continuation_job_id": continuation_id,
                                "parent_job_id": f"parent-{index:02d}",
                                "path": str(recipe_path.resolve()),
                                "size_bytes": recipe_path.stat().st_size,
                                "sha256": core.sha256_file(recipe_path),
                                "num_epochs": epochs,
                            }
                        )
                        jobs.append(
                            {
                                "job_id": continuation_id,
                                "stage": core.CONTINUATION_STAGE,
                                "tolerance_minutes": tolerance,
                                "fold": fold,
                                "seed": seed,
                            }
                        )
                        index += 1
            core.write_json(
                core._registry_path(root),
                {
                    "branch_recipes_frozen": True,
                    "terminal_complete": False,
                    "jobs": jobs,
                },
            )
            core.write_csv(
                root / "registry/branch_recipe_manifest.csv",
                manifest_rows,
            )

            with experiment.unet_nolp_profile():
                paths = experiment._write_branch_epoch_report(root)
            self.assertEqual(len(paths), 4)
            with paths[0].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 80)
            self.assertEqual(rows[0]["shared_branch_epochs_E"], "1")
            summary = core.read_json(paths[1])
            self.assertEqual(summary["global"]["cell_count"], 80)
            self.assertEqual(summary["global"]["minimum"], 1)
            self.assertEqual(summary["global"]["maximum"], 3)
            mtimes = {path: path.stat().st_mtime_ns for path in paths}
            with experiment.unet_nolp_profile():
                experiment._write_branch_epoch_report(root)
            self.assertEqual(
                {path: path.stat().st_mtime_ns for path in paths},
                mtimes,
            )

            paths[2].write_text("tampered\n", encoding="utf-8")
            with (
                experiment.unet_nolp_profile(),
                self.assertRaisesRegex(ValueError, "report drift"),
            ):
                experiment._write_branch_epoch_report(root)

    def test_wrapper_is_inside_the_frozen_recursive_code_ledger(self) -> None:
        wrapper = "scripts/rq123/news_first_vol_film_unet_nolp_10seed.py"
        self.assertIn(wrapper, experiment.SOURCE_CODE_RELATIVE_PATHS)
        with experiment.unet_nolp_profile():
            self.assertIn(wrapper, core.SOURCE_CODE_RELATIVE_PATHS)
            relative = {
                path.relative_to(core.REPO_ROOT).as_posix()
                for path in core._code_paths()
            }
        self.assertIn(
            wrapper,
            relative,
        )

    def test_new_worker_module_is_used_only_inside_profile(self) -> None:
        job = {"job_id": "parent_05m_f1_2023q1_seed_42_parent_current_only"}
        config = {
            "runtime": {"python_executable": "/opt/py312/bin/python"},
            "source_config_path": "/tmp/frozen-unet.yaml",
        }
        with experiment.unet_nolp_profile():
            command = core._worker_command(
                Path("/tmp/unet-root"),
                job,
                config=config,
                dry_run=False,
                resume=True,
            )
        self.assertEqual(
            command[0:3],
            [
                "/opt/py312/bin/python",
                "-m",
                experiment.WORKER_MODULE,
            ],
        )
        self.assertIn("--resume", command)

        legacy_command = core._worker_command(
            Path("/tmp/legacy-root"),
            job,
            config=config,
            dry_run=True,
            resume=False,
        )
        self.assertEqual(legacy_command[2], LEGACY_DEFAULTS["WORKER_MODULE"])
        self.assertIn("--worker-dry-run", legacy_command)

    def test_legacy_defaults_and_status_dispatch_contract_do_not_drift(self) -> None:
        for name, expected in LEGACY_DEFAULTS.items():
            self.assertEqual(getattr(core, name), expected)
        parsed = core._parser().parse_args(["status"])
        self.assertEqual(parsed.config, LEGACY_DEFAULTS["DEFAULT_CONFIG"])
        self.assertEqual(parsed.output_dir, LEGACY_DEFAULTS["DEFAULT_OUTPUT_DIR"])
        with mock.patch.object(
            core,
            "status_experiment",
            return_value={"status": "terminal_complete", "jobs": 400},
        ) as status:
            result = core.run_action(
                "status",
                config_path=LEGACY_DEFAULTS["DEFAULT_CONFIG"],
                output_dir=LEGACY_DEFAULTS["DEFAULT_OUTPUT_DIR"],
            )
        status.assert_called_once_with(LEGACY_DEFAULTS["DEFAULT_OUTPUT_DIR"])
        self.assertEqual(result, {"status": "terminal_complete", "jobs": 400})


class TestUnetMatrixAndTrainingContract(unittest.TestCase):
    def test_config_model_contract_and_400_cell_matrix_are_exact(self) -> None:
        with experiment.unet_nolp_profile():
            config = core.load_config(experiment.DEFAULT_CONFIG)
            specs = core.planned_specs()
            model_contract = core.model_contract(config)
            architecture = core.architecture_profile_contract(config)

        self.assertEqual(len(specs), 400)
        self.assertEqual(len({core.job_id(row) for row in specs}), 400)
        self.assertEqual(
            Counter(str(row["stage"]) for row in specs),
            {"parents": 80, "continuations": 80, "text_branches": 240},
        )
        self.assertEqual(
            Counter(int(row["tolerance_minutes"]) for row in specs),
            {5: 240, 30: 160},
        )
        self.assertEqual(config["matrix"]["expected_training_jobs"], 400)
        self.assertEqual(config["matrix"]["expected_prediction_cells"], 400)
        self.assertEqual(core.EXPECTED_PREDICTION_CELLS, 400)
        core.validate_gpu_balance(specs)
        block_assignments = {
            (
                int(row["tolerance_minutes"]),
                str(row["fold"]),
                int(row["seed"]),
            ): int(row["gpu_id"])
            for row in specs
        }
        self.assertEqual(len(block_assignments), 80)
        self.assertEqual(Counter(block_assignments.values()), {0: 40, 1: 40})
        self.assertEqual(
            model_contract["generator_conditioning_mode"],
            "film_unet_mask_coords_v1",
        )
        self.assertEqual(
            model_contract["critic_conditioning_mode"],
            "lp_disabled_same_shape_v1",
        )
        self.assertEqual(model_contract["generator_parameters"], 827_745)
        self.assertEqual(model_contract["critic_parameters"], 729_157)
        self.assertEqual(model_contract["total_parameters"], 1_556_902)
        self.assertEqual(
            architecture["architecture_profile_sha256"],
            experiment.EXPECTED_ARCHITECTURE_PROFILE_SHA256,
        )
        self.assertEqual(
            model_contract["model_contract_sha256"],
            EXPECTED_MODEL_CONTRACT_SHA256,
        )
        self.assertEqual(
            core.sha256_file(experiment.DEFAULT_CONFIG),
            EXPECTED_CONFIG_SHA256,
        )

    def test_parent_continuation_and_branch_epoch_lr_contract(self) -> None:
        with experiment.unet_nolp_profile():
            config = core.load_config(experiment.DEFAULT_CONFIG)
        training = config["training"]
        self.assertEqual(
            core._stage_num_epochs(training, core.PARENT_STAGE),
            30,
        )
        self.assertEqual(
            core._stage_num_epochs(training, core.CONTINUATION_STAGE),
            240,
        )
        self.assertEqual(
            core._stage_early_stopping_min_epochs(training, core.PARENT_STAGE),
            30,
        )
        self.assertEqual(
            core._stage_early_stopping_min_epochs(training, core.CONTINUATION_STAGE),
            30,
        )
        for epochs in (1, 17, 240):
            frozen_recipe = {
                "num_epochs": epochs,
                "generator_lr_trace": [
                    {"epoch": epoch, "lr": 5.0e-7} for epoch in range(1, epochs + 1)
                ],
                "discriminator_lr_trace": [
                    {"epoch": epoch, "lr": 2.5e-7} for epoch in range(1, epochs + 1)
                ],
            }
            self.assertEqual(
                core._stage_num_epochs(
                    training,
                    core.BRANCH_STAGE,
                    recipe=frozen_recipe,
                ),
                epochs,
            )
            self.assertEqual(
                [row["epoch"] for row in frozen_recipe["generator_lr_trace"]],
                list(range(1, epochs + 1)),
            )
            self.assertEqual(
                [row["epoch"] for row in frozen_recipe["discriminator_lr_trace"]],
                list(range(1, epochs + 1)),
            )

    def test_zero_warmup_is_explicit_and_fail_closed(self) -> None:
        with experiment.unet_nolp_profile():
            config = core.load_config(experiment.DEFAULT_CONFIG)
            self.assertEqual(config["training"]["lr_warmup_epochs"], 0)
            invalid = copy.deepcopy(config)
            invalid["training"]["lr_warmup_epochs"] = 1
            with self.assertRaisesRegex(ValueError, "zero LR warmup"):
                core.validate_config(invalid)

    def test_recipe_manifest_requires_80_complete_paired_lr_traces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index in range(80):
                epochs = index % 3 + 1
                payload = {
                    "refit_mode": "frozen_epoch_lr_replay_v1",
                    "num_epochs": epochs,
                    "generator_lr_trace": [
                        {"epoch": epoch, "lr": 5.0e-7} for epoch in range(1, epochs + 1)
                    ],
                    "discriminator_lr_trace": [
                        {"epoch": epoch, "lr": 5.0e-7} for epoch in range(1, epochs + 1)
                    ],
                }
                path = core.write_json(
                    root / "registry/branch_recipes" / f"cont-{index}.json",
                    payload,
                )
                rows.append(
                    {
                        "continuation_job_id": f"cont-{index}",
                        "parent_job_id": f"parent-{index}",
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": core.sha256_file(path),
                        "num_epochs": epochs,
                    }
                )
            core.write_csv(
                root / "registry/branch_recipe_manifest.csv",
                rows,
            )
            recipes = core._validate_recipe_manifest(root)
            self.assertEqual(len(recipes), 80)
            self.assertEqual(recipes["cont-79"]["num_epochs"], 2)

            invalid_path = Path(rows[0]["path"])
            invalid = core.read_json(invalid_path)
            invalid["discriminator_lr_trace"] = []
            core.write_json(invalid_path, invalid)
            rows[0]["size_bytes"] = invalid_path.stat().st_size
            rows[0]["sha256"] = core.sha256_file(invalid_path)
            core.write_csv(
                root / "registry/branch_recipe_manifest.csv",
                rows,
            )
            with self.assertRaisesRegex(ValueError, "Invalid branch recipe"):
                core._validate_recipe_manifest(root)


class TestUnetFullState(unittest.TestCase):
    def test_exact_parameter_counts_and_full_state_next_update_equivalence(
        self,
    ) -> None:
        with experiment.unet_nolp_profile():
            config = core.load_config(experiment.DEFAULT_CONFIG)
        torch.manual_seed(8128)
        control = _components(config)
        (
            control_g,
            control_d,
            control_g_optimizer,
            control_d_optimizer,
            control_g_scheduler,
            control_d_scheduler,
        ) = control
        self.assertEqual(sum(p.numel() for p in control_g.parameters()), 827_745)
        self.assertEqual(sum(p.numel() for p in control_d.parameters()), 729_157)

        _parameter_step(control_g, control_g_optimizer, scale=0.4)
        _parameter_step(control_d, control_d_optimizer, scale=-0.3)
        control_g_scheduler.step(1.0)
        control_d_scheduler.step(1.0)
        saved_g_scheduler = copy.deepcopy(control_g_scheduler.state_dict())
        saved_d_scheduler = copy.deepcopy(control_d_scheduler.state_dict())

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "unet-parent.pt"
            digest = save_full_training_state(
                state_path,
                generator=control_g,
                discriminator=control_d,
                generator_optimizer=control_g_optimizer,
                discriminator_optimizer=control_d_optimizer,
                generator_scheduler=control_g_scheduler,
                discriminator_scheduler=control_d_scheduler,
                loader_generator=torch.Generator().manual_seed(991),
                completed_epoch=30,
                lineage=_lineage(),
                contract_sha256=hashlib.sha256(b"contract").hexdigest(),
            )

            _parameter_step(control_g, control_g_optimizer, scale=0.2)
            _parameter_step(control_d, control_d_optimizer, scale=-0.1)
            control_g_scheduler.step(2.0)
            control_d_scheduler.step(2.0)

            torch.manual_seed(17)
            restored = _components(config)
            (
                restored_g,
                restored_d,
                restored_g_optimizer,
                restored_d_optimizer,
                restored_g_scheduler,
                restored_d_scheduler,
            ) = restored
            loaded = load_full_training_state(
                state_path,
                digest,
                generator=restored_g,
                discriminator=restored_d,
                generator_optimizer=restored_g_optimizer,
                discriminator_optimizer=restored_d_optimizer,
                generator_scheduler=restored_g_scheduler,
                discriminator_scheduler=restored_d_scheduler,
                loader_generator=torch.Generator(),
                expected_lineage=_lineage(),
                restore_schedulers=True,
                map_location="cpu",
            )
            self.assertEqual(loaded["completed_epoch"], 30)
            _assert_nested_equal(
                self, restored_g_scheduler.state_dict(), saved_g_scheduler
            )
            _assert_nested_equal(
                self, restored_d_scheduler.state_dict(), saved_d_scheduler
            )

            _parameter_step(restored_g, restored_g_optimizer, scale=0.2)
            _parameter_step(restored_d, restored_d_optimizer, scale=-0.1)
            restored_g_scheduler.step(2.0)
            restored_d_scheduler.step(2.0)

        _assert_nested_equal(self, control_g.state_dict(), restored_g.state_dict())
        _assert_nested_equal(self, control_d.state_dict(), restored_d.state_dict())
        _assert_nested_equal(
            self,
            control_g_optimizer.state_dict(),
            restored_g_optimizer.state_dict(),
        )
        _assert_nested_equal(
            self,
            control_d_optimizer.state_dict(),
            restored_d_optimizer.state_dict(),
        )
        _assert_nested_equal(
            self,
            control_g_scheduler.state_dict(),
            restored_g_scheduler.state_dict(),
        )
        _assert_nested_equal(
            self,
            control_d_scheduler.state_dict(),
            restored_d_scheduler.state_dict(),
        )


if __name__ == "__main__":
    unittest.main()
