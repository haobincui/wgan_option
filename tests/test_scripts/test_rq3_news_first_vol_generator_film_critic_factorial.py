from __future__ import annotations

from copy import deepcopy
import csv
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.rq3 import main as rq3_main  # noqa: E402
from scripts.rq3 import (  # noqa: E402
    news_first_vol_generator_film_critic_factorial as factorial,
)
from wgan_option.config import Config, load_config  # noqa: E402
from wgan_option.models.gan_model import WGAN_GP  # noqa: E402
from wgan_option.utils.merged_xlsx import (  # noqa: E402
    create_configured_vol_surface_dataloaders,
)


CONFIG = ROOT / "configs/rq3/news_first_vol_generator_film_critic_factorial.yaml"
SOURCE_5M = (
    ROOT / "data/processed/rq3/"
    "news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_v1/"
    "tolerance_05m/merged_vol.xlsx"
)


def _self_hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    output = dict(payload)
    output[field] = factorial._payload_sha256(output)
    return output


def _write_freeze_candidate(root: Path, development_ids: list[str]) -> Path:
    selection_path = factorial._write_json(
        root / "analysis" / "film_critic_q3_selection.json",
        _self_hashed({"schema_version": 1, "winner": {}}, "selection_sha256"),
    )
    recipe_rows = []
    for job_id in development_ids:
        recipe_path = factorial._write_json(
            root / "analysis" / "refit_recipes" / f"{job_id}.json",
            {
                "schema_version": 1,
                "refit_mode": factorial.REFIT_MODE,
                "num_epochs": 1,
                "generator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
                "discriminator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
            },
        )
        recipe_rows.append(
            {
                "development_job_id": job_id,
                "recipe_path": str(recipe_path.resolve()),
                "recipe_sha256": factorial._sha256_file(recipe_path),
            }
        )
    manifest = _self_hashed(
        {
            "schema_version": 1,
            "refit_mode": factorial.REFIT_MODE,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": factorial._sha256_file(selection_path),
            "recipe_count": 16,
            "recipes": recipe_rows,
        },
        "manifest_sha256",
    )
    factorial._write_json(root / "analysis" / "refit_recipe_manifest.json", manifest)
    return selection_path


def _write_terminal_fixture(root: Path) -> None:
    (root / "registry" / "jobs").mkdir(parents=True)
    jobs: list[dict[str, object]] = []
    for stage, count, roles in (
        (factorial.DEVELOPMENT_STAGE, 16, factorial.DEVELOPMENT_ARTIFACT_ROLES),
        (factorial.REFIT_STAGE, 16, factorial.REFIT_ARTIFACT_ROLES),
    ):
        for index in range(count):
            job_id = f"{stage}_{index:02d}"
            config_sha = f"config-{job_id}"
            jobs.append(
                {
                    "job_id": job_id,
                    "experiment_stage": stage,
                    "config_sha256": config_sha,
                }
            )
            artifacts = []
            for role in roles:
                path = root / "runs" / job_id / f"{role}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{job_id}:{role}\n", encoding="utf-8")
                artifacts.append(
                    {
                        "artifact_role": role,
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": factorial._sha256_file(path),
                    }
                )
            factorial._write_json(
                factorial._job_status_path(root, job_id),
                {
                    "job_id": job_id,
                    "experiment_stage": stage,
                    "config_sha256": config_sha,
                    "status": "completed",
                    "artifacts": artifacts,
                },
            )
    factorial._write_json(
        root / "registry" / "jobs.json",
        {
            "experiment_kind": factorial.EXPERIMENT_KIND,
            "status": "completed",
            "q4_evaluated": True,
            "jobs": jobs,
        },
    )
    factorial._write_json(
        root / "registry" / "experiment_status.json", {"status": "completed"}
    )
    required_files = (
        "analysis/film_critic_q3_pair_metrics.csv.gz",
        "analysis/film_critic_q3_cell_scores.csv",
        "analysis/film_critic_q3_architecture_contrasts.csv",
        "analysis/film_critic_q3_text_contrasts.csv",
        "analysis/film_critic_q3_factorial_effects.csv",
        "analysis/film_critic_q3_selection.json",
        "analysis/refit_recipe_manifest.json",
        "analysis/film_critic_q4_pair_metrics.csv.gz",
        "analysis/film_critic_q4_primary_contrasts.csv",
        "analysis/film_critic_q4_30m_secondary.csv",
        "analysis/film_critic_q4_summary.json",
        "data_windows/q4/common_05m_q4.xlsx",
        "data_windows/q4/q4_window_manifest.json",
        "q4_checkpoint_allowlist.csv",
        "report/film_critic_factorial_conclusion.md",
        "report/film_critic_factorial_conclusion.html",
        "resource_usage.csv",
        "resource_summary.csv",
        "resource_summary.json",
        "code_hashes.csv",
        "config_hashes.csv",
        "source_hashes.csv",
        "model_contract_manifest.json",
        "resolved_config.yaml",
        "rolling_split_manifest.csv",
        "task_registry.csv",
    )
    for relative in required_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{relative}\n", encoding="utf-8")
    development_ids = [
        str(job["job_id"])
        for job in jobs
        if job["experiment_stage"] == factorial.DEVELOPMENT_STAGE
    ]
    for job_id in development_ids:
        path = root / "analysis" / "refit_recipes" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"recipe:{job_id}\n", encoding="utf-8")
    for directory, count in (
        ("analysis/predictions/q3_development_best_learned", 16),
        ("analysis/predictions/q4_refit_final_05m", 8),
        ("analysis/predictions/q4_refit_final_30m", 8),
    ):
        for index in range(count):
            base = root / directory / f"prediction_{index:02d}"
            base.parent.mkdir(parents=True, exist_ok=True)
            Path(f"{base}.csv.gz").write_text("prediction\n", encoding="utf-8")
            Path(f"{base}.manifest.json").write_text("manifest\n", encoding="utf-8")
    placeholder = _self_hashed(
        {
            "schema_version": 1,
            "status": "passed",
            "terminal_output_artifact_count": 1,
            "terminal_output_manifest_pending": False,
        },
        "payload_sha256",
    )
    factorial._write_json(root / "qa.json", placeholder)
    prospective_count = len(factorial._terminal_deliverable_paths(root)) + 1
    qa = _self_hashed(
        {
            "schema_version": 1,
            "status": "passed",
            "terminal_output_artifact_count": prospective_count,
            "terminal_output_manifest_pending": False,
        },
        "payload_sha256",
    )
    factorial._write_json(root / "qa.json", qa)


class FactorialContractTests(unittest.TestCase):
    def test_exact_axes_parameter_counts_and_fingerprints(self) -> None:
        resolved = factorial.resolve_config(CONFIG)
        grid = factorial._surface_grid_contract()
        self.assertEqual(
            grid["maturity_days_grid"],
            [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38],
        )
        self.assertEqual(grid["surface_shape"], [16, 16])
        self.assertEqual(len(grid["surface_grid_sha256"]), 64)
        training = resolved["models"]["wgan"]["training"]
        for generator_mode in factorial.GENERATOR_MODES:
            for critic_mode in factorial.CRITIC_MODES:
                spec = {
                    "generator_conditioning_mode": generator_mode,
                    "critic_conditioning_mode": critic_mode,
                    "text_ablation_mode": "real_text",
                }
                contract = factorial._conditioning_contract(spec)
                config = Config(
                    **{
                        **training,
                        "generator_conditioning_mode": generator_mode,
                        "critic_conditioning_mode": critic_mode,
                        "support_mask_mode": "raw_joint",
                        "cuda": False,
                    }
                )
                model = WGAN_GP(
                    config,
                    strike_grid=grid["strike_grid"],
                    maturity_grid_days=grid["maturity_days_grid"],
                    embedding_dim=1024,
                )
                observed = sum(
                    parameter.numel()
                    for module in (model.G, model.D)
                    for parameter in module.parameters()
                )
                self.assertEqual(observed, contract["expected_wgan_parameters"])
                self.assertEqual(
                    model.G.generator_conditioning_fingerprint,
                    contract["generator_conditioning_fingerprint"],
                )
                self.assertEqual(
                    model.D.critic_conditioning_fingerprint,
                    contract["critic_conditioning_fingerprint"],
                )

    def test_stage_matrices_are_unique_and_every_axis_is_gpu_balanced(self) -> None:
        development = factorial._balanced_assignments(
            factorial.factorial_specs(factorial.DEVELOPMENT_STAGE),
            gpu_ids=(0, 1),
            slots_per_gpu=8,
        )
        refit = factorial._balanced_assignments(
            factorial.factorial_specs(factorial.REFIT_STAGE),
            gpu_ids=(0, 1),
            slots_per_gpu=8,
            invert_gpu=True,
        )
        self.assertEqual(len(development), 16)
        self.assertEqual(len(refit), 16)
        self.assertEqual({row["wave"] for row in development}, {1})
        keys = (
            "generator_conditioning_mode",
            "critic_conditioning_mode",
            "text_ablation_mode",
            "tolerance_minutes",
            "seed",
        )
        self.assertEqual(
            len({tuple(row[key] for key in keys) for row in development}), 16
        )
        by_cell = {tuple(row[key] for key in keys): row for row in development}
        for row in refit:
            self.assertNotEqual(
                by_cell[tuple(row[key] for key in keys)]["gpu_id"], row["gpu_id"]
            )
        fallback = factorial._balanced_assignments(
            factorial.factorial_specs(), gpu_ids=(0, 1), slots_per_gpu=4
        )
        self.assertEqual({row["wave"] for row in fallback}, {1, 2})

    def test_refit_payload_has_exact_training_only_replay_contract(self) -> None:
        resolved = factorial.resolve_config(CONFIG)
        spec = factorial.factorial_specs(factorial.REFIT_STAGE)[0]
        recipe = {
            "schema_version": 1,
            "refit_mode": factorial.REFIT_MODE,
            "num_epochs": 2,
            "generator_lr_trace": [
                {"epoch": 1, "lr": 5e-7},
                {"epoch": 2, "lr": 2.5e-7},
            ],
            "discriminator_lr_trace": [
                {"epoch": 1, "lr": 5e-7},
                {"epoch": 2, "lr": 5e-7},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = factorial._write_json(root / "recipe.json", recipe)
            payload = factorial._training_payload(
                resolved,
                root,
                spec=spec,
                stage=factorial.REFIT_STAGE,
                recipe_path=recipe_path,
            )
        self.assertEqual(payload["news_first_refit_mode"], factorial.REFIT_MODE)
        self.assertEqual(payload["num_epochs"], 2)
        self.assertEqual(
            payload["news_first_train_end_utc"],
            payload["news_first_validation_end_utc"],
        )
        for key in (
            "news_first_materialize_validation_loader",
            "news_first_materialize_test_loader",
            "use_reduce_lr_on_plateau",
            "use_early_stopping",
            "evaluate_initial_checkpoint",
        ):
            self.assertFalse(payload[key], key)
        self.assertEqual(
            payload["generator_conditioning_mode"],
            "bottleneck_concat_v1",
        )
        self.assertEqual(payload["critic_conditioning_mode"], "lp_concat_v1")

    @unittest.skipUnless(SOURCE_5M.is_file(), "exact-TTM workbook is unavailable")
    def test_refit_real_config_and_loader_materialize_training_only(self) -> None:
        """Exercise the production Config and loader, not merely the payload dict."""

        resolved = factorial.resolve_config(CONFIG)
        source = pd.read_excel(SOURCE_5M, sheet_name="gan_input_ready")
        origin = pd.to_datetime(source["effective_origin_utc"], utc=True)
        selected = source.loc[
            (origin < pd.Timestamp("2023-10-01T00:00:00Z"))
            & source["surface_training_eligible"].astype(bool)
        ].head(4)
        self.assertGreaterEqual(len(selected), 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window_dir = root / "data_windows" / "pre_q4"
            window_dir.mkdir(parents=True)
            window = window_dir / "tolerance_05m_pre_q4.xlsx"
            with pd.ExcelWriter(window, engine="openpyxl") as writer:
                selected.to_excel(writer, sheet_name="gan_input_ready", index=False)
            recipe = {
                "schema_version": 1,
                "refit_mode": factorial.REFIT_MODE,
                "num_epochs": 1,
                "generator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
                "discriminator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
            }
            recipe_path = factorial._write_json(root / "recipe.json", recipe)
            payload = factorial._training_payload(
                resolved,
                root,
                spec=factorial.factorial_specs(factorial.REFIT_STAGE)[0],
                stage=factorial.REFIT_STAGE,
                recipe_path=recipe_path,
            )
            payload.update(
                {
                    "data_path": str(window),
                    "news_first_common_eval_data_path": str(window),
                    "output_root": str(root / "run"),
                    "cuda": False,
                }
            )
            config_path = root / "refit.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
            )
            config = load_config(config_path)
            bundle = create_configured_vol_surface_dataloaders(config)
        self.assertIsNone(bundle.val_loader)
        self.assertIsNone(bundle.test_loader)
        self.assertEqual(bundle.val_items, [])
        self.assertEqual(bundle.test_items, [])
        self.assertEqual(bundle.val_samples, 0)
        self.assertEqual(bundle.test_samples, 0)
        self.assertGreater(bundle.train_samples, 0)
        self.assertEqual(bundle.split_metadata["mode"], "news_first_training_only")
        self.assertFalse(bundle.split_metadata["validation_materialized"])
        self.assertTrue(
            all(
                pd.Timestamp(value) < pd.Timestamp("2023-10-01T00:00:00Z")
                for value in bundle.train_timestamps
            )
        )

    def test_q4_window_cannot_materialize_before_explicit_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir()
            factorial._write_json(
                root / "registry" / "jobs.json",
                {
                    "experiment_kind": factorial.EXPERIMENT_KIND,
                    "q4_gate_open": False,
                    "refit_complete": True,
                    "jobs": [],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "explicit open gate"):
                factorial._materialize_q4_common_window(
                    factorial.resolve_config(CONFIG), root
                )
            self.assertFalse((root / "data_windows" / "q4").exists())

    def test_recipe_schema_is_exact_and_epoch_trace_is_contiguous(self) -> None:
        valid = {
            "schema_version": 1,
            "refit_mode": factorial.REFIT_MODE,
            "num_epochs": 1,
            "generator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
            "discriminator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
        }
        factorial._validate_refit_recipe(valid)
        changed = deepcopy(valid)
        changed["source_job_id"] = "not-part-of-the-core-schema"
        with self.assertRaisesRegex(ValueError, "exactly"):
            factorial._validate_refit_recipe(changed)
        changed = deepcopy(valid)
        changed["generator_lr_trace"][0]["epoch"] = 2
        with self.assertRaisesRegex(ValueError, "epoch1..N"):
            factorial._validate_refit_recipe(changed)

    def test_complete_unregistered_freeze_is_recovered_without_rerunning_hooks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development_ids = [f"development_{index:02d}" for index in range(16)]
            (root / "registry").mkdir(parents=True)
            factorial._write_json(
                root / "registry" / "jobs.json",
                {
                    "experiment_kind": factorial.EXPERIMENT_KIND,
                    "selection_frozen": False,
                    "jobs": [
                        {
                            "job_id": job_id,
                            "experiment_stage": factorial.DEVELOPMENT_STAGE,
                        }
                        for job_id in development_ids
                    ],
                },
            )
            selection_path = _write_freeze_candidate(root, development_ids)
            with (
                patch.object(factorial, "_validate_root_lineage"),
                patch.object(factorial, "_validate_stage_complete"),
                patch.object(
                    factorial,
                    "_register_frozen_refit_recipes",
                    return_value=selection_path,
                ) as register,
                patch.object(factorial, "run_film_critic_q3_analysis") as analysis,
                patch.object(factorial, "freeze_refit_recipes") as recipes,
            ):
                factorial.freeze_factorial_selection(root)
            analysis.assert_not_called()
            recipes.assert_not_called()
            register.assert_called_once_with(root.resolve(), selection_path)

    def test_partial_unregistered_freeze_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = [
                {
                    "job_id": f"development_{index:02d}",
                    "experiment_stage": factorial.DEVELOPMENT_STAGE,
                }
                for index in range(16)
            ]
            selection = _self_hashed(
                {"schema_version": 1, "winner": {}}, "selection_sha256"
            )
            factorial._write_json(
                root / "analysis" / "film_critic_q3_selection.json", selection
            )
            with self.assertRaisesRegex(
                ValueError, "Partial Q3 selection/refit freeze"
            ):
                factorial._recoverable_analysis_freeze_candidate(root, development)

    def test_registration_journal_recovers_interrupted_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry").mkdir(parents=True)
            development = [
                {
                    "job_id": f"development_{index:02d}",
                    "experiment_stage": factorial.DEVELOPMENT_STAGE,
                }
                for index in range(16)
            ]
            refit = [
                {
                    "job_id": f"refit_{index:02d}",
                    "experiment_stage": factorial.REFIT_STAGE,
                    "config_sha256": f"config-{index:02d}",
                }
                for index in range(16)
            ]
            factorial._write_json(
                root / "registry" / "jobs.json",
                {
                    "experiment_kind": factorial.EXPERIMENT_KIND,
                    "selection_frozen": False,
                    "jobs": development,
                },
            )
            config_target = root / "config-target.yaml"
            config_target.write_text("config\n", encoding="utf-8")
            for name in ("source_hashes.csv", "code_hashes.csv"):
                (root / name).write_text("anchor\n", encoding="utf-8")
            final_registry = {
                "experiment_kind": factorial.EXPERIMENT_KIND,
                "selection_frozen": True,
                "selection_sha256": "selection",
                "source_hashes_sha256": factorial._sha256_file(
                    root / "source_hashes.csv"
                ),
                "code_hashes_sha256": factorial._sha256_file(root / "code_hashes.csv"),
                "jobs": development + refit,
            }
            journal = _self_hashed(
                {
                    "schema_version": 1,
                    "transaction": "selection_registration_v1",
                    "final_registry": final_registry,
                    "config_rows": [
                        {
                            "source_role": "target",
                            "path": str(config_target.resolve()),
                            "size_bytes": config_target.stat().st_size,
                            "sha256": factorial._sha256_file(config_target),
                        }
                    ],
                },
                "payload_sha256",
            )
            factorial._write_json(
                factorial._selection_registration_journal_path(root), journal
            )
            with (
                patch.object(factorial, "_write_registry_exports"),
                patch.object(factorial, "_validate_root_lineage"),
                patch.object(factorial, "_validate_registered_refit_freeze"),
            ):
                factorial._recover_selection_registration(root)
            recovered = factorial._read_json(root / "registry" / "jobs.json")
            self.assertTrue(recovered["selection_frozen"])
            self.assertEqual(len(recovered["jobs"]), 32)
            self.assertFalse(
                factorial._selection_registration_journal_path(root).exists()
            )
            for job in refit:
                self.assertTrue(
                    factorial._job_status_path(root, str(job["job_id"])).is_file()
                )

    def test_refit_job_schema_binds_parent_and_conditioning_contract(self) -> None:
        resolved = factorial.resolve_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data_windows" / "pre_q4"
            data.mkdir(parents=True)
            for tolerance in factorial.TOLERANCES:
                (data / f"tolerance_{tolerance:02d}m_pre_q4.xlsx").write_bytes(
                    b"lineage-only-test"
                )
            recipes = {}
            for spec in factorial.factorial_specs(factorial.DEVELOPMENT_STAGE):
                development_id = factorial._job_id(factorial.DEVELOPMENT_STAGE, spec)
                recipe = {
                    "schema_version": 1,
                    "refit_mode": factorial.REFIT_MODE,
                    "num_epochs": 1,
                    "generator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
                    "discriminator_lr_trace": [{"epoch": 1, "lr": 5e-7}],
                }
                recipes[development_id] = factorial._write_json(
                    root / "recipes" / f"{development_id}.json", recipe
                )
            jobs, _ = factorial._build_jobs(
                resolved,
                root,
                stage=factorial.REFIT_STAGE,
                slots_per_gpu=8,
                recipes=recipes,
            )
        self.assertEqual(len(jobs), 16)
        for job in jobs:
            self.assertEqual(
                job["parent_development_job_id"], job["development_job_id"]
            )
            self.assertIn("conditioning_contract_sha256", job)
            self.assertIn("model_contract_sha256", job)
            self.assertEqual(len(job["conditioning_contract_sha256"]), 64)
            self.assertEqual(len(job["model_contract_sha256"]), 64)

    def test_resource_summary_does_not_mislabel_process_hours_as_gpu_hours(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry" / "jobs").mkdir(parents=True)
            jobs = []
            for gpu_id in (0, 1):
                for slot in (0, 1):
                    job_id = f"job_{gpu_id}_{slot}"
                    jobs.append(
                        {
                            "job_id": job_id,
                            "experiment_stage": factorial.DEVELOPMENT_STAGE,
                            "wave": 1,
                            "gpu_id": gpu_id,
                            "gpu_slot": slot,
                        }
                    )
                    factorial._write_json(
                        factorial._job_status_path(root, job_id),
                        {
                            "status": "completed",
                            "started_at_utc": "2026-08-21T10:00:00Z",
                            "completed_at_utc": "2026-08-21T11:00:00Z",
                        },
                    )
            factorial._write_json(
                root / "registry" / "jobs.json",
                {
                    "experiment_kind": factorial.EXPERIMENT_KIND,
                    "jobs": jobs,
                },
            )
            (root / "resource_usage.csv").write_text(
                "timestamp_utc,wave,gpu_index,sample_status,memory_used_mib,utilization_gpu_pct\n"
                "2026-08-21T10:30:00Z,1,0,ok,100,50\n"
                "2026-08-21T10:30:00Z,1,1,ok,120,60\n",
                encoding="utf-8",
            )
            csv_path, json_path = factorial._write_factorial_resource_summary(root)
            summary = factorial._read_json(json_path)
            rows = pd.read_csv(csv_path)
        self.assertAlmostEqual(summary["total_job_process_hours"], 4.0)
        self.assertAlmostEqual(summary["total_physical_gpu_hours"], 2.0)
        self.assertEqual(len(rows[rows["scope"] == "job_process"]), 4)
        self.assertEqual(len(rows[rows["scope"] == "physical_gpu_wave"]), 2)
        self.assertTrue(
            summary["accounting_contract"]["do_not_sum_job_process_hours_as_gpu_hours"]
        )

    def test_terminal_manifest_is_anchored_complete_and_excludes_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_terminal_fixture(root)
            manifest = factorial._finalize_terminal_lineage(root)
            rows = factorial._validate_terminal_output_manifest(root)
            self.assertNotIn(str(manifest.resolve()), {row["path"] for row in rows})
            qa = factorial._read_json(root / "qa.json")
            self.assertFalse(qa["terminal_output_manifest_pending"])
            self.assertEqual(qa["terminal_output_artifact_count"], len(rows))
            (root / "report" / "film_critic_factorial_conclusion.md").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "artifact drift"):
                factorial._validate_terminal_output_manifest(root)

    def test_terminal_manifest_rejects_recorded_size_and_role_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_terminal_fixture(root)
            manifest_path = factorial._finalize_terminal_lineage(root)
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["size_bytes"] = str(int(rows[0]["size_bytes"]) + 1)
            factorial._write_csv(manifest_path, rows, tuple(rows[0]))
            registry = factorial._read_json(root / "registry" / "jobs.json")
            registry["terminal_output_manifest_sha256"] = factorial._sha256_file(
                manifest_path
            )
            factorial._write_json(root / "registry" / "jobs.json", registry)
            with self.assertRaisesRegex(ValueError, "artifact drift"):
                factorial._validate_terminal_output_manifest(root)
            rows[0]["size_bytes"] = str((Path(rows[0]["path"])).stat().st_size)
            rows[1]["artifact_role"] = rows[0]["artifact_role"]
            factorial._write_csv(manifest_path, rows, tuple(rows[0]))
            registry["terminal_output_manifest_sha256"] = factorial._sha256_file(
                manifest_path
            )
            factorial._write_json(root / "registry" / "jobs.json", registry)
            with self.assertRaisesRegex(ValueError, "uniqueness"):
                factorial._validate_terminal_output_manifest(root)

    def test_terminal_finalize_rejects_pending_selection_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_terminal_fixture(root)
            factorial._write_json(
                factorial._selection_registration_journal_path(root),
                {"payload_sha256": "pending"},
            )
            with self.assertRaisesRegex(ValueError, "pending selection transaction"):
                factorial._finalize_terminal_lineage(root)
            self.assertFalse((root / "output_hashes.csv").exists())

    def test_completed_postprocess_retry_is_read_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = Mock()
            with (
                patch.object(factorial, "_validate_root_lineage"),
                patch.object(
                    factorial,
                    "_load_registry",
                    return_value={
                        "status": "completed",
                        "terminal_output_manifest_path": str(
                            root / "output_hashes.csv"
                        ),
                    },
                ),
                patch.object(
                    factorial, "_validate_completed_q4_outputs"
                ) as q4_validate,
                patch.object(
                    factorial, "_validate_terminal_output_manifest"
                ) as terminal_validate,
                patch.object(factorial, "_write_experiment_status") as status_write,
            ):
                observed = factorial.postprocess_factorial_experiment(
                    root, report_hook=report
                )
            self.assertEqual(observed, root.resolve())
            q4_validate.assert_called_once_with(root.resolve())
            terminal_validate.assert_called_once_with(root.resolve())
            report.assert_not_called()
            status_write.assert_not_called()

    def test_historical_q4_exposure_is_revalidated_from_external_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.csv.gz"
            path.write_text("historical evidence\n", encoding="utf-8")
            summary = {
                "historical_q4_exposure": {
                    "historical_prediction_path": str(path.resolve()),
                    "historical_prediction_sha256": factorial._sha256_file(path),
                    "current_pair_count": 143,
                    "overlapping_pair_count": 143,
                    "pair_overlap_label": "143/143",
                    "historically_exposed": True,
                    "interpretation": (
                        "retrospective_frozen_exploratory_not_confirmatory"
                    ),
                }
            }
            factorial._validate_historical_q4_exposure(summary)
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence/path/hash drift"):
                factorial._validate_historical_q4_exposure(summary)


class FactorialCliTests(unittest.TestCase):
    def test_cli_exposes_all_actions_and_defaults_to_safe_prepare(self) -> None:
        parser = rq3_main.build_parser()
        command = "train-news-first-vol-generator-film-critic-factorial"
        parsed = parser.parse_args([command])
        self.assertEqual(parsed.action, "prepare")
        for action in (
            "prepare",
            "benchmark",
            "dry-run",
            "launch-development",
            "freeze-selection",
            "launch-refit",
            "evaluate-q4",
            "postprocess",
            "worker",
            "qa",
        ):
            self.assertEqual(parser.parse_args([command, action]).action, action)

    def test_cli_dispatches_without_importing_training_at_parser_build(self) -> None:
        command = "train-news-first-vol-generator-film-critic-factorial"
        expected = Path("/tmp/factorial-dispatch")
        with patch.object(
            factorial,
            "run_news_first_vol_generator_film_critic_factorial",
            return_value=expected,
        ) as mocked:
            observed = rq3_main.main([command, "qa", "--output-dir", str(expected)])
        self.assertEqual(observed, expected)
        self.assertEqual(mocked.call_args.kwargs["action"], "qa")


if __name__ == "__main__":
    unittest.main()
