from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from scripts.rq3 import news_first_vol_legacy_architecture_seed as sweep
from scripts.rq3 import news_first_vol_legacy_architecture_seed_analysis as analysis
from scripts.rq3 import main as rq3_main
from wgan_option.config import Config
from wgan_option.models.gan_model import WGAN_GP


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rq3/news_first_vol_legacy_architecture_seed.yaml"


def _signed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["payload_sha256"] = sweep._payload_sha256(result)
    return result


def _terminal_qa(predicted: int) -> dict[str, object]:
    return _signed(
        {
            "schema_version": 1,
            "experiment_kind": sweep.EXPERIMENT_KIND,
            "status": "passed",
            "failures": [],
            "local_job_count": 9,
            "local_completed": 9,
            "external_reference_count": 3,
            "evaluation_cell_count": 12,
            "prediction_count": 12,
            "prediction_manifest_count": 12,
            "q4_access": "forbidden",
            "predicted_terminal_output_artifact_count": predicted,
        }
    )


def _materialize_completed_terminal_root(root: Path) -> list[Path]:
    (root / "registry").mkdir(parents=True)
    (root / "report").mkdir(parents=True)
    registry_path = root / "registry/jobs.json"
    experiment_status_path = root / "registry/experiment_status.json"
    task_registry_path = root / "task_registry.csv"
    resource_path = root / "resource_usage.csv"
    qa_path = root / "qa.json"
    report_path = root / "report/legacy_architecture_q3_conclusion.html"
    registry = {
        "experiment_kind": sweep.EXPERIMENT_KIND,
        "status": "completed_q3_exploratory_q4_forbidden",
        "jobs": [
            {"job_id": f"job_{index}", "job_spec_sha256": f"sha_{index}"}
            for index in range(9)
        ],
    }
    sweep._write_json(registry_path, registry)
    sweep._write_json(
        experiment_status_path,
        _signed(
            {
                "schema_version": 1,
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "completed_q3_exploratory_q4_forbidden",
            }
        ),
    )
    task_registry_path.write_text("job_id\n", encoding="utf-8")
    resource_path.write_text("job_id\n", encoding="utf-8")
    report_path.write_text("<html>complete</html>\n", encoding="utf-8")
    predicted = sweep._predicted_terminal_output_artifact_count(root)
    sweep._write_json(qa_path, _terminal_qa(predicted))
    snapshot_path = sweep._write_final_registry_snapshot(root, qa_path=qa_path)
    terminal_path = root / "terminal_output_sha256.csv"
    rows = [
        sweep._manifest_row(
            f"terminal_output:{path.relative_to(root).as_posix()}", path
        )
        for path in sweep._terminal_manifest_candidates(root)
    ]
    sweep._write_csv(terminal_path, rows, tuple(rows[0]))
    sweep._validate_terminal_output_manifest(root, repair_registry_anchor=True)
    return [
        resource_path,
        task_registry_path,
        registry_path,
        experiment_status_path,
        qa_path,
        snapshot_path,
        terminal_path,
        report_path,
    ]


class LegacyArchitectureContractTests(unittest.TestCase):
    def test_matrix_has_nine_local_and_three_frozen_reference_cells(self) -> None:
        self.assertEqual(len(sweep.all_specs()), 12)
        self.assertEqual(len(sweep.local_specs()), 9)
        references = [spec for spec in sweep.all_specs() if spec["external_reference"]]
        self.assertEqual(len(references), 3)
        self.assertEqual({int(spec["seed"]) for spec in references}, {42, 202, 404})
        self.assertTrue(
            all(
                (
                    spec["generator_conditioning_mode"],
                    spec["critic_conditioning_mode"],
                )
                == sweep.REFERENCE_CELL
                for spec in references
            )
        )

    def test_gpu_assignment_is_four_five_and_seed_paired(self) -> None:
        rows = sweep._assigned_specs()
        counts = [sum(int(row["gpu_id"]) == gpu for row in rows) for gpu in (0, 1)]
        self.assertEqual(sorted(counts), [4, 5])
        by_key = {
            (
                row["generator_conditioning_mode"],
                row["critic_conditioning_mode"],
                int(row["seed"]),
            ): int(row["gpu_id"])
            for row in rows
        }
        for seed in sweep.SEEDS:
            anchor_gpu = by_key[(sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[0], seed)]
            interaction_gpu = by_key[
                (sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[1], seed)
            ]
            film_gpu = by_key[(sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[0], seed)]
            self.assertNotEqual(anchor_gpu, interaction_gpu)
            self.assertEqual(interaction_gpu, film_gpu)
        expected_sequences = {
            (sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[0]): (0, 1, 0),
            (sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[1]): (1, 0, 1),
            (sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[0]): (1, 0, 1),
        }
        for cell, sequence in expected_sequences.items():
            self.assertEqual(
                tuple(by_key[(*cell, seed)] for seed in sweep.SEEDS), sequence
            )

    def test_four_actual_model_parameter_contracts(self) -> None:
        resolved = sweep.resolve_config(CONFIG)
        training = resolved["models"]["wgan"]["training"]
        grid = sweep.factorial._surface_grid_contract()
        for generator in sweep.GENERATOR_MODES:
            for critic in sweep.CRITIC_MODES:
                spec = {
                    "generator_conditioning_mode": generator,
                    "critic_conditioning_mode": critic,
                }
                contract = sweep._model_contract(resolved, spec)
                config = Config(
                    **{
                        **training,
                        "generator_conditioning_mode": generator,
                        "critic_conditioning_mode": critic,
                        "support_mask_mode": "raw_joint",
                        "cuda": False,
                    }
                )
                model = WGAN_GP(
                    config,
                    strike_grid=np.asarray(grid["strike_grid"]),
                    maturity_grid_days=np.asarray(grid["maturity_days_grid"]),
                    embedding_dim=1024,
                )
                generator_count = sum(
                    parameter.numel() for parameter in model.G.parameters()
                )
                critic_count = sum(
                    parameter.numel() for parameter in model.D.parameters()
                )
                self.assertEqual(
                    generator_count, contract["expected_generator_parameters"]
                )
                self.assertEqual(critic_count, contract["expected_critic_parameters"])
                self.assertEqual(
                    generator_count + critic_count,
                    contract["expected_wgan_parameters"],
                )
                self.assertEqual(
                    contract["architecture_profile_sha256"],
                    "5285d6c973d66b3b0fa78491bed99a7d707c68b02b8a4a6634ecbbea50c44124",
                )
                if (generator, critic) == sweep.REFERENCE_CELL:
                    self.assertEqual(
                        contract["model_contract_sha256"],
                        "55890dec55d5739030aa73a44a36d619aa85d571f951da6b59ba783e14887ce4",
                    )

    def test_training_payload_diff_from_formal_is_allowlisted(self) -> None:
        resolved = sweep.resolve_config(CONFIG)
        formal_root = Path(
            sweep._sweep(resolved)["external_reference"]["experiment_root"]
        )
        formal_config = yaml.safe_load(
            (
                formal_root / "configs/development_q3_capacity_selection/"
                "dev_film_nolp_legacy_lr_5e_07_seed_042_real_05m.yaml"
            ).read_text(encoding="utf-8")
        )
        allowed = {
            "data_path",
            "news_first_common_eval_data_path",
            "output_root",
            "generator_conditioning_mode",
            "critic_conditioning_mode",
            "news_first_model_contract_sha256",
        }
        for generator in sweep.GENERATOR_MODES:
            for critic in sweep.CRITIC_MODES:
                payload = sweep._training_payload(
                    resolved,
                    Path("/tmp/legacy_architecture_contract_only").resolve(),
                    {
                        "generator_conditioning_mode": generator,
                        "critic_conditioning_mode": critic,
                        "text_ablation_mode": "real_text",
                        "tolerance_minutes": 5,
                        "seed": 42,
                    },
                )
                self.assertEqual(set(payload), set(formal_config))
                differences = {
                    key for key in payload if payload[key] != formal_config[key]
                }
                self.assertLessEqual(differences, allowed)

    def test_reference_manifest_is_exact_and_not_latest_scanned(self) -> None:
        resolved = sweep.resolve_config(CONFIG)
        manifest = sweep._reference_manifest(resolved)
        self.assertEqual(manifest["reference_count"], 3)
        self.assertEqual(manifest["capacity_selection"]["point_leader"], "legacy")
        self.assertEqual(
            manifest["capacity_selection"]["candidate_statistical_support"],
            "descriptive_only",
        )
        for row in manifest["references"]:
            self.assertIn(f"seed_{int(row['seed']):03d}", row["source_job_id"])
            self.assertEqual(
                row["generator_conditioning_mode"], sweep.REFERENCE_CELL[0]
            )
            self.assertEqual(row["critic_conditioning_mode"], sweep.REFERENCE_CELL[1])
            self.assertNotIn("latest", row["source_run_dir"].lower())
            self.assertEqual(len(row["source_status_sha256"]), 64)
            self.assertEqual(
                int(row["source_gpu_id"]),
                dict(zip(sweep.SEEDS, (1, 0, 1)))[int(row["seed"])],
            )
            self.assertEqual(
                set(row["artifacts"]),
                {
                    "generator_best_learned",
                    "discriminator_best_learned",
                    "best_learned_checkpoint",
                    "training_metrics_csv",
                },
            )

    def test_prepare_writes_only_nine_training_configs_and_allows_pre_q4(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            sweep.prepare_experiment(CONFIG, root)
            registry = sweep._registry(root)
            self.assertEqual(len(registry["jobs"]), 9)
            self.assertEqual(len(list((root / "configs").glob("*.yaml"))), 9)
            self.assertTrue((root / "data_windows/tolerance_05m_pre_q4.xlsx").is_file())
            references = sweep._validate_references(root)
            self.assertEqual(len(references), 3)
            self.assertEqual(
                sum(Path(row["prediction"]["path"]).is_file() for row in references),
                3,
            )
            self.assertEqual(sweep._forbidden_q4_artifacts(root), [])
            forbidden = root / "data_windows/q4_common_05m.xlsx"
            forbidden.touch()
            self.assertEqual(sweep._forbidden_q4_artifacts(root), [forbidden])

    def test_completed_local_job_requires_exact_development_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = []
            for role in sweep.factorial.DEVELOPMENT_ARTIFACT_ROLES:
                path = root / f"{role}.bin"
                path.write_bytes(role.encode("utf-8"))
                artifacts.append(sweep._manifest_row(role, path))
            job = {"job_id": "job", "config_sha256": "config"}
            status = {
                "status": "completed",
                "config_sha256": "config",
                "artifacts": artifacts,
            }
            self.assertTrue(sweep._completed_local_job_valid(root, job, status))
            status["artifacts"] = artifacts[:-1]
            self.assertFalse(sweep._completed_local_job_valid(root, job, status))

    def test_terminal_manifest_is_registry_anchored_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "registry").mkdir()
            payload = root / "artifact.txt"
            payload.write_text("frozen", encoding="utf-8")
            registry = {
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "completed_q3_exploratory_q4_forbidden",
                "jobs": [
                    {"job_id": f"job_{index}", "job_spec_sha256": f"sha_{index}"}
                    for index in range(9)
                ],
            }
            sweep._write_json(root / "registry/jobs.json", registry)
            experiment_status = root / "registry/experiment_status.json"
            task_registry = root / "task_registry.csv"
            qa_path = root / "qa.json"
            sweep._write_json(
                experiment_status,
                _signed(
                    {
                        "schema_version": 1,
                        "experiment_kind": sweep.EXPERIMENT_KIND,
                        "status": "completed_q3_exploratory_q4_forbidden",
                    }
                ),
            )
            task_registry.write_text("job_id\n", encoding="utf-8")
            predicted = sweep._predicted_terminal_output_artifact_count(root)
            sweep._write_json(qa_path, _terminal_qa(predicted))
            sweep._write_final_registry_snapshot(root, qa_path=qa_path)
            # Simulate interruption after QA+snapshot but before terminal manifest.
            self.assertEqual(
                sweep._predicted_terminal_output_artifact_count(root), predicted
            )
            sweep._write_json(
                qa_path,
                _terminal_qa(sweep._predicted_terminal_output_artifact_count(root)),
            )
            sweep._write_final_registry_snapshot(root, qa_path=qa_path)
            manifest = root / "terminal_output_sha256.csv"
            rows = [
                sweep._manifest_row(
                    f"terminal_output:{path.relative_to(root).as_posix()}", path
                )
                for path in sweep._terminal_manifest_candidates(root)
            ]
            self.assertEqual(len(rows), predicted)
            sweep._write_csv(manifest, rows, tuple(rows[0]))
            with self.assertRaisesRegex(ValueError, "registry anchor drift"):
                sweep._validate_terminal_output_manifest(root)
            sweep._validate_terminal_output_manifest(root, repair_registry_anchor=True)
            sweep._validate_terminal_output_manifest(root)
            payload.write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Terminal output hash drift"):
                sweep._validate_terminal_output_manifest(root)

    def test_terminal_manifest_rejects_live_registry_job_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "registry").mkdir()
            registry = {
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "completed_q3_exploratory_q4_forbidden",
                "jobs": [
                    {"job_id": f"job_{index}", "job_spec_sha256": f"sha_{index}"}
                    for index in range(9)
                ],
            }
            sweep._write_json(root / "registry/jobs.json", registry)
            experiment_status = root / "registry/experiment_status.json"
            task_registry = root / "task_registry.csv"
            qa_path = root / "qa.json"
            sweep._write_json(
                experiment_status,
                _signed(
                    {
                        "schema_version": 1,
                        "experiment_kind": sweep.EXPERIMENT_KIND,
                        "status": "completed_q3_exploratory_q4_forbidden",
                    }
                ),
            )
            task_registry.write_text("job_id\n", encoding="utf-8")
            predicted = sweep._predicted_terminal_output_artifact_count(root)
            sweep._write_json(qa_path, _terminal_qa(predicted))
            sweep._write_final_registry_snapshot(root, qa_path=qa_path)
            rows = [
                sweep._manifest_row(
                    f"terminal_output:{path.relative_to(root).as_posix()}", path
                )
                for path in sweep._terminal_manifest_candidates(root)
            ]
            sweep._write_csv(root / "terminal_output_sha256.csv", rows, tuple(rows[0]))

            live = sweep._registry(root)
            live["jobs"][0]["job_spec_sha256"] = "tampered"
            sweep._write_json(root / "registry/jobs.json", live)
            with self.assertRaisesRegex(
                ValueError, "Live registry drift from frozen terminal snapshot"
            ):
                sweep._validate_terminal_output_manifest(
                    root, repair_registry_anchor=True
                )

    def test_terminal_manifest_rejects_semantically_tampered_qa(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "registry").mkdir()
            registry = {
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "completed_q3_exploratory_q4_forbidden",
                "jobs": [
                    {"job_id": f"job_{index}", "job_spec_sha256": f"sha_{index}"}
                    for index in range(9)
                ],
            }
            sweep._write_json(root / "registry/jobs.json", registry)
            sweep._write_json(
                root / "registry/experiment_status.json",
                _signed(
                    {
                        "schema_version": 1,
                        "experiment_kind": sweep.EXPERIMENT_KIND,
                        "status": "completed_q3_exploratory_q4_forbidden",
                    }
                ),
            )
            (root / "task_registry.csv").write_text("job_id\n", encoding="utf-8")
            predicted = sweep._predicted_terminal_output_artifact_count(root)
            qa_path = root / "qa.json"
            sweep._write_json(qa_path, _terminal_qa(predicted))
            snapshot_path = sweep._write_final_registry_snapshot(root, qa_path=qa_path)

            qa = sweep._read_json(qa_path)
            qa["status"] = "failed"
            qa["failures"] = ["tampered"]
            qa["payload_sha256"] = sweep._payload_sha256(
                {key: value for key, value in qa.items() if key != "payload_sha256"}
            )
            sweep._write_json(qa_path, qa)
            snapshot = sweep._read_json(snapshot_path)
            snapshot["qa_sha256"] = sweep._sha256_file(qa_path)
            snapshot["payload_sha256"] = sweep._payload_sha256(
                {
                    key: value
                    for key, value in snapshot.items()
                    if key != "payload_sha256"
                }
            )
            sweep._write_json(snapshot_path, snapshot)
            rows = [
                sweep._manifest_row(
                    f"terminal_output:{path.relative_to(root).as_posix()}", path
                )
                for path in sweep._terminal_manifest_candidates(root)
            ]
            sweep._write_csv(root / "terminal_output_sha256.csv", rows, tuple(rows[0]))
            with self.assertRaisesRegex(ValueError, "Terminal QA payload/count drift"):
                sweep._validate_terminal_output_manifest(
                    root, repair_registry_anchor=True
                )

    def test_cli_exposes_safe_actions_and_dispatches(self) -> None:
        command = "train-news-first-vol-legacy-architecture-seed-sweep"
        parser = rq3_main.build_parser()
        self.assertEqual(parser.parse_args([command]).action, "prepare")
        actions = (
            "prepare",
            "dry-run",
            "launch",
            "worker",
            "analyze",
            "postprocess",
            "qa",
            "status",
            "run-pipeline",
        )
        for action in actions:
            self.assertEqual(parser.parse_args([command, action]).action, action)
        expected = Path("/tmp/legacy-architecture-dispatch")
        with patch.object(
            sweep,
            "run_news_first_vol_legacy_architecture_seed",
            return_value=expected,
        ) as mocked:
            observed = rq3_main.main([command, "qa", "--output-dir", str(expected)])
        self.assertEqual(observed, expected)
        self.assertEqual(mocked.call_args.kwargs["action"], "qa")

    def test_completed_run_pipeline_resume_is_strictly_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            critical_paths = _materialize_completed_terminal_root(root)
            before = {
                path: (path.stat().st_mtime_ns, sweep._sha256_file(path))
                for path in critical_paths
            }
            with (
                patch.object(sweep, "_validate_root") as validate_root,
                patch.object(
                    sweep,
                    "_launch",
                    side_effect=AssertionError("completed pipeline must not launch"),
                ),
            ):
                observed = sweep.run_pipeline(CONFIG, root, resume=True)
            self.assertEqual(
                observed, root / "report/legacy_architecture_q3_conclusion.html"
            )
            validate_root.assert_called_once_with(root.resolve())
            after = {
                path: (path.stat().st_mtime_ns, sweep._sha256_file(path))
                for path in critical_paths
            }
            self.assertEqual(after, before)


class LegacyArchitectureStatisticsTests(unittest.TestCase):
    @staticmethod
    def _synthetic_pairs() -> pd.DataFrame:
        rows = []
        cell_effects = {
            "gconcat_dlp": 0.0,
            "gconcat_dnolp": -0.20,
            "gfilm_dlp": 0.10,
            "gfilm_dnolp": -0.30,
        }
        modes = {
            "gconcat_dlp": (sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[0]),
            "gconcat_dnolp": (sweep.GENERATOR_MODES[0], sweep.CRITIC_MODES[1]),
            "gfilm_dlp": (sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[0]),
            "gfilm_dnolp": (sweep.GENERATOR_MODES[1], sweep.CRITIC_MODES[1]),
        }
        for cell, effect in cell_effects.items():
            generator, critic = modes[cell]
            for seed in sweep.SEEDS:
                for session in range(10):
                    for pair in range(3):
                        rows.append(
                            {
                                "architecture_cell": cell,
                                "generator_conditioning_mode": generator,
                                "critic_conditioning_mode": critic,
                                "seed": seed,
                                "session_id": f"s{session}",
                                "pair_id": f"s{session}_p{pair}",
                                "model_mae": 2.0 + effect,
                                "persistence_mae": 2.5,
                            }
                        )
        return pd.DataFrame(rows)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        frame = self._synthetic_pairs()
        anchor = frame[frame["architecture_cell"].eq("gconcat_dlp")]
        candidate = frame[frame["architecture_cell"].eq("gconcat_dnolp")]
        keys = ["seed", "session_id", "pair_id"]
        paired = candidate[keys + ["model_mae"]].merge(
            anchor[keys + ["model_mae"]],
            on=keys,
            suffixes=("_candidate", "_anchor"),
            validate="one_to_one",
        )
        paired["difference"] = (
            paired["model_mae_candidate"] - paired["model_mae_anchor"]
        )
        left = analysis._two_level_paired_bootstrap(paired, replicates=500, seed=7)
        right = analysis._two_level_paired_bootstrap(paired, replicates=500, seed=7)
        self.assertEqual(left, right)
        self.assertLess(left["ci_95_upper"], 0)

    def test_holm_three_candidate_anchor_family_and_gate(self) -> None:
        with patch.object(analysis, "BOOTSTRAP_REPLICATES", 500):
            scores, contrasts, selection = analysis._score_and_contrasts(
                self._synthetic_pairs()
            )
            persistence = analysis._persistence_contrasts(self._synthetic_pairs())
            factorial = analysis._secondary_factorial_effects(self._synthetic_pairs())
        self.assertEqual(len(scores), 4)
        self.assertEqual(len(contrasts), 3)
        self.assertEqual(selection["holm_family_size"], 3)
        supported = set(selection["statistically_supported_candidates"])
        self.assertIn("gconcat_dnolp", supported)
        self.assertIn("gfilm_dnolp", supported)
        self.assertNotIn("gfilm_dlp", supported)
        self.assertFalse(selection["q4_read"])
        self.assertEqual(len(persistence), 4)
        self.assertTrue(
            persistence["inference_role"]
            .eq("secondary_cell_vs_persistence_holm4")
            .all()
        )
        self.assertEqual(len(factorial), 3)
        self.assertTrue(
            factorial["inference_role"].eq("secondary_factorial_effect_holm3").all()
        )
        effects = factorial.set_index("factorial_effect")["mean_difference"]
        self.assertAlmostEqual(
            effects["generator_main_film_minus_concat"], 0.0, places=12
        )
        self.assertAlmostEqual(effects["critic_main_nolp_minus_lp"], -0.3, places=12)
        self.assertAlmostEqual(
            effects["generator_by_critic_interaction"], -0.2, places=12
        )


if __name__ == "__main__":
    unittest.main()
