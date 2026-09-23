from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.rq3 import news_first_vol_film_nolp_capacity_seed as sweep
from scripts.rq3 import news_first_vol_film_nolp_capacity_seed_analysis as analysis
from scripts.rq3.main import build_parser
from wgan_option.config import Config
from wgan_option.models.gan_model import WGAN_GP


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rq3/news_first_vol_film_nolp_capacity_seed.yaml"


def _recovery_fixture(
    root: Path, *, extra_code_drift: bool = False
) -> tuple[dict[str, str], str, str]:
    (root / "registry").mkdir(parents=True)
    source = root / "fixture_source.bin"
    config = root / "fixture_config.bin"
    extra_code = root / "fixture_code.py"
    sentinel = root / "runs" / "immutable_training_sentinel.bin"
    sentinel.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    config.write_bytes(b"config")
    extra_code.write_bytes(b"code")
    sentinel.write_bytes(b"trained")
    sweep._write_csv(
        root / "source_hashes.csv",
        [sweep._manifest_row("source", source)],
        ("artifact_role", "path", "size_bytes", "sha256"),
    )
    sweep._write_csv(
        root / "config_hashes.csv",
        [sweep._manifest_row("config", config)],
        ("artifact_role", "path", "size_bytes", "sha256"),
    )
    extra_row = sweep._manifest_row("code:fixture_code.py", extra_code)
    if extra_code_drift:
        extra_row["sha256"] = "f" * 64
    code_rows = [
        {
            "artifact_role": (
                "code:scripts/rq3/news_first_vol_film_nolp_capacity_seed.py"
            ),
            "path": str(Path(sweep.__file__).resolve()),
            "size_bytes": sweep.Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SIZE,
            "sha256": sweep.Q4_RECOVERY_ORIGINAL_ORCHESTRATOR_SHA256,
        },
        extra_row,
    ]
    sweep._write_csv(
        root / "code_hashes.csv",
        code_rows,
        ("artifact_role", "path", "size_bytes", "sha256"),
    )
    registry = {
        "schema_version": 1,
        "experiment_kind": sweep.EXPERIMENT_KIND,
        "status": "refit_complete_q4_locked",
        "selection_frozen": True,
        "refit_complete": True,
        "q4_gate_open": True,
        "q4_window_materialized": False,
        "q4_loader_created": False,
        "q4_predictions_generated": False,
        "q4_evaluated": False,
        "jobs": [{"job_id": f"job_{index:02d}"} for index in range(72)],
    }
    sweep._write_json(root / "registry" / "jobs.json", registry)
    sweep._write_experiment_status(
        root,
        "failed",
        current_stage="q4",
        error=sweep.Q4_RECOVERY_FAILURE,
    )
    manifests = {
        name: sweep._sha256_file(root / name)
        for name in ("source_hashes.csv", "code_hashes.csv", "config_hashes.csv")
    }
    return (
        manifests,
        sweep._sha256_file(root / "registry" / "jobs.json"),
        sweep._sha256_file(root / "registry" / "experiment_status.json"),
    )


class FilmNoLPCapacitySeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resolved = sweep.resolve_config(CONFIG)

    def test_frozen_matrix_and_cli(self) -> None:
        specs = sweep.experiment_specs(sweep.DEVELOPMENT_STAGE)
        self.assertEqual(len(specs), 36)
        self.assertEqual(
            {row["capacity_profile"] for row in specs}, set(sweep.PROFILES)
        )
        self.assertEqual({row["seed"] for row in specs}, set(sweep.SEEDS))
        self.assertEqual({row["tolerance_minutes"] for row in specs}, {5, 30})
        args = build_parser().parse_args(
            ["train-news-first-vol-film-nolp-capacity-seed-sweep", "status"]
        )
        self.assertEqual(args.action, "status")

    def test_gpu_assignment_is_balanced_and_refit_reversed(self) -> None:
        development = sweep._assign(
            sweep.experiment_specs(sweep.DEVELOPMENT_STAGE),
            slots_per_gpu=18,
            invert=False,
        )
        refit = sweep._assign(
            sweep.experiment_specs(sweep.REFIT_STAGE),
            slots_per_gpu=18,
            invert=True,
        )
        left = {
            (row["capacity_profile"], row["seed"], row["tolerance_minutes"]): row[
                "gpu_id"
            ]
            for row in development
        }
        right = {
            (row["capacity_profile"], row["seed"], row["tolerance_minutes"]): row[
                "gpu_id"
            ]
            for row in refit
        }
        self.assertTrue(all(right[key] == 1 - gpu for key, gpu in left.items()))

    def test_all_six_parameter_contracts_are_real(self) -> None:
        expected = {
            "micro": (21164, 4756, 25920),
            "tiny": (48828, 10651, 59479),
            "small": (124424, 25861, 150285),
            "medium": (356688, 69961, 426649),
            "large": (697048, 132301, 829349),
            "legacy": (4020288, 729157, 4749445),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data_windows/pre_q4").mkdir(parents=True)
            for tolerance in sweep.TOLERANCES:
                source = (
                    Path(self.resolved["datasets"]["root"])
                    / f"tolerance_{tolerance:02d}m/merged_vol.xlsx"
                )
                (
                    root / f"data_windows/pre_q4/tolerance_{tolerance:02d}m_pre_q4.xlsx"
                ).symlink_to(source)
            for profile in sweep.PROFILES:
                spec = next(
                    row
                    for row in sweep.experiment_specs(sweep.DEVELOPMENT_STAGE)
                    if row["capacity_profile"] == profile
                )
                payload = sweep._training_payload(
                    self.resolved, root, spec=spec, stage=sweep.DEVELOPMENT_STAGE
                )
                config = Config(**payload)
                model = WGAN_GP(
                    config,
                    np.asarray(
                        self.resolved["film_nolp_capacity_seed_sweep"]["strike_grid"]
                    ),
                    np.asarray(
                        self.resolved["film_nolp_capacity_seed_sweep"][
                            "maturity_days_grid"
                        ]
                    ),
                    1024,
                )
                generator = sum(parameter.numel() for parameter in model.G.parameters())
                critic = sum(parameter.numel() for parameter in model.D.parameters())
                self.assertEqual(
                    (generator, critic, generator + critic), expected[profile]
                )

    def test_nolp_critic_ignores_embedding_without_removing_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data_windows/pre_q4").mkdir(parents=True)
            for tolerance in sweep.TOLERANCES:
                source = (
                    Path(self.resolved["datasets"]["root"])
                    / f"tolerance_{tolerance:02d}m/merged_vol.xlsx"
                )
                (
                    root / f"data_windows/pre_q4/tolerance_{tolerance:02d}m_pre_q4.xlsx"
                ).symlink_to(source)
            spec = sweep.experiment_specs(sweep.DEVELOPMENT_STAGE)[0]
            config = Config(
                **sweep._training_payload(
                    self.resolved, root, spec=spec, stage=sweep.DEVELOPMENT_STAGE
                )
            )
            model = WGAN_GP(
                config,
                np.linspace(0.97, 1.03, 16),
                np.asarray([1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38]),
                1024,
            )
            model.D.eval()
            device = next(model.D.parameters()).device
            current = torch.rand(2, 1, 16, 16, device=device)
            future = torch.rand(2, 1, 16, 16, device=device)
            left = model.D(current, future, torch.zeros(2, 1024, device=device))
            right = model.D(current, future, torch.randn(2, 1024, device=device))
            self.assertTrue(torch.equal(left, right))

    def test_two_level_bootstrap_preserves_three_seeds(self) -> None:
        rows = []
        for seed in sweep.SEEDS:
            for session in range(3):
                for pair in range(2):
                    rows.append(
                        {
                            "seed": seed,
                            "session_id": f"s{session}",
                            "pair_id": f"p{session}_{pair}",
                            "difference": -1e-5,
                        }
                    )
        result = analysis._two_level(
            pd.DataFrame(rows), value_column="difference", seed=20260822
        )
        self.assertLess(result["ci_95_upper"], 0.0)

    def test_grid_contract_is_exact_ttm(self) -> None:
        contract = sweep._grid_contract(self.resolved)
        self.assertEqual(
            contract["surface_grid_sha256"],
            "7b72f2d3d12a55999415351ba071f8d87863ed57445f1bbf03543e39f9f186b8",
        )

    def test_capacity_q4_materializer_uses_own_kind_and_common_5m_only(self) -> None:
        resolved = deepcopy(self.resolved)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "datasets" / "tolerance_05m" / "merged_vol.xlsx"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"source-5m")
            resolved["datasets"]["root"] = str(root / "datasets")
            pair_ids = [f"p{index}" for index in range(143)] + [
                f"p{index}" for index in range(24)
            ]
            frame = pd.DataFrame(
                {
                    "effective_origin_utc": ["2023-10-02T12:00:00Z"] * 167,
                    "pair_id": pair_ids,
                    "session_id": [f"s{index % 45}" for index in range(167)],
                }
            )
            (root / "registry").mkdir()
            sweep._write_json(
                root / "registry" / "jobs.json",
                {
                    "experiment_kind": sweep.EXPERIMENT_KIND,
                    "selection_frozen": True,
                    "refit_complete": True,
                    "q4_gate_open": True,
                    "jobs": [],
                },
            )
            with (
                patch.object(sweep, "_validate_frozen_selection"),
                patch.object(sweep, "_validate_allowlist"),
                patch.object(sweep.factorial, "_supported_frame", return_value=frame),
                patch.object(
                    sweep.factorial,
                    "_materialize_q4_common_window",
                    side_effect=AssertionError("factorial helper must not be used"),
                ),
                patch("pandas.read_excel", return_value=frame) as read_excel,
            ):
                workbook, manifest_path = sweep._materialize_q4_common_window(
                    resolved, root, resume=False
                )
            self.assertTrue(workbook.is_file())
            manifest = sweep._read_json(manifest_path)
            self.assertEqual(manifest["experiment_kind"], sweep.EXPERIMENT_KIND)
            self.assertEqual(
                manifest["supported_counts"], sweep.EXPECTED_Q4_COMMON_COUNTS
            )
            self.assertEqual(manifest["broad_30m_q4_rows_read"], 0)
            self.assertIn("tolerance_05m", str(read_excel.call_args.args[0]))
            self.assertNotIn("tolerance_30m", str(read_excel.call_args.args[0]))

    def test_capacity_q4_materializer_reads_zero_rows_before_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            sweep._write_json(
                root / "registry" / "jobs.json",
                {
                    "experiment_kind": sweep.EXPERIMENT_KIND,
                    "selection_frozen": True,
                    "refit_complete": True,
                    "q4_gate_open": False,
                    "jobs": [],
                },
            )
            with patch("pandas.read_excel") as read_excel:
                with self.assertRaisesRegex(RuntimeError, "explicit open gate"):
                    sweep._materialize_q4_common_window(
                        self.resolved, root, resume=False
                    )
            read_excel.assert_not_called()
            self.assertFalse((root / "data_windows" / "q4").exists())

    def test_known_q4_failure_recovery_is_anchored_and_does_not_touch_training(
        self,
    ) -> None:
        evidence = {
            "completed_job_status_count": 72,
            "required_training_artifact_count": 756,
            "checkpoint_artifact_count": 360,
            "q4_allowlist_row_count": 72,
            "q3_prediction_count": 36,
            "q3_prediction_manifest_count": 36,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchors, registry_sha, status_sha = _recovery_fixture(root)
            original_manifests = {name: (root / name).read_bytes() for name in anchors}
            sentinel = root / "runs" / "immutable_training_sentinel.bin"
            sentinel_sha = sweep._sha256_file(sentinel)
            with (
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_MANIFEST_SHA256", anchors),
                patch.object(
                    sweep, "Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256", registry_sha
                ),
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_STATUS_SHA256", status_sha),
                patch.object(
                    sweep, "_recovery_immutable_evidence", return_value=evidence
                ) as immutable,
            ):
                ledger_path = sweep._register_q4_recovery(root)
                sweep._validate_recovery_bundle(root, require_registered=True)
            registry = sweep._read_json(root / "registry" / "jobs.json")
            ledger = sweep._read_json(ledger_path)
            self.assertTrue(registry["q4_recovery_applied"])
            self.assertFalse(registry["q4_gate_open"])
            self.assertEqual(ledger["immutable_evidence"], evidence)
            self.assertEqual(len(ledger["approved_code_changes"]), 1)
            self.assertEqual(ledger["q4_rows_read_before_recovery"], 0)
            self.assertFalse(ledger["training_jobs_rewritten"])
            self.assertEqual(sweep._sha256_file(sentinel), sentinel_sha)
            self.assertGreaterEqual(immutable.call_count, 3)
            for name, content in original_manifests.items():
                self.assertEqual((root / name).read_bytes(), content)
                anchor = ledger["anchored_manifests"][name]
                self.assertEqual(anchor["size_bytes"], (root / name).stat().st_size)
                self.assertEqual(anchor["sha256"], anchors[name])

    def test_q4_recovery_rejects_any_second_code_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchors, registry_sha, status_sha = _recovery_fixture(
                root, extra_code_drift=True
            )
            with (
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_MANIFEST_SHA256", anchors),
                patch.object(
                    sweep, "Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256", registry_sha
                ),
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_STATUS_SHA256", status_sha),
            ):
                with self.assertRaisesRegex(ValueError, "Unapproved recovery code"):
                    sweep._register_q4_recovery(root)
            self.assertFalse(sweep._recovery_directory(root).exists())

    def test_q4_recovery_rejects_pre_recovery_registry_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchors, registry_sha, status_sha = _recovery_fixture(root)
            registry = sweep._read_json(root / "registry" / "jobs.json")
            registry["unapproved_extra_field"] = True
            sweep._write_json(root / "registry" / "jobs.json", registry)
            with (
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_MANIFEST_SHA256", anchors),
                patch.object(
                    sweep, "Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256", registry_sha
                ),
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_STATUS_SHA256", status_sha),
            ):
                with self.assertRaisesRegex(ValueError, "registry/status anchor"):
                    sweep._register_q4_recovery(root)
            self.assertFalse(sweep._recovery_directory(root).exists())

    def test_recovery_registration_retry_repairs_interrupted_status_write(self) -> None:
        evidence = {"completed_job_status_count": 72}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchors, registry_sha, status_sha = _recovery_fixture(root)
            with (
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_MANIFEST_SHA256", anchors),
                patch.object(
                    sweep, "Q4_RECOVERY_ORIGINAL_REGISTRY_SHA256", registry_sha
                ),
                patch.object(sweep, "Q4_RECOVERY_ORIGINAL_STATUS_SHA256", status_sha),
                patch.object(
                    sweep, "_recovery_immutable_evidence", return_value=evidence
                ),
            ):
                sweep._register_q4_recovery(root)
                sweep._write_experiment_status(
                    root,
                    "failed",
                    current_stage="q4",
                    error=sweep.Q4_RECOVERY_FAILURE,
                )
                sweep._register_q4_recovery(root)
            status = sweep._read_json(root / "registry" / "experiment_status.json")
            self.assertEqual(status["status"], "refit_complete_q4_recovered_locked")
            self.assertEqual(status["current_stage"], "q4_recovery")

    def test_resource_summary_accepts_canonical_real_telemetry_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "resource_usage.csv").write_text(
                (
                    "timestamp_utc,wave,gpu_index,gpu_uuid,gpu_name,"
                    "utilization_gpu_pct,utilization_memory_pct,memory_used_mib,"
                    "memory_total_mib,power_draw_w,temperature_c,sample_status,error\n"
                    "2026-08-21T15:40:09Z,1,0,u0,NVIDIA A30,80,1,100,"
                    "24576,30,33,ok,\n"
                    "2026-08-21T15:40:10Z,1,0,u0,NVIDIA A30,100,1,120,"
                    "24576,30,33,ok,\n"
                    "2026-08-21T15:40:09Z,1,1,u1,NVIDIA A30,50,1,90,"
                    "24576,30,33,ok,\n"
                ),
                encoding="utf-8",
            )
            path = sweep._resource_summary(root)
            result = pd.read_csv(path)
            self.assertEqual(result["sample_count"].tolist(), [2, 1])
            self.assertEqual(result["peak_memory_mib"].tolist(), [120.0, 90.0])
            self.assertEqual(result["mean_utilization_percent"].tolist(), [90.0, 50.0])
            self.assertEqual(
                result.columns.tolist(),
                [
                    "gpu_id",
                    "sample_count",
                    "peak_memory_mib",
                    "mean_utilization_percent",
                ],
            )

    def test_resource_summary_accepts_explicit_legacy_alias_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "resource_usage.csv").write_text(
                (
                    "gpu_index,memory_used_mib,utilization_gpu_percent,sample_status\n"
                    "0,100,70,ok\n"
                    "1,200,90,ok\n"
                ),
                encoding="utf-8",
            )
            result = pd.read_csv(sweep._resource_summary(root))
            self.assertEqual(result["mean_utilization_percent"].tolist(), [70.0, 90.0])

    def test_resource_summary_rejects_mixed_or_missing_utilization_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usage = root / "resource_usage.csv"
            usage.write_text(
                (
                    "gpu_index,memory_used_mib,utilization_gpu_pct,"
                    "utilization_gpu_percent,sample_status\n"
                    "0,100,70,70,ok\n1,100,70,70,ok\n"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                sweep._resource_summary(root)
            usage.write_text(
                "gpu_index,memory_used_mib,sample_status\n0,100,ok\n1,100,ok\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                sweep._resource_summary(root)

    def test_single_code_transition_rejects_any_extra_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.py"
            extra = root / "extra.py"
            target.write_bytes(b"old-target")
            extra.write_bytes(b"stable")
            previous = [
                sweep._manifest_row("code:target", target),
                sweep._manifest_row("code:extra", extra),
            ]
            previous_size = target.stat().st_size
            previous_sha = sweep._sha256_file(target)
            target.write_bytes(b"new-target")
            rows, change = sweep._single_code_transition(
                previous,
                target_role="code:target",
                target_path=target,
                previous_size_bytes=previous_size,
                previous_sha256=previous_sha,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(change["previous_sha256"], previous_sha)
            extra.write_bytes(b"drifted")
            with self.assertRaisesRegex(ValueError, "Unapproved postprocess"):
                sweep._single_code_transition(
                    previous,
                    target_role="code:target",
                    target_path=target,
                    previous_size_bytes=previous_size,
                    previous_sha256=previous_sha,
                )

    def test_partial_postprocess_recovery_is_atomic_idempotent_and_tamper_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry/q4_recovery_v1").mkdir(parents=True)
            for name in (
                sweep.Q4_RECOVERY_LEDGER,
                sweep.Q4_RECOVERY_CURRENT_CODE_HASHES,
                sweep.Q4_RECOVERY_PRE_OUTPUT_HASHES,
            ):
                (root / "registry/q4_recovery_v1" / name).write_bytes(
                    f"v1-{name}".encode()
                )
            (root / "report").mkdir()
            report_paths = {
                "report/film_nolp_capacity_conclusion.md": b"markdown",
                "report/film_nolp_capacity_conclusion.html": b"html",
                "resource_usage.csv": b"telemetry",
                "analysis/film_nolp_capacity_q4_summary.json": b"summary",
                "data_windows/q4/q4_window_manifest.json": b"window",
                "data_windows/q4/common_05m_q4.xlsx": b"workbook",
            }
            for relative, content in report_paths.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            registry = {
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "q4_evaluated",
                "selection_frozen": True,
                "refit_complete": True,
                "q4_window_materialized": True,
                "q4_loader_created": True,
                "q4_predictions_generated": True,
                "q4_evaluated": True,
                "jobs": [],
            }
            sweep._write_json(root / "registry/jobs.json", registry)
            sweep._write_experiment_status(root, "q4_evaluated", current_stage="q4")
            registry_sha = sweep._sha256_file(root / "registry/jobs.json")
            status_sha = sweep._sha256_file(root / "registry/experiment_status.json")
            v1_anchors = {
                name: sweep._sha256_file(root / "registry/q4_recovery_v1" / name)
                for name in (
                    sweep.Q4_RECOVERY_LEDGER,
                    sweep.Q4_RECOVERY_CURRENT_CODE_HASHES,
                    sweep.Q4_RECOVERY_PRE_OUTPUT_HASHES,
                )
            }
            partial_anchors = {
                relative: sweep._sha256_file(root / relative)
                for relative in report_paths
            }
            fake_code = root / "fixed_orchestrator.py"
            fake_code.write_bytes(b"fixed")
            current_rows = [sweep._manifest_row("code:orchestrator", fake_code)]
            approved_change = {
                "artifact_role": "code:orchestrator",
                "path": str(fake_code.resolve()),
                "previous_size_bytes": 3,
                "previous_sha256": "a" * 64,
                "recovery_size_bytes": fake_code.stat().st_size,
                "recovery_sha256": sweep._sha256_file(fake_code),
            }
            evidence = {
                "completed_job_status_count": 72,
                "q4_prediction_cache_count": 36,
            }
            with (
                patch.object(sweep, "POSTPROCESS_RECOVERY_V1_ANCHORS", v1_anchors),
                patch.object(
                    sweep,
                    "POSTPROCESS_RECOVERY_PRE_REGISTRY_SHA256",
                    registry_sha,
                ),
                patch.object(
                    sweep, "POSTPROCESS_RECOVERY_PRE_STATUS_SHA256", status_sha
                ),
                patch.object(
                    sweep,
                    "POSTPROCESS_RECOVERY_PARTIAL_FILE_SHA256",
                    partial_anchors,
                ),
                patch.object(sweep, "_validate_formal_q4_recovery_v1_anchor"),
                patch.object(
                    sweep,
                    "_postprocess_recovery_code_candidate",
                    return_value=(current_rows, approved_change),
                ),
                patch.object(
                    sweep, "_postprocess_immutable_evidence", return_value=evidence
                ),
            ):
                ledger = sweep._register_postprocess_recovery(root)
                first = ledger.read_bytes()
                sweep._register_postprocess_recovery(root)
                self.assertEqual(ledger.read_bytes(), first)
                sweep._validate_postprocess_recovery_bundle(
                    root, require_registered=True
                )
                manifest = (
                    root
                    / sweep.POSTPROCESS_RECOVERY_DIRECTORY
                    / sweep.POSTPROCESS_RECOVERY_CURRENT_CODE_HASHES
                )
                manifest.write_bytes(manifest.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "manifest anchor"):
                    sweep._validate_postprocess_recovery_bundle(
                        root, require_registered=True
                    )

    def test_postprocess_recovery_rejects_interrupted_temporary_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = root / "registry/.postprocess_recovery_v2.123.tmp"
            stale.mkdir(parents=True)
            (stale / "partial.csv").write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit inspection/removal"):
                sweep._validate_known_postprocess_recovery_candidate(root)

    def test_final_snapshot_allows_only_terminal_manifest_registry_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            registry = {
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "completed",
                "q4_evaluated": True,
                "jobs": [],
            }
            sweep._write_json(root / "registry/jobs.json", registry)
            sweep._write_experiment_status(root, "completed", current_stage="terminal")
            snapshot = {
                "schema_version": 1,
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "registry": registry,
                "experiment_status": sweep._read_json(
                    root / "registry/experiment_status.json"
                ),
                "created_at_utc": "2026-08-22T00:00:00Z",
            }
            snapshot["payload_sha256"] = sweep._payload_sha256(snapshot)
            sweep._write_json(root / "registry/final_registry_snapshot.json", snapshot)
            live = dict(registry)
            live.update(
                terminal_output_manifest_path=str(
                    (root / "output_hashes.csv").resolve()
                ),
                terminal_output_manifest_sha256="a" * 64,
                terminal_output_artifact_count=1,
            )
            sweep._write_json(root / "registry/jobs.json", live)
            sweep._validate_final_registry_snapshot(root)
            live["unapproved_after_snapshot"] = True
            sweep._write_json(root / "registry/jobs.json", live)
            with self.assertRaisesRegex(ValueError, "snapshot contract drift"):
                sweep._validate_final_registry_snapshot(root)

    def test_resume_repairs_only_manifest_before_registry_anchor_gap(self) -> None:
        def build_unanchored(root: Path) -> None:
            (root / "registry").mkdir()
            registry = {
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "status": "completed",
                "q4_evaluated": True,
                "jobs": [],
            }
            sweep._write_json(root / "registry/jobs.json", registry)
            sweep._write_experiment_status(root, "completed", current_stage="terminal")
            (root / "artifact.bin").write_bytes(b"immutable")
            snapshot = {
                "schema_version": 1,
                "experiment_kind": sweep.EXPERIMENT_KIND,
                "registry": registry,
                "experiment_status": sweep._read_json(
                    root / "registry/experiment_status.json"
                ),
                "created_at_utc": "2026-08-22T00:00:00Z",
            }
            snapshot["payload_sha256"] = sweep._payload_sha256(snapshot)
            sweep._write_json(root / "registry/final_registry_snapshot.json", snapshot)
            qa_path = root / "qa.json"
            qa = {
                "schema_version": 1,
                "status": "passed",
                "terminal_output_artifact_count": len(sweep._all_local_files(root)) + 1,
            }
            qa["payload_sha256"] = sweep._payload_sha256(qa)
            sweep._write_json(qa_path, qa)
            files = sweep._all_local_files(root)
            rows = [
                {
                    "artifact_role": f"output:{path.relative_to(root).as_posix()}",
                    "relative_path": path.relative_to(root).as_posix(),
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sweep._sha256_file(path),
                }
                for path in files
            ]
            sweep._write_csv(root / "output_hashes.csv", rows, tuple(rows[0]))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_unanchored(root)
            qa_path = root / "qa.json"
            with (
                patch.object(sweep, "_validate_root"),
                patch.object(sweep, "qa_experiment", return_value=qa_path),
            ):
                observed = sweep.postprocess(root, resume=True)
            self.assertEqual(observed, qa_path)
            registry = sweep._read_json(root / "registry/jobs.json")
            self.assertEqual(
                registry["terminal_output_manifest_sha256"],
                sweep._sha256_file(root / "output_hashes.csv"),
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_unanchored(root)
            registry = sweep._read_json(root / "registry/jobs.json")
            registry["unapproved_after_snapshot"] = True
            sweep._write_json(root / "registry/jobs.json", registry)
            before = (root / "registry/jobs.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "snapshot contract drift"):
                sweep._repair_unanchored_terminal_manifest(root)
            self.assertEqual((root / "registry/jobs.json").read_bytes(), before)

    def test_terminal_manifest_requires_live_registry_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "output_hashes.csv"
            manifest.write_text("artifact_role,relative_path,path,size_bytes,sha256\n")
            registry = {
                "status": "completed",
                "terminal_output_manifest_path": str(manifest.resolve()),
                "terminal_output_manifest_sha256": sweep._sha256_file(manifest),
                "terminal_output_artifact_count": 0,
            }
            sweep._validate_terminal_manifest_registry_anchor(
                root, registry, manifest, 0
            )
            manifest.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registry anchor"):
                sweep._validate_terminal_manifest_registry_anchor(
                    root, registry, manifest, 0
                )

    def test_completed_postprocess_retry_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "output_hashes.csv"
            manifest.write_text("terminal\n", encoding="utf-8")
            before = manifest.read_bytes()
            expected = root / "qa.json"
            completed_registry = {
                "status": "completed",
                "terminal_output_manifest_path": str(manifest.resolve()),
                "terminal_output_manifest_sha256": sweep._sha256_file(manifest),
                "terminal_output_artifact_count": 0,
            }
            with (
                patch.object(sweep, "_validate_root"),
                patch.object(sweep, "_registry", return_value=completed_registry),
                patch.object(sweep, "qa_experiment", return_value=expected) as qa,
                patch(
                    "scripts.rq3.news_first_vol_film_nolp_capacity_seed_report.render_report"
                ) as render,
            ):
                observed = sweep.postprocess(root)
            self.assertEqual(observed, expected)
            self.assertEqual(manifest.read_bytes(), before)
            qa.assert_called_once_with(root.resolve())
            render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
