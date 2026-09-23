"""Tests for isolated multi-GPU text-effect prediction scheduling."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction as parallel,
)


PROFILE_A = "a" * 64
PROFILE_B = "b" * 64
WORKER = (
    "tests.test_scripts."
    "test_rq3_news_first_vol_film_unet_pure_cnn_backbone_text_effect_parallel_prediction:"
    "fixture_prediction_worker"
)


def fixture_prediction_worker(
    *, unit: dict[str, object], logical_device: int, resume: bool
) -> dict[str, object]:
    """Spawn-safe fixture with the same contract expected from a core wrapper."""

    del resume
    if bool(unit.get("fail")):
        raise RuntimeError("intentional evaluator failure")
    path = Path(str(unit["artifact_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prediction_unit_id": unit["prediction_unit_id"],
                "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": logical_device,
                "noise": unit["noise_bank_profile_sha256"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "noise_bank_profile_sha256": unit["noise_bank_profile_sha256"],
        "artifacts": [
            {
                "role": "pair_metrics",
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": parallel.sha256_file(path),
            }
        ],
        "metadata": {"fixture": True},
    }


def _unit(
    root: Path,
    identifier: str,
    *,
    seed: int,
    fold: str,
    kind: str = "standard_test",
    profile: str = PROFILE_A,
    fail: bool = False,
) -> dict[str, object]:
    return {
        "prediction_unit_id": identifier,
        "prediction_kind": kind,
        "seed": seed,
        "fold": fold,
        "arm": "film_lp_matched",
        "checkpoint_sha256": "c" * 64,
        "noise_bank_namespace": f"test/{fold}/seed_{seed}",
        "noise_bank_profile_sha256": profile,
        "artifact_path": str((root / f"{identifier}.csv").resolve()),
        "expected_artifact_roles": ["pair_metrics"],
        "fail": fail,
    }


class ParallelPredictionTests(unittest.TestCase):
    def test_group_assignment_is_stable_and_keeps_seed_fold_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = [
                _unit(root, "b", seed=22, fold="fold_b"),
                _unit(root, "a2", seed=11, fold="fold_a"),
                _unit(root, "a1", seed=11, fold="fold_a"),
            ]
            assigned = parallel.assign_units_to_physical_gpus(units, gpu_ids=(3, 7))
        by_group: dict[tuple[int, str], set[int]] = {}
        for row in assigned:
            by_group.setdefault((int(row["seed"]), str(row["fold"])), set()).add(
                int(row["physical_gpu_id"])
            )
            self.assertEqual(row["logical_device"], 0)
        self.assertTrue(all(len(values) == 1 for values in by_group.values()))
        self.assertEqual(
            [row["prediction_unit_id"] for row in assigned], ["a1", "a2", "b"]
        )

    def test_parallel_cells_use_physical_visibility_and_logical_device_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            artifacts = base / "artifacts"
            control = base / "control"
            artifacts.mkdir()
            units = [
                _unit(artifacts, "std_a", seed=11, fold="fold_a"),
                _unit(
                    artifacts,
                    "int_a",
                    seed=11,
                    fold="fold_a",
                    kind="matched_checkpoint_intervention",
                ),
                _unit(artifacts, "std_b", seed=22, fold="fold_b", profile=PROFILE_B),
                _unit(
                    artifacts,
                    "traj_b",
                    seed=22,
                    fold="fold_b",
                    kind="validation_trajectory",
                    profile=PROFILE_B,
                ),
            ]
            manifest = parallel.run_parallel_prediction_units(
                units,
                worker_entrypoint=WORKER,
                control_dir=control,
                artifact_root=artifacts,
                gpu_ids=(3, 7),
                workers_per_gpu={3: 1, 7: 1},
            )
            self.assertEqual(manifest["prediction_unit_count"], 4)
            self.assertEqual(
                set(manifest["shared_noise_profiles"].values()), {PROFILE_A, PROFILE_B}
            )
            for unit in units:
                cell_path = parallel.prediction_cell_manifest_path(
                    control, unit["prediction_unit_id"]
                )
                cell = parallel._read_signed(cell_path)
                self.assertEqual(cell["logical_device"], 0)
                self.assertEqual(
                    cell["cuda_visible_devices"], str(cell["physical_gpu_id"])
                )
                payload = json.loads(
                    Path(unit["artifact_path"]).read_text(encoding="utf-8")
                )
                self.assertEqual(payload["logical_device"], 0)
                self.assertEqual(payload["visible"], str(cell["physical_gpu_id"]))

            execution = control / "parallel_prediction_execution_manifest.json"
            before = execution.stat().st_mtime_ns
            repeated = parallel.run_parallel_prediction_units(
                units,
                worker_entrypoint=WORKER,
                control_dir=control,
                artifact_root=artifacts,
                gpu_ids=(3, 7),
                workers_per_gpu=1,
                resume=True,
            )
            self.assertEqual(repeated["payload_sha256"], manifest["payload_sha256"])
            self.assertEqual(execution.stat().st_mtime_ns, before)

    def test_partial_resume_reuses_valid_cell_and_runs_only_missing_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            artifacts = base / "artifacts"
            control = base / "control"
            artifacts.mkdir()
            units = [
                _unit(artifacts, "first", seed=11, fold="fold_a"),
                _unit(artifacts, "second", seed=22, fold="fold_b", profile=PROFILE_B),
            ]
            parallel.run_parallel_prediction_units(
                units,
                worker_entrypoint=WORKER,
                control_dir=control,
                artifact_root=artifacts,
                workers_per_gpu=1,
            )
            execution = control / "parallel_prediction_execution_manifest.json"
            execution.unlink()
            second_manifest = parallel.prediction_cell_manifest_path(control, "second")
            second_manifest.unlink()
            Path(units[1]["artifact_path"]).unlink()
            first_artifact = Path(units[0]["artifact_path"])
            first_mtime = first_artifact.stat().st_mtime_ns
            result = parallel.run_parallel_prediction_units(
                units,
                worker_entrypoint=WORKER,
                control_dir=control,
                artifact_root=artifacts,
                workers_per_gpu=1,
                resume=True,
            )
            self.assertEqual(result["prediction_unit_count"], 2)
            self.assertEqual(first_artifact.stat().st_mtime_ns, first_mtime)
            self.assertTrue(Path(units[1]["artifact_path"]).is_file())

    def test_tampered_completed_cell_fails_closed_instead_of_rerunning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            artifacts = base / "artifacts"
            control = base / "control"
            artifacts.mkdir()
            units = [_unit(artifacts, "only", seed=11, fold="fold_a")]
            parallel.run_parallel_prediction_units(
                units,
                worker_entrypoint=WORKER,
                control_dir=control,
                artifact_root=artifacts,
                workers_per_gpu=1,
            )
            Path(units[0]["artifact_path"]).write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(parallel.ParallelPredictionError, "drift"):
                parallel.run_parallel_prediction_units(
                    units,
                    worker_entrypoint=WORKER,
                    control_dir=control,
                    artifact_root=artifacts,
                    workers_per_gpu=1,
                    resume=True,
                )

    def test_failure_stops_same_gpu_queue_before_next_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            artifacts = base / "artifacts"
            control = base / "control"
            artifacts.mkdir()
            units = [
                _unit(artifacts, "a_fail", seed=11, fold="fold_a", fail=True),
                _unit(artifacts, "z_not_started", seed=11, fold="fold_a"),
            ]
            with self.assertRaisesRegex(
                parallel.ParallelPredictionError, "stopped after worker failure"
            ):
                parallel.run_parallel_prediction_units(
                    units,
                    worker_entrypoint=WORKER,
                    control_dir=control,
                    artifact_root=artifacts,
                    workers_per_gpu=1,
                )
            self.assertFalse(Path(units[1]["artifact_path"]).exists())
            journal = json.loads(
                (control / "prediction_journal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(journal["latest"]["status"], "failed")

    def test_noise_profile_drift_within_namespace_is_rejected_before_launch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = [
                _unit(root, "a", seed=11, fold="fold_a", profile=PROFILE_A),
                _unit(root, "b", seed=11, fold="fold_a", profile=PROFILE_B),
            ]
            with self.assertRaisesRegex(
                parallel.ParallelPredictionError, "noise lineage"
            ):
                parallel.validate_shared_noise_lineage(units)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
