"""Regression tests for the provenance-preserving text-effect recovery runner."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_10seed as experiment,
)
from scripts.rq3 import (
    news_first_vol_film_unet_pure_cnn_backbone_text_effect_recovery as recovery,
)


class TextEffectRecoveryTests(unittest.TestCase):
    def test_frozen_main_attestation_preserves_original_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "stages/backbones/code_hashes.csv"
            manifest.parent.mkdir(parents=True)
            source = (
                experiment.REPO_ROOT / recovery.FROZEN_MAIN_RELATIVE_PATH
            ).resolve()
            pd.DataFrame(
                [
                    {
                        "artifact_role": "code:frozen-main",
                        "path": str(source),
                        "size_bytes": source.stat().st_size,
                        "sha256": experiment.sha256_file(source),
                    }
                ]
            ).to_csv(manifest, index=False)

            attestation = recovery._frozen_main_attestation(root)

            self.assertEqual(
                attestation["frozen_experiment_source_sha256"],
                experiment.sha256_file(source),
            )
            self.assertFalse(attestation["frozen_manifests_modified"])
            self.assertEqual(
                experiment.sha256_file(manifest),
                attestation["frozen_code_manifest_sha256"],
            )

    def test_frozen_main_attestation_rejects_source_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "stages/backbones/code_hashes.csv"
            manifest.parent.mkdir(parents=True)
            source = (
                experiment.REPO_ROOT / recovery.FROZEN_MAIN_RELATIVE_PATH
            ).resolve()
            pd.DataFrame(
                [
                    {
                        "artifact_role": "code:frozen-main",
                        "path": str(source),
                        "size_bytes": source.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ]
            ).to_csv(manifest, index=False)

            with self.assertRaisesRegex(ValueError, "source drift"):
                recovery._frozen_main_attestation(root)

    def test_checkpoint_router_installs_complete_pure_and_film_profiles(self) -> None:
        root = Path("/tmp/text-effect-recovery-profile-test").resolve()
        roots = experiment._stage_roots(root)
        observed: list[dict[str, object]] = []

        def probe(_stage_root: Path) -> Path:
            observed.append(
                {
                    "jobs": experiment.direct.EXPECTED_TRAINING_JOBS,
                    "arms": tuple(experiment.direct.DIRECT_ARMS),
                    "fairness": experiment.direct._freeze_initial_state_fairness,
                    "inference": experiment.direct.INFERENCE_DETERMINISM_KIND,
                }
            )
            return Path(_stage_root)

        original_freezer = experiment.direct._freeze_checkpoints
        with mock.patch.object(
            experiment.direct, "_freeze_checkpoints", side_effect=probe
        ) as frozen_probe:
            patched_probe = experiment.direct._freeze_checkpoints
            with recovery._checkpoint_freeze_router(root):
                with experiment._pure_stage_profile(
                    main_root=root,
                    stage_name="backbones",
                    arm=experiment.PARENT_ARM,
                    use_graft=False,
                ):
                    experiment.direct._freeze_checkpoints(roots["backbones"])
                with experiment._film_stage_profile(
                    main_root=root, stage_name="film_continuations"
                ):
                    experiment.direct._freeze_checkpoints(roots["film_continuations"])
            self.assertIs(experiment.direct._freeze_checkpoints, patched_probe)
            self.assertEqual(frozen_probe.call_count, 2)
        self.assertIs(experiment.direct._freeze_checkpoints, original_freezer)

        self.assertEqual(observed[0]["jobs"], 40)
        self.assertEqual(observed[0]["arms"], (experiment.PARENT_ARM,))
        self.assertIs(
            observed[0]["fairness"], experiment._stage_freeze_initial_state_fairness
        )
        self.assertEqual(
            observed[0]["inference"], "cnn_unet_pure_no_text_10seed_inference_v1"
        )
        self.assertEqual(observed[1]["jobs"], 200)
        self.assertEqual(observed[1]["arms"], experiment.FILM_ARMS)
        self.assertIs(
            observed[1]["fairness"], experiment._stage_freeze_initial_state_fairness
        )
        self.assertEqual(
            observed[1]["inference"], "film_text_10seed_lr2p5e5_inference_v1"
        )

    def test_checkpoint_router_restores_shared_freezer_after_failure(self) -> None:
        root = Path("/tmp/text-effect-recovery-failure-test").resolve()
        original = experiment.direct._freeze_checkpoints
        with mock.patch.object(
            experiment.direct,
            "_freeze_checkpoints",
            side_effect=RuntimeError("probe failure"),
        ):
            patched = experiment.direct._freeze_checkpoints
            with self.assertRaisesRegex(RuntimeError, "probe failure"):
                with (
                    recovery._checkpoint_freeze_router(root),
                    experiment._pure_stage_profile(
                        main_root=root,
                        stage_name="backbones",
                        arm=experiment.PARENT_ARM,
                        use_graft=False,
                    ),
                ):
                    experiment.direct._freeze_checkpoints(
                        experiment._stage_roots(root)["backbones"]
                    )
            self.assertIs(experiment.direct._freeze_checkpoints, patched)
        self.assertIs(experiment.direct._freeze_checkpoints, original)

    def test_resource_limits_use_shared_cap_and_keep_memory_reserve(self) -> None:
        snapshot = {
            "gpus": [
                {
                    "gpu_index": gpu,
                    "gpu_uuid": f"GPU-{gpu}",
                    "memory_used_mib": 20_177,
                    "memory_total_mib": 24_576,
                }
                for gpu in (0, 1)
            ],
            "compute_processes": [
                {"gpu_uuid": f"GPU-{gpu}", "pid": 100 + gpu} for gpu in (0, 1)
            ],
        }

        limits = recovery._workers_by_gpu(
            snapshot,
            gpu_ids=(0, 1),
            shared_cap=4,
            idle_cap=10,
            worker_memory_mib=450,
            reserve_mib=1_536,
        )

        self.assertEqual(limits, {0: 4, 1: 4})
        snapshot["compute_processes"] = []
        for row in snapshot["gpus"]:
            row["memory_used_mib"] = 14
        self.assertEqual(
            recovery._workers_by_gpu(
                snapshot,
                gpu_ids=(0, 1),
                shared_cap=4,
                idle_cap=10,
                worker_memory_mib=450,
                reserve_mib=1_536,
            ),
            {0: 10, 1: 10},
        )

    def test_batch_selection_never_moves_a_job_across_gpus(self) -> None:
        pending = [
            {"job_id": f"g{gpu}-{index}", "gpu_id": gpu}
            for gpu in (0, 1)
            for index in range(5)
        ]
        selected = recovery._select_batch(pending, {0: 2, 1: 3})
        self.assertEqual(
            [row["job_id"] for row in selected],
            ["g0-0", "g0-1", "g1-0", "g1-1", "g1-2"],
        )
        self.assertTrue(
            all(row["job_id"].startswith(f"g{row['gpu_id']}") for row in selected)
        )


if __name__ == "__main__":
    unittest.main()
