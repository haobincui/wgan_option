from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.rq3 import news_first_vol_architecture_window_study as study
from scripts.rq3 import news_first_vol_architecture_window_study_analysis as analysis


class ArchitectureWindowStudyContractTests(unittest.TestCase):
    def test_architecture_registry_is_exact_and_capacity_profiles_are_frozen(
        self,
    ) -> None:
        config = study.load_config(
            "configs/rq3/news_first_vol_generator_architecture_3seed_5m.yaml"
        )
        plan = study.planned_cells(config)
        self.assertEqual(
            Counter(row["stage"] for row in plan["new_training_jobs"]),
            {"screen": 9, "formal": 36, "scaled": 24},
        )
        self.assertEqual(len(plan["reused_baseline_cells"]), 24)
        screen = [row for row in plan["new_training_jobs"] if row["stage"] == "screen"]
        self.assertEqual(Counter(row["gpu_id"] for row in screen), {0: 5, 1: 4})
        self.assertEqual(config["matrix"]["screen"]["num_epochs"], 60)
        self.assertEqual(
            config["matrix"]["screen"]["conditioning_learning_rates"],
            [1.0e-5, 2.5e-5, 5.0e-5],
        )
        self.assertTrue(all("conditioning_learning_rate" in row for row in screen))
        self.assertNotIn("matched_capacity_profiles", config["matrix"])
        matched = config["matrix"]["matched_capacity"]
        self.assertEqual(
            config["interpretation"], "retrospective_rolling_development_3seed"
        )
        self.assertEqual(
            matched["target_generator_parameters"],
            study.TARGET_MATCHED_GENERATOR_PARAMETERS,
        )
        self.assertEqual(
            matched["expected_winner_generator_parameters"],
            study.MATCHED_GENERATOR_PARAMETER_COUNTS,
        )
        self.assertEqual(
            set(matched["deterministic_search_grid"]), set(study.ARCHITECTURE_MODES)
        )
        self.assertTrue(
            all(
                row["capacity_profile"]
                == study._matched_capacity_profile_name(row["generator_mode"])
                for row in plan["new_training_jobs"]
                if row["stage"] in {"screen", "formal"}
            )
        )

    def test_matched_capacity_grid_search_freezes_and_drives_screen_and_formal(
        self,
    ) -> None:
        config = study.load_config(
            "configs/rq3/news_first_vol_generator_architecture_3seed_5m.yaml"
        )
        counts = {
            "crossattn_unet_mask_coords_v1": {
                28: 727_604,
                29: 750_757,
                30: 774_648,
                31: 799_277,
                32: 824_644,
                33: 850_749,
                34: 877_592,
                35: 905_173,
                36: 933_492,
            },
            "transformer_tokens_mask_coords_v1": {
                256: 735_873,
                320: 785_217,
                384: 834_561,
                448: 883_905,
                512: 933_249,
            },
            "stylemod_unet_mask_coords_v1": {
                26: 725_719,
                27: 750_591,
                28: 776_201,
                29: 802_549,
                30: 829_635,
                31: 857_459,
                32: 886_021,
                33: 915_321,
                34: 945_359,
            },
        }

        def parameter_count(mode: str, profile: dict[str, object]) -> int:
            if mode == "transformer_tokens_mask_coords_v1":
                key = int(profile["gen_transformer_ffn_dim"])
            else:
                key = int(profile["gen_base_channels"])
            return counts[mode][key]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            frozen_config = {
                key: value
                for key, value in config.items()
                if key not in {"source_config_path", "source_config_sha256"}
            }
            (root / "resolved_config.yaml").write_text(
                yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8"
            )
            with mock.patch.object(
                study, "_generator_parameter_count", side_effect=parameter_count
            ):
                path = study._freeze_matched_capacity_profiles(config, root)
                artifact = study._read_signed(path, kind=study.MATCHED_CAPACITY_KIND)
                self.assertEqual(
                    artifact["target_generator_parameters"],
                    study.TARGET_MATCHED_GENERATOR_PARAMETERS,
                )
                self.assertEqual(artifact["maximum_relative_deviation"], 0.05)
                self.assertEqual(
                    artifact["expected_winner_generator_parameters"],
                    study.MATCHED_GENERATOR_PARAMETER_COUNTS,
                )
                self.assertEqual(
                    artifact["search_grid_sha256"],
                    study._payload_sha256(artifact["deterministic_search_grid"]),
                )
                profiles = artifact["profiles"]
                self.assertEqual(
                    {
                        mode: int(profile["actual_generator_parameters"])
                        for mode, profile in profiles.items()
                    },
                    study.MATCHED_GENERATOR_PARAMETER_COUNTS,
                )
                self.assertEqual(
                    profiles["crossattn_unet_mask_coords_v1"]["gen_base_channels"],
                    32,
                )
                self.assertEqual(
                    profiles["transformer_tokens_mask_coords_v1"][
                        "gen_transformer_ffn_dim"
                    ],
                    384,
                )
                self.assertEqual(
                    profiles["stylemod_unet_mask_coords_v1"]["gen_base_channels"],
                    30,
                )
                for profile in profiles.values():
                    unsigned = {
                        key: value
                        for key, value in profile.items()
                        if key != "profile_sha256"
                    }
                    self.assertEqual(
                        profile["profile_sha256"], study._payload_sha256(unsigned)
                    )

                registry = {
                    "kind": study.REGISTRY_KIND,
                    "study_kind": study.ARCHITECTURE_KIND,
                    "line": "architecture",
                    "matched_capacity_profiles_path": str(path.resolve()),
                    "matched_capacity_profiles_sha256": study._sha256_file(path),
                }
                study._write_signed(root / "registry/task_registry.json", registry)
                study._verify_matched_capacity_profiles(root, registry)
                plan = study.planned_cells(config)["new_training_jobs"]
                for stage in ("screen", "formal"):
                    for mode in study.ARCHITECTURE_MODES:
                        job = next(
                            row
                            for row in plan
                            if row["stage"] == stage and row["generator_mode"] == mode
                        )
                        capacity = study._job_capacity(root, job)
                        self.assertEqual(
                            capacity["actual_generator_parameters"],
                            study.MATCHED_GENERATOR_PARAMETER_COUNTS[mode],
                        )

                tampered = dict(artifact)
                tampered_profiles = {
                    mode: dict(profile) for mode, profile in profiles.items()
                }
                cross = tampered_profiles["crossattn_unet_mask_coords_v1"]
                cross.update(
                    gen_base_channels=31,
                    gen_hidden_dim=992,
                    generator_parameters=799_277,
                    actual_generator_parameters=799_277,
                    absolute_deviation=abs(799_277 - 827_745),
                    relative_deviation=abs(799_277 - 827_745) / 827_745,
                )
                cross["profile_sha256"] = study._payload_sha256(
                    {
                        key: value
                        for key, value in cross.items()
                        if key != "profile_sha256"
                    }
                )
                tampered["profiles"] = tampered_profiles
                study._write_signed(path, tampered)
                tampered_registry = {
                    **registry,
                    "matched_capacity_profiles_sha256": study._sha256_file(path),
                }
                with self.assertRaisesRegex(ValueError, "not grid optima"):
                    study._verify_matched_capacity_profiles(
                        root, tampered_registry, recompute_winners=True
                    )

    def test_window_registry_is_120_logical_96_new_and_never_retrains_5m(self) -> None:
        config = study.load_config(
            "configs/rq3/news_first_vol_alignment_tolerance_3seed.yaml"
        )
        plan = study.planned_cells(config)
        self.assertEqual(
            config["interpretation"], "retrospective_rolling_development_3seed"
        )
        self.assertEqual(len(plan["new_training_jobs"]), 96)
        self.assertEqual(len(plan["reused_baseline_cells"]), 24)
        self.assertEqual(
            {int(row["tolerance_minutes"]) for row in plan["new_training_jobs"]},
            {10, 15, 20, 30},
        )
        self.assertEqual(
            {int(row["tolerance_minutes"]) for row in plan["reused_baseline_cells"]},
            {5},
        )
        self.assertEqual(config["analysis"]["expected_total_evaluation_units"], 324)
        self.assertEqual(
            study._prediction_text_conditions(line="window", conditional=True),
            ("matched", "zero"),
        )
        self.assertNotIn(
            "generator_film_learning_rate", config["matrix"]["arms"]["film_lp_matched"]
        )
        self.assertNotIn("film_learning_rate", config["training"])

    def test_window_matched_and_zero_embeddings_do_not_require_shuffle_donors(
        self,
    ) -> None:
        matched = np.arange(1024, dtype=np.float32)
        vectors = {"pair": matched, "donor": matched[::-1].copy()}
        resolved, donor = study._prediction_embedding(
            pair_id="pair",
            condition="matched",
            lp_vectors=vectors,
            donor_by_pair={},
        )
        np.testing.assert_array_equal(resolved, matched)
        self.assertIsNone(donor)

        resolved, donor = study._prediction_embedding(
            pair_id="pair",
            condition="zero",
            lp_vectors=vectors,
            donor_by_pair={},
        )
        np.testing.assert_array_equal(resolved, np.zeros(1024, dtype=np.float32))
        self.assertIsNone(donor)

        resolved, donor = study._prediction_embedding(
            pair_id="pair",
            condition="shuffle",
            lp_vectors=vectors,
            donor_by_pair={"pair": "donor"},
        )
        np.testing.assert_array_equal(resolved, vectors["donor"])
        self.assertEqual(donor, "donor")

    def test_contrast_families_and_window_relations_are_decision_complete(self) -> None:
        frame = pd.DataFrame(
            {
                "generator_mode": [
                    *study.ARCHITECTURE_MODES,
                    "film_unet_mask_coords_v1",
                    study.ARCHITECTURE_MODES[0],
                    "film_unet_mask_coords_v1",
                    "cnn_unet_mask_coords_v1",
                ],
                "model_id": [
                    *(f"formal:{mode}" for mode in study.ARCHITECTURE_MODES),
                    "film_reference",
                    "scaled:validation_point_leader_scaled",
                    "scaled:film_reference_scaled",
                    "pure_cnn_reference",
                ],
            }
        )
        architecture = analysis._architecture_contrasts(frame)
        primary = [
            row
            for row in architecture
            if row["family_id"] == "architecture_primary_new_modes_vs_film_holm3"
        ]
        self.assertEqual(len(primary), 3)
        self.assertTrue(all(row["apply_holm"] for row in primary))
        window = analysis._window_contrasts()
        counts = Counter(row["family_id"] for row in window if row["apply_holm"])
        self.assertEqual(counts["window_between_models_common5_holm5"], 5)
        self.assertEqual(counts["window_within_film_lp_matched_common5_holm4"], 4)
        self.assertEqual(counts["window_within_pure_cnn_no_text_common5_holm4"], 4)
        self.assertFalse(any("window_text_reliance" in family for family in counts))
        text_rows = [
            row for row in window if row["scope"] == "text_intervention_descriptive"
        ]
        self.assertEqual(len(text_rows), 5)
        self.assertTrue(all(not row["apply_holm"] for row in text_rows))
        expanded = [row for row in window if "_vs_train05" in row["contrast_id"]]
        self.assertTrue(
            all(
                row["comparison_relation"]
                == "expanded_training_window_vs_5m_on_common5"
                for row in expanded
            )
        )
        text_counts = Counter(
            row["family_id"]
            for row in architecture
            if row["family_id"].startswith("architecture_text_reliance_")
        )
        self.assertEqual(len(text_counts), 6)
        self.assertEqual(set(text_counts.values()), {2})

    def test_pair_window_metadata_uses_conservative_maximum_article_overlap(
        self,
    ) -> None:
        end = pd.Timestamp("2023-01-03T14:05:00Z")
        overlaps = [0, 5, 1, 3, 0]
        frame = pd.DataFrame(
            {
                "pair_id": ["mixed", "mixed", "partial", "partial", "pre"],
                "article_id": ["a", "b", "c", "d", "e"],
                "sample_id": [f"sample-{index}" for index in range(5)],
                "news_row_id": list(range(5)),
                "news_available_time_utc": [
                    end - pd.Timedelta(minutes=value) for value in overlaps
                ],
                "current_window_start_utc": [end - pd.Timedelta(minutes=5)] * 5,
                "current_window_end_utc": [end] * 5,
                "window_relation": [
                    "current_pre_news_target_post_news",
                    "current_fully_post_news",
                    "current_partially_post_news",
                    "current_partially_post_news",
                    "current_pre_news_target_post_news",
                ],
                "post_news_current_overlap_minutes": [0, 5, 1, 3, 0],
            }
        )
        collapsed = study._collapse_pair_window_metadata(frame).set_index("pair_id")
        self.assertEqual(
            collapsed.at["mixed", "window_relation"], "current_fully_post_news"
        )
        self.assertEqual(
            int(collapsed.at["mixed", "post_news_current_overlap_minutes"]), 5
        )
        self.assertEqual(
            int(collapsed.at["mixed", "post_news_current_overlap_min_minutes"]), 0
        )
        self.assertEqual(int(collapsed.at["mixed", "window_relation_variant_count"]), 2)
        self.assertEqual(
            int(collapsed.at["mixed", "post_news_current_overlap_variant_count"]),
            2,
        )
        self.assertEqual(
            collapsed.at["partial", "window_relation"],
            "current_partially_post_news",
        )
        self.assertEqual(
            int(collapsed.at["partial", "post_news_current_overlap_minutes"]), 3
        )
        self.assertEqual(
            collapsed.at["pre", "window_relation"],
            "current_pre_news_target_post_news",
        )

    def test_pair_window_metadata_rejects_relation_overlap_mismatch(self) -> None:
        end = pd.Timestamp("2023-01-03T14:05:00Z")
        frame = pd.DataFrame(
            {
                "pair_id": ["pair"],
                "article_id": ["article"],
                "sample_id": ["sample"],
                "news_row_id": [1],
                "news_available_time_utc": [end - pd.Timedelta(minutes=3)],
                "current_window_start_utc": [end - pd.Timedelta(minutes=5)],
                "current_window_end_utc": [end],
                "window_relation": ["current_pre_news_target_post_news"],
                "post_news_current_overlap_minutes": [3],
            }
        )
        with self.assertRaisesRegex(ValueError, "disagrees with source timestamps"):
            study._collapse_pair_window_metadata(frame)

    def test_pair_window_metadata_uses_lp_article_deduplication(self) -> None:
        end = pd.Timestamp("2023-01-03T14:05:00Z")
        frame = pd.DataFrame(
            {
                "pair_id": ["pair", "pair", "pair"],
                "article_id": ["same", "same", "other"],
                "sample_id": ["sample-1", "sample-2", "sample-3"],
                "news_row_id": [1, 2, 3],
                "news_available_time_utc": [
                    end,
                    end - pd.Timedelta(minutes=5),
                    end - pd.Timedelta(minutes=3),
                ],
                "current_window_start_utc": [end - pd.Timedelta(minutes=5)] * 3,
                "current_window_end_utc": [end] * 3,
                "window_relation": [
                    "current_pre_news_target_post_news",
                    "current_fully_post_news",
                    "current_partially_post_news",
                ],
                "post_news_current_overlap_minutes": [0, 5, 3],
            }
        )
        row = study._collapse_pair_window_metadata(frame).iloc[0]
        self.assertEqual(int(row["aligned_article_count"]), 2)
        self.assertEqual(int(row["post_news_current_overlap_minutes"]), 3)
        self.assertEqual(int(row["post_news_current_overlap_variant_count"]), 2)

    def test_pair_window_metadata_rejects_bad_source_timestamp(self) -> None:
        end = pd.Timestamp("2023-01-03T14:05:00Z")
        frame = pd.DataFrame(
            {
                "pair_id": ["pair"],
                "article_id": ["article"],
                "sample_id": ["sample"],
                "news_row_id": [1],
                "news_available_time_utc": ["not-a-timestamp"],
                "current_window_start_utc": [end - pd.Timedelta(minutes=5)],
                "current_window_end_utc": [end],
                "window_relation": ["current_pre_news_target_post_news"],
                "post_news_current_overlap_minutes": [0],
            }
        )
        with self.assertRaisesRegex(
            ValueError, "Invalid article-level window timestamp"
        ):
            study._collapse_pair_window_metadata(frame)

    def test_real_5m_pair_window_source_counts_and_order_are_frozen(self) -> None:
        root = Path(
            "data/processed/rq3/"
            "news_first_vol_surfaces_q097_103_ttm01_38_exact_ttm_v1/"
            "tolerance_05m"
        )
        frame = pd.read_excel(
            root / "merged_vol.xlsx",
            sheet_name="gan_input_ready",
            usecols=[
                "pair_id",
                "article_id",
                "sample_id",
                "news_row_id",
                "news_available_time_utc",
                "current_window_start_utc",
                "current_window_end_utc",
                "window_relation",
                "post_news_current_overlap_minutes",
            ],
        )
        support = pd.read_csv(
            root / "surface_support_audit.csv.gz",
            usecols=[
                "pair_id",
                "surface_training_eligible",
                "joint_zero_support",
                "joint_strict_support_cell_count",
            ],
        )
        eligible = study._as_bool(
            support["surface_training_eligible"], label="eligible"
        )
        zero = study._as_bool(support["joint_zero_support"], label="zero")
        support = support.loc[
            eligible
            & ~zero
            & pd.to_numeric(support["joint_strict_support_cell_count"]).gt(0)
        ]
        frame["pair_id"] = frame["pair_id"].astype(str)
        frame = frame.loc[
            frame["pair_id"].isin(set(support["pair_id"].astype(str)))
        ].copy()
        expected_order = list(dict.fromkeys(frame["pair_id"]))
        collapsed = study._collapse_pair_window_metadata(frame)
        self.assertEqual(len(frame), 1_283)
        self.assertEqual(len(collapsed), 1_026)
        self.assertEqual(collapsed["pair_id"].tolist(), expected_order)
        self.assertEqual(int(collapsed["aligned_article_count"].sum()), 1_280)
        self.assertEqual(
            int(collapsed["window_relation_variant_count"].gt(1).sum()), 16
        )
        self.assertEqual(
            int(collapsed["post_news_current_overlap_variant_count"].gt(1).sum()),
            28,
        )

    def test_window_nesting_checks_all_four_adjacent_tolerances(self) -> None:
        self.assertEqual(
            study._adjacent_window_tolerance_pairs(),
            ((5, 10), (10, 15), (15, 20), (20, 30)),
        )

    def test_failed_benchmark_workspace_is_preserved_before_fresh_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary)
            legacy = study._fresh_benchmark_workspace(control)
            self.assertEqual(legacy, control / "benchmark_workspace")
            legacy.mkdir()
            (legacy / "failure.txt").write_text("preserve\n", encoding="utf-8")
            retry = study._fresh_benchmark_workspace(control)
            self.assertEqual(
                retry,
                control / "benchmark_attempts/attempt_0001/benchmark_workspace",
            )
            retry.mkdir(parents=True)
            next_retry = study._fresh_benchmark_workspace(control)
            self.assertEqual(
                next_retry,
                control / "benchmark_attempts/attempt_0002/benchmark_workspace",
            )
            self.assertEqual(
                (legacy / "failure.txt").read_text(encoding="utf-8"), "preserve\n"
            )

    def test_hierarchical_bootstrap_really_resamples_three_levels(self) -> None:
        rows = []
        for seed in study.SEEDS:
            for fold in study.FOLDS:
                for session in ("a", "b"):
                    rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "session_id": f"{fold}-{session}",
                            "target_mae_candidate": 0.9,
                            "target_mae_reference": 1.0,
                        }
                    )
        draws = analysis._hierarchical_draws(
            pd.DataFrame(rows), replicates=25, rng=np.random.default_rng(17)
        )
        self.assertEqual(draws.shape, (25,))
        self.assertTrue(np.isfinite(draws).all())
        self.assertTrue((draws < 0).all())

    def test_vectorized_bootstrap_matches_slow_cluster_reference(self) -> None:
        rows = []
        for seed_index, seed in enumerate(study.SEEDS):
            for fold_index, fold in enumerate(study.FOLDS):
                for session_index in range(3):
                    for pair_index in range(session_index + 1):
                        baseline = 1.0 + seed_index + fold_index / 10 + pair_index / 100
                        rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "session_id": f"{fold}-s{session_index}",
                                "target_mae_candidate": baseline
                                * (0.9 + session_index / 100),
                                "target_mae_reference": baseline,
                            }
                        )
        frame = pd.DataFrame(rows)
        fast = analysis._hierarchical_draws(
            frame, replicates=50, rng=np.random.default_rng(219)
        )
        rng = np.random.default_rng(219)
        seeds = np.asarray(sorted(frame["seed"].unique()))
        slow = []
        for _ in range(50):
            sampled = []
            for seed_index in rng.integers(0, len(seeds), size=len(seeds)):
                seed_rows = frame.loc[frame["seed"].eq(seeds[int(seed_index)])]
                folds = sorted(seed_rows["fold"].unique())
                for fold_index in rng.integers(0, len(folds), size=len(folds)):
                    fold_rows = seed_rows.loc[
                        seed_rows["fold"].eq(folds[int(fold_index)])
                    ]
                    sessions = sorted(fold_rows["session_id"].unique())
                    for session_index in rng.integers(
                        0, len(sessions), size=len(sessions)
                    ):
                        sampled.append(
                            fold_rows.loc[
                                fold_rows["session_id"].eq(sessions[int(session_index)])
                            ]
                        )
            draw = pd.concat(sampled, ignore_index=True)
            slow.append(
                math.log(
                    draw["target_mae_candidate"].mean()
                    / draw["target_mae_reference"].mean()
                )
            )
        np.testing.assert_allclose(fast, np.asarray(slow), rtol=0, atol=1e-14)

    def test_global_slot_lock_survives_supervisor_descriptor_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal_root"
            descriptor, _slot = study._acquire_global_gpu_slot(
                root, gpu_id=0, capacity=1
            )
            os.set_inheritable(descriptor, True)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(0.5)"],
                pass_fds=(descriptor,),
            )
            os.close(descriptor)
            try:
                self.assertIsNone(
                    study._acquire_global_gpu_slot(root, gpu_id=0, capacity=1)
                )
            finally:
                child.wait(timeout=5)
            reacquired = study._acquire_global_gpu_slot(root, gpu_id=0, capacity=1)
            self.assertIsNotNone(reacquired)
            assert reacquired is not None
            study._release_global_gpu_slot(reacquired[0])

    def test_cli_exposes_requested_aliases(self) -> None:
        action = next(
            item
            for item in study._parser()._actions
            if getattr(item, "dest", "") == "action"
        )
        self.assertTrue(
            {"screen-lr", "launch-training", "freeze-selection"}.issubset(
                set(action.choices)
            )
        )

    def test_checkpoint_update_audit_requires_both_generator_and_critic_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "run_root"
            run_dir = output_root / "stamp"
            checkpoint_dir = run_dir / "checkpoints"
            metric_dir = run_dir / "metrics"
            checkpoint_dir.mkdir(parents=True)
            metric_dir.mkdir()
            for name, value, count in (
                ("generator_initial_epoch0.pt", 0.0, 1),
                ("generator_best_learned.pt", 1.0, 1),
                ("discriminator_initial_epoch0.pt", 2.0, 729_157),
                ("discriminator_best_learned.pt", 3.0, 729_157),
            ):
                torch.save(
                    {"state_dict": {"weight": torch.full((count,), value)}},
                    checkpoint_dir / name,
                )
            (metric_dir / "best_learned_checkpoint.json").write_text(
                "{}\n", encoding="utf-8"
            )
            pd.DataFrame(
                [
                    {
                        "epoch": 1,
                        "val_recon": 1.0,
                        "g_recon": 1.0,
                        "g_total": 1.0,
                        "d_total": 1.0,
                        "gp": 1.0,
                    }
                ]
            ).to_csv(metric_dir / "training_metrics.csv", index=False)
            result = study._validate_benchmark_run(
                {"output_root": str(output_root), "generator_parameters": 1}
            )
            self.assertTrue(result["generator_updated"])
            self.assertTrue(result["critic_updated"])

    def test_checkpoint_tensor_hash_supports_scalars_and_detects_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scalar.pt"
            torch.save(
                {
                    "state_dict": {
                        "float": torch.tensor(1.25, dtype=torch.float32),
                        "integer": torch.tensor(7, dtype=torch.int64),
                    }
                },
                path,
            )
            original = study._checkpoint_tensor_sha256(path)
            torch.save(
                {
                    "state_dict": {
                        "float": torch.tensor(1.5, dtype=torch.float32),
                        "integer": torch.tensor(7, dtype=torch.int64),
                    }
                },
                path,
            )
            self.assertNotEqual(study._checkpoint_tensor_sha256(path), original)

    def test_checkpoint_tensor_hash_is_stride_independent_and_metadata_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
            self.assertFalse(noncontiguous.is_contiguous())

            def checkpoint(name: str, key: str, tensor: torch.Tensor) -> Path:
                path = root / name
                torch.save({"state_dict": {key: tensor}}, path)
                return path

            noncontiguous_hash = study._checkpoint_tensor_sha256(
                checkpoint("noncontiguous.pt", "weight", noncontiguous)
            )
            contiguous_hash = study._checkpoint_tensor_sha256(
                checkpoint("contiguous.pt", "weight", noncontiguous.contiguous())
            )
            self.assertEqual(noncontiguous_hash, contiguous_hash)
            self.assertNotEqual(
                noncontiguous_hash,
                study._checkpoint_tensor_sha256(
                    checkpoint("key.pt", "other_weight", noncontiguous)
                ),
            )
            self.assertNotEqual(
                noncontiguous_hash,
                study._checkpoint_tensor_sha256(
                    checkpoint("shape.pt", "weight", noncontiguous.reshape(2, 6))
                ),
            )
            self.assertNotEqual(
                noncontiguous_hash,
                study._checkpoint_tensor_sha256(
                    checkpoint("dtype.pt", "weight", noncontiguous.to(torch.float64))
                ),
            )

    def test_checkpoint_tensor_hash_preserves_legacy_non_scalar_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy_supported.pt"
            state = {
                "bias": torch.tensor([1.5, -2.0], dtype=torch.float64),
                "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            }
            torch.save({"state_dict": state}, path)
            legacy_digest = hashlib.sha256()
            for key in sorted(state):
                tensor = state[key].detach().cpu().contiguous()
                legacy_digest.update(str(key).encode("utf-8"))
                legacy_digest.update(str(tensor.dtype).encode("ascii"))
                legacy_digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
                legacy_digest.update(tensor.view(torch.uint8).numpy().tobytes())
            self.assertEqual(
                study._checkpoint_tensor_sha256(path), legacy_digest.hexdigest()
            )

    def test_output_sha_manifest_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = root / "artifact.txt"
            item.write_text("frozen\n", encoding="utf-8")
            study._write_output_sha_manifest(root)
            study._verify_output_sha_manifest(root)
            item.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA drift"):
                study._verify_output_sha_manifest(root)

    def test_control_output_sha_manifest_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary)
            log = control / "pipeline.log"
            log.write_text("done\n", encoding="utf-8")
            study._write_control_output_sha_manifest(control)
            study._verify_control_output_sha_manifest(control)
            log.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "control output SHA drift"):
                study._verify_control_output_sha_manifest(control)

    def test_terminal_run_pipeline_cli_does_not_rewrite_control_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "terminal"
            config = {"output_root": str(root)}
            with (
                mock.patch.object(study, "load_config", return_value=config),
                mock.patch.object(study, "run_pipeline", return_value=root),
                mock.patch.object(study, "_is_terminal_root", return_value=True),
                mock.patch.object(study, "_write_control_output_sha_manifest") as write,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(
                    study.main(
                        [
                            "run-pipeline",
                            "--config",
                            "unused.yaml",
                            "--output-dir",
                            str(root),
                            "--resume",
                        ]
                    ),
                    0,
                )
            write.assert_not_called()

    def test_first_terminal_completion_freezes_control_after_final_print(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "terminal"
            events: list[str] = []
            with (
                mock.patch.object(
                    study, "load_config", return_value={"output_root": str(root)}
                ),
                mock.patch.object(study, "run_pipeline", return_value=root),
                mock.patch.object(
                    study, "_is_terminal_root", side_effect=[False, True]
                ),
                mock.patch.object(
                    study,
                    "_write_control_output_sha_manifest",
                    side_effect=lambda _path: events.append("manifest"),
                ),
                mock.patch(
                    "builtins.print",
                    side_effect=lambda *_a, **_k: events.append("print"),
                ),
            ):
                study.main(
                    [
                        "run-pipeline",
                        "--config",
                        "unused.yaml",
                        "--output-dir",
                        str(root),
                        "--resume",
                    ]
                )
            self.assertEqual(events, ["print", "manifest"])

    def test_window_reuse_binds_exact_old_and_with20_5m_semantics(self) -> None:
        config = study.load_config(
            "configs/rq3/news_first_vol_alignment_tolerance_3seed.yaml"
        )
        evidence = study._reused_5m_semantic_equivalence(config)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(
            evidence["workbook_semantic_sha256"],
            config["data"]["reused_5m_semantic_equivalence"][
                "expected_workbook_semantic_sha256"
            ],
        )
        self.assertEqual(len(evidence["inputs"]), 4)

    def test_worker_passes_inherited_global_slot_to_training_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            descriptor = os.open(Path(temporary) / "slot.lock", os.O_CREAT | os.O_RDWR)
            try:
                with mock.patch.dict(
                    os.environ,
                    {study.GLOBAL_GPU_SLOT_FD_ENV: str(descriptor)},
                    clear=False,
                ):
                    self.assertEqual(
                        study._inherited_global_slot_descriptors(), (descriptor,)
                    )
                    with mock.patch.object(
                        study.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ) as run:
                        study._run_training_subprocess(["training"], env={})
                    self.assertEqual(run.call_args.kwargs["pass_fds"], (descriptor,))
            finally:
                os.close(descriptor)
        with mock.patch.dict(
            os.environ, {study.GLOBAL_GPU_SLOT_FD_ENV: "99999999"}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid global GPU slot FD"):
                study._inherited_global_slot_descriptors()

    def test_launch_failure_terminates_whole_worker_process_group_before_slot_release(
        self,
    ) -> None:
        process = mock.Mock(pid=43210)
        process.poll.return_value = None
        handle = mock.Mock()
        events: list[tuple[str, int]] = []

        def kill_group(_pid: int, sent_signal: int) -> None:
            events.append(("signal", sent_signal))
            if sent_signal in (0, signal.SIGKILL):
                raise ProcessLookupError

        def release(_descriptor: int) -> None:
            events.append(("release", _descriptor))

        with (
            mock.patch.object(study.os, "killpg", side_effect=kill_group),
            mock.patch.object(study, "_release_global_gpu_slot", side_effect=release),
        ):
            study._terminate_worker_process_groups(
                {"job": (process, handle, {}, 17, 0)}
            )
        self.assertIn(("signal", signal.SIGTERM), events)
        self.assertLess(
            events.index(("signal", signal.SIGTERM)), events.index(("release", 17))
        )
        process.wait.assert_called_once_with()
        handle.close.assert_called_once_with()

    def test_prediction_bundle_rejects_raw_cell_prediction_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            allowlist_path = root / "allowlist.json"
            study._write_signed(
                allowlist_path,
                {
                    "kind": "architecture_window_evaluation_allowlist_v1",
                    "checkpoint_rows": [
                        {
                            "path": str(checkpoint),
                            "sha256": study._sha256_file(checkpoint),
                            "size_bytes": checkpoint.stat().st_size,
                        }
                    ],
                },
            )
            plan_path = root / "plan.json"
            study._write_signed(
                plan_path,
                {
                    "kind": study.PREDICTION_PLAN_KIND,
                    "total_evaluation_units": 1,
                    "evaluations": [{"evaluation_id": "eval"}],
                },
            )
            test_panel = root / "test.csv"
            test_panel.write_text("pair_id\npair\n", encoding="utf-8")
            test_manifest_path = root / "evaluation/test_input_manifest.json"
            test_manifest_path.parent.mkdir()
            study._write_signed(
                test_manifest_path,
                {
                    "kind": "architecture_window_test_inputs_v1",
                    "artifacts": [
                        {
                            "artifact_role": "test panel",
                            "path": str(test_panel),
                            "sha256": study._sha256_file(test_panel),
                            "size_bytes": test_panel.stat().st_size,
                        }
                    ],
                },
            )
            raw_prediction = root / "raw.csv"
            raw_prediction.write_text("pair_id,prediction\npair,1\n", encoding="utf-8")
            cell_metrics = root / "cell.csv"
            pd.DataFrame([{"evaluation_id": "eval", "pair_id": "pair"}]).to_csv(
                cell_metrics, index=False
            )
            cell_manifest = root / "cell.json"
            study._write_signed(
                cell_manifest,
                {
                    "kind": "architecture_window_prediction_cell_v1",
                    "prediction_path": str(raw_prediction),
                    "prediction_sha256": study._sha256_file(raw_prediction),
                    "pair_metrics_path": str(cell_metrics),
                    "pair_metrics_sha256": study._sha256_file(cell_metrics),
                    "row_count": 1,
                },
            )
            prediction_manifest = root / "prediction_manifest.csv"
            pd.DataFrame(
                [
                    {
                        "evaluation_id": "eval",
                        "manifest_path": str(cell_manifest),
                        "manifest_sha256": study._sha256_file(cell_manifest),
                    }
                ]
            ).to_csv(prediction_manifest, index=False)
            aggregate = root / "aggregate.csv"
            pd.DataFrame([{"evaluation_id": "eval", "pair_id": "pair"}]).to_csv(
                aggregate, index=False
            )
            registry = {
                "predictions_frozen": True,
                "checkpoint_allowlist_path": str(allowlist_path),
                "checkpoint_allowlist_sha256": study._sha256_file(allowlist_path),
                "prediction_plan_path": str(plan_path),
                "prediction_plan_sha256": study._sha256_file(plan_path),
                "test_input_manifest_path": str(test_manifest_path),
                "test_input_manifest_sha256": study._sha256_file(test_manifest_path),
                "pair_metrics_path": str(aggregate),
                "pair_metrics_sha256": study._sha256_file(aggregate),
                "prediction_manifest_path": str(prediction_manifest),
                "prediction_manifest_sha256": study._sha256_file(prediction_manifest),
                "prediction_pair_rows": 1,
            }
            with mock.patch.object(study, "_verify_prepare_lineage"):
                study._verify_prediction_bundle(root, registry)
                raw_prediction.write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "cell raw prediction"):
                    study._verify_prediction_bundle(root, registry)

    def test_predict_rejects_source_config_drift_before_opening_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(
                    study,
                    "load_config",
                    return_value={"source_config_sha256": "new"},
                ),
                mock.patch.object(
                    study,
                    "_load_registry",
                    return_value={"source_config_sha256": "frozen"},
                ),
                mock.patch.object(study, "_materialize_test_inputs") as materialize,
            ):
                with self.assertRaisesRegex(ValueError, "source config drift"):
                    study.predict("unused.yaml", temporary)
            materialize.assert_not_called()

    def test_materialized_worker_rejects_scientific_contract_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            job = {
                "job_id": "job",
                "job_spec_sha256": "spec",
                "stage": "window",
                "arm": "pure_cnn_no_text",
                "generator_mode": "cnn_unet_mask_coords_v1",
                "seed": 42,
                "tolerance_minutes": 10,
            }
            study._write_signed(
                root / "registry/task_registry.json",
                {
                    "kind": study.REGISTRY_KIND,
                    "study_kind": study.WINDOW_KIND,
                    "line": "window",
                },
            )
            (root / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "training": {
                            "prediction_mc_samples": 64,
                            "backbone_learning_rate": 5.0e-7,
                            "text_learning_rate": 2.5e-6,
                            "critic_learning_rate": 5.0e-7,
                        }
                    }
                ),
                encoding="utf-8",
            )
            training = {
                "generator_conditioning_mode": "cnn_unet_mask_coords_v1",
                "news_first_materialize_test_loader": False,
                "news_first_materialize_validation_loader": True,
                "seed": 42,
                "news_first_dataset_tolerance_minutes": 10,
                "news_first_full_training_state_mode": "none",
                "news_first_full_training_state_contract_path": "",
                "news_first_full_training_state_contract_sha256": "",
                "news_first_refit_mode": "none",
                "news_first_refit_recipe_path": "",
                "news_first_refit_recipe_sha256": "",
                "news_first_graft_state_path": "",
                "news_first_graft_state_sha256": "",
                "generator_noise_mode": "gaussian",
                "noise_dim": 32,
                "residual_output_mode": "identity_softplus_residual",
                "support_mask_mode": "raw_joint",
                "generator_current_input_mode": "current_support_masked",
                "critic_conditioning_mode": "lp_disabled_same_shape_v1",
                "generator_learning_rate": 5.0e-7,
                "discriminator_learning_rate": 5.0e-7,
                "learning_rate": 5.0e-7,
                "reduce_lr_min_lr": 5.0e-8,
                "num_epochs": 240,
                "early_stopping_min_epochs": 30,
                "early_stopping_patience": 20,
                "batch_size": 15,
                "discriminator_iter": 5,
                "validation_mc_samples": 16,
                "lr_warmup_epochs": 0,
                "use_early_stopping": True,
                "best_checkpoint_metric": "val_hybrid_score",
                "text_embedding_mode": "lp",
                "generator_optimizer_profile": "uniform_v1",
                "generator_text_learning_rate": 0.0,
                "generator_text_min_learning_rate": 0.0,
                "generator_film_learning_rate": 0.0,
                "generator_film_min_learning_rate": 0.0,
                "generator_conditioning_learning_rate": 0.0,
                "generator_conditioning_min_learning_rate": 0.0,
            }
            training_path = root / "job.yaml"
            training_path.write_text(yaml.safe_dump(training), encoding="utf-8")
            study._write_signed(
                root / "registry/materialized_worker_specs.json",
                {
                    "kind": study.WORKER_SPEC_KIND,
                    "study_kind": study.WINDOW_KIND,
                    "jobs": [
                        {
                            "job_id": "job",
                            "job_spec_sha256": "spec",
                            "training_config_path": str(training_path),
                            "training_config_sha256": study._sha256_file(training_path),
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "scientific training contract"):
                study._materialized_worker_spec(root, job)

    def test_materialized_conditional_worker_accepts_computed_lr_floor_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "registry").mkdir()
            job = {
                "job_id": "screen_job",
                "job_spec_sha256": "spec",
                "stage": "screen",
                "arm": "lp_matched",
                "generator_mode": "crossattn_unet_mask_coords_v1",
                "conditioning_learning_rate": 1.0e-5,
                "seed": 42,
                "tolerance_minutes": 5,
            }
            study._write_signed(
                root / "registry/task_registry.json",
                {
                    "kind": study.REGISTRY_KIND,
                    "study_kind": study.ARCHITECTURE_KIND,
                    "line": "architecture",
                },
            )
            (root / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "training": {
                            "prediction_mc_samples": 64,
                            "backbone_learning_rate": 5.0e-7,
                            "text_learning_rate": 2.5e-6,
                            "critic_learning_rate": 5.0e-7,
                        }
                    }
                ),
                encoding="utf-8",
            )
            training = {
                "generator_conditioning_mode": "crossattn_unet_mask_coords_v1",
                "news_first_materialize_test_loader": False,
                "news_first_materialize_validation_loader": True,
                "seed": 42,
                "news_first_dataset_tolerance_minutes": 5,
                "news_first_full_training_state_mode": "none",
                "news_first_full_training_state_contract_path": "",
                "news_first_full_training_state_contract_sha256": "",
                "news_first_refit_mode": "none",
                "news_first_refit_recipe_path": "",
                "news_first_refit_recipe_sha256": "",
                "news_first_graft_state_path": "",
                "news_first_graft_state_sha256": "",
                "generator_noise_mode": "gaussian",
                "noise_dim": 32,
                "residual_output_mode": "identity_softplus_residual",
                "support_mask_mode": "raw_joint",
                "generator_current_input_mode": "current_support_masked",
                "critic_conditioning_mode": "lp_disabled_same_shape_v1",
                "generator_learning_rate": 5.0e-7,
                "discriminator_learning_rate": 5.0e-7,
                "learning_rate": 5.0e-7,
                "reduce_lr_min_lr": 5.0e-8,
                "num_epochs": 60,
                "early_stopping_min_epochs": 60,
                "early_stopping_patience": 60,
                "batch_size": 16,
                "discriminator_iter": 5,
                "validation_mc_samples": 16,
                "lr_warmup_epochs": 0,
                "use_early_stopping": True,
                "best_checkpoint_metric": "val_hybrid_score",
                "text_embedding_mode": "lp",
                "generator_optimizer_profile": "conditioning_split_lr_v2",
                "generator_text_learning_rate": 2.5e-6,
                "generator_text_min_learning_rate": 2.5e-6 * 0.1,
                "generator_film_learning_rate": 0.0,
                "generator_film_min_learning_rate": 0.0,
                "generator_conditioning_learning_rate": 1.0e-5,
                "generator_conditioning_min_learning_rate": 1.0e-5 * 0.1,
            }
            training_path = root / "screen_job.yaml"

            def write_inputs() -> None:
                training_path.write_text(yaml.safe_dump(training), encoding="utf-8")
                study._write_signed(
                    root / "registry/materialized_worker_specs.json",
                    {
                        "kind": study.WORKER_SPEC_KIND,
                        "study_kind": study.ARCHITECTURE_KIND,
                        "jobs": [
                            {
                                "job_id": "screen_job",
                                "job_spec_sha256": "spec",
                                "training_config_path": str(training_path),
                                "training_config_sha256": study._sha256_file(
                                    training_path
                                ),
                            }
                        ],
                    },
                )

            write_inputs()
            self.assertEqual(
                study._materialized_worker_spec(root, job)["job_id"], "screen_job"
            )
            training["generator_text_min_learning_rate"] = math.nextafter(
                2.5e-6 * 0.1, math.inf
            )
            write_inputs()
            with self.assertRaisesRegex(ValueError, "Conditioning LR contract drift"):
                study._materialized_worker_spec(root, job)


if __name__ == "__main__":
    unittest.main()
