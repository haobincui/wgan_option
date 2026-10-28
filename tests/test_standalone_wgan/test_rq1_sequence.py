from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import (  # noqa: E402
    FilmWGANTrainConfig,
    build_sample_config,
    config_to_dict,
)
from film_wgan.inference import FilmWGANSampler, _short_atm_metrics  # noqa: E402
from film_wgan.sequence_data import (  # noqa: E402
    create_sequence_data_bundle,
    load_sequence_samples,
)
from film_wgan.models import TemporalFilmWGANGenerator  # noqa: E402
from film_wgan.trainer import FilmWGANTrainer  # noqa: E402
from scripts.rq1_sequence.dataset_builder import (  # noqa: E402
    NewsWindow,
    SequenceCandidate,
    _build_news_windows,
    _read_market_index,
)
from scripts.rq1_sequence.rq1_sequence_experiment import (  # noqa: E402
    CONTINUED_VARIANT,
    MATCHED_VARIANT,
    VARIANTS,
    _cluster_bootstrap,
    _paired_differences,
)
from wgan_option.market.treasury_sessions import (  # noqa: E402
    TreasuryGlobexSessionCalendar,
)
from wgan_option.surface_generation.market_index import (  # noqa: E402
    connect_market_index,
    insert_surface_anchor,
    refresh_valid_pairs,
    set_metadata,
)


def _pair_id(current: str, target: str) -> str:
    return hashlib.sha256(f"{current}|{target}".encode("utf-8")).hexdigest()[:20]


class RQ1SequenceDataTests(unittest.TestCase):
    def _write_bundle(self, root: Path) -> tuple[Path, Path]:
        sample_count = 6
        history = np.full((sample_count, 12, 2, 2), 0.20, dtype=np.float32)
        support = np.ones_like(history, dtype=np.uint8)
        steps = np.ones((sample_count, 12), dtype=np.uint8)
        steps[0, :6] = 0
        support[0, :6] = 0
        history[0, :6] = 0
        targets = np.full((sample_count, 2, 2), 0.21, dtype=np.float32)
        matched = np.ones((sample_count, 389), dtype=np.float32)
        count = np.zeros_like(matched)
        count[:, -5:] = 1.0
        shuffled = np.full_like(matched, 2.0)
        has_news = np.asarray([1, 0, 1, 1, 0, 1], dtype=np.float32)
        no_news = has_news <= 0.0
        matched[no_news] = 0.0
        count[no_news] = 0.0
        shuffled[no_news] = 0.0
        npz_path = root / "sequence_dataset.npz"
        np.savez_compressed(
            npz_path,
            history_surfaces=history,
            history_support_masks=support,
            history_step_masks=steps,
            target_surfaces=targets,
            target_support_masks=np.ones_like(targets, dtype=np.uint8),
            matched_text_features=matched,
            count_text_features=count,
            shuffled_text_features=shuffled,
            has_news=has_news,
            strike_grid=np.asarray([0.99, 1.01], dtype=np.float32),
            maturity_days_grid=np.asarray([7.0, 21.0], dtype=np.float32),
        )
        rows = []
        for index in range(sample_count):
            current = f"2023-01-03T10:{index * 5:02d}:00Z"
            target = f"2023-01-03T10:{index * 5 + 5:02d}:00Z"
            anchors = [
                (
                    pd.Timestamp(current) - pd.Timedelta(minutes=5 * offset)
                ).strftime("%Y-%m-%dT%H:%M:00Z")
                for offset in range(11, -1, -1)
            ]
            rows.append(
                {
                    "global_index": index,
                    "sample_id": f"seq_{index}",
                    "surface_pair_id": _pair_id(current, target),
                    "split": ("train", "train", "val", "val", "test", "test")[index],
                    "current_snapshot_time_utc": current,
                    "target_snapshot_time_utc": target,
                    "history_anchor_times_json": json.dumps(anchors),
                    "history_valid_steps": int(steps[index].sum()),
                    "supported_cell_count": 4,
                    "short_atm_supported_cell_count": 4,
                    "short_atm_eligible": 1,
                    "trading_day": "2023-01-03",
                    "news_count_total": int(has_news[index]),
                }
            )
        manifest_path = root / "split_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        return npz_path, manifest_path

    def _config(
        self,
        npz_path: Path,
        manifest_path: Path,
        *,
        variant: str,
        policy: str = "full_history",
    ) -> FilmWGANTrainConfig:
        return FilmWGANTrainConfig(
            data_path=str(npz_path),
            sample_unit="surface_pair",
            split_strategy="predefined_manifest",
            split_manifest_path=str(manifest_path),
            surface_input_mode="rolling_history",
            history_steps=12,
            history_min_valid_steps=6,
            history_input_policy=policy,
            temporal_encoder_mode="masked_gru",
            sequence_text_variant=variant,
            text_output_dim=389,
            normalize_text_embedding=False,
            conditioning_mode="residual_film",
            min_samples_for_training=1,
        )

    def test_no_text_is_strict_zero_and_current_only_masks_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path, manifest_path = self._write_bundle(Path(temp_dir))
            current_config = self._config(
                npz_path,
                manifest_path,
                variant="no_text",
                policy="current_only",
            )
            samples = load_sequence_samples(current_config)
            current_bundle = create_sequence_data_bundle(current_config)
            full_bundle = create_sequence_data_bundle(
                self._config(
                    npz_path,
                    manifest_path,
                    variant="no_text",
                    policy="full_history",
                )
            )
            history_batch = next(iter(current_bundle.train_loader))[0]
        self.assertEqual(len(samples), 6)
        for sample in samples:
            np.testing.assert_array_equal(sample.text_embedding, np.zeros(389))
            self.assertEqual(float(sample.metadata["has_text"]), 0.0)
            self.assertGreaterEqual(int(sample.history_step_mask.sum()), 6)
            self.assertEqual(sample.metadata["history_effective_valid_steps"], 1)
        step_counts = history_batch[:, :, 2].mean(dim=(2, 3)).sum(dim=1)
        torch.testing.assert_close(step_counts, torch.ones_like(step_counts))
        np.testing.assert_array_equal(
            current_bundle.normalization_stats.current_log_mean,
            full_bundle.normalization_stats.current_log_mean,
        )

    def test_no_news_rows_are_zero_for_matched_lp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            npz_path, manifest_path = self._write_bundle(Path(temp_dir))
            samples = load_sequence_samples(
                self._config(npz_path, manifest_path, variant="matched_lp")
            )
        self.assertGreater(float(np.linalg.norm(samples[0].text_embedding)), 0.0)
        self.assertEqual(float(np.linalg.norm(samples[1].text_embedding)), 0.0)
        self.assertEqual(float(samples[1].metadata["has_text"]), 0.0)

    def test_temporal_checkpoint_generate_result_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            npz_path, manifest_path = self._write_bundle(root)
            config = replace(
                self._config(
                    npz_path,
                    manifest_path,
                    variant="matched_lp",
                ),
                cuda=False,
                output_root=str(root / "training"),
                gen_base_channels=4,
                disc_base_channels=4,
                gen_res_blocks=1,
                disc_res_blocks=1,
                temporal_hidden_dim=16,
                text_hidden_dim=16,
                text_out_dim=8,
                fusion_hidden_dim=24,
                batch_size=2,
                num_workers=0,
                num_epochs=1,
                critic_iter=1,
                eval_mc_samples=2,
                lambda_gp=0.0,
                checkpoint_warmup_epochs=0,
                use_early_stopping=False,
            )
            trainer = FilmWGANTrainer(config)
            trainer.train()
            checkpoint_path = trainer.run_dir / "checkpoints/film_wgan_best.pt"
            self.assertTrue(checkpoint_path.is_file())
            output_path = trainer.run_dir / "temporal_smoke_generate"
            sample_config = build_sample_config(
                training_values=config_to_dict(trainer.config),
                checkpoint_path=checkpoint_path,
                overrides={
                    "split": "test",
                    "selection_mode": "head",
                    "selection_count": 1,
                    "mc_samples": 2,
                    "cuda": False,
                    "output_dir": str(output_path),
                    "save_json": True,
                    "save_plots": False,
                    "save_full_atm_timeseries": False,
                },
            )
            result_path = FilmWGANSampler(sample_config).sample()
            summary = pd.read_csv(result_path / "summary.csv")
            self.assertEqual(len(summary), 1)
            self.assertEqual(len(list((result_path / "samples").glob("*.json"))), 1)
            for column in (
                "history_observed_valid_steps",
                "history_effective_valid_steps",
                "short_atm_supported_cell_count",
                "short_atm_metric_status",
            ):
                self.assertIn(column, summary.columns)


class RQ1TemporalModelTests(unittest.TestCase):
    def _generator(self) -> TemporalFilmWGANGenerator:
        return TemporalFilmWGANGenerator(
            surface_height=4,
            surface_width=4,
            history_steps=12,
            embedding_dim=389,
            noise_dim=8,
            base_channels=4,
            res_blocks=1,
            temporal_hidden_dim=16,
            text_hidden_dim=12,
            text_out_dim=8,
            fusion_hidden_dim=24,
            text_dropout=0.0,
            text_gate_initial_value=0.0,
        ).eval()

    def test_zero_gate_matches_no_text_output(self):
        torch.manual_seed(7)
        generator = self._generator()
        history = torch.randn(2, 12, 3, 4, 4)
        history[:, :, 2] = 1.0
        noise = torch.randn(2, 8)
        text = torch.randn(2, 389)
        with torch.no_grad():
            conditioned = generator(history, text, noise, torch.ones(2))
            no_text = generator(history, torch.zeros_like(text), noise, torch.zeros(2))
        torch.testing.assert_close(conditioned, no_text, rtol=0.0, atol=0.0)

    def test_masked_history_step_does_not_update_hidden_state(self):
        torch.manual_seed(9)
        generator = self._generator()
        first = torch.randn(1, 12, 3, 4, 4)
        first[:, :, 2] = 1.0
        first[:, 3, 2] = 0.0
        second = first.clone()
        second[:, 3, :2] = 1_000.0
        noise = torch.randn(1, 8)
        text = torch.zeros(1, 389)
        with torch.no_grad():
            output_a = generator(first, text, noise, torch.zeros(1))
            output_b = generator(second, text, noise, torch.zeros(1))
        torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)


class RQ1SequenceNewsWindowTests(unittest.TestCase):
    def test_toy_raw_sqlite_builds_same_session_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "market_surface_index.sqlite"
            connection = connect_market_index(database_path)
            params = {
                "business_days": [5, 21],
                "percent_strikes": [[0.98, 1.02], [0.98, 1.02]],
                "implied_vols": [[0.20, 0.21], [0.22, 0.23]],
            }
            anchors = pd.date_range(
                "2023-01-03T13:05:00Z",
                "2023-01-03T14:05:00Z",
                freq="5min",
            )
            try:
                set_metadata(
                    connection,
                    "immutable_settings",
                    {
                        "window_minutes": 5,
                        "candidate_anchor_source": (
                            "positive_price_volume_option_minutes"
                        ),
                    },
                )
                for anchor in anchors:
                    insert_surface_anchor(
                        connection,
                        anchor_time_utc=anchor,
                        surface_model="raw",
                        surface_params=params,
                        source_target_utc=anchor,
                        source_direction="backward",
                    )
                self.assertEqual(refresh_valid_pairs(connection), 12)
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                connection.close()

            surface_anchors, candidates, audit = _read_market_index(
                database_path,
                TreasuryGlobexSessionCalendar(),
            )
            self.assertEqual(len(surface_anchors), 13)
            self.assertEqual(len(candidates), 7)
            self.assertEqual(audit["surface_model_counts"], {"raw": 13})
            self.assertTrue(all(sum(item.history_valid) >= 6 for item in candidates))
            self.assertTrue(all(item.history_valid[-1] for item in candidates))
            self.assertEqual(len({item.session_id for item in candidates}), 1)

    def test_future_news_is_not_included(self):
        candidate = SequenceCandidate(
            origin=pd.Timestamp("2023-01-03T10:00:00Z"),
            target=pd.Timestamp("2023-01-03T10:05:00Z"),
            history_anchors=tuple(
                pd.Timestamp("2023-01-03T10:00:00Z")
                - pd.Timedelta(minutes=5 * offset)
                for offset in range(11, -1, -1)
            ),
            history_valid=tuple([True] * 12),
            session_id="toy",
        )
        news = pd.DataFrame(
            {
                "available_time": pd.to_datetime(
                    ["2023-01-03T09:59:00Z", "2023-01-03T10:01:00Z"],
                    utc=True,
                ),
                "article_key": ["past", "future"],
                "embedding_sha256": ["past_hash", "future_hash"],
            }
        )
        catalog = {
            "past": np.asarray([1.0, 0.0], dtype=np.float32),
            "future": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        windows, _audit = _build_news_windows([candidate], news, catalog, 2)
        window: NewsWindow = windows[candidate.pair_id]
        self.assertEqual(window.article_ids, ("past",))
        self.assertNotIn("future", window.article_ids)


class RQ1SequenceStatisticsTests(unittest.TestCase):
    def test_short_atm_without_supported_cells_is_missing_not_zero(self):
        surface = np.full((2, 2), 0.2, dtype=np.float32)
        metrics = _short_atm_metrics(
            generated_surface=surface,
            current_surface=surface,
            target_surface=surface,
            recon_weights_surface=torch.ones((2, 2)),
            atm_short_mask_surface=torch.ones((2, 2)),
            support_mask=np.zeros((2, 2), dtype=bool),
        )
        self.assertTrue(np.isnan(float(metrics["atm_short_pure_mae"])))
        self.assertEqual(float(metrics["short_atm_supported_cell_count"]), 0.0)
        self.assertEqual(
            metrics["short_atm_metric_status"],
            "no_supported_short_atm_cells",
        )

    def test_comparator_minus_focal_direction_is_positive_for_better_lp(self):
        rows = []
        for variant in VARIANTS:
            error = 1.0 if variant == MATCHED_VARIANT else 2.0
            if variant == CONTINUED_VARIANT:
                error = 3.0
            rows.append(
                {
                    "fold": "2023Q1",
                    "seed": 42,
                    "surface_pair_id": "pair",
                    "trading_day": "2023-01-03",
                    "current_snapshot_time_utc": "2023-01-03T10:00:00Z",
                    "variant": variant,
                    "surface_mae": error,
                    "short_atm_mae": error,
                    "energy_score": error,
                    "calibration_error": error,
                    "calendar_violation_rate": error,
                    "butterfly_violation_rate": error,
                    "current_mae": 4.0,
                    "current_atm_short_pure_mae": 4.0,
                }
            )
        differences = _paired_differences(pd.DataFrame(rows))
        primary = differences[
            (differences["contrast"] == "incremental_lp_text")
            & (differences["metric"] == "surface_mae")
        ].iloc[0]
        self.assertEqual(float(primary["difference"]), 2.0)
        self.assertEqual(primary["difference_direction"], "comparator_minus_focal")

    def test_cluster_bootstrap_reports_positive_mean(self):
        frame = pd.DataFrame(
            {
                "trading_day": ["2023-01-03", "2023-01-03", "2023-01-04"],
                "difference": [1.0, 2.0, 3.0],
            }
        )
        result = _cluster_bootstrap(frame, iterations=100, seed=7)
        self.assertEqual(float(result["mean_difference"]), 2.0)
        self.assertEqual(int(result["cluster_count"]), 2)


if __name__ == "__main__":
    unittest.main()
