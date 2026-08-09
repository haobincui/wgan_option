import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from film_wgan.config import FilmWGANTrainConfig, _validate_train_fields  # noqa: E402
from film_wgan.io import write_csv  # noqa: E402
from film_wgan.models import FilmWGANCritic, FilmWGANGenerator  # noqa: E402
from film_wgan.protocol import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION_V3,
    CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING,
    DIAGNOSTICS_SCHEMA_VERSION,
    MATCHING_NEGATIVE_SOURCE_PLAN_VERSION,
    TEXT_ALIGNMENT_PLAN_VERSION,
    TRAINING_PROTOCOL_VERSION_V3,
)
from film_wgan.trainer import (  # noqa: E402
    FilmWGANTrainer,
    aggregate_gradient_penalty_diagnostics,
    aggregate_transition_delivery_diagnostics,
    summarize_matching_logits,
)


class TestExactDiagnosticAggregationV3(unittest.TestCase):
    def test_gp_aggregation_is_sample_weighted_and_has_exact_quantiles(self):
        rows = [
            {
                "gp_raw_norm_count": 2.0,
                "gp_raw_norm_sum": 1.0,
                "gp_raw_norm_sum_squares": 1.0,
                "gp_raw_norm_min": 0.0,
                "gp_raw_norm_max": 1.0,
                "gp_raw_norm_outside_count": 1.0,
                "gp_unscaled_penalty_sum": 1.0,
                "gp_unsupported_max_abs_gradient": 0.0,
                "gp_raw_norm_values": [0.0, 1.0],
            },
            {
                "gp_raw_norm_count": 1.0,
                "gp_raw_norm_sum": 2.0,
                "gp_raw_norm_sum_squares": 4.0,
                "gp_raw_norm_min": 2.0,
                "gp_raw_norm_max": 2.0,
                "gp_raw_norm_outside_count": 1.0,
                "gp_unscaled_penalty_sum": 1.0,
                "gp_unsupported_max_abs_gradient": 1.0e-8,
                "gp_raw_norm_values": [2.0],
            },
        ]
        metrics = aggregate_gradient_penalty_diagnostics(rows)
        self.assertAlmostEqual(metrics["gp_raw_norm_mean"], 1.0)
        self.assertAlmostEqual(metrics["gp_raw_norm_std"], math.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(metrics["gp_raw_norm_p05"], 0.1)
        self.assertAlmostEqual(metrics["gp_raw_norm_p50"], 1.0)
        self.assertAlmostEqual(metrics["gp_raw_norm_p95"], 1.9)
        self.assertAlmostEqual(metrics["gp_unscaled_penalty"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["gp_raw_norm_outside_0p5_1p5_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["gp_unsupported_max_abs_gradient"], 1.0e-8)

    def test_transition_aggregation_is_cell_weighted(self):
        metrics = aggregate_transition_delivery_diagnostics(
            [
                {
                    "g_transition_supported_count": 2.0,
                    "g_transition_clipped_count": 1.0,
                    "g_transition_clipped_abs_gap_sum": 0.5,
                    "g_transition_clipped_max_abs_gap": 0.5,
                    "g_transition_unclipped_raw_log_max_abs_error": 1.0e-7,
                    "g_transition_unclipped_normalized_max_abs_error": 2.0e-7,
                },
                {
                    "g_transition_supported_count": 3.0,
                    "g_transition_clipped_count": 1.0,
                    "g_transition_clipped_abs_gap_sum": 0.2,
                    "g_transition_clipped_max_abs_gap": 0.2,
                    "g_transition_unclipped_raw_log_max_abs_error": 3.0e-7,
                    "g_transition_unclipped_normalized_max_abs_error": 4.0e-7,
                },
            ]
        )
        self.assertAlmostEqual(metrics["g_transition_clipped_fraction"], 0.4)
        self.assertAlmostEqual(metrics["g_transition_clipped_mean_abs_gap"], 0.35)
        self.assertAlmostEqual(metrics["g_transition_clipped_max_abs_gap"], 0.5)
        self.assertEqual(metrics["g_transition_unclipped_raw_log_max_abs_error"], 3.0e-7)

    def test_matching_summary_aggregates_at_target_level(self):
        metrics = summarize_matching_logits(
            torch.tensor([2.0, 0.0]),
            torch.tensor([[1.0, 3.0], [-1.0, -2.0]]),
            total_targets=4,
            prefix="heldout",
        )
        self.assertEqual(metrics["heldout_eligible_targets"], 2.0)
        self.assertEqual(metrics["heldout_eligible_fraction"], 0.5)
        self.assertEqual(metrics["heldout_pairwise_accuracy"], 0.75)


class _SpyMatchingCritic(torch.nn.Module):
    conditioning_mode = "transition_matching"

    def __init__(self):
        super().__init__()
        self.seen_text: list[torch.Tensor] = []

    def matching_logits(self, transition, text, support_mask=None, has_text=None):
        del transition, support_mask, has_text
        self.seen_text.append(text.detach().clone())
        return text[:, :1]


class _TinyProbeGenerator(torch.nn.Module):
    conditioning_mode = "residual_film"
    noise_dim = 0

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.full((4,), 0.01))
        self.adapter = torch.nn.Parameter(torch.full((4,), 0.002))

    def forward(self, current, text, noise=None, has_text=None):
        del current, noise, has_text
        return self.backbone.unsqueeze(0) + text[:, :1] * self.adapter.unsqueeze(0)

    def text_adapter_parameters(self):
        return iter((self.adapter,))

    def film_regularization(self):
        return self.adapter.square().mean()


class _TinyProbeCritic(torch.nn.Module):
    conditioning_mode = "transition_matching"

    def forward(self, future, current, text, has_text=None, support_mask=None):
        del current, text, has_text
        support = torch.ones_like(future) if support_mask is None else support_mask
        return (future * support).flatten(start_dim=1).mean(dim=1, keepdim=True)

    def matching_logits(self, transition, text, support_mask=None, has_text=None):
        del has_text
        support = torch.ones_like(transition) if support_mask is None else support_mask
        encoded = (transition * support).flatten(start_dim=1).sum(dim=1, keepdim=True)
        encoded = encoded / support.flatten(start_dim=1).sum(dim=1, keepdim=True)
        return encoded * text[:, :1]


class _FailingProbeCritic(_TinyProbeCritic):
    def forward(self, future, current, text, has_text=None, support_mask=None):
        del future, current, text, has_text, support_mask
        raise RuntimeError("intentional probe failure")


class _ProbeDataset:
    def __init__(
        self,
        has_text_values: list[float],
        supported_cell_counts: list[int],
    ):
        self.has_text_values = has_text_values
        self.supported_cell_counts = supported_cell_counts

    def __len__(self):
        return len(self.has_text_values)

    def __getitem__(self, index: int):
        support = torch.zeros(4, dtype=torch.float32)
        support[: self.supported_cell_counts[index]] = 1.0
        return (
            torch.zeros(2, 2, 2, dtype=torch.float32),
            torch.tensor([float(index + 1), 1.0], dtype=torch.float32),
            torch.zeros(4, dtype=torch.float32),
            torch.full((4,), 0.2, dtype=torch.float32),
            torch.full((4,), 0.21, dtype=torch.float32),
            support,
            torch.tensor(self.has_text_values[index], dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class TestCanonicalMatcherWiringV3(unittest.TestCase):
    def test_carrier_text_cannot_change_canonical_positive_logits(self):
        trainer = FilmWGANTrainer(FilmWGANTrainConfig(cuda=False))
        critic = _SpyMatchingCritic()
        trainer.critic = critic
        bank = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        trainer._matching_text_banks_by_split["train"] = bank
        trainer._matching_positive_source_indices_by_split["train"] = torch.tensor([2, 0])
        transition = torch.ones(2, 1, 2, 2)
        support = torch.ones_like(transition)
        eligible = torch.tensor([0, 1])
        donors = torch.tensor([[1], [2]])
        indices = torch.tensor([0, 1])
        first = trainer._transition_matching_logits(
            transition,
            torch.full((2, 2), 999.0),
            support,
            eligible,
            donors,
            sample_indices=indices,
        )
        second = trainer._transition_matching_logits(
            transition,
            torch.full((2, 2), -999.0),
            support,
            eligible,
            donors,
            sample_indices=indices,
        )
        torch.testing.assert_close(critic.seen_text[0], bank[torch.tensor([2, 0])])
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_autograd_probe_helper_does_not_write_parameter_grad(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        parameter.grad = torch.tensor([7.0, 8.0])
        before = parameter.grad.clone()
        loss = parameter.square().sum()
        all_norm, adapter_norm = FilmWGANTrainer._probe_gradient_norms(
            loss,
            [parameter],
            adapter_parameter_ids={id(parameter)},
            retain_graph=False,
        )
        self.assertAlmostEqual(all_norm, math.sqrt(20.0), places=6)
        self.assertEqual(all_norm, adapter_norm)
        torch.testing.assert_close(parameter.grad, before)
        self.assertTrue(parameter.requires_grad)

    @staticmethod
    def _tiny_diagnostic_trainer() -> FilmWGANTrainer:
        config = FilmWGANTrainConfig(
            cuda=False,
            conditioning_mode="residual_film",
            critic_conditioning_mode="transition_matching",
            normalize_target_delta=True,
            normalize_current_surface=True,
            normalize_text_embedding=False,
            lambda_adv=0.1,
            lambda_critic_matching=0.1,
            lambda_generator_matching=0.01,
            matching_negative_count=1,
            matching_min_supported_cells=1,
            use_calendar_constraint=False,
            use_butterfly_constraint=False,
            use_smooth_constraint=False,
            use_recon_constraint=True,
            lambda_recon=1.0,
            use_atm_short_loss=False,
            adv_warmup_epochs=0,
        )
        trainer = FilmWGANTrainer(config)
        trainer.generator = _TinyProbeGenerator()
        trainer.critic = _TinyProbeCritic()
        trainer.normalization = SimpleNamespace(
            current_log_mean=torch.zeros(4),
            current_log_std=torch.ones(4),
            delta_mean=torch.zeros(4),
            delta_std=torch.ones(4),
            text_mean=torch.zeros(2),
            text_std=torch.ones(2),
        )
        trainer.bundle = SimpleNamespace(
            surface_shape=(2, 2),
            val_items=[object(), object(), object()],
        )
        trainer._recon_weights_flat = torch.ones(4)
        trainer._atm_short_mask_flat = torch.ones(4)
        text = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        current = torch.zeros(3, 1, 2, 2)
        current_flat = torch.full((3, 4), 0.2)
        target = current_flat + 0.01
        delta = torch.zeros(3, 4)
        has_text = torch.ones(3)
        sample_indices = torch.arange(3)
        trainer._matching_gradient_probe_batch = (
            current,
            text,
            delta,
            current_flat,
            target,
            has_text,
            sample_indices,
        )
        trainer._matching_donor_indices_by_split["train"] = torch.tensor(
            [[1], [2], [0]]
        )
        trainer._matching_text_banks_by_split["train"] = text
        trainer._matching_positive_source_indices_by_split["train"] = torch.arange(3)
        return trainer

    @classmethod
    def _prepared_probe_trainer(
        cls,
        *,
        has_text_values: list[float],
        supported_cell_counts: list[int],
    ) -> FilmWGANTrainer:
        trainer = cls._tiny_diagnostic_trainer()
        trainer.config = FilmWGANTrainConfig(
            **{
                **trainer.config.__dict__,
                "surface_support_mode": "raw_observed",
                "matching_gradient_probe_size": 64,
                "matching_gradient_probe_seed": 20260809,
            }
        )
        dataset = _ProbeDataset(has_text_values, supported_cell_counts)
        trainer.bundle.train_loader = SimpleNamespace(dataset=dataset)
        sample_count = len(dataset)
        trainer._matching_donor_indices_by_split["train"] = torch.stack(
            [
                (torch.arange(sample_count) + 1) % sample_count,
            ],
            dim=1,
        )
        trainer._matching_text_banks_by_split["train"] = torch.stack(
            [dataset[index][1] for index in range(sample_count)]
        )
        trainer._matching_positive_source_indices_by_split["train"] = torch.arange(
            sample_count
        )
        trainer._prepare_matching_gradient_probe()
        return trainer

    def test_zero_lp_probe_preparation_has_no_carrier_eligible_rows(self):
        trainer = self._prepared_probe_trainer(
            has_text_values=[0.0] * 70,
            supported_cell_counts=[4] * 70,
        )
        self.assertIsNone(trainer._matching_gradient_probe_batch)
        metrics = trainer._evaluate_matching_gradient_probe()
        self.assertEqual(metrics["diag_g_probe_active"], 0.0)
        self.assertEqual(metrics["diag_g_probe_eligible_targets"], 0.0)
        self.assertTrue(math.isnan(metrics["diag_g_all_parameter_ratio"]))

    def test_probe_preparation_filters_insufficient_carrier_support(self):
        trainer = self._prepared_probe_trainer(
            has_text_values=[1.0] * 70,
            supported_cell_counts=[0] * 70,
        )
        self.assertIsNone(trainer._matching_gradient_probe_batch)

    def test_matched_probe_selects_64_and_remains_active(self):
        trainer = self._prepared_probe_trainer(
            has_text_values=[1.0] * 70,
            supported_cell_counts=[4] * 70,
        )
        self.assertIsNotNone(trainer._matching_gradient_probe_batch)
        self.assertEqual(trainer._matching_gradient_probe_batch[0].size(0), 64)
        trainer._current_epoch = 3
        metrics = trainer._evaluate_matching_gradient_probe()
        self.assertEqual(metrics["diag_g_probe_active"], 1.0)
        self.assertEqual(metrics["diag_g_probe_eligible_targets"], 64.0)

    def test_zero_weight_or_empty_carrier_never_enters_probe_forward(self):
        trainer = self._prepared_probe_trainer(
            has_text_values=[1.0] * 70,
            supported_cell_counts=[4] * 70,
        )
        trainer._current_epoch = 3
        original_forward = trainer.generator.forward

        def fail_forward(*args, **kwargs):
            del args, kwargs
            raise AssertionError("probe forward must remain inactive")

        trainer.generator.forward = fail_forward
        try:
            trainer.config.lambda_generator_matching = 0.0
            zero_weight = trainer._evaluate_matching_gradient_probe()
            self.assertEqual(zero_weight["diag_g_probe_active"], 0.0)
            self.assertEqual(zero_weight["diag_g_probe_eligible_targets"], 64.0)

            trainer.config.lambda_generator_matching = 0.01
            batch = list(trainer._matching_gradient_probe_batch)
            batch[6] = torch.zeros_like(batch[6])
            trainer._matching_gradient_probe_batch = tuple(batch)
            empty_carrier = trainer._evaluate_matching_gradient_probe()
            self.assertEqual(empty_carrier["diag_g_probe_active"], 0.0)
            self.assertEqual(empty_carrier["diag_g_probe_eligible_targets"], 0.0)
        finally:
            trainer.generator.forward = original_forward

    def test_full_gradient_probe_is_finite_and_has_no_training_side_effects(self):
        trainer = self._tiny_diagnostic_trainer()
        trainer._current_epoch = 3
        trainer.generator.eval()
        trainer.critic.train()
        trainer.generator.backbone.requires_grad_(False)
        for parameter in trainer.generator.parameters():
            parameter.grad = torch.full_like(parameter, 7.0)
        grads_before = [parameter.grad.clone() for parameter in trainer.generator.parameters()]
        params_before = [parameter.detach().clone() for parameter in trainer.generator.parameters()]
        flags_before = [
            parameter.requires_grad for parameter in trainer.generator.parameters()
        ]
        rng_before = torch.random.get_rng_state().clone()
        metrics = trainer._evaluate_matching_gradient_probe()
        self.assertEqual(metrics["diag_g_probe_active"], 1.0)
        self.assertEqual(metrics["diag_g_probe_eligible_targets"], 3.0)
        self.assertTrue(math.isfinite(metrics["diag_g_all_parameter_ratio"]))
        self.assertGreaterEqual(
            metrics["diag_g_matching_output_grad_rms_p95"],
            metrics["diag_g_matching_output_grad_rms_median"],
        )
        self.assertFalse(trainer.generator.training)
        self.assertTrue(trainer.critic.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertNotEqual(
            metrics["diag_g_all_parameter_ratio"],
            metrics["diag_g_text_adapter_ratio"],
        )
        for parameter, value, gradient, requires_grad in zip(
            trainer.generator.parameters(), params_before, grads_before, flags_before
        ):
            torch.testing.assert_close(parameter, value)
            self.assertTrue(torch.equal(parameter.grad, gradient))
            self.assertEqual(parameter.requires_grad, requires_grad)

    def test_gradient_probe_exception_restores_flags_and_grad_bit_patterns(self):
        trainer = self._tiny_diagnostic_trainer()
        trainer._current_epoch = 3
        trainer.critic = _FailingProbeCritic()
        trainer.generator.backbone.requires_grad_(False)
        trainer.generator.backbone.grad = None
        trainer.generator.adapter.grad = torch.tensor(
            [7.0, -3.0, 2.0, 11.0], dtype=torch.float32
        )
        flags_before = [
            parameter.requires_grad for parameter in trainer.generator.parameters()
        ]
        gradients_before = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in trainer.generator.parameters()
        ]
        with self.assertRaisesRegex(RuntimeError, "intentional probe failure"):
            trainer._evaluate_matching_gradient_probe()
        for parameter, requires_grad, gradient in zip(
            trainer.generator.parameters(), flags_before, gradients_before
        ):
            self.assertEqual(parameter.requires_grad, requires_grad)
            if gradient is None:
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.equal(parameter.grad, gradient))

    def test_inactive_gradient_probe_reports_nan_ratios(self):
        trainer = self._tiny_diagnostic_trainer()
        trainer.config = FilmWGANTrainConfig(
            **{
                **trainer.config.__dict__,
                "adv_warmup_epochs": 2,
            }
        )
        trainer._current_epoch = 1
        metrics = trainer._evaluate_matching_gradient_probe()
        self.assertEqual(metrics["diag_g_probe_active"], 0.0)
        self.assertTrue(math.isnan(metrics["diag_g_all_parameter_ratio"]))

    def test_probe_schema_is_stable_across_warmup_and_active_epochs(self):
        trainer = self._tiny_diagnostic_trainer()
        trainer.config = FilmWGANTrainConfig(
            **{
                **trainer.config.__dict__,
                "adv_warmup_epochs": 2,
            }
        )
        rows = []
        for epoch in (1, 2, 3):
            trainer._current_epoch = epoch
            rows.append(
                {
                    "epoch": epoch,
                    **trainer._evaluate_matching_gradient_probe(),
                }
            )

        self.assertEqual(rows[0].keys(), rows[1].keys())
        self.assertEqual(rows[0].keys(), rows[2].keys())
        self.assertEqual(rows[0]["diag_g_probe_active"], 0.0)
        self.assertEqual(rows[1]["diag_g_probe_active"], 0.0)
        self.assertEqual(rows[2]["diag_g_probe_active"], 1.0)
        self.assertEqual(rows[0]["g_matching_gradient_probe_samples"], 3.0)
        self.assertEqual(
            rows[0]["g_matching_gradient_probe_effective_lambda"], 0.0
        )
        self.assertTrue(math.isnan(rows[0]["g_matching_gradient_norm_all"]))
        self.assertTrue(
            math.isfinite(rows[2]["g_matching_gradient_norm_all"])
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_metrics.csv"
            write_csv(path, rows)
            with path.open("r", encoding="utf-8", newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(len(written), 3)
            self.assertIn("g_nonmatching_gradient_norm_text_adapter", written[0])
            self.assertIn("g_matching_gradient_ratio_text_adapter", written[0])

    def test_heldout_diagnostics_restore_modes_rng_and_parameters(self):
        trainer = self._tiny_diagnostic_trainer()
        batch = trainer._matching_gradient_probe_batch
        trainer.bundle.val_loader = [batch]
        trainer._matching_donor_indices_by_split["val"] = torch.tensor(
            [[1], [2], [0]]
        )
        trainer._matching_text_banks_by_split["val"] = (
            trainer._matching_text_banks_by_split["train"].clone()
        )
        trainer._matching_positive_source_indices_by_split["val"] = torch.arange(3)
        trainer.generator.train()
        trainer.critic.eval()
        params_before = [parameter.detach().clone() for parameter in trainer.generator.parameters()]
        rng_before = torch.random.get_rng_state().clone()
        metrics = trainer._evaluate_matching_diagnostics()
        self.assertIn("val_matching_real_pairwise_accuracy", metrics)
        self.assertEqual(metrics["val_matching_mask_only_pairwise_accuracy"], 0.5)
        self.assertTrue(trainer.generator.training)
        self.assertFalse(trainer.critic.training)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        for parameter, value in zip(trainer.generator.parameters(), params_before):
            torch.testing.assert_close(parameter, value)

    def test_generator_step_reports_exact_supported_clip_counts(self):
        trainer = self._tiny_diagnostic_trainer()
        trainer.generator.backbone.data.copy_(
            torch.tensor([20.0, -20.0, 0.01, 0.02])
        )
        trainer.generator_optimizer = torch.optim.Adam(
            trainer.generator.parameters(), lr=1.0e-4
        )
        trainer._strike_grid = torch.arange(2, dtype=torch.float32)
        trainer._maturity_days_grid = torch.arange(2, dtype=torch.float32)
        batch = trainer._matching_gradient_probe_batch
        current, text, _delta, current_flat, target, has_text, indices = batch
        support = torch.tensor([[1.0, 0.0, 1.0, 1.0]]).expand(3, -1).clone()
        trainer._current_epoch = 3
        metrics = trainer._generator_step(
            current,
            text,
            current_flat,
            target,
            has_text,
            support,
            indices,
        )
        self.assertEqual(metrics["g_transition_supported_count"], 9.0)
        self.assertEqual(metrics["g_transition_clipped_count"], 3.0)
        self.assertAlmostEqual(metrics["g_saturation_rate"], 1.0 / 3.0)
        self.assertLess(
            metrics["g_transition_unclipped_raw_log_max_abs_error"], 1.0e-5
        )


class TestCheckpointProtocolV3(unittest.TestCase):
    def test_explicit_v3_checkpoint_persists_protocol_and_diagnostics(self):
        config = FilmWGANTrainConfig(
            cuda=False,
            training_protocol_version=TRAINING_PROTOCOL_VERSION_V3,
            run_fingerprint_sha256="a" * 64,
        )
        trainer = FilmWGANTrainer(config)
        trainer.generator = FilmWGANGenerator(
            surface_height=4,
            surface_width=4,
            embedding_dim=3,
            noise_dim=2,
            base_channels=4,
            res_blocks=0,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=8,
            conditioning_mode="residual_film",
        )
        trainer.critic = FilmWGANCritic(
            surface_height=4,
            surface_width=4,
            embedding_dim=3,
            base_channels=4,
            res_blocks=0,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=8,
            conditioning_mode="residual_film",
            critic_conditioning_mode="transition_matching",
        )
        stats = SimpleNamespace(
            current_log_mean=np.zeros(16),
            current_log_std=np.ones(16),
            delta_mean=np.zeros(16),
            delta_std=np.ones(16),
            text_mean=np.zeros(3),
            text_std=np.ones(3),
        )
        trainer.bundle = SimpleNamespace(
            surface_shape=(4, 4),
            strike_grid=np.arange(4),
            maturity_days_grid=np.arange(4),
            embedding_dim=3,
            normalization_stats=stats,
            text_transform_path="",
            text_transform_sha256="",
            surface_support_path="",
            surface_support_sha256="",
        )
        trainer._latest_diagnostics = {"gp_raw_norm_mean": 1.0}
        payload = trainer._checkpoint_payload()
        self.assertEqual(payload["checkpoint_schema_version"], CHECKPOINT_SCHEMA_VERSION_V3)
        self.assertEqual(payload["training_protocol_version"], TRAINING_PROTOCOL_VERSION_V3)
        self.assertEqual(payload["run_fingerprint_sha256"], "a" * 64)
        self.assertEqual(payload["diagnostics"]["gp_raw_norm_mean"], 1.0)
        self.assertIn("matching_negative_source_plan_version", payload)
        self.assertIn("matching_eligible_samples_by_split", payload)
        self.assertEqual(payload["native_positive_as_negative_count"], 0)

    @staticmethod
    def _load_parent_with_metadata(
        directory: str,
        metadata_updates: dict[str, object] | None = None,
        *,
        remove_field: str = "",
    ) -> FilmWGANTrainer:
        generator = FilmWGANGenerator(
            surface_height=4,
            surface_width=4,
            embedding_dim=3,
            noise_dim=2,
            base_channels=4,
            res_blocks=0,
            text_hidden_dim=8,
            text_out_dim=4,
            fusion_hidden_dim=8,
            conditioning_mode="residual_film",
        )
        alignment_sha = "a" * 64
        negative_sha = "b" * 64
        checkpoint = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION_V3,
            "training_protocol_version": TRAINING_PROTOCOL_VERSION_V3,
            "critic_architecture_version": (
                CRITIC_ARCHITECTURE_VERSION_TRANSITION_MATCHING
            ),
            "run_fingerprint_sha256": "c" * 64,
            "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "text_alignment_plan_version": TEXT_ALIGNMENT_PLAN_VERSION,
            "matching_negative_source_plan_version": (
                MATCHING_NEGATIVE_SOURCE_PLAN_VERSION
            ),
            "text_alignment_plan_sha256": alignment_sha,
            "matching_negative_source_plan_sha256": negative_sha,
            "surface_shape": [4, 4],
            "embedding_dim": 3,
            "current_surface_channels": 1,
            "surface_support_sha256": "",
            "conditioning_mode": "residual_film",
            "text_transform_sha256": "",
            "generator_state_dict": generator.state_dict(),
        }
        checkpoint.update(metadata_updates or {})
        if remove_field:
            checkpoint.pop(remove_field)
        checkpoint_path = Path(directory) / "parent.pt"
        torch.save(checkpoint, checkpoint_path)
        trainer = FilmWGANTrainer(
            FilmWGANTrainConfig(
                cuda=False,
                conditioning_mode="residual_film",
                critic_conditioning_mode="transition_matching",
                training_protocol_version=TRAINING_PROTOCOL_VERSION_V3,
                initial_generator_checkpoint_path=str(checkpoint_path),
            )
        )
        trainer.generator = generator
        trainer.bundle = SimpleNamespace(
            surface_shape=(4, 4),
            embedding_dim=3,
            surface_support_sha256="",
            text_transform_sha256="",
        )
        trainer._text_alignment_plan_sha256 = alignment_sha
        trainer._matching_negative_source_plan_sha256 = negative_sha
        trainer._load_initial_generator_checkpoint()
        return trainer

    def test_v3_parent_requires_complete_protocol_metadata(self):
        required_fields = (
            "run_fingerprint_sha256",
            "diagnostics_schema_version",
            "text_alignment_plan_version",
            "matching_negative_source_plan_version",
        )
        for field_name in required_fields:
            with self.subTest(field=field_name), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, field_name):
                    self._load_parent_with_metadata(
                        directory,
                        remove_field=field_name,
                    )

    def test_v3_parent_rejects_malformed_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "run_fingerprint_sha256"
        ):
            self._load_parent_with_metadata(
                directory,
                {"run_fingerprint_sha256": "G" * 64},
            )

    def test_v3_parent_accepts_complete_protocol_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self._load_parent_with_metadata(directory)
            self.assertTrue(trainer._parent_checkpoint_sha256)


class TestV3CheckpointSelectionFirewall(unittest.TestCase):
    @staticmethod
    def _config(directory: str, **updates) -> FilmWGANTrainConfig:
        alignment = Path(directory) / "alignment.csv"
        negatives = Path(directory) / "negatives.csv"
        alignment.touch()
        negatives.touch()
        values = dict(
            cuda=False,
            conditioning_mode="residual_film",
            critic_conditioning_mode="transition_matching",
            critic_text_dropout=0.0,
            gradient_penalty_mode="support_masked",
            lambda_critic_matching=0.1,
            training_protocol_version=TRAINING_PROTOCOL_VERSION_V3,
            run_fingerprint_sha256="b" * 64,
            text_alignment_plan_path=str(alignment),
            matching_negative_source_plan_path=str(negatives),
            checkpoint_metric="val_mae",
            extra_checkpoint_metrics=[],
        )
        values.update(updates)
        return FilmWGANTrainConfig(**values)

    def test_matching_metric_cannot_select_v3_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fixes checkpoint_metric=val_mae"):
                _validate_train_fields(
                    self._config(directory, checkpoint_metric="val_matching_real_loss")
                )

    def test_extra_diagnostic_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "extra_checkpoint_metrics=\\[\\]"):
                _validate_train_fields(
                    self._config(directory, extra_checkpoint_metrics=["val_mae_gap_vs_current"])
                )

    def test_v3_protocol_locks_negative_count_and_gradient_probe(self):
        invalid_updates = (
            ("matching_negative_count", 1),
            ("matching_gradient_probe_size", 0),
            ("matching_gradient_probe_seed", 7),
        )
        with tempfile.TemporaryDirectory() as directory:
            for field_name, value in invalid_updates:
                with self.subTest(field=field_name), self.assertRaisesRegex(
                    ValueError, field_name
                ):
                    _validate_train_fields(
                        self._config(directory, **{field_name: value})
                    )

    def test_v3_protocol_locks_deterministic_critic_gp(self):
        invalid_updates = (
            ("critic_text_dropout", 0.1),
            ("gradient_penalty_mode", "legacy_full_grid"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for field_name, value in invalid_updates:
                with self.subTest(field=field_name), self.assertRaises(ValueError):
                    _validate_train_fields(
                        self._config(directory, **{field_name: value})
                    )


if __name__ == "__main__":
    unittest.main()
