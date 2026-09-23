"""Focused contracts for the FiLM text-signal probe core."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.rq3.news_first_vol_film_unet_text_signal_probe_core import (
    _job_epochs,
    _paired_job_comparison,
    _save_adapter_state,
    _set_global_gate_optimizer_phase,
    load_probe_adapter_state,
)
from wgan_option.models.generator import Generator
from wgan_option.models.text_signal_adapter import (
    SMALL_NORMAL_STARTUP_MODE,
    convert_film_unet_to_text_signal_adapter,
)


def _source_generator() -> Generator:
    torch.manual_seed(491)
    generator = Generator(
        channels=1,
        embedding_dim=1024,
        noise_dim=32,
        surface_height=16,
        surface_width=16,
        base_channels=32,
        res_blocks=0,
        text_hidden_dim=256,
        text_out_dim=128,
        hidden_dim=1024,
        residual_output_mode="identity_softplus_residual",
        generator_noise_mode="gaussian",
        generator_current_input_mode="current_support_masked",
        generator_conditioning_mode="film_unet_mask_coords_v1",
        strike_grid=torch.linspace(0.97, 1.03, steps=16),
        maturity_grid_days=torch.tensor(
            [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38]
        ),
    ).eval()
    with torch.no_grad():
        generator.residual_head.weight.normal_(0.0, 0.01)
        generator.residual_head.bias.fill_(0.005)
    return generator


def _update(
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...],
) -> None:
    current, text, noise, mask, target = batch
    optimizer.zero_grad(set_to_none=True)
    prediction = adapter(current, text, noise=noise, current_support_mask=mask)
    torch.nn.functional.l1_loss(prediction, target).backward()
    optimizer.step()


def _random_batch() -> tuple[torch.Tensor, ...]:
    current = 0.1 + 0.3 * torch.rand(1, 1, 16, 16)
    text = torch.randn(1, 1024)
    noise = torch.randn(1, 32)
    mask = torch.ones_like(current)
    target = current + 0.01 * torch.randn_like(current)
    return current, text, noise, mask, target


class ProbeStateTests(unittest.TestCase):
    def test_gated_state_requires_contracts_and_releases_on_epoch_eleven(self) -> None:
        source = _source_generator()
        optimizer_contract = {
            "mode": "separate_text_and_film_lr_v1",
            "text_encoder_learning_rate": 2.5e-6,
            "global_film_learning_rate": 5.0e-7,
            "global_gate_learning_rate": 5.0e-7,
        }
        gate_schedule = {
            "mode": "fixed_then_learned_direct_hard_clamped_v1",
            "initial": 0.01,
            "maximum": 0.1,
            "freeze_epochs": 10,
        }

        def build() -> tuple[torch.nn.Module, torch.optim.Adam]:
            torch.manual_seed(81)
            adapter = convert_film_unet_to_text_signal_adapter(
                source,
                startup_mode=SMALL_NORMAL_STARTUP_MODE,
                spatial_rank=0,
                baseline_sha256="generator-sha",
                global_residual_gate_initial=0.01,
                global_residual_gate_max=0.1,
            )
            groups = adapter.trainable_parameter_groups()
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": list(groups["text_encoder"]),
                        "lr": 2.5e-6,
                        "group_name": "text_encoder",
                    },
                    {
                        "params": list(groups["global_film"]),
                        "lr": 5.0e-7,
                        "group_name": "global_film",
                    },
                    {
                        "params": list(groups["global_gate"]),
                        "lr": 5.0e-7,
                        "group_name": "global_gate",
                    },
                ],
                betas=(0.5, 0.9),
            )
            return adapter, optimizer

        first, first_optimizer = build()
        self.assertTrue(
            _set_global_gate_optimizer_phase(
                first_optimizer,
                enabled=True,
                epoch=10,
                freeze_epochs=10,
                target_learning_rate=5.0e-7,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gated.pt"
            _save_adapter_state(
                path,
                first,
                first_optimizer,
                epoch=10,
                learning_rate=2.5e-6,
                baseline_generator_sha="generator-sha",
                baseline_critic_sha="critic-sha",
                optimizer_contract=optimizer_contract,
                gate_schedule=gate_schedule,
            )
            restored, restored_optimizer = build()
            with self.assertRaisesRegex(ValueError, "requires optimizer"):
                load_probe_adapter_state(
                    path,
                    restored,
                    restored_optimizer,
                    expected_generator_sha256="generator-sha",
                    expected_critic_sha256="critic-sha",
                )
            epoch = load_probe_adapter_state(
                path,
                restored,
                restored_optimizer,
                expected_generator_sha256="generator-sha",
                expected_critic_sha256="critic-sha",
                expected_optimizer_contract=optimizer_contract,
                expected_gate_schedule=gate_schedule,
            )
        self.assertEqual(epoch, 10)
        self.assertFalse(
            _set_global_gate_optimizer_phase(
                restored_optimizer,
                enabled=True,
                epoch=epoch + 1,
                freeze_epochs=10,
                target_learning_rate=5.0e-7,
            )
        )
        gates = restored.trainable_parameter_groups()["global_gate"]
        before = tuple(parameter.detach().clone() for parameter in gates)
        _update(restored, restored_optimizer, _random_batch())
        restored.clamp_global_residual_gates_()
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, gates, strict=True)
            )
        )

    def test_gate_optimizer_is_frozen_for_ten_epochs_then_released(self) -> None:
        text_parameter = torch.nn.Parameter(torch.tensor(1.0))
        gate_parameter = torch.nn.Parameter(torch.tensor(0.01))
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [text_parameter],
                    "lr": 2.5e-6,
                    "group_name": "text_encoder",
                },
                {
                    "params": [gate_parameter],
                    "lr": 5.0e-7,
                    "group_name": "global_gate",
                },
            ]
        )
        for epoch in range(1, 11):
            self.assertTrue(
                _set_global_gate_optimizer_phase(
                    optimizer,
                    enabled=True,
                    epoch=epoch,
                    freeze_epochs=10,
                    target_learning_rate=5.0e-7,
                )
            )
            self.assertEqual(optimizer.param_groups[1]["lr"], 0.0)
        self.assertFalse(
            _set_global_gate_optimizer_phase(
                optimizer,
                enabled=True,
                epoch=11,
                freeze_epochs=10,
                target_learning_rate=5.0e-7,
            )
        )
        self.assertEqual(optimizer.param_groups[1]["lr"], 5.0e-7)

    def test_full_adapter_state_reproduces_the_exact_next_update(self) -> None:
        source = _source_generator()
        torch.manual_seed(17)
        first = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=0,
            baseline_sha256="generator-sha",
        )
        first_optimizer = torch.optim.Adam(
            [parameter for parameter in first.parameters() if parameter.requires_grad],
            lr=2.5e-6,
            betas=(0.5, 0.9),
        )
        torch.manual_seed(90)
        _update(first, first_optimizer, _random_batch())

        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "adapter.pt"
            _save_adapter_state(
                state_path,
                first,
                first_optimizer,
                epoch=1,
                learning_rate=2.5e-6,
                baseline_generator_sha="generator-sha",
                baseline_critic_sha="critic-sha",
            )
            expected_batch = _random_batch()
            _update(first, first_optimizer, expected_batch)
            expected = {
                name: parameter.detach().clone()
                for name, parameter in first.named_parameters()
                if parameter.requires_grad
            }

            torch.manual_seed(999)
            restored = convert_film_unet_to_text_signal_adapter(
                source,
                startup_mode=SMALL_NORMAL_STARTUP_MODE,
                spatial_rank=0,
                baseline_sha256="generator-sha",
            )
            restored_optimizer = torch.optim.Adam(
                [
                    parameter
                    for parameter in restored.parameters()
                    if parameter.requires_grad
                ],
                lr=2.5e-6,
                betas=(0.5, 0.9),
            )
            epoch = load_probe_adapter_state(
                state_path,
                restored,
                restored_optimizer,
                expected_generator_sha256="generator-sha",
                expected_critic_sha256="critic-sha",
            )
            self.assertEqual(epoch, 1)
            replay_batch = _random_batch()
            for expected_value, observed_value in zip(
                expected_batch, replay_batch, strict=True
            ):
                self.assertTrue(torch.equal(expected_value, observed_value))
            _update(restored, restored_optimizer, replay_batch)
            for name, parameter in restored.named_parameters():
                if parameter.requires_grad:
                    self.assertTrue(torch.equal(parameter, expected[name]), name)

    def test_state_rejects_baseline_drift(self) -> None:
        source = _source_generator()
        adapter = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=0,
            baseline_sha256="generator-sha",
        )
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in adapter.parameters()
                if parameter.requires_grad
            ],
            lr=2.5e-6,
        )
        _update(adapter, optimizer, _random_batch())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.pt"
            _save_adapter_state(
                path,
                adapter,
                optimizer,
                epoch=1,
                learning_rate=2.5e-6,
                baseline_generator_sha="generator-sha",
                baseline_critic_sha="critic-sha",
            )
            with self.assertRaisesRegex(ValueError, "Generator binding mismatch"):
                load_probe_adapter_state(
                    path,
                    adapter,
                    optimizer,
                    expected_generator_sha256="different",
                    expected_critic_sha256="critic-sha",
                )


class ProbeAnalysisTests(unittest.TestCase):
    def test_job_epochs_reads_orchestrator_max_epochs(self) -> None:
        self.assertEqual(_job_epochs({"max_epochs": 1}), 1)
        self.assertEqual(_job_epochs({"max_epochs": 60}), 60)
        self.assertEqual(_job_epochs({"max_epochs": 240}), 240)
        with self.assertRaisesRegex(ValueError, "support"):
            _job_epochs({"max_epochs": 2})

    def test_cross_job_comparison_is_session_paired(self) -> None:
        session_ids = [f"s{index:02d}" for index in range(34)]
        rows = []
        for index in range(110):
            rows.append(
                {
                    "pair_id": f"p{index:03d}",
                    "session_id": session_ids[index % 34],
                    "mae_matched": 0.9 + index * 1.0e-5,
                }
            )
        reference = [
            {**row, "mae_matched": float(row["mae_matched"]) / 0.9} for row in rows
        ]
        with tempfile.TemporaryDirectory() as temporary:
            focal_path = Path(temporary) / "focal.csv"
            reference_path = Path(temporary) / "reference.csv"
            pd.DataFrame(rows).to_csv(focal_path, index=False)
            pd.DataFrame(reference).to_csv(reference_path, index=False)
            result = _paired_job_comparison(focal_path, reference_path, seed=19)
        self.assertAlmostEqual(result["estimate"], np.log(0.9), places=12)
        self.assertLess(result["ci_upper"], 0.0)
        self.assertEqual(result["nonworse_sessions"], 34)


if __name__ == "__main__":
    unittest.main()
