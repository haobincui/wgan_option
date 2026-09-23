"""Focused contracts for the frozen FiLM text-signal adapter."""

from __future__ import annotations

import copy
import unittest

import torch

from wgan_option.models.generator import Generator
from wgan_option.models.text_signal_adapter import (
    BOUNDED_GLOBAL_GATE_MODE,
    SMALL_NORMAL_STARTUP_MODE,
    TEXT_SIGNAL_ADAPTER_STATE_SCHEMA,
    ZERO_STARTUP_MODE,
    convert_film_unet_to_text_signal_adapter,
)


BASE_GENERATOR_PARAMETERS = 827_745
SPATIAL_PARAMETERS = 231_168
SPATIAL_MODEL_PARAMETERS = 1_058_913
TEXT328_MODEL_PARAMETERS = 1_058_345
GATED_GLOBAL_MODEL_PARAMETERS = 827_751


def _generator() -> Generator:
    return Generator(
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
        strike_grid=torch.linspace(0.8, 1.2, steps=16),
        maturity_grid_days=torch.tensor(
            [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38]
        ),
    )


def _nontrivial_generator() -> Generator:
    torch.manual_seed(923)
    generator = _generator().eval()
    with torch.no_grad():
        generator.residual_head.weight.normal_(mean=0.0, std=0.02)
        generator.residual_head.bias.fill_(0.01)
        for layer in (
            *generator.encoder_film_layers,
            generator.bottleneck_film_layer,
            *generator.decoder_film_layers,
        ):
            layer.projection.weight.normal_(mean=0.0, std=0.01)
            layer.projection.bias.normal_(mean=0.0, std=0.01)
    return generator


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(812)
    current = 0.15 + 0.2 * torch.rand(batch_size, 1, 16, 16)
    text = torch.randn(batch_size, 1024)
    noise = torch.randn(batch_size, 32)
    mask = torch.ones_like(current)
    mask[:, :, ::3, ::4] = 0.0
    return current, text, noise, mask


class TestFrozenFiLMTextAdapter(unittest.TestCase):
    def test_bounded_global_gate_preserves_zero_text_and_has_six_parameters(
        self,
    ) -> None:
        source = _nontrivial_generator()
        current, text, noise, mask = _inputs()
        zero = torch.zeros_like(text)
        with torch.no_grad():
            expected_zero = source(current, zero, noise, mask)

        torch.manual_seed(613)
        gated = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=0,
            global_residual_gate_initial=0.01,
            global_residual_gate_max=0.1,
        ).eval()
        with torch.no_grad():
            observed_zero = gated(current, zero, noise, mask)

        self.assertLessEqual(float((observed_zero - expected_zero).abs().max()), 1e-7)
        self.assertEqual(gated.parameter_count, GATED_GLOBAL_MODEL_PARAMETERS)
        names = gated.trainable_parameter_names()
        self.assertEqual(set(names), {"text_encoder", "global_film", "global_gate"})
        self.assertEqual(len(names["global_gate"]), 6)
        self.assertTrue(
            all(name.endswith(".residual_gate_value") for name in names["global_gate"])
        )
        gates = gated.global_residual_gate_values()
        self.assertEqual(len(gates), 6)
        for value in gates:
            self.assertAlmostEqual(float(value), 0.01, places=8)
        diagnostics = gated.activation_diagnostics(text)
        for site in diagnostics["global"]:
            self.assertTrue(
                torch.allclose(
                    site["effective_gamma"],
                    site["residual_gate"] * site["gamma"],
                    rtol=0.0,
                    atol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    site["effective_beta"],
                    site["residual_gate"] * site["beta"],
                    rtol=0.0,
                    atol=0.0,
                )
            )
        self.assertEqual(
            gated.adapter_contract()["global_residual_gate"]["mode"],
            BOUNDED_GLOBAL_GATE_MODE,
        )

        gate_parameters = gated.trainable_parameter_groups()["global_gate"]
        with torch.no_grad():
            gate_parameters[0].fill_(-1.0)
            gate_parameters[1].fill_(1.0)
        gated.clamp_global_residual_gates_()
        observed = [float(value) for value in gated.global_residual_gate_values()]
        self.assertEqual(observed[0], 0.0)
        self.assertAlmostEqual(observed[1], 0.1, places=7)
        self.assertTrue(all(-1.0e-8 <= value <= 0.10000001 for value in observed))

        with self.assertRaisesRegex(ValueError, "requires spatial_rank=0"):
            convert_film_unet_to_text_signal_adapter(
                source,
                startup_mode=SMALL_NORMAL_STARTUP_MODE,
                spatial_rank=2,
                global_residual_gate_initial=0.01,
                global_residual_gate_max=0.1,
            )

    def test_conversion_preserves_source_and_zero_text_forward(self) -> None:
        source = _nontrivial_generator()
        current, text, noise, mask = _inputs()
        zero_text = torch.zeros_like(text)
        source_state = {
            key: value.detach().clone() for key, value in source.state_dict().items()
        }
        with torch.no_grad():
            expected_zero = source(current, zero_text, noise, mask)
            expected_text = source(current, text, noise, mask)

        torch.manual_seed(41)
        probe = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=2,
            baseline_sha256="checkpoint-file-sha",
        ).eval()
        with torch.no_grad():
            observed_zero = probe(current, zero_text, noise, mask)
            source_zero_after = source(current, zero_text, noise, mask)
            source_text_after = source(current, text, noise, mask)

        self.assertLessEqual(float((observed_zero - expected_zero).abs().max()), 1e-7)
        self.assertTrue(torch.equal(source_zero_after, expected_zero))
        self.assertTrue(torch.equal(source_text_after, expected_text))
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, source_state[key]), key)

        for module_name in (
            "surface_encoder",
            "bottleneck_conv",
            "decoder_convs",
            "residual_head",
        ):
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in getattr(probe.backbone, module_name).parameters()
                ),
                module_name,
            )
        self.assertEqual(probe.base_generator_sha256, "checkpoint-file-sha")

    def test_centered_encoder_is_deterministic_and_zero_startup_is_null(self) -> None:
        source = _nontrivial_generator()
        current, text, noise, mask = _inputs()
        torch.manual_seed(7)
        probe = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=ZERO_STARTUP_MODE,
            spatial_rank=2,
        ).train()

        first_encoding = probe.encode_text(text)
        second_encoding = probe.encode_text(text)
        zero_encoding = probe.encode_text(torch.zeros_like(text))
        self.assertTrue(torch.equal(first_encoding, second_encoding))
        self.assertEqual(int(torch.count_nonzero(zero_encoding)), 0)
        self.assertFalse(
            any(isinstance(module, torch.nn.Dropout) for module in probe.modules())
        )

        with torch.no_grad():
            matched_output = probe(current, text, noise, mask)
            zero_output = probe(current, torch.zeros_like(text), noise, mask)
        self.assertTrue(torch.equal(matched_output, zero_output))

    def test_parameter_counts_groups_and_spatial_diagnostics(self) -> None:
        source = _nontrivial_generator()
        self.assertEqual(
            sum(parameter.numel() for parameter in source.parameters()),
            BASE_GENERATOR_PARAMETERS,
        )

        torch.manual_seed(55)
        spatial_probe = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=2,
        )
        self.assertEqual(spatial_probe.spatial_parameter_count, SPATIAL_PARAMETERS)
        self.assertEqual(spatial_probe.parameter_count, SPATIAL_MODEL_PARAMETERS)
        self.assertEqual(
            spatial_probe.parameter_count - BASE_GENERATOR_PARAMETERS,
            SPATIAL_PARAMETERS,
        )

        names = spatial_probe.trainable_parameter_names()
        groups = spatial_probe.trainable_parameter_groups()
        self.assertEqual(set(names), {"text_encoder", "global_film", "spatial_film"})
        self.assertEqual(set(groups), set(names))
        self.assertEqual(
            {name for values in names.values() for name in values},
            {
                name
                for name, parameter in spatial_probe.named_parameters()
                if parameter.requires_grad
            },
        )

        _, text, _, _ = _inputs()
        diagnostics = spatial_probe.activation_diagnostics(text)
        expected_shapes = (
            (2, 32, 16, 16),
            (2, 64, 8, 8),
            (2, 128, 4, 4),
            (2, 128, 4, 4),
            (2, 64, 8, 8),
            (2, 32, 16, 16),
        )
        self.assertEqual(len(diagnostics["global"]), 6)
        self.assertEqual(
            tuple(tuple(row["gamma"].shape) for row in diagnostics["spatial"]),
            expected_shapes,
        )
        self.assertTrue(
            all(float(row["gamma"].std()) > 0.0 for row in diagnostics["spatial"])
        )

        torch.manual_seed(55)
        capacity_probe = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=0,
            adapter_text_out_dim=328,
        )
        self.assertEqual(capacity_probe.parameter_count, TEXT328_MODEL_PARAMETERS)
        self.assertEqual(capacity_probe.spatial_parameter_count, 0)

    def test_small_normal_startup_reaches_text_and_film_on_first_update(self) -> None:
        source = _nontrivial_generator()
        current, text, noise, mask = _inputs()
        torch.manual_seed(101)
        probe = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=2,
        ).train()
        prediction = probe(current, text, noise, mask)
        target = current + 0.015 * torch.randn_like(current)
        torch.nn.functional.mse_loss(prediction, target).backward()

        grouped_names = probe.trainable_parameter_names()
        named_parameters = dict(probe.named_parameters())
        for group in ("text_encoder", "global_film", "spatial_film"):
            gradients = [
                named_parameters[name].grad
                for name in grouped_names[group]
                if name.endswith("weight")
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients), group)
            self.assertTrue(
                any(
                    gradient is not None
                    and bool(torch.isfinite(gradient).all())
                    and float(gradient.abs().sum()) > 0.0
                    for gradient in gradients
                ),
                group,
            )

        for name, parameter in probe.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)

    def test_adapter_only_state_round_trip_is_fail_closed(self) -> None:
        source = _nontrivial_generator()
        current, text, noise, mask = _inputs()
        torch.manual_seed(31)
        first = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=2,
            baseline_sha256="frozen-checkpoint",
        ).eval()
        payload = first.extract_adapter_state()
        self.assertEqual(payload["schema_version"], TEXT_SIGNAL_ADAPTER_STATE_SCHEMA)
        self.assertFalse(any("surface_encoder" in key for key in payload["state_dict"]))

        torch.manual_seed(99)
        restored = convert_film_unet_to_text_signal_adapter(
            source,
            startup_mode=SMALL_NORMAL_STARTUP_MODE,
            spatial_rank=2,
            baseline_sha256="frozen-checkpoint",
        ).eval()
        with torch.no_grad():
            before = restored(current, text, noise, mask)
            expected = first(current, text, noise, mask)
        self.assertFalse(torch.equal(before, expected))
        restored.load_adapter_state(payload)
        with torch.no_grad():
            observed = restored(current, text, noise, mask)
        self.assertTrue(torch.equal(observed, expected))

        wrong_baseline = copy.deepcopy(payload)
        wrong_baseline["contract"]["base_generator_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            restored.load_adapter_state(wrong_baseline)

        missing_key = copy.deepcopy(payload)
        missing_key["state_dict"].pop(next(iter(missing_key["state_dict"])))
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            restored.load_adapter_state(missing_key)


if __name__ == "__main__":
    unittest.main()
