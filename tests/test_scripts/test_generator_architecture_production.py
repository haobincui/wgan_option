"""Production contracts for the versioned RQ3 Generator architectures."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from wgan_option.config import Config
from wgan_option.models.common import (
    BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    critic_conditioning_fingerprint,
    generator_conditioning_fingerprint,
    generator_current_input_fingerprint,
    generator_noise_fingerprint,
    residual_output_fingerprint,
)
from wgan_option.models.discriminator import Discriminator
from wgan_option.models.gan_model import WGAN_GP
from wgan_option.models.generator import (
    GENERATOR_PARAMETER_ROLES,
    Generator,
    StyleModulatedConv2d,
)
from wgan_option.utils.inference_helpers import load_vol_generator


EXACT_MONEYNESS = np.linspace(0.97, 1.03, 16, dtype=np.float32)
EXACT_TTM = np.asarray(
    [1, 2, 3, 6, 7, 8, 9, 10, 14, 15, 17, 21, 26, 30, 35, 38],
    dtype=np.float32,
)
NEW_MODES = (
    CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
)


def _mode_kwargs(mode: str, *, small: bool) -> dict[str, object]:
    if mode == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "crossattn_heads": 4,
            "crossattn_text_tokens": 4,
            "crossattn_dim": 16 if small else 64,
        }
    if mode == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
        return {
            "transformer_model_dim": 16 if small else 128,
            "transformer_layers": 1 if small else 3,
            "transformer_heads": 4,
            "transformer_ffn_dim": 32 if small else 384,
            "transformer_dropout": 0.0,
        }
    return {"style_dim": 16 if small else 256, "style_demodulate": True}


def _generator(mode: str, *, small: bool = True) -> Generator:
    return Generator(
        channels=1,
        embedding_dim=12 if small else 1024,
        noise_dim=32,
        surface_height=16,
        surface_width=16,
        base_channels=(4 if small else (30 if mode == NEW_MODES[2] else 32)),
        res_blocks=0,
        text_hidden_dim=8 if small else 256,
        text_out_dim=6 if small else 128,
        hidden_dim=128 if small else 1024,
        residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
        generator_noise_mode="gaussian",
        generator_current_input_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        generator_conditioning_mode=mode,
        strike_grid=EXACT_MONEYNESS,
        maturity_grid_days=EXACT_TTM,
        **_mode_kwargs(mode, small=small),
    )


def _state_hashes(model: Generator) -> tuple[str, str]:
    values = hashlib.sha256()
    keys = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        header = f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode()
        values.update(header)
        values.update(tensor.detach().cpu().numpy().tobytes())
        keys.update(header)
    return values.hexdigest(), keys.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


class ProductionGeneratorArchitectureTest(unittest.TestCase):
    def test_identity_four_channel_input_and_architecture_shapes(self) -> None:
        current = torch.rand(2, 1, 16, 16) + 0.1
        mask = torch.ones_like(current)
        mask[:, :, 0, 0] = 0.0
        text = torch.randn(2, 12)
        noise = torch.randn(2, 32)
        for mode in NEW_MODES:
            with self.subTest(mode=mode):
                model = _generator(mode).eval()
                captured: list[torch.Tensor] = []
                first_module = (
                    model.surface_token_projection
                    if mode
                    == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE
                    else model.surface_encoder[0]
                )
                handle = first_module.register_forward_pre_hook(
                    lambda _module, inputs: captured.append(inputs[0].detach().clone())
                )
                result = model(
                    current,
                    text,
                    noise=noise,
                    current_support_mask=mask,
                )
                handle.remove()
                torch.testing.assert_close(result, current, rtol=0.0, atol=0.0)
                self.assertFalse(hasattr(model, "fusion"))
                if mode == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
                    self.assertEqual(tuple(captured[0].shape), (2, 256, 4))
                    torch.testing.assert_close(
                        captured[0][:, 0, :2],
                        torch.stack(
                            [
                                current[:, 0, 0, 0] * mask[:, 0, 0, 0],
                                mask[:, 0, 0, 0],
                            ],
                            dim=1,
                        ),
                    )
                else:
                    self.assertEqual(tuple(captured[0].shape), (2, 4, 16, 16))
                    torch.testing.assert_close(captured[0][:, :1], current * mask)
                    torch.testing.assert_close(captured[0][:, 1:2], mask)

    def test_crossattn_transformer_and_style_structure(self) -> None:
        cross = _generator(CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE)
        calls: list[tuple[int, ...]] = []
        handles = [
            layer.register_forward_pre_hook(
                lambda _module, inputs: calls.append(tuple(inputs[0].shape))
            )
            for layer in cross.cross_attention_layers
        ]
        cross(
            torch.ones(2, 1, 16, 16),
            torch.randn(2, 12),
            noise=torch.randn(2, 32),
            current_support_mask=torch.ones(2, 1, 16, 16),
        )
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [(2, 16, 4, 4), (2, 8, 8, 8), (2, 4, 16, 16)])
        self.assertTrue(
            all(
                float(layer.residual_gate) == 0.0
                for layer in cross.cross_attention_layers
            )
        )

        transformer = _generator(
            TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        )
        tokens: list[torch.Tensor] = []
        handle = transformer.transformer_encoder.register_forward_pre_hook(
            lambda _module, inputs: tokens.append(inputs[0].detach())
        )
        transformer(
            torch.ones(2, 1, 16, 16),
            torch.randn(2, 12),
            noise=torch.randn(2, 32),
            current_support_mask=torch.ones(2, 1, 16, 16),
        )
        handle.remove()
        self.assertEqual(tuple(tokens[0].shape), (2, 258, 16))

        style = _generator(STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE)
        convolutions = [
            module
            for module in style.modules()
            if isinstance(module, StyleModulatedConv2d)
        ]
        self.assertEqual(len(convolutions), 6)
        self.assertTrue(all(module.demodulate for module in convolutions))
        self.assertTrue(
            all(
                int(torch.count_nonzero(module.style_projection.weight)) == 0
                and int(torch.count_nonzero(module.style_projection.bias)) == 0
                for module in convolutions
            )
        )

    def test_three_real_updates_start_all_parameter_roles(self) -> None:
        current = torch.rand(2, 1, 16, 16) + 0.1
        target = current + 0.05 * torch.randn_like(current)
        mask = torch.ones_like(current)
        text = torch.randn(2, 12)
        noise = torch.randn(2, 32)
        for mode in NEW_MODES:
            with self.subTest(mode=mode):
                torch.manual_seed(17)
                model = _generator(mode).train()
                original = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(
                        current,
                        text,
                        noise=noise,
                        current_support_mask=mask,
                    )
                    (prediction - target).square().mean().backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    current,
                    text,
                    noise=noise,
                    current_support_mask=mask,
                )
                (prediction - target).square().mean().backward()
                roles = model.parameter_roles()
                self.assertEqual(
                    set(roles), {name for name, _ in model.named_parameters()}
                )
                self.assertEqual(set(roles.values()), set(GENERATOR_PARAMETER_ROLES))
                for role in GENERATOR_PARAMETER_ROLES:
                    role_parameters = [
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if roles[name] == role
                    ]
                    self.assertTrue(role_parameters)
                    self.assertTrue(
                        any(
                            parameter.grad is not None
                            and bool(torch.isfinite(parameter.grad).all())
                            and int(torch.count_nonzero(parameter.grad)) > 0
                            for _, parameter in role_parameters
                        )
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(parameter.detach(), original[name])
                            for name, parameter in role_parameters
                        )
                    )

    def test_exact_matched_scaled_and_critic_parameter_counts(self) -> None:
        self.assertEqual(
            {
                mode: sum(p.numel() for p in _generator(mode, small=False).parameters())
                for mode in NEW_MODES
            },
            {
                CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: 824_644,
                TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE: 834_561,
                STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: 829_635,
            },
        )
        scaled = {
            CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: dict(
                base_channels=42, crossattn_dim=192
            ),
            TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE: dict(
                transformer_model_dim=128,
                transformer_layers=6,
                transformer_heads=4,
                transformer_ffn_dim=640,
                transformer_dropout=0.0,
            ),
            STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: dict(
                base_channels=56, style_dim=128
            ),
        }
        counts = {}
        for mode, overrides in scaled.items():
            kwargs = _mode_kwargs(mode, small=False)
            kwargs.update(overrides)
            base_channels = int(kwargs.pop("base_channels", 32))
            model = Generator(
                channels=1,
                embedding_dim=1024,
                noise_dim=32,
                surface_height=16,
                surface_width=16,
                base_channels=base_channels,
                text_hidden_dim=256,
                text_out_dim=128,
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
                generator_current_input_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
                generator_conditioning_mode=mode,
                strike_grid=EXACT_MONEYNESS,
                maturity_grid_days=EXACT_TTM,
                **kwargs,
            )
            counts[mode] = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(
            counts,
            {
                CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: 1_655_352,
                TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE: 1_725_441,
                STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: 1_657_101,
            },
        )
        film_counts = {}
        for capacity, base_channels in (("matched", 32), ("scaled", 54)):
            film = Generator(
                channels=1,
                embedding_dim=1024,
                noise_dim=32,
                surface_height=16,
                surface_width=16,
                base_channels=base_channels,
                res_blocks=0,
                text_hidden_dim=256,
                text_out_dim=128,
                residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
                generator_current_input_mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
                generator_conditioning_mode=FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                strike_grid=EXACT_MONEYNESS,
                maturity_grid_days=EXACT_TTM,
            )
            film_counts[capacity] = sum(
                parameter.numel() for parameter in film.parameters()
            )
        self.assertEqual(film_counts, {"matched": 827_745, "scaled": 1_631_823})
        critic = Discriminator(
            channels=1,
            embedding_dim=1024,
            surface_height=16,
            surface_width=16,
            base_channels=32,
            res_blocks=0,
            text_hidden_dim=128,
            hidden_dim=786,
            critic_conditioning_mode="lp_disabled_same_shape_v1",
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in critic.parameters()), 729_157
        )

    def test_new_modes_fail_closed_on_exact_contract(self) -> None:
        base = {
            "channels": 1,
            "embedding_dim": 12,
            "noise_dim": 32,
            "surface_height": 16,
            "surface_width": 16,
            "generator_conditioning_mode": NEW_MODES[0],
            "generator_current_input_mode": CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            "residual_output_mode": IDENTITY_RESIDUAL_OUTPUT_MODE,
        }
        for overrides, pattern in (
            ({"channels": 2}, "channels=1"),
            ({"surface_width": 8}, "exact 16x16"),
            ({"noise_dim": 16}, "Gaussian32"),
            ({"generator_noise_mode": "zero"}, "Gaussian32"),
            ({"residual_output_mode": "legacy_softplus"}, "identity residual"),
            (
                {"generator_current_input_mode": "full_current"},
                "current_support_masked",
            ),
            ({"res_blocks": 1}, "gen_res_blocks=0"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, pattern):
                    Generator(**{**base, **overrides})

        with self.assertRaisesRegex(ValueError, "explicit current_support_mask"):
            _generator(NEW_MODES[0])(
                torch.ones(1, 1, 16, 16),
                torch.ones(1, 12),
                noise=torch.ones(1, 32),
            )

        config_base = {
            "generator_conditioning_mode": NEW_MODES[0],
            "generator_current_input_mode": CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            "support_mask_mode": "raw_joint",
            "residual_output_mode": IDENTITY_RESIDUAL_OUTPUT_MODE,
        }
        for overrides, pattern in (
            ({"channels": 2}, "channels=1"),
            ({"strike_bins": 8}, "exact 16x16"),
            ({"noise_dim": 16}, "Gaussian32"),
            ({"gen_res_blocks": 1}, "gen_res_blocks=0"),
        ):
            with self.subTest(config_overrides=overrides):
                with self.assertRaisesRegex(ValueError, pattern):
                    Config(**{**config_base, **overrides})

    def _config(self, root: Path, mode: str, **overrides: object) -> Config:
        values: dict[str, object] = {
            "cuda": False,
            "generator_conditioning_mode": mode,
            "generator_current_input_mode": CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
            "support_mask_mode": "raw_joint",
            "residual_output_mode": IDENTITY_RESIDUAL_OUTPUT_MODE,
            "generator_optimizer_profile": "conditioning_split_lr_v2",
            "learning_rate": 5e-7,
            "generator_learning_rate": 5e-7,
            "generator_text_learning_rate": 2.5e-6,
            "generator_conditioning_learning_rate": 2.5e-5,
            "generator_text_min_learning_rate": 2.5e-7,
            "generator_conditioning_min_learning_rate": 2.5e-6,
            "reduce_lr_min_lr": 5e-8,
            "gen_base_channels": 4,
            "gen_text_hidden_dim": 8,
            "gen_text_out_dim": 6,
            "gen_crossattn_dim": 16,
            "gen_transformer_model_dim": 16,
            "gen_transformer_layers": 1,
            "gen_transformer_heads": 4,
            "gen_transformer_ffn_dim": 32,
            "gen_transformer_dropout": 0.0,
            "gen_style_dim": 16,
            "disc_base_channels": 2,
            "disc_text_hidden_dim": 2,
            "disc_hidden_dim": 4,
            "embedding_dim": 12,
            "models_path": str(root / "models"),
            "outputs_path": str(root / "models"),
            "samples_path": str(root / "samples"),
            "metrics_path": str(root / "metrics"),
        }
        values.update(overrides)
        return Config(**values)

    def test_v2_optimizer_roles_learning_rates_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, NEW_MODES[0])
            model = WGAN_GP(config, EXACT_MONEYNESS, EXACT_TTM, 12)
            self.assertEqual(
                [group["group_name"] for group in model.g_optimizer.param_groups],
                list(GENERATOR_PARAMETER_ROLES),
            )
            self.assertEqual(
                [group["lr"] for group in model.g_optimizer.param_groups],
                [5e-7, 2.5e-6, 2.5e-5],
            )
            self.assertEqual(
                model._generator_scheduler_min_learning_rates(),
                {
                    "backbone": 5e-8,
                    "text_encoder": 2.5e-7,
                    "conditioning_module": 2.5e-6,
                },
            )

            legacy_fallback = self._config(
                root,
                NEW_MODES[0],
                generator_conditioning_learning_rate=0.0,
                generator_conditioning_min_learning_rate=0.0,
                generator_film_learning_rate=1e-5,
                generator_film_min_learning_rate=1e-6,
            )
            fallback_model = WGAN_GP(
                legacy_fallback,
                EXACT_MONEYNESS,
                EXACT_TTM,
                12,
            )
            self.assertEqual(
                fallback_model._generator_optimizer_group_learning_rates()[
                    "conditioning_module"
                ],
                1e-5,
            )
            self.assertEqual(
                fallback_model._generator_scheduler_min_learning_rates()[
                    "conditioning_module"
                ],
                1e-6,
            )

            backbone_fallback = self._config(
                root,
                NEW_MODES[0],
                generator_conditioning_learning_rate=0.0,
                generator_conditioning_min_learning_rate=0.0,
            )
            backbone_fallback_model = WGAN_GP(
                backbone_fallback,
                EXACT_MONEYNESS,
                EXACT_TTM,
                12,
            )
            self.assertEqual(
                backbone_fallback_model._generator_optimizer_group_learning_rates()[
                    "conditioning_module"
                ],
                5e-7,
            )
            self.assertEqual(
                backbone_fallback_model._generator_scheduler_min_learning_rates()[
                    "conditioning_module"
                ],
                5e-8,
            )

            film = self._config(
                root,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            )
            film_model = WGAN_GP(film, EXACT_MONEYNESS, EXACT_TTM, 12)
            self.assertEqual(
                [group["group_name"] for group in film_model.g_optimizer.param_groups],
                list(GENERATOR_PARAMETER_ROLES),
            )

    def test_new_mode_checkpoint_inference_round_trip(self) -> None:
        sample = SimpleNamespace(
            current_surface=np.ones((1, 16, 16), dtype=np.float32),
            strike_grid=EXACT_MONEYNESS,
            maturity_grid_days=EXACT_TTM,
        )
        for mode in NEW_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = self._config(root, mode)
                model = WGAN_GP(config, EXACT_MONEYNESS, EXACT_TTM, 12).G
                checkpoint = {
                    "config": asdict(config),
                    "state_dict": model.state_dict(),
                    "embedding_dim": 12,
                    "residual_output_mode": model.residual_output_mode,
                    "residual_output_fingerprint": residual_output_fingerprint(
                        model.residual_output_mode
                    ),
                    "generator_noise_mode": model.generator_noise_mode,
                    "generator_noise_fingerprint": generator_noise_fingerprint(
                        model.generator_noise_mode, model.noise_dim
                    ),
                    "generator_current_input_mode": model.generator_current_input_mode,
                    "generator_current_input_fingerprint": (
                        generator_current_input_fingerprint(
                            model.generator_current_input_mode
                        )
                    ),
                    "generator_conditioning_mode": mode,
                    "generator_conditioning_fingerprint": (
                        generator_conditioning_fingerprint(mode)
                    ),
                    "critic_conditioning_mode": config.critic_conditioning_mode,
                    "critic_conditioning_fingerprint": critic_conditioning_fingerprint(
                        config.critic_conditioning_mode
                    ),
                }
                path = root / "checkpoint.pt"
                torch.save(checkpoint, path)
                loaded, _, embedding_dim = load_vol_generator(
                    path,
                    sample,
                    torch.device("cpu"),
                )
                self.assertEqual(embedding_dim, 12)
                self.assertEqual(loaded.generator_conditioning_mode, mode)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value,
                        loaded.state_dict()[name],
                        rtol=0.0,
                        atol=0.0,
                    )

    def test_all_old_modes_keep_golden_state_rng_and_fingerprint(self) -> None:
        golden = {
            BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE: (
                "60a06a3ac5fe7fc5a5b045283208e8d0500495d36bdf4f6cdcee8d459d988ecf",
                "ff9d8ae4d7e41f83adfa4be70ad86997daed0d25370f92e1a9b6f6179c5277a4",
                "b61674cdf033fe4cf8e961df4cf0232d3346d5d1693ca28173b2240ba07e4f17",
                "e9e7911e20c9e63ea2498b98c44cee7a0acd5cbca7bfb03843d200210e063b84",
            ),
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: (
                "fee027e437cfc9f23717b6e1141babcb81e052d732ea25f46293f69b0dde6674",
                "619eb5efba1c15e0ed84323d34b121aaf04a74c87ff5746a409e5e459a31e851",
                "6c47660ba3e5a30af6c7fc31f20f886e999ff10c9a2133b3868a5e5ab31a8b8e",
                "dbfab468ae8806b4b994e10ad4727e1455899b16eaeee5f004c0fa55e912a25a",
            ),
            FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE: (
                "05d6107f3ca5b363e6095d590a283c11710c708d7490b2ffcc9317b29a8a844d",
                "232022a875c331359cdecf695b64073f359a25fda87becfbcca3edf2da5d08b5",
                "b61674cdf033fe4cf8e961df4cf0232d3346d5d1693ca28173b2240ba07e4f17",
                "91cde7910067a6dee6922a9496951f29ddfe4e7bd49057003e5b92f18a532483",
            ),
            FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE: (
                "8525a4a76e78d1085f2bd807423f704f1e67cd3acba8cb5eb59fb78e33c94584",
                "1d8a29c1e6d3546211f86ae6d5ea7214ff95eeb3d22cb92677cd3d46ef149a1b",
                "d88bb829301857d5c5ee433a11d871ce1c7f75fc38b8dbc07afac7ebf53663ce",
                "db0b497fe82ab70c7ec1de3c28e08a45f16c2aad457fc801690b0b65bb364d07",
            ),
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE: (
                "d55d130d7aeadcbae37cda1bb0183bbca298dd1a64c4ac11e7c67b0fea47fdf1",
                "56f425d836c25922c7834216f469b05b87267196a5f2167834075c3f152c5c76",
                "6c47660ba3e5a30af6c7fc31f20f886e999ff10c9a2133b3868a5e5ab31a8b8e",
                "0b5722ab56a728cd04a5b2ed9d0cc487ec553406f3da23e08be2adb4463685a6",
            ),
            FILM_UNET_GENERATOR_CONDITIONING_MODE: (
                "c54daf241ccf9f053c191c85a8d065b63c3cb06894ad9a4b2b7cd263392813dc",
                "194263d482d2d021a93a8c3f5828b81f46cb540b6e1d4da50056ed3dc2f171b5",
                "10a85f7b43ad42c8b5d994b9e0a1276b0fec969c15a8d56a2a754bee2481f0ba",
                "17cda9dc44b19e1440e6c47173b42fd482e51212dbb5b1b406c0aba210322663",
            ),
        }
        unet_modes = {
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        }
        coordinate_modes = {
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        }
        for mode, expected in golden.items():
            with self.subTest(mode=mode):
                torch.manual_seed(908172)
                kwargs: dict[str, object] = {}
                if mode in unet_modes:
                    kwargs["generator_current_input_mode"] = (
                        CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
                    )
                if mode in coordinate_modes:
                    kwargs.update(
                        strike_grid=EXACT_MONEYNESS,
                        maturity_grid_days=EXACT_TTM,
                    )
                model = Generator(
                    channels=1,
                    embedding_dim=17,
                    noise_dim=5,
                    surface_height=16,
                    surface_width=16,
                    base_channels=3,
                    res_blocks=0,
                    text_hidden_dim=11,
                    text_out_dim=7,
                    hidden_dim=19,
                    residual_output_mode=IDENTITY_RESIDUAL_OUTPUT_MODE,
                    generator_conditioning_mode=mode,
                    **kwargs,
                )
                state_hash, key_hash = _state_hashes(model)
                observed = (
                    state_hash,
                    key_hash,
                    _tensor_hash(torch.random.get_rng_state()),
                    model.generator_conditioning_fingerprint,
                )
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
