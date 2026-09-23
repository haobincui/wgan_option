"""Frozen-backbone text adapters for the FiLM mask-and-coordinate U-Net.

The probe model in this module is deliberately separate from :mod:`generator`.
It converts an already-loaded ``film_unet_mask_coords_v1`` Generator into a
controlled text-only training surface without changing the legacy Generator
or its checkpoint contract.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgan_option.models.common import (
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    ZERO_GENERATOR_NOISE_MODE,
    apply_residual_surface_output,
    generator_noise_tensor,
)
from wgan_option.models.generator import Generator


TEXT_SIGNAL_ADAPTER_STATE_SCHEMA = "text_adapter_probe_state_v1"
ZERO_STARTUP_MODE = "zero"
SMALL_NORMAL_STARTUP_MODE = "small_normal"
_STARTUP_MODES = {ZERO_STARTUP_MODE, SMALL_NORMAL_STARTUP_MODE}
_SMALL_NORMAL_STD = 1e-3
BOUNDED_GLOBAL_GATE_MODE = "direct_hard_clamped_per_site_v1"


def _tensor_sha256(values: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    return _tensor_sha256(list(module.state_dict().items()))


def _initialize_projection(projection: nn.Linear, startup_mode: str) -> None:
    if startup_mode == ZERO_STARTUP_MODE:
        nn.init.zeros_(projection.weight)
    elif startup_mode == SMALL_NORMAL_STARTUP_MODE:
        nn.init.normal_(projection.weight, mean=0.0, std=_SMALL_NORMAL_STD)
    else:  # pragma: no cover - caller validation gives a clearer error
        raise ValueError(f"Unsupported startup_mode: {startup_mode!r}")
    nn.init.zeros_(projection.bias)


class _BaselinePreservingGlobalFiLM(nn.Module):
    """Trainable global FiLM residual over a frozen zero-text FiLM offset."""

    def __init__(
        self,
        conditioning_dim: int,
        base_gamma: torch.Tensor,
        base_beta: torch.Tensor,
        *,
        startup_mode: str,
        residual_gate_initial: float | None = None,
        residual_gate_max: float = 0.1,
    ) -> None:
        super().__init__()
        if base_gamma.ndim != 1 or base_beta.shape != base_gamma.shape:
            raise ValueError(
                "FiLM base offsets must be matching one-dimensional tensors"
            )
        channels = int(base_gamma.numel())
        self.projection = nn.Linear(int(conditioning_dim), channels * 2)
        _initialize_projection(self.projection, startup_mode)
        self.register_buffer("base_gamma", base_gamma.detach().clone())
        self.register_buffer("base_beta", base_beta.detach().clone())
        self.residual_gate_max = float(residual_gate_max)
        if residual_gate_initial is None:
            self.register_parameter("residual_gate_value", None)
        else:
            initial = float(residual_gate_initial)
            if not 0.0 < initial < self.residual_gate_max:
                raise ValueError(
                    "residual_gate_initial must be strictly between 0 and "
                    f"residual_gate_max, got {initial} and {self.residual_gate_max}"
                )
            self.residual_gate_value = nn.Parameter(
                torch.tensor(
                    initial,
                    device=base_gamma.device,
                    dtype=base_gamma.dtype,
                )
            )

    def residual_gate(self, reference: torch.Tensor) -> torch.Tensor:
        if self.residual_gate_value is None:
            return reference.new_ones(())
        return self.residual_gate_value.clamp(0.0, self.residual_gate_max)

    def clamp_residual_gate_(self) -> None:
        """Project the directly optimized gate back onto its frozen bounds."""

        if self.residual_gate_value is not None:
            with torch.no_grad():
                self.residual_gate_value.clamp_(0.0, self.residual_gate_max)

    def adapter_modulation(
        self, centered_conditioning: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The Linear bias remains in the parameter-count/state contract, while
        # subtracting it makes the adapter exactly null when conditioning is 0.
        projected = self.projection(centered_conditioning)
        projected = projected - self.projection.bias
        gamma, beta = projected.chunk(2, dim=-1)
        return gamma[:, :, None, None], beta[:, :, None, None]

    def forward(
        self, features: torch.Tensor, centered_conditioning: torch.Tensor
    ) -> torch.Tensor:
        base_gamma = self.base_gamma[None, :, None, None]
        base_beta = self.base_beta[None, :, None, None]
        gamma, beta = self.adapter_modulation(centered_conditioning)
        baseline = (1.0 + base_gamma) * features + base_beta
        gate = self.residual_gate(features)
        return baseline + gate * (gamma * features + beta)


class _LowRankSpatialFiLM(nn.Module):
    """A text-dependent spatial FiLM residual over two fixed grid bases."""

    def __init__(
        self,
        conditioning_dim: int,
        channels: int,
        basis: torch.Tensor,
        *,
        rank: int,
        startup_mode: str,
    ) -> None:
        super().__init__()
        if basis.ndim != 3 or int(basis.shape[0]) != int(rank):
            raise ValueError(
                "Spatial basis must have shape [rank, height, width], got "
                f"{tuple(basis.shape)}"
            )
        self.channels = int(channels)
        self.rank = int(rank)
        self.projection = nn.Linear(
            int(conditioning_dim), 2 * self.channels * self.rank
        )
        _initialize_projection(self.projection, startup_mode)
        self.register_buffer("basis", basis.detach().clone())

    def modulation(
        self, centered_conditioning: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.projection(centered_conditioning)
        projected = projected - self.projection.bias
        coefficients = projected.view(
            int(projected.shape[0]), 2, self.channels, self.rank
        )
        maps = torch.einsum("bqcr,rhw->bqchw", coefficients, self.basis)
        return maps[:, 0], maps[:, 1]

    def forward(
        self, features: torch.Tensor, centered_conditioning: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.modulation(centered_conditioning)
        return gamma * features + beta


class FrozenFiLMTextAdapterGenerator(nn.Module):
    """Text-only probe converted from a loaded FiLM U-Net Generator.

    The convolutional path and residual head are copied and frozen.  The
    source Generator's deterministic zero-text FiLM outputs are stored as
    immutable buffers.  A newly initialized text encoder and FiLM residuals
    are the only trainable path, so ``forward(..., text_embedding=zeros)``
    reproduces the source model's eval-mode zero-text output.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        startup_mode: str = ZERO_STARTUP_MODE,
        spatial_rank: int = 0,
        adapter_text_out_dim: int = 128,
        baseline_sha256: str | None = None,
        global_residual_gate_initial: float | None = None,
        global_residual_gate_max: float = 0.1,
    ) -> None:
        super().__init__()
        if startup_mode not in _STARTUP_MODES:
            raise ValueError(
                f"startup_mode must be one of {sorted(_STARTUP_MODES)}, "
                f"got {startup_mode!r}"
            )
        if int(spatial_rank) not in {0, 2}:
            raise ValueError("spatial_rank must be 0 or 2")
        if int(spatial_rank) == 2 and int(adapter_text_out_dim) != 128:
            raise ValueError("rank-2 spatial FiLM requires adapter_text_out_dim=128")
        if (
            generator.generator_conditioning_mode
            != FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        ):
            raise ValueError(
                "Text adapter requires generator_conditioning_mode="
                f"{FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE!r}"
            )
        if len(generator.surface_encoder) != 3:
            raise ValueError("Text adapter requires exactly three U-Net encoders")
        if len(generator.encoder_film_layers) != 3:
            raise ValueError("Text adapter requires exactly three encoder FiLM sites")
        if len(generator.decoder_convs) != 2 or len(generator.decoder_film_layers) != 2:
            raise ValueError("Text adapter requires exactly two U-Net decoders")

        source_parameter = next(generator.parameters())
        device = source_parameter.device
        dtype = source_parameter.dtype
        source_parameter_count = sum(
            int(parameter.numel()) for parameter in generator.parameters()
        )
        source_state_sha256 = _module_state_sha256(generator)
        self.base_generator_sha256 = baseline_sha256 or source_state_sha256
        self.base_generator_state_sha256 = source_state_sha256
        self.base_generator_parameter_count = source_parameter_count
        self.startup_mode = startup_mode
        self.spatial_rank = int(spatial_rank)
        self.adapter_text_out_dim = int(adapter_text_out_dim)
        self.global_residual_gate_initial = (
            None
            if global_residual_gate_initial is None
            else float(global_residual_gate_initial)
        )
        self.global_residual_gate_max = float(global_residual_gate_max)
        if self.global_residual_gate_initial is not None and self.spatial_rank:
            raise ValueError(
                "The bounded Global FiLM gate probe requires spatial_rank=0"
            )

        source_text_encoder = generator.text_encoder
        first_linear = source_text_encoder[0]
        second_linear = source_text_encoder[4]
        if not isinstance(first_linear, nn.Linear) or not isinstance(
            second_linear, nn.Linear
        ):
            raise ValueError("Unexpected source text encoder structure")
        embedding_dim = int(first_linear.in_features)
        text_hidden_dim = int(first_linear.out_features)
        self.embedding_dim = embedding_dim
        self.text_hidden_dim = text_hidden_dim

        base_offsets = self._capture_zero_text_offsets(generator, embedding_dim)
        self.backbone = copy.deepcopy(generator)
        # Preserve the public attributes consumed by the existing trainer and
        # inference helpers even though checkpoint persistence is intentionally
        # handled through the adapter-only state API below.
        for attribute in (
            "noise_dim",
            "surface_height",
            "surface_width",
            "base_channels",
            "res_blocks",
            "residual_output_mode",
            "residual_output_fingerprint",
            "generator_noise_mode",
            "generator_noise_fingerprint",
            "generator_current_input_mode",
            "generator_current_input_fingerprint",
            "generator_conditioning_mode",
            "generator_conditioning_fingerprint",
        ):
            setattr(self, attribute, getattr(generator, attribute))
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        self.backbone.text_encoder = self._new_text_encoder(
            embedding_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
            output_dim=self.adapter_text_out_dim,
            device=device,
            dtype=dtype,
        )
        global_layers = [
            _BaselinePreservingGlobalFiLM(
                self.adapter_text_out_dim,
                gamma.to(device=device, dtype=dtype),
                beta.to(device=device, dtype=dtype),
                startup_mode=startup_mode,
                residual_gate_initial=self.global_residual_gate_initial,
                residual_gate_max=self.global_residual_gate_max,
            ).to(device=device, dtype=dtype)
            for gamma, beta in base_offsets
        ]
        self.backbone.encoder_film_layers = nn.ModuleList(global_layers[:3])
        self.backbone.bottleneck_film_layer = global_layers[3]
        self.backbone.decoder_film_layers = nn.ModuleList(global_layers[4:])

        self.spatial_film_layers = nn.ModuleList()
        if self.spatial_rank:
            full_basis = torch.cat(
                [
                    self.backbone.moneyness_coordinate,
                    self.backbone.log_ttm_coordinate,
                ],
                dim=1,
            )[0].to(device=device, dtype=dtype)
            channels_and_sizes = (
                (int(self.backbone.base_channels), (16, 16)),
                (int(self.backbone.base_channels) * 2, (8, 8)),
                (int(self.backbone.base_channels) * 4, (4, 4)),
                (int(self.backbone.base_channels) * 4, (4, 4)),
                (int(self.backbone.base_channels) * 2, (8, 8)),
                (int(self.backbone.base_channels), (16, 16)),
            )
            expected_surface = (
                int(self.backbone.surface_height),
                int(self.backbone.surface_width),
            )
            if expected_surface != (16, 16):
                raise ValueError(
                    "rank-2 spatial FiLM currently requires a 16x16 surface"
                )
            for channels, size in channels_and_sizes:
                basis = F.interpolate(
                    full_basis[None],
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
                self.spatial_film_layers.append(
                    _LowRankSpatialFiLM(
                        self.adapter_text_out_dim,
                        channels,
                        basis,
                        rank=self.spatial_rank,
                        startup_mode=startup_mode,
                    ).to(device=device, dtype=dtype)
                )

        # The converted module follows the source train/eval mode, but its text
        # encoder contains Identity instead of Dropout and is deterministic in
        # either mode.
        self.train(generator.training)
        self._assert_freezing_contract()

    @staticmethod
    def _new_text_encoder(
        *,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=False),
        ).to(device=device, dtype=dtype)

    @staticmethod
    def _ordered_source_film_layers(generator: Generator) -> tuple[nn.Module, ...]:
        return (
            *generator.encoder_film_layers,
            generator.bottleneck_film_layer,
            *generator.decoder_film_layers,
        )

    @classmethod
    def _capture_zero_text_offsets(
        cls, generator: Generator, embedding_dim: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        first_parameter = next(generator.text_encoder.parameters())
        zero_text = torch.zeros(
            1,
            embedding_dim,
            device=first_parameter.device,
            dtype=first_parameter.dtype,
        )
        training_states = {
            module: module.training for module in generator.text_encoder.modules()
        }
        generator.text_encoder.eval()
        try:
            with torch.no_grad():
                encoded_zero = generator.text_encoder(zero_text)
                offsets: list[tuple[torch.Tensor, torch.Tensor]] = []
                for layer in cls._ordered_source_film_layers(generator):
                    projected = layer.projection(encoded_zero)[0]
                    gamma, beta = projected.chunk(2, dim=-1)
                    offsets.append((gamma.detach().clone(), beta.detach().clone()))
                return tuple(offsets)
        finally:
            for module, training in training_states.items():
                module.training = training

    def _ordered_global_film_layers(
        self,
    ) -> tuple[_BaselinePreservingGlobalFiLM, ...]:
        return (
            *self.backbone.encoder_film_layers,
            self.backbone.bottleneck_film_layer,
            *self.backbone.decoder_film_layers,
        )

    def _assert_freezing_contract(self) -> None:
        allowed_prefixes = (
            "backbone.text_encoder.",
            "backbone.encoder_film_layers.",
            "backbone.bottleneck_film_layer.",
            "backbone.decoder_film_layers.",
            "spatial_film_layers.",
        )
        leaked = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith(allowed_prefixes)
        ]
        if leaked:
            raise RuntimeError(f"Frozen-backbone parameter leak: {leaked}")

    def encode_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        if (
            text_embedding.ndim != 2
            or int(text_embedding.shape[1]) != self.embedding_dim
        ):
            raise ValueError(
                "Expected text_embedding shape [batch, "
                f"{self.embedding_dim}], got {tuple(text_embedding.shape)}"
            )
        encoded = self.backbone.text_encoder(text_embedding)
        encoded_zero = self.backbone.text_encoder(torch.zeros_like(text_embedding))
        return encoded - encoded_zero

    def _activate_site(
        self,
        site: int,
        features: torch.Tensor,
        centered_text: torch.Tensor,
    ) -> torch.Tensor:
        global_layer = self._ordered_global_film_layers()[site]
        modulated = global_layer(features, centered_text)
        if self.spatial_rank:
            modulated = modulated + self.spatial_film_layers[site](
                features, centered_text
            )
        return F.leaky_relu(modulated, negative_slope=0.2, inplace=False)

    def _delta(
        self,
        encoder_surface: torch.Tensor,
        centered_text: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        encoder_features: list[torch.Tensor] = []
        x = encoder_surface
        for site, conv in enumerate(self.backbone.surface_encoder):
            x = self._activate_site(site, conv(x), centered_text)
            encoder_features.append(x)

        spatial_noise = noise[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        x = self.backbone.bottleneck_conv(torch.cat([x, spatial_noise], dim=1))
        x = self._activate_site(3, x, centered_text)

        x = F.interpolate(
            x,
            size=encoder_features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.backbone.decoder_convs[0](torch.cat([x, encoder_features[1]], dim=1))
        x = self._activate_site(4, x, centered_text)

        x = F.interpolate(
            x,
            size=encoder_features[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.backbone.decoder_convs[1](torch.cat([x, encoder_features[0]], dim=1))
        x = self._activate_site(5, x, centered_text)
        return self.backbone.residual_head(x)

    def forward(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        noise: torch.Tensor | None = None,
        current_support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(current_surface.shape[0])
        if noise is None:
            noise = generator_noise_tensor(
                current_surface,
                batch_size=batch_size,
                noise_dim=int(self.backbone.noise_dim),
                mode=self.backbone.generator_noise_mode,
                preserve_gaussian_rng_progression=self.training,
            )
        expected_shape = (batch_size, int(self.backbone.noise_dim))
        if tuple(noise.shape) != expected_shape:
            raise ValueError(
                f"Expected generator noise shape {expected_shape}, got {tuple(noise.shape)}"
            )
        if self.backbone.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE:
            noise = torch.zeros_like(noise)

        encoder_surface = self.backbone._film_unet_encoder_input(
            current_surface, current_support_mask
        )
        centered_text = self.encode_text(text_embedding)
        delta = self._delta(encoder_surface, centered_text, noise)
        return apply_residual_surface_output(
            current_surface,
            delta,
            mode=self.backbone.residual_output_mode,
        )

    def activation_diagnostics(
        self, text_embedding: torch.Tensor, *, detach: bool = True
    ) -> dict[str, Any]:
        """Return centered text and the six global/spatial modulation tensors."""

        centered = self.encode_text(text_embedding)
        global_modulations = []
        spatial_modulations = []
        for site, layer in enumerate(self._ordered_global_film_layers()):
            gamma, beta = layer.adapter_modulation(centered)
            residual_gate = layer.residual_gate(centered)
            global_modulations.append(
                {
                    "site": site,
                    "gamma": gamma,
                    "beta": beta,
                    "effective_gamma": residual_gate * gamma,
                    "effective_beta": residual_gate * beta,
                    "base_gamma": layer.base_gamma,
                    "base_beta": layer.base_beta,
                    "residual_gate": residual_gate,
                }
            )
            if self.spatial_rank:
                spatial_gamma, spatial_beta = self.spatial_film_layers[site].modulation(
                    centered
                )
                spatial_modulations.append(
                    {
                        "site": site,
                        "gamma": spatial_gamma,
                        "beta": spatial_beta,
                    }
                )
        diagnostics: dict[str, Any] = {
            "centered_text": centered,
            "global": tuple(global_modulations),
            "spatial": tuple(spatial_modulations),
        }
        if not detach:
            return diagnostics

        def detach_value(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach()
            if isinstance(value, dict):
                return {key: detach_value(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return tuple(detach_value(item) for item in value)
            return value

        return detach_value(diagnostics)

    def trainable_parameter_names(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {
            "text_encoder": [],
            "global_film": [],
            "global_gate": [],
            "spatial_film": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone.text_encoder."):
                groups["text_encoder"].append(name)
            elif name.startswith("spatial_film_layers."):
                groups["spatial_film"].append(name)
            elif name.endswith(".residual_gate_value"):
                groups["global_gate"].append(name)
            else:
                groups["global_film"].append(name)
        return {key: tuple(values) for key, values in groups.items() if values}

    def trainable_parameter_groups(
        self,
    ) -> dict[str, tuple[nn.Parameter, ...]]:
        named_parameters = dict(self.named_parameters())
        return {
            group: tuple(named_parameters[name] for name in names)
            for group, names in self.trainable_parameter_names().items()
        }

    def global_residual_gate_values(self) -> tuple[torch.Tensor, ...]:
        """Return the six bounded gate values, or an empty tuple when disabled."""

        if self.global_residual_gate_initial is None:
            return ()
        reference = next(self.parameters())
        return tuple(
            layer.residual_gate(reference)
            for layer in self._ordered_global_film_layers()
        )

    def clamp_global_residual_gates_(self) -> None:
        """Project all enabled per-site gates to ``[0, maximum]`` in place."""

        for layer in self._ordered_global_film_layers():
            layer.clamp_residual_gate_()

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    @property
    def spatial_parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.spatial_film_layers.parameters()
        )

    def adapter_contract(self) -> dict[str, Any]:
        coordinate_sha256 = _tensor_sha256(
            [
                ("moneyness", self.backbone.moneyness_coordinate),
                ("log_ttm", self.backbone.log_ttm_coordinate),
            ]
        )
        contract = {
            "schema_version": TEXT_SIGNAL_ADAPTER_STATE_SCHEMA,
            "base_generator_sha256": self.base_generator_sha256,
            "base_generator_state_sha256": self.base_generator_state_sha256,
            "base_generator_parameter_count": self.base_generator_parameter_count,
            "model_parameter_count": self.parameter_count,
            "embedding_dim": self.embedding_dim,
            "text_hidden_dim": self.text_hidden_dim,
            "adapter_text_out_dim": self.adapter_text_out_dim,
            "startup_mode": self.startup_mode,
            "spatial_rank": self.spatial_rank,
            "coordinate_sha256": coordinate_sha256,
        }
        # Preserve the exact historical contract when the optional gate is
        # disabled so existing adapter-only states remain readable.
        if self.global_residual_gate_initial is not None:
            contract["global_residual_gate"] = {
                "mode": BOUNDED_GLOBAL_GATE_MODE,
                "initial": self.global_residual_gate_initial,
                "maximum": self.global_residual_gate_max,
                "sites": 6,
            }
        return contract

    @staticmethod
    def _adapter_state_prefixes() -> tuple[str, ...]:
        return (
            "backbone.text_encoder.",
            "backbone.encoder_film_layers.",
            "backbone.bottleneck_film_layer.",
            "backbone.decoder_film_layers.",
            "spatial_film_layers.",
        )

    def extract_adapter_state(self) -> dict[str, Any]:
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(self._adapter_state_prefixes())
        }
        return {
            "schema_version": TEXT_SIGNAL_ADAPTER_STATE_SCHEMA,
            "contract": self.adapter_contract(),
            "state_dict": state_dict,
        }

    def load_adapter_state(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != TEXT_SIGNAL_ADAPTER_STATE_SCHEMA:
            raise ValueError("Invalid text adapter state schema")
        if payload.get("contract") != self.adapter_contract():
            raise ValueError("Text adapter state contract mismatch")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("Text adapter state_dict must be a mapping")
        own_state = self.state_dict()
        expected_keys = {
            key for key in own_state if key.startswith(self._adapter_state_prefixes())
        }
        observed_keys = set(state)
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys)
            unexpected = sorted(observed_keys - expected_keys)
            raise ValueError(
                "Text adapter state keys mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for key in sorted(expected_keys):
                source = state[key]
                target = own_state[key]
                if not isinstance(source, torch.Tensor):
                    raise ValueError(f"Text adapter state {key!r} is not a tensor")
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise ValueError(
                        f"Text adapter state {key!r} tensor contract mismatch"
                    )
                target.copy_(source.to(device=target.device))


def convert_film_unet_to_text_signal_adapter(
    generator: Generator,
    *,
    startup_mode: str = ZERO_STARTUP_MODE,
    spatial_rank: int = 0,
    adapter_text_out_dim: int = 128,
    baseline_sha256: str | None = None,
    global_residual_gate_initial: float | None = None,
    global_residual_gate_max: float = 0.1,
) -> FrozenFiLMTextAdapterGenerator:
    """Copy and convert a loaded FiLM U-Net without mutating the source."""

    return FrozenFiLMTextAdapterGenerator(
        generator,
        startup_mode=startup_mode,
        spatial_rank=spatial_rank,
        adapter_text_out_dim=adapter_text_out_dim,
        baseline_sha256=baseline_sha256,
        global_residual_gate_initial=global_residual_gate_initial,
        global_residual_gate_max=global_residual_gate_max,
    )


__all__ = [
    "BOUNDED_GLOBAL_GATE_MODE",
    "FrozenFiLMTextAdapterGenerator",
    "SMALL_NORMAL_STARTUP_MODE",
    "TEXT_SIGNAL_ADAPTER_STATE_SCHEMA",
    "ZERO_STARTUP_MODE",
    "convert_film_unet_to_text_signal_adapter",
]
