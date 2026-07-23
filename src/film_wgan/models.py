"""FiLM-conditioned WGAN models: text embedding modulates surface CNN features via Feature-wise Linear Modulation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

VOL_FLOOR = 1e-4
VOL_CEIL = 5.0
_LOG_VOL_FLOOR = math.log(VOL_FLOOR)
_LOG_VOL_CEIL = math.log(VOL_CEIL)


def _conv2d_out_size(size: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


class FiLMLayer(nn.Module):
    """Generate channel-wise affine parameters (gamma, beta) from a conditioning vector."""

    def __init__(self, conditioning_dim: int, num_channels: int):
        super().__init__()
        self.projection = nn.Linear(conditioning_dim, num_channels * 2)
        # With `(1 + gamma) * x + beta`, identity-preserving FiLM means `gamma=0`, not `gamma=1`.
        nn.init.zeros_(self.projection.weight[:num_channels])
        nn.init.zeros_(self.projection.weight[num_channels:])
        nn.init.zeros_(self.projection.bias[:num_channels])
        nn.init.zeros_(self.projection.bias[num_channels:])

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        params = self.projection(conditioning)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class _FiLMResidualConvBlock(nn.Module):
    """Residual block with FiLM conditioning applied after each convolution."""

    def __init__(self, channels: int, conditioning_dim: int, *, conditioning_mode: str = "film"):
        super().__init__()
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.film1 = FiLMLayer(conditioning_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.film2 = FiLMLayer(conditioning_dim, channels)
        if self.conditioning_mode in {"concat", "residual_film", "projection"}:
            self.film1.requires_grad_(False)
            self.film2.requires_grad_(False)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        if self.conditioning_mode == "film":
            out = self.film1(out, conditioning)
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.2)
        out = self.conv2(out)
        if self.conditioning_mode == "film":
            out = self.film2(out, conditioning)
        return torch.nn.functional.leaky_relu(residual + out, negative_slope=0.2)


class FilmWGANGenerator(nn.Module):
    """CNN generator with FiLM conditioning: text embedding modulates surface feature extraction."""

    def __init__(
        self,
        surface_height: int,
        surface_width: int,
        embedding_dim: int,
        noise_dim: int,
        base_channels: int,
        res_blocks: int,
        text_hidden_dim: int,
        text_out_dim: int,
        fusion_hidden_dim: int,
        conditioning_mode: str = "film",
        text_dropout: float = 0.0,
        text_gate_initial_value: float = 0.0,
    ):
        super().__init__()
        self.surface_height = int(surface_height)
        self.surface_width = int(surface_width)
        self.surface_dim = int(surface_height * surface_width)
        self.embedding_dim = int(embedding_dim)
        self.noise_dim = int(noise_dim)
        self.num_res_blocks = max(0, int(res_blocks))
        self.conditioning_mode = str(conditioning_mode).strip().lower()
        if self.conditioning_mode not in {"film", "concat", "residual_film"}:
            raise ValueError("conditioning_mode must be one of ['film', 'concat', 'residual_film'].")

        text_layers: list[nn.Module] = [
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if self.conditioning_mode == "residual_film":
            text_layers.append(nn.Dropout(float(text_dropout)))
        text_layers.extend(
            [
                nn.Linear(text_hidden_dim, text_out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        if self.conditioning_mode == "residual_film":
            text_layers.append(nn.Dropout(float(text_dropout)))
        self.text_encoder = nn.Sequential(*text_layers)

        c = base_channels
        self.conv1 = nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1)
        self.film1 = FiLMLayer(text_out_dim, c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.film2 = FiLMLayer(text_out_dim, c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.film3 = FiLMLayer(text_out_dim, c * 4)
        if self.conditioning_mode in {"concat", "residual_film"}:
            self.film1.requires_grad_(False)
            self.film2.requires_grad_(False)
            self.film3.requires_grad_(False)

        self.res_blocks = nn.ModuleList(
            [
                _FiLMResidualConvBlock(c * 4, text_out_dim, conditioning_mode=self.conditioning_mode)
                for _ in range(self.num_res_blocks)
            ]
        )

        reduced_h = _conv2d_out_size(_conv2d_out_size(surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(surface_width))
        self.surface_feat_dim = int(c * 4 * reduced_h * reduced_w)

        fusion_input_dim = self.surface_feat_dim + noise_dim
        if self.conditioning_mode != "residual_film":
            fusion_input_dim += text_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(fusion_hidden_dim, self.surface_dim),
        )
        if self.conditioning_mode == "residual_film":
            self.residual_bottleneck_film = FiLMLayer(text_out_dim, c * 4)
            adapter_hidden = max(64, int(text_hidden_dim))
            self.text_adapter = nn.Sequential(
                nn.Linear(self.surface_feat_dim + text_out_dim, adapter_hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(float(text_dropout)),
                nn.Linear(adapter_hidden, self.surface_dim),
            )
            nn.init.normal_(self.text_adapter[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.text_adapter[-1].bias)
            gate_value = float(text_gate_initial_value)
            if abs(gate_value) >= 1.0:
                raise ValueError("text_gate_initial_value must be strictly between -1 and 1.")
            raw_gate = math.atanh(gate_value) if gate_value else 0.0
            self.text_gate = nn.Parameter(torch.tensor(raw_gate, dtype=torch.float32))

    @staticmethod
    def _has_text_mask(
        has_text: torch.Tensor | None,
        *,
        batch_size: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if has_text is None:
            return torch.ones(batch_size, 1, device=reference.device, dtype=reference.dtype)
        mask = has_text.to(device=reference.device, dtype=reference.dtype).reshape(batch_size, -1)
        if mask.shape[1] != 1:
            raise ValueError(f"has_text must contain one value per sample, got {tuple(mask.shape)}.")
        return mask

    def text_adapter_parameters(self):
        if self.conditioning_mode != "residual_film":
            return iter(())
        modules = (self.text_encoder, self.residual_bottleneck_film, self.text_adapter)
        parameters = [parameter for module in modules for parameter in module.parameters()]
        parameters.append(self.text_gate)
        return iter(parameters)

    def backbone_parameters(self):
        if self.conditioning_mode != "residual_film":
            return iter(self.parameters())
        adapter_ids = {id(parameter) for parameter in self.text_adapter_parameters()}
        return (parameter for parameter in self.parameters() if id(parameter) not in adapter_ids)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(bool(trainable))

    def film_regularization(self) -> torch.Tensor:
        if self.conditioning_mode != "residual_film":
            return torch.zeros((), device=next(self.parameters()).device)
        values = [torch.tanh(self.text_gate).pow(2)]
        values.extend(parameter.pow(2).mean() for parameter in self.residual_bottleneck_film.parameters())
        values.extend(parameter.pow(2).mean() for parameter in self.text_adapter.parameters())
        return torch.stack([value.reshape(()) for value in values]).mean()

    def forward(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        noise: torch.Tensor | None = None,
        has_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = current_surface.size(0)
        if self.noise_dim <= 0:
            noise = torch.empty(batch_size, 0, device=current_surface.device, dtype=current_surface.dtype)
        elif noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=current_surface.device, dtype=torch.float32)
        elif int(noise.shape[1]) != self.noise_dim:
            raise ValueError(f"Expected noise dimension {self.noise_dim}, got {noise.shape[1]}.")

        text_features = self.text_encoder(text_embedding)
        text_mask = self._has_text_mask(
            has_text,
            batch_size=batch_size,
            reference=text_features,
        )
        if self.conditioning_mode == "residual_film":
            text_features = text_features * text_mask

        x = self.conv1(current_surface)
        if self.conditioning_mode == "film":
            x = self.film1(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        if self.conditioning_mode == "film":
            x = self.film2(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        if self.conditioning_mode == "film":
            x = self.film3(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        for res_block in self.res_blocks:
            x = res_block(x, text_features)

        surface_features = x.flatten(start_dim=1)
        if self.conditioning_mode != "residual_film":
            fused = torch.cat([surface_features, text_features, noise], dim=1)
            return self.fusion(fused)

        base_delta = self.fusion(torch.cat([surface_features, noise], dim=1))
        modulated = self.residual_bottleneck_film(x, text_features)
        text_delta = self.text_adapter(
            torch.cat([(modulated - x).flatten(start_dim=1), text_features], dim=1)
        )
        return base_delta + text_mask * torch.tanh(self.text_gate) * text_delta


class FilmWGANCritic(nn.Module):
    """CNN critic with FiLM conditioning: text embedding modulates joint surface feature extraction."""

    def __init__(
        self,
        surface_height: int,
        surface_width: int,
        embedding_dim: int,
        base_channels: int,
        res_blocks: int,
        text_hidden_dim: int,
        text_out_dim: int,
        fusion_hidden_dim: int,
        conditioning_mode: str = "film",
        critic_conditioning_mode: str = "inherit",
        text_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_res_blocks = max(0, int(res_blocks))
        generator_conditioning = str(conditioning_mode).strip().lower()
        requested_critic_conditioning = str(critic_conditioning_mode).strip().lower()
        if requested_critic_conditioning == "inherit":
            requested_critic_conditioning = (
                "projection" if generator_conditioning == "residual_film" else generator_conditioning
            )
        self.conditioning_mode = requested_critic_conditioning
        if self.conditioning_mode not in {"film", "concat", "projection"}:
            raise ValueError("critic conditioning must be one of ['film', 'concat', 'projection'].")

        text_layers: list[nn.Module] = [
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if self.conditioning_mode == "projection":
            text_layers.append(nn.Dropout(float(text_dropout)))
        text_layers.extend(
            [
                nn.Linear(text_hidden_dim, text_out_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        if self.conditioning_mode == "projection":
            text_layers.append(nn.Dropout(float(text_dropout)))
        self.text_encoder = nn.Sequential(*text_layers)

        c = base_channels
        self.conv1 = nn.Conv2d(2, c, kernel_size=3, stride=2, padding=1)
        self.film1 = FiLMLayer(text_out_dim, c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.film2 = FiLMLayer(text_out_dim, c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.film3 = FiLMLayer(text_out_dim, c * 4)
        if self.conditioning_mode in {"concat", "projection"}:
            self.film1.requires_grad_(False)
            self.film2.requires_grad_(False)
            self.film3.requires_grad_(False)

        self.res_blocks_list = nn.ModuleList(
            [
                _FiLMResidualConvBlock(c * 4, text_out_dim, conditioning_mode=self.conditioning_mode)
                for _ in range(self.num_res_blocks)
            ]
        )

        reduced_h = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_height)))
        reduced_w = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_width)))
        self.surface_feat_dim = int(c * 4 * reduced_h * reduced_w)

        if self.conditioning_mode == "projection":
            self.classifier = nn.Sequential(
                nn.Linear(self.surface_feat_dim, fusion_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(fusion_hidden_dim, 1),
            )
            self.surface_projection = nn.Linear(self.surface_feat_dim, text_out_dim)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.surface_feat_dim + text_out_dim, fusion_hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(fusion_hidden_dim, 1),
            )

    def forward(
        self,
        future_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        has_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_features = self.text_encoder(text_embedding)
        text_mask = FilmWGANGenerator._has_text_mask(
            has_text,
            batch_size=future_surface.size(0),
            reference=text_features,
        )
        if self.conditioning_mode == "projection":
            text_features = text_features * text_mask

        stacked = torch.cat([current_surface, future_surface], dim=1)

        x = self.conv1(stacked)
        if self.conditioning_mode == "film":
            x = self.film1(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        if self.conditioning_mode == "film":
            x = self.film2(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        if self.conditioning_mode == "film":
            x = self.film3(x, text_features)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)

        for res_block in self.res_blocks_list:
            x = res_block(x, text_features)

        surface_features = x.flatten(start_dim=1)
        if self.conditioning_mode == "projection":
            unconditional_score = self.classifier(surface_features)
            projected_surface = self.surface_projection(surface_features)
            projection_score = (
                projected_surface * text_features
            ).sum(dim=1, keepdim=True) / math.sqrt(float(projected_surface.shape[1]))
            return unconditional_score + text_mask * projection_score
        combined = torch.cat([surface_features, text_features], dim=1)
        return self.classifier(combined)


def reconstruct_future_surface(current_surface_flat: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Reconstruct future volatility levels from current levels and a log-IV increment.

    Clamps future_log into [log(VOL_FLOOR), log(VOL_CEIL)] before exp to prevent overflow
    when the generator emits large deltas during early training.
    """

    current_log = torch.log(torch.clamp(current_surface_flat, min=VOL_FLOOR))
    future_log = torch.clamp(current_log + delta, min=_LOG_VOL_FLOOR, max=_LOG_VOL_CEIL)
    return torch.exp(future_log).clamp(min=VOL_FLOOR, max=VOL_CEIL)
