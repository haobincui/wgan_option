import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wgan_option.models.common import (
    BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
    FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_GENERATOR_CONDITIONING_MODE,
    FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    FULL_CURRENT_GENERATOR_INPUT_MODE,
    GAUSSIAN_GENERATOR_NOISE_MODE,
    IDENTITY_RESIDUAL_OUTPUT_MODE,
    LEGACY_RESIDUAL_OUTPUT_MODE,
    STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
    ZERO_GENERATOR_NOISE_MODE,
    ResidualConvBlock,
    apply_residual_surface_output,
    conv2d_out_size,
    generator_current_encoder_input,
    generator_current_input_fingerprint,
    generator_conditioning_fingerprint,
    generator_noise_fingerprint,
    generator_noise_tensor,
    normalize_generator_noise_mode,
    normalize_generator_conditioning_mode,
    normalize_generator_current_input_mode,
    normalize_residual_output_mode,
    residual_output_fingerprint,
    zero_initialize_residual_head,
)

# Keep underscore-prefixed aliases so existing checkpoint-loading code
# that may reference these names via pickle/torch.load keeps working.
_ResidualConvBlock = ResidualConvBlock
_conv2d_out_size = conv2d_out_size

GENERATOR_PARAMETER_ROLES = (
    "backbone",
    "text_encoder",
    "conditioning_module",
)


def _text_encoder(
    embedding_dim: int,
    text_hidden_dim: int,
    text_out_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(embedding_dim), int(text_hidden_dim)),
        nn.LayerNorm(int(text_hidden_dim)),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.1),
        nn.Linear(int(text_hidden_dim), int(text_out_dim)),
        nn.LeakyReLU(0.2, inplace=True),
    )


class SurfaceTextCrossAttention(nn.Module):
    """Let spatial surface queries attend to virtual LP-text tokens."""

    def __init__(self, channels: int, attention_dim: int, heads: int):
        super().__init__()
        if int(attention_dim) <= 0 or int(heads) <= 0:
            raise ValueError("attention_dim and heads must be positive")
        if int(attention_dim) % int(heads):
            raise ValueError("attention_dim must be divisible by heads")
        self.query_projection = nn.Linear(int(channels), int(attention_dim))
        self.attention = nn.MultiheadAttention(
            int(attention_dim), int(heads), batch_first=True
        )
        self.output_projection = nn.Linear(int(attention_dim), int(channels))
        self.output_norm = nn.LayerNorm(int(channels))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        queries = self.query_projection(features.flatten(start_dim=2).transpose(1, 2))
        attended, _ = self.attention(
            queries,
            text_tokens,
            text_tokens,
            need_weights=False,
        )
        residual = self.output_norm(self.output_projection(attended))
        residual = residual.transpose(1, 2).reshape(batch_size, channels, height, width)
        return features + self.residual_gate * residual


class StyleModulatedConv2d(nn.Module):
    """Per-example modulated convolution with optional weight demodulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        *,
        stride: int = 1,
        demodulate: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.demodulate = bool(demodulate)
        self.weight = nn.Parameter(
            torch.randn(self.out_channels, self.in_channels, 3, 3)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.style_projection = nn.Linear(int(style_dim), self.in_channels)
        nn.init.zeros_(self.style_projection.weight)
        nn.init.zeros_(self.style_projection.bias)
        self.weight_scale = 1.0 / math.sqrt(float(self.in_channels * 3 * 3))

    def forward(
        self,
        features: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        if int(channels) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {int(channels)}"
            )
        modulation = (1.0 + self.style_projection(style)).view(
            batch_size, 1, self.in_channels, 1, 1
        )
        weight = self.weight.unsqueeze(0) * self.weight_scale * modulation
        if self.demodulate:
            weight = weight * torch.rsqrt(
                weight.square().sum(dim=(2, 3, 4), keepdim=True) + 1e-8
            )
        grouped_features = features.reshape(
            1, batch_size * self.in_channels, height, width
        )
        grouped_weight = weight.reshape(
            batch_size * self.out_channels, self.in_channels, 3, 3
        )
        output = F.conv2d(
            grouped_features,
            grouped_weight,
            stride=self.stride,
            padding=1,
            groups=batch_size,
        )
        return output.reshape(
            batch_size, self.out_channels, output.shape[-2], output.shape[-1]
        ) + self.bias.view(1, -1, 1, 1)


class _FiLMLayer(nn.Module):
    """Identity-initialized channel-wise affine modulation from encoded LP text."""

    def __init__(self, conditioning_dim: int, channels: int):
        super().__init__()
        self.projection = nn.Linear(int(conditioning_dim), int(channels) * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self, features: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.projection(conditioning).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * features + beta


class Generator(nn.Module):
    """
    Conditional generator:
    (current surface, news embedding, noise) -> next surface
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        noise_dim: int,
        surface_height: int,
        surface_width: int,
        base_channels: int = 32,
        res_blocks: int = 0,
        text_hidden_dim: int = 256,
        text_out_dim: int = 128,
        hidden_dim: int = 512,
        residual_output_mode: str = LEGACY_RESIDUAL_OUTPUT_MODE,
        generator_noise_mode: str = GAUSSIAN_GENERATOR_NOISE_MODE,
        generator_current_input_mode: str = FULL_CURRENT_GENERATOR_INPUT_MODE,
        generator_conditioning_mode: str = BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        strike_grid: object | None = None,
        maturity_grid_days: object | None = None,
        crossattn_heads: int = 4,
        crossattn_text_tokens: int = 4,
        crossattn_dim: int = 128,
        transformer_model_dim: int = 96,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 384,
        transformer_dropout: float = 0.1,
        style_dim: int = 128,
        style_demodulate: bool = True,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.surface_height = surface_height
        self.surface_width = surface_width
        self.base_channels = base_channels
        self.res_blocks = max(0, int(res_blocks))
        self.residual_output_mode = normalize_residual_output_mode(residual_output_mode)
        self.residual_output_fingerprint = residual_output_fingerprint(
            self.residual_output_mode
        )
        self.generator_noise_mode = normalize_generator_noise_mode(generator_noise_mode)
        self.generator_noise_fingerprint = generator_noise_fingerprint(
            self.generator_noise_mode,
            self.noise_dim,
        )
        self.generator_current_input_mode = normalize_generator_current_input_mode(
            generator_current_input_mode
        )
        self.generator_current_input_fingerprint = generator_current_input_fingerprint(
            self.generator_current_input_mode
        )
        self.generator_conditioning_mode = normalize_generator_conditioning_mode(
            generator_conditioning_mode
        )
        self.generator_conditioning_fingerprint = generator_conditioning_fingerprint(
            self.generator_conditioning_mode
        )
        self.crossattn_heads = int(crossattn_heads)
        self.crossattn_text_tokens = int(crossattn_text_tokens)
        self.crossattn_dim = int(crossattn_dim)
        self.transformer_model_dim = int(transformer_model_dim)
        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ffn_dim = int(transformer_ffn_dim)
        self.transformer_dropout = float(transformer_dropout)
        self.style_dim = int(style_dim)
        self.style_demodulate = bool(style_demodulate)
        architecture_modes = {
            CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        }
        if (
            self.generator_conditioning_mode
            in {
                FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
                FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                *architecture_modes,
            }
            and self.res_blocks != 0
        ):
            raise ValueError(
                f"{self.generator_conditioning_mode} requires gen_res_blocks=0"
            )

        if self.generator_conditioning_mode in architecture_modes:
            if int(channels) != 1:
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires channels=1"
                )
            if (int(surface_height), int(surface_width)) != (16, 16):
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires an exact 16x16 grid"
                )
            if (
                self.generator_noise_mode != GAUSSIAN_GENERATOR_NOISE_MODE
                or int(self.noise_dim) != 32
            ):
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires Gaussian32"
                )
            if self.residual_output_mode != IDENTITY_RESIDUAL_OUTPUT_MODE:
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires identity residual output"
                )
            if (
                self.generator_current_input_mode
                != CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            ):
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires "
                    "generator_current_input_mode='current_support_masked'"
                )
            self._validate_architecture_hyperparameters()
            self._register_film_unet_coordinates(
                strike_grid=strike_grid,
                maturity_grid_days=maturity_grid_days,
            )
            if (
                self.generator_conditioning_mode
                == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ):
                self._build_crossattn_unet(
                    embedding_dim=embedding_dim,
                    text_hidden_dim=text_hidden_dim,
                    text_out_dim=text_out_dim,
                )
            elif (
                self.generator_conditioning_mode
                == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ):
                self._build_transformer_tokens(
                    embedding_dim=embedding_dim,
                    text_hidden_dim=text_hidden_dim,
                    text_out_dim=text_out_dim,
                )
            else:
                self._build_stylemod_unet(
                    embedding_dim=embedding_dim,
                    text_hidden_dim=text_hidden_dim,
                    text_out_dim=text_out_dim,
                )
            return

        if self.generator_conditioning_mode in {
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        }:
            if int(channels) != 1:
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires channels=1"
                )
            if (
                self.generator_current_input_mode
                != CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE
            ):
                raise ValueError(
                    f"{self.generator_conditioning_mode} requires "
                    "generator_current_input_mode='current_support_masked'"
                )
            input_channels = 1
            if self.generator_conditioning_mode in {
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            }:
                input_channels = 4
                self._register_film_unet_coordinates(
                    strike_grid=strike_grid,
                    maturity_grid_days=maturity_grid_days,
                )
            if (
                self.generator_conditioning_mode
                == CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ):
                self._build_cnn_unet(
                    input_channels=input_channels,
                    embedding_dim=embedding_dim,
                    text_hidden_dim=text_hidden_dim,
                    text_out_dim=text_out_dim,
                )
            else:
                self._build_film_unet(
                    input_channels=input_channels,
                    embedding_dim=embedding_dim,
                    text_hidden_dim=text_hidden_dim,
                    text_out_dim=text_out_dim,
                )
            return

        encoder_layers = [
            nn.Conv2d(channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(self.res_blocks):
            encoder_layers.append(_ResidualConvBlock(base_channels * 4))
        self.surface_encoder = nn.Sequential(*encoder_layers)

        reduced_h = _conv2d_out_size(_conv2d_out_size(surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(surface_width))
        self.surface_feat_dim = (base_channels * 4) * reduced_h * reduced_w

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        bottleneck_text_concat = (
            self.generator_conditioning_mode
            != FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
        )
        fusion_dim = self.surface_feat_dim + noise_dim
        if bottleneck_text_concat:
            fusion_dim += text_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, surface_height * surface_width),
        )
        if self.residual_output_mode == IDENTITY_RESIDUAL_OUTPUT_MODE:
            zero_initialize_residual_head(self.fusion[-1])

        self.encoder_film_layers = nn.ModuleList()
        if self.generator_conditioning_mode in {
            FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
            FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
        }:
            # FiLM is an additive experimental branch.  Preserve the global CPU
            # RNG state so paired concat/FiLM construction leaves every shared
            # parameter and the caller's subsequent RNG stream unchanged.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.encoder_film_layers.extend(
                    [
                        _FiLMLayer(text_out_dim, base_channels),
                        _FiLMLayer(text_out_dim, base_channels * 2),
                        _FiLMLayer(text_out_dim, base_channels * 4),
                    ]
                )
            finally:
                torch.random.set_rng_state(cpu_rng_state)

    def _validate_architecture_hyperparameters(self) -> None:
        mode = self.generator_conditioning_mode
        if mode == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
            if self.crossattn_heads <= 0 or self.crossattn_text_tokens <= 0:
                raise ValueError(
                    "Cross-attention heads and text-token count must be positive"
                )
            if self.crossattn_dim <= 0 or self.crossattn_dim % self.crossattn_heads:
                raise ValueError("crossattn_dim must be divisible by crossattn_heads")
        elif mode == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
            if self.transformer_model_dim <= 0 or self.transformer_heads <= 0:
                raise ValueError("Transformer dimensions must be positive")
            if self.transformer_model_dim % self.transformer_heads:
                raise ValueError(
                    "transformer_model_dim must be divisible by transformer_heads"
                )
            if self.transformer_layers <= 0 or self.transformer_ffn_dim <= 0:
                raise ValueError(
                    "Transformer layers and FFN dimension must be positive"
                )
            if not 0.0 <= self.transformer_dropout < 1.0:
                raise ValueError("transformer_dropout must be in [0, 1)")
        elif mode == STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE:
            if self.style_dim <= 0:
                raise ValueError("style_dim must be positive")
            if not self.style_demodulate:
                raise ValueError(
                    "stylemod_unet_mask_coords_v1 requires demodulated convolutions"
                )

    def _build_crossattn_unet(
        self,
        *,
        embedding_dim: int,
        text_hidden_dim: int,
        text_out_dim: int,
    ) -> None:
        channels = self.base_channels
        self.surface_encoder = nn.ModuleList(
            [
                nn.Conv2d(4, channels, 3, padding=1),
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            ]
        )
        self.text_encoder = _text_encoder(
            embedding_dim,
            text_hidden_dim,
            text_out_dim,
        )
        self.text_token_projection = nn.Linear(
            int(text_out_dim), self.crossattn_text_tokens * self.crossattn_dim
        )
        self.bottleneck_conv = nn.Conv2d(
            channels * 4 + self.noise_dim,
            channels * 4,
            3,
            padding=1,
        )
        self.decoder_convs = nn.ModuleList(
            [
                nn.Conv2d(channels * 6, channels * 2, 3, padding=1),
                nn.Conv2d(channels * 3, channels, 3, padding=1),
            ]
        )
        self.cross_attention_layers = nn.ModuleList(
            [
                SurfaceTextCrossAttention(
                    channels * 4,
                    self.crossattn_dim,
                    self.crossattn_heads,
                ),
                SurfaceTextCrossAttention(
                    channels * 2,
                    self.crossattn_dim,
                    self.crossattn_heads,
                ),
                SurfaceTextCrossAttention(
                    channels,
                    self.crossattn_dim,
                    self.crossattn_heads,
                ),
            ]
        )
        self.residual_head = nn.Conv2d(channels, 1, 1)
        zero_initialize_residual_head(self.residual_head)

    def _build_transformer_tokens(
        self,
        *,
        embedding_dim: int,
        text_hidden_dim: int,
        text_out_dim: int,
    ) -> None:
        model_dim = self.transformer_model_dim
        self.surface_token_projection = nn.Linear(4, model_dim)
        self.row_embedding = nn.Parameter(
            torch.randn(self.surface_height, model_dim) * 0.02
        )
        self.column_embedding = nn.Parameter(
            torch.randn(self.surface_width, model_dim) * 0.02
        )
        self.text_encoder = _text_encoder(
            embedding_dim,
            text_hidden_dim,
            text_out_dim,
        )
        self.text_token_projection = nn.Linear(int(text_out_dim), model_dim)
        self.noise_token_encoder = nn.Sequential(
            nn.Linear(self.noise_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.transformer_input_norm = nn.LayerNorm(model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ffn_dim,
            dropout=self.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.transformer_layers,
        )
        self.transformer_output_norm = nn.LayerNorm(model_dim)
        self.residual_head = nn.Linear(model_dim, 1)
        zero_initialize_residual_head(self.residual_head)

    def _build_stylemod_unet(
        self,
        *,
        embedding_dim: int,
        text_hidden_dim: int,
        text_out_dim: int,
    ) -> None:
        channels = self.base_channels
        self.text_encoder = _text_encoder(
            embedding_dim,
            text_hidden_dim,
            text_out_dim,
        )
        self.style_vector_projection = nn.Linear(int(text_out_dim), self.style_dim)
        self.surface_encoder = nn.ModuleList(
            [
                StyleModulatedConv2d(
                    4,
                    channels,
                    self.style_dim,
                    demodulate=self.style_demodulate,
                ),
                StyleModulatedConv2d(
                    channels,
                    channels * 2,
                    self.style_dim,
                    stride=2,
                    demodulate=self.style_demodulate,
                ),
                StyleModulatedConv2d(
                    channels * 2,
                    channels * 4,
                    self.style_dim,
                    stride=2,
                    demodulate=self.style_demodulate,
                ),
            ]
        )
        self.bottleneck_conv = StyleModulatedConv2d(
            channels * 4 + self.noise_dim,
            channels * 4,
            self.style_dim,
            demodulate=self.style_demodulate,
        )
        self.decoder_convs = nn.ModuleList(
            [
                StyleModulatedConv2d(
                    channels * 6,
                    channels * 2,
                    self.style_dim,
                    demodulate=self.style_demodulate,
                ),
                StyleModulatedConv2d(
                    channels * 3,
                    channels,
                    self.style_dim,
                    demodulate=self.style_demodulate,
                ),
            ]
        )
        self.residual_head = nn.Conv2d(channels, 1, 1)
        zero_initialize_residual_head(self.residual_head)

    def _build_film_unet(
        self,
        *,
        input_channels: int,
        embedding_dim: int,
        text_hidden_dim: int,
        text_out_dim: int,
    ) -> None:
        """Build the fully convolutional FiLM encoder-decoder branch."""

        base_channels = self.base_channels
        self.surface_encoder = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channels,
                    base_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(
                    base_channels,
                    base_channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    base_channels * 2,
                    base_channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        reduced_h = _conv2d_out_size(_conv2d_out_size(self.surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(self.surface_width))
        self.surface_feat_dim = (base_channels * 4) * reduced_h * reduced_w

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.bottleneck_conv = nn.Conv2d(
            base_channels * 4 + self.noise_dim,
            base_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.decoder_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    base_channels * 6,
                    base_channels * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(
                    base_channels * 3,
                    base_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]
        )
        self.residual_head = nn.Conv2d(
            base_channels, 1, kernel_size=1, stride=1, padding=0
        )
        zero_initialize_residual_head(self.residual_head)

        # All FiLM projections are deliberately identity-initialized. Preserve
        # the caller's RNG stream because Linear initializes before being zeroed.
        cpu_rng_state = torch.random.get_rng_state()
        try:
            self.encoder_film_layers = nn.ModuleList(
                [
                    _FiLMLayer(text_out_dim, base_channels),
                    _FiLMLayer(text_out_dim, base_channels * 2),
                    _FiLMLayer(text_out_dim, base_channels * 4),
                ]
            )
            self.bottleneck_film_layer = _FiLMLayer(text_out_dim, base_channels * 4)
            self.decoder_film_layers = nn.ModuleList(
                [
                    _FiLMLayer(text_out_dim, base_channels * 2),
                    _FiLMLayer(text_out_dim, base_channels),
                ]
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)

    @staticmethod
    def _consume_text_encoder_initialization_rng(
        *,
        embedding_dim: int,
        text_hidden_dim: int,
        text_out_dim: int,
    ) -> None:
        """Match FiLM U-Net construction RNG without retaining text modules.

        This keeps every shared convolution and the subsequently constructed
        Critic identical under a paired seed.  The temporary module is never
        assigned to ``self``, so it contributes no parameters or computation
        to the pure-CNN model.
        """

        omitted_text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(text_hidden_dim, text_out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        del omitted_text_encoder

    def _build_cnn_unet(
        self,
        *,
        input_channels: int,
        embedding_dim: int,
        text_hidden_dim: int,
        text_out_dim: int,
    ) -> None:
        """Build a mask-and-coordinate U-Net with no text or FiLM modules."""

        base_channels = self.base_channels
        self.surface_encoder = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channels,
                    base_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(
                    base_channels,
                    base_channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    base_channels * 2,
                    base_channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        reduced_h = _conv2d_out_size(_conv2d_out_size(self.surface_height))
        reduced_w = _conv2d_out_size(_conv2d_out_size(self.surface_width))
        self.surface_feat_dim = (base_channels * 4) * reduced_h * reduced_w

        self._consume_text_encoder_initialization_rng(
            embedding_dim=embedding_dim,
            text_hidden_dim=text_hidden_dim,
            text_out_dim=text_out_dim,
        )
        self.bottleneck_conv = nn.Conv2d(
            base_channels * 4 + self.noise_dim,
            base_channels * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.decoder_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    base_channels * 6,
                    base_channels * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Conv2d(
                    base_channels * 3,
                    base_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            ]
        )
        self.residual_head = nn.Conv2d(
            base_channels, 1, kernel_size=1, stride=1, padding=0
        )
        zero_initialize_residual_head(self.residual_head)

    @staticmethod
    def _normalized_coordinate(
        values: object | None,
        *,
        expected_size: int,
        logarithmic: bool,
        name: str,
    ) -> torch.Tensor:
        if values is None:
            return torch.linspace(-1.0, 1.0, steps=expected_size)
        coordinate = torch.as_tensor(values, dtype=torch.float32).flatten()
        if int(coordinate.numel()) != int(expected_size):
            raise ValueError(
                f"{name} must contain {expected_size} values, got "
                f"{int(coordinate.numel())}"
            )
        if not bool(torch.isfinite(coordinate).all()):
            raise ValueError(f"{name} values must be finite")
        if logarithmic:
            if bool((coordinate <= 0.0).any()):
                raise ValueError(f"{name} values must be strictly positive")
            coordinate = coordinate.log()
        minimum = coordinate.min()
        span = coordinate.max() - minimum
        if float(span) == 0.0:
            return torch.zeros_like(coordinate)
        return 2.0 * (coordinate - minimum) / span - 1.0

    def _register_film_unet_coordinates(
        self,
        *,
        strike_grid: object | None,
        maturity_grid_days: object | None,
    ) -> None:
        moneyness = self._normalized_coordinate(
            strike_grid,
            expected_size=self.surface_width,
            logarithmic=False,
            name="strike_grid",
        )
        log_ttm = self._normalized_coordinate(
            maturity_grid_days,
            expected_size=self.surface_height,
            logarithmic=True,
            name="maturity_grid_days",
        )
        moneyness_grid = moneyness.view(1, 1, 1, -1).expand(
            1, 1, self.surface_height, self.surface_width
        )
        log_ttm_grid = log_ttm.view(1, 1, -1, 1).expand(
            1, 1, self.surface_height, self.surface_width
        )
        # Coordinates are determined by the separately fingerprinted surface
        # grid, so avoid duplicating them in model checkpoints.
        self.register_buffer(
            "moneyness_coordinate", moneyness_grid.contiguous(), persistent=False
        )
        self.register_buffer(
            "log_ttm_coordinate", log_ttm_grid.contiguous(), persistent=False
        )

    def _film_unet_encoder_input(
        self,
        current_surface: torch.Tensor,
        current_support_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        masked_surface = generator_current_encoder_input(
            current_surface,
            current_support_mask,
            mode=CURRENT_SUPPORT_MASKED_GENERATOR_INPUT_MODE,
        )
        # The helper above performs the authoritative shape, finite, binary,
        # and nonempty validation before this second channel is materialized.
        support_mask = current_support_mask.to(
            device=current_surface.device,
            dtype=current_surface.dtype,
            non_blocking=True,
        )
        if support_mask.ndim == 3:
            support_mask = support_mask.unsqueeze(1)
        if self.generator_conditioning_mode == FILM_UNET_GENERATOR_CONDITIONING_MODE:
            return masked_surface
        batch_size = int(current_surface.shape[0])
        coordinates = [
            self.moneyness_coordinate.expand(batch_size, -1, -1, -1),
            self.log_ttm_coordinate.expand(batch_size, -1, -1, -1),
        ]
        return torch.cat([masked_surface, support_mask, *coordinates], dim=1)

    @staticmethod
    def _film_activate(
        features: torch.Tensor,
        film_layer: _FiLMLayer,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        return F.leaky_relu(
            film_layer(features, text_features),
            negative_slope=0.2,
            inplace=False,
        )

    def _film_unet_delta(
        self,
        encoder_surface: torch.Tensor,
        text_features: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a surface residual without flattening spatial features."""

        encoder_features: list[torch.Tensor] = []
        x = encoder_surface
        for conv, film_layer in zip(
            self.surface_encoder, self.encoder_film_layers, strict=True
        ):
            x = self._film_activate(conv(x), film_layer, text_features)
            encoder_features.append(x)

        spatial_noise = noise[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        x = self.bottleneck_conv(torch.cat([x, spatial_noise], dim=1))
        x = self._film_activate(x, self.bottleneck_film_layer, text_features)

        x = F.interpolate(
            x,
            size=encoder_features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.decoder_convs[0](torch.cat([x, encoder_features[1]], dim=1))
        x = self._film_activate(x, self.decoder_film_layers[0], text_features)

        x = F.interpolate(
            x,
            size=encoder_features[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.decoder_convs[1](torch.cat([x, encoder_features[0]], dim=1))
        x = self._film_activate(x, self.decoder_film_layers[1], text_features)
        return self.residual_head(x)

    def _cnn_unet_delta(
        self,
        encoder_surface: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a residual through the text-free convolutional path."""

        encoder_features: list[torch.Tensor] = []
        x = encoder_surface
        for conv in self.surface_encoder:
            x = F.leaky_relu(conv(x), negative_slope=0.2, inplace=False)
            encoder_features.append(x)

        spatial_noise = noise[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        x = F.leaky_relu(
            self.bottleneck_conv(torch.cat([x, spatial_noise], dim=1)),
            negative_slope=0.2,
            inplace=False,
        )

        x = F.interpolate(
            x,
            size=encoder_features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = F.leaky_relu(
            self.decoder_convs[0](torch.cat([x, encoder_features[1]], dim=1)),
            negative_slope=0.2,
            inplace=False,
        )

        x = F.interpolate(
            x,
            size=encoder_features[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = F.leaky_relu(
            self.decoder_convs[1](torch.cat([x, encoder_features[0]], dim=1)),
            negative_slope=0.2,
            inplace=False,
        )
        return self.residual_head(x)

    def _crossattn_unet_delta(
        self,
        encoder_surface: torch.Tensor,
        text_features: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        encoder_features: list[torch.Tensor] = []
        features = encoder_surface
        for convolution in self.surface_encoder:
            features = F.leaky_relu(
                convolution(features),
                negative_slope=0.2,
                inplace=False,
            )
            encoder_features.append(features)
        text_tokens = self.text_token_projection(text_features).reshape(
            text_features.shape[0],
            self.crossattn_text_tokens,
            self.crossattn_dim,
        )
        spatial_noise = noise[:, :, None, None].expand(
            -1, -1, features.shape[-2], features.shape[-1]
        )
        features = F.leaky_relu(
            self.bottleneck_conv(torch.cat([features, spatial_noise], dim=1)),
            negative_slope=0.2,
            inplace=False,
        )
        features = self.cross_attention_layers[0](features, text_tokens)
        features = F.interpolate(
            features,
            size=encoder_features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = F.leaky_relu(
            self.decoder_convs[0](torch.cat([features, encoder_features[1]], dim=1)),
            negative_slope=0.2,
            inplace=False,
        )
        features = self.cross_attention_layers[1](features, text_tokens)
        features = F.interpolate(
            features,
            size=encoder_features[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = F.leaky_relu(
            self.decoder_convs[1](torch.cat([features, encoder_features[0]], dim=1)),
            negative_slope=0.2,
            inplace=False,
        )
        features = self.cross_attention_layers[2](features, text_tokens)
        return self.residual_head(features)

    def _transformer_tokens_delta(
        self,
        encoder_surface: torch.Tensor,
        text_features: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(encoder_surface.shape[0])
        surface_tokens = encoder_surface.permute(0, 2, 3, 1).reshape(
            batch_size,
            self.surface_height * self.surface_width,
            4,
        )
        surface_tokens = self.surface_token_projection(surface_tokens)
        positions = (
            self.row_embedding[:, None, :] + self.column_embedding[None, :, :]
        ).reshape(
            1,
            self.surface_height * self.surface_width,
            self.transformer_model_dim,
        )
        surface_tokens = surface_tokens + positions
        text_token = self.text_token_projection(text_features).unsqueeze(1)
        noise_token = self.noise_token_encoder(noise).unsqueeze(1)
        tokens = torch.cat([text_token, noise_token, surface_tokens], dim=1)
        encoded = self.transformer_encoder(self.transformer_input_norm(tokens))
        surface_encoded = self.transformer_output_norm(encoded[:, 2:, :])
        return (
            self.residual_head(surface_encoded)
            .transpose(1, 2)
            .reshape(
                batch_size,
                1,
                self.surface_height,
                self.surface_width,
            )
        )

    def _stylemod_unet_delta(
        self,
        encoder_surface: torch.Tensor,
        text_features: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        style = self.style_vector_projection(text_features)
        encoder_features: list[torch.Tensor] = []
        features = encoder_surface
        for convolution in self.surface_encoder:
            features = F.leaky_relu(
                convolution(features, style),
                negative_slope=0.2,
                inplace=False,
            )
            encoder_features.append(features)
        spatial_noise = noise[:, :, None, None].expand(
            -1, -1, features.shape[-2], features.shape[-1]
        )
        features = F.leaky_relu(
            self.bottleneck_conv(
                torch.cat([features, spatial_noise], dim=1),
                style,
            ),
            negative_slope=0.2,
            inplace=False,
        )
        features = F.interpolate(
            features,
            size=encoder_features[1].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = F.leaky_relu(
            self.decoder_convs[0](
                torch.cat([features, encoder_features[1]], dim=1),
                style,
            ),
            negative_slope=0.2,
            inplace=False,
        )
        features = F.interpolate(
            features,
            size=encoder_features[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = F.leaky_relu(
            self.decoder_convs[1](
                torch.cat([features, encoder_features[0]], dim=1),
                style,
            ),
            negative_slope=0.2,
            inplace=False,
        )
        return self.residual_head(features)

    def parameter_role(self, name: str) -> str:
        """Return the stable split-LR role for one Generator parameter."""

        if name.startswith("text_encoder."):
            return "text_encoder"
        if self.generator_conditioning_mode in {
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        } and name.startswith(
            (
                "encoder_film_layers.",
                "bottleneck_film_layer.",
                "decoder_film_layers.",
            )
        ):
            return "conditioning_module"
        if (
            self.generator_conditioning_mode
            == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            and name.startswith(("text_token_projection.", "cross_attention_layers."))
        ):
            return "conditioning_module"
        if (
            self.generator_conditioning_mode
            == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            and name.startswith(
                (
                    "text_token_projection.",
                    "noise_token_encoder.",
                    "transformer_input_norm.",
                    "transformer_encoder.",
                    "transformer_output_norm.",
                )
            )
        ):
            return "conditioning_module"
        if self.generator_conditioning_mode == (
            STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        ) and (
            name.startswith("style_vector_projection.") or ".style_projection." in name
        ):
            return "conditioning_module"
        return "backbone"

    def parameter_roles(self) -> dict[str, str]:
        """Return a complete, disjoint named-parameter-to-role mapping."""

        roles = {name: self.parameter_role(name) for name, _ in self.named_parameters()}
        invalid = set(roles.values()) - set(GENERATOR_PARAMETER_ROLES)
        if invalid:
            raise RuntimeError(f"Generator emitted invalid parameter roles: {invalid}")
        if len(roles) != sum(1 for _ in self.named_parameters()):
            raise RuntimeError("Generator parameter-role mapping is incomplete")
        return roles

    def _film_surface_features(
        self,
        encoder_surface: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run the three fixed convolution stages with pre-activation FiLM."""

        if self.res_blocks != 0 or len(self.surface_encoder) != 6:
            raise RuntimeError(
                f"{self.generator_conditioning_mode} requires exactly three "
                "encoder convolutions and gen_res_blocks=0"
            )
        x = encoder_surface
        for stage, (conv_index, activation_index) in enumerate(
            ((0, 1), (2, 3), (4, 5))
        ):
            x = self.surface_encoder[conv_index](x)
            x = self.encoder_film_layers[stage](x, text_features)
            x = self.surface_encoder[activation_index](x)
        return x.flatten(start_dim=1)

    def forward(
        self,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
        noise: torch.Tensor = None,
        current_support_mask: torch.Tensor | None = None,
    ):
        batch_size = current_surface.size(0)
        if noise is None:
            noise = generator_noise_tensor(
                current_surface,
                batch_size=batch_size,
                noise_dim=self.noise_dim,
                mode=self.generator_noise_mode,
                preserve_gaussian_rng_progression=self.training,
            )
        expected_shape = (batch_size, self.noise_dim)
        if tuple(noise.shape) != expected_shape:
            raise ValueError(
                f"Expected generator noise shape {expected_shape}, got {tuple(noise.shape)}"
            )
        if self.generator_noise_mode == ZERO_GENERATOR_NOISE_MODE:
            # The persisted mode is authoritative even if an external caller
            # accidentally supplies a nonzero tensor.
            noise = torch.zeros_like(noise)

        if self.generator_conditioning_mode in {
            FILM_UNET_GENERATOR_CONDITIONING_MODE,
            FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
        }:
            encoder_surface = self._film_unet_encoder_input(
                current_surface,
                current_support_mask,
            )
        else:
            encoder_surface = generator_current_encoder_input(
                current_surface,
                current_support_mask,
                mode=self.generator_current_input_mode,
            )
        if (
            self.generator_conditioning_mode
            == CNN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
        ):
            delta = self._cnn_unet_delta(encoder_surface, noise)
        else:
            text_features = self.text_encoder(text_embedding)
            if (
                self.generator_conditioning_mode
                == CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ):
                delta = self._crossattn_unet_delta(
                    encoder_surface,
                    text_features,
                    noise,
                )
            elif (
                self.generator_conditioning_mode
                == TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ):
                delta = self._transformer_tokens_delta(
                    encoder_surface,
                    text_features,
                    noise,
                )
            elif (
                self.generator_conditioning_mode
                == STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE
            ):
                delta = self._stylemod_unet_delta(
                    encoder_surface,
                    text_features,
                    noise,
                )
            elif self.generator_conditioning_mode in {
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            }:
                delta = self._film_unet_delta(
                    encoder_surface,
                    text_features,
                    noise,
                )
            elif self.generator_conditioning_mode in {
                FILM_CONV_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
                FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE,
            }:
                surface_features = self._film_surface_features(
                    encoder_surface,
                    text_features,
                )
            else:
                surface_features = self.surface_encoder(encoder_surface).flatten(
                    start_dim=1
                )
            if self.generator_conditioning_mode not in {
                FILM_UNET_GENERATOR_CONDITIONING_MODE,
                FILM_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                CROSSATTN_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                TRANSFORMER_TOKENS_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
                STYLEMOD_UNET_MASK_COORDS_GENERATOR_CONDITIONING_MODE,
            }:
                fusion_parts = [surface_features]
                if (
                    self.generator_conditioning_mode
                    != FILM_CONV_NO_BOTTLENECK_CONCAT_GENERATOR_CONDITIONING_MODE
                ):
                    fusion_parts.append(text_features)
                fusion_parts.append(noise)
                delta = self.fusion(torch.cat(fusion_parts, dim=1)).view(
                    batch_size, 1, self.surface_height, self.surface_width
                )

        return apply_residual_surface_output(
            current_surface,
            delta,
            mode=self.residual_output_mode,
        )
