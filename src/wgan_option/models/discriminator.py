import torch
import torch.nn as nn


def _conv2d_out_size(size: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, dilation: int = 1) -> int:
    return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


class Discriminator(nn.Module):
    """
    Conditional critic:
    score(real/fake next surface | current surface, text embedding)
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        surface_height: int,
        surface_width: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.surface_height = surface_height
        self.surface_width = surface_width

        self.surface_encoder = nn.Sequential(
            nn.Conv2d(channels * 2, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        reduced_h = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_height)))
        reduced_w = _conv2d_out_size(_conv2d_out_size(_conv2d_out_size(surface_width)))
        self.surface_feat_dim = 128 * reduced_h * reduced_w

        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.surface_feat_dim + 128, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        next_surface: torch.Tensor,
        current_surface: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat([current_surface, next_surface], dim=1)
        surface_features = self.surface_encoder(stacked).flatten(start_dim=1)
        text_features = self.text_encoder(text_embedding)
        combined = torch.cat([surface_features, text_features], dim=1)
        return self.classifier(combined)
