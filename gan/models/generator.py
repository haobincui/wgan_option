import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, R, T):
        super(Generator, self).__init__()
        self.init_size = R // 16  # Prepare the initial size
        self.init_size_t = T // 16

        self.fc = nn.Linear(z_dim, 512 * self.init_size * self.init_size_t)

        self.t_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, self.init_size, self.init_size_t)
        x = self.t_conv_layers(x)
        return x
