import torch
import torch.nn as nn


# class Generator(nn.Module):
#     def __init__(self, z_dim, R, T):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(z_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, R * T),
#             nn.Tanh()
#         )
#         self.R = R
#         self.T = T
#
#     def forward(self, z):
#         return self.model(z).view(-1, self.R, self.T)
#


class Generator(nn.Module):
    def __init__(self, z_dim, R, T):
        super(Generator, self).__init__()

        # Start with a fully connected layer
        # We map the latent vector z into a smaller feature map that will be reshaped before being upsampled
        self.fc = nn.Linear(z_dim, 128 * int(R / 4) * int(T / 4))

        self.t_conv_layers = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Second transposed convolution
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Final transposed convolution to get to the desired (R, T) shape
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Calculate the size we need to reshape the initial dense layer output to
        self.init_size = (128, int(R / 4), int(T / 4))

    def forward(self, z):
        # Map z to the reshaped dense layer
        x = self.fc(z)
        x = x.view(-1, *self.init_size)
        # Apply transposed convolutions
        x = self.t_conv_layers(x)
        # Final reshape to get rid of the extra channel dimension
        return x.view(-1, *x.shape[2:4])
