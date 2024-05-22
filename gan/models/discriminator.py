import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, R, T):
        super(Discriminator, self).__init__()

        # Input shape: (batch_size, 1, R, T)
        # We use 1 input channel because our data is like grayscale in structure
        self.model = nn.Sequential(
            # First convolution
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Second convolution
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third convolution
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Compute the output size after the conv layers
        self.output_size = self._get_conv_output((1, R, T))

        # Final fully connected layer
        self.final_layer = nn.Sequential(
            nn.Linear(self.output_size, 1),
            nn.Sigmoid()
        )

    def _get_conv_output(self, shape):
        # Utility function for calculating the size of the output after convolution layers
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.model(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension if not present
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.final_layer(x)
        return x
