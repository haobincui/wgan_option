import torch
import numpy as np
from torch.utils.data import DataLoader

from config import config
from gan.models.gan_model import WGAN_GP
from gan.utils.dataloader import create_random_dataloader


dataloader = create_random_dataloader(640, 32, 32)

# Initialize WGAN model

gan = WGAN_GP()

# Train the model

print("Training the model...")
gan.train(dataloader)
print("Training complete!")
