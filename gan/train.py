import torch
import numpy as np
from torch.utils.data import DataLoader

from config import config
from gan.models.gan_model import WGAN_GP
from gan.utils.dataloader import OptionReturnDataset

# Unpack configurations
R = config['R']
T = config['T']
R = 64
T = 64
num_samples = config['num_samples']
z_dim = config['z_dim']
lr = config['lr']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
device = torch.device(config['device'])
clip_value = config['clip_value']
n_critic = config['n_critic']

# Create synthetic option return data
option_returns = np.random.rand(num_samples, R, T)
option_returns_tensor = torch.tensor(option_returns, dtype=torch.float32)

# Initialize dataset and dataloader
dataset = OptionReturnDataset(option_returns_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize WGAN model

gan = WGAN_GP()

# Train the model


gan.train(dataloader)
