from dataclasses import dataclass

import torch

# Hyperparameters for the WGAN model
config = {
    "R": 8,  # Number of different returns
    "T": 8,  # Number of different times-to-maturity
    "num_samples": 1000,  # Number of samples in synthetic option return data
    "z_dim": 100,  # Dimension of the latent space
    "lr": 0.00005,  # Learning rate for optimizers (lower is often better for WGAN)
    "num_epochs": 50,  # Number of epochs to train the GAN
    "batch_size": 128,  # Batch size for training
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device to train on
    "n_critic": 5,  # Number of critic updates per generator update
    "data_path": "data/",  # Path to the dataset directory
    "models_path": "models/",  # Path to save trained models
    "outputs_path": "outputs/"  # Path to save outputs
}


@dataclass
class Config:
    channels: int = 1
    # R: int = 8
    # T: int = 8
    num_samples: int = 1000
    # z_dim: int = 100
    learning_rate: float = 0.00005
    num_epochs: int = 50  # generate 50 epochs
    batch_size: int = 64
    beta_1: float = 0.5
    beta_2: float = 0.999

    cuda: bool = torch.cuda.is_available()

    discriminator_iter: int = 5
    # data_path: str = "data/"
    models_path: str = "./model_params/"
    outputs_path: str = "./output/"


default_config = Config()
