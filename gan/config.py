import torch

# Hyperparameters for the WGAN model
config = {
    "R": 224,  # Number of different returns
    "T": 224,  # Number of different times-to-maturity
    "num_samples": 1000,  # Number of samples in synthetic option return data
    "z_dim": 100,  # Dimension of the latent space
    "lr": 0.00005,  # Learning rate for optimizers (lower is often better for WGAN)
    "num_epochs": 50,  # Number of epochs to train the GAN
    "batch_size": 128,  # Batch size for training
    "device": "cpu",  # Device to train on
    "clip_value": 0.01,  # Weight clipping for the critic in WGAN
    "n_critic": 5,  # Number of critic updates per generator update
    "data_path": "data/",  # Path to the dataset directory
    "models_path": "models/",  # Path to save trained models
    "outputs_path": "outputs/"  # Path to save outputs
}
