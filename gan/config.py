# config.py
import torch

config = {
    "R": 5,  # Number of different returns
    "T": 10,  # Number of different times-to-maturity
    "num_samples": 1000,  # Number of samples in synthetic option return data
    "z_dim": 100,  # Dimension of the latent space
    "lr": 0.0002,  # Learning rate for optimizers
    "num_epochs": 50,  # Number of epochs to train the GAN
    "batch_size": 64,  # Batch size for training
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device to train on
    "data_path": "data/",  # Path to the dataset directory
    "models_path": "models/",  # Path to save trained models
    "outputs_path": "outputs/"  # Path to save outputs
}
# Update the device based on CUDA availability
if not torch.cuda.is_available():
    config["device"] = "cpu"
