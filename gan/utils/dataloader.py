import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils


# from utils.fashion_mnist import MNIST, FashionMNIST


class OptionReturnDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_dataloader(raw_data: np.ndarray, batch_size=64, shuffle=True):
    """
    Create a DataLoader for the Option Returns dataset.
    :param number_of_data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    # np_data = np.random.rand(number_of_data, x_length, y_length)

    # add a channel dimension
    data_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(data_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # sample shape: (batch_size, 1, x_length, y_length)

    return dataloader


def create_random_dataloader(number_of_data: int, x_length: int, y_length: int, batch_size=64, shuffle=True):
    """
    Create a DataLoader for the Option Returns dataset.
    :param number_of_data:
    :param x_length:
    :param y_length:
    :param batch_size:
    :param shuffle:
    :return:
    """
    np_data = np.random.rand(number_of_data, x_length, y_length)

    # add a channel dimension
    data_tensor = torch.tensor(np_data, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(data_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # sample shape: (batch_size, 1, x_length, y_length)

    return dataloader
