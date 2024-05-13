
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, values, frequencies):
        """
        values: Numpy array of shape (N, D) where N is the number of distinct data points and D is the dimension of each data point.
        frequencies: Numpy array of shape (N,) where N is the number of distinct data points and each element is the frequency of the corresponding value in 'values'.
        """
        self.values = values
        self.frequencies = frequencies

        # We create an expanded list of indices based on frequencies
        # This will allow us to use random choice efficiently
        self.expanded_indices = np.repeat(np.arange(len(values)), frequencies)

    def __len__(self):
        # Total number of samples is the sum of the frequencies
        return self.expanded_indices.shape[0]

    def __getitem__(self, index):
        # Get the data point index from the expanded indices
        data_point_index = self.expanded_indices[index]
        data_point = self.values[data_point_index]
        return data_point

# Example usage:
# values = np.array([[1, 2], [3, 4], [5, 6]])
# frequencies = np.array([10, 20, 30])
# dataset = CustomDataset(values, frequencies)
# dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
