"""Utility functions and classes for SINDy-SHRED.

This module consolidates common utilities used across the SINDy-SHRED codebase.
"""

import torch


def get_device():
    """Detect and return the best available compute device.

    Returns
    -------
    device : torch.device
        The best available device (MPS > CUDA > CPU).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class TimeSeriesDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for time series data.

    Takes input sequence of sensor measurements with shape (batch_size, lags, num_sensors)
    and corresponding measurements of high-dimensional state.

    Parameters
    ----------
    X : torch.Tensor
        Input sensor trajectories, shape (n_samples, lags, num_sensors).
    Y : torch.Tensor
        Target high-dimensional states, shape (n_samples, state_dim).
    """

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len
