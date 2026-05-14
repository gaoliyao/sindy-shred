"""Data processing utilities for SINDy-SHRED."""

from scipy.io import loadmat
import numpy as np
import scipy.linalg

# Re-export TimeSeriesDataset for backward compatibility
from utils import TimeSeriesDataset

__all__ = ['TimeSeriesDataset', 'load_data', 'qr_place']


def load_data(name):
    '''Takes string denoting data name and returns the corresponding (N x m) array 
    (N samples of m dimensional state)'''
    if name == 'SST':
        load_X = loadmat('Data/SST_data.mat')['Z'].T
        print(load_X.shape)
        mean_X = np.mean(load_X, axis=0)
        sst_locs = np.where(mean_X != 0)[0]
        return load_X[:, sst_locs]
        

def qr_place(data_matrix, num_sensors):
    '''Takes a (m x N) data matrix consisting of N samples of an m dimensional state and
    number of sensors, returns QR placed sensors and U_r for the SVD X = U S V^T'''
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    rankapprox = u[:, :num_sensors]
    q, r, pivot = scipy.linalg.qr(rankapprox.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, rankapprox

