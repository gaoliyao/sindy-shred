import torch
from scipy.io import loadmat
import numpy as np
import scipy.linalg

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len

def load_data(name):
    '''Takes string denoting data name and returns the corresponding (N x m) array 
    (N samples of m dimensional state)'''
    if name == 'SST':
        load_X = loadmat('/home/marsgao/pyshred/Data/SST_data.mat')['Z'].T
        print(load_X.shape)
        mean_X = np.mean(load_X, axis=0)
        sst_locs = np.where(mean_X != 0)[0]
        return load_X[:, sst_locs]

    if name == 'AO3':
        load_X = np.load('/home/marsgao/pyshred/Data/short_svd_O3.npy')
        return load_X

    if name == 'ISO':
#         load_X = np.load('/home/marsgao/pyshred/Data/numpy_isotropic.npy')
#         print(load_X.shape)
        load_X = np.load('/home/marsgao/pyshred/Data/numpy_isotropic.npy').reshape(-1, 350*350)
        return load_X
    
    if name == 'PEN':
#         load_X = np.load('/home/marsgao/pyshred/Data/numpy_isotropic.npy')
#         print(load_X.shape)
        load_X = np.load('/home/marsgao/pyshred/Data/pendulum_smoothed.npy').reshape(-1, 27*24)
        load_X = load_X[:389,:]
        print(load_X.shape)
        return load_X
    
    if name == 'FLOW':
        load_X = np.load('/home/marsgao/pyshred/Data/flow_over_cylinder_extended.npy').reshape(-1, 400*1000)
        print(load_X.shape)
        return load_X
    
    if name == 'METRONOME':
        load_X = np.load('/home/marsgao/pyshred/Data/metronome_synchronization.npy').reshape(-1, 48*72)
        print(load_X.shape)
        return load_X
    
    if name == 'SUN':
        load_X = np.load('/home/marsgao/pyshred/Data/sun_data_aoa_pooling_normalized.npy').reshape(-1, 271*271)
        print(load_X.shape)
        return load_X
    
    if name == 'KOL':
        load_X = np.load('/home/marsgao/pyshred/Data/original_kolflow_long_re30.npy').reshape(6000, 80*80, 2)
        print(load_X.shape)
        return load_X
    
    if name == 'KOL_RE20':
        load_X = np.load('/home/marsgao/pyshred/Data/original_kolflow_long_re20.npy').reshape(6000, 80*80, 2)
        print(load_X.shape)
        return load_X
    
    if name == 'KOL_RE40':
        load_X = np.load('/home/marsgao/pyshred/Data/original_kolflow_long_re40.npy').reshape(6000, 128*128, 2)
        print(load_X.shape)
        return load_X
        


def qr_place(data_matrix, num_sensors):
    '''Takes a (m x N) data matrix consisting of N samples of an m dimensional state and
    number of sensors, returns QR placed sensors and U_r for the SVD X = U S V^T'''
    u, s, v = np.linalg.svd(data_matrix, full_matrices=False)
    rankapprox = u[:, :num_sensors]
    q, r, pivot = scipy.linalg.qr(rankapprox.T, pivoting=True)
    sensor_locs = pivot[:num_sensors]
    return sensor_locs, rankapprox

