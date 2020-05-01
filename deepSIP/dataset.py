# imports -- standard
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['NumpyXDataset', 'NumpyXYDataset']

class NumpyXDataset(Dataset):
    '''
    create torch tensor dataset from X numpy ndarray

    Parameters
    ----------
    X : np.ndarray of shape (number of spectra, number of wavelength bins)
        pre-processed spectra
    device : str, optional
             device to push tensors to, defaults to GPU if available

    Attributes
    ----------
    X : torch.tensor of shape (number of spectra, 1, number of wavelength bins)
    '''
    def __init__(self, X, device = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() \
                                       else 'cpu')
        else:
            self.device = device
        if type(X) is not np.ndarray:
            raise ValueError('inputs must be np.ndarray type')
        if len(X.shape) == 2:
            # X should be (batch size, N channels = 1, length of signal)
            self.X = X.reshape((X.shape[0], 1, X.shape[1])).astype(np.float32)
        elif len(X.shape) == 3:
            self.X = X.astype(np.float32)
        else:
            raise ValueError('Input X expected to have 2 or 3 components.')
        self.X = torch.from_numpy(self.X).to(self.device)

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, i):
        return self.X[i]

class NumpyXYDataset(NumpyXDataset):
    '''
    create torch tensor dataset from X, Y numpy ndarrays

    Parameters
    ----------
    X : np.ndarray of shape (number of spectra, number of wavelength bins)
        pre-processed spectra
    Y : np.ndarray of shape (number of spectra, number of targets)
        targets
    **kwargs
        arbitrary keyword arguments for NumpyXDataset constructor

    Attributes
    ----------
    X : torch.tensor of shape (number of spectra, 1, number of wavelength bins)
    Y : torch.tensor of shape (number of spectra, number of targets)
    '''
    def __init__(self, X, Y, **kwargs):
        if type(X) != type(Y):
            raise ValueError('X and Y must be the same type')
        NumpyXDataset.__init__(self, X, **kwargs)
        self.Y = torch.from_numpy(Y.astype(np.float32)).to(self.device)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
