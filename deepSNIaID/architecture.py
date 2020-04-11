# imports -- standard
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

__all__ = ['DropoutCNN']

# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def _conv_length(L_in, kernel, padding = 0, dilation = 1, stride = 1):
    '''compute output length of Conv1d'''
    return np.floor((L_in + 2 * padding - dilation * (kernel - 1) - 1) / \
                    stride + 1).astype(int)

def _maxpool_length(L_in, kernel, padding = 0, dilation = 1, stride = None):
    '''compute output length of MaxPool1d layer'''
    if stride is None:
        stride = kernel
    return np.floor((L_in + 2 * padding - dilation * (kernel - 1) - 1) / \
                    stride + 1).astype(int)

# -----------------------------------------------------------------------------
# Model Components
# -----------------------------------------------------------------------------

def _conv_block(filters_in, filters_out, L_in,
                kernel = 5, pooling = 2, dropout = 0.):
    '''create and return convolution block'''
    model_components = OrderedDict()
    model_components['convolution'] = nn.Conv1d(filters_in, filters_out, kernel,
                                                padding = kernel // 2)
    L = _conv_length(L_in, kernel, padding = kernel // 2)
    model_components['activation'] = nn.ReLU()
    if pooling > 0:
        model_components['pooling'] = nn.MaxPool1d(pooling)
        L = _maxpool_length(L, pooling)
    if dropout > 0.:
        model_components['dropout'] = nn.Dropout(dropout)
    return nn.Sequential(model_components), L

def _fc_block(filters_in, filters_out, activate = True, dropout = 0.):
    '''create and return fully connected block'''
    model_components = OrderedDict()
    model_components['linear'] = nn.Linear(filters_in, filters_out)
    if activate:
        model_components['activation'] = nn.ReLU()
    if dropout > 0.:
        model_components['dropout'] = nn.Dropout(dropout)
    return nn.Sequential(model_components)

# -----------------------------------------------------------------------------
# Core Model Architecture
# -----------------------------------------------------------------------------

class DropoutCNN(nn.Module):
    '''
    core model architecture

    4 x (Conv + ReLU + Max Pooling + Dropout) + (Linear + ReLU + Dropout)
    + Linear

    Parameters
    ----------
    spec_len : int,
               number of wavelength bins for pre-processed spectra
    kernel : odd int, optional
             convolutional kernel size
    filters : int, optional
              number of filters in first convolution layer
    fc_size : int, optional
              number of neurons in fully connected layer
    drop_rate : float, optional
                dropout probability

    Attributes
    ----------
    conv[1-4] : torch.nn.Sequential
                convolution block [1-4]
    fc : torch.nn.Sequential
         fully connected block
    out : toch.nn.Sequential
          output block
    '''

    def __init__(self, spec_len, kernel = 15, filters = 8,
                 fc_size = 32, drop_rate = 0.1):
        super(DropoutCNN, self).__init__()

        # cast types
        spec_len = int(spec_len)
        kernel = int(kernel)
        filters = int(filters)
        drop_rate = float(drop_rate)
        fc_size = int(fc_size)

        self.conv1, L = _conv_block(1, filters, spec_len, kernel = kernel,
                                    pooling = 2, dropout = drop_rate)
        self.conv2, L = _conv_block(filters, filters * 2, L,
                                    kernel = kernel, pooling = 2,
                                    dropout = drop_rate)
        self.conv3, L = _conv_block(filters * 2, filters * 4, L,
                                    kernel = kernel, pooling = 2,
                                    dropout = drop_rate)
        self.conv4, L = _conv_block(filters * 4, filters * 4, L,
                                    kernel = kernel, pooling = 2,
                                    dropout = drop_rate)
        self.fc = _fc_block(filters * 4 * L, fc_size,
                            activate = True, dropout = drop_rate)
        self.out = _fc_block(fc_size, 1, activate = False)

    def forward(self, x):
        '''
        forward pass

        Parameters
        ----------
        x : torch.tensor of shape (batch size, 1, number of wavelength bins)
            inputs to the network
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).flatten(start_dim = 1)
        x = self.fc(x)
        return self.out(x)
