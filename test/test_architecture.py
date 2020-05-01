# pytest --cov-report term-missing --cov=deepSIP test/

# imports -- standard
import numpy as np
import torch
from torch import nn
import pytest

# imports -- custom
from deepSIP.architecture import DropoutCNN
from deepSIP.architecture import _conv_length, _maxpool_length
from deepSIP.architecture import _conv_block, _fc_block

def test_conv_length():
    assert _conv_length(100, 1) == 100
    assert _conv_length(100, 3) == 98

def test_maxpool_length():
    assert _maxpool_length(100, 1) == 100
    assert _maxpool_length(100, 2) == 50
    assert _maxpool_length(101, 2) == 50

def test_conv_block():
    block, L = _conv_block(1, 32, 100, kernel = 5, pooling = 4, dropout = 0.1)
    assert type(block) is nn.Sequential
    assert type(L) in [int, np.int32, np.int64]
    for element in ['convolution', 'activation', 'pooling', 'dropout']:
        assert hasattr(block, element)
    assert block.convolution.kernel_size == (5,)
    assert type(block.activation) is nn.ReLU
    assert block.pooling.kernel_size == 4
    assert block.dropout.p == 0.1

def test_fc_block():
    block = _fc_block(100, 100, activate = True, dropout = 0.1)
    assert type(block) is nn.Sequential
    for element in ['linear', 'activation', 'dropout']:
        assert hasattr(block, element)
    assert block.linear.in_features == 100
    assert block.linear.out_features == 100
    assert type(block.activation) is nn.ReLU
    assert block.dropout.p == 0.1
    block = _fc_block(100, 100, activate = False, dropout = 0.)
    assert not hasattr(block, 'activation')
    assert not hasattr(block, 'dropout')

def test_DropoutCNN():
    network = DropoutCNN(1000, kernel = 5, drop_rate = 0.15, fc_size = 256)
    for i in range(4):
        assert hasattr(network, 'conv{}'.format(i + 1))
    assert hasattr(network, 'fc')
    assert hasattr(network.fc, 'activation')
    assert hasattr(network, 'out')
    x = torch.zeros((100, 1, 1000))
    assert network.forward(x).size() == (100, 1)
    assert network(x).size() == (100, 1)
    with pytest.raises(RuntimeError):
        network(torch.zeros((100, 1, 15)))
