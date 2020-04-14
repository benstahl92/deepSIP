# pytest --cov-report term-missing --cov=deepSNIaID test/

# imports -- standard
import numpy as np
import pytest

# imports -- custom
from deepSNIaID.dataset import NumpyXDataset, NumpyXYDataset

def test_NumpyXDataset():
    X = np.ones((10, 5)).astype(np.float32)
    with pytest.raises(ValueError):
        ds = NumpyXDataset(5)
    ds = NumpyXDataset(X)
    assert len(ds) == 10
    xx = ds[3]
    assert (xx.numpy() == X[3]).all()

def test_NumpyXYDataset():
    X = np.ones((10, 5)).astype(np.float32)
    Y = 5 * np.random.rand(10, 1).astype(np.float32) + 10
    ds = NumpyXYDataset(X, Y, device = 'cpu')
    assert ds.device == 'cpu'
    assert ds.X.shape == (10, 1, 5)
    assert len(ds.X) == len(ds.Y)
    assert (ds.Y.numpy() == Y).all()
    xx, yy = ds[3]
    assert (xx.numpy() == X[3]).all()
    assert (yy.numpy() == Y[3]).all()
    X = np.ones(10)
    with pytest.raises(ValueError):
        ds = NumpyXYDataset(X, Y)
    with pytest.raises(ValueError):
        ds = NumpyXYDataset(5, Y)
