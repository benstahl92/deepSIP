# pytest --cov-report term-missing --cov=deepSIP test/

# imports -- standard
import os
import numpy as np
import torch
import pytest

# imports -- custom
from deepSIP import utils
from deepSIP.utils import _log_bin_loop, _log_bin_vec
from deepSIP.architecture import _fc_block, DropoutCNN

# globals for testing
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'spectra',
                         'sn2016coj-20160610.245-ui.flm')
REDSHIFT = 0.004483
WAVE, FLUX = utils.load_txt_spectrum(SPEC_FILE)

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def test_load_txt_spectrum():
	wave, flux = utils.load_txt_spectrum(SPEC_FILE)
	assert type(wave) == type(np.array([0.]))
	assert len(wave) == len(flux)

def test_saveloadnet(tmpdir):
    utils.reset_state()
    block1 = _fc_block(10, 10, activate = False, dropout = 0.)
    block2 = _fc_block(10, 10, activate = False, dropout = 0.)
    assert (block1.linear.bias.data != block2.linear.bias.data).all()
    filename = str(tmpdir.join('tmp.pth'))
    utils.savenet(block1, filename)
    utils.loadnet(block2, filename, 'cpu')
    assert (block1.linear.bias.data == block2.linear.bias.data).all()

# -----------------------------------------------------------------------------
# Spectrum Operations
# -----------------------------------------------------------------------------

def test_deredshift():
	w = utils.deredshift(WAVE, REDSHIFT)
	assert (w < WAVE).all()
	w = utils.deredshift(WAVE, 0)
	assert (w == WAVE).all()

def test_smooth():
    f = utils.smooth(FLUX, 45, 5)
    assert len(f) == len(FLUX)
    assert FLUX.max() >= f.max()
    assert FLUX.std() >= f.std()

def test_get_continuum():
    f = utils.get_continuum(FLUX, 451, 3)
    assert len(f) == len(FLUX)
    assert f.min() >= FLUX.min()
    assert f.max() <= FLUX.max()

def test_log_wave():
    w = utils.log_wave((3400, 8900), 1024)
    assert len(w) == 1024
    assert w[1] - w[0] < w[-1] - w[-2]

def test_log_bin_methods():
	lw_l, lflux_l, valid_mask_l = _log_bin_loop(WAVE, FLUX, (3400, 8900), 1024)
	lw_v, lflux_v, valid_mask_v = _log_bin_vec(WAVE, FLUX, (3400, 8900), 1024)
	assert (lw_l == lw_v).all()
	assert (lflux_l == lflux_v).all()
	assert (valid_mask_l == valid_mask_v).all()

def test_log_bin():
    lwave, lflux, valid_mask = utils.log_bin(WAVE, FLUX, (3400, 8900), 1024)
    assert len(lwave) == 1024
    assert len(lwave) == len(lflux)

def test_normalization():
    f = utils.normalize_flux(FLUX)
    assert len(f) == len(FLUX)
    assert f.min() >= -0.5
    assert f.max() <= 0.5

def test_apodize():
    f = utils.apodize(FLUX, 0.05)
    assert np.abs(FLUX[0]) > np.abs(f[0])
    assert np.abs(FLUX[-1]) > np.abs(f[-1])

# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------

def test_redshift():
	w = utils.redshift(WAVE, REDSHIFT)
	assert (w > WAVE).all()
	w = utils.redshift(WAVE, 0)
	assert (w == WAVE).all()

def test_drop_ends():
    w, f = utils.drop_ends(WAVE, FLUX, 0)
    assert (w == WAVE).all()
    assert (f == FLUX).all()
    w, f = utils.drop_ends(WAVE, FLUX, 0.25)
    assert len(w) < len(WAVE)
    assert len(f) < len(FLUX)
    assert len(w) == len(f)

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def test_reset_state():
    utils.reset_state(seed = 100)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ['PYTHONHASHSEED'] == '100'

# -----------------------------------------------------------------------------
# Target Scaling
# -----------------------------------------------------------------------------

def test_VoidScaler():
    Y = np.ones((10, 2))
    scaler = utils.VoidScaler(*[0, 2], **{'unused_kwd': 30})
    assert scaler == scaler.fit(Y)
    assert (Y == scaler.fit_transform(Y)).all()
    assert (Y == scaler.inverse_transform(Y)).all()

def test_LinearScaler():
    Y = np.ones((10, 2))
    scaler = utils.LinearScaler(*[0, 2])
    assert (Y == scaler.fit_transform(2 * Y)).all()
    assert (2 * Y == scaler.inverse_transform(Y)).all()

# -----------------------------------------------------------------------------
# Type Conversion
# -----------------------------------------------------------------------------

def test_torch2numpy():
    t1 = torch.zeros(2)
    t2 = torch.zeros((2,3))
    t3 = torch.zeros((2,3,4))
    assert type(t1) is torch.Tensor
    a1 = utils.torch2numpy(t1)
    assert type(a1) is np.ndarray
    assert a1.shape == t1.size()
    a2, a3 = utils.torch2numpy(t2, t3)
    assert type(a2) is np.ndarray
    assert a2.shape == t2.size()
    assert type(a3) is np.ndarray
    assert a3.shape == t3.size()
    with pytest.raises(ValueError):
        utils.torch2numpy(5)

# -----------------------------------------------------------------------------
# Network Utilities
# -----------------------------------------------------------------------------

def test_count_params():
    block1 = _fc_block(10, 10, activate = False, dropout = 0.)
    assert utils.count_params(block1) == 110 # 10 x 10 weights + 10 biases

def test_init_weights():
    block1 = _fc_block(10, 10, activate = False, dropout = 0.)
    assert (block1.linear.bias.data != 0.1).any()
    block1.apply(utils.init_weights)
    assert (block1.linear.bias.data == 0.1).all()

def test_stochastic_predict():
    X = torch.ones((20, 1, 40))
    #block = _fc_block(10, 1, activate = False, dropout = 0.)
    block = DropoutCNN(40, kernel = 1, filters = 2, fc_size = 4, drop_rate = 0.)
    mu, std = utils.stochastic_predict(block, X, seed = 100, mcnum = 5)
    assert len(mu) == 20
    assert (std == 0).all()
    mu1, std1 = utils.stochastic_predict(block, X, seed = 100, mcnum = 5)
    assert (mu1 == mu).all()
    mu2, std2 = utils.stochastic_predict(block, X, sigmoid = True,
                                         seed = 100, mcnum = 5)
    assert (mu1 != mu2).any()

def test_WrappedModel():
    block = _fc_block(10, 10, activate = False, dropout = 0.)
    wblock = utils.WrappedModel(block)
    assert (block.linear.bias.data == wblock.module.linear.bias.data).all()
    t1 = torch.ones((5, 1, 10))
    assert (block(t1) == wblock(t1)).all()

def test_VoidLRScheduler():
    scheduler = utils.VoidLRScheduler()
    initial_state = vars(scheduler)
    scheduler._reset()
    assert vars(scheduler) == initial_state
    scheduler.step()
    assert vars(scheduler) == initial_state

# -----------------------------------------------------------------------------
# Custom Metrics and Wrappers
# -----------------------------------------------------------------------------

def test_regression_metrics():
    y0 = np.arange(1, 11) * 1. # cast to float dtype
    y1 = np.arange(11, 21) * 1. # cast to float dtype
    r = utils.rmse(y0, y1)
    wr = utils.wrmse(y0, y1, y1)
    assert r == 10. # all diffs are 10
    assert r == wr
    assert wr == utils.wrmse(y0, y1, 5 * y1)
    d = utils.regression_metrics(y0, y1, y1)
    assert len(d.keys()) == 4 # rmse, wrmse, mr, slope
    assert d['rmse_1'] == r
    assert d['wrmse_1'] == wr
    assert (r == utils.rmse(y0.reshape(-1, 1), y1.reshape(-1, 1))).all()
    assert (wr == utils.wrmse(y0.reshape(-1, 1), y1.reshape(-1, 1),
                             y1.reshape(-1, 1))).all()
    assert (r == utils.rmse(y0.reshape(1, -1), y1.reshape(1, -1))).all()
    assert (wr == utils.wrmse(y0.reshape(1, -1), y1.reshape(1, -1),
                             y1.reshape(1, -1))).all()

def test_classification_metrics():
    y = np.ones(10)
    p25 = 0.25 * np.ones(10)
    p75 = 0.75 * np.ones(10)
    assert (utils.classify(p25) == np.zeros(10)).all()
    assert (utils.classify(p75) == np.ones(10)).all()
    assert (utils.classify(p75, threshold = 0.8) == np.zeros(10)).all()
    y[:5] = 0.
    p = p25
    p[5:] = 0.75
    d = utils.classification_metrics(y, p)
    assert len(d.keys()) == 3 # acc, f1, auc
    assert d['acc'] == 1.
    assert d['auc'] == 1.
    d = utils.classification_metrics(y, p[::-1])
    assert d['acc'] == 0.
    assert d['auc'] == 0.
    p[5] = 0.25
    d = utils.classification_metrics(y, p)
    assert d['acc'] == 0.9
    assert d['auc'] == 0.9
