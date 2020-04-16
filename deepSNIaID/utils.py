# imports -- standard
import os
import json
import random
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def load_txt_spectrum(filename):
    '''
    load spectrum from txt file with first two columns as wave flux

    Parameters
    ----------
    filename : str
               name of text file to read spectrum from

    Returns
    -------
    wave : np.array
           wavelength grid
    flux : np.array
           fluxes on wavelength grid
    '''
    data = pd.read_csv(filename, delim_whitespace = True,
                       header = None, comment = '#')
    wave = data.loc[:, 0].values
    flux = data.loc[:, 1].values
    return wave, flux

def savenet(network, filename):
    '''
    save network parameters

    Parameters
    ----------
    network : DropoutCNN or torch network
              network to save
    filename : str
               name of file to save network to
    '''
    torch.save(network.state_dict(), filename)

def loadnet(network, filename, device = 'cpu'):
    '''
    load network parameters

    Parameters
    ----------
    network : DropoutCNN
              network to load weights and biases into
    filename : str
               name of file to load network from
    device : str, optional
             device to load network onto ('cpu' by default)
    '''
    if device == 'cpu':
        sd = torch.load(filename, map_location = 'cpu')
    else:
        sd = torch.load(filename)
    network.load_state_dict(sd)

# -----------------------------------------------------------------------------
# Spectrum Operations
# -----------------------------------------------------------------------------

def deredshift(wave, z):
    '''
    transform wavelength to rest frame

    Parameters
    ----------
    wave : np.array
           wavelength grid
    z : float
        redshift of SN

    Returns
    -------
    np.array
        rest-frame wavelength grid
    '''
    return wave / (1 + z)

def smooth(flux, window, order):
    '''
    smooth using a savitzky-golay filter

    Parameters
    ----------
    flux : np.array
           fluxes
    window : int
             window size in angstroms for smoothing
    smoothing : int
                polynomial order for smoothing

    Returns
    -------
    np.array
        smoothed spectrum
    '''
    return savgol_filter(flux, window, order)

def get_continuum(flux, window, order):
    '''
    model continuum from *heavily* smoothed spectrum

    Notes
    -----
    see documentiation for smooth, which this function simply wraps
    '''
    return smooth(flux, window, order)

def log_wave(bounds, n_bins):
    '''
    compute logarithmic wavelength grid

    Parameters
    ----------
    bounds : tuple, list, or other iterable of length 2
             lower and upper limit of logarithmic wavelength array
    n_bins : int
             number of bins in logarithmic wavelength grid

    Returns
    -------
    np.array
        logarithmic wavelength grid
    '''
    dl = np.log(bounds[1] / bounds[0]) / n_bins
    n = np.arange(0.5, n_bins + 0.5)
    return bounds[0] * np.exp(n * dl)

def _log_bin_loop(wave, flux, bounds, n_bins):
    '''rebin onto logarithmic wavelength grid (adapted from astrodash/SNID)'''

    # compute needed values and setup destination array
    lwave = log_wave(bounds, n_bins)
    dl = np.log(bounds[1] / bounds[0]) / n_bins
    lflux = np.zeros(n_bins)

    # iterate over wavelength bins
    for i in range(len(wave)):

        # find boundaries
        if i == 0:
            s0 = 0.5 * (3 * wave[i] - wave[i + 1])
            s1 = 0.5 * (wave[i] + wave[i + 1])
        elif i == len(wave) - 1:
            s0 = 0.5 * (wave[i - 1] + wave[i])
            s1 = 0.5 * (3 * wave[i] - wave[i - 1])
        else:
            s0 = 0.5 * (wave[i - 1] + wave[i])
            s1 = 0.5 * (wave[i] + wave[i + 1])
        s0log = np.log(s0 / bounds[0]) / dl + 1
        s1log = np.log(s1 / bounds[0]) / dl + 1
        dnu = s1 - s0

        # rebin
        for j in range(int(s0log), int(s1log)):
            if (j < 0) or (j >= n_bins):
                continue
            lflux[j] += flux[i] / (s1log - s0log) * dnu

    # find valid region and set zero elsewhere
    valid_mask = np.logical_and(lwave >= wave.min(), lwave <= wave.max())
    lflux[~valid_mask] = 0.

    return lwave, lflux, valid_mask

def _log_bin_vec(wave, flux, bounds, n_bins):
    '''rebin onto logarithmic wavelength grid (adapted from astrodash/SNID)'''

    # compute new wavelength grid
    lwave = log_wave(bounds, n_bins)

    # find valid region and use to mask input wave and flux
    mask = np.logical_and(wave >= lwave.min(), wave <= lwave.max())
    wave = wave[mask]
    flux = flux[mask]

    # setup needed arrays
    dl = np.log(bounds[1] / bounds[0]) / n_bins
    lflux = np.zeros(n_bins)
    s0, s1 = np.zeros(len(wave)), np.zeros(len(wave))

    # find boundaries
    s0[0] = 0.5 * (3 * wave[0] - wave[1])
    s1[0] = 0.5 * (wave[0] + wave[1])
    s0[-1] = 0.5 * (wave[-2] + wave[-1])
    s1[-1] = 0.5 * (3 * wave[-1] - wave[-2])
    s0[1:-1] = 0.5 * (wave[:-2] + wave[1:-1])
    s1[1:-1] = 0.5 * (wave[1:-1] + wave[2:])
    s0log = np.log(s0 / bounds[0]) / dl + 1
    s1log = np.log(s1 / bounds[0]) / dl + 1
    dnu = s1 - s0

    # rebin
    n_loops = s1log.astype(int) - s0log.astype(int)
    indices = np.flatnonzero(n_loops)
    values = s0log.astype(int)[indices]
    prepend = values[0] if values[0] < 0 else False
    if prepend is not False:
        values[0] = 0
        n_loops[0] += prepend
    n_loops = n_loops[indices][values < n_bins]
    fluxes = ((flux / (s1log - s0log) * dnu)[indices])[values < n_bins]
    fluxes = np.repeat(fluxes, n_loops)
    min_idx = min(values)
    m_cand = max(values) + n_loops[-1]
    max_idx = (m_cand if (m_cand < n_bins) else n_bins)
    lflux[min_idx:max_idx] = fluxes[:(max_idx - min_idx)]

    # find valid region
    valid_mask = np.logical_and(lwave >= wave.min(), lwave <= wave.max())
    lflux[~valid_mask] = 0.

    return lwave, lflux, valid_mask

def log_bin(wave, flux, bounds, n_bins):
    '''
    rebin onto logarithmic wavelength grid (adapted from astrodash/SNID)

    Parameters
    ----------
    wave : np.array
           wavelength grid
    flux : np.array
           fluxes on wavelength grid
    bounds : tuple, list, or other iterable of length 2
             lower and upper limit of logarithmic wavelength array
    n_bins : int
             number of bins in logarithmic wavelength grid

    Returns
    -------
    lwave : np.array
            logarithmic wavelength grid
    lflux : np.array
            rebinned fluxes on grid
    valid_mask : boolean array
                 selection mask for signal on grid
    '''
    try:
        return _log_bin_vec(wave, flux, bounds, n_bins)
    except Exception as e:
        print('Warning: problem (' + str(e) + ') encountered when ' + \
              'attempting vectorized log binning')
        print('Using slower looping method')
        return _log_bin_loop(wave, flux, bounds, n_bins)

def normalize_flux(flux):
    '''
    normalize flux range of 1 centered at 0

    Parameters
    ----------
    flux : np.array
           fluxes

    Returns
    -------
    np.array
        normalized flux
    '''
    nflux = (flux - flux.min()) / (flux.max() - flux.min())
    return nflux - 0.5

def apodize(flux, end_pct):
    '''
    taper spectrum at each end to zero using hanning window

    Parameters
    ----------
    flux : np.array
           fluxes
    end_pct : float
              percentage to taper at each end

    Returns
    -------
    np.array
        apodized flux
    '''
    size = int(end_pct * len(flux))
    h = np.hanning(2 * size)
    window = np.concatenate((h[:size], np.ones(len(flux) - 2 * size), h[-size:]))
    return flux * window

# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------

def redshift(wave, z):
    '''
    transform wavelength to observer frame

    Parameters
    ----------
    wave : np.array
           rest wavelength grid
    z : float
        redshift of SN relative to observer

    Returns
    -------
    np.array
        observer-frame wavelength grid
    '''
    return wave * (1 + z)

def drop_ends(wave, flux, max_drop_frac):
    '''
    drop random fraction of spectrum from ends

    Parameters
    ----------
    wave : np.array
           wavelength grid
    flux : np.array
           fluxes on wavelength grid
    max_drop_frac : float
                    maximum fraction to be dropped from each end

    Returns
    -------
    np.array, np.array
        truncated wave and flux arrays
    '''
    if int(max_drop_frac * len(wave) / 2) <= 1:
        return wave, flux
    left = np.random.randint(0, int(max_drop_frac * len(wave)))
    right = np.random.randint(int(len(wave) * (1 - max_drop_frac)), len(wave))
    return wave[left:right], flux[left:right]

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def reset_state(seed = 100):
    '''
    set all seeds for reproducibility

    Parameters
    ----------
    seed : int, optional
           seed for random number generator

    Notes
    -----
    seeds are set for torch (including cuda), numpy, random, and PHYTHONHASH;
    this appears to be sufficient to ensure reproducibility
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# -----------------------------------------------------------------------------
# Target Scaling
# -----------------------------------------------------------------------------

class VoidScaler:
    '''
    sklearn conforming scaler that has no effect (i.e. identity transformation)

    Parameters
    ----------
    *args
        arbitrary arguments with no effect
    **kwargs
        arbitrary keyword arguments with no effect
    '''
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X
    def fit_transform(self, X):
        return self.transform(X)

class LinearScaler(VoidScaler):
    '''
    sklearn conforming that behaves as MinMaxScaler but min and max are fixed

    Parameters
    ----------
    [min,max]imum : float or np.array
                    [min,max]imum value(s) used for transformation

    Attributes
    ----------
    [min,max] : float or np.array
                see Parameters
    '''
    def __init__(self, minimum = 0., maximum = 1.):
        self.min = minimum
        self.max = maximum
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min

# -----------------------------------------------------------------------------
# Type Conversion
# -----------------------------------------------------------------------------

def torch2numpy(*tensors):
    '''
    convert input torch tensors to numpy ndarrays

    Parameters
    ----------
    *tensors
        torch tensors to convert to corresponding numpy ndarrays

    Returns
    -------
    ndarrays: np.ndarrays
        numpy ndarrays corresponding to input torch tensors
    '''
    ndarrays = []
    for tensor in tensors:
        if type(tensor) is not torch.Tensor:
            raise ValueError('input type ({}) must be a torch.Tensor object!')
        ndarrays.append(tensor.cpu().detach().numpy())
    if len(ndarrays) == 1:
        return ndarrays[0]
    return ndarrays

# -----------------------------------------------------------------------------
# Network Utilities
# -----------------------------------------------------------------------------

def count_params(network):
    '''
    count trainable parameters in network

    Parameters
    ----------
    network : torch network
              network to count trainable parameters for

    Returns
    -------
    int
        number of trainable parameters in input network
    '''
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

def init_weights(network_component):
    '''
    initialize weights (normal) and biases (slight positive)

    Parameters
    ----------
    network_component : component torch network
                        component to initialize
    '''
    if type(network_component) in [nn.Linear, nn.Conv1d]:
        with torch.no_grad():
            nn.init.normal_(network_component.weight, mean = 0., std = 0.1)
            network_component.bias.data.fill_(0.1)

def stochastic_predict(network, X, mcnum = 75, sigmoid = False, seed = None,
                       scaler = None, status = False):
    '''
    perform mcnum stochastic forward passes, return mean and std np.arrays

    Parameters
    ----------
    network : torch network
              network to generate predictions with
    X : torch tensor
        inputs to network
    mcnum : int, optional
            number of stochastic forward passes to perform
    sigmoid : bool, optional
              apply sigmoid to outputs
    seed : int, optional
           seed for random number generator
    scaler : sklearn conforming scaler
             scaler to inverse transform outputs with
    status : bool, optional
             show status bars

    Returns
    -------
    mean : np.ndarray
           mean prediction per input
    std : np.ndarray
          std of predictions per input

    Notes
    -----
    Does mcnum forward passes on each row of input, one at a time, with
    re-seeding between each row. This ensures that predictions on a given input
    will be the same regardless of the order of inputs. It is possible the
    algorithm could be optimized for speed, but in tests it performs comparably
    to the previous implementation that suffered from modest reproducibility
    issues.
    '''
    if scaler is None:
        scaler = VoidScaler()
    network.train() # keep dropout so network is probabilistic
    with torch.no_grad():
        if status:
            X = tqdm(X)
        # do predictions one at a time make reproducible regardless of order
        predictions = np.zeros((mcnum, len(X), 1))
        for i, XX in enumerate(X):
            if type(seed) is int:
                reset_state(seed = seed)
            # reshape for network input and repeat in mcnum times
            XX = XX.reshape(1, 1, XX.size()[-1]).repeat(mcnum, 1, 1)
            pred = network(XX)
            if sigmoid:
                pred = torch.sigmoid(pred)
            predictions[:,i,:] = scaler.inverse_transform(torch2numpy(pred))
    return predictions.mean(axis = 0), predictions.std(axis = 0)

class WrappedModel(nn.Module):
    '''
    wrapper to seamlessly load network on cpu from dicts saved with DataParallel

    Parameters
    ----------
    network : torch network
              network to wrap

    Attributes
    ----------
    module : torch network
             see Parameters
    '''
    def __init__(self, network):
        super(type(self), self).__init__()
        self.module = network
    def forward(self, *x):
        '''pass arguments to forward of module'''
        return self.module(*x)

class VoidLRScheduler:
    '''torch conforming LR scheduler with no effect'''
    def __init__(self):
        pass
    def _reset(self):
        pass
    def step(self, *args):
        pass

# -----------------------------------------------------------------------------
# Custom Metrics and Wrappers
# -----------------------------------------------------------------------------

def rmse(y, yhat):
    '''
    root-mean-square error

    Parameters
    ----------
    y : np.ndarray
        true labels for each unique target
    yhat : np.ndarray
           predictions for each unique target

    Returns
    -------
    rmse : np.ndarray
           root-mean-square error for each unique target
    '''
    assert y.shape == yhat.shape
    return np.sqrt( np.mean((y - yhat)**2, axis = 0) )

def wrmse(y, yhat, sigma_hat):
    '''
    inverse predicted variance weighted root-mean-square error

    Parameters
    ----------
    y : np.ndarray
        true labels for each unique target
    yhat : np.ndarray
           predictions for each unique target
    sigma_hat : np.ndarray
                predicted uncertainties for each unique target

    Returns
    -------
    wrmse : np.ndarray
            weighted root-mean-square error for each unique target
    '''
    assert y.shape == yhat.shape
    assert yhat.shape == sigma_hat.shape
    return np.sqrt( np.sum((y - yhat)**2 * (sigma_hat **-2), axis = 0) / \
                    np.sum(sigma_hat**-2, axis = 0) )

def outlier(y, yhat):
    '''
    largest absolute residual

    Parameters
    ----------
    y : np.ndarray
        true labels for each unique target
    yhat : np.ndarray
           predictions for each unique target

    Returns
    -------
    mr : np.ndarray
         maximum absolute residual for each unique target
    '''
    return np.max(np.abs(y - yhat), axis = 0)

def slope(y, yhat):
    '''
    slope of linear fit to yhat as a function of y

    Parameters
    ----------
    y : np.array
        true labels
    yhat : np.array
           predictions

    Returns
    -------
    slope : float
            slope of fitted line to yhat vs y
    '''
    assert y.shape == yhat.shape
    assert len(y.shape) == 1
    try:
        p = np.polyfit(y, yhat, 1)
    except np.linalg.LinAlgError:
        print('Warning: error in slope calculation. Setting to nan')
        p = [np.nan]
    return p[0]

def regression_metrics(y, yhat, sigma_hat, key_prepend = ''):
    '''
    compute all regression-related metrics

    Parameters
    ----------
    y : np.ndarray
        true labels for each unique target
    yhat : np.ndarray
           predictions for each unique target
    sigma_hat : np.ndarray
                predicted uncertainties for each unique target
    key_prepend : str, optional
                  label to prepend output dict keys with

    Returns
    -------
    metrics : dict
              all computed regression metrics for each unique target
    '''
    metrics = {}
    rms = rmse(y, yhat)
    wrms = wrmse(y, yhat, sigma_hat)
    mr = outlier(y, yhat)
    if type(rms) is not np.ndarray:
        rms = [rms]
        wrms = [wrms]
        mr = [mr]
    for i in range(len(rms)):
        metrics['{}rmse_{}'.format(key_prepend, i + 1)] = rms[i]
        metrics['{}wrmse_{}'.format(key_prepend, i + 1)] = wrms[i]
        metrics['{}mr_{}'.format(key_prepend, i + 1)] = mr[i]
        if len(y.shape) > 1:
            yy, yh = y[:, i], yhat[:, i]
        else:
            yy, yh = y, yhat
        metrics['{}slope_{}'.format(key_prepend, i + 1)] = slope(yy, yh)
    return metrics

def classify(p, threshold = 0.5):
    '''
    given probabilities, perform binary classification using threshold

    Parameters
    ----------
    p : np.array
        probabilities for binary classification
    threshold : float, optional
                decision threshold for binary classification

    Returns
    -------
    np.array
        binary classifications
    '''
    p = p.copy() # don't overwrite input
    p[p > threshold] = 1.
    p[p <= threshold] = 0.
    return p

def classification_metrics(y, p, threshold = 0.5, key_prepend = ''):
    '''
    compute all classification related metrics

    Parameters
    ----------
    y : np.array
        true labels
    p : np.array
        probabilities for binary classification
    threshold : float, optional
                decision threshold for binary classification
    key_prepend : str, optional
                  label to prepend output dict keys with

    Returns
    -------
    metrics : dict
              all computed classification metrics
    '''
    yhat = classify(p, threshold = threshold)
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    auc = roc_auc_score(y, p)
    return {key_prepend + 'acc': acc,
            key_prepend + 'f1': f1,
            key_prepend + 'auc': auc}
