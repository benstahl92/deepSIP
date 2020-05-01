# pytest --cov-report term-missing --cov=deepSIP test/

# imports -- standard
import os
import numpy as np
import pandas as pd
import pytest

# imports -- custom
from deepSIP.preprocessing import Spectrum, EvaluationSpectra, TVTSpectra

# globals for testing
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'spectra',
                         'sn2016coj-20160610.245-ui.flm')
REDSHIFT = 0.004483
SPECTRUM = Spectrum(SPEC_FILE, REDSHIFT)
SPECTRA_FILE = os.path.join(os.path.dirname(TEST_DIR), 'example', 'spectra.csv')
DF = pd.read_csv(SPECTRA_FILE)
DF['filename'] = DF['filename'].apply(lambda f: \
                                      os.path.join(os.path.dirname(TEST_DIR),
                                      'example', 'spectra', f))
DATASET = EvaluationSpectra(DF)
TVT = TVTSpectra(DF)

def test_Spectrum_init():
    s_obsframe = Spectrum(SPEC_FILE, REDSHIFT)
    s_z0 = Spectrum(SPEC_FILE, 0)
    s_restframe = Spectrum(SPEC_FILE, REDSHIFT, obsframe = False)
    assert (s_z0.wave == s_restframe.wave).all()
    assert (s_obsframe.wave < s_restframe.wave).all()
    assert s_obsframe.z == REDSHIFT
    assert len(s_obsframe.lwave) == s_obsframe.lwave_n_bins

def test_Spectrum_process():
    _f = SPECTRUM._process(SPECTRUM.wave, SPECTRUM.flux)
    f = SPECTRUM.process()
    af = SPECTRUM.augprocess(0.01, 100)
    assert (_f == f).all()
    assert (f != af).any()
    assert f[0] == f[-1]
    assert f[0] == 0.5
    assert f.max() <= 1.
    assert f.min() >= 0.

def test_Spectrum_plot_raise():
    with pytest.raises(ValueError):
        SPECTRUM.plot(show = 'unsupported show type')

def test_EvaluationSpectra_init():
    with pytest.raises(TypeError):
        ds = EvaluationSpectra(DF.values)
    with pytest.raises(ValueError):
        ds = EvaluationSpectra(DF.drop(['SN'], axis = 'columns'))
    ds = EvaluationSpectra(DF.drop(['obsframe'], axis = 'columns'))
    assert 'obsframe' in ds.spectra.columns
    assert (ds.spectra['obsframe'] == True).all()
    assert 'Spectrum' in DATASET.spectra.columns
    assert isinstance(DATASET.spectra.loc[0, 'Spectrum'], Spectrum)

def test_EvaluationSpectra_len():
    assert len(DATASET) == 50

def test_save_load(tmpdir):
    savefile = tmpdir.join('tmp.sav')
    DATASET.savefile = savefile
    with pytest.raises(FileNotFoundError):
        DATASET.load()
    DATASET.test_attr = 1
    DATASET.save()
    DATASET.load()
    assert hasattr(DATASET, 'test_attr')
    assert DATASET.test_attr == 1
    ds = EvaluationSpectra(str(savefile))
    assert ds.savefile == savefile
    assert hasattr(ds, 'test_attr')
    assert ds.test_attr == 1

def test_EvaluationSpectra_process():
    DATASET.process()
    assert hasattr(DATASET, 'X')
    assert DATASET.X.shape == (50, 1024)
    X = DATASET.X
    delattr(DATASET, 'X')
    DATASET.process(status = True)
    assert (DATASET.X == X).all()

def test_EvaluationSpectra_to_npy(tmpdir):
    DATASET.savefile = str(tmpdir.join('tmp.sav'))
    DATASET.to_npy()
    assert len(tmpdir.listdir()) == 1
    X = np.load(DATASET.savefile.replace('.sav', '.npy'))
    assert (X == DATASET.X).all()

def test_TVTSpectra_init():
    assert TVT.savefile == 'tvt.spectra.sav'
    with pytest.raises(ValueError):
        tvt = TVTSpectra(DF.drop(['dm15'], axis = 'columns'))
    with pytest.raises(ValueError):
        tvt = TVTSpectra(DF, prep = 4)
    with pytest.raises(ValueError):
        tvt = TVTSpectra(DF, aug_num = 2)
    assert 'class' in TVT.spectra.columns
    assert (TVT.spectra_out['class'] == 0.).all()
    assert (TVT.spectra_in['class'] == 1.).all()
    assert len(TVT.spectra_out) + len(TVT.spectra_in) == len(TVT.spectra)
    tvt = TVTSpectra(DF, prep = 2)
    assert ('phase' in tvt.Ycol) and ('dm15' in tvt.Ycol)

def test_TVTSpectra_split():
    assert 'purpose' not in TVT.spectra.columns
    TVT.split()
    assert 'purpose' in TVT.spectra.columns
    purpose = TVT.spectra['purpose']
    TVT.spectra.drop(['purpose'], axis = 1, inplace = True)
    TVT.split() # check consistency
    assert (purpose.values == TVT.spectra['purpose'].values).all()
    TVT2 = TVTSpectra(DF, prep = 2)
    TVT2.split()
    for purpose in ['train', 'val', 'test']: # ensure prep 1/2 consistency
        assert (TVT.spectra.loc[(TVT.spectra['purpose'] == purpose) & \
                                (TVT.spectra['class'] == 1.), 'filename'].values\
                == TVT2.spectra.loc[TVT2.spectra['purpose'] == purpose, \
                                    'filename'].values).all()
    TVT.spectra.drop(['purpose'], axis = 1, inplace = True)
    TVT.split(method = 'randomized')
    vc = TVT.spectra['purpose'].value_counts('normalize')
    assert len(vc) == 3
    assert TVT.test_frac - 0.05 < vc.loc['test'] < TVT.test_frac + 0.05
    assert TVT.val_frac - 0.05 < vc.loc['val'] < TVT.val_frac + 0.05

def test_TVTSpectra_process():
    TVT.spectra = TVT.spectra.drop(['purpose'], axis = 1)
    assert 'purpose' not in TVT.spectra.columns
    assert not hasattr(TVT, 'trainX')
    TVT.process(status = False)
    for purpose in ['train', 'val', 'test']:
        assert hasattr(TVT, '{}X'.format(purpose))
        assert hasattr(TVT, '{}Y'.format(purpose))

def test_TVTSpectra_augprocess():
    TVT.aug_num = 110 # shorten execution time for testing
    delattr(TVT, 'testX')
    TVT.augprocess()
    assert hasattr(TVT, 'testX')
    assert hasattr(TVT, 'augX')
    assert hasattr(TVT, 'augY')
    assert len(TVT.augY) + len(TVT.trainY) == TVT.aug_num
    tvt = TVTSpectra(DF, prep = 2, aug_num = 110)
    tvt.augprocess(status = False)
    assert hasattr(tvt, 'testX')
    assert hasattr(tvt, 'augX')
    assert hasattr(tvt, 'augY')
    assert len(tvt.augY) + len(tvt.trainY) == tvt.aug_num

def test_TVTSpectra_to_npy(tmpdir):
    TVT.savefile = str(tmpdir.join('tmp.sav'))
    TVT.to_npy()
    assert len(tmpdir.listdir()) == 8
    for purpose in ['train', 'val', 'test', 'aug']:
        X = np.load(TVT.savefile.replace('.sav', '.{}.X.npy'.format(purpose)))
        assert (X == getattr(TVT, '{}X'.format(purpose))).all()
        Y = np.load(TVT.savefile.replace('.sav', '.{}.Y.npy'.format(purpose)))
        assert (Y == getattr(TVT, '{}Y'.format(purpose))).all()
