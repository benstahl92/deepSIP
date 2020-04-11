# imports -- standard
import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
tqdm.pandas()
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import pySNID, if available
try:
    import pySNID as ps
    _SNID = True
except ImportError:
    _SNID = False

# imports -- custom
from deepSNIaID import utils

__all__ = ['Spectrum', 'EvaluationSpectra', 'TVTSpectra']

class Spectrum:
    '''
    container for individual spectra

    Parameters
    ----------
    filename : str
               name of text file to read spectrum from
    z : float
        redshift of SN
    obsframe : bool, optional
               indicates spectrum is in observer frame (and thus needs to be
               de-redshifted)

    Attributes
    ----------
    signal_window_angstroms : int
                              window size in angstroms for signal smoothing
    signal_smoothing_order : int
                             polynomial order for signal smoothing
    continuum_window_angstroms : int
                                 window size in angstroms for continuum
                                 smoothing
    continuum_smoothing_order : int
                                polynomial order for continuum smoothing
    lwave_bounds : tuple, list, or other iterable of length 2
                   lower and upper limit of logarithmic wavelength array
    lwave_n_bins : int
                   number of bins in logarithmic wavelength grid
    lwave : np.array
            logarithmic wavelength grid
    apodize_end_pct : float
                      percentage from each end of flux array to apodize
    aug_drop_frac : float
                    maximum percentage of each of flux array to drop during
                    augmentation
    wave : np.array
           rest frame wavelength grid
    flux : np.array
           fluxes on wavelength grid
    '''

    def __init__(self, filename, z, obsframe = True):

        # store arguments
        self.filename = filename
        self.z = z
        self.obsframe = obsframe

        # set static parameters
        self.signal_window_angstroms = 100
        self.signal_smoothing_order = 5
        self.continuum_window_angstroms = 3000
        self.continuum_smoothing_order = 3
        self.lwave_bounds = (3450, 7500)
        self.lwave_n_bins = 1024
        self.lwave = utils.log_wave(self.lwave_bounds, self.lwave_n_bins)
        self.apodize_end_pct = 0.05
        self.aug_drop_frac = 0.1

        # load spectrum and shift to rest frame if necessary
        self.wave, self.flux = utils.load_txt_spectrum(self.filename)
        if self.obsframe:
            self.wave = utils.deredshift(self.wave, self.z)

    def _process(self, wave, flux):
        '''internal method for processing'''

        # smooth spectrum
        angstroms_per_pix = np.abs(np.mean(wave[1:] - wave[:-1]))
        signal_window_pix = int(np.ceil( (self.signal_window_angstroms / \
                                          angstroms_per_pix) / 2) * 2 - 1)
        flux = utils.smooth(flux, signal_window_pix,
                            self.signal_smoothing_order)

        # get continuum using similar method
        continuum_window_pix = int(np.ceil( (self.continuum_window_angstroms / \
                                             angstroms_per_pix) / 2) * 2 - 1)
        if continuum_window_pix > len(flux):
            continuum_window_pix = int(np.ceil( (len(flux) / 1.4) / 2) * 2 - 1)
        continuum = utils.get_continuum(flux, continuum_window_pix,
                                        self.continuum_smoothing_order)

        # subtract continuum
        flux -= continuum

        # log bin
        wave, flux, valid_mask = utils.log_bin(wave, flux,
                                               self.lwave_bounds,
                                               self.lwave_n_bins)

        # scaling and apodization
        flux[valid_mask] = utils.normalize_flux(flux[valid_mask])
        flux[valid_mask] = utils.apodize(flux[valid_mask], self.apodize_end_pct)

        # offset and return
        flux += 0.5
        return flux

    def process(self):
        '''
        process spectrum with no augmentation

        complete pre-processing: smoothes spectrum, identifies and subtracts
        pseudo-continuum, does log binning, scales flux and apodizes edges

        Returns
        -------
        np.array
            fully pre-processed flux array
        '''
        return self._process(self.wave, self.flux)

    def augprocess(self, augz, sig_wA):
        '''
        process spectrum with augmentation

        perturbs wavelength by multiplicative (1 + augz),
        drops ends randomly up to a maximum of aug_drop_frac,
        modify signal_window_angstroms to sig_wA,
        and then perform pre-processing according to process method

        Returns
        -------
        np.array
            augmented and fully pre-processed flux array
        '''
        wave = utils.redshift(self.wave, augz)
        wave, flux = utils.drop_ends(wave, self.flux, self.aug_drop_frac)
        self.signal_window_angstroms = sig_wA
        return self._process(wave, flux)

    def plot(self, show = 'processed'):
        '''
        plot original or processed spectrum for quick inspection

        Parameters
        ----------
        show : str, optional
               type of plot to show ('processed' or 'original')
        '''
        if show not in ['processed', 'original']:
            raise ValueError('show must be either "processed" or "original"')
        fig, ax = plt.subplots(1, 1, figsize = (6, 3))
        ax.plot(self.wave if show == 'original' else self.lwave,
                self.flux if show == 'original' else self.process(),
                color = 'black')
        ax.set_xlabel('Wavelength (\u212B)')
        ax.set_ylabel('$f_\lambda$'if show == 'original' else 'scaled flux')
        ax.set_title('{} ({})'.format(self.filename, show))
        plt.show()

    def SNID(self, **kwargs):
        '''
        run SNID on spectrum using pySNID (if available)

        Parameters
        ----------
        **kwargs
            arbitrary keyword arguments for pySNID

        Returns
        -------
        pySNID outputs
        '''
        if not _SNID:
            print('method unavailable without pySNID installed')
            return
        return ps.pySNID(self.filename, self.z if self.obsframe else 0.,
                         **kwargs)

class EvaluationSpectra:
    '''
    class for preparing spectra for evaluation

    Parameters
    ----------
    spectra : pd.DataFrame
              spectra to prepare for evaluation; must have columns columns of
              [SN, filename, z] and optionally obsframe as bool
    savefile : str, optional
               name of save file

    Attributes
    ----------
    X : np.ndarray with dimensions (number of spectra, lwave_n_bins)
        processed spectra (attribute set by process method)
    SNID_results : pd.DataFrame
                   pySNID results for each spectrum
                   (attribute set by SNID method)
    '''

    def __init__(self, spectra, savefile = 'eval.spectra.sav'):

        if '.sav' in spectra: # preempt init if given a save file
            self.savefile = spectra
            self.load()
            return

        # store spectra dataframe after basic validity checking
        if type(spectra) is not pd.DataFrame:
            raise TypeError('spectra needs to be of type pd.DataFrame')
        if not pd.Series(['SN', 'filename', 'z']).isin(spectra.columns).all():
            raise ValueError('spectra must have columns of [SN, filename, z]')
        if 'obsframe' not in spectra.columns: # assume all are in obs frame
            spectra['obsframe'] = True
        self.spectra = spectra

        self.savefile = savefile

        # instantiate Spectrum instances
        self.spectra['Spectrum'] = self.spectra.apply( \
                                   lambda row: Spectrum(row['filename'],
                                                        row['z'],
                                                        row['obsframe']),
                                                        axis = 1 )

    def __len__(self):
        return len(self.spectra)

    def save(self):
        '''save current state to savefile'''
        with open(self.savefile, 'wb') as f:
            pkl.dump(self.__dict__, f)

    def load(self):
        '''load from savefile'''
        with open(self.savefile, 'rb') as f:
            tmp = pkl.load(f)
        self.__dict__.update(tmp)

    def _process(self, status):
        '''internal method for processing'''
        fn = lambda spec: pd.Series(spec.process())
        if status:
            X = self.spectra['Spectrum'].progress_apply(fn)
        else:
            X = self.spectra['Spectrum'].apply(fn)
        return X

    def process(self, status = True):
        '''
        process all loaded spectra with no augmentation

        Parameters
        ----------
        status : bool, optional
                 show status bars
        '''
        self.X = self._process(status).values

    def SNID(self, selection = 'all', status = True, **kwargs):
        '''
        run SNID on selected (via boolean array) spectra in dataset

        Parameters
        ----------
        selection : boolean array
                    spectra from data set to run SNID on via pySNID
        status : bool, optional
                 show status bars
        **kwargs
            arbitrary keyword arguments for pySNID
        '''
        if selection is 'all':
            selection = [True] * len(self)
        fn = lambda spec: pd.Series(spec.SNID(**kwargs))
        if status:
            r = self.spectra.loc[selection, 'Spectrum'].progress_apply(fn)
        else:
            r = self.spectra.loc[selection, 'Spectrum'].apply(fn)
        r.columns = ['type', 'bmatch', 'goodnum', 'subtype', 'z', 'z_err',
                     'age', 'age_err']
        self.SNID_results = r
        if not os.path.exists('snid_files'):
            os.makedirs('snid_files')
        os.system('mv *snid.output snid_files/')
        os.system('mv snid.param snid_files/')

    def SNID_to_csv(self):
        '''write SNID results to csv file'''
        self.SNID_results.to_csv(self.savefile.replace('.sav', '.SNID.csv'),
                                 index = False)

    def to_npy(self):
        '''write processed spectra to .npy file'''
        np.save(self.savefile.replace('.sav', '.npy'), self.X,
                allow_pickle = True)

class TVTSpectra(EvaluationSpectra):
    '''
    class for preparing Training, Validation, and Testing sets

    Parameters
    ----------
    spectra : pd.DataFrame
              spectra to prepare; must have columns columns of
              [SN, filename, z] and optionally obsframe as bool
    savefile : str, optional
               name of save file
    prep : int, optional
           preparation mode (1 for all spectra, 2 for domain-restricted subset)
    val_frac : float, optional
               fraction of full set to split for validation
    test_frac : float, optional
                fraction of full set to split for testing
    phase_bounds : tuple, list, or other iterable of length 2, optional
                   lower and upper phase limits of domain
    dm15_bounds : tuple, list, or other iterable of length 2, optional
                  lower and upper dm15 limits of domain

    Attributes
    ----------
    spectra_[out,in] : pd.DataFrame
                       subset of spectra that are [out,in] selected domain
    spectra_aug : pd.DataFrame
                  augmented spectra
    Ycol : list
           columns in spectra that correspond to labels
    [train,aug,val,test]X : np.ndarray
                            processed spectra (attribute set by [aug]process
                            method)
    [train,aug,val,test]Y : np.ndarray
                            targets (attribute set by [aug]process method)

    Other Parameters
    ----------------
    phase_binsize : int or float, optional
                    phase bin size for pseudo-stratified splitting
    dm15_binsize : int or float, optional
                   dm15 bin size for pseudo-stratified splitting
    aug_num : int, optional
              final size of training set after augmentation
    aug_z_bounds : tuple, list, or other iterable of length 2, optional
                   lower and upper limits of randomly selected redshifts for
                   augmented spectra
    aug_signal_window_angstroms : tuple, list, or other iterable of length 2
                                  lower and upper limits of randomly selected
                                  signal windows for augmented spectra
    random_state : int, optional
                   seed for random number generator
    '''

    def __init__(self, spectra, savefile = 'tvt.spectra.sav',
                 prep = 1, val_frac = 0.1, test_frac = 0.1,
                 phase_bounds = (-10, 18), dm15_bounds = (0.85, 1.55),
                 phase_binsize = 4, dm15_binsize = 0.1,
                 aug_num = 5000, aug_z_bounds = (-0.004, 0.004),
                 aug_signal_window_angstroms = (50, 150), random_state = 100):

        # invoke inherited init instructions
        EvaluationSpectra.__init__(self, spectra, savefile = savefile)

        # do extra validation on spectra columns
        if not pd.Series(['phase', 'dm15']).isin(self.spectra.columns).all():
            raise ValueError('spectra must have columns of [phase, dm15]')

        self.prep = int(prep)
        if self.prep not in [1, 2]:
            raise ValueError('prep should be 1 (class) or 2 (phase/dm15 regr)')
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.phase_bounds = phase_bounds
        self.dm15_bounds = dm15_bounds
        self.phase_binsize = phase_binsize
        self.dm15_binsize = dm15_binsize
        if aug_num < len(self):
            raise ValueError('aug_num must exceed number of unmodified samples')
        self.aug_num = aug_num
        self.aug_z_bounds = aug_z_bounds
        self.aug_signal_window_angstroms = aug_signal_window_angstroms
        self.random_state = random_state

        # select for specific model being prepped for
        sel = ((spectra['phase'] >= self.phase_bounds[0]) &
               (spectra['phase'] < self.phase_bounds[1]) &
               (spectra['dm15'] >= self.dm15_bounds[0]) &
               (spectra['dm15'] < self.dm15_bounds[1]))
        self.spectra['class'] = 0.
        self.spectra.loc[sel, 'class'] = 1.
        self.spectra_out = spectra.loc[~sel].reset_index()
        self.spectra_in = spectra.loc[sel].reset_index()
        self.Ycol = ['class', 'phase', 'e_phase', 'dm15', 'e_dm15']

    def _split_randomized(self, spectra):
        '''split into train/val/test subsets by randomly sampling'''
        spectra['purpose'] = 'train'
        spectra = spectra.sample(frac = 1, replace = False,
                                 random_state = self.random_state)
        idx = spectra.index
        spectra.loc[idx[:int(self.test_frac * len(spectra))],'purpose'] = 'test'
        spectra.loc[idx[-int(self.val_frac * len(spectra)):],'purpose'] = 'val'
        return spectra

    def _split_stratified(self, spectra):
        '''randomly split into train/val/test subsets that are stratified'''
        spectra['purpose'] = 'train'
        spectra = spectra.sample(frac = 1, replace = False,
                                 random_state = self.random_state)
        for pctr in np.arange(self.phase_bounds[0] + self.phase_binsize / 2,
                                 self.phase_bounds[1], self.phase_binsize):
            for dctr in np.arange(self.dm15_bounds[0] + self.dm15_binsize / 2,
                                     self.dm15_bounds[1], self.dm15_binsize):
                sel = ((spectra['phase'] >= pctr - self.phase_binsize / 2) &
                       (spectra['phase'] < pctr + self.phase_binsize / 2) &
                       (spectra['dm15'] >= dctr - self.dm15_binsize / 2) &
                       (spectra['dm15'] < dctr + self.dm15_binsize / 2))
                test_samples = int(np.ceil(self.test_frac * sel.sum()))
                val_samples = int(np.ceil(self.val_frac * sel.sum()))
                idx = sel.index[sel]
                spectra.loc[idx[:test_samples], 'purpose'] = 'test'
                spectra.loc[idx[-val_samples:], 'purpose'] = 'val'
        return spectra

    def split(self, force = False, method = 'stratified'):
        '''
        split spectra into train/val/test subsets

        two splitting methods are available:
            1. 'stratified' - split in-domain spectra in pseudo-stratified
                              fashion by randomly selected subsets from bins
            2. 'randomized' - random selection

        Parameters
        ----------
        force : bool, optional
                overwrite pre-existing splits if True
        method : str, optional
                 splitting method ('stratified' or 'randomized')
        '''
        if ('purpose' in self.spectra.columns) and (not force):
            print('training/validation/testing sets already assigned')
            print('use keyword "force = True" to override')
            return
        elif method == 'stratified':
            spectra_in = self._split_stratified(self.spectra_in.copy())
        else:
            spectra_in = self._split_randomized(self.spectra_in.copy())

        if self.prep == 1: # ensure same in-domain tvt allocation
            spectra_out = self._split_randomized(self.spectra_out.copy())
            self.spectra = pd.concat((spectra_in, spectra_out),
                                     ignore_index = True)
        else:
            self.spectra = spectra_in

    def process(self, status = True):
        '''
        process all loaded spectra with no augmentation

        Parameters
        ----------
        status : bool, optional
                 show status bars
        '''
        self.split() # make sure subsets are selected
        X = self._process(status)
        for purpose in ['train', 'val', 'test']:
            sel = self.spectra['purpose'] == purpose
            setattr(self, '{}X'.format(purpose), X.loc[sel].values)
            setattr(self, '{}Y'.format(purpose),
                    self.spectra.loc[sel, self.Ycol].values)

    def augprocess(self, status = True):
        '''
        process all loaded spectra with augmentation

        Parameters
        ----------
        status : bool, optional
                 show status bars
        '''

        if not hasattr(self, 'testX'):
            self.process(status = status)

        # select only training spectra (and split if needed)
        train = self.spectra.loc[self.spectra['purpose'] == 'train'].copy()

        # generate set to augment
        self.spectra_aug = train.sample(n = self.aug_num - len(train),
                                        replace = True,
                                        random_state = self.random_state)
        self.spectra_aug['aug_z'] = np.random.rand(len(self.spectra_aug)) * \
                                    ( self.aug_z_bounds[1] - \
                                      self.aug_z_bounds[0] ) + \
                                    self.aug_z_bounds[0]
        self.spectra_aug['sig_wA'] = np.random.rand(len(self.spectra_aug)) * \
                                    ( self.aug_signal_window_angstroms[1] - \
                                      self.aug_signal_window_angstroms[0] ) + \
                                    self.aug_signal_window_angstroms[0]

        # augment
        fn = lambda row: pd.Series(row['Spectrum'].augprocess(row['aug_z'],
                                                              row['sig_wA']))
        if status:
            self.augX = self.spectra_aug.progress_apply(fn, axis = 1).values
        else:
            self.augX = self.spectra_aug.apply(fn, axis = 1).values
        self.augY = self.spectra_aug[self.Ycol].values

    def to_npy(self):
        '''write processed spectra and targets to .npy files'''
        purposes = ['train', 'val', 'test']
        if hasattr(self, 'augX'):
            purposes.append('aug')
        for purpose in purposes:
            np.save(self.savefile.replace('.sav', '.{}.X.npy'.format(purpose)),
                    getattr(self, '{}X'.format(purpose)), allow_pickle = True)
            np.save(self.savefile.replace('.sav', '.{}.Y.npy'.format(purpose)),
                    getattr(self, '{}Y'.format(purpose)), allow_pickle = True)
