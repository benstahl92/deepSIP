# imports -- standard
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torchsummary

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

# imports -- custom
from deepSNIaID import utils
from deepSNIaID.architecture import DropoutCNN
from deepSNIaID.dataset import NumpyXDataset, NumpyXYDataset

__all__ = ['Train', 'Sweep']

class Train:
    '''
    network training

    Parameters
    ----------
    train[X,Y] : np.ndarray
                 training [inputs,outputs]
    val[X,Y] : np.ndarray
               validation [inputs,outputs]
    test[X,Y] : np.ndarray, optional
                testing [inputs,outputs]
    Ylim : tuple, list, or other iterable of length 2, optional
           lower and upper limits for for utils.LinearScaler on outputs (Y)
    seed : int, optional
           seed for random number generator
    threshold : float, optional
                minimum threshold for 'in' classification by Domain model
    regression : bool, optional
                 toggle for regression (determines scalers used)
    mcnum : int, optional
            number of stochastic forward passes to perform
    kernel : odd int, optional
             convolutional kernel size
    filters : int, optional
              number of filters in first convolution layer
    fc_size : int, optional
              number of neurons in fully connected layer
    drop_rate : float, optional
                dropout probability
    epochs : int, optional
             number of training epochs
    lr : float, optional
         initial learning rate
    batch_size : int, optional
                 batch size for training
    weight_decay : float, optional
                   weight decay for training
    verbose : bool, optional
              show network summary and status bars
    save : bool, optional
           flag for saving training history and trained model
    savedir : str, optional
              directory for save files

    Attributes
    ----------
    device : torch.device
             device type being used (GPU if available, else CPU)
    network : DropoutCNN
              network to train (may be wrapped in DataParallel if on GPU)
    Yscaler : VoidScaler or LinearScaler
              scaler for Y labels
    optimizer : torch optimizer (Adam)
                optimizer for training
    scheduler : VoidLRScheduler or MultiStepLR
                learning rate scheduler
    loss : torch loss
           loss for training

    Other Parameters
    ----------------
    early_stop : length-1 array_like, optional
                 early stopping threshold on validation RMSE for regression mode
    lr_decay_steps : array_like, optional
                     epochs at which to decay learning rate by factor 10
    wandb : wandb instance, optional
            wandb instance for run tracking
    '''

    def __init__(self, trainX, trainY, valX, valY, testX = None, testY = None,
                 Ylim = [0., 1.],
                 seed = 100, threshold = 0.5, regression = True, mcnum = 100,
                 kernel = 15, filters = 16, fc_size = 32, drop_rate = 0.1,
                 epochs = 75, early_stop = [0.],
                 lr_decay_steps = [45, 60, 70], lr = 1e-3,
                 batch_size = 16, weight_decay = 1e-4,
                 verbose = True,  wandb = None, save = True, savedir = './'):

        # store needed inputs
        self.seed = seed
        self.threshold = threshold
        self.regression = regression
        self.mcnum = mcnum
        self.epochs = epochs
        self.early_stop = early_stop
        self.verbose = verbose
        self.wandb = wandb
        self.save = save
        self.savedir = savedir

        # seed for reproducibility
        utils.reset_state(seed = self.seed)

        # instantiate network using GPU, if available
        network = DropoutCNN(trainX.shape[-1], kernel = kernel,
                             filters = filters, fc_size = fc_size,
                             drop_rate = drop_rate)
        network.apply(utils.init_weights)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Running on {} GPUs.'.format(torch.cuda.device_count()))
            torch.cuda.empty_cache()
            self.network = nn.DataParallel(network).to(self.device)
        else:
            self.device = torch.device('cpu')
            print('No GPU available. Training on CPU...')
            self.network = network.to(self.device)

        # setup scaler
        if self.regression:
            self.Yscaler = utils.LinearScaler(*Ylim)
        else: # classification, so no scaling needed
            self.Yscaler = utils.VoidScaler()
        trainY = self.Yscaler.fit_transform(trainY)

        # load and prepare data
        self.train_loader = DataLoader(NumpyXYDataset(trainX, trainY),
                                       batch_size = batch_size, shuffle = True)
        self.valX = NumpyXDataset(valX).X
        self.valY = valY
        if testX is not None:
            self.testX = NumpyXDataset(testX).X
        else:
            self.testX = testX
        self.testY = testY

        # instantiate optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = lr,
                                          weight_decay = weight_decay)

        # instantiate lr scheduler, if needed
        if hasattr(lr_decay_steps, '__len__') and (type(lr_decay_steps) != str):
            self.scheduler = MultiStepLR(self.optimizer, lr_decay_steps,
                                         gamma = 0.1)
        else:
            self.scheduler = utils.VoidLRScheduler()

        # instantiate training loss
        if self.regression:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

        # show model summary if requested
        if self.verbose:
            torchsummary.summary(self.network, (1, trainX.shape[-1]))

    def train_epoch(self):
        '''
        perform training steps for single epoch

        Returns
        -------
        dict
            training metrics for epoch (loss and lr-current)
        '''
        self.network.train()
        for i, (inputs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if self.regression:
                outputs = self.network(inputs.to(self.device))
            else:
                outputs = torch.sigmoid(self.network(inputs.to(self.device)))
            loss = self.loss(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()
        return {'loss': loss.item(),
                'lr-current': self.optimizer.param_groups[0]['lr']}

    def test_epoch(self, X, Y, label = ''):
        '''
        perform validation or testing steps for single epoch

        Parameters
        ----------
        X : torch.tensor
            inputs
        Y : torch.tensor
            outputs
        label : str, optional
                label to prepend output dict keys with (e.g. 'val' or 'test')

        Returns
        -------
        metrics : dict
                  validation or testing metrics for epoch
        '''
        with torch.no_grad():
            mu, sigma = utils.stochastic_predict(self.network, X,
                                                 mcnum = self.mcnum,
                                                 sigmoid = not self.regression,
                                                 seed = None, # don't reset net
                                                 scaler = self.Yscaler)
            if self.regression:
                metrics = utils.regression_metrics(Y, mu, sigma,
                                                   key_prepend = label)
            else:
                metrics = utils.classification_metrics(Y, mu,
                                                       self.threshold,
                                                       key_prepend = label)
        return metrics

    def train(self):
        '''train network'''

        utils.reset_state(seed = self.seed)

        # store information
        training_history = []
        if self.wandb is not None:
            self.wandb.watch(self.network)
            self.wandb.log({'n_params': utils.count_params(self.network)})

        # iterate through epochs
        epochs = range(self.epochs)
        if self.verbose:
            epochs = tqdm(epochs)
        for epoch in epochs:

            # update learning rate --- pytorch 1.0 placement
            self.scheduler.step(epoch)

            # train and validate
            train_dict = self.train_epoch()
            val_dict = self.test_epoch(self.valX, self.valY, label = 'val_')
            if self.testX is not None:
                test_dict = self.test_epoch(self.testX, self.testY,
                                            label = 'test_')
                val_dict = {**val_dict, **test_dict}

            # log results
            epoch_dict = {**train_dict, **val_dict, 'epoch': epoch}
            training_history.append(epoch_dict)
            if self.wandb is not None:
                self.wandb.log(epoch_dict)

            # check early stopping criteria at 20 epochs
            if (epoch == 20) and (self.regression):
                quit = False
                for i, tol in enumerate(self.early_stop):
                    if epoch_dict['val_rmse_{}'.format(i + 1)] > tol:
                        quit = True
                if quit:
                    break

        if self.save:
            time = str(datetime.now())[:10]
            self.root = os.path.join(self.savedir, 'network.{}'.format(time))
            pd.DataFrame(training_history).to_csv(self.root + '.csv',
                                                  index = False)
            utils.savenet(self.network, self.root + '.pth')

class Sweep:
    '''
    sweep (search) through hyperparameters using wandb

    Parameters
    ----------
    train[X,Y] : np.ndarray
                 training [inputs,outputs]
    val[X,Y] : np.ndarray
               validation [inputs,outputs]
    entity : str
             wandb entity
    project : str
              wandb project
    kernels : array_like
              convolutional kernel sizes to sweep over
    filters : array_like
              numbers of filters in first convolution layer to sweep over
    fc_sizes : array_like
               numbers of neurons in fully connected layer to sweep over
    drop_rates : array_like
                 dropout probabilities to sweep over
    batch_sizes : array_like
                  batch sizes to sweep over
    lrs : array_like
          initial learning rates to sweep over
    weight_decays : array_like
                    weight decays to sweep over
    seed : int, optional
           seed for random number generator
    regression : bool, optional
                 toggle for regression (determines scalers used)
    mcnum : int, optional
            number of stochastic forward passes to perform
    epochs : int, optional
             number of training epochs
    Ylim : tuple, list, or other iterable of length 2, optional
           lower and upper limits for for utils.LinearScaler on outputs (Y)
    sweep_method : str, optional
                   method for sweep ('random' or 'grid')
    test[X,Y] : np.ndarray, optional
                testing [inputs,outputs]

    Attributes
    ----------
    sweep_config : dict
                   wandb sweep configurations

    Other Parameters
    ----------------
    early_stop : length-1 array_like, optional
                 early stopping threshold on validation RMSE for regression mode
    '''

    def __init__(self, trainX, trainY, valX, valY, entity, project,
                 kernels, filters, fc_sizes,
                 drop_rates, batch_sizes, lrs, weight_decays, seed = 100,
                 regression = True, mcnum = 100, epochs = 75, early_stop = [0.],
                 Ylim = [0., 1.], sweep_method = 'random',
                 testX = None, testY = None):

        if not _WANDB:
            print('cannot do sweeps without wandb installed')
            return

        # validate inputs
        if 'WANDB_API_KEY' not in os.environ:
            raise EnvironmentError('WANDB_API_KEY environment variable not set')
        if sweep_method not in ['random', 'grid']:
            raise ValueError('supported sweep methods are random or grid')

        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY
        self.entity = entity
        self.project = project
        self.seed = seed
        self.regression = regression
        self.mcnum = mcnum
        self.epochs = epochs
        self.early_stop = early_stop
        self.Ylim = Ylim

        # make sweep grid
        self.sweep_config = {'method': sweep_method,
                             'parameters': {
                                'kernel': {'values': kernels},
                                'filters': {'values': filters},
                                'fc_size': {'values': fc_sizes},
                                'drop_rate': {'values': drop_rates},
                                'batch_size': {'values': batch_sizes},
                                'lr': {'values': lrs},
                                'weight_decay': {'values': weight_decays}}
                             }

    def sweep(self, tags = [], saveroot = None):
        '''
        run sweep

        Parameters
        ----------
        tags : list, optional
               list of strings to add as tags to sweep runs
        saveroot : str, optional
                   root name to use for saving
        '''

        # setup root saving directoring
        if saveroot is not None:
            if not os.path.exists(saveroot):
                os.mkdir(saveroot)

        # define internal train function to wrap Train
        def train():
            # setup wandb
            config_defaults = {'kernel': 5, 'filters': 8, 'fc_size': 32,
                               'drop_rate': 0.1, 'batch_size': 16, 'lr': 1e-3,
                               'weight_decay': 1e-4}
            tags.append(datetime.today().strftime('%Y-%m-%d'))
            wandb.init(config = config_defaults, tags = tags)
            config = wandb.config

            # create run results directory
            save = False
            if saveroot is not None:
                runpath = os.path.join(saveroot, wandb.run.id)
                os.mkdir(runpath)
                save = True

            # instantiate trainer and run
            trainer = Train(self.trainX, self.trainY, self.valX, self.valY,
                            testX = self.testX, testY = self.testY,
                            Ylim = self.Ylim,
                            kernel = config.kernel, filters = config.filters,
                            drop_rate = config.drop_rate,
                            epochs = self.epochs, early_stop = self.early_stop,
                            fc_size = config.fc_size,
                            batch_size = config.batch_size, lr = config.lr,
                            weight_decay = config.weight_decay,
                            verbose = False,
                            mcnum = self.mcnum, regression = self.regression,
                            seed = self.seed, wandb = wandb,
                            save = save, savedir = runpath)
            trainer.train()

        # run sweep
        sweep_id = wandb.sweep(self.sweep_config, entity = self.entity,
                               project = self.project)
        wandb.agent(sweep_id, train)
