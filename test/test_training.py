# pytest --cov-report term-missing --cov=deepSNIaID test/

# imports -- standard
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import pytest

# imports -- custom
from deepSNIaID import utils
from deepSNIaID.training import Train, Sweep
from deepSNIaID.architecture import DropoutCNN

# globals for testing
X1 = np.ones((10, 1, 100))
Y1 = np.ones((10, 1))

def test_Train_init():
    trainer = Train(X1, Y1, X1, Y1)
    assert trainer.mcnum == 75 # default
    assert type(trainer.network) is DropoutCNN
    assert trainer.train_loader.batch_size == 16 # default
    assert trainer.valX.size()[0] == X1.shape[0]
    assert type(trainer.optimizer) is torch.optim.Adam
    assert trainer.optimizer.defaults['lr'] == 1e-3 # default
    assert trainer.optimizer.defaults['weight_decay'] == 1e-4 # default
    assert trainer.regression
    assert type(trainer.loss) is nn.MSELoss
    del trainer
    trainer = Train(X1, Y1, X1, Y1, testX = X1, testY = Y1,
                    kernel = 3, fc_size = 16, lr_decay_steps = None,
                    regression = False, verbose = False)
    assert not trainer.regression
    assert isinstance(trainer.scheduler, utils.VoidLRScheduler)
    assert type(trainer.loss) is nn.BCELoss
    assert trainer.network.conv1.convolution.kernel_size == (3,)
    assert trainer.network.fc.linear.out_features == 16

def test_Train_traintest_epoch():
    Y1[0,0] = 0. # must have both classes present
    trainer = Train(X1, Y1, X1, Y1, regression = False)
    init_w = trainer.network.fc.linear.weight.data.clone()
    init_b = trainer.network.fc.linear.bias.data.clone()
    t = trainer.train_epoch()
    assert len(t.keys()) == 2
    assert type(t['loss']) in [float, np.float32, np.float64]
    assert type(t['lr-current']) in [float, np.float32, np.float64]
    trained_w = trainer.network.fc.linear.weight.data.clone()
    trained_b = trainer.network.fc.linear.bias.data.clone()
    assert (trained_w != init_w).any()
    assert (trained_b != init_b).any()
    v = trainer.test_epoch(trainer.valX, trainer.valY)
    assert len(v.keys()) == 3
    assert (trainer.network.fc.linear.weight.data == trained_w).all()
    assert (trainer.network.fc.linear.bias.data == trained_b).all()

def test_Train_train(tmpdir):
    trainer = Train(X1, Y1, X1, Y1, testX = X1, testY = Y1,
                    epochs = 3, savedir = str(tmpdir))
    trainer.train()
    assert len(tmpdir.listdir()) == 2 # 2 from net
    csv = [str(f) for f in tmpdir.listdir() if 'csv' in str(f)][0]
    csv = pd.read_csv(csv)
    assert csv.shape[0] == 3

def test_Sweep_init():
    if 'WANDB_API_KEY' in os.environ:
        os.environ.pop('WANDB_API_KEY')
    with pytest.raises(EnvironmentError):
        sweep = Sweep(X1, Y1, X1, Y1, 'null', 'null', [5, 7], [32, 64],
                      [64, 128], [64, 128], [64, 128], [0.1, 0.2], [64, 128],
                      [5e-4, 1e-3], [1e-4, 1e-3])
    os.environ['WANDB_API_KEY'] = 'null'
    with pytest.raises(ValueError):
        sweep = Sweep(X1, Y1, X1, Y1, 'null', 'null', [5, 7], [32, 64],
                      [64, 128], [64, 128], [64, 128], [0.1, 0.2], [64, 128],
                      [5e-4, 1e-3], [1e-4, 1e-3], sweep_method = 'not allowed')
    sweep = Sweep(X1, Y1, X1, Y1, 'null', 'null', [5, 7], [32, 64],
                  [1, 2], [3, 4], [5, 6], [0.1, 0.2], [64, 128],
                  [5e-4, 1e-3], [1e-4, 1e-3])
    assert type(sweep.sweep_config) is dict
    assert len(sweep.sweep_config.keys()) == 2
    assert sweep.sweep_config['parameters']['fc_size']['values'] == [1, 2]
    assert sweep.sweep_config['parameters']['kernel']['values'] == [5, 7]
