# imports --standard
import os
import numpy as np
import pandas as pd

# imports -- custom
from deepSIP.architecture import DropoutCNN

__all__ = ['models', 'model_path']

model_path = os.path.join(os.path.dirname(__file__))

columns = ['kernel', 'filters', 'fc_size', 'drop_rate', 'Ymin', 'Ymax']

network_params = [[25, 8, 16, 0.01, 0., 1.],
                  [25, 8, 16, 0.01, -10, 18],
                  [35, 8, 128, 0.01, 0.85, 1.55]]
models = pd.DataFrame(data = network_params, columns = columns,
                      index = ['Domain', 'Phase', 'dm15'])
