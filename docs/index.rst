.. deepSIP documentation master file, created by
   sphinx-quickstart on Thu Apr  9 14:22:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

deepSIP
==========

deepSIP (deep learning of Supernova Ia Parameters) is an open-source toolkit for measuring the phase and light-curve shape (parameterized by SNooPy's :math:`\Delta m_{15}`) of a Type Ia Supernova (SN Ia) from an optical spectrum. The primary contents of the package are a set of three trained Convolutional Neural Networks (CNNs) for the aforementioned purposes, but tools for preprocessing spectra, modifying the neural architecture, training models, and sweeping through hyperparameters are also included.

The entire code base is available on GitHub: https://github.com/benstahl92/deepSIP

If you use deepSIP in your research, please cite the following paper:

Installation
^^^^^^^^^^^^

First, you'll need to clone deepSIP and enter its directory.

.. code-block:: bash

   git clone https://github.com/benstahl92/deepSIP.git
   cd deepSIP

(optional) It is recommended that you use a virtual environment for deepSIP and its dependencies.

.. code-block:: bash

   python -m venv dsenv # create virtual environment (one time only)
   source dsenv/bin/active # each time you need to activate the environment
   deactivate # if/when you need to leave the environment

Install dependencies and deepSIP.

.. code-block:: bash

   pip install -r requirements.txt
   pip install .

Standard Usage
^^^^^^^^^^^^^^

.. code-block:: python

   from deepSIP import deepSIP
   ds = deepSIP()
   # spectra is a pd.DataFrame with columns including ['SN', 'filename', 'z']
   predictions = ds.predict(spectra, threshold = 0.5, status = True)

Placeholder for example section on GitHub

Full Documentation
^^^^^^^^^^^^^^^^^^

Beyond standard usage, there may be occasions to use the underlying toolkit provided by deepSIP. We therefore provide full documentation of its capabilities below.

.. toctree::
   :maxdepth: 2

   deepSIP.model.rst
   deepSIP.preprocessing.rst
   deepSIP.dataset.rst
   deepSIP.architecture.rst
   deepSIP.training.rst
   deepSIP.utils.rst

Contributing
^^^^^^^^^^^^

We welcome community involvement in the form of bug fixes, architecture improvements and new trained models, additional high-quality data (spectra and photometry), and expanded functionality. To those wishing to participate, we ask that you fork the `deepSIP repository <https://github.com/benstahl92/deepSIP>`_ and issue a pull request with your changes (along with unit tests and a description of your contribution).

deepSIP is developed and maintained by Benjamin Stahl under the supervision of Prof. Alex Filippenko at UC Berkeley.
