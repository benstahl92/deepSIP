.. deepSNIaID documentation master file, created by
   sphinx-quickstart on Thu Apr  9 14:22:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

deepSNIaID
==========

deepSNIaID (deep learning for Supernova Ia IDentification) is an open-source toolkit for measuring the phase and light-curve shape (parameterized by SNooPy's :math:`\Delta m_{15}`) of a Type Ia Supernova (SN Ia) from an optical spectrum. The primary contents of the package are a set of three trained Convolutional Neural Networks (CNNs) for the aforementioned purposes, but tools for preprocessing spectra, modifying the neural architecture, training models, and sweeping through hyperparameters are also included.

The entire code base is available on GitHub: https://github.com/benstahl92/deepSNIaID

If you use deepSNIaID in your research, please cite the following paper:

Installation
^^^^^^^^^^^^

First, you'll need to clone deepSNIaID and enter its directory.

.. code-block:: bash

   git clone https://github.com/benstahl92/deepSNIaID.git
   cd deepSNIaID

(optional) It is recommended that you use a virtual environment for deepSNIaID and its dependencies.

.. code-block:: bash

   python -m venv dsnenv # create virtual environment (one time only)
   source dsnenv/bin/active # each time you need to activate the environment
   deactivate # if/when you need to leave the environment

Install dependencies and deepSNIaID.

.. code-block:: bash

   pip install -r requirements.txt
   pip install .

Standard Usage
^^^^^^^^^^^^^^

.. code-block:: python

   from deepSNIaID import deepSNIaID
   dsn = deepSNIaID()
   # spectra is a pd.DataFrame with columns including ['SN', 'filename', 'z']
   predictions = dsn.predict(spectra, threshold = 0.5, status = True)

Placeholder for example section on GitHub

Full Documentation
^^^^^^^^^^^^^^^^^^

Beyond standard usage, there may be occasions to use the underlying toolkit provided by deepSNIaID. We therefore provide full documentation of its capabilities below.

.. toctree::
   :maxdepth: 2

   deepSNIaID.model.rst
   deepSNIaID.preprocessing.rst
   deepSNIaID.dataset.rst
   deepSNIaID.architecture.rst
   deepSNIaID.training.rst
   deepSNIaID.utils.rst

Contributing
^^^^^^^^^^^^

We welcome community involvement in the form of bug fixes, architecture improvements and new trained models, additional high-quality data (spectra and photometry), and expanded functionality. To those wishing to participate, we ask that you fork the `deepSNIaID repository <https://github.com/benstahl92/deepSNIaID>`_ and issue a pull request with your changes (along with unit tests and a description of your contribution).

deepSNIaID is developed and maintained by Benjamin Stahl under the supervision of Prof. Alex Filippenko at UC Berkeley.
