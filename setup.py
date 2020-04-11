# imports -- standard
import ast
from setuptools import setup, find_packages

# Get version without importing module
with open('deepSNIaID/__init__.py') as f:
    mod = ast.parse(f.read())

setup(name='deepSNIaID',
      version=mod.body[-1].value.s,
      description='Deep Learning for Supernova Ia IDentification',
      url='https://github.com/benstahl92/deepSNIaID',
      author='Benjamin Stahl',
      author_email='benjamin_stahl@berkeley.edu',
      license='GPL-v3',
      packages=find_packages(),
      include_package_data=True,
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Intended Audience :: Science/Research']
      )
