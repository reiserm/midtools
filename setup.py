#!/usr/bin/env python
import setuptools
from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import os
from distutils.command.sdist import sdist

# load the description from the README.md file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    cmdclass={'sdist': sdist},
    name = 'midtools',
    version = '0.0.1',
    packages=setuptools.find_packages(),
    license = 'BSD-III',
    author = 'Mario Reiser',
    author_email = 'mario.mkel@gmail.com',
    url = 'https://github.com/reiserm/midtools',
    download_url = '',
    keywords = ['data analysis', 'XFEL', 'MID'],
    description="Analysis tools to process MID measurements (runs).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires ='>= 3.6',
    install_requires = [
        'numpy',
        'pandas',
        'pyfai',
        'h5py',
        'extra_data',
        'extra_geom',
        'dask[complete]',
        'Xana',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
  ],
)

