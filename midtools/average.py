# standard python packages
import numpy as np
import scipy.integrate as integrate
from time import time
import pickle
import xarray
import copy

# reading AGIPD data provided by XFEL
from extra_data import RunDirectory, stack_detector_data, open_run
from extra_geom import AGIPD_1MGeometry
from extra_data.components import AGIPD1M

# DASK
import dask.array as da
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar

import pdb


def average(calibrator, last_train=None, chunks=None, axis='train_pulse',
            **kwargs):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        run (DataCollection): the run objection, e.g., created by RunDirectory.
    """

    axisl = []
    if 'train' in axis:
        axisl.append('trainId')
    if 'pulse' in axis:
        axisl.append('pulseId')

    arr = calibrator.data.copy()
    npulses = np.unique(arr.pulseId.values).size

    if len(axisl) == 1:
        axis = axisl[0]
        arr = arr.unstack('train_pulse')
        arr = arr[..., arr.trainId < arr.trainId[last_train], :]
        arr = arr.transpose(axis, ..., 'pixels')
    if len(axisl) == 2:
        axis = 'train_pulse'
        arr = arr[:last_train * npulses]

    if chunks is None:
        chunks = {axis: 128}

    if calibrator.worker_corrections['asic_commonmode']:
        arr = calibrator._asic_commonmode_xarray(arr)
        if len(axisl) == 1:
            arr = arr.unstack('train_pulse')
            arr = arr.stack(pixels=('module', 'dim_0', 'dim_1'))
            arr = arr.transpose(axis, ..., 'pixels')

    if calibrator.worker_corrections['dropletize']:
        arr = calibrator._dropletize(arr)

    print("Start computation", flush=True)
    arr = arr.chunk(chunks)

    average = arr.mean(axis, skipna=True, keepdims=True).persist()
    progress(arr)
    average = np.squeeze(average.values.reshape(-1, 16, 512, 128))

    variance = arr.var(axis, skipna=True, keepdims=True).persist()
    progress(variance)
    variance = np.squeeze(variance.values.reshape(-1, 16, 512, 128))

    return {'average': average, 'variance': variance}
