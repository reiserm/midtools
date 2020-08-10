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

def average(run, last=None, npulses=None, first_cell=1, mask=None,
        to_counts=False, apply_internal_mask=False, client=None, geom=None,
        adu_per_photon=65, max_trains=10_000, **kwargs):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        run (DataCollection): the run objection, e.g., created by RunDirectory.
    """

    agp = AGIPD1M(run, min_modules=16)
    arr = agp.get_dask_array('image.data')
    print("Got dask array", flush=True)

    # coords are module, dim_0, dim_1, trainId, pulseId after unstack
    arr = arr.unstack()
    is_proc = True if len(arr.dims) == 5 else False
    print(is_proc, arr)

    # take maximum 200 trains for the simple average
    # skip trains to get max_trains trains
    last = min(200, last)
    train_step = 1

    # select pulses and skip the first one
    if is_proc:
        arr = arr[..., :last:train_step, first_cell:npulses+first_cell]
    else:
        arr = arr[:, 0, ..., :last:train_step, first_cell:npulses+first_cell]
    npulses = arr.shape[-1]

    if to_counts:
        arr.data = np.floor((arr.data + 0.5*adu_per_photon) / adu_per_photon)
        arr.data[arr.data<0] = 0
        # arr.data.astype('float32')

    if apply_internal_mask:
        mask_int = agp.get_dask_array('image.mask')
        mask_int = mask_int.unstack()
        mask_int = mask_int[..., :last, first_cell:npulses+first_cell]
        arr = arr.where((mask_int.data <= 0) | (mask_int.data>=8))
        # mask[(mask_int.data > 0) & (mask_int.data<8)] = 0

    print(arr)

    print("Start computation", flush=True)
    # store some single frames
    last_train_pulse = min(npulses*100, arr.shape[0])
    frames = np.array(arr[:last_train_pulse:npulses*10])
    frames = frames.reshape(-1, 16, 512, 128)

    arr = arr.chunk({'trainId': 1})
    arr = client.persist(arr.mean('trainId', skipna=True))
    progress(arr)

    return np.array(arr)
