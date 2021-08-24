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


def average(
    calibrator, trainIds=None, max_trains=10, chunks=None, axis="train_pulse", **kwargs
):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        run (DataCollection): the run objection, e.g., created by RunDirectory.
    """

    axisl = []
    if "train" in axis:
        axisl.append("trainId")
    if "pulse" in axis:
        axisl.append("pulseId")

    arr = calibrator.data.copy(deep=False)

    if len(axisl) == 1:
        axis = axisl[0]
        arr = arr.sel(trainId=trainIds[:max_trains], method="nearest")
        arr = arr.transpose(axis, ..., "pixels")
    if len(axisl) == 2:
        arr = arr.sel(trainId=trainIds[:max_trains], method="nearest")
        arr = arr.stack(train_pulse=("trainId", "pulseId"))
        axis = "train_pulse"

    if chunks is None:
        chunks = {axis: 1}

    if calibrator.worker_corrections["asic_commonmode"]:
        arr = calibrator._asic_commonmode_xarray(arr)
        if len(axisl) == 1:
            arr = arr.unstack("train_pulse")
            arr = arr.stack(pixels=("module", "dim_0", "dim_1"))
            arr = arr.transpose(axis, ..., "pixels")

    if calibrator.worker_corrections["dropletize"]:
        arr = calibrator._dropletize(arr)

    print("Start computation", flush=True)
    arr = arr.chunk(chunks)

    average = arr.mean(axis, skipna=True, keepdims=True)
    average = average.unstack()
    average = average.transpose(axis, ...)

    variance = arr.var(axis, skipna=True, keepdims=True)
    variance = variance.unstack()
    variance = variance.transpose(axis, ...)

    print("Shape of averge data")
    print(average.shape, variance.shape)

    return {
        "average": np.asarray(average.compute()),
        "variance": np.asarray(variance.compute()),
    }
