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

def statistics(calibrator, last=None, mask=None, setup=None, geom=None,
        max_trains=10_000, chunks=None, hist_range=(-200, 800), savname=None,
        **kwargs):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        calibrator (Calibrator): the data broker.

        last (int, optional): last train index. Defaults None. Set to small
            number for testing.

        mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0. If
            None, no pixel is masked.

        setup (Xana.Setup, optional): Xana setup. Defaults to None.

        geom (geometry, optional): AGIPD1M geometry.

        hist_range(tuple, optional): tuple with lower and upper histogram
            boundaries. Defaults to (-200, 800)

        savname (str, optional): Prefix of filename under which the results are
            saved automatically. Defaults to azimuthal_integration. An
            incrementing number is added not to overwrite previous results.

    Returns:
        dict: Dictionary containing the results depending on the method it
            contains the keys:
              * 'histogram': counts
              * 'centers': bin centers
    """

    def statistics_worker(data, return_centers=False):
        # Check data type
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, xarray.DataArray):
            data = np.array(data.data)
        else:
            raise(ValueError(f"Data type {type(data)} not understood."))

        ind = np.isfinite(data)
        counts, edges = np.histogram(data[ind], bins=500,
                range=hist_range)

        if return_centers:
            centers = edges[:-1] + (edges[1] - edges[0]) / 2
            return centers, counts
        else:
            return counts

    t_start = time()

    if chunks is None:
        chunks = {'train_pulse': 128, 'pixels': 128*512}

    arr = calibrator._get_data(last_train=last)
    npulses = np.unique(arr.pulseId.values).size

    print("Start computation", flush=True)
    arr = arr.chunk(chunks)
    pdb.set_trace()
    centers = statistics_worker(arr[0], return_centers=True)[0]

    dim = arr.get_axis_num("pixels")
    arr = da.apply_along_axis(statistics_worker, dim, \
        arr.data, dtype='float32', shape=(500,))

    arr = arr.persist()
    progress(arr)

    savdict = {'centers': centers, 'counts': arr.compute()}
    del arr
    return savdict
