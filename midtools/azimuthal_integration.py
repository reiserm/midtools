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

from . import worker_functions as wf
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector

import pdb


class Agipd1m(Detector):
    def __init__(self, *args, **kwargs):

        super().__init__(
            pixel1=2e-4,
            pixel2=2e-4,
            max_shape=(8192, 128),
        )
        self.shape = self.dim = (8192, 128)
        self.aliases = ["Agipd1m"]
        self.IS_CONTIGUOUS = False
        self.mask = np.zeros(self.shape)


def azimuthal_integration(
    calibrator,
    method="average",
    last=None,
    mask=None,
    setup=None,
    geom=None,
    max_trains=10_000,
    chunks=None,
    savname=None,
    sample_detector=8,
    photon_energy=9,
    center=(512, 512),
    distortion_array=None,
    **kwargs,
):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        calibrator (Calibrator): the data broker.

        method (str, optional): how to integrate. Defaults to average. Use
            average to average all trains and pulses. Use single to calculate
            the azimuthally integrated intensity per pulse.

        last (int, optional): last train index. Defaults None. Set to small
            number for testing.

        mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0. If
            None, no pixel is masked.

        setup (Xana.Setup, optional): Xana setup. Defaults to None.

        geom (geometry, optional): AGIPD1M geometry.

        savname (str, optional): Prefix of filename under which the results are
            saved automatically. Defaults to azimuthal_integration. An
            incrementing number is added not to overwrite previous results.

    Returns:
        dict: Dictionary containing the results depending on the method it
            contains the keys:
              * for average:
                * 'soq': the azimuthal intensity
                * 'avr': the average image (16,512,128)
                * 'img2d': the repositioned image in 2D
              * for single:
                * 'soq-pr': azimuthal intensity pulse resolved
                * 'q(nm-1)': the q-values in inverse nanometers
    """

    worker_corrections = ("asic_commonmode", "dropletize")

    @wf._xarray2numpy(dtype="int16")
    @wf._calibrate_worker(calibrator, worker_corrections)
    def integrate_azimuthally(train_data, integrator=None, returnq=True):

        s = (8192, 128)
        train_data = train_data.reshape(-1, *s)
        I_v = np.empty((train_data.shape[0], 500))
        for ip, pulse_data in enumerate(train_data):
            pyfai_mask = np.isnan(pulse_data) | (pulse_data < 0) | ~mask.reshape(s)
            q, I = integrator.integrate1d(
                pulse_data, 500, mask=pyfai_mask, unit="q_nm^-1", dummy=np.nan
            )
            I_v[ip] = I
        if returnq:
            return q, I_v
        else:
            return I_v.astype("float32")

    t_start = time()

    if chunks is None:
        chunks = {
            "single": {"trainId": 1, "train_data": -1},
        }

    # get the azimuthal integrator
    ai = copy.copy(setup.ai)

    arr = calibrator.data.copy(deep=False)
    arr = arr.unstack()
    arr = arr.stack(train_data=("pulseId", "module", "dim_0", "dim_1"))
    npulses = np.unique(arr.pulseId.values).size

    print("Start computation", flush=True)
    if method == "single":
        arr = arr.chunk(chunks["single"])
        dim = arr.get_axis_num("train_data")
        q = integrate_azimuthally(np.ones(arr[0].shape), integrator=ai)[0]
        res = da.apply_along_axis(
            integrate_azimuthally,
            dim,
            arr.data,
            dtype="float32",
            shape=(
                npulses,
                500,
            ),
            returnq=False,
            integrator=ai,
        )

        res = res.persist()
        del arr
        progress(res)
        res = res.reshape(-1, 500)

        savdict = {"q(nm-1)": q, "soq-pr": res}
        return savdict
    else:
        raise (
            ValueError(
                f"Method {method} not understood. \
                        Choose between 'average' and 'single'."
            )
        )
