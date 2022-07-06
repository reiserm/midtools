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
    integrate='1d',
    mask=None,
    setup=None,
    geom=None,
    chunks=None,
    savname=None,
    sample_detector=8,
    photon_energy=9,
    center=(512, 512),
    distortion_array=None,
    nqbins=300,
    nphibins=180,
    **kwargs,
):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        calibrator (Calibrator): the data broker.

        method (str, optional): how to integrate. Defaults to average. Use
            average to average all trains and pulses. Use single to calculate
            the azimuthally integrated intensity per pulse.

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
    def integrate_azimuthally(train_data, integrator=None, integrate='1d', returnq=True):

        s = (8192, 128)
        train_data = train_data.reshape(-1, *s)
        out_shape = [train_data.shape[0], nqbins]
        if integrate == '2d':
            out_shape.append(nphibins)

        print(out_shape)
        I_v = np.zeros(out_shape, dtype=np.float32)
        for ipulse in range(out_shape[0]):
            pulse_data = train_data[ipulse]
            pyfai_mask = np.isnan(pulse_data) | (pulse_data < 0) | ~mask.reshape(s)
            if integrate == '1d':
                phi = 0
                q, I_v[ipulse] = integrator.integrate1d(
                    pulse_data, nqbins, mask=pyfai_mask, unit="q_nm^-1", dummy=np.nan
                )
            elif integrate == '2d':
                Iphiq, q, phi = integrator.integrate2d(
                    pulse_data, nqbins, nphibins, mask=pyfai_mask, unit="q_nm^-1", dummy=np.nan
                )
                I_v[ipulse] = Iphiq.T
        if returnq:
            return q, phi, I_v
        else:
            return I_v.astype("float32")

    t_start = time()
    integrate = integrate.lower()

    # get the azimuthal integrator
    ai = copy.copy(setup.ai)

    arr = calibrator.data.copy(deep=False)
    npulses = np.unique(arr.pulseId.values).size

    print("Start computation", flush=True)
    if method == "single":
        q, phi, I_v = integrate_azimuthally(np.ones((16 * 512, 128)), integrator=ai, integrate=integrate)
        if integrate == '1d':
            output_core_dims = [['pulseId', 'qI']]
            output_sizes = {'qI': nqbins}
            coords = {"qI": q}
        elif integrate == '2d':
            output_core_dims = [['pulseId', 'qI', 'phi']]
            output_sizes = {'qI': nqbins, 'phi':nphibins}
            coords = {"qI": q, 'phi':phi}

        darr = xarray.apply_ufunc(
            integrate_azimuthally,
            arr,
            input_core_dims=[['pulseId', "pixels"]],
            exclude_dims=set(["pixels"]),
            output_core_dims=output_core_dims,
            dask_gufunc_kwargs={"output_sizes": output_sizes, "allow_rechunk": True},
            dask="parallelized",
            vectorize=True,
            kwargs=dict(returnq=False, integrator=ai, integrate=integrate),
            output_dtypes=(np.float32),
        )
        darr = darr.assign_coords(coords)
        darr = darr.persist()

        darr = darr.chunk({"trainId": 10, "pulseId": -1, "qI": nqbins})

        savdict = {"q(nm-1)": darr.qI, 'phi':phi, "azimuthal-intensity": darr}
        return savdict
    else:
        raise (
            ValueError(
                f"Method {method} not understood. \
                        Choose between 'average' and 'single'."
            )
        )
