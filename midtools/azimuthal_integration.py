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

        super().__init__(pixel1=2e-4, pixel2=2e-4, max_shape=(8192,128),)
        self.shape = self.dim = (8192,128)
        self.aliases = ['Agipd1m']
        self.IS_CONTIGUOUS = False
        self.mask = np.zeros(self.shape)

def azimuthal_integration(calibrator, method='average', last=None,
    mask=None, setup=None, geom=None, max_trains=10_000, chunks=None,
    savname=None, sample_detector=8, photon_energy=9, center=(512, 512),
    distortion_array=None, **kwargs):
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

    worker_corrections = ('asic_commonmode', 'dropletize')

    @wf._xarray2numpy
    @wf._calibrate_worker(calibrator, worker_corrections)
    def integrate_azimuthally(data, returnq=True):

        # detector = Agipd1m()
        # detector.set_pixel_corners(distortion_array)
        # ai = AzimuthalIntegrator(detector=detector, dist=sample_detector)
        # ai.setFit2D(sample_detector * 1000, center[0], center[1])
        # ai.wavelength = 12.39 / photon_energy * 1e-10
        data = data.reshape(16, 512, 128)
        data[(data<0)|(data>6)] = np.nan
        ai = AzimuthalIntegrator(
            dist=sample_detector,
            pixel1=geom.pixel_size,
            pixel2=geom.pixel_size,
            poni1=center[1] * geom.pixel_size,
            poni2=center[0] * geom.pixel_size,
        )

        ai.wavelength = 12.39 / photon_energy * 1e-10
        data_2d = geom.position_modules_fast(data)[0]
        wnan = np.isnan(data_2d)
        q, I = ai.integrate1d(data_2d, 500, mask=wnan, unit='q_nm^-1', dummy=np.nan)
        if returnq:
            return q, I
        else:
            return I.astype('float32')

    t_start = time()

    if chunks is None:
        chunks = {'average': {'train_pulse': 8, 'pixels': 16*512*128},
                  'single': {'train_pulse': 32, 'pixels': 16*128*512}}

    # take maximum 200 trains for the simple average
    # skip trains to get max_trains trains
    if method == 'average':
        last = min(200, last)
        train_step = 1
    elif method == 'single':
        train_step = (last // max_trains) + 1

    arr = calibrator.data.copy(deep=False)
    npulses = np.unique(arr.pulseId.values).size

    print("Start computation", flush=True)
    if method == 'average':
        # store some single frames
        last_train_pulse = min(npulses*100, arr.shape[0])
        frames = arr[:last_train_pulse:npulses*10].values
        frames = frames.reshape(-1, 16, 512, 128)

        arr = arr.chunk(chunks['average'])
        arr = arr.mean('train_pulse', skipna=True).persist()
        progress(arr)
        arr = arr.values.reshape(16, 512, 128)

        # aziumthal integration
        q, I = integrate_azimuthally(arr)
        img2d = geom.position_modules_fast(arr)[0]

        savdict = {"soq":(q,I), "avr2d":img2d, "avr":arr, "frames": frames}
        del arr, frames

        if savname is None:
            return savdict
        else:
            savname = f'./{savname}_Iq_{int(time())}.pkl'
            pickle.dump(savdict, open(savname, 'wb'))

    elif method == 'single':
        arr = arr.chunk(chunks['single'])

        dim = arr.get_axis_num("pixels")
        q = integrate_azimuthally(np.ones(arr[0].shape))[0]
        arr = da.apply_along_axis(integrate_azimuthally, dim,
            arr.data, dtype='float32', shape=(500,), returnq=False)

        res = arr.persist()
        del arr
        progress(res)

        savdict = {"q(nm-1)":q, "soq-pr":res}
        return savdict
    else:
        raise(ValueError(f"Method {method} not understood. \
                        Choose between 'average' and 'single'."))

