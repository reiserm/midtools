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

from .corrections import commonmode_series, commonmode_frame

import pdb

def azimuthal_integration(run, method='average', partition="upex",
    quad_pos=None, verbose=False, last=None, npulses=None, first_cell=1,
    mask=None, to_counts=False, apply_internal_mask=False, setup=None,
    client=None, geom=None, savname=None, adu_per_photon=65, **kwargs):
    """Calculate the azimuthally integrated intensity of a run using dask.

    Args:
        run (DataCollection): the run objection, e.g., created by RunDirectory.

        method (str, optional): how to integrate. Defaults to average. Use
            average to average all trains and pulses. Use single to calculate
            the azimuthally integrated intensity per pulse.

        partition (str, optional): Maxwell partition. Defaults upex. upex for
            users, exfel for  XFEL employees.

        quad_pos (list, optional): list of the four quadrant positions. If not
            provided, the latest configuration will be used.

        verbose (bool, optional): whether to print output. Defaults to True.

        last (int, optional): last train index. Defaults None. Set to small
            number for testing.

        mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0. If
            None, no pixel is masked.

        to_count (bool, optional): Convert the adus to photon counts based on
            thresholding and the value of adu_per_photon.

        apply_internal_mask (bool, optional): Read and apply the mask calculated
            from the calibration pipeline. Defaults to True.

        setup (Xana.Setup, optional): Xana setup. Defaults to None.

        geom (geometry, optional):

        savname (str, optional): Prefix of filename under which the results are
            saved automatically. Defaults to azimuthal_integration. An
            incrementing number is added not to overwrite previous results.

        adu_per_photon ((int,float), optional): number of ADUs per photon.

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

    def integrate_azimuthally(data, returnq=True):
        # Check data type
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, xarray.DataArray):
            data = np.array(data.data)
        else:
            raise(ValueError(f"Data type {type(data)} not understood."))

        # do azimuthal integration
        wnan = np.isnan(data)
        q, I = ai.integrate1d(data.reshape(8192,128),
                              500,
                              mask=~(mask.reshape(8192,128).astype('bool'))
                                    | wnan.reshape(8192,128),
                              unit='q_nm^-1', dummy=np.nan)
        if returnq:
            return q, I
        else:
            return I

    t_start = time()

    # get the azimuthal integrator
    ai = copy.copy(setup.ai)

    agp = AGIPD1M(run, min_modules=16)
    arr = agp.get_dask_array('image.data')
    print("Got dask array", flush=True)

    # coords are module, dim_0, dim_1, trainId, pulseId after unstack
    arr = arr.unstack()

    # take maximum 200 trains for the simple average
    if method == 'average':
        last = min(200, last)

    # select pulses and skip the first one
    arr = arr[..., :last, first_cell:npulses+first_cell]
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

    arr = arr.stack(train_pulse=('trainId', 'pulseId'),
                    pixels=('module', 'dim_0', 'dim_1'))
    arr = arr.transpose('train_pulse', 'pixels')
    print(arr)

    # apply common mode correction
    dim = arr.get_axis_num("pixels")
    arr = da.apply_along_axis(commonmode_frame, dim, arr.data)

    print("Start computation", flush=True)
    if method == 'average':
        # store some single frames
        last_train_pulse = min(npulses*100, arr.shape[0])
        frames = np.array(arr[:last_train_pulse:npulses*10])
        frames = frames.reshape(-1, 16, 512, 128)

        arr = arr.chunk({'train_pulse': 8, 'pixels': 16*512*128})
        arr = client.persist(arr.mean('train_pulse', skipna=True))
        progress(arr)

        arr = arr.reshape(-1, 16, 512, 128)

        # aziumthal integration
        q, I = integrate_azimuthally(arr)
        img2d = geom.position_modules_fast(arr.data)[0]

        savdict = {"soq":(q,I), "avr2d":img2d, "avr":np.array(arr),
                    "frames": frames}
        del arr, frames

        if savname is None:
            return savdict
        else:
            savname = f'./{savname}_Iq_{int(time())}.pkl'
            pickle.dump(savdict, open(savname, 'wb'))

    elif method == 'single':
        arr = arr.stack(pixels=('module','dim_0','dim_1'))
        arr = arr.chunk({'train_pulse': 128, 'pixels': 128*512})
        arr = arr.transpose('train_pulse', 'pixels')

        dim = arr.get_axis_num("pixels")
        q = integrate_azimuthally(arr[0])[0]
        arr = da.apply_along_axis(integrate_azimuthally, dim, \
            arr.data, dtype='float32', shape=(500,), returnq=False)

        arr = client.persist(arr)
        progress(arr)

        savdict = {"q(nm-1)":q, "soq-pr":np.array(arr)}
        del arr

        if savname is None:
            return savdict
        else:
            savname = f'./{savname}_Iqpr_{int(time())}.pkl'
            pickle.dump(savdict, open(savname, 'wb'))
    else:
        raise(ValueError(f"Method {method} not understood. \
                        Choose between 'average' and 'single'."))

    elapsed_time = time() - t_start
    if local_client:
        cluster.close()
        client.close()
    if verbose:
        print(f"Finished: elapsed time: {elapsed_time/60:.2f}min")
        print(f"Filename is {savname}")

    return
