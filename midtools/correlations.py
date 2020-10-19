import pdb
# standard python packages
import numpy as np
import scipy.integrate as integrate
import scipy.ndimage as ndimage
from time import time
import pickle
import xarray
import h5py
import copy

# main analysis software can be installed by: pip install Xana
from Xana import Xana
import Xana.Setup
from Xana.XpcsAna.pyxpcs3 import pyxpcs
from Xana.XpcsAna.xpcsmethods import ttc_to_g2

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


def ttc_to_g2_ufunc(ttc, dims, time=None):
    return  xarray.apply_ufunc(
                    ttc_to_g2,
                    ttc,
                    input_core_dims=[dims],
                    kwargs={'time': time},
                    dask='parallelized',
                    output_dtypes=[float],)


def convert_ttc(ttc):
    ttc = ttc.reshape(int(np.sqrt(ttc.size)), -1)
    g2 = ttc_to_g2(ttc)
    return g2[:,1]


def blur_gauss(U, sigma=2.0, truncate=4.0):

    V = U.copy()
    V[np.isnan(U)] = 0
    VV = ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0*U.copy() + 1
    W[np.isnan(U)] = 0
    WW = ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    return VV / WW


def correlate(calibrator, method='intra_train', last=None, qmap=None,
    mask=None, setup=None, q_range=None, save_ttc=False, h5filename=None,
    norm_xgm=False, chunks=None, **kwargs):
    """Calculate XPCS correlation functions of a run using dask.

    Args:
        calibrator (Calibrator): the data broker.

        method (str, optional): along which axis to correlate. Default is to
            calculate the correlation function per train.

        last (int, optional): last train index. Defaults None. Set to small
            number for testing.

        mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0.

        setup ((Xana.Setup), optional): Xana setupfile. Defaults to None.
            If str, include path in the filename. Otherwise provide Xana Setup
            instance.

        qrange (dict): Information on the q-bins. Should contain the keys:
            * q_first, q_last, nsteps
            or
            * q_first, q_last, qwidth

    Returns:
        dict: Dictionary containing the results depending on the method it
            contains the keys:
            * 'corf': the xpcs correlation function
    """

    worker_corrections = ('asic_commonmode', 'cell_commonmode', 'dropletize')

    @wf._xarray2numpy
    @wf._calibrate_worker(calibrator, worker_corrections)
    def calculate_correlation(data, return_='all', blur=True, **kwargs):

        data[(data<0)|(data>6)] = np.nan
        if blur:
            for i, image in enumerate(data):
                image[~mask] = np.nan
                data[i] = blur_gauss(image, sigma=3.0, truncate=4.0)

        data = data.reshape(-1, 8192, 128)
        wnan = np.isnan(np.sum(data, axis=0))
        xpcs_mask = mask_2d & ~wnan.reshape(8192,128)

        rois = [np.where((qmap_2d>qi) & (qmap_2d<qf) & xpcs_mask) for qi,qf in
            zip(qarr[:-1],qarr[1:])]

        # get only the q-bins in range
        qv = qarr[:len(rois)] + (qarr[1] - qarr[0])/2.

        out = pyxpcs(data, rois, mask=xpcs_mask, nprocs=1, verbose=False,
                **kwargs)

        if method == 'intra_train':
            corf = out['corf']
            t = corf[1:,0]
            corf = corf[1:,1:]
        elif method == 'intra_train_ttc':
            corf = np.dstack(out['twotime_corf'].values()).T
            t = out['twotime_xy']

        if return_ == 'all':
            return t, corf, qv
        elif return_ == 'corf':
            return corf.astype('float32')
        elif return_ == 'ttc':
            pass

    if chunks is None:
        chunks = {'intra_train': {'trainId': 1, 'train_data': 16*512*128},
                  'intra_train_ttc': {'trainId': 1, 'train_data': 16*512*128}}


    arr = calibrator.data.copy(deep=False)
    npulses = np.unique(arr.pulseId.values).size

    if norm_xgm:
        with h5py.File(h5filename, 'r') as f:
            xgm = f["pulse_resolved/xgm/energy"][:]
        xgm = xgm[:last, :npulses]
        xgm = xgm / xgm.mean(1)[:, None]
        print(f"Read XGM data from {h5filename} for normalization")
        arr = arr / xgm[None, None, None, ...]

    arr = arr.unstack()
    trainId = arr.trainId
    arr = arr.stack(train_data=('pulseId', 'module', 'dim_0', 'dim_1'))
    arr = arr.chunk(chunks[method])
    npulses = arr.shape[1] // (16*512*128)

    if 'nsteps' in q_range:
        qarr = np.linspace(q_range['q_first'],
                           q_range['q_last'],
                           q_range['nsteps'])
    elif 'qwidth' in q_range:
        qarr = np.arange(q_range['q_first'],
                         q_range['q_last'],
                         q_range['qwidth'])
    else:
        raise ValueError("Could not define q-values. "
                         "q_range not defined properly: "
                         f"{q_range}")

    qmap_2d = qmap.reshape(16*512,128)
    mask_2d = mask.reshape(16*512,128)

    # do the actual calculation
    if 'intra_train' in method:
        t, corf, qv = calculate_correlation(np.ones(arr[0].shape), return_='all',
                **kwargs)

        dim = arr.get_axis_num("train_data")
        out = da.apply_along_axis(calculate_correlation, dim, arr.data,
            dtype='float32', shape=corf.shape, return_='corf', **kwargs)

        if 'ttc' in method:
            dset = xarray.Dataset(
                    {
                        "ttc": (["trainId", "qv", "t1", "t2"], out),
                        },
                    coords={
                        "trainId": trainId,
                        "t1": t,
                        "t2": t,
                        "qv": qv,
                        },
                    )
            g2 = da.apply_along_axis(convert_ttc, 1,
                    dset["ttc"].stack(meas=("trainId", "qv"), data=("t1", "t2")),
                    dtype='float32', shape=(t.size,))
            g2 = g2.reshape(dset.trainId.size, qv.size, t.size).swapaxes(1, 2)
            dset = dset.assign_coords(t_cor=t[1:])
            dset = dset.assign(g2=(("trainId", "t_cor", "qv"), g2[:,1:]))
        else:
            dset = xarray.Dataset(
                    {
                        "g2": (["trainId", "t_cor", "qv"], out),
                        },
                    coords={
                        "trainId": trainId,
                        "t_cor": t,
                        "qv": qv,
                        },
                    )

        dset = dset.persist()
        progress(dset)
        del arr, out

        savdict = {"q(nm-1)": dset.qv.values,
                   "t(s)": dset.t_cor.values,
                   "corf": dset.get('g2', None),
                   "ttc": dset.get('ttc', None)}
        return savdict
    else:
        raise(ValueError(f"Method {method} not understood. "
                         "Choose between 'intra_train' and 'intra_train_ttc."))
    return
