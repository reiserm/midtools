import pdb
# standard python packages
import numpy as np
import scipy.integrate as integrate
from time import time
import pickle
import xarray
import h5py
import copy

# main analysis software can be installed by: pip install Xana
from Xana import Xana
import Xana.Setup
from Xana.XpcsAna.pyxpcs3 import pyxpcs

# reading AGIPD data provided by XFEL
from extra_data import RunDirectory, stack_detector_data, open_run
from extra_geom import AGIPD_1MGeometry
from extra_data.components import AGIPD1M

# DASK
import dask.array as da
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar

from .corrections import _asic_commonmode_worker, _dropletize_worker


def correlate(calibrator, method='per_train', last=None, qmap=None,
    mask=None, setup=None, q_range=None, save_ttc=False, h5filename=None,
    norm_xgm=False, chunks=None, **kwargs):
    """Calculate XPCS correlation functions of a run using dask.

    Args:
        calibrator (Calibrator): the data broker.

        last (int, optional): last train index. Defaults None. Set to small
            number for testing.

        mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0. If None,
            no pixel is masked.

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

    def calculate_correlation(data, return_='all', adu_per_photon=58,
                              worker_corections=False, **kwargs):
        # Check data type
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, xarray.DataArray):
            data = np.array(data.data)
        else:
            raise(ValueError(f"Data type {type(data)} not understood."))

        if bool(worker_corections):
            if 'asic_commonmode' in worker_corections:
                data = _asic_commonmode_worker(data, mask, adu_per_photon)
            if 'dropletize' in worker_corections:
                data = _dropletize_worker(data, adu_per_photon)

        data = data.reshape(npulses, 8192, 128)
        wnan = np.isnan(np.sum(data, axis=0))
        xpcs_mask = mask_2d & ~wnan.reshape(8192,128)

        rois = [np.where((qmap_2d>qi) & (qmap_2d<qf) & xpcs_mask) for qi,qf in
            zip(qarr[:-1],qarr[1:])]

        # get only the q-bins in range
        qv = qarr[:len(rois)] + (qarr[1] - qarr[0])/2.

        out = pyxpcs(data, rois, mask=xpcs_mask, nprocs=1, verbose=False,
                **kwargs)

        corf = out['corf']
        t = corf[1:,0]
        corf = corf[1:,1:]
        # ttc = out['twotime_corf'][0]
        if return_ == 'all':
            return t, corf, qv
        elif return_ == 'corf':
            return corf.astype('float32')
        elif return_ == 'ttc':
            pass

    if chunks is None:
        chunks = {'per_train': {'trainId': 1, 'train_data': 16*512*128}}

    arr = calibrator.data.copy()
    npulses = np.unique(arr.pulseId.values).size
    adu_per_photon = calibrator.adu_per_photon
    worker_corections = list(dict(filter(lambda x: x[1],
        calibrator.worker_corrections.items())).keys())

    if norm_xgm:
        with h5py.File(h5filename, 'r') as f:
            xgm = f["pulse_resolved/xgm/energy"][:]
        xgm = xgm[:last, :npulses]
        xgm = xgm / xgm.mean(1)[:, None]
        print(f"Read XGM data from {h5filename} for normalization")
        arr = arr / xgm[None, None, None, ...]

    arr = arr.unstack()
    arr = arr.stack(train_data=('pulseId', 'module', 'dim_0', 'dim_1'))
    arr = arr.chunk(chunks['per_train'])
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
    if method == 'per_train':
        t, corf, qv = calculate_correlation(arr[0], return_='all',
                **kwargs)

        dim = arr.get_axis_num("train_data")
        out = da.apply_along_axis(calculate_correlation, dim, arr.data,
            dtype='float32', shape=corf.shape, return_='corf',
            worker_corections=worker_corections, adu_per_photon=adu_per_photon,
            **kwargs)

        out = out.persist()
        progress(out)

        savdict = {"q(nm-1)": qv,
                   "t(s)": t,
                   "corf": out.compute()}
        del out
        return savdict
    else:
        raise(ValueError(f"Method {method} not understood. \
                        Choose between 'average' and 'single'."))
    return
