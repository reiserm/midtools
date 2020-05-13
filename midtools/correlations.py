import pdb
# standard python packages
import numpy as np
import scipy.integrate as integrate
from time import time
import pickle
import xarray
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


def correlate(run, method='per_train', last=None, qmap=None,
	mask=None,  npulses=None, to_counts=False, apply_internal_mask=True, 
    setup=None, adu_per_photon=65, q_range=None, client=None,  **kwargs):
    """Calculate XPCS correlation functions of a run using dask.

    Args:
        run (DataCollection): the run objection, e.g., created by RunDirectory.

        last (int, optional): last train index. Defaults None. Set to small
            number for testing.

        mask (np.ndarray, optional): Good pixels are 1 bad pixels are 0. If None,
            no pixel is masked.

        npulses (int): Number of pulses per pulse train. Defaults to None.

        to_count (bool, optional): Convert the adus to photon counts based on
            thresholding and the value of adu_per_photon.

        apply_internal_mask (bool, optional): Read and apply the mask calculated
            from the calibration pipeline. Defaults to True.

        setup ((Xana.Setup), optional): Xana setupfile. Defaults to None.
            If str, include path in the filename. Otherwise provide Xana Setup
            instance.

        adu_per_photon ((int,float), optional): number of ADUs per photon.

        client (dask.distributed.Client): Defaults to None.

        qrange (dict): Information on the q-bins. Should contain the keys: 
            * q_first
            * q_last
            * nsteps

    Returns:
        dict: Dictionary containing the results depending on the method it
            contains the keys:
            * 'corf': the xpcs correlation function
    """

    def calculate_correlation(data, return_='all',  **kwargs):
        # Check data type
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, xarray.DataArray):
            data = np.array(data.data)
        else:
            raise(ValueError(f"Data type {type(data)} not understood."))
        
        data = data.reshape(npulses,8192,128)
        wnan = np.isnan(np.sum(data, axis=0))
        xpcs_mask = mask_2d & ~wnan.reshape(8192,128)

        rois = [np.where((qr_tmp>qi) & (qr_tmp<qf) & xpcs_mask) for qi,qf in
            zip(qarr[:-1],qarr[1:])]

        out = pyxpcs(data, rois, mask=xpcs_mask, nprocs=1, verbose=False, **kwargs)

        corf = out['corf']
        t = corf[1:,0]
        qv = corf[0,1:]
        corf = corf[1:,1:]
        if return_ == 'all':
            return t, corf, qv
        elif return_ == 'corf':
            return corf.astype('float32')

    # getting the data
    agp = AGIPD1M(run, min_modules=16)
    # data = agp.get_dask_array('image.data')
    arr = agp.get_dask_array('image.data')
    if to_counts:
        arr.data = np.floor((arr.data + 0.5*adu_per_photon) / adu_per_photon)
        arr.data[arr.data<0] = 0
        # arr.data.astype('float32')

    if apply_internal_mask:
        mask_int = agp.get_dask_array('image.mask')
        arr = arr.where((mask_int.data <= 0) | (mask_int.data>=8))
        # mask[(mask_int.data > 0) & (mask_int.data<8)] = 0

    arr = arr.where(mask[:,None,:,:])

    arr = arr.unstack()
    arr = arr.transpose('trainId', 'pulseId', 'module', 'dim_0', 'dim_1')
    arr = arr[:last,:npulses]
    arr = arr.stack(pixels=('pulseId', 'module','dim_0', 'dim_1'))
    arr = arr.chunk({'trainId':8})

    qmin, qmax, n = [q_range[x] for x in ['q_first', 'q_last', 'nsteps']]
    qarr = np.linspace(qmin,qmax,n)
    qv = qarr + (qarr[1] - qarr[0])/2.
    qr_tmp = qmap.reshape(16*512,128)
    mask_2d = mask.reshape(16*512,128)

    # do the actual calculation
    if method == 'per_train':
        t, corf, _ = calculate_correlation(arr[0], return_='all',
                **kwargs)

        dim = arr.get_axis_num("pixels")
        out = da.apply_along_axis(calculate_correlation, dim, arr.data,
            dtype='float32', shape=corf.shape, return_='corf',
            **kwargs)
        out = out.persist()
        progress(out)

        savdict = {"q(nm-1)":qv, "t(s)":t, "corf":out}
        return savdict
    else:
        raise(ValueError(f"Method {method} not understood. \
                        Choose between 'average' and 'single'."))
    return
