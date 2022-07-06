import pdb

# standard python packages
import numpy as np
from scipy import integrate
from scipy import ndimage
from time import time
from itertools import cycle
import pickle
import xarray
import h5py
import copy

# main analysis software can be installed by: pip install Xana
from Xana import Xana
import Xana.Setup
from Xana.XpcsAna.pyxpcs3 import pyxpcs
from Xana.SaxsAna.defineqrois import defineqrois
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
from .masking import mask_radial, mask_asics


def ttc_to_g2_ufunc(ttc, dims, time=None):
    return xarray.apply_ufunc(
        ttc_to_g2,
        ttc,
        input_core_dims=[dims],
        kwargs={"time": time},
        dask="parallelized",
        output_dtypes=[float],
    )


def convert_ttc(ttc):
    # ttc = ttc.reshape(int(np.sqrt(ttc.size)), -1)
    g2 = ttc_to_g2(ttc)
    return g2[:, 1]


def convert_ttc_err(ttc):
    # ttc = ttc.reshape(int(np.sqrt(ttc.size)), -1)
    g2 = ttc_to_g2(ttc)
    return g2[:, 2]


def blur_gauss(U, sigma=2.0, truncate=4.0):

    V = U.copy()
    V[np.isnan(U)] = 0
    VV = ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0 * U.copy() + 1
    W[np.isnan(U)] = 0
    WW = ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)

    return VV / WW


def mask_rolling(data, windows=None):
    """Identifies bad pixels based on the rolling average over time.

    Args:
        data (np.ndarray): data to be used for masking.
        window (list): list of integers with window size for running average.
    """

    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def movingaverage(values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, "valid")
        return sma

    if windows is None:
        windows = [2, 4, 8, 16]

    nt, s = data.shape
    data = data.reshape(nt, -1)

    good_pixels = np.ones(data.shape[1])
    nuniq = np.unique(data).size
    if nuniq == 1:
        return good_pixels.astype("bool").reshape(s)
    for N in windows:
        roll = np.apply_along_axis(lambda x, N: running_mean(x, N), 0, data, N)
        roll = roll.mean(0).round(3)
        vals, invs, cnts = np.unique(roll, return_counts=True, return_inverse=True)
        inds = np.asarray(
            [np.where(cnts == x)[0][0] for x in np.sort(cnts)[-nuniq + 1 :]]
        )
        pixels = np.asarray([1 if x == i else 0 for x in invs for i in inds])
        pixels = pixels.reshape(-1, nuniq - 1).sum(1)
        good_pixels *= pixels
    return good_pixels.astype("bool").reshape(s)


def update_rois(data, rois, doplot=False):

    for i in range(len(rois)):
        roi = rois[i]
        npix = len(roi)
        pixels = mask_rolling(data[:, roi])
        rois[i] = rois[i][pixels]
        if doplot:
            figure()
            pcolors = sns.color_palette("crest", pixels.size)
            for ip, p in enumerate(rois[i]):
                plot(data[:, p], "o", color=pcolors[ip])
    return rois



def update_mask(data, rmap, mask):

    # average = data.mean(0)
    # variance = data.var(0)
    mask = mask_radial(data, rmap, mask, lower_quantile=0.01)

    # mask = mask_radial(
    #     variance / average ** 2,
    #     rmap,
    #     mask=mask,
    #     lower_quantile=0.01,
    # )

    # mask = mask_asics(mask)
    return mask


def get_qarr(r, par='q', scale='log'):
    scale = r.get('scale', scale)
    if "nsteps" in r:
        if scale == 'log':
            arr = np.logspace(
                np.log10(r[f"{par}_first"]), np.log10(r[f"{par}_last"]), r["nsteps"]
            )
        else:
            arr = np.linspace(
                r[f"{par}_first"], r[f"{par}_last"], r["nsteps"]
            )

    elif f"{par}_width" in r:
        arr = np.arange(r[f"{par}_first"], r[f"{par}_last"], r[f"{par}_width"])
    else:
        raise ValueError(
            "Could not define parameter-ranges. "
            "range input not defined properly: "
            f"{r}"
        )
    return arr

def get_q_phi_pixels(setup, d):
    """Input is a dictionary of xpcs_opt."""
    q_range = d['q_range']
    qarr = get_qarr(q_range, par='q')
    qarr = [(qarr, (qarr[1]-qarr[0])),]

    if 'phi_range' in d:
        phi_range = d['phi_range']
        phiarr = get_qarr(phi_range, par='phi', scale='lin')
        phiarr = [(phiarr, (phiarr[1]-phiarr[0])),]
    else:
        phiarr = [(0,360)]

    Isaxs = 0  # dummy
    print(qarr, flush=1)
    print(phiarr, flush=1)
    defineqrois(setup, Isaxs, qv_init=qarr, phiv_init=phiarr, mirror=True)

    return setup


def get_xpcs_rois(qarr, qmap, valid):
    rois = []
    bad_qs = []
    for i, (qi, qf) in enumerate(zip(qarr[:-1], qarr[1:])):
        roi = np.where((qmap.flatten() > qi) & (qmap.flatten() < qf) & valid.flatten())[
            0
        ]
        if len(roi) > 100:
            rois.append(roi)
        else:
            roi = np.where((qmap.flatten() > qi) & (qmap.flatten() < qf))[0]
            rois.append(roi)
            bad_qs.append(i)
    return rois, bad_qs


def correlate(
    calibrator,
    method="intra_train",
    qmap=None,
    mask=None,
    setup=None,
    q_range=None,
    save_ttc=False,
    h5filename=None,
    norm_xgm=False,
    chunks=None,
    batch_size=16,
    **kwargs,
):
    """Calculate XPCS correlation functions of a run using dask.

    Args:
        calibrator (Calibrator): the data broker.

        method (str, optional): along which axis to correlate. Default is to
            calculate the correlation function per train.

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

    worker_corrections = ("asic_commonmode", "cell_commonmode", "dropletize")

    @wf._xarray2numpy(dtype="float32")
    @wf._calibrate_worker(calibrator, worker_corrections)
    def calculate_correlation(data, return_="all", blur=False, **kwargs):

        if blur:
            for i, image in enumerate(data):
                image[~mask] = np.nan
                data[i] = blur_gauss(image, sigma=3.0, truncate=4.0)

        data = data.reshape(-1, 16 * 512 * 128)
        valid = (data >= 0).all(0) & mask.flatten()

        # valid = update_mask(np.mean(data, axis=0, keepdims=True), qmap.flatten(), valid)
        # valid = valid.flatten()

        rois, bad_qs = get_xpcs_rois(qarr, qmap, valid)
        # rois = update_rois_beta(data, rois);
        rois = [np.unravel_index(x, (16 * 512, 128)) for x in rois]

        # get only the q-bins in range
        qv = qarr[: len(rois)] + (qarr[1] - qarr[0]) / 2.0

        if return_ == "all":
            out = pyxpcs(
                data.reshape(-1, 16 * 512, 128),
                rois,
                mask=valid.reshape(16 * 512, 128),
                nprocs=1,
                verbose=False,
                **kwargs,
            )
            corf = out["corf"]
            t = out["twotime_xy"]
            corf = corf[1:, 1:]
            return t, corf, qv

        elif return_ == "ttc":
            data = data.reshape(-1, npulses, 16 * 512, 128)
            ntrains = data.shape[0]
            ttcs = [[] for _ in range(len(strides) + 1)]
            data_indices = np.arange(ntrains)
            for istride, stride in enumerate(strides):
                xdata_indices = np.roll(data_indices, -1 * abs(stride))
                for i, j in zip(data_indices, xdata_indices):
                    train = data[i]
                    crossdata = data[j]
                    out = pyxpcs(
                        train,
                        rois,
                        mask=valid.reshape(16 * 512, 128),
                        nprocs=1,
                        crossdata=crossdata,
                        verbose=False,
                        **kwargs,
                    )
                    if istride == 0:
                        ttcs[0].append(list(out["twotime_corf"].values()))
                    ttcs[istride + 1].append(list(out["xtwotime_corf"].values()))
            return np.stack(ttcs)

    arr = calibrator.data.copy(deep=False)
    npulses = len(arr.pulseId)

    # arr = arr.unstack('train_pulse')
    trainId = arr.trainId
    arr["trainId"] = np.arange(len(trainId))

    qarr = get_qarr(q_range)
    t, _, qv = calculate_correlation(
        np.ones((npulses, 16, 512, 128)), return_="all", **kwargs
    )

    arr = arr.assign_coords(batch=arr.trainId // batch_size)
    arr = arr.groupby("batch")
    strides = np.array([1, 10])

    # do the actual calculation
    if "intra_train" in method:

        dset = xarray.apply_ufunc(
            calculate_correlation,
            arr,
            input_core_dims=[["trainId", "pulseId", "pixels"]],
            exclude_dims=set(["pulseId", "pixels"]),
            output_core_dims=[["stride", "trainId", "qv", "t1", "t2"]],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "stride": len(strides) + 1,
                    "qv": len(qv),
                    "t1": len(t),
                    "t2": len(t),
                },
                "allow_rechunk": True,
            },
            dask="parallelized",
            vectorize=True,
            kwargs=dict(return_="ttc", **kwargs),
            output_dtypes=(np.float32),
        )
        dset = dset.drop("batch")
        coords = {
            "trainId": trainId,
            "stride": np.append(0, strides),
            "t1": t,
            "t2": t,
            "qv": qv,
        }
        dset = dset.to_dataset(name="ttc")
        dset = dset.assign_coords(coords)
        dset = dset.chunk({"trainId": 1})

        dset = dset.persist()
        progress(dset)

        # if "ttc" in method:

        # g2 = xarray.apply_ufunc(
        #     convert_ttc,
        #     dset['ttc'],
        #     input_core_dims=[['t1', 't2']],
        #     exclude_dims=set(['t1', 't2']),
        #     output_core_dims=[['t_cor']],
        #     dask_gufunc_kwargs={'allow_rechunk':True, 'output_sizes':{'t_cor':len(t)}},
        #     dask='parallelized',
        #     vectorize=True,
        #     output_dtypes=(np.float32),
        # )
        # g2 = g2.rename('g2')
        # g2 = g2.assign_coords({'t_cor': t})
        # g2 = g2.isel(t_cor=slice(1,None))
        # g2 = g2.persist()

        # del arr, ttc, xttc

        savdict = {
            "q(nm-1)": dset.qv,
            "t(s)": dset.t1,
            "strides": dset.stride,
            # "corf": np.ones((dset.trainId.size, dset.qv.size)),
            "ttc": dset.get("ttc", None),
        }
        return savdict
    else:
        raise (
            ValueError(
                f"Method {method} not understood. "
                "Choose between 'intra_train' and 'intra_train_ttc."
            )
        )
    return
