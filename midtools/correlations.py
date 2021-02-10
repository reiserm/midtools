import pdb

# standard python packages
import numpy as np
import numba
from numba import prange
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
    ttc = ttc.reshape(int(np.sqrt(ttc.size)), -1)
    g2 = ttc_to_g2(ttc)
    return g2[:, 1]

def convert_ttc_err(ttc):
    ttc = ttc.reshape(int(np.sqrt(ttc.size)), -1)
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


@numba.jit(parallel=True, nopython=True)
def update_rois_beta(data, rois, rng=(-0.5, 2)):
    def calc_beta(c):
        n = c.shape[0]
        ind = np.sum(c >= 0, 0)
        ind = ind == n
        s = np.sum(c, 0)
        s1 = np.sum(c == 1, 0)
        s0 = np.sum(c == 0, 0)

        beta = s0 / s1 - n / s
        beta[~ind] = -2
        return beta

    for i in prange(len(rois)):
        roi = rois[i]
        beta = calc_beta(data[:, roi])

        rois[i] = rois[i][(beta > rng[0]) & (beta < rng[1])]
        if len(rois[i]) == 0:
            rois[i] = np.array([0])

    return rois


def update_mask(mask, data, rmap):

    average = data.mean(0)
    variance = data.var(0)
    mask = mask_radial(average, rmap, mask)

    mask = mask_radial(
        variance / average ** 2,
        rmap,
        mask=mask,
        lower_quantile=0.01,
    )

    mask = mask_asics(mask)
    return mask


def correlate(
    calibrator,
    method="intra_train",
    last=None,
    qmap=None,
    mask=None,
    setup=None,
    q_range=None,
    save_ttc=False,
    h5filename=None,
    norm_xgm=False,
    chunks=None,
    **kwargs,
):
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
        # valid = update_mask(valid, data, qmap.flatten())
        # valid = valid.flatten()
        rois = []
        bad_qs = []
        for i, (qi, qf) in enumerate(zip(qarr[:-1], qarr[1:])):
            roi = np.where((qmap.flatten() > qi) & (qmap.flatten() < qf) & valid)[0]
            if len(roi) > 100:
                rois.append(roi)
            else:
                roi = np.where((qmap.flatten() > qi) & (qmap.flatten() < qf))[0]
                rois.append(roi)
                bad_qs.append(i)

        # rois = update_rois_beta(data, rois);
        rois = [np.unravel_index(x, (16 * 512, 128)) for x in rois]

        # get only the q-bins in range
        qv = qarr[: len(rois)] + (qarr[1] - qarr[0]) / 2.0

        out = pyxpcs(
            data.reshape(-1, 16 * 512, 128),
            rois,
            mask=valid.reshape(16 * 512, 128),
            nprocs=1,
            verbose=False,
            **kwargs,
        )

        if method == "intra_train":
            corf = out["corf"]
            t = corf[1:, 0]
            corf = corf[1:, 1:]
        elif method == "intra_train_ttc":
            corf = np.dstack(out["twotime_corf"].values()).T
            t = out["twotime_xy"]

        if return_ == "all":
            return t, corf, qv
        elif return_ == "corf":
            for bad_q in bad_qs:
                corf[bad_q] *= 0
                corf[bad_q] -= 100
            return corf.astype("float32")
        elif return_ == "ttc":
            pass

    if chunks is None:
        chunks = {
            "intra_train": {"trainId": 1, "train_data": 16 * 512 * 128},
            "intra_train_ttc": {"trainId": 1, "train_data": 16 * 512 * 128},
        }

    arr = calibrator.data.copy(deep=False)
    npulses = np.unique(arr.pulseId.values).size

    if norm_xgm:
        with h5py.File(h5filename, "r") as f:
            xgm = f["pulse_resolved/xgm/energy"][:]
        xgm = xgm[:last, :npulses]
        xgm = xgm / xgm.mean(1)[:, None]
        print(f"Read XGM data from {h5filename} for normalization")
        arr = arr / xgm[None, None, None, ...]

    arr = arr.unstack()
    trainId = arr.trainId
    arr = arr.stack(train_data=("pulseId", "module", "dim_0", "dim_1"))
    arr = arr.chunk(chunks[method])
    npulses = arr.shape[1] // (16 * 512 * 128)

    if "nsteps" in q_range:
        qarr = np.logspace(
            np.log10(q_range["q_first"]), np.log10(q_range["q_last"]), q_range["nsteps"]
        )
    elif "qwidth" in q_range:
        qarr = np.arange(q_range["q_first"], q_range["q_last"], q_range["qwidth"])
    else:
        raise ValueError(
            "Could not define q-values. " "q_range not defined properly: " f"{q_range}"
        )

    # do the actual calculation
    if "intra_train" in method:
        t, corf, qv = calculate_correlation(
            np.ones(arr[0].shape), return_="all", **kwargs
        )

        dim = arr.get_axis_num("train_data")
        out = da.apply_along_axis(
            calculate_correlation,
            dim,
            arr.data,
            dtype="float32",
            shape=corf.shape,
            return_="corf",
            **kwargs,
        )

        if "ttc" in method:
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
            g2 = da.apply_along_axis(
                convert_ttc,
                1,
                dset["ttc"].stack(meas=("trainId", "qv"), data=("t1", "t2")),
                dtype="float32",
                shape=(t.size,),
            )
            g2 = g2.reshape(dset.trainId.size, qv.size, t.size).swapaxes(1, 2)
            dset = dset.assign_coords(t_cor=t[1:])
            dset = dset.assign(g2=(("trainId", "t_cor", "qv"), g2[:, 1:]))
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

        savdict = {
            "q(nm-1)": dset.qv.values,
            "t(s)": dset.t_cor.values,
            "corf": dset.get("g2", None),
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
