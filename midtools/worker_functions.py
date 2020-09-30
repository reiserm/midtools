import numpy as np
import warnings
import xarray as xr
import dask
import bottleneck  as bn
from functools import wraps


def _xarray2numpy(func):
    @wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            data = data.astype('float32')
        elif isinstance(data, xr.core.dataarray.DataArray):
            data = data.values.astype('float32')
        elif isinstance(data, dask.array.core.Array):
            data = data.values.astype('float32')
        else:
            raise(ValueError(f"Data type {type(data)} not understood."))
        return func(data, *args, **kwargs)
    return wrapper


def _calibrate_worker(calibrator, corrections):
    def calibrate(func):
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            # corrections are method specific therefore, the corrections argument
            # has to be defined for each worker separately
            worker_corrections = list(dict(filter(lambda x: x[1] and x[0] in corrections,
                calibrator.worker_corrections.items())).keys())
            subshape = calibrator.subshape
            adu_per_photon = calibrator.adu_per_photon
            mask = calibrator.mask
            if 'asic_commonmode' in worker_corrections:
                data = _asic_commonmode_worker(data, mask, adu_per_photon, subshape)
            if 'cell_commonmode' in worker_corrections:
                data = _cell_commonmode_worker(data, mask, adu_per_photon)
            if 'dropletize' in worker_corrections:
                data = _dropletize_worker(data, adu_per_photon)
            return func(data, *args, **kwargs)
        return wrapper
    return calibrate


def _to_asics(data, reverse=False, subshape=(64, 64)):
    """convert module to subasics and vise versa"""
    nrows, ncols = subshape
    nv = 512 // nrows
    nh = 128 // ncols

    if data.ndim == 3:
        if not reverse:
            return (data
                    .reshape(-1, nv, nrows, nh, ncols)
                    .swapaxes(2, 3)
                    .reshape(-1, 16, nrows, ncols))
        else:
            return (data
                    .reshape(-1, nv, nh, nrows, ncols)
                    .swapaxes(2, 3)
                    .reshape(-1, 512, 128))
    elif data.ndim == 2:
        if not reverse:
            return (data
                    .reshape(nv, nrows, nh, ncols)
                    .swapaxes(1, 2)
                    .reshape(16, nrows, ncols))
        else:
            return (data
                    .reshape(nv, nh, nrows, ncols)
                    .swapaxes(1, 2)
                    .reshape(512,128))


def _asic_commonmode_worker(frames, mask, adu_per_photon=58,
        subshape=(64, 64)):
    """Asic commonmode to be used in apply_along_axis"""
    frames = frames.reshape(-1, 16, 512, 128)
    frames[:, ~mask] = np.nan
    asics = (_to_asics(frames.reshape(-1, 512, 128), subshape=subshape)
             .reshape(-1, *subshape))
    asic_median = np.nanmedian(asics, axis=(1, 2))
    indices = asic_median <= (adu_per_photon / 2)
    asics[indices] -= asic_median[indices, None, None]
    ascis = asics.reshape(-1, 16, *subshape)
    asics = _to_asics(asics, reverse=True, subshape=subshape)
    frames = asics.reshape(-1, 16, 512, 128)
    return frames


def _cell_commonmode_worker(frames, mask, adu_per_photon=58, window=16):
    """Time axis Commonmode to be used in apply_along_axis"""
    frames[:, ~mask] = np.nan
    move_mean = bn.move_mean(frames, window, axis=0, min_count=1)
    frames = np.where(move_mean <= (adu_per_photon / 2), frames - move_mean,
                frames)
    return frames


def _dropletize_worker(arr, adu_per_photon=58):
    """Convert adus to photon counts."""
    arr = np.floor((arr + .5 * adu_per_photon) / adu_per_photon)
    arr = np.where(arr >= 0, arr, 0)
    return arr
