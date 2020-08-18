import numpy as np
from extra_data.components import AGIPD1M
import dask.array as da
import xarray as xr
from dask.distributed import Client, progress
import warnings

import pdb

class Calibrator:
    """Calibrate AGIPD dataset"""
    adu_per_photon = 66
    mask = np.ones((16, 512, 128), 'bool')
    worker_corrections = {'asic_commonmode': False}

    def __init__(self, run, last_train=10_000, pulses_per_train=100,
            first_cell=1, train_step=1, pulse_step=1,
            dark_run_number=None, mask=None,
            apply_internal_mask=False, dropletize=False, stripes=None,
            baseline=False, asic_commonmode=False,):

        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = run
        self.last_train = last_train
        self.pulses_per_train = pulses_per_train
        self.first_cell = first_cell
        self.train_step = train_step
        self.pulse_step = pulse_step

        Calibrator.mask  = mask
        self.xmask = xr.DataArray(self.mask,
                dims=('module', 'dim_0', 'dim_1'),
                coords={'module': np.arange(16),
                        'dim_0': np.arange(512),
                        'dim_1': np.arange(128)})

        self.is_proc = None
        #: (str, None): Directory of dark run. Determined by dark_run_number
        self.darkdir = None
        #: (DataCollection, None): DataCollection of HG dark run.
        self.darkrun = None
        #: (np.ndarray, None): average dark over trains
        self.avr_dark = None
        # setting the dark run also sets the other dark attributes
        self.dark_run_number = dark_run_number

        self.corrections = {'baseline': baseline, # baseline has to be applied
                                                  # before any masking
                            'masking': True,
                            'internal_masking': apply_internal_mask,
                            'dropletize': dropletize,}

        Calibrator.worker_corrections['asic_commonmode'] = asic_commonmode
        self.stripes = stripes
        self.data = None


    @property
    def dark_run_number(self):
        """int: Run number of high gain dark."""
        return self.__dark_number


    @dark_run_number.setter
    def dark_run_number(self, number):
        if isinstance(number, int):
            if 'proc' in self.datdir:
                raise ValueError("Use dark options only with raw data")
            self.darkdir = re.sub('r\d{4}', f'r{number:04}', self.datdir)
            self.darkrun = RunDirectory(self.darkdir)

            # process darsk
            self.avr_dark = average(self.darkrun,
                                    last=10,
                                    npulses=self.pulses_per_train,)
            print(type(self.avr_dark), self.avr_dark.shape)
        elif number is None:
            pass
        else:
            raise TypeError(f'Invalid dark run number type {type(number)}.')
        self.__dark_run_number = number


    @property
    def stripes(self):
        """Pixels behind the Ta stripes for baseline correction."""
        return self.__stripes


    @stripes.setter
    def stripes(self, stripes):
        if isinstance(stripes, str):
            stripes = np.load(stripes).astype('bool')
            if stripes.shape != (16, 512, 128):
                raise ValueError("Stripe mask has wrong shape: "
                                 f"shape is {stripe.shape}")
            self.corrections['baseline'] = True
        elif stripes is None:
            pass
        else:
            raise TypeError(f'Invalid dark run number type {type(number)}.')
        self.__stripes = stripes


    def _get_data(self, train_step=None, last_train=None):
        """Read data and apply corrections"""

        print("Requesting data...", flush=True)
        self._agp = AGIPD1M(self.run, min_modules=16)
        arr = self._agp.get_dask_array('image.data')
        print("Loaded dask array.", flush=True)

        # coords are module, dim_0, dim_1, trainId, pulseId after unstack
        arr = arr.unstack()
        self.is_proc = True if len(arr.dims) == 5 else False
        print(f"Is processed: {self.is_proc}")

        arr = self._slice(arr, train_step, last_train)
        arr = arr.stack(train_pulse=('trainId', 'pulseId'))
        arr = arr.transpose('train_pulse', ...)

        for correction, options in self.corrections.items():
            if bool(options):
                print(f"Apply {correction}")
                arr = getattr(self, "_" + correction)(arr)

        for correction, flag in self.worker_corrections.items():
            if bool(flag):
                print(f"Apply {correction} on workers.")

        arr = arr.stack(pixels=('module', 'dim_0', 'dim_1'))
        arr = arr.transpose('train_pulse', 'pixels')
        self.data = arr
        return arr


    def _slice(self, arr, train_step=None, last_train=None):
        """Select trains and pulses."""
        train_step = train_step if bool(train_step) else self.train_step
        last_train = last_train if bool(last_train) else self.last_train

        train_slice = slice(0, last_train, train_step)
        pulse_slice = slice(self.first_cell,
                            self.first_cell + self.pulses_per_train,
                            self.pulse_step)
        if self.is_proc:
            arr = arr[..., train_slice, pulse_slice]
        else:
            # drop gain map
            arr = arr[:, 0, ..., train_slice, pulse_slice]
            arr = arr.rename({'dim_1': 'dim_0',
                              'dim_2': 'dim_1'})
        return arr


    def _masking(self, arr):
        """Mask bad pixels with provided use mask."""
        return arr.where(self.xmask)


    def _dropletize(self, arr):
        """Convert adus to photon counts."""
        arr = np.floor(
            (arr + .5*self.adu_per_photon) / self.adu_per_photon)
        arr = arr.where(arr >= 0, other=0)
        return arr


    def _internal_masking(self, arr):
        """Exclude pixels based on calibration pipeline."""
        mask_int = self._agp.get_dask_array('image.mask')
        mask_int = mask_int.unstack()
        mask_int = self._slice(mask_int)
        mask_int = mask_int.stack(train_pulse=('trainId', 'pulseId'))
        arr = arr.where((mask_int <= 0) | (mask_int >= 8))
        return arr


    def _baseline(self, arr):
        """Baseline correction based on Ta stripes"""
        pix = arr.where(self.stripes[None, ...])
        module_median = pix.median(['dim_0', 'dim_1'], skipna=True)
        if np.sum(np.isfinite(module_median)).compute() == 0:
            warnings.warn("All module medians are nan. Check masks")
        module_median = module_median.where(np.isfinite(module_median), 0)
        arr = arr - module_median
        return arr


    def _asic_commonmode(self, arr):
        return self._asic_commonmode_xarray(arr)


    @classmethod
    def _calib_worker(cls, frames):
        """Apply corrections on workers."""
        if cls.worker_corrections['asic_commonmode']:
            frames = cls._asic_commonmode_worker(frames)
        return frames


    def _asic_commonmode_ufunc(self, arr):
        """Apply commonmode on asic level"""

        # arr = arr.unstack()
        # arr = arr.stack(frame_data=('module', 'dim_0', 'dim_1'))
        # npulses = np.unique(arr.pulseId.values).size
        arr = arr.chunk({'pixels': -1})

        arr = xr.apply_ufunc(self._asic_commonmode_worker,
                             arr,
                             vectorize=True,
                             input_core_dims=[['pixels']],
                             output_core_dims=[['pixels']],
                             output_dtypes=[np.int],
                             dask='parallelized',
                             )
        # dim = arr.get_axis_num("train_data")
        # arr.data = da.apply_along_axis(worker, dim, arr.data, dtype='float32')
        #         # shape=(npulses*16*512*128))
        # arr = arr.persist()
        # progress(arr)
        # arr = arr.unstack()
        # arr = arr.stack(train_pulse=('trainId', 'pulseId'),
        #         pixels=('module', 'dim_0', 'dim_1'))
        return arr


    def _asic_commonmode_xarray(self, arr):
        """Apply commonmode on asic level using xarray."""
        arr = arr.unstack()
        arr = arr.stack(train_pulse_module=['trainId', 'pulseId', 'module'])
        arr = arr.chunk({'train_pulse_module': 1024})

        arr = arr.assign_coords(asc_0=('dim_0', np.repeat(np.arange(1,9), 64)),
                                  asc_1=('dim_1', np.repeat([-1, 1], 64)))
        arr = arr.assign_coords(asc=arr.asc_0 * arr.asc_1)
        asics = arr.groupby('asc')

        asic_median = asics.median(skipna=True)
        asic_median = asic_median.where(asic_median < self.adu_per_photon/2,
                                        other=0)
        asics -= asic_median

        arr = asics.drop(('asc', 'asc_0', 'asc_1'))
        arr = arr.unstack()
        arr = arr.stack(train_pulse=('trainId', 'pulseId'))
        return arr


def _to_asics(data, reverse=False):
    """convert module to asics and vise versa"""
    nrows = ncols = 64
    if data.ndim == 3:
        if not reverse:
            return (data
                    .reshape(-1, 8, nrows, 2, ncols)
                    .swapaxes(2, 3)
                    .reshape(-1, 16, nrows, ncols))
        else:
            return (data
                    .reshape(-1, 8, 2, nrows, ncols)
                    .swapaxes(2, 3)
                    .reshape(-1, 512, 128))
    elif data.ndim == 2:
        if not reverse:
            return (data
                    .reshape(8, nrows, 2, ncols)
                    .swapaxes(1, 2)
                    .reshape(16, nrows, ncols))
        else:
            return (data
                    .reshape(8, 2, nrows, ncols)
                    .swapaxes(1, 2)
                    .reshape(512,128))


def _asic_commonmode_worker(frames, mask, adu_per_photon=58):
    """Asic commonmode to be used in apply_along_axis"""
    frames = frames.reshape(-1, 16, 512, 128)
    frames[:, ~mask] = np.nan
    asics = (_to_asics(frames.reshape(-1, 512, 128))
             .reshape(-1, 64, 64))
    asic_median = np.nanmedian(asics, axis=(1, 2))
    indices = asic_median <= adu_per_photon / 2
    asics[indices] -= asic_median[indices, None, None]
    ascis = asics.reshape(-1, 16, 64, 64)
    asics = _to_asics(asics, reverse=True)
    frames = asics.reshape(-1, 16, 512, 128)
    return frames


