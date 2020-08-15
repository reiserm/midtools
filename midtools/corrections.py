import numpy as np
from extra_data.components import AGIPD1M
from .average import average
import dask.array as da
import xarray as xr

import pdb

class Calibrator:
    """Calibrate AGIPD dataset"""

    def __init__(self, run, last_train=10_000, pulses_per_train=100,
            first_cell=1, train_step=1, pulse_step=1,
            dark_run_number=None, mask=None, adu_per_photon=62,
            apply_internal_mask=False, dropletize=False, stripes=None,
            baseline=False, asic_commonmode=False,):

        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = run
        self.last_train = last_train
        self.pulses_per_train = pulses_per_train
        self.first_cell = first_cell
        self.train_step = train_step
        self.pulse_step = pulse_step

        self.mask = mask
        self.adu_per_photon = adu_per_photon

        self.is_proc = None
        #: (str, None): Directory of dark run. Determined by dark_run_number
        self.darkdir = None
        #: (DataCollection, None): DataCollection of HG dark run.
        self.darkrun = None
        #: (np.ndarray, None): average dark over trains
        self.avr_dark = None
        # setting the dark run also sets the other dark attributes
        self.dark_run_number = dark_run_number

        self.corrections = {'apply_internal_mask': apply_internal_mask,
                            'baseline': baseline,
                            'asic_commonmode': asic_commonmode,
                            'dropletize': dropletize,}

        self.stripes = stripes


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

        train_step = train_step if bool(train_step) else self.train_step
        last_train = last_train if bool(last_train) else self.last_train

        print("Requesting data...", flush=True)
        agp = AGIPD1M(self.run, min_modules=16)
        arr = agp.get_dask_array('image.data')
        print("Loaded dask array.", flush=True)

        # coords are module, dim_0, dim_1, trainId, pulseId after unstack
        arr = arr.unstack()
        self.is_proc = True if len(arr.dims) == 5 else False
        print(f"Is processed: {self.is_proc}")

        train_slice = slice(0, last_train, train_step)
        pulse_slice = slice(self.first_cell,
                            self.first_cell + self.pulses_per_train,
                            self.pulse_step)

        # select pulses and skip the first one
        if self.is_proc:
            arr = arr[..., train_slice, pulse_slice]
        else:
            # drop gain map
            arr = arr[:, 0, ..., train_slice, pulse_slice]
            arr = arr.rename({'dim_1': 'dim_0',
                              'dim_2': 'dim_1'})

        if self.corrections['apply_internal_mask']:
            arr = self._apply_internal_mask(agp, arr, train_slice, pulse_slice)

        arr = arr.stack(train_pulse=('trainId', 'pulseId'),
                        pixels=('module', 'dim_0', 'dim_1'))

        # apply corrections
        for correction, options in self.corrections.items():
            if bool(options) and 'apply' not in correction:
                print(f"Apply correction: {correction}")
                arr = getattr(self, "_" + correction)(arr)

        arr = arr.transpose('train_pulse', 'pixels')
        return arr


    def _dropletize(self, arr):
        """Convert adus to photon counts."""
        arr = np.floor(
            (arr + .5*self.adu_per_photon) / self.adu_per_photon)
        arr = arr.where(arr >= 0, other=0)
        return arr


    def _apply_internal_mask(self, agp, arr, train_slice,  pulse_slice):
        """Exclude pixels based on calibration pipeline."""
        mask_int = agp.get_dask_array('image.mask')
        mask_int = mask_int.unstack()
        if self.is_proc:
            mask_int = mask_int[..., train_slice, pulse_slice]
        else:
            # drop gain map
            mask_int = mask_int[:, 0, ..., train_slice, pulse_slice]
            mask_int = mask_int.rename({'dim_1': 'dim_0',
                                        'dim_2': 'dim_1'})

        arr = arr.where((mask_int.data <= 0) | (mask_int.data >= 8))
        return arr


    def _baseline(self, arr):
        """Baseline correction based on Ta stripes"""
        arr = arr.unstack('pixels')
        pix = arr.where(self.stripes[None, ...])
        module_median = pix.median(['dim_0', 'dim_1'], skipna=True)
        arr = arr - module_median
        arr = arr.stack(pixels=('module', 'dim_0', 'dim_1'))
        return arr


    @staticmethod
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


    def _asic_commonmode(self, arr):
        """Apply commonmode on asic level."""
        arr = arr.unstack('pixels')
        arr = arr.where(self.mask[None, ...])
        arr = arr.unstack('train_pulse')
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
        arr = arr.stack(train_pulse=('trainId', 'pulseId'),
                pixels=('module', 'dim_0', 'dim_1'))
        return arr







