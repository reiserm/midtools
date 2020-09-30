import numpy as np
from extra_data.components import AGIPD1M
import dask.array as da
import xarray as xr
from dask.distributed import Client, progress
import warnings
import h5py as h5
import bottleneck  as bn

import pdb

class Calibrator:
    """Calibrate AGIPD dataset"""
    adu_per_photon = 66
    mask = np.ones((16, 512, 128), 'bool')


    def __init__(self, run, cell_ids, train_ids, flatfield_run_number=None,
            dark_run_number=None, mask=None, is_dark=False, is_flatfield=False,
            apply_internal_mask=False, dropletize=False, stripes=None,
            baseline=False, asic_commonmode=False, subshape=(64, 64),
            cell_commonmode=False, cellCM_window=2):

        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = run
        #: bool: True if data is a dark run
        self.is_dark = is_dark
        #: bool: True if data is a flatfield run
        self.is_flatfield = is_flatfield
        #: tuple: asic (or subasic) shape for asic commonmode. Default (64, 64)
        self.subshape = subshape

        self.cell_ids = cell_ids
        self.train_ids = train_ids

        # corrections applied on Dask lazy array
        self.corrections = {'dark_subtraction': False,
                            'baseline': baseline, # baseline has to be applied
                                                  # before any masking
                            'masking': True,
                            'internal_masking': apply_internal_mask,
                            # 'dropletize': dropletize,
                            }
        # corrections applied on each worker
        self.worker_corrections = {'asic_commonmode': asic_commonmode,
                                   'cell_commonmode': cell_commonmode,
                                   'dropletize': dropletize}

        #: DataArray: pixel mask as xarray.DataArray
        self.xmask = xr.DataArray(self.mask,
                dims=('module', 'dim_0', 'dim_1'),
                coords={'module': np.arange(16),
                        'dim_0': np.arange(512),
                        'dim_1': np.arange(128)})

        self.is_proc = None
        #: (np.ndarray): average dark over trains
        self.avr_dark = None
        #: (np.ndarray): mask calculated from darks
        self.dark_mask = None
        #: str: file with dark data.
        self.darkfile = None
        # setting the dark run also sets the previous dark attributes
        self.dark_run_number = dark_run_number

        #: (np.ndarray): flatfield data
        self.flatfield = None
        #: (np.ndarray): mask calculated from flatfields
        self.flatfield_mask = None
        #: str: file with flatfield data.
        self.flatfieldfile = None
        # setting the flatfield run also sets the previous attributes
        self.flatfield_run_number = flatfield_run_number

        self.stripes = stripes

        # Darks will not be calibrated (overwrite configfile)
        if self.is_dark or self.is_flatfield:
            for correction in ['corrections', 'worker_corrections']:
                correction_dict = getattr(self, correction)
                for key in correction_dict:
                    correction_dict[key] = False

        #: DataAarray: the run AGIPD data
        self.data = None


    def __getstate__(self):
        """needed for apply dask.apply_along_axis

        We return only those attributes needed by the worker functions
        """
        attrs = ['adu_per_photon', 'worker_corrections', 'subshape', 'mask']
        return {attr: getattr(self, attr) for attr in attrs}


    @property
    def dark_run_number(self):
        """The run number and the file index to load the average dark"""
        return self.__dark_run_number


    @dark_run_number.setter
    def dark_run_number(self, number):
        if number is None:
            pass
        elif len(number) == 2:
            self.darkfile = f"r{number[0]:04}-dark_{number[1]:03}.h5"
            with h5.File(self.darkfile, 'r') as f:
                avr_dark = f['dark/intensity'][:]
                pulse_ids = f['identifiers/pulse_ids'][:].flatten()
                self.dark_mask = self.xmask.copy(data=f['dark/mask'][:])
            xdark = xr.DataArray(avr_dark,
                    dims=('pulseId', 'module', 'dim_0', 'dim_1'),
                    coords={'pulseId': pulse_ids,
                            'module': np.arange(16),
                            'dim_0': np.arange(512),
                            'dim_1': np.arange(128)})
            xdark = xdark.transpose('module', 'dim_0', 'dim_1', 'pulseId')
            self.avr_dark = xdark
            self.is_proc = False
            self.corrections['dark_subtraction'] = True
            # internal mask available only in processed data
            self.corrections['internal_masking'] = False
        else:
            raise ValueError("Dark input parameter could not be processed:\n"
                             f"{number}")
        self.__dark_run_number = number


    @property
    def flatfield_run_number(self):
        """The run number and the file index to load the flatfield"""
        return self.__flatfield_run_number


    @flatfield_run_number.setter
    def flatfield_run_number(self, number):
        if number is None:
            pass
        elif len(number) == 2:
            self.flatfieldfile = f"r{number[0]:04}-flatfield_{number[1]:03}.h5"
            with h5.File(self.flatfieldfile, 'r') as f:
                flatfield = f['flatfield/intensity'][:]
                pulse_ids = f['identifiers/pulse_ids'][:].flatten()
                self.flatfield_mask = self.xmask.copy(data=f['flatfield/mask'][:])
            xflatfield = xr.DataArray(flatfield,
                            dims=('pulseId', 'module', 'dim_0', 'dim_1'),
                            coords={'pulseId': pulse_ids,
                                    'module': np.arange(16),
                                    'dim_0': np.arange(512),
                                    'dim_1': np.arange(128)})
            xflatfield = xflatfield.transpose('module', 'dim_0', 'dim_1', 'pulseId')
            self.flatfield = xflatfield
        else:
            raise ValueError("Flatfield input parameter could not be processed:\n"
                             f"{number}")
        self.__flatfield_run_number = number


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
            if self.dark_mask is not None:
                print("Combining dark-mask and stripe-mask")
                stripes &= self.dark_mask.values
        elif stripes is None:
            pass
        else:
            raise TypeError(f'Invalid dark run number type {type(number)}.')
        self.__stripes = stripes


    def _get_data(self):
        """Read data and apply corrections

        Args:
            train_step (int): step between trains for slicing.

            last_train (int): last train index to process.

        Return:
           DataArray: of the run data.

        """

        print("Requesting data...", flush=True)
        self._agp = AGIPD1M(self.run, min_modules=16)
        arr = self._agp.get_dask_array('image.data')
        print("Loaded dask array.", flush=True)

        # coords are module, dim_0, dim_1, trainId, pulseId after unstack
        arr = arr.unstack()
        self.is_proc = True if len(arr.dims) == 5 else False
        print(f"Is processed: {self.is_proc}")

        arr = self._slice(arr)

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


    def _slice(self, arr):
        """Select trains and pulses."""
        arr = arr.sel(trainId=self.train_ids).sel(pulseId=self.cell_ids)
        if not self.is_proc:
            arr = arr.rename({'dim_1': 'dim_0',
                              'dim_2': 'dim_1'})
            # slicing dark should not be neccessary as xarrays
            # broadcasting takes coords into account
            #if self.avr_dark is not None:
            #    # first cell has been cut when averaging dark
            #    dark_slice = slice(
            #            self.first_cell - 1,
            #            (self.first_cell - 1 +
            #                self.pulses_per_train * self.pulse_step),
            #            self.pulse_step)
            #    self.avr_dark = self.avr_dark[..., dark_slice]
        return arr


    def _dark_subtraction(self, arr):
        """Subtract average darks from train data."""
        arr = arr.unstack()
        arr = arr - self.avr_dark
        arr = arr.stack(train_pulse=('trainId', 'pulseId'))
        arr = arr.transpose('train_pulse', ...)
        return arr


    def _masking(self, arr):
        """Mask bad pixels with provided use mask."""
        if self.dark_mask is not None:
            print('using dark mask')
            arr =  arr.where(self.xmask & self.dark_mask)
            if self.flatfield_mask is not None:
                print('using flatfield mask')
                arr = arr.where(self.flatfield_mask)
            return arr
        else:
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
        # if np.sum(np.isfinite(module_median)).compute() == 0:
        #     warnings.warn("All module medians are nan. Check masks")
        module_median = module_median.where((np.isfinite(module_median) &
                                 (module_median < .5*self.adu_per_photon)), 0)
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
        nrows, ncols = self.subshape
        nv = 512 // nrows
        nh = 128 // ncols // 2
        nhl = sorted(set([x * y for x in [-1, 1] for y in range(1, 1+nh)]))

        arr = arr.unstack()
        arr = arr.stack(train_pulse_module=['trainId', 'pulseId', 'module'])
        arr = arr.chunk({'train_pulse_module': 1024})

        arr = arr.assign_coords(asc_0=('dim_0',
                                       np.repeat(np.arange(1,nv+1), nrows)),
                                  asc_1=('dim_1', np.repeat(nhl, ncols)))
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


def _create_asic_mask():
    """Mask asic borders."""
    asic_mask = np.ones((512, 128))
    asic_mask[:3, :] = 0
    asic_mask[-3:, :] = 0
    asic_mask[:, -3:] = 0
    asic_mask[:, :3] = 0
    asic_mask[:, 62:66] = 0
    for i in range(1,9):
        c = 64*i
        asic_mask[c-2:c+2, :] = 0
    asic_mask = np.rollaxis(np.dstack([asic_mask]*16), axis=-1)
    return asic_mask.astype('bool')


def _create_ta_mask():
    """Mask Ta stripes."""
    ta_mask = [np.zeros((512, 128)) for _ in range(4)]
    ta_mask[0][41:47, :] = 1
    ta_mask[1][103:109, :] = 1
    ta_mask[2][41:47, :] = 1
    ta_mask[3][95:101, :] = 1
    ta_mask = np.rollaxis(np.repeat(np.dstack(ta_mask), 4, axis=2), axis=-1)
    return ta_mask.astype('bool')


def _create_mask_from_dark(dark_counts, dark_variance, pvals=(.2, .5)):
    """Use dark statistics to make a mask."""
    darkmask = _create_asic_mask()
    ndarks = dark_counts.shape[0]
    median_list = []
    for idark in range(ndarks):
        for ii, (image, p) in enumerate(
            zip([dark_counts[idark], dark_variance[idark]/dark_counts[idark]],
                pvals)):
            nmean = np.nanmean(image[darkmask])
            nstd = np.nanstd(image[darkmask])
            darkmask[(image<(nmean*(1-p))) | (image>(nmean*(1+p)))] = 0
            median_list.append(nmean)
    median_list = np.vstack((median_list[::2], median_list[1::2]))
    return darkmask.astype('bool'), median_list


def _create_mask_from_flatfield(counts, variance, average_limits=(4000, 5800),
        variance_limits=(200, 1500)):
    """Use flatfield data to make a mask."""
    ffmask = _create_asic_mask()
    medians = np.nanmedian(counts[:, ffmask], axis=1)
    medians_var = np.nanmedian(variance[:, ffmask], axis=1)
    counts = counts.mean(0)
    variance = variance.mean(0)
    ffmask[(counts < average_limits[0]) |
           (counts > average_limits[1])] = 0
    ffmask[(variance < variance_limits[0]) |
           (variance > variance_limits[1])] = 0
    return ffmask.astype('bool'), np.vstack((medians, medians_var))
