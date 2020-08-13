import numpy as np
from extra_data.components import AGIPD1M
from .average import average
import dask.array as da


class Calibrator:
    """Calibrate AGIPD dataset"""

    def __init__(self, run, last_train=10_000, pulses_per_train=100,
            first_cell=1, train_step=1, pulse_step=1, dark_run_number=None,
            apply_internal_mask=False, dropletize=False, stripes=None,
            baseline=False,):

        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = run
        self.last_train = last_train
        self.pulses_per_train = pulses_per_train
        self.first_cell = first_cell
        self.train_step = train_step
        self.pulse_step = pulse_step

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
                            'dropletize': dropletize,
                            'baseline': baseline}


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
        elif stripes is None:
            pass
        else:
            raise TypeError(f'Invalid dark run number type {type(number)}.')
        self.corrections['baseline'] = True
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

        arr = arr.stack(train_pulse=('trainId', 'pulseId'),
                        pixels=('module', 'dim_0', 'dim_1'))
        arr = arr.transpose('train_pulse', 'pixels')

        for correction, options in self.corrections.items():
            if bool(options):
                getattr(self, "_" + correction)()

                # apply common mode correction
                # dim = arr.get_axis_num("pixels")
                # arr = da.apply_along_axis(commonmode_frame, dim, arr.data)
        return arr


    def _dropletize(self, arr):
        """Convert adus to photon counts."""
        arr = arr.ufunc.floor(
            (arr.data + 0.5*self.adu_per_photon) / self.adu_per_photon)
        arr[arr < 0] = 0
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

        arr = arr.where((mask_int.data <= 0) | (mask_int.data>=8))
        return arr


#       def _baseline(self, arr):
#           """Baseline correction based on Ta stripes"""
#           arr = arr.unstack()
#           def func(a, module_number):
#               stripe = self.stripes[module_number] == 1
#               median = np.median(a[stripe])
#                -= e[np.argmax(h)]
#           tmp = da.apply_along_axis(func,)


    @staticmethod
    def _module2asics(data, reverse=False):
        """convert module to asics and vise versa"""
        nrows = ncols = 64
        if data.ndim == 3:
            if not reverse:
                return data.reshape(-1, 8, nrows, 2, ncols).swapaxes(2,3).reshape(-1, 16, nrows, ncols)
            else:
                return data.reshape(-1, 8, 2, nrows, ncols).swapaxes(2,3).reshape(-1,512,128)
        elif data.ndim == 2:
            if not reverse:
                return data.reshape(8, nrows, 2, ncols).swapaxes(1,2).reshape(16, nrows, ncols)
            else:
                return data.reshape(8, 2, nrows, ncols).swapaxes(1,2).reshape(512,128)


    def _commonmode_module(self, module, range_=(-30, 31)):
        """find the zero photon peak and center it around zeros."""
        asics = self.module2asics(module)
        for asic in asics:
            h, e = np.histogram(asic, bins=np.arange(*range_))
            asic -= e[np.argmax(h)]
        return self.module2asics(asics, reverse=True)


    def _commonmode_frame(self, frame):
        """apply commonmode correction to each module of a frame"""
        print(frame)
        is_flat = True if frame.ndim == 1 else False
        if is_flat:
            frame = frame.reshape(16, 512, 128)
        for i, module in enumerate(frame):
            frame[i] = self.commonmode_module(module)
        if is_flat:
            frame = frame.flatten()
        return frame


    def _commonmode_series(self, series):
        """apply commonmode to all frames of a train"""
        for i, frame in enumerate(series):
            series[i] = self.commonmode_frame(frame)
        return series

