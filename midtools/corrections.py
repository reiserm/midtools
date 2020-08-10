import numpy as np
from extra_data.components import AGIPD1M


class Calibrator:
    """Calibrate AGIPD dataset"""

    def __init__(self, run, corrections, last=10_000):

        self.run = run
        #: dict: Dictionary of dictionaries. The first level keys are the
        #        corrections to be applied. The options are given by the second
        #        level.
        self.corrections = corrections
        self.is_pros = None
        self.last = last


    def __call__(self, train):
        """Apply corrections per train"""
        if bool(self.dark):
            assert dark.shape == train.shape
            train -= dark
        if bool(self.commonmode):
            train = commonmode_series(series)
        return train


    def _get_data(self, ):

        print("Requesting data...", flush=True)
        agp = AGIPD1M(run, min_modules=16)
        arr = agp.get_dask_array('image.data')
        print("Got dask array", flush=True)

        # coords are module, dim_0, dim_1, trainId, pulseId after unstack
        arr = arr.unstack()

        # take maximum 200 trains for the simple average
        # skip trains to get max_trains trains
        if method == 'average':
            last = min(200, last)
            train_step = 1
        elif method == 'single':
            train_step = (last // max_trains) + 1

        # select pulses and skip the first one
        arr = arr[..., :last:train_step, first_cell:npulses+first_cell]
        npulses = arr.shape[-1]

        if to_counts:
            arr.data = np.floor((arr.data + 0.5*adu_per_photon) / adu_per_photon)
            arr.data[arr.data<0] = 0
            # arr.data.astype('float32')

        if apply_internal_mask:
            mask_int = agp.get_dask_array('image.mask')
            mask_int = mask_int.unstack()
            mask_int = mask_int[..., :last, first_cell:npulses+first_cell]
            arr = arr.where((mask_int.data <= 0) | (mask_int.data>=8))
            # mask[(mask_int.data > 0) & (mask_int.data<8)] = 0

        arr = arr.stack(train_pulse=('trainId', 'pulseId'),
                        pixels=('module', 'dim_0', 'dim_1'))
        arr = arr.transpose('train_pulse', 'pixels')
        print(arr)

        # apply common mode correction
        dim = arr.get_axis_num("pixels")
        arr = da.apply_along_axis(commonmode_frame, dim, arr.data)


    @staticmethod
    def module2asics(data, reverse=False):
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


    def commonmode_module(self, module, range_=(-30, 31)):
        """find the zero photon peak and center it around zeros."""
        asics = self.module2asics(module)
        for asic in asics:
            h, e = np.histogram(asic, bins=np.arange(*range_))
            asic -= e[np.argmax(h)]
        return self.module2asics(asics, reverse=True)


    def commonmode_frame(self, frame):
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


    def commonmode_series(self, series):
        """apply commonmode to all frames of a train"""
        for i, frame in enumerate(series):
            series[i] = self.commonmode_frame(frame)
        return series

