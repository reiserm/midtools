import numpy as np

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


def commonmode_module(module, range_=(-30, 31)):
    """find the zero photon peak and center it around zeros."""
    asics = module2asics(module)
    for asic in asics:
        h, e = np.histogram(asic, bins=np.arange(*range_))
        asic -= e[np.argmax(h)]
    return module2asics(asics, reverse=True)


def commonmode_frame(frame):
    """apply commonmode correction to each module of a frame"""
    print(frame)
    is_flat = True if frame.ndim == 1 else False
    if is_flat:
        frame = frame.reshape(16, 512, 128)
    for i, module in enumerate(frame):
        frame[i] = commonmode_module(module)
    if is_flat:
        frame = frame.flatten()
    return frame


def commonmode_series(series):
    """apply commonmode to all frames of a train"""
    for i, frame in enumerate(series):
        series[i] = commonmode_frame(frame)
    return series
