import numpy as np


def mask_radial(arr, rmap, mask=None, upper_quantile=0.99, nbins=32):
    """This function masks pixels basted on the statistics in radial rois.

    Concretely, it considers logarithmically spaced concentric rings and masks outliers
    based on the value of upper_quantile.
    """

    if mask is None:
        mask = np.ones(16 * 512 * 128)

    mask = mask.flatten().astype(bool)
    m0 = arr.reshape(-1, 16 * 512 * 128).mean(0)
    rmap = rmap.flatten()
    r_min, r_max = rmap.min(), rmap.max()
    rrng = np.logspace(np.log10(r_min), np.log10(r_max), nbins)

    for ri in range(len(rrng) - 1):
        roi = (rmap > rrng[ri]) & (rmap < rrng[ri + 1]) & mask & np.isfinite(m0)
        thres = np.quantile(m0[roi], upper_quantile)
        mask[roi] = m0[roi] < thres

    return mask.reshape(16, 512, 128).astype(bool)
