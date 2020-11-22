import numpy as np
from .worker_functions import _to_asics


def mask_asics(mask):
    asics = _to_asics(mask).reshape(-1, 64, 64)
    asics *= (asics.sum(-1) > (64 * 0.50))[:, :, None]
    asics *= (asics.sum(-2) > (64 * 0.50))[:, None, :]

    for d in range(2):
        hl = list(np.where(asics.sum(-(d + 1)) == 0))
        dhl = np.diff(hl[1])
        tmp = hl[1].copy()
        hl2 = list(hl)
        for i in np.where((dhl > 1) & (dhl < 8))[0]:
            hl2[1] = np.append(hl2[1], hl[1][i + 1] - np.arange(4))
            hl2[0] = np.append(hl2[0], np.ones(4) * hl[0][i + 1])
        ind = hl2[1] >= 0
        hl2 = [x[ind].astype(int) for x in hl2]
        if d == 0:
            asics[hl2[0], hl2[1]] = 0
        else:
            asics[hl2[0], :, hl2[1]] = 0

    asics[asics.sum((1, 2)) < (64 ** 2 / 10)] = 0
    return _to_asics(asics, reverse=True)


def mask_radial(arr, rmap, mask=None, upper_quantile=0.99, lower_quantile=0, nbins=32):
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
        if not sum(roi):
            continue
        thres = [-100_000, np.quantile(m0[roi], upper_quantile)]
        if lower_quantile > 0:
            thres[0] = np.quantile(m0[roi], lower_quantile)
        mask[roi] = (m0[roi] > thres[0]) & (m0[roi] < thres[1])

    return mask.reshape(16, 512, 128).astype(bool)
