import os
import re
import numpy as np
from functools import reduce
from collections import namedtuple
from functools import wraps
import matplotlib as mpl
from functools import reduce
from collections import namedtuple
from functools import wraps
import xarray as xr

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
import seaborn as sns

from scipy.stats import sem
from Xana.XpcsAna.pyxpcs3 import pyxpcs
from Xana.Xfit.fitg2global import G2
from Xana.Xfit.fit_basic import fit_basic

import dask.array as da

from .correlations import convert_ttc

import pdb



def calc_speckle_contrast(pixel_size, speckle_size):
    """Calculate the speckle size in micro meters.

    Args:
        pixel_size (float): pixel size in micro meters.
        beam_size (float): beam size in micro meters.

    Returns:
        float: speckle contrast.
    """
    return 1 / (1 +  (pixel_size / speckle_size)**2)


def calc_speckle_size(beam_size, distance=5, energy=9):
    """Calculate the speckle size in micro meters.

    Args:
        beam_size (float): beam size in micro meters.
        distance (float, optional): sample detector distance in meters.
        energy (float, optional): photon energy in kilo electronvolt.

    Returns:
        float: speckle size in micro meters.
    """
    return distance * 12.39 / energy * 1e-10 / beam_size * 1e12


def calc_snr(contrast, intensity,  pixels=1, patterns=1, repetitions=1):
    """Calculate the XPCS signal to noise ratio

    Args:
        contrast (float): Speckle contrast.
        intensity (float): Average number of photons per pixel.
        pixels (int, optional): Number of pixels in ROI.
        patterns (int, optional): Number of images.
        repetitions (int, optional): Number of repetitions.

    Returns:
        float: the signal to noise ratio.
    """
    return contrast * intensity * np.sqrt(pixels * patterns * repetitions)


def calc_flux(energy, attenuation=1., photon_energy=9,
              T_lenses=0.586, T_apperture=0.75, T_air=0.39):
    """"Calculate the photon flux at MID

    Args:
        energy (float): pulse energy in micro Joule.

        attenuation (float, optional): attenuation factor through absorbers.
            a values of `1` would correspond to 100\% transmission.

        photon_energy (float, optional): photon energy in keV.

        T_lenses (float, optional): transmission of lenses.

        T_apperture (float, optional): transmission of apperture.

    Returns:
        Number of photons per pulse.

    """
    h_planck = 6.626e-34
    wavelength = 12.39 / photon_energy * 1e-10
    photon_energy_J = h_planck * 3e8 / wavelength
    energy_J = energy * 1e-6
    photons_per_pules = energy_J / photon_energy_J
    return photons_per_pules  * attenuation * T_lenses * T_apperture * T_air


def calc_dose(flux, npulses=1, photon_energy=9, absorption=np.exp(-1),
             volume_fraction=1., beam_size=10, thickness=1.5,
             density=1.):
    """"Calculate the photon flux at MID

    Args:
        flux (float): photon flux per pulse

        npulses (int, optional): number of pulses.

        photon_energy (float, optional): photon energy in keV.

        absorption (float, optional): sample absorption.

        volume_fraction (float, optional): volume fraction.

        beam_size (float, optional): beam_size in micro meters.

        thickness (float, optional): sample thickness in milli meters.

        density (float, optional): sample mass density in gram per cubic
            centimeter.

    Returns:
        Absorbed dose in kGy.

    """
    h_planck = 6.626e-34
    wavelength = 12.39 / photon_energy * 1e-10
    photon_energy_J = h_planck * 3e8 / wavelength
    total_energy_J = photon_energy_J * flux * npulses
    return (total_energy_J * absorption * volume_fraction /
            ((beam_size*1e-6)**2 * thickness*1e-3 * density*1e3) / 1000)


def find_percent(x):
    """Find percent in given str and return as float"""
    l = re.findall("\d{,3}\.{,1}\d{1,3}(?=\%)", x)
    if len(l) > 1:
        print("Found more than one value. Consider only first occurrence.")
    elif len(l) == 0:
        return np.nan
    return float(l[0])


def find_size(x):
    """Find size of nanoparticles in given str and return as float"""
    l = re.findall("\d{1,}\s*(?=nm)", x)
    if len(l) > 1:
        print("Found more than one value. Consider only first occurrence.")
    elif len(l) == 0:
        return np.nan
    return float(l[0])


def is_processed(run, path='/gpfs/exfel/exp/MID/202022/p002693/'):
    path = path.rstrip('/proc/') + '/proc/'
    proc_folders = os.listdir(path)
    if f"r{run:04}" in proc_folders:
        return True
    else:
        return False


def plot_xgm(run):
    """Plot the train resolved XGM data"""

    # get data as pandas dataframe
    df = run.get_dataframe(fields=[("*_XGM/*", "*.photonFlux"),])
    df.reset_index(inplace=True)
    col = list(filter(lambda x: 'photonFlux' in x, df.columns))[0]

    # make the plot
    fig, ax = subplots() # make a figure
    df.plot(y=col, ax=ax, label='photon flux', legend=False)
    ax.set_xlabel('train index')
    ax.set_ylabel('photon flux ($\mu J$)')
    ax.set_title(f"run {run_id}")
    ax.minorticks_on()


def plot_xgm_pr(run, last_pulse=None, train_step=100):
    # the first argument is the data source, the second one the attribute
    arr = run.get_array('SA2_XTD1_XGM/XGM/DOOCS:output', 'data.intensityTD')

    if last_pulse is None:
        last_pulse = arr.shape[-1]

    pulse_intensities = arr[::train_step,:last_pulse]
    train_ids = run.train_ids[:last_pulse:train_step]

    with sns.color_palette('Reds_d'):
        plot(pulse_intensities.T, alpha=.7)
    xlabel('pulse index')
    ylabel('intensity (uJ)')

    gca().minorticks_on()


def get_run_number(x):
    return int(re.findall('(?<=r)\d{4}', x)[0])


def check_contents(filename, return_h5structure=False):
    h5_structure = {
           'META': {
                "/identifiers/pulses_per_train": [None],
                "/identifiers/pulse_ids": [None],
                "/identifiers/train_ids": [None],
                "/identifiers/train_indices": [None],
            },
            'DIAGNOSTICS': {
                "/pulse_resolved/xgm/energy": [None],
                "/pulse_resolved/xgm/pointing_x": [None],
                "/pulse_resolved/xgm/pointing_y": [None],
            },
            'SAXS': {
                "/pulse_resolved/azimuthal_intensity/q": [None],
            },
            'XPCS': {
                "/train_resolved/correlation/q": [None],
                "/train_resolved/correlation/t": [None],
            },
            'FRAMES': {
                "/average/intensity": [None],
            },
            'STATISTICS': {
                "/pulse_resolved/statistics/centers": [None],
            },
        }
    if return_h5structure:
        return h5_structure
    else:
        content = {}
        with h5.File(filename, 'r') as f:
            for key, datasets in h5_structure.items():
                content[key] = True
                for dataset in datasets:
                    try:
                        if dataset in f:
                            continue
                        else:
                            content[key] = False
                            break
                    except Exception as e:
                        print(str(e))
                        content[key] = False
        return content


def is_complete(filename):
    return all(list(check_contents(filename).values()))


def visit_h5(filename):
    def func(arg):
        """print shape of datasets"""
        if isinstance(f[arg], h5.Group):
            print("Group:", arg)
        elif isinstance(f[arg], h5.Dataset):
            print(f"  - {arg} dataset with shape\n    {f[arg].shape}")
        else:
            pass
    with h5.File(filename, 'r') as f:
        f.visit(func)


def get_datasets(directory):
    """Search a folder for datasets processed by midtools

    Args:
        directory (str): directory to search.

    Returns:
        namedtuple: `Datasets` that can be passed to the Dataset class. It
            contains lists of HDF5-files, run numbers, indices, time-stamps.

    Examples:
        Use the output like `Dataset(datasets)`.

    """
    Datasets = namedtuple('Datasets', ['file',
                                       'run',
                                       'index',
                                       'time',
                                       'complete'])
    filenames = list(filter(lambda x: bool(re.search('\d{3,10}.h5', x)),
                            os.listdir(directory)))

    filenames = [directory + x for x in filenames]
    filenames.sort(key=os.path.getmtime)
    mtimes = [os.path.getmtime(x) for x in filenames]
    runs = [get_run_number(x) for x in filenames]
    state = [is_complete(x) for x in filenames]
    indices = list(map(lambda x: int(re.search('\d{3}(?=\.h5)', x).group(0)),
                       filenames))

    datasets = Datasets(filenames, runs, indices, mtimes, state)
    return datasets


def _run_to_filename(func):
    @wraps(func)
    def wrapper(obj, runid, index, *args, **kwargs):
        filename = obj._get_dataset_file(runid, index)
        setupfile = filename.replace('analysis', 'setup').replace('h5', 'yml')
        obj.config = None #Dataset._read_setup(setupfile)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{filename}\ndoes not exiist")
        return func(obj, filename, *args, **kwargs)
    return wrapper


def autocorr(x):
    x = (x - np.mean(x)) / np.std(x)
    result = np.correlate(x, x, mode='full') / len(x)
    return result[result.size//2:]


def rescale(y, mn, mx, rng=(0, 1)):
    p = (rng[1]-rng[0])/(mx-mn)
    return p * (y - mn) + rng[0], p


def add_colorbar(ax, vec, label=None, cmap='magma', discrete=False,
                 tick_step=1, qscale=0, location='right', show_offset=False,
                 **kwargs):

    ncolors = len(vec)
    tick_indices = np.arange(0, len(vec), tick_step)
    vec = np.array(vec)[tick_indices]
    vec *= 10**qscale
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)

    change_ticks = False
    if discrete:
        norm = mpl.colors.NoNorm()
        change_ticks = True
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap, ncolors)
        elif isinstance(cmap, (list, np.ndarray)):
            cmap = mpl.colors.ListedColormap(cmap)
    else:
        cmap = plt.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vec.min(), vmax=vec.max())


    if location == 'right':
        orientation = 'vertical'
    elif location == 'top':
        orientation = 'horizontal'

    cax = mpl.colorbar.make_axes(ax, location=location, **kwargs)[0]
    cb = mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap,
                                   orientation=orientation)

    # set up color bar ticks
    if location == 'top':
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')

    cb.set_label(label)
    if change_ticks:
        cb.set_ticks(tick_indices)
        cb.set_ticklabels(list(map(lambda x: "$%.{}f$".format(3-qscale) % x, vec)))
        if qscale and show_offset:
            cb.ax.text(1.,1.04, r'$\times 10^{{-{}}}$'.format(qscale),
                        transform=cb.ax.transAxes)
    cb.ax.invert_yaxis()
    cb.ax.set_in_layout(True)


class Interpreter:
    """Class to explore MID datasets.
    """
    def __init__(self, datasets, metadata=None, geom=None, mask=None, max_len=100,
            **kwargs):
        self.datasets = datasets
        self.metadata = metadata
        self.run = None, 0
        self.xgm = None
        self.geom = geom
        self.mask = mask
        self.pulses = None
        self.trains = None
        self.image = None
        self.frames = None
        self.qI = None
        self.qv = None
        self.g2 = None
        self.info = None
        self.max_len = max_len
        self.config = None
        self.kwargs = kwargs


    def iter_trainids(self, run, subset=None):
        indices = np.array(self.datasets.index)[np.where(np.array(self.datasets.run) == run)[0]]
        if subset is not None:
            indices = np.intersect1d(indices, subset)
        for index in indices:
            yield index, self.load_identifier(run, index)[2]


    def iter_files(self, run, subset=None):
        indices = np.array(self.datasets.index)[np.where(np.array(self.datasets.run) == run)[0]]
        files = np.array(self.datasets.file)[np.where(np.array(self.datasets.run) == run)[0]]
        if subset is not None:
            indices, in_list = np.intersect1d(indices, subset, return_indices=True)[:2]
        for index, ind in zip(indices, in_list):
            yield index, files[ind]


    def __str__(self,):
        if self.info is None:
            return "MID Dataset"
        else:
            return self.info


    def _get_dataset_file(self, runid, index=None):
        if index is None:
            cond = np.array(self.datasets.run) == runid
        else:
            cond = ((np.array(self.datasets.run) == runid)
                         & (np.array(self.datasets.index) == index))
        return self.datasets.file[np.where(cond)[0][0]]

    @property
    def run(self):
        return self.__run


    @run.setter
    def run(self, identifier):
        run, index = identifier
        self.__run = run
        if isinstance(run, int):
            self.info =  f"RUN: {self.run}({index})\n"
            if self.metadata is not None:
                if self.run in self.metadata.index:
                    meta = self.metadata.loc[self.run]
                    cols = ['sample', 'att (%)']
                    self.info += "\n".join([col + f": {meta[col]}" for col in cols])
                else:
                    pass


    def _rebin(self, arr, avr=False):
        """rebin along first axis and return rebinned array and bin size"""
        new_len = self.max_len
        if arr.shape[0] > new_len:
            old_shape = arr.shape
            rest = (arr.shape[0]%new_len)
            if rest:
                arr = arr[:-rest]
            arr = arr.reshape(new_len, -1, *old_shape[1:])
            binned = arr.shape[1]
            if avr:
                arr = np.nanmean(arr, axis=1)
            else:
                arr = arr[:, 0, ...]
            return arr, binned
        else:
            return arr, 1


    @_run_to_filename
    def load_identifier(self, filename):
        keysh5 = ["identifiers/pulse_ids",
                "identifiers/train_indices",
                "identifiers/train_ids",
                "identifiers/all_trains"]

        args = ['pulseIds', 'train_indices', 'trainIds', 'all_trains']
        out = {arg: None for arg in args}
        with h5.File(filename, 'r') as f:
            for key, arg in zip(keysh5, out.keys()):
                if key in f:
                    data = np.array(f[key])
                    out[arg] = data
                    setattr(self, arg, data)

        return list(out.values())


    @_run_to_filename
    def load_xgm(self, filename):
        with h5.File(filename, 'r') as f:
            xgm = f["pulse_resolved/xgm/energy"][:]
            pulses = f["identifiers/pulse_ids"][:]
            trains = f["identifiers/train_indices"][:]
        self.xgm = xgm
        self.pulses = pulses
        self.trains = trains
        return xgm, pulses, trains


    def plot_xgm(self, run, index=0, rebin_kws=None):
        """Plot the train resolved XGM data"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get('rebin_kws', {})

        self.run = run, index
        data, pulses, trains = self.load_xgm(run, index=index)
        data, tstep = self._rebin(data, **rebin_kws)
        trains = np.arange(data.shape[0]) * tstep

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9,4), sharey=True,
                                constrained_layout=True)
        axs = np.ravel(axs)

        xlabels = ['pulse index', 'train index']
        colors = sns.color_palette('Blues_d', trains.size, desat=1)

        for i, (ax, xl) in enumerate(zip(axs, xlabels)):
            if i == 0:
                for t, color in zip(data, colors):
                    ax.plot(pulses, t, color=color, alpha=.4)
                ax.set_ylabel('photon flux ($\mu J$)')
            if i == 1:
                ax.plot(trains, np.mean(data, 1))
            ax.set_xlabel(xl)
            ax.set_title(f"run {run}, {len(pulses)} pulses per train")
            ax.minorticks_on()
        fig.suptitle(self.info.replace('\n', ', '), fontsize=16)


    @_run_to_filename
    def load_images(self, filename):
        keysh5 = ["average/intensity",
                  "average/variance"]

        args = ['frames', 'variance']
        out = {arg: None for arg in args}
        with h5.File(filename, 'r') as f:
            for key, arg in zip(keysh5, out.keys()):
                if key in f:
                    data = np.array(f[key])
                    out[arg] = data
                    setattr(self, arg, data)

        return list(out.values())


    def plot_images(self, run, index=0, vmin=None, vmax=None, log=True):
        """Plot average image and some frames"""

        self.run = run, index
        frames = self.load_images(run, index=index)[0].reshape(-1, 16, 512, 128)
        if self.mask is not None:
            frames[:, ~self.mask] = np.nan
        avr = frames.mean(0)
        avr[avr <= 0] = 1e-3

        # make the plot
        fig = plt.figure(figsize=(12,8), constrained_layout=True)
        gs_main = fig.add_gridspec(4, 8)

        ax_main = fig.add_subplot(gs_main[:, :4])
        subax = [fig.add_subplot(gs_main[i:i+2, j:j+2])
                 for i in range(0,4,2) for j in range(4,8,2)]

        if log:
            norm = LogNorm()
        else:
            norm = None
        im = ax_main.imshow(self.geom.position_modules_fast(avr)[0], cmap='magma',
                            norm=norm, vmin=vmin, vmax=vmax)
        ax_main.set_title('average trains')

        for i, (ax, frame) in enumerate(zip(subax, frames)):
            if self.mask is not None:
                frame[~self.mask] = np.nan
            if self.geom is not None:
                slc = slice(400,-400)
                frame = self.geom.position_modules_fast(frame)[0][slc, slc]
            else:
                frame = frame.reshape(-1, 1024)

            ax.imshow(frame, norm=LogNorm(), vmin=vmin, vmax=vmax,
                    cmap='magma')

            ax.set_title(f"train {i}")
        plt.colorbar(im, ax=subax[1::2], shrink=.5, label='intensity')
        fig.suptitle(self.info.replace('\n', ', '), fontsize=16)


    @_run_to_filename
    def load_azimuthal_intensity(self, filename):
        with h5.File(filename, 'r') as f:
            q = f["pulse_resolved/azimuthal_intensity/q"][:]
            I = f["pulse_resolved/azimuthal_intensity/I"][:]
            trains = f["identifiers/train_indices"][:]
        self.qI = (q, I)
        self.trains = trains
        return q, I, np.repeat(trains, I.shape[0]/trains.size)


    def plot_azimuthal_intensity(self, run, index=0, rebin_kws=None, vmin=None, vmax=None):
        """Plot the pulse resolved azimuthal intensity"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get('rebin_kws', {})

        self.run = run, index
        q, I, trains = self.load_azimuthal_intensity(run, index=index)
        I, pstep = self._rebin(I, **rebin_kws)
        pulses = np.arange(I.shape[0]) * pstep

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9,4),
                                constrained_layout=True)
        axs = np.ravel(axs)

        if I.shape[0]:
            cmap = 'Reds'
            alpha = .4
        else:
            cmap = 'Set1'
            alpha = 1

        for i, (ax,) in enumerate(zip(axs,)):
            if i == 0:
                im = ax.pcolor(q, pulses, I, cmap='inferno', vmin=vmin, vmax=vmax)
                ax.set_ylabel(f'pulse step {pstep} '
                              f'average: {rebin_kws.get("avr", False)}')
                plt.colorbar(im, ax=ax, shrink=.6)
                ax.set_xscale('log')
            if i == 1:
                colors = sns.color_palette(cmap, I.shape[0], desat=.7)
                for i in range(I.shape[0]):
                    ax.loglog(q, I[i], color=colors[i], alpha=alpha)
                ax.set_ylabel('intensity (a.d.u.)')
                add_colorbar(ax, pulses, cmap=cmap, shrink=0.6,
                             label=(f'pulse step {pstep} '
                                f'average: {rebin_kws.get("avr", False)}'))
            ax.set_xlabel("q ($nm^{-1}$)")
            ax.set_ylim(np.nanmean(I[:,-50:]), None)
            ax.minorticks_on()
        fig.suptitle(self.info.replace('\n', ' | '), fontsize=16)


    @_run_to_filename
    def load_correlation_functions(self, filename):
        keysh5 = ["train_resolved/correlation/t",
                  "train_resolved/correlation/q",
                  "train_resolved/correlation/g2",
                  "/train_resolved/correlation/ttc",
                  "identifiers/train_indices"]

        args = ['t_cor', 'qv', 'g2', 'ttc', 'train_indices']
        out = {arg: None for arg in args}
        with h5.File(filename, 'r') as f:
            for key, arg in zip(keysh5, out.keys()):
                if key in f:
                    data = np.array(f[key])
                    out[arg] = data
                    setattr(self, arg, data)

        return list(out.values())


    def plot_correlation_functions(self, run, index=0, qval=0.1, g2_offset=0,
            ylim=(.98, None),  clim=(None, None), rebin_kws=None):
        """Plot correlation functions"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get('rebin_kws', {})

        self.run = run, index
        t, q, g2, ttc, trains = self.load_correlation_functions(run, index=index)
        qind = np.argmin(np.abs(q-qval))
        qval = q[qind]
        t = t*1e6
        g2, tstep = self._rebin(g2, **rebin_kws)
        trains = trains[:g2.shape[0]*tstep:tstep]
        g2 -= g2_offset

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9,4),
                                constrained_layout=True)
        axs = np.ravel(axs)

        if g2.shape[0]:
            cmap = 'Purples'
            alpha = .4
        else:
            cmap = 'Set1'
            alpha = 1

        for i, (ax,) in enumerate(zip(axs,)):
            if i == 0:
                im = ax.pcolor(t, trains, g2[...,qind],
                               cmap='inferno', vmin=clim[0], vmax=clim[1],)
                ax.set_xscale('log')
                ax.set_ylabel(f'train step {tstep} '
                              f'average: {rebin_kws.get("avr", False)}')
                ax.set_title(f'$q={qval:.3}\,nm^{{-1}}$')
                plt.colorbar(im, ax=ax, shrink=.6)
            if i == 1:
                colors = sns.color_palette(cmap, g2.shape[0], desat=.7)
                for i in range(g2.shape[0]):
                    ax.semilogx(t, g2[i, :, qind], color=colors[i],
                                alpha=alpha)
                ax.set_ylabel('$g_2$')
                ax.set_title(f'$q={qval:.3}\,nm^{{-1}}$')
                add_colorbar(ax, trains, cmap=cmap, shrink=0.6,
                     label=(f'train step {tstep} '
                            f'average: {rebin_kws.get("avr", False)}'))
                ax.set_ylim(ylim)
            ax.set_xlabel("t ($\mu s$)")
            ax.minorticks_on()
        fig.suptitle(self.info.replace('\n', ' | '), fontsize=16)


    @_run_to_filename
    def load_statistics(self, filename):
        tmp = np.zeros((500))
        count_path = '/pulse_resolved/statistics/counts'
        centers_path = 'pulse_resolved/statistics/centers'
        with h5.File(filename, 'r') as f:
            if count_path in f:
                counts = f[count_path][:]
            else:
                counts = np.ones(500)
            if centers_path in f:
                centers = f[centers_path][:]
            else:
                centers = np.ones(500)
            trains = f["identifiers/train_indices"][:]
        self.stats = (centers, counts)
        self.trains = trains
        return centers, counts, trains


    def plot_statistics(self, run, index=0, rebin_kws=None):
        """Plot correlation functions"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get('rebin_kws', {})

        self.run = run, index
        centers, counts, pulses = self.load_statistics(run, index=index)
        xgm = self.load_xgm(run, index=index)[0]

        agipd_pulse_intensity = counts.mean(1)
        xgm_pulse_intensity = xgm.flatten()
        train_pulse = np.arange(xgm_pulse_intensity.size)

        counts, pstep = self._rebin(counts, **rebin_kws)
        xgm = self._rebin(xgm, **rebin_kws)[0]
        pulses = np.arange(counts.shape[0]) * pstep

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9,4),
                                constrained_layout=True)
        axs = np.ravel(axs)

        if counts.shape[0]:
            cmap = 'Blues'
            alpha = .4
        else:
            cmap = 'Set1'
            alpha = 1
        for i, (ax,) in enumerate(zip(axs,)):

            if i == 0:
                #im = ax.scatter(xgm_pulse_intensity, agipd_pulse_intensity,
                #                c=train_pulse, cmap=cmap)
                ax.set_ylabel(f'agipd pulse intensity (a.d.u.)')
                ax.set_xlabel(f'xgm pulse intensity ($\mu$J)')

            if i == 1:
                colors = sns.color_palette(cmap, counts.shape[0], desat=.7)
                for cnt, col in zip(counts, colors):
                    pl = ax.semilogy(centers, cnt, color=col, alpha=alpha)
                add_colorbar(ax, pulses, cmap=cmap, shrink=0.6,
                             label=(f'pulse step {pstep} '
                                f'average: {rebin_kws.get("avr", False)}'))
                ax.set_xlabel('intensity (a.d.u.)')
                ax.set_ylabel("counts")

            ax.minorticks_on()
        fig.suptitle(self.info.replace('\n', ' | '), fontsize=16)


    @_run_to_filename
    def load_sample_sources(self, filename):
        h5keys = ["identifiers/train_indices",
                  "/train_resolved/sample_position/y",
                  "/train_resolved/sample_position/z",
                  "/train_resolved/linkam-stage/temperature"]
        keys = ['trains', 'y', 'z', 'T']

        data = {}
        with h5.File(filename, 'r') as f:
            for h5key, key in zip(h5keys, keys):
                if h5key in f:
                    data[key] = np.array(f[h5key][:])
                else:
                    data[key] = -42 * np.ones_like(data['trains'])
        return data


    def plot_sample_sources(self, run, index=0, rebin_kws=None):
        """Plot sample position and temperature"""

        self.run = run, index
        data = self.load_sample_sources(run, index=index)
        trains, y, z, T = list(data.values())

        yu, yc = np.unique(np.round(np.diff(y), 2), return_counts=True)
        yu = yu[(yc>2).nonzero()]
        yc = yc[(yc>2).nonzero()]
        zu, zc = np.unique(np.abs(np.round(np.diff(z), 3)), return_counts=True)
        ind = ((zc>10)&(np.abs(zu)<.4)).nonzero()
        zu = zu[ind]
        zc = zc[ind]
        print(f"total {len(trains)} trains")
        print(f"unique y steps (mm) {yu} occurred {yc}")
        print(f"unique z steps (mm) {zu} occurred {zc}")
        print(f"unique T steps (K) {np.unique(np.diff(T))}")
        print(f"area: {y.max()- y.min():.2f} x {z.max()- z.min():.2f} (y * z)")

        fig, axs = plt.subplots(2, 2, figsize=(6,6), constrained_layout=True)

        ax = axs.flat[0]
        ax.scatter(y, z, c=T, cmap='RdYlGn', alpha=.05)
        ax.set_xlabel('y coordinate')
        ax.set_ylabel('z coordinate')
        ax.set_title('Sample position')

        ax = axs.flat[1]
        t = f'LINKAM T'
        ax.scatter(trains, T, c=T, s=.1, cmap='RdYlGn', alpha=.05)
        ax.set_title(t)
        ax.set_ylabel('T (deg C)')

        ax = axs.flat[2]
        ax.plot(trains, z, '.')
        ax.set_xlabel('trains')
        ax.set_ylabel('z coordinate')

        ax = axs.flat[3]
        ax.plot(trains, y, '.')
        ax.set_xlabel('trains')
        ax.set_ylabel('y coordinate')

        for ax in axs.flat:
            ax.minorticks_on()

        fig.suptitle(self.info.replace('\n', ' | '), fontsize=16)


    def to_xDataset(self, run, index=0, major='g2', subset=['xgm', 'g2', 'azI', 'stats']):
        """Load data into xarray Dataset"""

        xgm, pulses, trains = self.load_xgm(run, index)
        trains = self.load_identifier(run, index)[2]

        if self.metadata is not None:
            att = self.metadata.loc[run, 'att (%)'] / 100.
            # beam_size = self.metadata.loc[run, 'Beam size / um']
        flux = calc_flux(np.cumsum(xgm, axis=1), attenuation=att)
        dose = calc_dose(flux, npulses=1, photon_energy=9, absorption=0.57,
                     volume_fraction=300/1350, beam_size=10, density=1.35)

        dset = xr.Dataset(
                {
                    "xgm": (["trainId", "pulseId"], xgm),
                    "dose": (["trainId", "pulseId"], dose),
                    },
                coords={
                    "trainId": trains,
                    "pulseId": pulses,
                    },
                )
        for var in subset:
            if var == "azI":
                qI, I, trains = self.load_azimuthal_intensity(run, index)
                dset = dset.assign_coords({"qI": qI})
                dset = dset.assign({"azI": (["trainId", "pulseId", "qI"],
                                I.reshape(*xgm.shape, -1))})
            elif var == "g2":
                t, qv, g2, ttc, trains = self.load_correlation_functions(run, index)
                dset = dset.assign_coords({ "t_cor": t, "qv": qv,})
                dset["g2"] = (["trainId", "t_cor", "qv"], g2)
                dset['intercept'] = dset['g2'].isel(t_cor=0)
                dset['baseline'] = dset['g2'].sel(t_cor=max(dset.t_cor))
                if ttc is not None and 'ttc' in subset:
                    dset = dset.assign_coords({"t1": np.hstack(([0], t)),
                                               "t2": np.hstack(([0], t))})
                    dset['ttc'] = (['trainId', 'qv', 't1', 't2'], ttc)
            elif var == 'stats':
                centers, counts, trains = self.load_statistics(run, index)
                dset = dset.assign_coords({"centers": centers})
                dset["stats"] = (["trainId", "pulseId", "centers"],
                                 counts.reshape(*xgm.shape, -1))

        if "azI" in dset and "g2" in dset:
            dset['g2I'] = (dset['azI'][..., np.abs(dset.qI - dset.qv).argmin('qI')]
                           .mean('pulseId'))
        return dset


    @staticmethod
    def recalculate_g2(dset, pulses=None):
        t = dset.t1
        qv = dset.qv
        if pulses is None:
            pulses = np.arange(t.size)
        t = t[pulses]

        dset = dset.isel(t1=pulses, t2=pulses, t_cor=pulses[:-1])
        if 'trainId' in dset.coords:
            ttc = dset["ttc"].stack(meas=("trainId", "qv"), data=("t1", "t2"))
        else:
            ttc = dset["ttc"].stack(data=("t1", "t2"))

        g2 = da.apply_along_axis(convert_ttc, 1, ttc,
                dtype='float32', shape=(t.size,))
        if 'trainId' in dset.coords:
            g2 = g2.reshape(dset.trainId.size, qv.size, t.size).swapaxes(1, 2)
            dset = dset.update({'g2': (('trainId', 't_cor', 'qv'), g2[:,1:])})
        else:
            g2 = g2.reshape(qv.size, t.size).swapaxes(0, 1)
            dset = dset.update({'g2': (('t_cor', 'qv'), g2[1:])})

        return dset

