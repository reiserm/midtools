import os
import re
import shutil
import numpy as np
from collections import namedtuple
from functools import wraps
import matplotlib as mpl
from functools import reduce
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
    return 1 / (1 + (pixel_size / speckle_size) ** 2)


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


def calc_snr(contrast, intensity, pixels=1, patterns=1, repetitions=1):
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


def calc_flux(
    energy,
    attenuation=1.0,
    photon_energy=9,
    T_lenses=0.586,
    T_apperture=0.75,
    T_air=0.39,
):
    """ "Calculate the photon flux at MID

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
    return photons_per_pules * attenuation * T_lenses * T_apperture * T_air


def calc_dose(
    flux,
    npulses=1,
    photon_energy=9,
    absorption=np.exp(-1),
    volume_fraction=1.0,
    beam_size=10,
    thickness=1.5,
    density=1.0,
):
    """ "Calculate the photon flux at MID

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
    return (
        total_energy_J
        * absorption
        * volume_fraction
        / ((beam_size * 1e-6) ** 2 * thickness * 1e-3 * density * 1e3)
        / 1000
    )


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


def is_processed(run, path="/gpfs/exfel/exp/MID/202022/p002693/"):
    path = path.rstrip("/proc/") + "/proc/"
    proc_folders = os.listdir(path)
    if f"r{run:04}" in proc_folders:
        return True
    else:
        return False


def plot_xgm(run):
    """Plot the train resolved XGM data"""

    # get data as pandas dataframe
    df = run.get_dataframe(
        fields=[
            ("*_XGM/*", "*.photonFlux"),
        ]
    )
    df.reset_index(inplace=True)
    col = list(filter(lambda x: "photonFlux" in x, df.columns))[0]

    # make the plot
    fig, ax = subplots()  # make a figure
    df.plot(y=col, ax=ax, label="photon flux", legend=False)
    ax.set_xlabel("train index")
    ax.set_ylabel("photon flux ($\mu J$)")
    ax.set_title(f"run {run_id}")
    ax.minorticks_on()


def plot_xgm_pr(run, last_pulse=None, train_step=100):
    # the first argument is the data source, the second one the attribute
    arr = run.get_array("SA2_XTD1_XGM/XGM/DOOCS:output", "data.intensityTD")

    if last_pulse is None:
        last_pulse = arr.shape[-1]

    pulse_intensities = arr[::train_step, :last_pulse]
    train_ids = run.train_ids[:last_pulse:train_step]

    with sns.color_palette("Reds_d"):
        plot(pulse_intensities.T, alpha=0.7)
    xlabel("pulse index")
    ylabel("intensity (uJ)")

    gca().minorticks_on()


def get_run_number(x):
    return int(re.findall("(?<=r)\d{4}", x)[0])


def check_contents(filename, return_h5structure=False):
    h5_structure = {
        "META": {
            "/identifiers/pulses_per_train": [None],
            "/identifiers/pulse_ids": [None],
            "/identifiers/train_ids": [None],
            "/identifiers/train_indices": [None],
        },
        "DIAGNOSTICS": {
            "/pulse_resolved/xgm/energy": [None],
            "/pulse_resolved/xgm/pointing_x": [None],
            "/pulse_resolved/xgm/pointing_y": [None],
        },
        "SAXS": {
            "/pulse_resolved/azimuthal_intensity/q": [None],
        },
        "XPCS": {
            "/train_resolved/correlation/q": [None],
            "/train_resolved/correlation/t": [None],
        },
        "FRAMES": {
            "/average/intensity": [None],
        },
        "STATISTICS": {
            "/pulse_resolved/statistics/centers": [None],
        },
    }
    if return_h5structure:
        return h5_structure
    else:
        content = {}
        with h5.File(filename, "r") as f:
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

    with h5.File(filename, "r") as f:
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
    Datasets = namedtuple("Datasets", ["file", "run", "index", "time", "complete"])
    filenames = list(
        filter(lambda x: bool(re.search("\d{3,10}.h5", x)), os.listdir(directory))
    )

    filenames = [directory + x for x in filenames]
    filenames.sort(key=os.path.getmtime)
    mtimes = [os.path.getmtime(x) for x in filenames]
    runs = [get_run_number(x) for x in filenames]
    state = [is_complete(x) for x in filenames]
    indices = list(
        map(lambda x: int(re.search("\d{3}(?=\.h5)", x).group(0)), filenames)
    )

    datasets = Datasets(filenames, runs, indices, mtimes, state)
    return datasets


def get_datasets_from_table(df):
    """Search a folder for datasets processed by midtools

    Args:
        df (pd.DataFrame): DataFrame containing a `filename` column with the
            full path and name of the analysis file.

    Returns:
        namedtuple: `Datasets` that can be passed to the Dataset class. It
            contains lists of HDF5-files, run numbers, indices, time-stamps.

    Examples:
        Use the output like `Dataset(datasets)`.

    """
    Datasets = namedtuple(
        "Datasets", ["file", "proposal", "run", "index", "time", "complete"]
    )
    df = df.reset_index()
    if not "filename" in df.columns:
        raise KeyError(
            "No `filename` column found in dataframe. Provide a"
            "series of filenames that point to the HDF5 files in the DataFrame."
        )

    filenames = [AnaFile(x) for x in df["filename"]]
    proposals = [x for x in df["proposal"]]
    nfiles = len(filenames)
    mtimes = [os.path.getmtime(x.fullname) for x in filenames]
    runs = [x.run_number for x in filenames]
    state = ["complete"] * nfiles  # for backwards compatibility
    indices = [x.counter for x in filenames]

    filenames = [x.fullname for x in filenames]

    datasets = Datasets(filenames, proposals, runs, indices, mtimes, state)
    return datasets


def autocorr(x):
    x = (x - np.mean(x)) / np.std(x)
    result = np.correlate(x, x, mode="full") / len(x)
    return result[result.size // 2 :]


def rescale(y, mn, mx, rng=(0, 1)):
    p = (rng[1] - rng[0]) / (mx - mn)
    return p * (y - mn) + rng[0], p


def add_colorbar(
    ax,
    vec,
    label=None,
    cmap="magma",
    discrete=False,
    tick_step=1,
    qscale=0,
    location="right",
    show_offset=False,
    **kwargs,
):

    ncolors = len(vec)
    tick_indices = np.arange(0, len(vec), tick_step)
    vec = np.array(vec)[tick_indices]
    vec *= 10 ** qscale
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

    if location == "right":
        orientation = "vertical"
    elif location == "top":
        orientation = "horizontal"

    cax = mpl.colorbar.make_axes(ax, location=location, **kwargs)[0]
    cb = mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap, orientation=orientation)

    # set up color bar ticks
    if location == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

    cb.set_label(label)
    if change_ticks:
        cb.set_ticks(tick_indices)
        cb.set_ticklabels(list(map(lambda x: "$%.{}f$".format(3 - qscale) % x, vec)))
        if qscale and show_offset:
            cb.ax.text(
                1.0,
                1.04,
                r"$\times 10^{{-{}}}$".format(qscale),
                transform=cb.ax.transAxes,
            )
    cb.ax.invert_yaxis()
    cb.ax.set_in_layout(True)


class Interpreter:
    """Class to explore MID datasets."""

    FIGSIZE = (4, 3)

    def __init__(
        self, datasets, metadata=None, geom=None, mask=None, max_len=100, **kwargs
    ):
        self.datasets = datasets
        self.metadata = metadata
        self.info = None
        self.proposal = None
        self.run = None
        self.index = None
        self.filename = None
        self.dataset_index = 0  # sets default of the previous 4 attributes
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
        self.max_len = max_len
        self.config = None
        self.kwargs = kwargs
        self.filtered = {run: None for run in set(datasets.run)}

    def iter_trainids(self, run, subset=None):
        indices = np.array(self.datasets.index)[
            np.where(np.array(self.datasets.run) == run)[0]
        ]
        if subset is not None:
            indices = np.intersect1d(indices, subset)
        for index in indices:
            yield index, self.load_identifier()[2]

    def iter_files(self, run, subset=None):
        indices = np.array(self.datasets.index)[
            np.where(np.array(self.datasets.run) == run)[0]
        ]
        files = np.array(self.datasets.file)[
            np.where(np.array(self.datasets.run) == run)[0]
        ]
        if subset is not None:
            indices, in_list = np.intersect1d(indices, subset, return_indices=True)[:2]
        for index, ind in zip(indices, in_list):
            yield index, files[ind]

    def __str__(self):
        if self.info is None:
            return "MID Dataset"
        else:
            return self.info

    def __repr__(self):
        return self.__str__()

    def sel(self, **d):

        if "run" in d:
            self.run = d["run"]
        if len(d):
            cond = np.ones(len(self.datasets.file))
            for attr, val in d.items():
                cond *= np.array(getattr(self.datasets, attr)) == val
            if sum(cond) > 1:
                raise ValueError(
                    "Could not identify filename unambiuously. "
                    f"Found more than one datasets for run {self.run} and proposal {self.proposal}:\n"
                    f"In particular found indices: {[self.datasets.index[x] for x in np.where(cond)[0]]}"
                )
            elif sum(cond) == 0:
                raise ValueError(
                    "None of the datasets matches the given conditions:" f"{d}"
                )
            elif sum(cond) == 1:
                self.dataset_index = np.where(cond)[0][0]

    @property
    def run(self):
        return self.__run

    @run.setter
    def run(self, run):
        self.__run = run
        if isinstance(run, int):
            self.info = f"RUN: {self.run}\n"
            if self.metadata is not None:
                meta = self.metadata.loc[(self.proposal, self.run)]
                cols = ["sample", "att (%)"]
                self.info += "\n".join([col + f": {meta[col]}" for col in cols])

    @property
    def dataset_index(self):
        return self.__dataset_index

    @dataset_index.setter
    def dataset_index(self, index):
        self.__dataset_index = index
        self.proposal = self.datasets.proposal[index]
        self.run = self.datasets.run[index]
        self.index = self.datasets.index[index]
        self.filename = self.datasets.file[index]

    def _rebin(self, arr, avr=False):
        """rebin along first axis and return rebinned array and bin size"""
        new_len = self.max_len
        if arr.shape[0] > new_len:
            old_shape = arr.shape
            rest = arr.shape[0] % new_len
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

    def visit(self):
        """Print the content tree of the selected HDF5 file."""
        visit_h5(self.filename)

    def load_utility(self):
        keysh5 = [
            "utility/mask",
        ]

        args = ["mask"]
        out = {arg: None for arg in args}
        with h5.File(self.filename, "r") as f:
            for key, arg in zip(keysh5, out.keys()):
                if key in f:
                    data = np.array(f[key])
                    out[arg] = data
                    setattr(self, arg, data)

        return list(out.values())

    def load_identifier(self):
        keysh5 = [
            "identifiers/pulse_ids",
            "identifiers/train_indices",
            "identifiers/train_ids",
            "identifiers/all_trains",
        ]

        args = ["pulseIds", "train_indices", "trainIds", "all_trains"]
        out = {arg: None for arg in args}
        with h5.File(self.filename, "r") as f:
            for key, arg in zip(keysh5, out.keys()):
                if key in f:
                    data = np.array(f[key])
                    out[arg] = data
                    setattr(self, arg, data)

        return list(out.values())

    def load_xgm(self):
        with h5.File(self.filename, "r") as f:
            xgm = f["pulse_resolved/xgm/energy"][:]
            pulses = f["identifiers/pulse_ids"][:]
            trains = f["identifiers/train_indices"][:]
        self.xgm = xgm
        self.pulses = pulses
        self.trains = trains
        return xgm, pulses, trains

    def plot_xgm(self, rebin_kws=None):
        """Plot the train resolved XGM data"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get("rebin_kws", {})

        data, pulses, trains = self.load_xgm()
        data, trains = self._select_filtered(data, trains)
        data, tstep = self._rebin(data, **rebin_kws)
        trains = np.arange(data.shape[0]) * tstep

        # make the plot
        fig, axs = plt.subplots(
            1, 2, figsize=(9, 4), sharey=True, constrained_layout=True
        )
        axs = np.ravel(axs)

        xlabels = ["pulse index", "train index"]
        colors = sns.color_palette("Blues_d", trains.size, desat=1)

        for i, (ax, xl) in enumerate(zip(axs, xlabels)):
            if i == 0:
                for t, color in zip(data, colors):
                    ax.plot(pulses, t, color=color, alpha=0.4)
                ax.set_ylabel("photon flux ($\mu J$)")
            if i == 1:
                ax.plot(trains, np.mean(data, 1))
            ax.set_xlabel(xl)
            ax.set_title(f"run {self.run}, {len(pulses)} pulses per train")
            ax.minorticks_on()
        fig.suptitle(self.info.replace("\n", ", "), fontsize=16)

    def load_images(self):
        keysh5 = ["{}/intensity", "{}/variance"]

        args = ["frames", "variance"]
        out = {arg: None for arg in args}
        with h5.File(self.filename, "r") as f:
            for key, arg in zip(keysh5, out.keys()):
                for case in ["dark", "average"]:
                    key2 = key.format(case)
                    if key2 in f:
                        data = np.array(f[key2])
                        out[arg] = data
                        setattr(self, arg, data)

        return list(out.values())

    def position_modules(arr):
        arr = arr.reshape(-1, 16, 512, 128)
        return np.squeeze(self.geom.position_modules_fast(arr)[0])

    def plot_images(self, vmin=None, vmax=None, log=True):
        """Plot average image and some frames"""

        frames = self.load_images()[0].reshape(-1, 16, 512, 128)
        if self.mask is not None:
            frames[:, ~self.mask] = np.nan
        avr = frames.mean(0)
        avr[avr < 0] = 0

        # make the plot
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs_main = fig.add_gridspec(4, 8)

        ax_main = fig.add_subplot(gs_main[:, :4])
        subax = [
            fig.add_subplot(gs_main[i : i + 2, j : j + 2])
            for i in range(0, 4, 2)
            for j in range(4, 8, 2)
        ]

        if log:
            norm = LogNorm()
        else:
            norm = None
        im = ax_main.imshow(
            self.geom.position_modules_fast(avr)[0],
            cmap="magma",
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax_main.set_title("average trains")

        for i, (ax, frame) in enumerate(zip(subax, frames)):
            if self.mask is not None:
                frame[~self.mask] = np.nan
            if self.geom is not None:
                slc = slice(400, -400)
                frame = self.geom.position_modules_fast(frame)[0][slc, slc]
            else:
                frame = frame.reshape(-1, 1024)

            ax.imshow(frame, norm=norm, vmin=vmin, vmax=vmax, cmap="magma")

            ax.set_title(f"train {i}")
        plt.colorbar(im, ax=subax[1::2], shrink=0.5, label="intensity")
        fig.suptitle(self.info.replace("\n", ", "), fontsize=16)

    def load_azimuthal_intensity(self):
        with h5.File(self.filename, "r") as f:
            q = f["pulse_resolved/azimuthal_intensity/q"][:]
            I = f["pulse_resolved/azimuthal_intensity/I"][:]
            trains = f["identifiers/train_indices"][:]
        self.qI = (q, I)
        self.trains = trains
        return q, I, np.repeat(trains, I.shape[0] / trains.size)

    def plot_azimuthal_intensity(self, rebin_kws=None, vmin=None, vmax=None):
        """Plot the pulse resolved azimuthal intensity"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get("rebin_kws", {})

        q, I, trains = self.load_azimuthal_intensity()
        I, trains = self._select_filtered(I, trains)
        I, pstep = self._rebin(I, **rebin_kws)
        pulses = np.arange(I.shape[0]) * pstep

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        axs = np.ravel(axs)

        if I.shape[0]:
            cmap = "Reds"
            alpha = 0.4
        else:
            cmap = "Set1"
            alpha = 1

        for i, (ax,) in enumerate(
            zip(
                axs,
            )
        ):
            if i == 0:
                im = ax.pcolor(q, pulses, I, cmap="inferno", vmin=vmin, vmax=vmax)
                ax.set_ylabel(
                    f"pulse step {pstep} " f'average: {rebin_kws.get("avr", False)}'
                )
                plt.colorbar(im, ax=ax, shrink=0.6)
                ax.set_xscale("log")
            if i == 1:
                colors = sns.color_palette(cmap, I.shape[0], desat=0.7)
                for i in range(I.shape[0]):
                    ax.loglog(q, I[i], color=colors[i], alpha=alpha)
                ax.set_ylabel("intensity (a.d.u.)")
                add_colorbar(
                    ax,
                    pulses,
                    cmap=cmap,
                    shrink=0.6,
                    label=(
                        f"pulse step {pstep} " f'average: {rebin_kws.get("avr", False)}'
                    ),
                )
            ax.set_xlabel("q ($nm^{-1}$)")
            ax.set_ylim(np.nanmean(I[:, -50:]), None)
            ax.minorticks_on()
        fig.suptitle(self.info.replace("\n", " | "), fontsize=16)

    def load_correlation_functions(self):
        keysh5 = [
            "train_resolved/correlation/t",
            "train_resolved/correlation/q",
            "train_resolved/correlation/g2",
            "/train_resolved/correlation/ttc",
            "identifiers/train_indices",
        ]

        args = ["t_cor", "qv", "g2", "ttc", "train_indices"]
        out = {arg: None for arg in args}
        with h5.File(self.filename, "r") as f:
            for key, arg in zip(keysh5, out.keys()):
                if key in f:
                    data = np.array(f[key])
                    out[arg] = data
                    setattr(self, arg, data)

        return list(out.values())

    def plot_correlation_functions(
        self,
        qval=0.1,
        g2_offset=0,
        ylim=(0.98, None),
        clim=(None, None),
        rebin_kws=None,
    ):
        """Plot correlation functions"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get("rebin_kws", {})

        t, q, g2, ttc, trains = self.load_correlation_functions()
        g2, ttc, trains = self._select_filtered(g2, ttc, trains)
        qind = np.argmin(np.abs(q - qval))
        qval = q[qind]
        t = t * 1e6
        g2, tstep = self._rebin(g2, **rebin_kws)
        trains = trains[: g2.shape[0] * tstep : tstep]
        g2 -= g2_offset

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        axs = np.ravel(axs)

        if g2.shape[0]:
            cmap = "Purples"
            alpha = 0.4
        else:
            cmap = "Set1"
            alpha = 1

        for i, (ax,) in enumerate(
            zip(
                axs,
            )
        ):
            if i == 0:
                im = ax.pcolor(
                    t,
                    trains,
                    g2[..., qind],
                    cmap="inferno",
                    vmin=clim[0],
                    vmax=clim[1],
                )
                ax.set_xscale("log")
                ax.set_ylabel(
                    f"train step {tstep} " f'average: {rebin_kws.get("avr", False)}'
                )
                ax.set_title(f"$q={qval:.3}\,nm^{{-1}}$")
                plt.colorbar(im, ax=ax, shrink=0.6)
            if i == 1:
                colors = sns.color_palette(cmap, g2.shape[0], desat=0.7)
                for i in range(g2.shape[0]):
                    ax.semilogx(t, g2[i, :, qind], color=colors[i], alpha=alpha)
                ax.set_ylabel("$g_2$")
                ax.set_title(f"$q={qval:.3}\,nm^{{-1}}$")
                add_colorbar(
                    ax,
                    trains,
                    cmap=cmap,
                    shrink=0.6,
                    label=(
                        f"train step {tstep} " f'average: {rebin_kws.get("avr", False)}'
                    ),
                )
                ax.set_ylim(ylim)
            ax.set_xlabel("t ($\mu s$)")
            ax.minorticks_on()
        fig.suptitle(self.info.replace("\n", " | "), fontsize=16)

    def load_statistics(self):
        tmp = np.zeros((500))
        count_path = "/pulse_resolved/statistics/counts"
        centers_path = "pulse_resolved/statistics/centers"
        with h5.File(self.filename, "r") as f:
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

    def plot_statistics(self, rebin_kws=None):
        """Plot correlation functions"""

        if rebin_kws is None:
            rebin_kws = self.kwargs.get("rebin_kws", {})

        centers, counts, pulses = self.load_statistics()
        counts = self._select_filtered(counts)
        xgm = self._select_filtered(self.load_xgm())[0]

        agipd_pulse_intensity = counts.mean(1)
        xgm_pulse_intensity = xgm.flatten()
        train_pulse = np.arange(xgm_pulse_intensity.size)

        counts, pstep = self._rebin(counts, **rebin_kws)
        xgm = self._rebin(xgm, **rebin_kws)[0]
        pulses = np.arange(counts.shape[0]) * pstep

        # make the plot
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        axs = np.ravel(axs)

        if counts.shape[0]:
            cmap = "Blues"
            alpha = 0.4
        else:
            cmap = "Set1"
            alpha = 1
        for i, (ax,) in enumerate(
            zip(
                axs,
            )
        ):

            if i == 0:
                # im = ax.scatter(xgm_pulse_intensity, agipd_pulse_intensity,
                #                c=train_pulse, cmap=cmap)
                ax.set_ylabel(f"agipd pulse intensity (a.d.u.)")
                ax.set_xlabel(f"xgm pulse intensity ($\mu$J)")

            if i == 1:
                colors = sns.color_palette(cmap, counts.shape[0], desat=0.7)
                for cnt, col in zip(counts, colors):
                    pl = ax.semilogy(centers, cnt, color=col, alpha=alpha)
                add_colorbar(
                    ax,
                    pulses,
                    cmap=cmap,
                    shrink=0.6,
                    label=(
                        f"pulse step {pstep} " f'average: {rebin_kws.get("avr", False)}'
                    ),
                )
                ax.set_xlabel("intensity (a.d.u.)")
                ax.set_ylabel("counts")

            ax.minorticks_on()
        fig.suptitle(self.info.replace("\n", " | "), fontsize=16)

    def load_sample_sources(self):
        h5keys = [
            "identifiers/train_indices",
            "/train_resolved/sample_position/y",
            "/train_resolved/sample_position/z",
            "/train_resolved/linkam-stage/temperature",
        ]
        args = ["trains", "y", "z", "T"]

        out = {arg: None for arg in args}
        with h5.File(self.filename, "r") as f:
            for key, arg in zip(h5keys, out.keys()):
                if key in f:
                    data = np.array(f[key])
                else:
                    data = -42 * np.ones_like(out["trains"])
                out[arg] = data
                setattr(self, arg, data)

        return list(out.values())

    def plot_sample_sources(self, rebin_kws=None):
        """Plot sample position and temperature"""

        trains, y, z, T = self._select_filtered(*self.load_sample_sources())

        yu, yc = np.unique(np.round(np.diff(y), 2), return_counts=True)
        yu = yu[(yc > 2).nonzero()]
        yc = yc[(yc > 2).nonzero()]
        zu, zc = np.unique(np.abs(np.round(np.diff(z), 3)), return_counts=True)
        ind = ((zc > 10) & (np.abs(zu) < 0.4)).nonzero()
        zu = zu[ind]
        zc = zc[ind]
        print(f"total {len(trains)} trains")
        print(f"unique y steps (mm) {yu} occurred {yc}")
        print(f"unique z steps (mm) {zu} occurred {zc}")
        print(f"unique T steps (K) {np.unique(np.diff(T))}")
        print(f"area: {y.max()- y.min():.2f} x {z.max()- z.min():.2f} (y * z)")

        fig, axs = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)

        ax = axs.flat[0]
        ax.scatter(y, z, c=T, cmap="RdYlGn", alpha=0.05)
        ax.set_xlabel("y coordinate")
        ax.set_ylabel("z coordinate")
        ax.set_title("Sample position")

        ax = axs.flat[1]
        t = f"LINKAM T"
        ax.scatter(trains, T, c=T, s=0.1, cmap="RdYlGn", alpha=0.05)
        ax.set_title(t)
        ax.set_ylabel("T (deg C)")

        ax = axs.flat[2]
        ax.plot(trains, z, ".")
        ax.set_xlabel("trains")
        ax.set_ylabel("z coordinate")

        ax = axs.flat[3]
        ax.plot(trains, y, ".")
        ax.set_xlabel("trains")
        ax.set_ylabel("y coordinate")

        for ax in axs.flat:
            ax.minorticks_on()

        fig.suptitle(self.info.replace("\n", " | "), fontsize=16)

    def to_xDataset(self, major="g2", subset=["xgm", "g2", "azI", "stats"]):
        """Load data into xarray Dataset"""

        xgm, pulses, trains = self.load_xgm()
        trains = self.load_identifier()[2]

        if self.metadata is not None:
            att = self.metadata.loc[(self.proposal, self.run), "att (%)"] / 100.0
            # beam_size = self.metadata.loc[run, 'Beam size / um']
        flux = calc_flux(np.cumsum(xgm, axis=1), attenuation=att)
        dose = calc_dose(
            flux,
            npulses=1,
            photon_energy=9,
            absorption=0.57,
            volume_fraction=300 / 1350,
            beam_size=10,
            density=1.35,
        )

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
                qI, I, trains = self.load_azimuthal_intensity()
                dset = dset.assign_coords({"qI": qI})
                dset = dset.assign(
                    {"azI": (["trainId", "pulseId", "qI"], I.reshape(*xgm.shape, -1))}
                )
            elif var == "g2":
                t, qv, g2, ttc, trains = self.load_correlation_functions()
                dset = dset.assign_coords(
                    {
                        "t_cor": t,
                        "qv": qv,
                    }
                )
                dset["g2"] = (["trainId", "t_cor", "qv"], g2)
                dset["intercept"] = dset["g2"].isel(t_cor=0)
                dset["baseline"] = dset["g2"].sel(t_cor=max(dset.t_cor))
                if ttc is not None and "ttc" in subset:
                    dset = dset.assign_coords(
                        {"t1": np.hstack(([0], t)), "t2": np.hstack(([0], t))}
                    )
                    dset["ttc"] = (["trainId", "qv", "t1", "t2"], ttc)
            elif var == "stats":
                centers, counts, trains = self.load_statistics()
                dset = dset.assign_coords({"centers": centers})
                dset["stats"] = (
                    ["trainId", "pulseId", "centers"],
                    counts.reshape(*xgm.shape, -1),
                )
            elif var == "sample":
                trains, y, z, T = self.load_sample_sources()
                dset["y"] = (["trainId"], y)
                dset["z"] = (["trainId"], z)
                dset["T"] = (["trainId"], T)

        if "azI" in dset and "g2" in dset:
            dset["g2I"] = dset["azI"][..., np.abs(dset.qI - dset.qv).argmin("qI")].mean(
                "pulseId"
            )

        if self.filtered[self.run] is not None:
            print("Loaded filtered Dataset", self.run)
            dset = dset.sel(trainId=self.filtered[self.run])
        return dset

    @staticmethod
    def recalculate_g2(dset, pulses=None):
        t = dset.t1
        qv = dset.qv
        if pulses is None:
            pulses = np.arange(t.size)
        t = t[pulses]

        dset = dset.isel(t1=pulses, t2=pulses, t_cor=pulses[:-1])
        if "trainId" in dset.coords:
            ttc = dset["ttc"].stack(meas=("trainId", "qv"), data=("t1", "t2"))
        else:
            ttc = dset["ttc"].stack(data=("t1", "t2"))

        g2 = da.apply_along_axis(convert_ttc, 1, ttc, dtype="float32", shape=(t.size,))
        if "trainId" in dset.coords:
            g2 = g2.reshape(dset.trainId.size, qv.size, t.size).swapaxes(1, 2)
            dset = dset.update({"g2": (("trainId", "t_cor", "qv"), g2[:, 1:])})
        else:
            g2 = g2.reshape(qv.size, t.size).swapaxes(0, 1)
            dset = dset.update({"g2": (("t_cor", "qv"), g2[1:])})

        return dset

    def _select_filtered(self, *args):
        if self.filtered[self.run] is not None:
            trains = self.load_identifier()[2]
            ntrains = trains.size
            ind = np.asarray(
                np.intersect1d(trains, self.filtered[self.run], return_indices=True)[1]
            )
            new_args = []
            for arg in args:
                arg = np.asarray(arg)
                s = arg.shape
                s2 = s[0] // ntrains
                if s2 > 1:
                    arg = arg.reshape(ntrains, s2, *s[1:])
                arg = arg[ind]
                arg = arg.reshape(ind.size * s2, *s[1:])
                new_args.append(arg)
            return new_args
        else:
            return args

    def filter(
        self,
        show=False,
        verbose=False,
        xgm_lower=800,
        azI_percentiles=None,
        azI_nsiqma=3,
        mot_stepsize=50,
        mot_nsteps=2,
        subset=None,
        subsequent=True,
        ttc_kws=None,
    ):
        """Filter trains based on different conditions."""

        if subset is None:
            subset = ["xgm", "azI", "sample"]
        if "ttc" in subset and not "g2" in subset:
            subset.append("g2")
        if ttc_kws is None:
            ttc_kws = {}

        dset = self.to_xDataset(self.run, subset=subset)
        self.filtered[self.run] = np.unique(dset.trainId.values)
        trainIds = self.filtered[self.run]
        trainIds0 = trainIds.copy()

        if "sample" in subset:
            mpos = [dset["y"], dset["z"]]
            mpos = mpos[np.argmax([np.abs(x.min() - x.max()) for x in mpos])]
            trainIds = np.intersect1d(
                trainIds,
                dset.trainId[
                    (mpos > (mpos.min() + mot_nsteps * mot_stepsize / 1000))
                    & (mpos < (mpos.max() - mot_nsteps * mot_stepsize / 1000))
                ],
            )

        if "xgm" in subset:
            xgm = dset["xgm"].mean("pulseId")
            test_against = trainIds if subsequent else trainIds0
            xgm_thres = np.percentile(
                xgm.sel(trainId=test_against).values.flatten(), 10
            )
            xgm_thres = max(xgm_thres, xgm_lower)
            trainIds = np.intersect1d(trainIds, dset.trainId[(xgm > xgm_thres)])

        if "azI" in subset:
            test_against = trainIds if subsequent else trainIds0
            azI = dset["azI"].mean("pulseId").mean("qI")
            if bool(azI_percentiles):
                azI_thres = [
                    np.percentile(azI.sel(trainId=test_against).values.flatten(), x)
                    for x in azI_percentiles
                ]
            else:
                m = azI.sel(trainId=test_against).mean().values
                s = azI.sel(trainId=test_against).std().values
                azI_thres = [m - azI_nsiqma * s, m + azI_nsiqma * s]
            trainIds = np.intersect1d(
                trainIds, dset.trainId[(azI > azI_thres[0]) & (azI < azI_thres[1])]
            )

        self.filtered[self.run] = trainIds

        if verbose:
            print(f"Kept {trainIds.size/trainIds0.size*100:.1f}% of trains.")

        if show:
            colors = {"all": ["gray", 0.4], "sel": ["tab:red", 0.7]}
            fig, ax = plt.subplots(
                1,
                2,
                figsize=(self.FIGSIZE[0] * 2, self.FIGSIZE[1]),
                constrained_layout=True,
                sharey="row",
            )
            axc = np.array(ax)

            ranges = []
            for i, (n, tid) in enumerate(zip(["all", "sel"], [dset.trainId, trainIds])):
                ranges.append((2, max(xgm.values.flatten())))
                axc[0].hist(
                    xgm.sel(trainId=tid),
                    32,
                    range=ranges[0],
                    color=colors[n][0],
                    alpha=colors[n][1],
                )

                ranges.append((0, max(azI.values.flatten())))
                axc[1].hist(
                    azI.sel(trainId=tid),
                    32,
                    range=ranges[1],
                    color=colors[n][0],
                    alpha=colors[n][1],
                )

            axc[0].set_title("XGM")
            axc[0].set_xlabel("intensity (µJ)")
            axc[0].set_ylabel("counts")

            axc[1].set_title("AGIPD")
            axc[1].set_xlabel("intensity (ph/pix)")

            for a in axc:
                a.tick_params(
                    axis="both", direction="out", which="both", right=True, top=True
                )
                a.minorticks_on()

        dset = dset.sel(trainId=trainIds)

        if "ttc" in subset:
            dset = self.filter_ttc(dset, show=show, **ttc_kws)

        return dset

    def filter_ttc(self, dset, ttc_thres=(0.1, 2), qbin=3, show=False):
        """Filter data based on TTC values"""
        measure = (dset["ttc"] > ttc_thres[0]) & (dset["ttc"] < ttc_thres[1])
        ttc_unfiltered = dset["ttc"].copy(deep=True)
        dset["ttc"] = dset["ttc"].where(measure)

        if show:
            fig = plt.figure(
                figsize=(self.FIGSIZE[0] * 2, self.FIGSIZE[1] * 3),
                constrained_layout=True,
            )
            gs = fig.add_gridspec(6, 2)
            ax1 = fig.add_subplot(gs[:3, 0])
            ax2 = fig.add_subplot(gs[3:, 0])
            ax3 = fig.add_subplot(gs[:2, 1])
            ax4 = fig.add_subplot(gs[2:4, 1])
            ax5 = fig.add_subplot(gs[4:, 1])

            qcolors = sns.color_palette("inferno", dset.qv.size)
            for qi in range(dset.qv.size):
                dsetq = dset.isel(qv=qi)
                ax3.hist(dsetq["g2I"], 20, color=qcolors[qi], alpha=0.4, density=True)
            ax3.set_xlabel("average number photons")
            ax3.set_ylabel("pdf")
            ax3.set_xscale("log")

            im = ax2.imshow(
                dset["ttc"].isel(qv=qbin).mean("trainId"),
                origin="lower",
                interpolation="nearest",
            )
            ax2.set_xlabel("frame axis 0")
            ax2.set_ylabel("frame axis 1")
            ax2.text(
                0.22,
                0.92,
                "w/ contrast sorting",
                transform=ax2.transAxes,
                ha="center",
                bbox={"color": "w"},
            )

            vmin, vmax = im.get_clim()
            ax1.imshow(
                ttc_unfiltered.isel(qv=qbin).mean("trainId"),
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                interpolation="nearest",
            )
            ax1.set_xlabel("frame axis 0")
            ax1.set_ylabel("frame axis 1")
            ax1.text(
                0.22,
                0.92,
                "w/o contrast sorting",
                transform=ax1.transAxes,
                ha="center",
                bbox={"color": "w"},
            )

            ttcshape = ttc_unfiltered[0, 0, 0].size
            ind = np.tril_indices(ttcshape, k=-1)
            ind = np.ravel_multi_index(ind, (ttcshape, ttcshape))
            ttcflat = ttc_unfiltered.stack(ttcvals=('t1', 't2')).isel(ttcvals=ind)
            ttcflat = ttcflat.where(ttcflat>0)

            ax5.hist(
                np.ravel(ttcflat),
                64,
                range=(0.1, 10),
                color="tab:blue",
                alpha=0.7,
                density=True,
            )
            ax5.set_xlabel("TTC values")
            ax5.set_ylabel("pdf")

            ttcmin = ttcflat.min('ttcvals')
            ttcmax = ttcflat.max('ttcvals')
            ttcmean = ttcflat.mean('ttcvals')
            for data, marker, color in zip(
                [ttcmin, ttcmax], ["o", "x"], ["tab:red", "k"]
            ):
                ax4.scatter(dset["g2I"], data, alpha=0.3, marker=marker, color=color)

            ax4.set_xscale("log")
            ax4.set_yscale("log")
            ax4.set_xlabel("average number photons")
            ax4.set_ylabel("max(TTC)")

            for a in [ax1, ax2, ax3, ax4, ax5]:
                a.minorticks_on()

        return dset


class AnaFile:
    ANA_FMT = "r{:04d}-analysis_{:03d}.h5"
    SETUP_FMT = "r{:04d}-setup_{:03d}.yml"

    def __init__(self, filename, dirname=""):

        if isinstance(filename, (tuple, list)):
            filename = self.ANA_FMT.format(*filename)
            if os.path.isdir(dirname):
                filename = os.path.join(dirname, filename)

        self.fullname = os.path.abspath(filename)
        self.dirname = os.path.dirname(self.fullname)
        self.basename = os.path.basename(self.fullname)
        self._get_run_number()
        self._get_counter()
        self._get_setupfile()

    def __str__(self):
        return self.fullname

    def __repr__(self):
        return f"AnaFile('{self.fullname}')"

    def _get_run_number(
        self,
    ):
        self.run_number = int(re.findall("(?<=r)\d{4}", self.basename)[0])

    def _get_counter(
        self,
    ):
        self.counter = int(re.findall("(?<=_)\d{3}(?=\.h5)", self.basename)[0])

    def _get_setupfile(
        self,
    ):
        self.setupbase = self.basename.replace("analysis", "setup").replace(
            ".h5", ".yml"
        )
        self.setupfile = os.path.join(self.dirname, self.setupbase)
        if not os.path.isfile(self.setupfile):
            self.setupbase = None
            self.setupfile = None

    def rename(self, counter=None, path=None, copy=False, overwrite=False):
        if counter is None and path is None:
            print("Neither of counter or path provided. Nothing happened.")
            return
        func = shutil.copy if copy else shutil.move
        src = [self.fullname]
        dest = []
        if counter is None:
            counter = self.counter
        if path is None:
            path = self.dirname
        dest.append(os.path.join(path, self.ANA_FMT.format(self.run_number, counter)))
        if bool(self.setupfile):
            src.append(self.setupfile)
            dest.append(
                os.path.join(path, self.SETUP_FMT.format(self.run_number, counter))
            )

        for old_file, new_file in zip(src, dest):
            try:
                if os.path.isfile(new_file) and not overwrite:
                    print(f"Skipping existing file {new_file}.")
                    continue
                func(old_file, new_file)
            except FileNotFoundError:
                print(f"File: {old_file} not found")

    def remove(self):
        if os.path.isfile(self.fullname):
            os.remove(self.fullname)
        if bool(self.setupfile):
            os.remove(self.setupfile)

    def setuppars(self):
        if bool(self.setupfile):
            return MidData._read_setup(self.setupfile)
