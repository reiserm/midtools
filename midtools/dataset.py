#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import yaml
import re
import sys
import time
from datetime import datetime
import numpy as np
import h5py as h5
import argparse
from pathlib import Path
from shutil import copyfile

# XFEL packages
import extra_data
from extra_data import RunDirectory, open_run
from extra_geom import AGIPD_1MGeometry

# Dask and xarray
import dask
import dask_jobqueue
import dask.array as da
from dask.distributed import Client, progress, LocalCluster
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar
from dask.distributed import TimeoutError
from distributed.comm.core import CommClosedError
from functools import wraps

import Xana
from Xana import Setup

from .azimuthal_integration import azimuthal_integration
from .correlations import correlate, get_q_phi_pixels
from .average import average
from .statistics import statistics
from .calibration import Calibrator, _create_mask_from_dark, _create_mask_from_flatfield
from .masking import mask_radial, mask_asics

from functools import reduce

from .plotting import get_datasets
from .plotting import Interpreter

import pdb


def _exception_handler(max_attempts=3):
    def restart(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError, TimeoutError, CommClosedError) as e:
                    print(f"Restart due to {str(e)}. Attempt {attempt}")
                    attempt += 1

        return wrapper

    return restart


def print_now():
    now = datetime.now()
    print(now.strftime("%Y-%m-%d: %H-%M-%S"), flush=True)


class Dataset:

    METHODS = [
        "META",
        "DIAGNOSTICS",
        "FRAMES",
        "SAXS",
        "XPCS",
        "STATISTICS",
        "DARK",
        "FLATFIELD",
    ]

    def __init__(
        self,
        run_number,
        setupfile,
        proposal=None,
        analysis=None,
        datdir=None,
        first_train=0,
        last_train=1e6,
        pulses_per_train=500,
        dark_run_number=None,
        train_file="train-file.npy",
        pulse_file="pulse-file.npy",
        first_cell=2,
        train_step=1,
        pulse_step=1,
        is_dark=False,
        localcluster=False,
        is_flatfield=False,
        flatfield_run_number=None,
        out_dir="./",
        file_identifier=None,
        trainId_offset=0,
        **kwargs,
    ):
        """Dataset object to analyze MID datasets on Maxwell.

        Args:
            setupfile (str): Setupfile (.yml) that contains information on the
                setup parameters.
            analysis (str, optional): Flags of the analysis to perform.
                Defaults to '00'. `analysis` is a string of ones and zeros
                where a one means to perform the analysis and a zero means to
                omit the analysis. The analysis types are:

                +-------+-----------------------------+
                | flags  | analysis                   |
                +========+============================+
                |  1000  | average frames             |
                +--------+----------------------------+
                |  0100  | SAXS azimuthal integration |
                +--------+----------------------------+
                |  0010  | XPCS correlation functions |
                +--------+----------------------------+
                |  0001  | compute statistics         |
                +--------+----------------------------+

            last_train (int, optional): Index of last train to analyze. If not
                provided, all trains are processed.

            run_number (int, optional): Specify run number. Defaults to None.
                If not defined, the datdir in the setupfile has to contain the
                .h5 files.

            dark_run_number (int, optional): Dark run number for calibration.

            pulses_per_train (int, optional): Specify the number of pulses per
                train. If not provided, take all stored memory cells.

            train_step (int, optional): Stepsize for slicing the train axis.

            pulse_step (int, optional): Stepsize for slicing the pulse axis.

            is_dark (bool, optional): If True switch to dark routines, i.e.,
                average dark for dark subtraction, calculate mask from darks.

            is_flatfield (bool, optional): If True use flatfield algorithms.

            flatfield_run_number (int, optional): Run number of the processed
                flatfield for calibration.

        Note:
            A setupfile might look like this::

                # setup.yml file

                # Data
                datdir: /path/to/data/r0522

                # Maskfile
                mask: /path/to/mask/agipd_mask_tmp.npy

                # Beamline
                photon_energy: 9 # keV
                sample_detector: 8 # m
                pixel_size: 200 # um

                quadrant_positions:
                    dx: -18
                    dy: -15
                    q1: [-500, 650]
                    q2: [-550, -30]
                    q3: [ 570, -216]
                    q4: [ 620, 500]

                # XPCS
                xpcs_opt:
                    q_range:
                        q_first: .1 # smallest q in nm-1
                        q_last: 1.  # largest q in nm-1
                        steps: 10   # how many q-bins
        """

        self.out_dir = out_dir
        self.pulse_file = os.path.join(self.out_dir, pulse_file)
        self.train_file = os.path.join(self.out_dir, train_file)

        #: str: Path to the setupfile.
        self.setupfile = setupfile
        setup_pars = self._read_setup(setupfile)
        #: bool: True if current run is a dark run.
        self.is_dark = is_dark
        #: bool: True if current run is a flatfield run.
        self.is_flatfield = is_flatfield
        self.dark_run_number = dark_run_number
        self.flatfield_run_number = flatfield_run_number

        options = ["slurm", "calib"]
        options.extend(list(map(str.lower, self.METHODS[1:])))
        for option in options:
            option_name = "_".join([option, "opt"])
            attr_name = "_" + option_name
            attr_value = setup_pars.pop(option_name, {})
            setattr(self, attr_name, attr_value)

        #: bool: True if the SLURMCluster is running
        self._cluster_running = False
        self._localcluster = localcluster

        self.analysis = ["meta", "diagnostics"]
        # reduce computation to _compute_dark or _compute_flatfield
        if is_dark:
            self.analysis.extend(["dark"])
        elif is_flatfield:
            self.analysis.extend(["flatfield"])
        else:
            if analysis is not None:
                for flag, method in zip(analysis, self.METHODS[2:]):
                    if int(flag):
                        self.analysis.append(method.lower())

        self.run_number = run_number
        self.proposal = proposal
        self.datdir = setup_pars.pop("datdir", datdir)
        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = RunDirectory(self.datdir)
        self.mask = setup_pars.pop("mask", None)

        self.trainId_offset = trainId_offset
        self.pulses_per_train = pulses_per_train
        self.pulse_step = pulse_step
        self.train_step = train_step
        self.first_train_idx = first_train
        self.last_train_idx = last_train
        self.first_cell = first_cell

        # placeholder set when computing META
        self.cell_ids = None
        self.train_ids = None
        self.selected_train_ids = None
        self.ntrains = None
        self.train_indices = None

        # Experimental Setup
        #: tuple: Position of the direct beam in pixels
        self.center = setup_pars.pop("beamcenter", None)
        self.agipd_geom = setup_pars.pop("geometry", False)

        # save the other entries as attributes
        self.__dict__.update(setup_pars)
        del setup_pars

        #: Xana setup instance
        self.setup = None
        #: np.ndarray: qmap (16, 512, 128)
        self.qmap = None
        self._get_setup()

        #: dict: Structure of the HDF5 file
        self.h5_structure = self._make_h5structure()
        #: str: HDF5 file name.
        self.file_name = None
        self.file_identifier = file_identifier

        #: Calibrator: Calibrator instance for data pre-processing set by _compute_meta
        self._calibrator = None

        # placeholder attributes
        self._cluster = None  # the slurm cluster
        self._client = None  # the slurm cluster client

    def _make_h5structure(self):
        """Create the HDF5 data structure."""
        h5_structure = {
            "META": {
                "/identifiers/pulses_per_train": [True],
                "/identifiers/pulse_ids": [True],
                "/identifiers/train_ids": [None],
                "/identifiers/train_indices": [None],
                "/identifiers/complete_trains": [True],
                "/identifiers/all_trains": [True],
            },
            "DIAGNOSTICS": {
                "/pulse_resolved/xgm/energy": [None],
                "/identifiers/filtered_trains": [None],
                "/pulse_resolved/xgm/pointing_x": [None],
                "/pulse_resolved/xgm/pointing_y": [None],
                "/train_resolved/sample_position/y": [None],
                "/train_resolved/sample_position/z": [None],
                "/train_resolved/linkam-stage/temperature": [None],
            },
            "SAXS": {
                "/pulse_resolved/azimuthal_intensity/q": [True],
                "/pulse_resolved/azimuthal_intensity/phi": [None],
                "/pulse_resolved/azimuthal_intensity/I": [None],
            },
            "XPCS": {
                "/train_resolved/correlation/q": [True],
                "/train_resolved/correlation/t": [True],
                "/train_resolved/correlation/stride": [None],
                # "/train_resolved/correlation/g2": [None],
                "/train_resolved/correlation/ttc": [None],
            },
            "FRAMES": {
                "/average/intensity": [None],
                "/average/variance": [None],
                "/average/image_2d": [None],
                "/utility/mask": [None],
                "/average/train_ids": [None],
            },
            "STATISTICS": {
                "/pulse_resolved/statistics/centers": [True],
                "/pulse_resolved/statistics/counts": [None],
            },
            "DARK": {
                "/dark/intensity": [None],
                "/dark/variance": [None],
                "/dark/mask": [None],
                "/dark/median": [None],
            },
            "FLATFIELD": {
                "/flatfield/intensity": [None],
                "/flatfield/variance": [None],
                "/flatfield/mask": [None],
                "/flatfield/median": [None],
            },
        }
        return h5_structure

    def __str__(self):
        return "MID dataset."

    def __repr(self):
        return f"Dataset({self.setupfile})"

    @property
    def datdir(self):
        """str: Data directory."""
        return self.__datdir

    @datdir.setter
    def datdir(self, path):
        if path is None:
            return
        elif isinstance(path, dict):
            basedir = "/gpfs/exfel/exp/MID/"
            if (
                len(
                    list(
                        filter(
                            lambda x: x in path.keys(),
                            ["cycle", "proposal", "datatype", "run"],
                        )
                    )
                )
                == 4
            ):
                path = (
                    basedir
                    + f"/{path['cycle']}"
                    + f"/p{path['proposal']:06d}"
                    + f"/{path['datatype']}"
                    + f"/r{path['run']:04d}"
                )
        elif isinstance(path, str):
            if bool(re.search("r\d{4}", path)):
                self.run_number = path
            elif isinstance(self.run_number, int):
                path += f"/r{self.run_number:04d}"
            else:
                raise ValueError(
                    "Could not determine run number. Specify run "
                    "number with --run argument or pass data directory "
                    "with run folder (e.g., r0123.)"
                )
        elif isinstance(self.run_number, np.integer) and isinstance(self.proposal, np.integer):
            self.run = open_run(self.run_number, proposal, data='raw')
            self.__datdir = str(Path(self.run.files[0].filename).parent)
        elif isinstance(self.run, extra_data.reader.DataCollection):
            self.__datdir = str(Path(self.run.files[0].filename).parent)
        else:
            raise ValueError(f"Invalid data directory: {path}")

        # if self.is_dark or self.is_flatfield or self.dark_run_number is not None:
        #     path = path.replace("proc", "raw")
        #     print("Switched to raw data format")
        path = os.path.abspath(path)
        if os.path.exists(path):
            self.__datdir = path
        else:
            raise FileNotFoundError(f"Data directory {path} das not exist.")

    @property
    def agipd_geom(self):
        """AGIPD_1MGeometry: AGIPD geometry obtained from extra_data."""
        return self.__agipd_geom

    @agipd_geom.setter
    def agipd_geom(self, geom):
        if isinstance(geom, list):
            geom = AGIPD_1MGeometry.from_quad_positions(geom)
        elif isinstance(geom, str):
            geom = AGIPD_1MGeometry.from_crystfel_geom(geom)
        else:
            raise TypeError(f"Cannot create geometry from {type(geom)}.")

        self.__agipd_geom = geom
        dummy_img = np.zeros((16, 512, 128), "int8")
        if self.center is None:
            self.center = geom.position_modules_fast(dummy_img)[1][::-1]
        else:
            self.center = tuple(map(float, self.center))
        del dummy_img

    @property
    def run_number(self):
        """int: Number of the run."""
        return self.__run_number

    @run_number.setter
    def run_number(self, number):
        if isinstance(number, str):
            number = re.findall("(?<=r)\d{4}", number)
            if len(number):
                number = int(number[0])
            else:
                number = None
        elif isinstance(number, int):
            pass
        elif number is None:
            pass
        elif isinstance(number, extra_data.reader.DataCollection):
            self.run = number
            self.datdir = str(Path(self.run.files[0].filename).parent)
            return # run_number is updated in datdir setter
        else:
            raise TypeError(f"Invalid run number type {type(number)}.")
        self.__run_number = number

    @staticmethod
    def _read_setup(setupfile):
        """read setup parameters from config file

        Args:
            setupfile (str): Path to the setup file (YAML).

        Returns:
            dict: containing the setup parameters read from the setupfile.
        """
        with open(setupfile) as file:
            setup_pars = yaml.load(file, Loader=yaml.FullLoader)

            # include dx and dy in quadrant position
            if "quadrant_positions" in setup_pars.keys():
                quad_dict = setup_pars["quadrant_positions"]
                dx, dy = [quad_dict[x] for x in ["dx", "dy"]]
                quad_pos = [
                    (dx + quad_dict[f"q{x}"][0], dy + quad_dict[f"q{x}"][1])
                    for x in range(1, 5)
                ]
                if "geometry" not in setup_pars.keys():
                    setup_pars["geometry"] = quad_pos
                else:
                    print(
                        "Quadrant positions and geometry file defined by \
                            setupfile. Loading geometry file..."
                    )

            # if the key exists in the setup file without any entries, None is
            # returned. We convert those entries to empty dictionaires.
            for key, value in list(setup_pars.items()):
                if value is None:
                    setup_pars.pop(key)
            return setup_pars

    @property
    def mask(self):
        """np.ndarray: shape(16,512,128) Mask where `bad` pixels are 0
        and `good` pixels 1.
        """
        return self.__mask

    @mask.setter
    def mask(self, mask):
        local_mask_file = os.path.join(self.out_dir, "mask.npy")
        if os.path.isfile(local_mask_file):
            mask = np.load(local_mask_file)
            print(f"Loaded local mask-file: {local_mask_file}")
        elif mask is None:
            mask = np.ones((16, 512, 128), "bool")
        elif isinstance(mask, str):
            try:
                mask = np.load(mask)
            except Exception as e:
                print("Loading the mask failed. Error: ", e)
        elif isinstance(mask, np.ndarray):
            if mask.shape != (16, 512, 128):
                try:
                    mask = mask.reshape(16, 512, 128)
                except Exception as e:
                    print("Mask array has invalid shape: ", mask.shape)
        else:
            raise TypeError(f"Cannot read mask of type {type(mask)}.")

        self.__mask = np.array(mask).astype("bool")

    @staticmethod
    def _get_good_trains(run):
        """Function that finds all complete trains of the run.

        Args:
            run (DataCollection): XFEL data, e.g.,
                obtained from extra_data.RundDirectory.

        Return:
            tuple(np.ndarry, np.ndarry): First array contains the train_ids.
                Second array contains the corresponding indices.
        """

        "Stolen from Robert Rosca"
        trains_with_images = {}
        det_sources = filter(lambda x: "AGIPD1M" in x, run.detector_sources)
        for source_name in det_sources:
            good_trains_source = set()
            for source_file in run._source_index[source_name]:
                good_trains_source.update(
                    set(
                        source_file.train_ids[
                            source_file.get_index(source_name, "image")[1].nonzero()
                        ]
                    )
                )

            trains_with_images[source_name] = good_trains_source

        good_trains = sorted(
            reduce(set.intersection, list(trains_with_images.values()))
        )
        good_trains = np.array(good_trains)
        good_indices = np.intersect1d(run.train_ids, good_trains, return_indices=True)[
            1
        ]

        return good_trains, good_indices

    def _get_pulse_pattern(self, pulses_per_train=500, pulse_step=1):
        """determine pulses pattern from machine source"""
        source = "MID_RR_SYS/MDL/PULSE_PATTERN_DECODER"
        if os.path.isfile(self.pulse_file):
            pass
        elif source in self.run.all_sources and not (self.is_dark or self.is_flatfield):
            pulses = self.run.get_array(source, "sase2.pulseIds.value")
            pulse_spacing = np.unique(pulses.where(pulses > 199).diff("dim_0"))
            pulse_spacing = pulse_spacing[np.isfinite(pulse_spacing)].astype("int32")
            pulseIds = pulses // 200
            npulses = np.unique((pulseIds > 0).sum("dim_0"))
            if len(npulses) == 1:
                npulses = min(npulses[0], pulses_per_train)
                print(f"Analyzing {npulses} pulses per train.")
            else:
                print(f"Found various number of pulses per train {npulses}.")
                npulses = min(max(npulses), pulses_per_train)
            if len(pulse_spacing) == 1:
                pulse_spacing = pulse_step  # max(pulse_spacing[0]//2, pulse_step)
                print(f"Using pulse spacing of {pulse_spacing}.")
            else:
                print(
                    f"Varying the pulse spacing is not supported. Found {pulse_spacing}"
                )
                pulse_spacing = pulse_step  # max(min(pulse_spacing[0]//2), pulse_step)
                print(f"Using {pulse_spacing} cell_spacing")
        else:
            pulse_spacing = pulse_step
            npulses = pulses_per_train
            print(
                "Pulse pattern decoder not found. Using: "
                f"pulse spacing {pulse_spacing} and "
                f"{npulses} pulses per train"
            )
        return npulses, pulse_spacing

    def _get_cell_ids(self, train_idx=0):
        source = "MID_DET_AGIPD1M-1/DET/{}CH0:xtdf"
        i = 0
        while i < 500:
            for module in range(0, 16, 2):
                try:
                    s = source.format(module)
                    tid, train_data = self.run.select(
                        s, "image.pulseId"
                    ).train_from_index(train_idx)
                    cell_ids = train_data[s]["image.pulseId"]
                    return np.array(cell_ids).flatten()
                except (KeyError, ValueError):
                    pass
            i += 1
            train_idx += 10

        raise ValueError(
            "Unable to determine pulse ids. Probably the data "
            "source was not available."
        )

    def _get_setup(self):
        """initialize the Xana setup and the azimuthal integrator"""
        self.setup = Setup(detector="agipd1m")
        self.setup.mask = self.mask.copy()
        self.setup.make(
            **dict(
                center=self.center,
                wavelength=12.39 / self.photon_energy,
                distance=self.sample_detector,
            )
        )
        dist = self.agipd_geom.to_distortion_array()
        self.setup.detector.IS_CONTIGUOUS = False
        self.setup.detector.set_pixel_corners(dist)
        self.setup._update_ai()
        qmap = self.setup.ai.array_from_unit(unit="q_nm^-1")
        phimap = self.setup.ai.chiArray()
        #: np.ndarray: q-map
        self.qmap = qmap.reshape(16, 512, 128)
        #: np.ndarray: azimhuthal-angle-map
        self.phimap = phimap.reshape(16, 512, 128)

        self.setup.mask = self.setup.mask.reshape(-1,128)
        self.setup = get_q_phi_pixels(self.setup, self._xpcs_opt)
        self.setup.mask = self.setup.mask.reshape(16,512,128)

        #save array of rois
        img = np.zeros((16,512,128))
        img[~self.setup.mask] = np.nan
        img = img.reshape(-1, 128)
        for i, roi in enumerate(self.setup.qroi):
            img[roi] = i+1
        img = img.reshape(16,512,128)
        np.save(f"{self.out_dir}/rois.npy", self.__agipd_geom.position_modules_fast(img)[0])


    def _start_slurm_cluster(self):
        """Initialize the slurm cluster"""

        opt = self._slurm_opt
        # nprocs = 72//threads_per_process
        nprocs = opt.pop("nprocs", 1)
        threads_per_process = opt.pop("cores", 40)
        if self.is_dark or self.is_flatfield:
            nprocs = 1
        if isinstance(self.ntrains, np.integer):
            default_number_jobs = min(max(int(self.ntrains / 64), 4), 4)
        else:
            default_number_jobs = 4
        njobs = opt.pop("njobs", default_number_jobs)
        if self._localcluster:
            self._cluster = LocalCluster(
                n_workers=nprocs,
            )
        else:
            print(f"Submitting {njobs} jobs using {nprocs} processes per job.")
            self._cluster = SLURMCluster(
                queue=opt.get("partition", opt.pop("partition","upex")),
                processes=nprocs,
                cores=threads_per_process,
                memory=opt.pop("memory", "400GB"),
                log_directory="./dask_log/",
                local_directory="/scratch/",
                nanny=True,
                death_timeout=60 * 60,
                walltime="6:00:00",
                interface=opt.get('interface', "ib0"),  # or 'eth0'
                name="midtools",
            )
            self._cluster.scale(nprocs * njobs)
            # self._cluster.adapt(maximum_jobs=njobs)

        print(self._cluster)
        self._client = Client(self._cluster)
        print("Cluster dashboard link:", self._cluster.dashboard_link)

    def _stop_slurm_cluster(self):
        """Shut down the slurm cluster"""
        self._client.close()
        self._cluster.close()

    def _create_output_file(self):
        """Create the HDF5 output file."""
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        if self.is_dark:
            ftype = "dark"
        elif self.is_flatfield:
            ftype = "flatfield"
        else:
            ftype = "analysis"

        if self.file_identifier is None:
            # check existing files and determine counter
            existing = os.listdir(self.out_dir)
            search_str = f"(?<=r{self.run_number:04}-{ftype})" ".*\d{3,}(?=\.h5)"

            counter = map(re.compile(search_str).search, existing)
            counter = filter(lambda x: bool(x), counter)
            counter = list(map(lambda x: int(x[0][-3:]), counter))

            identifier = max(counter) + 1 if len(counter) else 0
        else:
            identifier = self.file_identifier

        # depricated include last and first
        # if self.is_dark or self.is_flatfield:
        #     filename = f"r{self.run_number:04}-analysis_{identifier:03}.h5"
        # else:
        #     # filename = f"r{self.run_number:04}-analysis_{self.first_train_idx}-{self.last_train_idx}_{identifier:03}.h5"
        #     filename = f"r{self.run_number:04}-analysis_{identifier:03}.h5"

        # this part is just a placeholder and simply creates the HDF5-file
        while True:
            try:
                filename = f"r{self.run_number:04}-{ftype}_{identifier:03}.h5"
                self.file_name = os.path.join(self.out_dir, filename)
                with h5.File(self.file_name, "a") as f:
                    for flag, method in zip(self.analysis, self.METHODS):
                        pass
                        # for path,(shape,dtype) in self.h5_structure[method].items():
                        #     pass
                        #     # f.create_dataset(path, shape=shape, dtype=dtype)
                break
            except OSError as e:
                print(str(e))
                print("Incrementing counter")
                if self.file_identifier is None:
                    identifier += 1

        with open(self.setupfile) as f:
            setup_pars = yaml.load(f, Loader=yaml.FullLoader)

        # attributes to save in copied setupfile
        attrs = [
            "datdir",
            "is_dark",
            "dark_run_number",
            "run_number",
            "pulses_per_train",
            "pulse_step",
            "is_flatfield",
            "flatfield_run_number",
        ]
        setup_pars.update({attr: getattr(self, attr) for attr in attrs})
        setup_pars["analysis"] = self.analysis

        # copy the setupfile
        new_setupfile = f"r{self.run_number:04}-setup_{identifier:03}.yml"
        new_setupfile = os.path.join(self.out_dir, new_setupfile)

        with open(new_setupfile, "w") as f:
            yaml.dump(setup_pars, f)

        self.setupfile = new_setupfile
        print(f"Filename: {self.file_name}")
        print(f"Setupfile: {self.setupfile}")

    def compute(self, create_file=True):
        """Start the actual computation based on the analysis attribute."""

        if create_file:
            self._create_output_file()
        try:
            for method in self.analysis:
                if method not in ["meta", "diagnostics"] and not self._cluster_running:
                    print(f"\n{'Fetching Data':-^50}")
                    self._start_slurm_cluster()
                    self._cluster_running = True
                    print("Initializing Data Calibrator...")
                    self._init_data_calibrator()
                    if self._calibrator.data is None:
                        self._calibrator._get_data()
                print(f"\n{method.upper():-^50}")
                repetition = 0
                while repetition < 5:
                    print_now()
                    success = getattr(self, f"_compute_{method.lower()}")()
                    if success:
                        break
                    else:
                        print(f"Compute {method} failed repetition {repetition}")
                        repetition += 1
                        self._client.restart()
                        self._calibrator._get_data()
                # print(f"{' Done ':-^50}")
                # if self._cluster_running:
                #     print('Restarting Cluster')
                #     self._client.restart()
        finally:
            if self._client is not None:
                self._stop_slurm_cluster()

    def _compute_meta(self):
        """Find complete trains."""
        # determine trainIds and pulseIds and their spacing
        self.pulses_per_train, self.pulse_step = self._get_pulse_pattern(
            self.pulses_per_train, self.pulse_step
        )

        #: np.ndarray: Array of pulse IDs.
        self.cell_ids = self._get_cell_ids()
        first_cell_index = np.where(self.cell_ids == self.first_cell)[0][0]
        cell_slice = slice(
            first_cell_index,
            first_cell_index + self.pulses_per_train * self.pulse_step,
            self.pulse_step,
        )
        self.cell_ids = self.cell_ids[cell_slice]
        #: int: Number of X-ray pulses per train.
        self.pulses_per_train = min(len(self.cell_ids), self.pulses_per_train)
        #: float: Delay time between two successive pulses.
        self.pulse_delay = np.diff(self.cell_ids)[0] * 220e-9

        all_trains = self.run.train_ids
        complete_train_ids, self.train_indices = self._get_good_trains(self.run)
        self.train_ids = complete_train_ids.copy()
        if os.path.isfile(self.train_file):
            print(f"Loaded train-file {self.train_file}")
            train_ids = np.load(self.train_file).astype("int64")
            self.train_ids, self.train_indices, _ = np.intersect1d(
                all_trains, train_ids, return_indices=True
            )

        print(f"{self.train_ids.size} of {len(all_trains)} trains are complete.")

        #: int: last train index to compute
        self.last_train_idx = min([self.last_train_idx, len(self.train_ids)])

        if self.first_train_idx > len(self.train_ids):
            raise ValueError(
                "First train index {self.first_train_idx} > number of trains {len(self.train_ids)}"
            )

        self.train_ids = self.train_ids[self.first_train_idx : self.last_train_idx]
        self.selected_train_ids = self.train_ids.copy()
        self.ntrains = len(self.train_ids)
        self.train_indices = self.train_indices[
            self.first_train_idx : self.last_train_idx
        ]
        print(
            f"Processing train {self.first_train_idx} to {self.last_train_idx}\n"
            f"{self.ntrains} of {len(all_trains)} trains will be processed."
        )

        data = {
            "pulses_per_train": self.pulses_per_train,
            "pulse_ids": self.cell_ids,
            "train_ids": self.train_ids,
            "train_indices": self.train_indices,
            "complete_train_ids": complete_train_ids,
            "all_train_ids": all_trains,
        }

        return self._write_to_h5(data, "META")

    def _init_data_calibrator(self):
        self._calibrator = Calibrator(
            self.run,
            cell_ids=self.cell_ids,
            train_ids=self.selected_train_ids,
            dark_run_number=self.dark_run_number,
            mask=self.mask.copy(),
            is_dark=self.is_dark,
            is_flatfield=self.is_flatfield,
            flatfield_run_number=self.flatfield_run_number,
            **self._calib_opt,
        )

    def _compute_diagnostics(self):
        """Read diagnostic data."""

        diagnostics_opt = dict(
                xgm_threshold=0,
        )
        diagnostics_opt.update(self._diagnostics_opt)

        print(f"Read XGM and control sources.")
        sources = {
            "SA2_XTD1_XGM/XGM/DOOCS:output": [
                "data.intensityTD",
                "data.xTD",
                "data.yTD",
            ],
            **{
                f"HW_MID_EXP_SAM_MOTOR_SSHEX_{x}": ["actualPosition.value"]
                for x in "YZ"
            },
            "MID_EXP_UPP/TCTRL/LINKAM": ["temperature.value"],
        }
        sources = dict(filter(lambda x: x[0] in self.run.all_sources, sources.items()))

        arr = {}
        for source in sources:
            for key in sources[source]:
                data = self.run.get_array(source, key)
                # we subtract 1 from the trainId to account for the shift
                # between AGIPD and other data sources.
                sel_trains = np.intersect1d(data.trainId, self.train_ids - self.trainId_offset)
                data = data.sel(trainId=sel_trains)
                method = None if len(sel_trains) == len(self.train_ids) else "nearest"
                data = data.reindex(trainId=self.train_ids - self.trainId_offset, method=method)
                if len(data.dims) == 2:
                    data = data[:, : len(self.cell_ids)]
                arr["/".join((source, key))] = data
                if ("XGM" in source) and (key == "data.intensityTD") and (diagnostics_opt['xgm_threshold']>0):
                    self.selected_train_ids = np.intersect1d(data.isel(
                        trainId=data.mean("dim_0") > diagnostics_opt['xgm_threshold']
                    ).trainId, self.train_ids)
                    print(f"{self.selected_train_ids.size}/{self.train_ids.size} remain after filtering by XGM threshold.")
                    arr['filtered_trains'] = self.selected_train_ids
                    self.ntrains = len(self.selected_train_ids)
                else:
                    if not 'filtered_trains' in arr:
                        arr['filtered_trains'] = self.train_ids

        return self._write_to_h5(arr, "DIAGNOSTICS")

    def _compute_saxs(self):
        """Perform the azimhuthal integration."""

        saxs_opt = dict(
            mask=self.mask,
            geom=self.agipd_geom,
            distortion_array=self.agipd_geom.to_distortion_array(),
            sample_detector=self.sample_detector,
            photon_energy=self.photon_energy,
            center=self.center,
            setup=self.setup,
            integrate='1D',
        )
        saxs_opt.update(self._saxs_opt)

        print("Compute pulse resolved SAXS", flush=True)
        out = azimuthal_integration(self._calibrator, method="single", **saxs_opt)

        if out["azimuthal-intensity"].shape[:2] == (
            self.ntrains,
            self.pulses_per_train,
        ):
            # out['azimuthal-intensity'].to_netcdf(self.file_name.replace('.h5', '.nc'))
            # print('wrote NetCDF file')
            print_now()
            print("Start writing to HDF5", flush=True)
            return self._write_to_h5(out, "SAXS")
        else:
            return False

    def _compute_xpcs(self):
        """Calculate correlation functions."""

        xpcs_opt = dict(
            mask=self.mask,
            qmap=self.qmap,
            setup=self.setup,
            dt=self.pulse_delay,
            use_multitau=False,
            rebin_g2=False,
            norm='symmetric',
            h5filename=self.file_name,
            method="intra_train",
        )
        xpcs_opt.update(self._xpcs_opt)

        print("Compute XPCS correlation funcions.", flush=True)
        out = correlate(self._calibrator, **xpcs_opt)

        if out["ttc"].shape[0] == self.ntrains:
            print_now()
            print("Start writing to HDF5", flush=True)
            return self._write_to_h5(out, "XPCS")
        else:
            return False

    def _compute_frames(self):
        """Averaging frames."""

        frames_opt = dict(axis="pulse", trainIds=self.selected_train_ids, max_trains=10)
        frames_opt["trainIds"] = frames_opt["trainIds"][
            :: frames_opt["trainIds"].size // frames_opt["max_trains"]
        ]
        frames_opt.update(self._frames_opt)

        print("Computing frames.", flush=True)
        out = average(self._calibrator, **frames_opt)

        img2d = self.agipd_geom.position_modules_fast(out["average"])[0]
        out["image2d"] = img2d

        out = self._update_mask(out)
        out["averaged_trains"] = frames_opt["trainIds"][: out["average"].shape[0]]

        print_now()
        print("Start writing to HDF5", flush=True)
        return self._write_to_h5(out, "FRAMES")

    def _update_mask(self, frames):
        """Update the mask based on the average intensity and variance.

        Args:
            frames (dict): should contain the keys `average` and `variance`.
        """
        assert ("average" in frames) and ("variance" in frames)

        print("Updating mask based on average image")
        print(
            f"Initially: masked {round((1-self.mask.sum()/self.mask.size)*100,2)}% of all pixels"
        )

        mask = mask_radial(frames["average"], self.qmap, self.mask)
        print(f"Average: masked {round((1-mask.sum()/mask.size)*100,2)}% of all pixels")

        mask = mask_radial(
            frames["variance"].mean(0) / frames["average"].mean(0) ** 2,
            self.qmap,
            mask=mask,
            lower_quantile=0.01,
        )
        print(
            f"Variance: masked {round((1-mask.sum()/mask.size)*100,2)}% of all pixels"
        )

        mask = mask_asics(mask)
        print(f"Asics: masked {round((1-mask.sum()/mask.size)*100,2)}% of all pixels")

        self.mask = mask
        self._calibrator.xmask = mask
        frames["mask"] = mask
        return frames

    def _compute_dark(self):
        """Averaging darks."""

        dark_opt = dict(
            axis="train", trainIds=self.train_ids, max_trains=len(self.train_ids)
        )
        dark_opt.update(self._dark_opt)
        # we need to remove the options for masking before passing the dict
        # to the average method
        pvals = dark_opt.pop("pvals", (0.2, 0.5))

        print("Computing darks.")
        out = average(self._calibrator, **dark_opt)
        darkmask, median = _create_mask_from_dark(
            out["average"], out["variance"], pvals=pvals
        )
        out["darkmask"] = darkmask
        out["median"] = median

        return self._write_to_h5(out, "DARK")

    def _compute_flatfield(self):
        """Process flatfield."""

        flatfield_opt = dict(
            axis="train", trainIds=self.train_ids, max_trains=len(self.train_ids)
        )
        flatfield_opt.update(self._flatfield_opt)

        # we need to remove the options for masking before passing the dict
        # to the average method
        average_limits = flatfield_opt.pop("average_limits", (3500, 6000))
        variance_limits = flatfield_opt.pop("variance_limits", (200, 1500))

        print("Computing flatfield.")
        out = average(self._calibrator, **flatfield_opt)
        ff_mask, median = _create_mask_from_flatfield(
            out["average"],
            out["variance"],
            average_limits=average_limits,
            variance_limits=variance_limits,
        )
        out["ffmask"] = ff_mask
        out["median"] = median

        return self._write_to_h5(out, "FLATFIELD")

    def _compute_statistics(self):
        """Calculate Histograms per image."""

        statistics_opt = dict(
            mask=self.mask,
            setup=self.setup,
            geom=self.agipd_geom,
            max_trains=200,
        )
        statistics_opt.update(self._statistics_opt)

        print("Compute pulse resolved statistics")
        out = statistics(self._calibrator, **statistics_opt)

        if out["counts"].shape[0] == (self.ntrains * self.pulses_per_train):
            return self._write_to_h5(out, "STATISTICS")
        else:
            return False

    def _write_to_h5(self, output, method):
        """Dump results in HDF5 file."""
        if self.file_name is not None:
            keys = list(self.h5_structure[method.upper()].keys())
            with h5.File(self.file_name, "r+") as f:
                for keyh5, (outkey, data) in zip(keys, output.items()):
                    if data is not None:
                        # check scalars and fixed_size
                        if np.isscalar(data):
                            f[keyh5] = data
                        else:
                            # print(outkey, keyh5, np.asarray(data).shape)
                            f.create_dataset(
                                keyh5,
                                data=np.asarray(data),
                                compression="gzip",
                                chunks=True,
                                # compression_opts=4,
                            )
            print("Wrote data to HDF5 file")
            return True
        else:
            warnings.warn("Results not saved. Filename not specified")
            return output

    def merge_files(self, subset=None, delete_file=False):
        """merge existing HDF5 files for a run"""
        datasets = get_datasets(self.out_dir)
        ds = Interpreter(datasets)
        all_trains = [t[1] for t in ds.iter_trainids(self.run_number, subset)]
        files = [x[1] for x in ds.iter_files(self.run_number, subset)]
        all_trains = np.unique(np.hstack(all_trains))
        ind_in_files = []
        first_in_all = []
        filenames = []
        for file_index, (index, trains) in enumerate(
            ds.iter_trainids(self.run_number, subset)
        ):
            if len(trains) == 0:
                continue
            ind_in_file, ind_all = np.intersect1d(
                trains, all_trains, return_indices=True
            )[1:]
            if not len(ind_in_file):
                continue
            ind_in_files.append(ind_in_file)
            first_in_all.append(ind_all[0])
            all_trains = np.delete(all_trains, ind_all)
            filenames.append(files[file_index])

        file_order = np.argsort(first_in_all)
        filenames = np.array(filenames)[file_order]

        self._create_output_file()

        keys = [x for y in self.h5_structure.values() for x in y.keys()]
        with h5.File(self.file_name, "a") as F:

            for filename in filenames:
                with h5.File(filename, "r") as f:
                    for method in self.h5_structure:
                        for key, value in self.h5_structure[method].items():
                            fixed_size = bool(value[0])
                            if key in f:
                                data = f[key]
                                s = data.shape
                                if not key in F:
                                    # check scalars and fixed_size
                                    if len(s) == 0:
                                        F[key] = np.array(data)
                                    else:
                                        F.create_dataset(
                                            key,
                                            data=data,
                                            compression="gzip",
                                            chunks=True,
                                            maxshape=(None, *s[1:]),
                                        )
                                else:
                                    if not fixed_size:
                                        F[key].resize((F[key].shape[0] + s[0]), axis=0)
                                        F[key][-s[0] :] = data
                if delete_file:
                    os.remove(filename)


def _get_parser():
    """Command line parser"""
    parser = argparse.ArgumentParser(
        prog="midtools",
        description="Analyze MID runs.",
    )
    parser.add_argument(
        "setupfile",
        type=str,
        help="the YAML file to configure midtools",
    )
    parser.add_argument(
        "analysis",
        type=str,
        help=(
            "which analysis to perform. List of 0s and 1s:\n"
            "1000 saves average data along specific axis,\n"
            "0100 SAXS routines,\n"
            "0010 XPCS routines,\n"
            "0001 statistics (histograms pulse resolved."
        ),
    )
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        help="Run number.",
        default=None,
    )
    parser.add_argument(
        "-dr",
        "--dark-run-number",
        type=int,
        help="Dark run number.",
        default=None,
        nargs=2,
    )
    parser.add_argument(
        "--last-train",
        type=int,
        help="last train to analyze.",
        default=1_000_000,
    )
    parser.add_argument(
        "--first-train",
        type=int,
        help="first train to analyze.",
        default=0,
    )
    parser.add_argument(
        "-ppt",
        "--pulses-per-train",
        type=int,
        help="number of pulses per train",
        default=500,
    )
    parser.add_argument(
        "-ts",
        "--train-step",
        type=int,
        help="spacing of trains",
        default=1,
    )
    parser.add_argument(
        "-ps",
        "--pulse-step",
        type=int,
        help="spacing of pulses",
        default=1,
    )
    parser.add_argument(
        "--is-dark",
        help="whether the run is a dark run",
        const=True,
        default=False,
        nargs="?",
    )
    parser.add_argument(
        "--is-flatfield",
        help="whether the run is a flatfield run",
        const=True,
        default=False,
        nargs="?",
    )
    parser.add_argument(
        "-ffr",
        "--flatfield-run",
        type=int,
        help="Flatfield run number.",
        default=None,
        nargs=2,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Output directory",
        default="./",
        nargs="?",
    )
    parser.add_argument(
        "--chunk",
        nargs="?",
        default=None,
        type=int,
        required=False,
        help="Split the number of trains in chunks of this size (default do not chunk)",
    )
    parser.add_argument(
        "--job-dir",
        default="/gpfs/exfel/data/scratch/reiserm/mid-proteins/jobs/",
        required=False,
        help="Directory for the slurm output and error files",
    )
    parser.add_argument(
        "--slurm",
        default=False,
        const=True,
        nargs="?",
        required=False,
        help="Run midtools on dedicated node with slurm job (default False)",
    )
    parser.add_argument(
        "--localcluster",
        default=False,
        const=True,
        nargs="?",
        required=False,
        help="Use dasks LocalCluster to run midtools locally (default False)",
    )
    parser.add_argument(
        "--file-identifier",
        default=None,
        type=int,
        nargs="?",
        required=False,
        help="Identifier at file ending. Default None.",
    )
    parser.add_argument(
        "--first-cell",
        type=int,
        help="Cell ID of the first AGIPD memory cell with X-rays.",
        required=False,
        default=2,
    )
    parser.add_argument(
        "--datdir",
        required=False,
        default=None,
        help="Path to the data. This argument is only used if the data directory is not provided in the setupfile.",
    )
    return parser


@_exception_handler(max_attempts=3)
def _submit_slurm_job(run, args, test=False):
    args = dict(args)
    job_dir = args.pop("job_dir", "./jobs")
    job_dir = os.path.abspath(job_dir)

    setupfile = args.pop("setupfile")
    analysis = args.pop("analysis")
    for key, val in list(args.items()):
        if not bool(val):
            args.pop(key)
        elif isinstance(val, (list, tuple)):
            args[key] = " ".join(map(str, val))

    midtools_args = " ".join(
        [f"--{arg.replace('_','-')} {val}" for arg, val in args.items()]
    )

    print(f"Generating sbatch jobs for run: {run}")

    if not os.path.exists(job_dir) and not test:
        os.mkdir(job_dir)

    env_dir = '/'.join(sys.executable.split('/')[:-2])

    TEMPLATE = """#!/bin/bash
#SBATCH --job-name=midtools
#SBATCH --output={job_file}.out
#SBATCH --error={job_file}.err
#SBATCH --partition=upex
#SBATCH --exclusive
#SBATCH --time 06:00:00

source {env_dir}/bin/activate
echo "SLURM_JOB_ID           $SLURM_JOB_ID"
type midtools

ulimit -n 4096
ulimit -c unlimited
midtools {setupfile} {analysis} -r {run} {midtools_args}

exit
"""

    print(f"With arguments: `{midtools_args}`")

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    random_id = np.random.randint(0, 999)
    job_name = f"{timestamp}-{random_id:03}-run{run}"
    job_file = f"{job_dir}/{job_name}.job"

    job = TEMPLATE.format_map(
        {
            "setupfile": setupfile,
            "analysis": analysis,
            "run": run,
            "job_name": job_name,
            "job_file": job_file,
            "midtools_args": midtools_args,
            "env_dir": env_dir,
        }
    )


    print(f"Generating and submitting sbatch for job {job_name}")

    if not test:
        with open(job_file, "w") as f:
            f.write(job)

        os.system(f"sbatch {job_file}")
    else:
        print(job)


def run_single(run_number, setupfile, analysis, **kwargs):

    t_start = time.time()

    dataset = Dataset(run_number, setupfile, analysis=analysis, **kwargs)
    print("Protein Mode")
    print(f"\n{' Starting Analysis ':-^50}")
    print(f"Analyzing {dataset.datdir}")
    dataset.compute()

    elapsed_time = time.time() - t_start
    print(f"\nFinished: elapsed time: {elapsed_time/60:.2f}min")
    print(f"Results saved under {dataset.file_name}\n")


def main():
    parser = _get_parser()
    args = vars(parser.parse_args())

    run = args.pop("run")
    if args["chunk"] is None:
        if args["slurm"]:
            args["slurm"] = False
            _submit_slurm_job(run, args)
        else:
            setupfile = args.pop("setupfile")
            analysis = args.pop("analysis")
            run_single(run, setupfile, analysis, **args)
    else:
        args["slurm"] = False
        first, last = args["first_train"], args["last_train"]
        chunksize = args.pop("chunk")
        for first in range(first, last, chunksize):
            args["first_train"] = first
            args["last_train"] = first + chunksize
            _submit_slurm_job(run, args)


if __name__ == "__main__":
    main()
