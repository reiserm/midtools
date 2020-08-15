#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import yaml
import re
import sys
import time
import numpy as np
import h5py as h5
import argparse
from shutil import copyfile

# XFEL packages
from extra_data import RunDirectory
from extra_geom import AGIPD_1MGeometry

# Dask
import dask
import dask_jobqueue
import dask.array as da
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar

import Xana
from Xana import Setup

from .azimuthal_integration import azimuthal_integration
from .correlations import correlate
from .average import average
from .corrections import Calibrator

from functools import reduce

import pdb

class Dataset:

    METHODS = ['META', 'DIAGNOSTICS', 'SAXS', 'XPCS', 'FRAMES', 'DARK']

    def __init__(self, setupfile, analysis='00', last_train=1e6,
            run_number=None, dark_run_number=None, pulses_per_train=500,
            train_step=1, pulse_step=1):
        """Dataset object to analyze MID datasets on Maxwell.

        Args:
            setupfile (str): Setupfile (.yml) that contains information on the
                setup parameters.
            analysis (str, optional): Flags of the analysis to perform.
                Defaults to '00'. `analysis` is a string of ones and zeros
                where a one means to perform the analysis and a zero means to
                omit the analysis. The analysis types are:

                +--------+----------------------------+
                | flags- | analysis                   |
                +========+============================+
                |  1000  | SAXS azimuthal integration |
                +--------+----------------------------+
                |  0100  | XPCS correlation functions |
                +--------+----------------------------+
                |  0010  | Average frames             |
                +--------+----------------------------+
                |  0001  | Compute darks              |
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

        #: str: Path to the setupfile.
        self.setupfile = setupfile
        setup_pars = self._read_setup(setupfile)

        options = ['slurm', 'calib']
        options.extend(list(map(str.lower, self.METHODS[2:])))
        for option in options:
            option_name = "_".join([option, 'opt'])
            attr_name = "_" + option_name
            attr_value = setup_pars.pop(option_name, {})
            setattr(self, attr_name, attr_value)

        #: bool: True if the SLURMCluster is running
        self._cluster_running  = False

        #: str: Flags of the analysis methods.
        self.analysis = '11' + analysis
        self.run_number = run_number
        self.datdir = setup_pars.pop('datdir', False)
        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = RunDirectory(self.datdir)
        self.mask = setup_pars.pop('mask', None)

        #: int: Number of X-ray pulses per train.
        self.pulse_ids = self._get_pulse_ids()
        #: np.ndarray: Array of pulse IDs.
        self.pulses_per_train = min([len(self.pulse_ids), pulses_per_train])
        self.pulse_ids = self.pulse_ids[:self.pulses_per_train]
        #: float: Delay time between two successive pulses.
        self.pulse_delay = np.diff(self.pulse_ids)[0]*220e-9
        #: np.ndarray: All train IDs.
        self.train_ids = np.array(self.run.train_ids)
        #: int: last train index to compute
        self.last_train_idx = min([last_train,len(self.train_ids)])
        #: int: Number of complete trains
        self.ntrains = min([len(self.train_ids), self.last_train_idx])
        #: np.ndarray: All train indices.
        self.train_indices = np.arange(self.train_ids.size)

        # Experimental Setup
        #: tuple: Position of the direct beam in pixels
        self.center = None
        self.agipd_geom = setup_pars.pop('geometry', False)

        # save the other entries as attributes
        self.__dict__.update(setup_pars)
        del setup_pars

        self.setup = Setup(detector='agipd1m')
        self.setup.mask = self.mask.copy()
        self.setup.make(**dict(center=self.center,
            wavelength=12.39/self.photon_energy,
            distance=self.sample_detector,))
        dist = self.agipd_geom.to_distortion_array()
        self.setup.detector.IS_CONTIGUOUS = False
        self.setup.detector.set_pixel_corners(dist)
        self.setup._update_ai()
        qmap = self.setup.ai.array_from_unit(unit='q_nm^-1');
        #: np.ndarray: q-map
        self.qmap = qmap.reshape(16,512,128);

        #: dict: Structure of the HDF5 file
        self.h5_structure = self._make_h5structure()
        #: str: HDF5 file name.
        self.file_name = None

        #: Calibrator: Calibrator instance for data pre-processing
        self._calibrator = Calibrator(self.run,
                                      last_train=self.last_train_idx,
                                      train_step=train_step,
                                      pulse_step=pulse_step,
                                      dark_run_number=dark_run_number,
                                      mask=self.mask.copy(),
                                      **self._calib_opt)

        # placeholder attributes
        self._cluster = None # the slurm cluster
        self._client = None # the slurm cluster client


    def _make_h5structure(self):
        """Create the HDF5 data structure.
        """
        h5_structure = {
           'META': {
                "/identifiers/pulses_per_train": [(1,), 'int8'],
                "/identifiers/pulse_ids": [(self.pulses_per_train,), 'int8'],
                "/identifiers/train_ids": [(self.ntrains,), 'int16'],
                "/identifiers/train_indices": [(self.ntrains,), 'int8'],
            },
            'DIAGNOSTICS': {
                "/pulse_resolved/xgm/energy": [
                    (self.ntrains,self.pulses_per_train), 'float32', ],
            },
            'SAXS': {
                "/pulse_resolved/azimuthal_intensity/q": [(500,), 'float32'],
                "/pulse_resolved/azimuthal_intensity/I": [
                    (self.ntrains*self.pulses_per_train, 500), 'float32',],
            },
            'XPCS': {
                "/train_resolved/correlation/q": [None, None],
                "/train_resolved/correlation/t": [None, None],
                "/train_resolved/correlation/g2": [None, None],
            },
            'FRAMES': {
                "/average/intensity": [None, None],
                "/average/variance": [None, None],
                "/average/image_2d": [None, 'float32'],
            },
            'DARK': {
                "/dark/intensity": [None, None],
                "/dark/variance": [None, None],
            },
            'STATISTICS': {
                "/pulse_resolved/histogram": [None, None],
                "/pulse_resolved/centers": [None, None],
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
        if isinstance(path, dict):
            basedir = '/gpfs/exfel/exp/MID/'
            if len(list(filter(lambda x: x in path.keys(),
                             ['cycle', 'proposal', 'datatype', 'run']))) == 4:
                path = basedir \
                    + f"/{path['cycle']}" \
                    + f"/p{path['proposal']:06d}" \
                    + f"/{path['datatype']}" \
                    + f"/r{path['run']:04d}"
        elif isinstance(path, str):
            if bool(re.search("r\d{4}",path)):
                self.run_number = path
            elif isinstance(self.run_number, int):
                path += f"/r{self.run_number:04d}"
            else:
                raise ValueError("Could not determine run number. Specify run "
                        "number with --run argument or pass data directory "
                        "with run folder (e.g., r0123.)")
        else:
            raise ValueError(f'Invalid data directory: {path}')

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
            raise TypeError(f'Cannot create geometry from {type(geom)}.')

        self.__agipd_geom = geom
        dummy_img = np.zeros((16,512,128), 'int8')
        self.center = geom.position_modules_fast(dummy_img)[1][::-1]
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
        else:
            raise TypeError(f'Invalid run number type {type(number)}.')
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
            if 'quadrant_positions' in setup_pars.keys():
                quad_dict = setup_pars['quadrant_positions']
                dx, dy = [quad_dict[x] for x in ['dx', 'dy']]
                quad_pos = [(dx+quad_dict[f"q{x}"][0],
                             dy+quad_dict[f"q{x}"][1]) for x in range(1,5)]
                if 'geometry' not in setup_pars.keys():
                    setup_pars['geometry'] = quad_pos
                else:
                    print('Quadrant positions and geometry file defined by \
                            setupfile. Loading geometry file...')

            # if the key exists in the setup file without any entries, None is
            # returned. We convert those entries to empty dictionaires.
            for key, value in setup_pars.items():
                if value is None:
                    setup_pars[key] = {}
            return setup_pars


    @property
    def mask(self):
        """np.ndarray: shape(16,512,128) Mask where `bad` pixels are 0
        and `good` pixels 1.
        """
        return self.__mask


    @mask.setter
    def mask(self, mask):
        if mask is None:
            mask = np.ones((16,512,128), 'bool')
        elif isinstance(mask, str):
            try:
                mask = np.load(mask)
            except Exception as e:
                print("Loading the mask failed. Error: ", e)
        elif isinstance(mask, np.ndarray):
            if mask.shape != (16,512,128):
                try:
                    mask = mask.reshape(16,512,128)
                except Exception as e:
                    print("Mask array has invalid shape: ", mask.shape)
        else:
            raise TypeError(f'Cannot read mask of type {type(mask)}.')

        self.__mask = np.array(mask).astype('bool')


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
        det_sources = filter(lambda x: 'AGIPD1M' in x, run.detector_sources)
        for source_name in det_sources:
            good_trains_source = set()
            for source_file in run._source_index[source_name]:
                good_trains_source.update(
                        set(source_file.train_ids[source_file.get_index(
                            source_name,'image')[1].nonzero()]
                            )
                )

            trains_with_images[source_name] = good_trains_source

        good_trains = sorted(reduce(set.intersection,
                                    list(trains_with_images.values())))
        good_trains = np.array(good_trains)
        good_indices = np.intersect1d(run.train_ids, good_trains,
                                      return_indices=True)[1]

        return good_trains, good_indices


    def _get_pulse_ids(self, train_idx=0):
        source = 'MID_DET_AGIPD1M-1/DET/{}CH0:xtdf'
        i = 0
        while i < 10:
            for module in range(16):
                try:
                    s = source.format(module)
                    tid, train_data = self.run.select(s,
                            'image.pulseId').train_from_index(train_idx)
                    pulse_ids = np.array(train_data[s]['image.pulseId'])
                    return pulse_ids
                except KeyError:
                    pass
            i += 1
            train_idx += 1
        raise ValueError("Unable to determine pulse ids. Probably the data \
                source was not available.")


    def _start_slurm_cluster(self):
        """Initialize the slurm cluster"""

        opt = self._slurm_opt
        # nprocs = 72//threads_per_process
        nprocs = opt.pop('nprocs', 12)
        threads_per_process = opt.pop('cores', nprocs)
        njobs = opt.pop('njobs', min(max(int(self.ntrains/64), 4), 12))
        print(f"\nSubmitting {njobs} jobs using {nprocs} processes per job.")
        self._cluster = SLURMCluster(
            queue=opt.get('partition',opt.pop('partition', 'exfel')),
            processes=nprocs,
            cores=threads_per_process,
            memory=opt.pop('memory', '768GB'),
            log_directory='./dask_log/',
            local_directory='/scratch/',
            nanny=True,
            death_timeout=3600,
            walltime="01:00:00",
            interface='ib0',
        )

        self._cluster.scale(nprocs*njobs)
        print(self._cluster)
        self._client = Client(self._cluster)
        self._calibrator._client = self._client
        print("Cluster dashboard link:", self._cluster.dashboard_link)


    def _stop_slurm_cluster(self):
        """Shut down the slurm cluster"""
        self._client.close()
        self._cluster.close()


    def _create_output_file(self):
        """Create the HDF5 output file.
        """

        # check existing files and determine counter
        existing = os.listdir('./')
        counter = map(re.compile(f"(?<=r{self.run_number:04}-analysis_)"
                              ".*\d{3,}(?=\.h5)").search, existing)
        counter = filter(lambda x: bool(x), counter)
        counter = list(map(lambda x: int(x[0]), counter))

        identifier = max(counter) + 1 if len(counter) else 0
        filename = f"./r{self.run_number:04}-analysis_{identifier:03}.h5"

        # copy the setupfile
        new_setupfile  = f"./r{self.run_number:04}-setup_{identifier:03}.yml"
        copyfile(self.setupfile, new_setupfile)
        self.setupfile = new_setupfile

        self.file_name = os.path.abspath(filename)

        with h5.File(self.file_name, 'a') as f:
            for flag, method in zip(self.analysis, self.METHODS):
                if int(flag):
                    for path,(shape,dtype) in self.h5_structure[method].items():
                        pass
                        # f.create_dataset(path, shape=shape, dtype=dtype)


    def compute(self, create_file=True):
        """Start the actual computation based on the analysis attribute.
        """

        if create_file:
            self._create_output_file()

        try:
            for i, (flag, method) in enumerate(zip(self.analysis, self.METHODS)):
                if int(flag):
                    if method not in ['META', 'DIAGNOSTICS'] and not self._cluster_running:
                        self._start_slurm_cluster()
                        self._cluster_running = True
                    print(f"\n{method:-^50}")
                    getattr(self,
                            f"_compute_{method.lower()}")()
                    print(f"{' Done ':-^50}")
                    # pdb.set_trace()
                    if self._cluster_running:
                        self._client.restart()
        finally:
            if self._client is not None:
                self._stop_slurm_cluster()


    def _compute_meta(self):
        """Find complete trains."""
        all_trains = len(self.train_ids)
        self.train_ids, self.train_indices = self._get_good_trains(self.run)
        print(f'{self.train_ids.size} of {all_trains} trains are complete.')
        self.last_train_idx = min([self.last_train_idx,self.train_indices.size])
        self.ntrains = min([self.train_ids.size, self.last_train_idx])
        self.train_ids = self.train_ids[:self.ntrains]
        self.train_indices = self.train_indices[:self.ntrains]
        print(f'{self.ntrains} of {all_trains} trains will be processed.')

        data = {'pulses_per_train': self.pulses_per_train,
                'pulse_ids': self.pulse_ids,
                'train_ids': self.train_ids,
                'train_indices': self.train_indices,}

        self._write_to_h5(data, 'META')


    def _compute_diagnostics(self):
        """Read diagnostic data. """
        print(f"Read XGM data.")
        arr = self.run.get_array('SA2_XTD1_XGM/XGM/DOOCS:output',
                                 'data.intensityTD')
        arr = {'data': arr[self.train_indices,:self.pulses_per_train]}
        self._write_to_h5(arr, 'DIAGNOSTICS')


    def _compute_saxs(self):
        """Perform the azimhuthal integration."""

        saxs_opt = dict(
	        mask=self.mask,
                setup=self.setup,
                geom=self.agipd_geom,
                last=self.ntrains,
                max_trains=200,
                )
        saxs_opt.update(self._saxs_opt)

        print('\nCompute pulse resolved SAXS')
        out = azimuthal_integration(self._calibrator, method='single',
                **saxs_opt)

        self._write_to_h5(out, 'SAXS')


    def _compute_xpcs(self):
        """Calculate correlation functions."""

        xpcs_opt = dict(
	        mask=self.mask,
                qmap = self.qmap,
                setup=self.setup,
                last=self.ntrains,
                dt=self.pulse_delay,
                use_multitau=False,
                rebin_g2=False,
                h5filename=self.file_name,
                )
        xpcs_opt.update(self._xpcs_opt)

        print('Compute XPCS correlation funcions.')
        out = correlate(self._calibrator, method='per_train', **xpcs_opt)
        self._write_to_h5(out, 'XPCS')


    def _compute_frames(self):
        """Averaging frames."""

        frames_opt = dict(axis='train')
        frames_opt.update(self._frames_opt)

        print('Computing frames.')
        out = average(self._calibrator, **frames_opt)

        img2d = self.agipd_geom.position_modules_fast(out['average'])[0]
        out.update({'image2d': img2d})

        self._write_to_h5(out, 'FRAMES')


    def _compute_dark(self):
        """Averaging darks."""

        dark_opt = dict(axis='train')
        dark_opt.update(self._dark_opt)

        print('Computing darks.')
        out = average(self._calibrator, **dark_opt)

        self._write_to_h5(out, 'DARK')


    # def _compute_statistics(self):
    #     """Perform the azimhuthal integration."""

    #     statistics_opt = dict(
    #             mask=self.mask,
    #             setup=self.setup,
    #             geom=self.agipd_geom,
    #             last=self.ntrains,
    #             max_trains=200,
    #             )
    #     statistics_opt.update(self._statistics_opt)

    #     print('\nCompute pulse resolved statistics')
    #     out = azimuthal_integration(self._calibrator, method='single',
    #             **saxs_opt)

    #     self._write_to_h5(out, 'SAXS')


    def _write_to_h5(self, output, method):
        """Dump results in HDF5 file."""
        with h5.File(self.file_name, 'r+') as f:
            keys = list(self.h5_structure[method].keys())
            for keyh5, item in zip(keys, output.values()):
                f[keyh5] = item


def _get_parser():
    """Command line parser
    """
    parser = argparse.ArgumentParser(
        prog='midtools',
        description='Analyze MID runs.',
    )
    parser.add_argument(
            'setupfile',
            type=str,
            help='the YAML file to configure midtools',
            )
    parser.add_argument(
            'analysis',
            type=str,
            help='which analysis to perform. List of 0s and 1s:\n \
                  10 runs SAXS routines,\n \
                  01 runs XPCS routines,\n \
                  11 runs both SAXS and XPCS routines.',
            )
    parser.add_argument(
            '-r',
            '--run',
            type=int,
            help='Run number.',
            default=None,
            )
    parser.add_argument(
            '-dr',
            '--dark_run',
            type=int,
            help='Dark run number.',
            default=None,
            )
    parser.add_argument(
            '--last',
            type=int,
            help='last train to analyze.',
            default=1_000_000,
            )
    parser.add_argument(
            '-ppt',
            '--pulses_per_train',
            metavar='npulses',
            type=int,
            help='number of pulses per train',
            default=500,
            )
    return parser


def main():

    # get the command line arguments
    parser = _get_parser()
    args = parser.parse_args()

    t_start = time.time()

    data = Dataset(args.setupfile,
                   analysis=args.analysis,
                   last_train=args.last,
                   run_number=args.run,
                   pulses_per_train=args.pulses_per_train,
                   dark_run_number=args.dark_run,
                   )
    print("Development Mode")
    print(f"\n{' Starting Analysis ':-^50}")
    print(f"Analyzing {data.ntrains} trains of {data.datdir}")
    data.compute()

    elapsed_time = time.time() - t_start
    print(f"Finished: elapsed time: {elapsed_time/60:.2f}min")
    print(f"Results saved under {data.file_name}")

if __name__ == "__main__":
    main()
