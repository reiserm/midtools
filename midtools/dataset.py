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

# XFEL packages
from extra_data import RunDirectory
from extra_geom import AGIPD_1MGeometry

# Dask
import dask.array as da
from dask.distributed import Client, progress
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar

from Xana import Setup

from .azimuthal_integration import azimuthal_integration
from .correlations import correlate

class Dataset:

    METHODS = ['META', 'DIAGNOSTICS', 'SAXS', 'XPCS']

    def __init__(self, setupfile, analysis='00', last_train=np.inf):
        """Dataset object to analyze MID datasets on Maxwell.

        Args:
            setupfile (str): Setupfile (.yml) that contains information on the
                setup parameters.
            analysis (str, optional): Flags of the analysis to perform. Defaults to '00'.
                analysis is a string of ones and zeros where a one means to perform the analysis
                and a zero means to omit the analysis. The analysis types are:

                +-------+----------------------------+
                | flags | analysis                   |
                +=======+============================+
                |  10   | SAXS azimuthal integration |
                +-------+----------------------------+
                |  01   | XPCS correlation functions |
                +-------+----------------------------+

        Note:
            A setupfile might look like this::

                # setup.yml file

                # Data
                datdir: /gpfs/exfel/exp/MID/202001/p002458/scratch/example_data/r0522

                # Maskfile
                mask: /gpfs/exfel/exp/MID/202001/p002458/scratch/midtools/agipd_mask_tmp.npy

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
                q_range:
                    q_first: .1 # smallest q in nm-1
                    q_last: 1.  # largest q in nm-1
                    steps: 10   # how many q-bins
        """

        #: str: Path to the setupfile.
        self.setupfile = setupfile
        setup_pars = self._read_setup(setupfile)

        #: str: Flags of the analysis methods.
        self.analysis = '11' + analysis

        #: str: Data directory
        self.datdir = setup_pars.pop('datdir', False)
        self.run_number = self.datdir

        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = RunDirectory(self.datdir)

        self.mask = setup_pars.pop('mask', None)

        #: tuple: Position of the direct beam in pixels
        self.center = None

        self.agipd_geom = setup_pars.pop('quadrant_positions', False)

        #: dict: options for the Slurm cluster
        self._slurm_opt = setup_pars.pop('slurm_opt', {})
        #: dict: options for the SAXS algorithm 
        self._saxs_opt = setup_pars.pop('saxs_opt', {})
        #: dict: options for the XPCS algorithm 
        self._xpcs_opt = setup_pars.pop('xpcs_opt', {})
        
        # save the other entries as attributes
        self.__dict__.update(setup_pars)

        #: int: Number of X-ray pulses per train.
        self.pulse_ids = self._get_pulse_ids(self.run)

        #: np.ndarray: Array of pulse IDs.
        self.pulses_per_train = len(self.pulse_ids)

        #: np.ndarray: All train IDs.
        self.train_ids = np.array(self.run.train_ids)

        #: int: last train index to compute
        self.last_train_idx = min([last_train,len(self.train_ids)])

        #: int: Number of complete trains
        self.ntrains = min([len(self.train_ids), self.last_train_idx])

        #: np.ndarray: All train indices.
        self.train_indices = np.arange(self.train_ids.size)

        self.setup = Setup(detector='agipd1m')
        self.setup.mask = self.mask.copy()
        self.setup.make(**dict(center=self.center,
            wavelength=12.39/self.photon_energy,
            distance=self.sample_detector,))
        dist = self.agipd_geom.to_distortion_array()
        self.setup.detector.IS_CONTIGUOUS = False
        self.setup.detector.set_pixel_corners(dist)
        self.setup.update_ai()

        qmap = self.setup.ai.array_from_unit(unit='q_nm^-1');
        #: np.ndarray: q-map
        self.qmap = qmap.reshape(16,512,128);

        del setup_pars

        #: dict: Structure of the HDF5 file
        self.h5_structure = self._make_h5structure()

        #: str: HDF5 file name.
        self.file_name = None

        # placeholder attributes
        self._cluster = None # the slurm cluster
        self._client = None # the slurm cluster client

    def _make_h5structure(self):
        """Create the HDF5 data structure.
        """
        h5_structure = {
            'META':{
                "/identifiers/pulses_per_train":[(1,),'int8'],
                "/identifiers/pulse_ids":[(self.pulses_per_train,), 'int8'],
                "/identifiers/train_ids":[(self.ntrains,), 'int16'],
                "/identifiers/train_indices":[(self.ntrains,), 'int8'],
            },
            'DIAGNOSTICS':{
                "/pulse_resolved/xgm/energy":[(self.ntrains,self.pulses_per_train),'float32'],
            },
            'SAXS':{
                "/average/azimuthal_intensity":[(2,500),'float32'],
                "/average/image_2d":[None, 'float32'],
                "/average/image":[(16,512,128), 'float32'],
                "/pulse_resolved/azimuthal_intensity/q":[(500,), 'float32'],
                "/pulse_resolved/azimuthal_intensity/I":[(self.ntrains*self.pulses_per_train,500),
                    'float32'],
            },
            'XPCS':{
                "/train_resolved/correlation/q":[None,None],
                "/train_resolved/correlation/t":[None,None],
                "/train_resolved/correlation/g2":[None,None],
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
            pass
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
                number = 0
        elif isinstance(number, int):
            pass
        else:
            raise TypeError(f'Invalid run number {type(number)}.')

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

            # make
            quad_dict = setup_pars['quadrant_positions']
            dx, dy = [quad_dict[x] for x in ['dx', 'dy']]
            quad_pos = [(dx+quad_dict[f"q{x}"][0], dy+quad_dict[f"q{x}"][1]) for x in range(1,5)]
            setup_pars['quadrant_positions'] = quad_pos
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
            mask = np.ones((16,512,128), 'int8')
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

        self.__mask = np.array(mask).astype('int8')

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

        mod_train_ids = run.get_dataframe(fields=[('*/DET/*', 'trailer.trainId')])
        corrupted_trains = mod_train_ids[mod_train_ids.isna().sum(1)>0].index.values
        good_indices = np.where(np.sum(np.isnan(mod_train_ids,), axis=1)==0)[0]

        mod_train_ids.reset_index(level=0, inplace=True)
        mod_train_ids.rename(columns={"index": "train_id"}, inplace=True)
        tmp = mod_train_ids.dropna(axis=0)
        good_trains = tmp['train_id']

        return good_trains, good_indices

    @staticmethod
    def _get_pulse_ids(run):
        source = f'MID_DET_AGIPD1M-1/DET/{0}CH0:xtdf'
        # pulses_per_train = run.get_data_counts(source, 'image.data').iloc[:10].max()
        tid, train_data = run.select(source, 'image.pulseId').train_from_index(0)
        pulse_ids = np.array(train_data[source]['image.pulseId'])
        return pulse_ids

    def _start_slurm_cluster(self):
        """Initialize the slurm cluster"""

        nprocs = 8
        njobs = max([int(self.ntrains/64),16])
        print(f"Submitting {njobs} jobs to the Cluster")
        opt = self._slurm_opt
        self._cluster = SLURMCluster(
            queue=opt.get('partition','exfel'),
            processes=nprocs, 
            cores=72, 
            memory='512GB',
            log_directory='./dask_log/',
            local_directory='./dask_tmp/',
            nanny=True,
            death_timeout=128,
            walltime="02:00:00",
        )

        self._cluster.scale(nprocs*njobs)
        self._client = Client(self._cluster)
        print("Cluster dashboard link:", self._cluster.dashboard_link)

    def _stop_slurm_cluster(self):
        """Shut down the slurm cluster"""
        self._client.close()
        self._cluster.close()

    def _create_output_file(self, filename=None):
        """Create the HDF5 output file.
        """

        if filename is None:
            filename = f"./r{self.run_number:04}-analysis.h5"

        filename = filename.replace(".h5","") +"_"+ str(int(time.time())) + ".h5"

        self.file_name = os.path.abspath(filename)

        with h5.File(self.file_name, 'a') as f:
            for flag, method in zip(self.analysis, self.METHODS):
                if int(flag):
                    for path,(shape,dtype) in self.h5_structure[method].items():
                        pass
                        # f.create_dataset(path, shape=shape, dtype=dtype)

    def compute(self, filename=None, create_file=True):
        """Start the actual computation based on the analysis attribute.
        """

        if create_file:
            self._create_output_file(filename)

        clst_running  = False
        try:
            for i, (flag, method) in enumerate(zip(self.analysis, self.METHODS)):
                if int(flag):
                    if method not in ['META', 'DIAGNOSTICS'] and not clst_running:
                        self._start_slurm_cluster()
                        clst_running = True
                    print(f"\n{method:-^50}")
                    getattr(self,
                            f"_compute_{method.lower()}")()
                    print(f"{' Done ':-^50}")

        finally:
            if self._client is not None:
                self._stop_slurm_cluster()

    def _compute_meta(self):
        """Find complete trains."""

        all_trains = len(self.train_ids)
        self.train_ids, self.train_indices = self._get_good_trains(self.run)
        self.last_train_idx = min([self.last_train_idx,self.train_indices.size])
        self.ntrains = min([len(self.train_ids), self.last_train_idx])
        print(f'{self.ntrains} of {all_trains} trains are complete.')

        # writing output to hdf5 file
        with h5.File(self.file_name, 'r+') as f:
            keys = list(self.h5_structure['META'].keys())
            values = [
                self.pulses_per_train,
                self.pulse_ids,
                self.train_ids,
                self.train_indices,
                ]
            for keyh5,item in zip(keys,values):
                f[keyh5] = item

    def _compute_diagnostics(self):
        """Read diagnostic data. """
        arr = self.run.get_array('SA2_XTD1_XGM/XGM/DOOCS:output', 'data.intensityTD')
        arr = arr[self.train_indices,:self.pulses_per_train]
        print(f"Read XGM data.")

        # writing output to hdf5 file
        with h5.File(self.file_name, 'r+') as f:
            keys = list(self.h5_structure['DIAGNOSTICS'].keys())
            for keyh5,item in zip(keys,[arr]):
                f[keyh5] = item

    def _compute_saxs(self):
        """Perform the azimhuthal integration."""

        # specify SAXS options
        saxs_opt = dict( 
	            mask=self.mask, 
                to_counts=False, 
                apply_internal_mask=True, 
                setup=self.setup,
                client=self._client,
                geom=self.agipd_geom, 
                adu_per_photon=65, 
                last=self.last_train_idx*self.pulses_per_train,
                )
        saxs_opt.update(self._saxs_opt)

        # compute average
        print('Compute average SAXS')
        out = azimuthal_integration(self.run, 
                method='average', **saxs_opt)

        # writing output to hdf5 file
        with h5.File(self.file_name, 'r+') as f:
            keys = list(self.h5_structure['SAXS'].keys())[:3]
            for keyh5,item in zip(keys,out.values()):
                f[keyh5] = item
        del out

        # compute pulse resolved
        print('\nCompute pulse resolved SAXS')
        out = azimuthal_integration(self.run, 
                method='single', **saxs_opt)

        # writing output to hdf5 file
        with h5.File(self.file_name, 'r+') as f:
            keys = list(self.h5_structure['SAXS'].keys())[3:]
            for keyh5,item in zip(keys,out.values()):
                f[keyh5] = item
        del out

    def _compute_xpcs(self):
        """Calculate correlation functions."""

        # specify XPCS options
        xpcs_opt = dict( 
	            mask=self.mask, 
                qmap = self.qmap,
                to_counts=False, 
                apply_internal_mask=True, 
                setup=self.setup,
                adu_per_photon=65, 
                last=self.last_train_idx,
                client=self._client,
                npulses=self.pulses_per_train,
                )
        xpcs_opt.update(self._xpcs_opt)

        # compute
        print('Compute XPCS correlation funcions.')
        out = correlate(self.run, method='per_train', **xpcs_opt)

        # writing output to hdf5 file
        with h5.File(self.file_name, 'r+') as f:
            keys = list(self.h5_structure['XPCS'].keys())
            for keyh5,item in zip(keys,out.values()):
                f[keyh5] = item
        del out

def main():
    t_start = time.time()
    args = list(sys.argv[1:])

    setupfile = args[0]
    analysis = ''
    last_train = np.inf
    if len(args) > 1:
        analysis = args[1]
    if len(args) > 2:
        last_train = int(args[2])
        print(f"Analyzing first {last_train} trains.")

    data = Dataset(setupfile, analysis, last_train)
    print("Analyzing ", data.datdir)
    data.compute()
    
    print(f"Found {data.train_indices.size} complete trains")
    elapsed_time = time.time() - t_start
    print(f"Finished: elapsed time: {elapsed_time/60:.2f}min")
    print(f"Results saved under {data.file_name}")

if __name__ == "__main__":
    main()
