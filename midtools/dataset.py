#!/usr/bin/env python
import os
import yaml
import numpy as np
from extra_data import RunDirectory
from extra_geom import AGIPD_1MGeometry

from midtools import azimuthal_integration

class Dataset:

    def __init__(self, setupfile):
        """Dataset class to handle MID datasets on Maxwell.

        Args:
            setupfile (str): Setupfile (.yml) that contains information on the
                setup parameters.

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

        self.setupfile = setupfile
        setup_pars = self._read_setup(setupfile)

        self.datdir = setup_pars.pop('datdir', False) #: str: Data directory
        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = RunDirectory(self.datdir)

        self.mask = setup_pars.pop('mask', None)

        #: tuple: Position of the direct beam in pixels
        self.center = None

        self.agipd_geom = setup_pars.pop('quadrant_positions', False)
        self.__dict__.update(setup_pars)

        del setup_pars

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
        self.center = geom.position_modules_fast(dummy_img)[1]
        del dummy_img


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
        """np.ndarray(int): shape(16,512,128) Mask where `bad` pixels are 0
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


if __name__ == "__main__":
    setupfile = '../setup_config/setup.yml'
    data = Dataset(setupfile)
    print(data.datdir)
