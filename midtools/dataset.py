#!/usr/bin/env python
import os
import yaml
import numpy as np
from extra_data import RunDirectory
from extra_geom import AGIPD_1MGeometry

class Dataset:
    
    def __init__(self, setupfile):
        """Dataset class to handle MID datasets on Maxwell.

        Args:
            setupfile (str): Setupfile (.yml) that contains information on the
                setup parameters.
                
        Note:
            An example setupfile might look like this:
            # setup.yml file
                        
            # data given as str or cycle, proposal, datatype
            datdir: /gpfs/exfel/exp/MID/202001/p002458/scratch/example_data/r0522
            #  cycle: 202001
            #  proposal: 2458
            #  datatype: proc # or raw
            
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
        """
        
        setup_pars = self.read_setup(setupfile)
        
        
        self.datdir = setup_pars.pop('datdir', False) #: str: Data directory
        #: DataCollection: e.g., returned from extra_data.RunDirectory
        self.run = RunDirectory(self.datdir) 
        
        #: tuple: Position of the direct beam in pixels
        self.center = None
        
        #: AGIPD_1MGeometry: 
        self.agipd_geom = setup_pars.pop('quadrant_positions', False)
        self.__dict__.update(setup_pars)
        
        #: str: Path to the mask-file
        self.mask = setup_pars.pop('mask', None)
        del setup_pars
    
    @property
    def datdir(self):
        return self.__datdir

    @datdir.setter
    def datdir(self, path):
        if isinstance(path, dict):
            basedir = '/gpfs/exfel/exp/MID/'
            if len(list(filter(lambda x: x in path.keys(), 
                               ['cycle', 'proposal', 'datatype']))):
                path = basedir + f"/{path['cycle']}/p{path['proposal']:06d}/{path['datatype']}"
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
    def read_setup(setupfile):
        """read setup parameters from config file
        """
        with open(setupfile) as file:
            setup_pars = yaml.load(file, Loader=yaml.FullLoader)
            
            # make 
            quad_dict = setup_pars['quadrant_positions']
            dx, dy = [quad_dict[x] for x in ['dx', 'dy']]
            quad_pos = [(dx+quad_dict[f"q{x}"][0], dy+quad_dict[f"q{x}"][1]) for x in range(1,5)]
            setup_pars['quadrant_positions'] = quad_pos
            return setup_pars
    
    
if __name__ == "__main__":
    setupfile = '../setup_config/setup.yml'
    data = Dataset(setupfile)
    print(data.datdir)
