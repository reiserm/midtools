Locations
=========

* The proposal folder: 
  ::

        /gpfs/exfel/exp/MID/202001/p002458/
* We are going to use the `scratch` folder for data anlysis during the experinemt and its preparation:
  ::

        /gpfs/exfel/exp/MID/202001/p002458/scratch/
  You can create your own subfolder in this directory to work on data analysis.

* We will process the data and store the results in an `HDF5`-file (see here h5_structure_). 
  The results are stored under:
  ::

        /gpfs/exfel/exp/MID/202001/p002458/scratch/datasets/


Structure of the Results
========================
.. _h5_structure:

The h5-file contains the following structure:

.. csv-table:: H5-structure
        :header: "Path"
        :widths: 30

        "identifiers"                       
        "identifirs/train_ids"
        "identifiers/train_indices"
        "pulse_resolved"
        "pulse_resolved/azimuthal_intensity" 
        "pulse_resolved/azimuthal_intensity/I"
        "pulse_resolved/azimuthal_intensity/q"
        "pulse_resolved/xgm"
        "pulse_resolved/xgm/energy" 
