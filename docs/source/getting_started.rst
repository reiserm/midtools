Locations
=========

* The proposal folder: 
  ::

        /gpfs/exfel/exp/MID/202001/p002458/
* We are going to use the `scratch` folder for data anlysis during the experinemt and its preparation:
  You can create your own subfolder in this directory to work on data analysis.
  ::

        /gpfs/exfel/exp/MID/202001/p002458/scratch/

* We will process the data and store the results in an HDF5-file (see `Structure of the HDF5-File`_). 
  The results are stored under:
  ::

        /gpfs/exfel/exp/MID/202001/p002458/scratch/datasets/


Structure of the HDF5-File
==========================

The results are saved in a single HDF5-file per run. In general, a counter is added in case a run is reprocessed not to overwrite old results.
We will only consider `full` trains, i.e., trains where all 16 AGIPD modules have been stored. Other trains are omitted.

The structure of the HDF5-file is shown in the following table. :code:`npulses` is the total number of pulses. :code:`nq` is the number of q-values. 
:code:`nx, ny` are the pixel in `x`- and `y`-direction, respectively.

.. csv-table:: results.h5-structure
        :header: "Path", "Type", "Content"
        :widths: 30, 5, 30

        "average/azimuthal_intensity", "array(nq,2)", "averaged I(q), first column q, second I"
        "average/image_2d", "array(ny,nx)", "average 2D image of the entire run"
        "identifiers/train_ids", "array(npulses)", "train_ids of full trains"
        "identifiers/train_indices", "array(npulses)", "train_indices of full trains"
        "pulse_resolved/azimuthal_intensity/I", "array(npulses,nq)", "I(q) for each pulse" 
        "pulse_resolved/azimuthal_intensity/q", "array(nq)", "q-values"
        "pulse_resolved/xgm/energy", "array(npulses)", "XGM energy in micro Joule" 
