.. _locations:

Locations
=========

* The proposal folder:
  ::

        /gpfs/exfel/exp/MID/202001/p002458/

* We are going to use the `scratch` folder for data anlysis during the
  experinemt and its preparation:  You can create your own subfolder in this
  directory to work on data analysis.
  ::

        /gpfs/exfel/exp/MID/202001/p002458/scratch/

* We will process the data and store the results in an HDF5-file
  (see `Structure of the HDF5-File`_). The results are stored under:
  ::

        /gpfs/exfel/exp/MID/202001/p002458/scratch/datasets/


.. _hdf5_structure:

Structure of the HDF5-File
==========================

The results are saved in a single HDF5-file per run. In general, a counter is
added in case a run is reprocessed not to overwrite old results. We will only
consider `full` trains, i.e., trains where all 16 AGIPD modules have been
stored. Other trains are omitted.

The structure of the HDF5-file is shown in the following table.
:code:`npulses` is the number of pulses per train and :code:`ntrains` is the
number of trains. :code:`nq` is the number of q-values. :code:`nx, ny` are the
pixel in `x`- and `y`-direction, respectively. :code:`ntimes` is the number of
delay times of the correlation function. :code:`nbins` is the number of bins
of the histogram.

..  note::
    Momentum transfers are recorded in inverse nanometers.


..
    :code:`-1` means that the dimension
    depends on the configuration, e.g., in case of the statistics module, the
    second dimension is the number of bins.

.. csv-table:: results.h5-structure
        :header: "Path", "Dimensions", "Content"
        :widths: 30, 5, 30

        "average/intensity", "(16,512,128)", "averge adu values."
        "average/variance", "(16,512,128", "variance"
        "average/image_2d", "(ny,nx)", "average 2D image"
        "identifiers/train_ids", "(ntrains)", "train_ids of full trains"
        "identifiers/train_indices", "(ntrains,)", "train_indices of full trains"
        "pulse_resolved/azimuthal_intensity/I", "(npulses x ntrains,nq)", "I(q) for each pulse"
        "pulse_resolved/azimuthal_intensity/q", "(nq)", "q-values"
        "pulse_resolved/xgm/energy", "(ntrains,npulses)", "XGM energy in micro Joule"
        "pulse_resolved/xgm/pointing_x", "(ntrains,npulses)", "horizontal pointing"
        "pulse_resolved/xgm/pointing_y", "(ntrains,npulses)", "vertical pointing"
        "train_resolved/correlation/q", "(nq,)", "XPCS q-bins"
        "train_resolved/correlation/t", "(ntimes,)", "XPCS delay times"
        "train_resolved/correlation/g2", "(ntrains,ntimes,nq)", "XPCS correlation functions"
        "/train_resolved/correlation/ttc", "(ntrains, ntimes, nq)", "two-time correlation functions"
        "/pulse_resolved/statistics/centers", "(npulses x ntrains, nbins)", "histogram bin centers"
        "/pulse_resolved/statistics/counts", "(npulses x ntrains, nbins)", "histogram bin counts"
        "/train_resolved/sample_position/y", "(ntrains, )", "horizontal sample position"
        "/train_resolved/sample_position/z", "(ntrains, )", "vertical sample positions"
        "/train_resolved/linkam-stage/temperature", "(ntrains, )", "sample temperature"

