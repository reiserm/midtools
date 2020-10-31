API
===

.. _concept:

Concept
-------

:class:`midtools.Dataset` is the basis of :mod:`midtools`. Amongst others it:

* provides the command line interface,
* reads the configuration file
* handles metadata,
* starts the SLURMCluster,
* runs analysis routines from submodules,
* and finally stores the results in an HDF5-file.

Data are provided and calibrated by :class:`midtools.Calibrator` after running
:meth:`midtools.Calibrator._get_data` which is called by
:class:`midtools.Dataset`. It includes the following data processing steps:

* loading the data as an :code:`xarray` using the :code:`extra_data` module,
* slicing trains and pulses,
* masking,
* binning,
* baseline correction.

Additionally, :mod:`midtools.corrections` provides functions to be applied on
individual workers.

:mod:`midtools` provides the following analysis submodules:

* :mod:`midtools.azimuthal_integration`,
* :mod:`midtools.correlation`,
* :mod:`midtools.statistics`,
* :mod:`midtools.average`.

The first three modules use the :code:`apply_along_axis` method from
:code:`dask.array` to apply an algorithm on each train or pulse of a run.


.. _the_dataset_class:

:mod:`The Dataset Class`
------------------------

.. autoclass:: midtools.Dataset
   :members:
   :special-members: __init__

.. argparse::
   :module: midtools.dataset
   :func: _get_parser


