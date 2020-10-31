Command Line Interface
======================

After installation, **midtools** can be run from the command line. This is a
summary of the parser arguments:

.. argparse::
   :module: midtools.dataset
   :func: _get_parser
   :prog: midtools


For example, the SAXS routine can be run on each pulse of a run by::

   >>> midtools /path/to/setupfile/setup.yml 10

The *setup.yml* file is a necessary argument. It contains information on the
current experimental setup, the data directory, etc. and defines other
parameters important for the analysis routines like the q-bins for the XPCS
analysis. The next argument is a serious of ones or zeros that determines which
analysis should be performed. See :ref:`the_dataset_class` for more details on
the arguments and their purpose.

.. warning:: The **run number** has to be specified in the setupfile by
             giving the full path to the data including the run folder
             */path/r0123*. The run directory can be ommited if the
             run argument is passed on the command line.

If the run directory is not included in the setupfile, it has to be specified.
For example, this would analyze the first 1000 trains of run 123 and run SAXS
and XPCS routines::

   >>> midtools /path/to/setupfile/setup.yml 11 --last 1000 --run 123

Additinally, the number of X-ray pulses per train can be specified::

   >>> midtools /path/to/setupfile/setup.yml 11 --last 1000 --run 123 --ppt 100

This would assume that 100 pulses per train (ppt) were deliverd by the machine.

.. note:: If you are insecure which arguments you should provide, pass all you
          know.


Working With Raw Data
---------------------

Midtools comes with simple data calibration routines for HG (high-gain) AGIPD
data including:

- dark subtraction,
- creating of a pixel mask based on dark and flatfield data,
- baseline shift based on Ta-stripes,
- flexible (spatial) commonmode correction (default: asic level 64x64 pixels),
- dropletizing by thresholding (converting ADUs to photons).

To work with raw data, a dark run has to be processed by passing the
`--is-dark`-flag::

   >>> midtools setup.yml 11 --last 500 --run 001 --ppt 200 --is-dark

The output file would be `r0001-dark_000.h5` (the counter might not be `000`).

To use a previously created dark file, pass the `--dark-run` argument.
The argument requires both the **run number** and the file **index**.
So if you want to use the dark file `r0001-dark_000.h5`,
you would run `midtools` like::

   >>> midtools setup.yml 11 --last 500 --run 011 --ppt 20 --dark-run 1 0


.. warning:: Make sure to use all pulses when processing the dark run.


Latest Changes and Remarks
--------------------------

`midtools` can read the pulse pattern from the machine control sources which in
principle allows to omit the `--ppt` or `--pulses-per-train` argument. However,
this source is not always available in the DAQ. Passing the argument manually
will also overwride the settings read from the control source.



