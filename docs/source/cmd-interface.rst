
Command Line Interface
======================

After installation, **midtools** can be run from the command line::

   >>> midtools /path/to/setupfile/setup.yml 01

The *setup.yml* file is a necessary argument. It contains information on the
current experimental setup, the data directory, etc. and defines other
parameters important for the analysis routines like the q-bins for the XPCS
analysis. The next argument is a serious of ones or zeros that determines which
analysis should be performed. See :ref:`the_dataset_class` for more details on
the arguments and their purpose.

A third argument is optionally setting the maximum number of trains to analyze::

   >>> midtools /path/to/setupfile/setup.yml 01 100

This would analyze the first 100 trains.
