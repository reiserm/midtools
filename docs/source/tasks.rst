Tasks to Prepare p2458
======================

1. Homework
-----------

Stay healthy.

2. Homework
-----------

The aim of the second *homework* is to familiarize yourself with working on the maxwell cluster. 
We will also try to establish `Slack` for communication in the data team so please post questions there.

Try to accomplish the following tasks:

* Decide which method you want to use for analyzing data:

  * Python or jupyter notebooks
  * MATLAB
  * etc.


* Access the Maxwell cluster with each of the following  methods:

  * Jupyter Hub `<https://max-jhub.desy.de/>`_
  * Secure Shell (SSH)
  * FastX `<https://max-display.desy.de:3443/>`_

* Locate the proposal folder and the :code:`r522_test.h5` file.
  The file contains the data I showed in the last session.

* Open the file and display the different datasets.

* Calculate the position of the peak in I(q) as a function of the incident X-ray pulse energy.


3. Homework
-----------

The :ref:`hdf5_structure` has changed. A new group has been added containing
the XPCS data. More precisely, the train resolved correlation functions, the 
corresponding delay time values, and the values of the q-bins are saved. 

.. note:: Another convenient way of exploring the HDF5 file is to use the 
          :code:`hdfview` program on Maxwell.
          * Use a shell on Maxwell, e.g., by using FastX.
          * run :code:`module load xray` this will make hdfview available.
          * run :code:`hdfview file-name` to open an HDF5 file.

I have processed a set of runs and saved the results in the 
datasets folder (see :ref:`locations`). To make sure that the results are 
not overwritten, a random number is added to the file name. It does not have 
any meaning.

The data have been measured with a Vycor sample varying the position of our 
nanofocus lenses; therefore, the beam size on the sample should vary between 
the measurements.

This brings us to the :ref:`3. Homework`:

* Check the new files and load the data as in the :ref:`2. Homework`.
* Characterize the measurements and search for an effect of the varied focal
  size.
* Prepare your results in maximum 4 figures for a breakout session on Tuesday.

As always, use Slack if you need more information or have questions.

