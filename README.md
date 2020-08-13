# midtools - Dask Tests

Further information can be found on
[ReadTheDocs](https://midtools.readthedocs.io/en/latest/index.html) (some parts
are a bit outdated)

## Installation

Before installing `midtools` install the most recent version of `Xana` from its
master branch.

Install `midtools` from GitHub:
```sh
git clone https://github.com/reiserm/midtools.git
cd midtools
pip install -e .
```

## Quickstart Guide

`midtools` can be executed from the command line after successful installation.

```sh
midtools [setupfile] [analysis-flags] -r RUN_NUMBER --last LAST_TRAIN
```

runs midtools with the configuration given in the setupfile (YAML). Specify
the run number and the maximum number of trains you want to analyze.
`analysis-flags` indicate by ones or zeros which analysis to include.

Currently, midtools can perform two different types of analysis:
- pulse resolved azimuthal integration: analysis-flags: `10` (see `midtools/azimhuthal_integration`)
- train resolved correlation functions: analysis-flags: `01` (see `midtools/correlations`)
- usee `11` to perform both types of analysis.

midtools loads AGIPD data as xarrays and stacks the dimensions such that
`dask.array.apply_along_axis` can be used to parallelize the computation.
Therefore, a `SLURMCluster` is spawned (see `midtools/dataset.py`).

The `Dataset` class defined in `midtools/dataset.py` determines the run
metadata, reads the setup-file and sets all neccessary options.
The data are loaded and calibrted by the `Calibrator` class defined in
`midtools/corrections.py`. Each of the sub routines loads the data by calling
the `_get_data()` method of a `Calibrator` instance.

## Optimizing Dask

Whether or not everything runs smoothly, depends mostly on the `SLURMCluster`
settings and the chunksize of the data both can be modified in the setupfile.
I put two example setupfiles in the `tests` folder.
- `tests/setup_dask_good.yml` contains parameter that work (probably not optimum)
- `tests/setup_dask_bad.yml` crushes especially when applying midtools to large runs.

Even when the program finishes without errors, the log files contain GIL warnings
complaining about too big chunks.

I reproduced the described behavior by running

```sh
midtools tests/setup_dask_[good|bad].yml 11 -r 201 -ppt 100 --last 100
```

The data directory (proposal) is defined in the setupfile. Here, we are aiming
for analyzing the first 100 trains of run 201. Each train contains 100 pulses
per train (-ppt). Again the `11` means, we would like to apply both analysis
methods.






