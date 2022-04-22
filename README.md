# midtools

Data analysis tools and example jupyter notebooks for MID experinemts.
Developed in preparation for experiment 2458.

Further information can be found on [ReadTheDocs](https://midtools.readthedocs.io/en/latest/index.html).


## Installation

### Recommended Installation

It is recommended to create a new virtual environmet when using `midtools`.
If you are working on the Maxwell cluster (DESY). Use the following steps:

```sh
mkdir my_project
cd my_project
module load maxwell anaconda-python/3.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade git+https://github.com/reiserm/midtools.git
```

The virutal environment can be deactivated by typing `deactivate`. 


### Example Jupyter Notebook

The example notebook `midtools/setup_fonfig/analyze-run.ipynb` demonstrates how to process data with midtools. 
Create an IPython kernel based on the environment you created before:

```sh
cd my_project
source .venv/bin/activate
pip install --upgrade ipykernel
python -m ipykernel install --user --name midtools
```

Run the notebook on Max-JHub. The kernel `midtools` should be available in the list of kernels.
