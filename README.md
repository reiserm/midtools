# midtools

Data analysis tools and example jupyter notebooks for MID experinemts.
Developed in preparation for experiment 2458.

Further information can be found on [ReadTheDocs](https://midtools.readthedocs.io/en/latest/index.html).


## Installation

### Recommended Installation

It is recommended to create a new virtual environmet when using `midtools`.
Make sure that all other environments, especially conda environments, are
deactivated. Let's assume you would like to work in a folder dedicated to your
project.

```sh
mkdir my_project
cd my_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/reiserm/Xana.git
pip install git+https://github.com/reiserm/midtools.git
```

The virutal can be deactivated by typing `deactivate`. Before running
`midtools` it is recommended to increase the limit of open files:

```sh
ulimit -n 4096
```


### Standard Installation

Install most recent version from GitHub:
```sh
git clone https://github.com/reiserm/midtools.git
cd midtools
pip install .
```
