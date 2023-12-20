# Installation

Here we describe how to install the DeepProg package. We assume that the installation will be done locally, using the `--user` flag from pip. Alternatively, the package can be installed using a virtual environment or globally with sudo. Both python2.7 or python3.6 (or higher) can be used. We tested the installation on a linux, OSX and Windows environment.

## Requirements
* Python 2 or 3 (Python3 is recommended)
* Either theano, tensorflow or CNTK (tensorflow is recommended)
* [theano](http://deeplearning.net/software/theano/install.html) (the used version for the manuscript was 0.8.2)
* [tensorflow](https://www.tensorflow.org/) as a more robust alternative to theano
* [cntk](https://github.com/microsoft/CNTK) CNTK is anoter DL library that can present some advantages compared to tensorflow or theano. See [https://docs.microsoft.com/en-us/cognitive-toolkit/](https://docs.microsoft.com/en-us/cognitive-toolkit/)
* scikit-learn (>=0.18)
* numpy, scipy
* lifelines
* (if using python3) scikit-survival
* (For distributed computing) ray (ray >= 0.8.4) framework
* (For hyperparameter tuning) scikit-optimize

## Tested python package versions
Python 3.8 (tested for Linux and OSX. For Windows Visual C++ is required and LongPathsEnabled shoud be set to 1 in windows registry)
* tensorflow == 2.4.1 (2.4.1 currently doesn't seem to work with python3.9)
* keras == 2.4.3
* ray == 0.8.4
* scikit-learn == 0.23.2
* scikit-survival == 0.14.0 (currently doesn't seem to work with python3.9)
* lifelines == 0.25.5
* scikit-optimize == 0.8.1 (currently doesn't seem to work with python3.9)
* mpld3 == 0.5.1

Since ray and tensorflow are rapidly evolving libraries, newest versions might unfortunatly break DeepProg's API. To avoid any dependencies issues, we recommand working inside a Python 3 [virtual environement](https://docs.python.org/3/tutorial/venv.html) (`virtualenv`) and install the tested packages.

### installation (local)

```bash
# The downloading can take few minutes due to the size of th git project
git clone https://github.com/lanagarmire/DeepProg.git
cd DeepProg

# (RECOMMENDED) install with conda
conda env create -n deepprog -f ./environment.yml python=3.8
conda activate deepprog
pip install -e . -r requirements_tested.txt

# (RECOMMENDED) to install the tested python library versions
pip install -e . -r requirements_tested.txt

# Basic installation (under python3/pip3)
pip3 install -e . -r requirements.txt
# To intall the distributed frameworks
pip3 install -e . -r requirements_distributed.txt
# Installing scikit-survival (python3 only)
pip3 install -r requirements_pip3.txt
# Install ALL required dependencies with the most up to date packages
pip install -e . -r requirements_all.txt


# **Ignore this if you are working under python3**
# python 3 is highly preferred, but DeepProg working with python2/pip2, however there is no support for scikit-survival in python2
pip2 install -e . -r requirements.txt
pip2 install -e . -r requirements_distributed.txt
```

### Installation with docker
We have created a docker image (`opoirion/deepprog_docker:v1`) with all the dependencies already installed. For the docker (and singularity) instruction, please refer to the docker [tutorial](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage_with_docker.html).

## Alternative deep-Learning packages installation

The required python packages can be installed using pip:

```bash
pip install theano --user # Original backend used OR
pip install tensorflow --user # Alternative backend for keras and default
pip install keras --user
```

## Alternative support for CNTK / theano / tensorflow
We originally used Keras with theano as backend plateform. However, [Tensorflow](https://www.tensorflow.org/) (currently the defaut background DL framework) or [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) are more recent DL framework that can be faster or more stable than theano. Because keras supports these 3 backends, it is possible to use them as alternative. To install CNTK, please refer to the official [guidelines](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) . To change backend, please configure the `$HOME/.keras/keras.json` file. (See official instruction [here](https://keras.io/backend/)).

The default configuration file: ` ~/.keras/keras.json` looks like this:

```json
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

### R installation (Alternative to Python lifelines)

In his first implementation, DeepProg used the R survival toolkits to fit the survival functions (cox-PH models) and compute the concordance indexes. These functions have been replaced with the python toolkits lifelines and scikit-survival for more convenience and avoid any compatibility issue. However, differences exists regarding the computation of the c-indexes using either python or R libraries. To use the original R functions, it is necessary to install the following R libraries.

* R
* the R "survival" package installed.
* rpy2 3.4.4 (for python2 rpy2 can be install with: pip install rpy2==2.8.6, for python3 pip3 install rpy2==2.8.6).


```R
install.packages("survival")
install.packages("glmnet")
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("survcomp")
```

Then, when instantiating a `SimDeep` or a `SimDeepBoosting` object, the option `use_r_packages` needs to be set to `True`.


## Visualisation module (Experimental)
To visualise test sets projected into the multi-omic survival space, it is required to install `mpld3` module.
Note that the pip version of mpld3 installed with pip on my computer presented a [bug](https://github.com/mpld3/mpld3/issues/434): `TypeError: array([1.]) is not JSON serializable `. However, the [newest](https://github.com/mpld3/mpld3) version of the mpld3 available from the github solved this issue. Rather than executing `pip install mpld3 --user` It is therefore recommended to install the newest version to avoid this issue directly from the github repository:

```bash
git clone https://github.com/mpld3/mpld3
cd mpld3
pip install -e . --user
```

### Distributed computation
* It is possible to use the python ray framework [https://github.com/ray-project/ray](https://github.com/ray-project/ray) to control the parallel computation of the multiple models. To use this framework, it is required to install it: `pip install ray`
* Alternatively, it is also possible to create the model one by one without the need of the ray framework

### Visualisation module (Experimental)
* To visualise test sets projected into the multi-omic survival space, it is required to install `mpld3` module: `pip install mpld3`
* Note that the pip version of mpld3 installed on my computer presented a [bug](https://github.com/mpld3/mpld3/issues/434): `TypeError: array([1.]) is not JSON serializable `. However, the [newest](https://github.com/mpld3/mpld3) version of the mpld3 available from the github solved this issue. It is therefore recommended to install the newest version to avoid this issue.

## Usage
* test if simdeep is functional (all the software are correctly installed): go to main folder (./DeepProg/) and run the following

```bash
  python3 test/test_simdeep.py -v #
  ```

* All the default parameters are defined in the config file: `./simdeep/config.py` but can be passed dynamically. Three types of parameters must be defined:
  * The training dataset (omics + survival input files)
    * In addition, the parameters of the test set, i.e. the omic dataset and the survival file
  * The parameters of the autoencoder (the default parameters works but it might be fine-tuned.
  * The parameters of the classification procedures (default are still good)


## Example scripts

Example scripts are availables in ./examples/ which will assist you to build a model from scratch with test and real data:

```bash
examples
├── example_hyperparameters_tuning.py
├── example_hyperparameters_tuning_with_test_dataset.py
├── example_with_dummy_data_distributed.py
├── example_with_dummy_data.py
└── load_3_omics_model.py
