# Installation

Here we describe how to install the DeepProg package. We assume that the installation will be done locally, using the `--user` flag from pip. Alternatively, the package can be installed using a virtual environment or globally with sudo. Both python2.7 or python3.6 (or higher) can be used. We only tested the installation on a linux environment but it should also work on a OSX environment.

## Requirements
* Python 2 or 3 (Python3 is recommended)
* Either theano, tensorflow or CNTK (theano is recommended)
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
* tensorflow == 2.4.1
* keras == 2.4.3
* ray == 0.8.4
* scikit-learn == 0.23.2
* scikit-survival == 0.14.0
* lifelines == 0.25.5
* scikit-optimize == 0.8.1
* mpld3 == 0.5.1

Since ray and tensorflow are rapidly evolving libraries, newest versions might unfortunatly break DeepProg's API. To avoid any dependencies issues, we recommand working inside a Python 3 [virtual environement](https://docs.python.org/3/tutorial/venv.html) (`virtualenv`) and install the tested packages.

### installation (local)

```bash
# The downloading can take few minutes due to the size of th git project
git clone https://github.com/lanagarmire/DeepProg.git
cd SimDeep
# Basic installation
pip3 install -e . -r requirements.txt
# To intall the distributed frameworks
pip3 install -r requirements_distributed.txt
# Installing scikit-survival (python3 only)
pip3 install -r requirements_pip3.txt

# DeepProg is working also with python2/pip2 however there is no support for scikit-survival in python2
pip2 install -r requirements.txt
pip2 install -r requirements_distributed.txt

# to install the tested python library versions
pip install -r requirements_tested.txt

# Install ALL required dependencies
pip install -r requirements_all.txt
```

## Deep-Learning packages installation

The required python packages can be installed using pip:

```bash
pip install theano --user # Original backend used OR
pip install tensorflow --user # Alternative backend for keras
pip install keras --user
```

## Support for CNTK / tensorflow
We originally used Keras with theano as backend plateform. However, [Tensorflow](https://www.tensorflow.org/) or [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) are more recent DL framework that can be faster or more stable than theano. Because keras supports these 3 backends, it is possible to use them as alternative to theano. To install CNTK, please refer to the official [guidelines](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) . To change backend, please configure the `$HOME/.keras/keras.json` file. (See official instruction [here](https://keras.io/backend/)).

The default configuration file: ` ~/.keras/keras.json` looks like this:

```json
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

### R installation (Depreciated)

In his first implementation, DeepProg used the R survival toolkits to fit the survival functions. Thse functions have been replaced with the python toolkits lifelines and scikit-survival for more convenience and avoid any compatibility issue.

* R
* the R "survival" package installed.
* rpy2 2.8.6 (for python2 rpy2 can be install with: pip install rpy2==2.8.6, for python3 pip3 install rpy2==2.8.6). It seems that newer version of rpy2 might not work due to a bug (not tested)


```R
install.package("survival")
install.package("glmnet")
source("https://bioconductor.org/biocLite.R")
biocLite("survcomp")
```


## Distributed computation
It is possible to use the python ray framework [https://github.com/ray-project/ray](https://github.com/ray-project/ray) to control the parallel computation of the multiple models. To use this framework, it is required to install it: `pip install ray --user`.
Alternatively, it is also possible to create the model one by one without the need of the ray framework

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
* test if simdeep is functional (all the software are correctly installed):

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
