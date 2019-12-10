# Installation

Here we describe how to install the DeepProg package. We assume that the installation will be done locally, using the `--user` flag from pip. Alternatively, the package can be installed using a virtual environment or globally with sudo. Both python2.7 or python3.6 (or higher) can be used. We only tested the installation on a linux environment but it should also work on a OSX environment.

## Requirements
* Python 2 or 3
* [theano](http://deeplearning.net/software/theano/install.html) (the used version for the manuscript was 0.8.2)
* [tensorflow](https://www.tensorflow.org/) as a more robust alternative to theano
* [cntk](https://github.com/microsoft/CNTK) CNTK is anoter DL library that can present some advantages compared to tensorflow or theano. See [https://docs.microsoft.com/en-us/cognitive-toolkit/](https://docs.microsoft.com/en-us/cognitive-toolkit/)
* R
* the R "survival" package installed.
* numpy, scipy
* scikit-learn (>=0.18)
* rpy2 2.8.6 (for python2 rpy2 can be install with: pip install rpy2==2.8.6, for python3 pip3 install rpy2==2.8.6). **It seems that newer version of rpy2 might not work due to a bug**. Thus, we highly recommand to install the 2.8.6 version for now.

## Tested versions
* python: 2.7.15, 3.6.6
* R: 3.5.0
* Numpy: 1.16.5
* Scipy: 1.2.2
* scikit-learn: 0.20.3
* keras: 2.3.1
* tensorflow: 1.14.0
* theano: 1.0.2
* rpy2: **2.8.6**


## Python packages installation

The required python packages can be installed using pip:

```bash
pip install theano --user # Original backend used OR
pip install tensorflow --user # Alternative backend for keras supposely for efficient
pip install keras --user
pip install rpy2==2.8.6 --user
```

## Support for CNTK / tensorflow
We originally used Keras with theano as backend plateform. However, [Tensorflow](https://www.tensorflow.org/) or [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) are more recent DL framework that can be faster or more stable than theano. Because keras supports these 3 backends, it is possible to use them as alternative to theano. To CNTK, please refer to the official [guidelines](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) . To change backend, please configure the `$HOME/.keras/keras.json` file. (See official instruction [here](https://keras.io/backend/)).

The default configuration file: ` ~/.keras/keras.json` looks like this:

```json
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

## R installation

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

## installation of the package

```bash
git clone https://github.com/lanagarmire/SimDeep.git
cd SimDeep
pip install -r requirements.txt --user
```

## Test
* test if simdeep is functional (all the software are correctly installed):

```bash
  python test/test_dummy_boosting_stacking.py -v # OR
  nosetests test -v # Improved version of python unit testing
  ```
