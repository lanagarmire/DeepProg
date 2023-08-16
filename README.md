# Survival Integration of Multi-omics using Deep-Learning (DeepProg)

This package allows to combine multi-omics data together with survival. Using autoencoders, the pipeline creates new features and identify those linked with survival, using CoxPH regression.
The omic data used in the original study are RNA-Seq, MiR and Methylation. However, this approach can be extended to any combination of omic data.

The current package contains the omic data used in the study and a copy of the model computed. However, it is very easy to recreate a new model from scratch using any combination of omic data.
The omic data and the survival files should be in tsv (Tabular Separated Values) format and examples are provided. The deep-learning framework uses Keras, which is a embedding of Theano / tensorflow/ CNTK.

A more complete documentation with API description is also available at [https://deepprog-garmires-lab.readthedocs.io/en/latest/index.html/](https://deepprog-garmires-lab.readthedocs.io/en/latest/index.html)

### Documentation section
* [Installation](https://deepprog-garmires-lab.readthedocs.io/en/latest/installation.html)
* [Tutorial: Simple DeepProg model](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage.html)
* [Tutorial: Ensemble of DeepProg model](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage_ensemble.html)
* [Tutorial: Advanced usage of DeepProg model](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage_advanced.html)
* [Tutorial: use DeepProg from the docker image](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage_with_docker.html)
* [Case study: Analyzing TCGA HCC dataset](https://deepprog-garmires-lab.readthedocs.io/en/latest/case_study.html)
* [Tutorial: Tuning DeepProg](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage_tuning.html)


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


```bash
pip3 install tensorflow

# Alternative to tensorflow, original backend used
pip3 install theano

#If you want to use theano or CNTK
nano ~/.keras/keras.json
```

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

# install with conda
conda env create -n deepprog -f ./environment.yml python=3.8
conda activate deepprog
pip install -e .

# (RECOMMENDED) to install the tested python library versions
pip install -e . -r requirements_tested.txt

##Alternative installations

# Basic installation
pip3 install -e . -r requirements.txt
# To intall the distributed frameworks
pip3 install -r requirements_distributed.txt
# Installing scikit-survival (python3 only)
pip3 install -r requirements_pip3.txt

# DeepProg is working also with python2/pip2 however there is no support for scikit-survival in python2
pip2 install -e . -r requirements.txt
pip2 install -e . -r requirements_distributed.txt

# Install ALL required dependencies with the most up to date packages
pip install -e . -r requirements_all.txt
```

### Installation with docker
We have created a docker image (`opoirion/deepprog_docker:v1`) with all the dependencies already installed. For the docker (and singularity) instruction, please refer to the docker [tutorial](https://deepprog-garmires-lab.readthedocs.io/en/latest/usage_with_docker.html) (see above).


### Support for CNTK / tensorflow
* We originally used Keras with theano as backend plateform. However, [Tensorflow](https://www.tensorflow.org/) (currently used as default) or [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/) are more recent DL framework that can be faster or more stable than theano. Because keras supports these 3 backends, it is possible to use them as alternative to theano. To change backend, please configure the `$HOME/.keras/keras.json` file. (See official instruction [here](https://keras.io/backend/)).

The default configuration file looks like this:

```json
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
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
  python3 test/test_simdeep.py -v

  # Individual examples
  python3 python examples/example_with_dummy_data.py
  python3 python examples/example_with_dummy_data_distributed.py
  python3 python examples/example_with_precomputed_labels.py
  python3 python examples/example_hyperparameters_tuning.py
  python3 python examples/example_hyperparameters_tuning_with_test_dataset.py
  ```

* All the default parameters are defined in the config file: `./simdeep/config.py` but can be passed dynamically. Three types of parameters must be defined:
  * The training dataset (omics + survival input files)
    * In addition, the parameters of the test set, i.e. the omic dataset and the survival file
  * The parameters of the autoencoder (the default parameters works but it might be fine-tuned.
  * The parameters of the classification procedures (default are still good)


## Example datasets and scripts
An omic .tsv file must have this format:

```bash
head mir_dummy.tsv

Samples        dummy_mir_0     dummy_mir_1     dummy_mir_2     dummy_mir_3 ...
sample_test_0  0.469656032287  0.347987447237  0.706633335508  0.440068758445 ...
sample_test_1  0.0453108219657 0.0234642968791 0.593393816691  0.981872970341 ...
sample_test_2  0.908784043793  0.854397550009  0.575879144667  0.553333958713 ...
...

```

a survival file must have this format:

```bash
head survival_dummy.tsv

Samples        days event
sample_test_0  134  1
sample_test_1  291  0
sample_test_2  125  1
sample_test_3  43   0
...

```

As examples, we included two datasets:
* A dummy example dataset in the `example/data/` folder:
```bash
examples
├── data
│   ├── meth_dummy.tsv
│   ├── mir_dummy.tsv
│   ├── rna_dummy.tsv
│   ├── rna_test_dummy.tsv
│   ├── survival_dummy.tsv
│   └── survival_test_dummy.tsv
```

* And a real dataset in the `data` folder. This dataset derives from the TCGA HCC cancer dataset. This dataset needs to be decompressed before processing:

```bash
data
├── meth.tsv.gz
├── mir.tsv.gz
├── rna.tsv.gz
└── survival.tsv

```

## Creating a simple DeepProg model with one autoencoder for each omic
First, we will build a model using the example dataset from `./examples/data/` (These example files are set as default in the config.py file). We will use them to show how to construct a single DeepProg model inferring a autoencoder for each omic

```python

# SimDeep class can be used to build one model with one autoencoder for each omic
from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData

help(SimDeep) # to see all the functions
help(LoadData) # to see all the functions related to loading datasets

# Defining training datasets
from simdeep.config import TRAINING_TSV
from simdeep.config import SURVIVAL_TSV

dataset = LoadData(training_tsv=TRAINING_TSV, survival_tsv=SURVIVAL_TSV)

simDeep = SimDeep(dataset=dataset) # instantiate the model with the dummy example training dataset defined in the config file
simDeep.load_training_dataset() # load the training dataset
simDeep.fit() # fit the model

# Defining test datasets
from simdeep.config import TEST_TSV
from simdeep.config import SURVIVAL_TSV_TEST

simDeep.load_new_test_dataset(TEST_TSV, fname_key='dummy', path_survival_file=SURVIVAL_TSV_TEST)

# The test set is a dummy rna expression (generated randomly)
print(simDeep.dataset.test_tsv) # Defined in the config file
# The data type of the test set is also defined to match an existing type
print(simDeep.dataset.data_type) # Defined in the config file
simDeep.predict_labels_on_test_dataset() # Perform the classification analysis and label the set dataset

print(simDeep.test_labels)
print(simDeep.test_labels_proba)

simDeep.save_encoders('dummy_encoder.h5')

```
## Creating a DeepProg model using an ensemble of submodels

Secondly, we will build a more complex DeepProg model constituted of an ensemble of sub-models each originated from a subset of the data. For that purpose, we need to use the `SimDeepBoosting` class:

```python
from simdeep.simdeep_boosting import SimDeepBoosting

help(SimDeepBoosting)

from collections import OrderedDict


path_data = "../examples/data/"
# Example tsv files
tsv_files = OrderedDict([
          ('MIR', 'mir_dummy.tsv'),
          ('METH', 'meth_dummy.tsv'),
          ('RNA', 'rna_dummy.tsv'),
])

# File with survival event
survival_tsv = 'survival_dummy.tsv'

project_name = 'stacked_TestProject'
epochs = 10 # Autoencoder epochs. Other hyperparameters can be fine-tuned. See the example files
seed = 3 # random seed used for reproducibility
nb_it = 5 # This is the number of models to be fitted using only a subset of the training data
nb_threads = 2 # These treads define the number of threads to be used to compute survival function

boosting = SimDeepBoosting(
    nb_threads=nb_threads,
    nb_it=nb_it,
    split_n_fold=3,
    survival_tsv=survival_tsv,
    training_tsv=tsv_files,
    path_data=path_data,
    project_name=project_name,
    path_results=path_data,
    epochs=epochs,
    seed=seed)

# Fit the model
boosting.fit()
# Predict and write the labels
boosting.predict_labels_on_full_dataset()
# Compute internal metrics
boosting.compute_clusters_consistency_for_full_labels()
# COmpute the feature importance
boosting.compute_feature_scores_per_cluster()
# Write the feature importance
boosting.write_feature_score_per_cluster()

boosting.load_new_test_dataset(
    {'RNA': 'rna_dummy.tsv'}, # OMIC file of the test set. It doesnt have to be the same as for training
    'TEST_DATA_1', # Name of the test test to be used
    'survival_dummy.tsv', # [OPTIONAL] Survival file of the test set
)

# Predict the labels on the test dataset
boosting.predict_labels_on_test_dataset()
# Compute C-index
boosting.compute_c_indexes_for_test_dataset()
# See cluster consistency
boosting.compute_clusters_consistency_for_test_labels()

# [EXPERIMENTAL] method to plot the test dataset amongst the class kernel densities
boosting.plot_supervised_kernel_for_test_sets()
```

## Creating a distributed DeepProg model using an ensemble of submodels

We can allow DeepProg to distribute the creation of each submodel on different clusters/nodes/CPUs by using the ray framework.
The configuration of the nodes / clusters, or local CPUs to be used needs to be done when instanciating a new ray object with the ray [API](https://ray.readthedocs.io/en/latest/). It is however quite straightforward to define the number of instances launched on a local machine such as in the example below in which 3 instances are used.

```python
# Instanciate a ray object that will create multiple workers
import ray
ray.init(webui_host='0.0.0.0', num_cpus=3)
# More options can be used (e.g. remote clusters, AWS, memory,...etc...)
# ray can be used locally to maximize the use of CPUs on the local machine
# See ray API: https://ray.readthedocs.io/en/latest/index.html

boosting = SimDeepBoosting(
    ...
    distribute=True, # Additional option to use ray cluster scheduler
    ...
)
...
# Processing
...

# Close clusters and free memory
ray.shutdown()
```

## Hyperparameter search
DeepProg can accept various alernative hyperparameters to fit a model, including alterative clustering,  normalisation, embedding, choice of autoencoder hyperparameters, use/restrict embedding and survival selection, size of holdout samples, ensemble model merging criterion. Furthermore it can accept external methods to perform clustering / normalisation or embedding. To help ones to find the optimal combinaisons of hyperparameter for a given dataset, we implemented an optional hyperparameter search module based on sequencial-model optimisation search and relying on the [tune](https://docs.ray.io/en/master/tune.html) and [scikit-optimize](https://scikit-optimize.github.io/stable/) python libraries. The optional hyperparameter tuning will perform a non-random itertive grid search and will select each new set of hyperparameters based on the performance of the past iterations. The computation can be entirely distributed thanks to the ray interace (see above).

```python

from simdeep.simdeep_tuning import SimDeepTuning

# AgglomerativeClustering is an external class that can be used as
# a clustering algorithm since it has a fit_predict method
from sklearn.cluster.hierarchical import AgglomerativeClustering

args_to_optimize = {
    'seed': [100, 200, 300, 400],
    'nb_clusters': [2, 3, 4, 5],
    'cluster_method': ['mixture', 'coxPH',
                       AgglomerativeClustering],
    'use_autoencoders': (True, False),
    'class_selection': ('mean', 'max'),
}

tuning = SimDeepTuning(
    args_to_optimize=args_to_optimize,
    nb_threads=nb_threads,
    survival_tsv=SURVIVAL_TSV,
    training_tsv=TRAINING_TSV,
    path_data=PATH_DATA,
    project_name=PROJECT_NAME,
    path_results=PATH_DATA,
)

ray.init(webui_host='0.0.0.0')


tuning.fit(
    # We will use the holdout samples Cox-PH pvalue as objective
    metric='log_test_fold_pvalue',
    num_samples=25,
    # Experiment run concurently using ray as dispatcher
    max_concurrent=2,
    # In addition, each deeprog model will be distributed
    distribute_deepprog=True,
    iterations=1)

# We recommend using large `max_concurrent` and distribute_deepprog=True
# when a large number CPUs and large RAMs are availables

# Results
table = tuning.get_results_table()
print(table)
```

## Save / load models

Two mechanisms exist to save and load dataset.
First the models can be entirely saved and loaded using `dill` (pickle like) libraries.

```python
from simdeep.simdeep_utils import save_model
from simdeep.simdeep_utils import load_model

# Save previous boosting model
save_model(boosting, "./test_saved_model")

# Delete previous model
del boosting

# Load model
boosting = load_model("TestProject", "./test_saved_model")
boosting.predict_labels_on_full_dataset()

```

See an example of saving/loading model in the example file: `examples/load_and_save_models.py`

However, this mechanism presents a huge drawback since the models saved can be very large (all the hyperparameters/matrices... etc... are saved). Also, the equivalent dependencies and DL libraries need to be installed in both the machine computing the models and the machine used to load them which can lead to various errors.

A second solution is to save only the labels inferred for each submodel instance. These label files can then be loaded into a new DeepProg instance that will be used as reference for building the classifier.

```python

# Fitting a model
boosting.fit()
# Saving individual labels
boosting.save_test_models_classes(
    path_results=PATH_PRECOMPUTED_LABELS # Where to save the labels
    )

boostingNew = SimDeepBoosting(
        survival_tsv=SURVIVAL_TSV, # Same reference training set for `boosting` model
        training_tsv=TRAINING_TSV, # Same reference training set for `boosting` model
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
        distribute=False, # Option to use ray cluster scheduler (True or False)
    )

boostingNew.fit_on_pretrained_label_file(
    labels_files_folder=PATH_PRECOMPUTED_LABELS,
    file_name_regex="*.tsv")

boostingNew.predict_labels_on_full_dataset()
```

See the `examples/example_simdeep_start_from_pretrained_labels.py` example file.

## Example scripts

Example scripts are availables in ./examples/ which will assist you to build a model from scratch with test and real data:

```bash
examples
├── example_hyperparameters_tuning.py
├── example_hyperparameters_tuning_with_test_dataset.py
├── example_with_dummy_data_distributed.py
├── example_with_dummy_data.py
└── load_3_omics_model.py
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



## License

The project is licensed under the PolyForm Perimeter License 1.0.0.

(See https://polyformproject.org/licenses/)

## Citation

This package refers to our study published in Genome Biology: [Multi-omics-based pan-cancer prognosis prediction using an ensemble of deep-learning and machine-learning models](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-021-00930-x)

## Data avaibility

The matrices and the survival data used to compute the models are available here [https://doi.org/10.6084/m9.figshare.14832813.v1](https://doi.org/10.6084/m9.figshare.14832813.v1)

## contact and credentials
* Developer: Olivier Poirion (PhD)
* contact: opoirion@hawaii.edu, o.poirion@gmail.com
