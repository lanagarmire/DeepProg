# Survival Integration of Multi-omics using Deep-Learning (SimDeep)

This package allows to combine multi-omics data together with survival. Using autoencoder, the pipeline creates new features and identify those linked with survival, using CoxPH regression. Then, using K-means it clusters the samples. Finally, using the labels inferred, the pipeline create a supervised model to label samples from a new omic dataset. The original study of this package is "Deep Learning based multi-omics integration robustly predicts survivals in liver cancer" from K. Chaudhary, O. Poirion, Lianqun Liu, and L. X. Garmire.
The omic data used in the original study are RNA-Seq, MiR and Methylation. However, this approach can beextended to any combination of omic data.

The current package contains the omic data used in the study and a copy of the model computed. However, it is very easy to recreate a new model from scratch using any combination of omic data.

The omic data and the survival files should be in tsv (Tabular Separated Values) format and examples are provided. The deep-learning framework uses Keras, which is a embedding of Theano.


## Requirements
* [theano](http://deeplearning.net/software/theano/install.html) (the used version is 0.8.2)
* R
* the R "survival" package installed. (Launch R then:)

```R
install.package("survival")
```

* numpy, scipy
* scikit-learn (>=0.18)
* rpy2

## installation (local)

```bash
git clone https://github.com/lanagarmire/SimDeep.git
cd SimDeep
pip install -r requirements.txt --user
```

## usage
* test if simdeep is functional (all the software are correctly installed):

```bash
  python test/test_ssrge.py -v # OR
  nosetests test -v # Improved version of python unit testing
  ```

* All the default parameters are defined in the config file: `./simdeep/config.py` but can be passed dynamically. Three types of parameters must be defined:
  * The training dataset (omics + survival input files)
    * In addition, the parameters of the test set, i.e. the omic dataset and the survival file
  * The parameters of the autoencoder (the default parameters works but it might be fine-tuned.
  * The parameters of the classification procedures (default are still good)

* First, we will build a model using the example dataset, in `./examples/data/`
* An omic .tsv file must have this format:

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

These example files are set as default in the config.py file. We will use them as a demo:

```python

from simdeep.simdeep_analysis import SimDeep

help(SimDeep) # to see all the functions

simDeep = SimDeep() # instantiate the model
simDeep.load_training_dataset() # load the training dataset
simDeep.fit() # fit the model

simDeep.load_test_dataset()# Load the test dataset

# The test set is a dummy rna expression (generated randomly)
print simDeep.dataset.tsv_test # Defined in the config file
# The data type of the test set is also defined to match an existing type
print simDeep.dataset.data_type # Defined in the config file
print simDeep.dataset.data_type_test # Defined in the config file
simDeep.predict_labels_on_test_dataset() # Perform the classification analysis and label the set dataset

print simDeep.test_labels
print simDeep.test_labels_proba

simDeep.save_encoder('dummy_encoder.h5')

```

* The tsv dataset used in the original study and a copy of the model generated are available in the package
* These data are in `./data/` and must be decompressed:

```bash
cd data
gzip -d *.gz

```

* Then it is very easy to build a new model or to load the already compiled model:

```python
from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData # class to load and define the datasets
from collections import OrderedDict


dataset = LoadData()

path_data = "../data/"
# Tsv files used in the original study in the appropriate order
tsv_files = OrderedDict([
          ('MIR', 'mir.tsv'),
          ('METH', 'meth.tsv'),
          ('RNA', 'rna.tsv'),
])

# File with survival event
survival_tsv = 'survival.tsv'

dataset = LoadData(
        path_data=path_data,
        training_tsv=tsv_files,
        survival_tsv=survival_tsv,
)

# Instanciate a SimDeep instance
simDeep = SimDeep(dataset=dataset)
# load the training dataset
simDeep.load_training_dataset()
# Load the full model
simDeep.load_encoder('encoder_seed_s0_full.h5')
# identify nodes linked to survival
simDeep.look_for_survival_nodes()
# predict labels of the training set using kmeans
simDeep.predict_labels()

```


* Finally, two example scripts are availables in ./examples/ which will assist you to build a model from scratch and load and use the original model


## contact and credentials
* Developer: Olivier Poirion (PhD)
* contact: opoirion@hawaii.edu, o.poirion@gmail.com

* Developer: Kumardeep Chaudhary (PhD)
* contact: kumardee@hawaii.edu , ckumardeep@gmail.com