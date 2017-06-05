from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif

from collections import OrderedDict

from os.path import abspath
from os.path import split as pathsplit

# absolute path of this file
PATH_THIS_FILE = pathsplit(abspath(__file__))[0]

#################### SimDeep variable ##################
NB_CLUSTERS = 2 # Number of clusters
PVALUE_THRESHOLD = 0.05 # Threshold for survival significance to set a node as valid
########################################################

#################### Paths to data file ################
# path to the folder containing the data
# PATH_DATA = PATH_THIS_FILE + "/../examples/data/"
PATH_DATA = "/home/opoirion/data/survival_analysis_multiple/sijia/"

# name of the tsv file containing the survival data of the training set
SURVIVAL_TSV = '0601_raw_merged_survival.tsv'

# dict('data type', 'name of the tsv file which are inside PATH_DATA')
# These data will be stacked together to build the autoencoder
TRAINING_TSV = OrderedDict([
    # ('MIR', 'mir_dummy.tsv'),
    # ('METH', 'meth_dummy.tsv'),
    # ('RNA', 'rna_dummy.tsv'),
    ('SIJIA', '0601_brca_raw_merged_matrix.tsv'),

])

# name of the file containing the test dataset
TSV_TEST = '0601_brca_raw_merged_matrix.tsv'
# name of the tsv file containing the survival data of the test set
SURVIVAL_TSV_TEST = '0601_raw_merged_survival.tsv'
# name of the data type of the test set
DATA_TYPE_TEST = 'SIJIA'
# Path where to save load the Keras models
PATH_MODEL = PATH_THIS_FILE + '/../data/models/'
########################################################

##################### NORMALIZATION PROCEDURE ###########
MAD_SCALE=False
ROBUST_SCALE=False
MIN_MAX_SCALE=False
UNIT_NORM=False
#########################################################

##################### Autoencoder Variable ##############
# Dimensions of the intermediate layers before and after the middle hidden layer
# if LEVEL_DIMS == [500, 250] then there will be two hidden layers with 500 and 250 nodes
# before and after the hidden middle layer (5 hidden layers)
# if LEVEL_DIMS = [], then the autoencoder will have only one hidden layer
LEVEL_DIMS = [500]
# Number of nodes in the middle hidden layer
# (i.e. the new dimensions of the transformed data)
NEW_DIM = 100
# Percentage of edges being dropout at each training iteration (None for no dropout)
DROPOUT = 0.5
# L2 Regularization constant on the node activity
ACT_REG = 0.0001
# L1 Regularization constant on the weight
W_REG = 0.001
# Fraction of the dataset to be used as test set when building the autoencoder
DATA_SPLIT = 0.2
# activation function
ACTIVATION = 'tanh'
# Number of epoch
NB_EPOCH = 10
# Loss function to minimize
LOSS = 'binary_crossentropy'
# Optimizer (sgd for Stochastic Gradient Descent)
OPTIMIZER = 'sgd'
########################################################

################## CLASSIFIER ##########################
# Variables used to perform the supervized classification procedure
# to assign labels to the test set

# Top K features retained by omic type.
# If a new feature is added in the TRAINING_TSV variable this dict must be updated
NDIM_CLASSIF = {
    'RNA': 100,
    'MIR': 50,
    'METH': 50,
}


# Hyper parameters used to perform the grid search to find the best classifier
HYPER_PARAMETERS = [
    {'kernel': ['rbf'],
     'class_weight': [None, 'balanced'],
     'gamma': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001],
     'C': [1000, 750, 500, 250, 100, 50, 10, 5, 1, 0.1],
     'max_iter':[10000],
     },
    {'kernel': ['linear'],
     'class_weight': [None, 'balanced'],
     'C': [1000, 750, 500, 250, 100, 50, 10, 5, 1, 0.1],
     'max_iter':[10000],
     }
]

# grid search classifier using Support Vector Machine Classifier (SVC)
CLASSIFIER = GridSearchCV(SVC(), HYPER_PARAMETERS, cv=5)

# Feature selection method
SELECT_FEATURES_METHOD = {
    key: SelectKBest(k=NDIM_CLASSIF[key],
                       score_func=f_classif)
    for key in NDIM_CLASSIF
}
##########################################################

#################### Other variables #####################
SEED = None # for experiment reproducibility (if set to an integer)
##########################################################
