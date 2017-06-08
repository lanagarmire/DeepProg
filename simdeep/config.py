from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from collections import OrderedDict

from os.path import abspath
from os.path import split as pathsplit

from sklearn.model_selection import KFold

# absolute path of this file
PATH_THIS_FILE = pathsplit(abspath(__file__))[0]

#################### SimDeep variable ##################
NB_CLUSTERS = 2 # Number of clusters
CLUSTER_METHOD = 'mixture'
CLUSTER_EVAL_METHOD = 'bic'
CLASSIFIER_TYPE = 'mixture'
CLUSTER_ARRAY = [2,
                 3,
                 # 4, 5,
                 # 6, 7
]
PVALUE_THRESHOLD = 0.05 # Threshold for survival significance to set a node as valid
########################################################

#################### Paths to data file ################
# path to the folder containing the data

# PATH_DATA = PATH_THIS_FILE + "/../examples/data/"
PATH_DATA = "/home/opoirion/data/survival_analysis_multiple/sijia/raw_matrix_and_raw_testing/"

# name of the tsv file containing the survival data of the training set
SURVIVAL_TSV = 'raw_merged_survival.tsv'

# dict('data type', 'name of the tsv file which are inside PATH_DATA')
# These data will be stacked together to build the autoencoder
TRAINING_TSV = OrderedDict([
    ('CNV', '0607_raw_cnv_data.tsv'),
    # ('METH', '0607_raw_methyl_data.tsv'),
    # ('RNA', '0607_raw_expr_data.tsv'),
])

######## Cross-validation on the training set ############
CROSS_VALIDATION_INSTANCE = KFold(n_splits=5,
                                  shuffle=True,
                                  random_state=1)
TEST_FOLD = 0
##########################################################

TEST_TSV = {
    'CNV': '0607_raw_cnv_testing_data.tsv',
    # 'RNA': '0607_raw_expr_testing_data.tsv',
    # 'METH': '0607_raw_methyl_testing_data.tsv',
}

# name of the tsv file containing the survival data of the test set
SURVIVAL_TSV_TEST = 'raw_testing_merged_survival.tsv'

# Path where to save load the Keras models
PATH_MODEL = PATH_THIS_FILE + '/../data/models/'
########################################################

##################### NORMALIZATION PROCEDURE ###########
## Normalize before the autoencoder construction ########
TRAIN_ROBUST_SCALE = False
TRAIN_MAD_SCALE = False
TRAIN_MIN_MAX = False
TRAIN_NORM_SCALE = False
TRAIN_RANK_NORM = True
TRAIN_CORR_REDUCTION = True
TRAIN_CORR_RANK_NORM = False
#########################################################

##################### Autoencoder Variable ##############
# Dimensions of the intermediate layers before and after the middle hidden layer
# if LEVEL_DIMS == [500, 250] then there will be two hidden layers with 500 and 250 nodes
# before and after the hidden middle layer (5 hidden layers)
# if LEVEL_DIMS = [], then the autoencoder will have only one hidden layer
LEVEL_DIMS = []
# Number of nodes in the middle hidden layer
# (i.e. the new dimensions of the transformed data)
NEW_DIM = 100
# Percentage of edges being dropout at each training iteration (None for no dropout)
DROPOUT = 0.5
# L2 Regularization constant on the node activity
ACT_REG = False
# L1 Regularization constant on the weight
W_REG = False
# Fraction of the dataset to be used as test set when building the autoencoder
DATA_SPLIT = None
# activation function
ACTIVATION = 'tanh'
# Number of epoch
NB_EPOCH = 50
# Loss function to minimize
LOSS = 'mse'
# Optimizer (sgd for Stochastic Gradient Descent)
OPTIMIZER = 'adam'
########################################################

################## CLASSIFIER ##########################
# Variables used to perform the supervized classification procedure
# to assign labels to the test set

# Top K features retained by omic type.
# If a new feature is added in the TRAINING_TSV variable this dict must be updated
MIXTURE_PARAMS = {
    'covariance_type': 'spherical',
    'max_iter': 10000,
    'n_init': 100
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

##########################################################

#################### Other variables #####################
SEED = None # for experiment reproducibility (if set to an integer)
##########################################################
