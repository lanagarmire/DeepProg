from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from collections import OrderedDict

from os.path import abspath
from os.path import split as pathsplit

# absolute path of this file
PATH_THIS_FILE = pathsplit(abspath(__file__))[0]

#################### SimDeep variable ##################
NB_CLUSTERS = 3 # Number of clusters
CLUSTER_METHOD = 'mixture'
CLUSTER_EVAL_METHOD = 'silhouette'
CLASSIFIER_TYPE = 'svm'
CLUSTER_ARRAY = []
PVALUE_THRESHOLD = 0.01 # Threshold for survival significance to set a node as valid

#### Boosting values
NB_ITER = 2 # boosting iteration
NB_THREADS = 4 # number of simdeep instance launched in parallel
NB_FOLDS = 3 # for each instance, the original dataset is split in folds and one fold is left
CLASS_SELECTION = 'mean' # mean or max: the method used to select the final class, according to class probas
########################################################

#################### Paths to data file ################
# path to the folder containing the data

# PATH_DATA = PATH_THIS_FILE + "/../examples/data/"
PROJECT_NAME = 'DREAM challenge'
PATH_DATA = "/home/opoirion/data/survival_analysis_multiple/dream_myeloma_challenge/"

# name of the tsv file containing the survival data of the training set
SURVIVAL_TSV = 'Clinical_Data/globalClinTraining.csv'

# Field from the survival tsv file
SURVIVAL_FLAG = {'patient_id': '"Patient"',
                  'survival': '"D_OS"',
                 'event': '"D_OS_FLAG"'}

# dict('data type', 'name of the tsv file which are inside PATH_DATA')
# These data will be stacked together to build the autoencoder
TRAINING_TSV = OrderedDict([
    ('RNA', 'Expression_Data/microarray/EMTAB4032entrezIDlevel.csv'),
])

SEPARATOR = {
    'Expression_Data/rnaseq/MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_TPM.txt': '\t',
    'Expression_Data/rnaseq/MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_Counts.txt': '\t',
    'Expression_Data/microarray/GSE24080UAMSentrezIDlevel.csv': ',',
    'Expression_Data/microarray/EMTAB4032entrezIDlevel.csv': ',',
    'Expression_Data/microarray/GSE19784HOVON65entrezIDlevel.csv': ',',
    'Expression_Data/microarray/GSE9782APEXentrezIDlevel_mas5.csv': ','
    }

TEST_TSV = {
    'RNA': 'Expression_Data/microarray/GSE19784HOVON65entrezIDlevel.csv',
}

# name of the tsv file containing the survival data of the test set
SURVIVAL_TSV_TEST = SURVIVAL_TSV

# Path where to save load the Keras models
PATH_MODEL = '/home/opoirion/data/survival_analysis_multiple/models/'

# Path to generate png images
PATH_RESULTS = '/home/opoirion/code/d3visualisation/dream_challenge//'

######## Cross-validation on the training set ############
CROSS_VALIDATION_INSTANCE =  KFold(n_splits=3,
                                  shuffle=True,
                                  random_state=1
)
TEST_FOLD = 0
##########################################################
########################################################

##################### NORMALIZATION PROCEDURE ###########
## Normalize before the autoencoder construction ########
TRAIN_MIN_MAX = False
TRAIN_ROBUST_SCALE = False
TRAIN_MAD_SCALE = False
TRAIN_NORM_SCALE = False
TRAIN_RANK_NORM = True
TRAIN_CORR_REDUCTION = True
TRAIN_CORR_RANK_NORM = True
#########################################################

##################### Autoencoder Variable ##############
# Dimensions of the intermediate layers before and after the middle hidden layer
# if LEVEL_DIMS == [500, 250] then there will be two hidden layers with 500 and 250 nodes
# before and after the hidden middle layer (5 hidden layers)
# if LEVEL_DIMS = [], then the autoencoder will have only one hidden layer
LEVEL_DIMS_IN = [50]
LEVEL_DIMS_OUT = [50]
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
LOSS = 'binary_crossentropy'
# Optimizer (sgd for Stochastic Gradient Descent)
OPTIMIZER = 'adam'
########################################################

################## CLASSIFIER ##########################
# Variables used to perform the supervized classification procedure
# to assign labels to the test set

# Top K features retained by omic type.
# If a new feature is added in the TRAINING_TSV variable this dict must be updated
MIXTURE_PARAMS = {
    'covariance_type': 'diag',
    'max_iter': 1000,
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
