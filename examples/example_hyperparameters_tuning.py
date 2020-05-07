import os
os.environ['PYTHONHASHSEED']=str(2020)

import random
random.seed(2020)

from os.path import abspath
from os.path import split

from simdeep.simdeep_tuning import SimDeepTuning

import ray


def test_instance():
    """
    example of SimDeepBoosting
    """
    PATH_DATA = '{0}/../examples/data/'.format(split(abspath(__file__))[0])

    #Input file
    TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv'}
    SURVIVAL_TSV = 'survival_dummy.tsv'

    PROJECT_NAME = 'TestProjectTuning'
    nb_threads = 2 # Number of processes to be used to fit individual survival models

    ################ AUTOENCODER PARAMETERS ################
    ## Additional parameters for the autoencoders can be parsed in the hyperparameter tuning
    # LEVEL_DIMS_IN = [250]
    # LEVEL_DIMS_OUT = [250]
    # LOSS = 'binary_crossentropy'
    # OPTIMIZER = 'adam'
    # ACT_REG = 0
    # W_REG = 0
    # DROPOUT = 0.5
    # DATA_SPLIT = 0
    # ACTIVATION = 'tanh'
    #########################################################

    ################ ADDITIONAL PARAMETERS ##################
    # stack_multi_omic=STACK_MULTI_OMIC,
    # level_dims_in=LEVEL_DIMS_IN,
    # level_dims_out=LEVEL_DIMS_OUT,
    # loss=LOSS,
    # optimizer=OPTIMIZER,
    # act_reg=ACT_REG,
    # w_reg=W_REG,
    # dropout=DROPOUT,
    # data_split=DATA_SPLIT,
    # activation=ACTIVATION,
    # path_to_save_model=PATH_TO_SAVE_MODEL,
    # pvalue_threshold=PVALUE_THRESHOLD,
    # nb_selected_features=NB_SELECTED_FEATURES,
    # pvalue_threshold = 0.01
    # nb_selected_features = 10
    # stack_multi_omic = False
    #########################################################

    # ray.init(num_cpus=3)

    args_to_optimize = {
        'seed': [100, 200, 300, 400],
        'nb_clusters': [2, 5],
        'cluster_method': ['mixture', 'kmeans', 'coxPH'],
        'stack_multi_omic': (True, False)
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

    ray.init()
    tuning.fit(
        num_samples=20,
        max_concurrent=4,
        iterations=1)

    table = tuning.get_results_table()

    tuning.save_results_table()

    ray.shutdown()


if __name__ == '__main__':
    test_instance()
