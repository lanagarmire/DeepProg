from os.path import abspath
from os.path import split

from simdeep.simdeep_boosting import SimDeepBoosting


def test_instance():
    """
    example of SimDeepBoosting
    """
    PATH_DATA = '{0}/../examples/data/'.format(split(abspath(__file__))[0])

    #Input file
    TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv'}
    SURVIVAL_TSV = 'survival_dummy.tsv'

    PROJECT_NAME = 'TestProject'
    SEED = 3
    nb_it = 5 # Number of models to be built
    nb_threads = 2 # Number of processes to be used to fit individual survival models

    ################ AUTOENCODER PARAMETERS ################
    EPOCHS = 10
    ## Additional parameters for the autoencoders can be defined, see config.py file for details
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
    # PATH_TO_SAVE_MODEL = '/home/username/deepprog'
    # PVALUE_THRESHOLD = 0.01
    # NB_SELECTED_FEATURES = 10
    # STACK_MULTI_OMIC = False
    #########################################################

    boosting = SimDeepBoosting(
        nb_threads=nb_threads,
        nb_it=nb_it,
        split_n_fold=3,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
        epochs=EPOCHS,
        seed=SEED,
        cluster_method="coxPH",
        use_autoencoders=False,
        feature_surv_analysis=False,
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
    )

    boosting.fit()
    boosting.predict_labels_on_full_dataset()

    boosting.compute_clusters_consistency_for_full_labels()
    boosting.evalutate_cluster_performance()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()

    boosting.compute_feature_scores_per_cluster()
    boosting.write_feature_score_per_cluster()
    boosting.collect_number_of_features_per_omic()

    boosting.load_new_test_dataset(
        {'RNA': 'rna_dummy.tsv'}, # OMIC file of the test set. It doesnt have to be the same as for training
        'survival_dummy.tsv', # Survival file of the test set
        'dummy', # Name of the test test to be used
    )

    boosting.predict_labels_on_test_dataset()
    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()

    boosting.load_new_test_dataset(
        {'METH': 'meth_dummy.tsv'}, # OMIC file of the second test set.
        'survival_dummy.tsv', # Survival file of the test set
        'dummy_METH', # Name of the second test test
    )

    boosting.predict_labels_on_test_dataset()
    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()


if __name__ == '__main__':
    test_instance()
