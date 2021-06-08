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

    # Optional metadata FILE
    OPTIONAL_METADATA = "metadata_dummy.tsv"

    # Subsetting training set with only males from metadata:
    SUBSET_TRAINING_WITH_META = {'stage': ['I', 'II', 'III']}

    PROJECT_NAME = 'TestProject'
    SEED = 3
    nb_it = 5 # Number of models to be built
    nb_threads = 2 # Number of processes to be used to fit individual survival models

    ################ AUTOENCODER PARAMETERS ################
    EPOCHS = 10
    ## Additional parameters for the autoencoders can be defined, see config.py file for details
    #########################################################

    ################ ADDITIONAL PARAMETERS ##################
    # PATH_TO_SAVE_MODEL = '/home/username/deepprog'
    # PVALUE_THRESHOLD = 0.01
    # NB_SELECTED_FEATURES = 10
    # STACK_MULTI_OMIC = False
    #########################################################

    # IT is possible to define a custom normalisation
    # from sklearn.preprocessing import RobustScaler
    # norm = {
    #         'CUSTOM': RobustScaler,
    # }

    boosting = SimDeepBoosting(
        nb_threads=nb_threads,
        nb_it=nb_it,
        split_n_fold=3,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        # metadata_tsv=OPTIONAL_METADATA, # optional
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
        use_r_packages=True, # to use R functions from the survival and survcomp packages
        epochs=EPOCHS,
        seed=SEED,
        # normalization=norm,
        cluster_method='coxPH',
        metadata_usage='labels',
        use_autoencoders=True,
        feature_surv_analysis=True,
        feature_selection_usage="lasso",
        # subset_training_with_meta=SUBSET_TRAINING_WITH_META,
        # stack_multi_omic=True,
        # path_to_save_model=PATH_TO_SAVE_MODEL,
        # pvalue_threshold=PVALUE_THRESHOLD,
        # nb_selected_features=NB_SELECTED_FEATURES,
    )

    boosting.fit()
    boosting.predict_labels_on_full_dataset()

    boosting.save_models_classes()
    boosting.save_cv_models_classes()

    boosting.compute_clusters_consistency_for_full_labels()
    boosting.evalutate_cluster_performance()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()

    boosting.compute_feature_scores_per_cluster()
    boosting.compute_survival_feature_scores_per_cluster(pval_thres=0.10)
    boosting.write_feature_score_per_cluster()
    boosting.collect_number_of_features_per_omic()
    boosting.compute_pvalue_for_merged_test_fold()

    boosting.load_new_test_dataset(
        tsv_dict={'RNA': 'rna_dummy.tsv'}, # OMIC file of the test set. It doesnt have to be the same as for training
        path_survival_file='survival_dummy.tsv', # Optional survival file of the test set for computing validation log-rank pvalue
        fname_key='dummy', # Name of the test test to be used
    )

    boosting.predict_labels_on_test_dataset()
    boosting.save_test_models_classes()

    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()

    boosting.load_new_test_dataset(
        tsv_dict={'METH': 'meth_dummy.tsv'}, # OMIC file of the second test set.
        path_survival_file='survival_dummy.tsv', # Survival file of the test set
        fname_key='dummy_METH', # Name of the second test test
        metadata_file="metadata_dummy.tsv" # Optional metadata
    )

    boosting.predict_labels_on_test_dataset()
    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()


if __name__ == '__main__':
    test_instance()
