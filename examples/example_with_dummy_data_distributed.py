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
    EPOCHS = 10
    SEED = 3
    nb_it = 5
    nb_threads = 2

    # Import cluster scheduler
    import ray
    ray.init(num_cpus=3)
    # More options can be used (e.g. remote clusters, AWS, memory,...etc...)
    # ray can be used locally to maximize the use of CPUs on the local machine
    # See ray API: https://ray.readthedocs.io/en/latest/index.html

    boosting = SimDeepBoosting(
        # stack_multi_omic=True,
        nb_threads=nb_threads,
        nb_it=nb_it,
        split_n_fold=3,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
        epochs=EPOCHS,
        distribute=True, # Option to use ray cluster scheduler
        seed=SEED)

    boosting.fit()
    boosting.predict_labels_on_full_dataset()

    boosting.compute_clusters_consistency_for_full_labels()
    boosting.evalutate_cluster_performance()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()

    boosting.compute_feature_scores_per_cluster()
    boosting.collect_number_of_features_per_omic()

    boosting.write_feature_score_per_cluster()

    boosting.load_new_test_dataset(
        {'RNA': 'rna_dummy.tsv'}, # OMIC file of the test set. It doesnt have to be the same as for training
        'survival_dummy.tsv', # Survival file of the test set
        'dummy', # Name of the test test to be used
    )

    boosting.predict_labels_on_test_dataset()
    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    # boosting.plot_supervised_kernel_for_test_sets()
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
    # boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()

    # Close clusters and free memory
    ray.shutdown()


if __name__ == '__main__':
    test_instance()
