from simdeep.simdeep_boosting import SimDeepBoosting


def test_instance():
    """
    example of SimDeepBoosting script configured for the docker image

    The image can bepulled using the following commands:
    # DeepProg version including R and the R libraries for computing CoxPH and c-indexes
    docker pull opoirion/deepprog_docker:RVersion1

    # DeepProg version using uniquely python and lifelines library (Please set use_r_packages to False)
    docker pull opoirion/deepprog_docker:v1

    We will assume that
    a) the DeepProg package is installed in ~/code
    b) a ~/data directory exists (for storing the output)

    This script can be launched using the following docker command:

    docker run --rm --name deepshrimp \
                    -v ~/code/DeepProg/examples/data/:/input/ \
                     -v ~/data/:/output \
                     -v ~/code/DeepProg/examples/:/code \
                     opoirion/deepprog_docker:RVersion1 \
                     python3.8 /code/example_with_dummy_data_distributed_using_docker.py

    """
    PATH_DATA = '/input/' # Input folder inside the docker image to mount
    PATH_RESULTS = '/output/' # Output folder inside the docker image to mount

    #Input file
    TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv'}
    SURVIVAL_TSV = 'survival_dummy.tsv'

    PROJECT_NAME = 'TestProject'
    EPOCHS = 10
    SEED = 3
    nb_it = 5
    nb_threads = 2

    # Optional metadata FILE
    OPTIONAL_METADATA = "metadata_dummy.tsv"

    # Import cluster scheduler
    import ray
    # ray.init(num_cpus=3)
    # More options can be used (e.g. remote clusters, AWS, memory,...etc...)
    # ray can be used locally to maximize the use of CPUs on the local machine
    # See ray API: https://ray.readthedocs.io/en/latest/index.html

    boosting = SimDeepBoosting(
        nb_threads=nb_threads,
        nb_it=nb_it,
        split_n_fold=3,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_RESULTS,
        metadata_tsv=OPTIONAL_METADATA, # optional
        use_r_packages=True, # to use R functions from the survival and survcomp packages instead of python lifelines package
        metadata_usage='all',
        epochs=EPOCHS,
        distribute=False, # Option to use ray cluster scheduler, we used it as false for testing and preventing memory issue on local computer
        seed=SEED)

    boosting.fit()
    boosting.save_models_classes()
    boosting.save_cv_models_classes()

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
        'dummy', # Name of the test test to be used
        'survival_dummy.tsv', # Survival file of the test set
    )

    boosting.predict_labels_on_test_dataset()
    boosting.save_test_models_classes()

    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()

    boosting.load_new_test_dataset(
        {'METH': 'meth_dummy.tsv'}, # OMIC file of the second test set.
        'dummy_METH', # Name of the second test test
        'survival_dummy.tsv', # Survival file of the test set (optional)
    )

    boosting.predict_labels_on_test_dataset()
    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    # Experimental method to plot the test dataset amongst the class kernel densities
    boosting.plot_supervised_kernel_for_test_sets()
    boosting.plot_supervised_predicted_labels_for_test_sets()

    # Close clusters and free memory
    ray.shutdown()


if __name__ == '__main__':
    test_instance()
