"""
Load the 3-omics abd perform subtype detecion from the HCC dataset

tsv files used in the original study are available in the ./data folder of this project.
However, theses files must be decompressed using this function in linux:
gzip -d *.gz.

"""

# Python import needed
from simdeep.simdeep_boosting import SimDeepBoosting
from simdeep.config import PATH_THIS_FILE

from collections import OrderedDict

from os.path import isfile

from sys import exit


def main():
    """ Main function excecuted """
    path_data = PATH_THIS_FILE + "/../data/"

    # Testing if the files were decompressed in the good repository
    try:
        assert(isfile(path_data + "meth.tsv"))
        assert(isfile(path_data + "rna.tsv"))
        assert(isfile(path_data + "mir.tsv"))
    except AssertionError:
        print('gz files in {0} must be decompressed !\n exiting...'.format(path_data))
        exit(1)

    # Tsv files used in the original study in the appropriate order
    tsv_files = OrderedDict([
        ('MIR', 'mir.tsv'),
        ('METH', 'meth.tsv'),
        ('RNA', 'rna.tsv'),
    ])

    # File with survival event
    survival_tsv = 'survival.tsv'

    # As test dataset we will use the rna.tsv only
    tsv_test = {'RNA': 'rna.tsv'}
    # because it is the same data, we should use the same survival file
    test_survival = 'survival.tsv'

    PROJECT_NAME = 'HCC_dataset'
    EPOCHS = 10
    SEED = 10045
    nb_it = 10
    nb_threads = 2

    survival_flag = {
        'patient_id': 'Samples',
        'survival': 'days',
        'event': 'event'}

    # Instanciate a DeepProg instance
    boosting = SimDeepBoosting(
        nb_threads=nb_threads,
        nb_it=nb_it,
        split_n_fold=3,
        survival_tsv=survival_tsv,
        training_tsv=tsv_files,
        path_data=path_data,
        project_name=PROJECT_NAME,
        path_results=path_data,
        epochs=EPOCHS,
        survival_flag=survival_flag,
        seed=SEED)

    boosting.fit()

    # predict labels of the training

    boosting.predict_labels_on_full_dataset()
    boosting.compute_clusters_consistency_for_full_labels()
    boosting.evalutate_cluster_performance()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_cindex_for_full_dataset()

    boosting.compute_feature_scores_per_cluster()
    boosting.write_feature_score_per_cluster()

    # Finally, load test set
    boosting.load_new_test_dataset(
        tsv_test,
        test_survival,
        'test_RNA_only',
    )

    boosting.predict_labels_on_test_dataset()
    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_clusters_consistency_for_test_labels()

    #All the parameters are attributes of the SimDeep instance:

    # boosting.labels
    # boosting.test_labels
    # boosting.test_labels_proba
    # ... etc...


if __name__ == "__main__":
    main()
