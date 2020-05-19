"""
This example details how to optimize the choice of the hyperparameters to cluster
a multi-omic dataset using a reference test dataset as objective.

Multiple objective criteria can be used, such as test model final cox-PH pvalue,
, cluster consistency,
 c-index for out-of-bags samples or for the full labels, mix score, or sum of the pvalues

"""

from os.path import abspath
from os.path import split

from simdeep.simdeep_tuning import SimDeepTuning

import ray


def test_instance():
    """
    example of SimDeepBoosting
    """
    PATH_DATA = '{0}/../examples/data/'.format(split(abspath(__file__))[0])

    # Input file. We will only cluster on the RNA features, see below
    TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv'}
    SURVIVAL_TSV = 'survival_dummy.tsv'

    PROJECT_NAME = 'TestProjectTuning'

    # We will use the methylation value as test dataset
    test_datasets = {
        'testdataset1': ({'METH': 'meth_dummy.tsv'}, 'survival_dummy.tsv')
    }

    # AgglomerativeClustering is an external class that can be used as
    # a clustering algorithm since it has a fit_predict method
    from sklearn.cluster.hierarchical import AgglomerativeClustering

    args_to_optimize = {
        'seed': [
            100, 200, 300, 400,
        ],
        'nb_clusters': [2, 5],
        'cluster_method': [
            'mixture',
            AgglomerativeClustering
        ],
    }

    tuning = SimDeepTuning(
        args_to_optimize=args_to_optimize,
        test_datasets=test_datasets,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
        clustering_omics=['RNA'], # Only cluster on RNA
    )

    ray.init()

    # Possible metrics for test set: {
    #         "log_test_pval",
    #         "test_cindex",
    #         "test_consisentcy",
    #         "sum_log_pval",
    #     }

    tuning.fit(
        metric='log_test_pval',
        num_samples=10,
        distribute_deepprog=True,
        max_concurrent=2,
        # iterations is usefull to take into account the DL parameter fitting variations
        iterations=1,
    )

    table = tuning.get_results_table()
    tuning.save_results_table()

    ray.shutdown()


if __name__ == '__main__':
    test_instance()
