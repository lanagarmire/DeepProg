# Tutorial: Tuning DeepProg

DeepProg can accept various alernative hyperparameters to fit a model, including alterative clustering,  normalisation, embedding, choice of autoencoder hyperparameters, use/restrict embedding and survival selection, size of holdout samples, ensemble model merging criterion. Furthermore it can accept external methods to perform clustering / normalisation or embedding. To help ones to find the optimal combinaisons of hyperparameter for a given dataset, we implemented an optional hyperparameter search module based on sequencial-model optimisation search and relying on the [tune](https://docs.ray.io/en/master/tune.html) and [scikit-optimize](https://scikit-optimize.github.io/stable/) python libraries. The optional hyperparameter tuning will perform a non-random itertive grid search and will select each new set of hyperparameters based on the performance of the past iterations. The computation can be entirely distributed thanks to the ray interace (see above).

A DeepProg instance depends on a lot of hyperparameters. Most important hyperparameters to tune are:

* The combination of `-nb_it` (Number of sumbmodels), `-split_n_fold `(How each submodel is randomly constructed) and `-seed` (random seed).
* The number of clusters `-nb_clusters`
* The clustering algorithm (implemented: `kmeans`, `mixture`, `coxPH`, `coxPHMixture`)
* The preprocessing normalization (`-normalization` option, see `Tutorial: Advanced usage of DeepProg model`)
* The embedding used (`alternative_embedding` option)
* The way of creating the new survival features (`-feature_selection_usage` option)


## A first example

A first example of tuning is available in the [example](../../../examples/example_hyperparameters_tuning.py) folder (example_hyperparameters_tuning.py). The first part of the script defines the array of hyperparameters to screen. An instance of `SimdeepTuning` is created in which the output folder and the project name are defined.

```python

from simdeep.simdeep_tuning import SimDeepTuning

# AgglomerativeClustering is an external class that can be used as
# a clustering algorithm since it has a fit_predict method
from sklearn.cluster.hierarchical import AgglomerativeClustering

# Array of hyperparameters
args_to_optimize = {
    'seed': [100, 200, 300, 400],
    'nb_clusters': [2, 3, 4, 5],
    'cluster_method': ['mixture', 'coxPH',
                       AgglomerativeClustering],
    'use_autoencoders': (True, False),
    'class_selection': ('mean', 'max'),
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

```


The SimDeepTuning module requires the use of the `ray` and `tune` python modules.

```python

ray.init(webui_host='0.0.0.0', )

```


### SimDeepTuning hyperparameters
* `num_samples` is the number of experiements
* `distribute_deepprog` is used to further distribute each DeepProg instance into the ray framework. If set to True, be sure to either have a large number of CPUs to use and/or to use a small number of `max_concurrent` (which is the number of concurrent experiments run in parallel). `iterations` is the number of iterations to run for each experiment (results will be averaged).


DeepProg can be tuned using different objective metrics:
* `"log_test_fold_pvalue"`: uses the stacked *out of bags* samples (survival and labels) to predict the -log10(log-rank Cox-PH pvalue)
* `"log_full_pvalue"`: minimizes the Cox-PH log-rank pvalue of the model (This metric can lead to overfitting since it relies on all the samples included in the model)
* `"test_fold_cindex"`: Maximizes the mean c-index of the test folds.
* `"cluster_consistency"`: Maximizes the adjusted Rand scores computed for all the model pairs. (Higher values imply stable clusters)


```python
tuning.fit(
    # We will use the holdout samples Cox-PH pvalue as objective
    metric='log_test_fold_pvalue',
    num_samples=25,
    # Experiment run concurently using ray as dispatcher
    max_concurrent=2,
    # In addition, each deeprog model will be distributed
    distribute_deepprog=True,
    iterations=1)

# We recommend using large `max_concurrent` and distribute_deepprog=True
# when a large number CPUs and large RAMs are availables

# Results
table = tuning.get_results_table()
print(table)
```

## Tuning using one or multiple test datasets

The computation of labels and the associate metrics from external test datasets can be included in the tuning workflowand be used as objective metrics. Please refers to [example](../../../examples/example_hyperparameters_tuning_with_dataset.py) folder (see example_hyperparameters_tuning_with_dataset.py).

Let's define two dummy test datasets:

```python

    # We will use the methylation and the RNA value as test datasets
    test_datasets = {
        'testdataset1': ({'METH': 'meth_dummy.tsv'}, 'survival_dummy.tsv')
        'testdataset2': ({RNA: rna_dummy.tsv'}, 'survival_dummy.tsv')
    }
```

We then include these two datasets when instanciating the `SimDeepTuning` instance:

```python
    tuning = SimDeepTuning(
        args_to_optimize=args_to_optimize,
        test_datasets=test_datasets,
        survival_tsv=SURVIVAL_TSV,
        training_tsv=TRAINING_TSV,
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
    )
```

and Finally fits the model using a objective metric accounting for the test datasets:

* `"log_test_pval"` maximizes the sum of the -log10(log-rank Cox-PH pvalue) for each test dataset
* `"test_cindex"` maximizes the mean on the test C-indexes
* `"sum_log_pval"` maximizes the sum of the model -log10(log-rank Cox-PH pvalue) with all the test datasets p-value
* `"mix_score"`: maximizes the product of `"sum_log_pval"`, `"cluster_consistency"`, `"test_fold_cindex"`


```python
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
```

## Results

The results will be generated in the `path_results` folder and one results folder per experiement will be generated. The report of all the experiements and metrics will be written in the result tables generated in the `path_results` folder. Once a model achieve satisfactory performance, it is possible to directly use the model by loading the generated labels with the `fit_on_pretrained_label_file` API (see the section `Save / load models from precomputed sample labels`)

## Recommendation

* According to the number of the size N of the hyperparameter array(e.g. the number of combination ), it is recommanded to perform at least more than sqrt(N) experiment but a higher N will always allow to explore a higher hyperparameter space and increase the performance.
* `seed` is definitively a hyperparameter to screen, especially for small number of models `nb_its` (less than 50). It is recommanded to at least screen for 8-10 different seed when using `nb_it` < 20
* Please, test you configuration using a small `num_samples` first
