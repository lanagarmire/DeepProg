# Tutorial: Advanced usage of DeepProg model

## Visualisation
Once a DeepProg model is fitted, it might be interessant to obtain different visualisations of the samples for the training or the test sets, based on new survival features inferred by the autoencoders.For that purpose, we developped two methods to project the samples into a 2D space that can be called once a `SimDeepBoosting` or a `simDeep` is fitted.

```python
# boosting class instance fitted using the ensemble tutorial
boosting.plot_supervised_predicted_labels_for_test_sets()
```

The first method transforms the OMIC matrix activities into the new survival feature space inferred by the autoencoders and projects the samples into a 2D space using  PCA analysis. The figure creates a kernel density for each cluster and project the labels of the test set.

![kdplot 1](./img/stacked_TestProject_TEST_DATA_2_KM_plot_boosting_test_kde_2_cropped.png)

A second more sophisticated method uses the new features inferred by the autoencoders to compute new features by constructing a supervised network targetting the inferred subtype labels. The new set of features are then projected into a 2D space using PCA analysis. This second method might present more efficient visualisations of the different clusters since it is uses a supervised algorithm.

```python
boosting.plot_supervised_kernel_for_test_sets()
```

![kdplot 2](./img/stacked_TestProject_TEST_DATA_2_KM_plot_boosting_test_kde_1_cropped.png)

Note that these visualisation are not very efficient in that example dataset, since we have only a limited number of samples (40) and features. However, they might become more useful for real datasets.


## Hyperparameters

Hyperparameters can have a considerable influence on the accuracy of DeepProgs models. We set up the default hyperparameters to be used on a maximum of different datasets. However, specific datasets might require additional optimizations. Below, we are listing

### Number of clusters

The parameters `nb_clusters` is used to define the number of partitions to produce

```python
#Example
boosting = SimDeepBoosting(
    nb_clusters=3)
boosting.fit()
```

### Clustering algorithm

By default, DeepProg is using a gaussian mixture model from the scikit-learn library  to perform clustering. The hyperparameter of the model are customisable using the `mixture_params` parameter:

```python
# Default params from the config file:

MIXTURE_PARAMS = {
    'covariance_type': 'diag',
    'max_iter': 1000,
    'n_init': 100
    }

boosting = SimDeepBoosting(
    mixture_params=MIXTURE_PARAMS,
    nb_clusters=3,
    cluster_method='mixture' # Default
    )
```

In addition, two alternative clustering approaches are available `kmeans`, which refers to the scikit-learn KMeans class and `coxPH` which fits a L1 penalized multi-dimensional Cox-PH model and then dichotomize the samples into K groups using the  predicted suvival times. The L1 penalised Cox-PH model is fitted using scikit-survival `CoxnetSurvivalAnalysis`class for python3 so it cannot be computed when using python 2. Finally, external clustering class instances can be used as long as they have a `fit_predict` method returning an array of labels, and accepting a `nb_clusters` parameter.

```python
# External clustering class having fit_predict method
from sklearn.cluster.hierarchical import AgglomerativeClustering

boostingH = SimDeepBoosting(
        nb_clusters=3,
        cluster_method=AgglomerativeClustering # Default
    )


class DummyClustering:
    self __init__(self, nb_clusters):
        """ """
        self.nb_clusters

    def fit_predict(M):
        """ """
        import numpy as np
        return np.random.randint(0, self.nb_clusters, M.shape[0])


boostingDummy = SimDeepBoosting(
        nb_clusters=3,
        cluster_method=DummyClustering # Default
    )
```

### Normalisation

DeepProg uses by default a four-step normalisation for both training and test datasets:
1. Selection of the top 100 features according to the variances
2. Rank normalisation per sample
3. Sample-sample Correlation similarity transformation
4. Rank normalisation

```python
default_normalisation =  {
    'NB_FEATURES_TO_KEEP': 100,
    'TRAIN_RANK_NORM': True,
    'TRAIN_CORR_REDUCTION': True,
    'TRAIN_CORR_RANK_NORM': True,
}

boosting = SimDeepBoosting(
        normalization=default_normalisation
    )
```

However, it is possible to use other normalisation using external python classes that have `fit` and `fit_transform` methods.


```python
from sklearn.preprocessing import RobustScaler

custom_norm =  {
    'CUSTOM': RobustScaler,
}

boosting = SimDeepBoosting(
        normalization=custom_norm
    )

    ```

Finally, more alternative normalisations are proposed in the config file.


### Number of iterations and seed
TO COMPLETE

## Save / load models

Two mechanisms exist to save and load dataset.
First the models can be entirely saved and loaded using `dill` (pickle like) libraries.

```python
from simdeep.simdeep_utils import save_model
from simdeep.simdeep_utils import load_model

# Save previous boosting model
save_model(boosting, "./test_saved_model")

# Delete previous model
del boosting

# Load model
boosting = load_model("TestProject", "./test_saved_model")
boosting.predict_labels_on_full_dataset()

```

See an example of saving/loading model in the example file: `load_and_save_models.py`

However, this mechanism presents a huge drawback since the models saved can be very large (all the hyperparameters/matrices... etc... are saved). Also, the equivalent dependencies and DL libraries need to be installed in both the machine computing the models and the machine used to load them which can lead to various errors.

A second solution is to save only the labels inferred for each submodel instance. These label files can then be loaded into a new DeepProg instance that will be used as reference for building the classifier.

```python

# Fitting a model
boosting.fit()
# Saving individual labels
boosting.save_test_models_classes(
    path_results=PATH_PRECOMPUTED_LABELS # Where to save the labels
    )

boostingNew = SimDeepBoosting(
        survival_tsv=SURVIVAL_TSV, # Same reference training set for `boosting` model
        training_tsv=TRAINING_TSV, # Same reference training set for `boosting` model
        path_data=PATH_DATA,
        project_name=PROJECT_NAME,
        path_results=PATH_DATA,
        distribute=False, # Option to use ray cluster scheduler (True or False)
    )

boostingNew.fit_on_pretrained_label_file(
    labels_files_folder=PATH_PRECOMPUTED_LABELS,
    file_name_regex="*.tsv")

boostingNew.predict_labels_on_full_dataset()
```
