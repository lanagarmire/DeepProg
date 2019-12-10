# Tutorial: Advanced usage of DeepProg model

## Visualisation
Once a DeepProg model is fitted, it might be interessant to obtain different visualisations of the samples for the training or the test sets, based on new survival features inferred by the autoencoders.For that purpose, we developped two methods to project the samples into a 2D space that can be called once a `SimDeepBoosting` or a `simDeep` is fitted.

```python
boosting.plot_supervised_predicted_labels_for_test_sets()
```

This first method infers the OMIC matrics activities for the new features inferred by the autoencoders and project the samples into a 2D space created using the two first PCA components. The figure create a kernel density for each cluster and project the labels of the test set.

![kdplot 1](./img/stacked_TestProject_TEST_DATA_2_KM_plot_boosting_test_kde_2_cropped.png)

A second more sophisticated method uses the new features inferred by the autoencoders to infer new features by reconstructing a supervised network that targets the inferred label. The new set of features are then projected into a 2D space using PCA components. This second method might be present more efficient visualisations of the different clusters

![kdplot 2](./img/stacked_TestProject_TEST_DATA_2_KM_plot_boosting_test_kde_1_cropped.png)

Note that these visualisation are not very efficient in that example dataset, since we have only a limited number of samples (40). However, they might become more usefull for real datasets.
