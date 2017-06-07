"""
SimDeep main class
"""

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

from simdeep.deepmodel_base import DeepBase

from simdeep.config import NB_CLUSTERS
from simdeep.config import CLUSTER_ARRAY
from simdeep.config import PVALUE_THRESHOLD
from simdeep.config import SELECT_FEATURES_METHOD
from simdeep.config import CLASSIFIER

from simdeep.config import MAD_SCALE
from simdeep.config import ROBUST_SCALE
from simdeep.config import MIN_MAX_SCALE
from simdeep.config import UNIT_NORM
from simdeep.config import RANK_SCALE

from simdeep.config import CLUSTER_EVAL_METHOD
from simdeep.config import CLUSTER_METHOD

from simdeep.survival_utils import select_best_classif_params

from coxph_from_r import coxph
from coxph_from_r import surv_mean

from collections import Counter

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

import numpy as np

from numpy import hstack


def main():
    """
    DEBUG function
    """
    sim_deep = SimDeep()
    sim_deep.load_training_dataset()
    sim_deep.fit()
    sim_deep.load_test_dataset()
    sim_deep.predict_labels_on_test_dataset_v2()
    # sim_deep.predict_labels_on_test_dataset()


class SimDeep(DeepBase):
    """ """
    def __init__(self,
                 nb_clusters=NB_CLUSTERS,
                 pvalue_thres=PVALUE_THRESHOLD,
                 cluster_method=CLUSTER_METHOD,
                 cluster_eval_method=CLUSTER_EVAL_METHOD,
                 **kwargs):
        """
        ### AUTOENCODER PARAMETERS ###:
            dataset=None      ExtractData instance (load the dataset),
            f_matrix_train=None
            f_matrix_test=None
            f_matrix_train_out=None
            f_matrix_test_out=None

            level_dims = [500]
            level_dnn = [500]
            new_dim = 100
            dropout = 0.5
            act_reg = 0.0001
            w_reg = 0.001
            data_split = 0.2
            activation = 'tanh'
            nb_epoch = 10
            loss = 'binary_crossentropy'
            optimizer = 'sgd'
        """
        self.nb_clusters = nb_clusters
        self.pvalue_thres = pvalue_thres
        self.classifier_grid = CLASSIFIER
        self.classifier = None

        self.valid_node_ids_array = {}
        self.activities_array = {}

        self.activities_train = None
        self.activities_test = None

        self.test_labels = None
        self.test_labels_proba = None
        self.training_omic_list = []

        self._label_ordered_dict = {}

        self.cluster_method = cluster_method
        self.cluster_eval_method = cluster_eval_method

        DeepBase.__init__(self, **kwargs)

    def fit(self):
        """
        main function
        construct an autoencoder, predict nodes linked with survival
        and do clustering
        """
        self.construct_autoencoders()
        self.look_for_survival_nodes()
        labels = self.predict_labels()
        self.labels = labels

    def predict_labels_on_test_dataset(
            self,
            return_proba=False,
            mad_scale=MAD_SCALE,
            robust_scale=ROBUST_SCALE,
            min_max_scale=MIN_MAX_SCALE,
            unit_norm=UNIT_NORM,
            rank_scale=RANK_SCALE):
        """
        predict labels of the test set

        argument:
            :return_proba: Bool    if true return class probabilities instead
            :mad_scale: Bool (default False)    use the mad scaler normalisation
            robust_scale: Bool (default False)    use the robust scale transformation
            min_max_scale: Bool (default False)    use min max transformation
            unit_norm: Bool (default False)    use the unit norm normalization

        return:
            class labels (array), Cox-PH P-value
        """
        test_matrix = self.dataset.matrix_test
        ref_matrix = self.dataset.matrix_ref

        ref_matrix, test_matrix = self.dataset.transform_matrices(
            ref_matrix,
            test_matrix,
            mad_scale=mad_scale,
            robust_scale=robust_scale,
            min_max_scale=min_max_scale,
            unit_norm=unit_norm,
            rank_scale=rank_scale,
        )

        data_type = self.dataset.data_type_test
        nbdays, isdead = self.dataset.survival_test.T.tolist()

        print('feature selection...')
        embbed = SELECT_FEATURES_METHOD[data_type]

        train_matrix = embbed.fit_transform(ref_matrix, self.labels)
        test_matrix = embbed.transform(test_matrix)

        self._do_classification(train_matrix, test_matrix, self.labels)

        pvalue = coxph(self.test_labels, isdead, nbdays)

        print('Cox-PH p-value (Log-Rank) for inferred labels: {0}'.format(pvalue))

        labels = self.test_labels_proba if return_proba else self.test_labels

        return labels, pvalue

    def predict_labels_on_test_dataset_v2(self, predict_proba=False):
        """
        """
        # self.dataset.reorder_matrix_test()

        nbdays, isdead = self.dataset.survival_test.T.tolist()

        if self.dataset.test_tsv.keys() != self.training_omic_list:
            self.look_for_survival_nodes(self.dataset.test_tsv.keys())
            labels = self.predict_labels()

        activities_array = []

        for key in self.training_omic_list:
            self.dataset.reorder_test_matrix(key)

            test_matrix = self.dataset.matrix_test_array[key]
            encoder = self.encoder_array[key]
            valid_node_ids = self.valid_node_ids_array[key]

            activities_array.append(encoder.predict(test_matrix).T[valid_node_ids].T)

        activities_stacked = hstack(activities_array)

        test_labels = self.clustering.predict(activities_stacked)

        self.test_labels = [self._label_ordered_dict[label]
                            for label in test_labels]

        if predict_proba:
            self.test_labels_proba = self.clustering.predict_proba(activities_stacked)

        print('#### report of assigned cluster:')
        for key, value in Counter(self.test_labels).items():
            print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue = coxph(self.test_labels, isdead, nbdays)

        print('Cox-PH p-value (Log-Rank) for inferred labels: {0}'.format(pvalue))

        labels = self.test_labels

        return labels, pvalue

    def _do_classification(self, train_matrix, test_matrix, labels):
        """ """
        print('classification analysis...')

        self.classifier_grid.fit(train_matrix, labels)
        self.classifier, params = select_best_classif_params(self.classifier_grid)
        print('best params:', params)

        cvs = cross_val_score(self.classifier, train_matrix, labels, cv=5)
        print('cross val score: {0}'.format(np.mean(cvs)))
        self.classifier.set_params(probability=True)
        self.classifier.fit(train_matrix, labels)

        print('classification score:', self.classifier.score(train_matrix, labels))

        self.test_labels = self.classifier.predict(test_matrix)
        self.test_labels_proba = self.classifier.predict_proba(test_matrix)

        print('#### report of assigned cluster:')
        for key, value in Counter(self.test_labels).items():
            print('class: {0}, number of samples :{1}'.format(key, value))

    def _predict_best_k_for_cluster(self):
        """ """
        criterion = None
        best_k = None

        for k_cluster in CLUSTER_ARRAY:
            if self.cluster_method == 'mixture':
                self.clustering.set_params(n_components=k_cluster)
            else:
                self.clustering.set_params(n_clusters=k_cluster)

            self.clustering.fit(self.activities_train)

            if self.cluster_eval_method == 'bic':
                score = self.clustering.bic(self.activities_train)
            elif self.cluster_eval_method:
                score = calinski_harabaz_score(
                    self.activities_train,
                    self.clustering.predict(self.activities_train)
                )
            elif self.cluster_eval_method:
                score = silhouette_score(
                    self.activities_train,
                    self.clustering.predict(self.activities_train)
                )

            print('obtained {2}: {0} for k = {1}'.format(score, k_cluster,
                                                         self.cluster_eval_method))

            if criterion == None or score < criterion:
                criterion, best_k = score, k_cluster

        print('best k: {0}'.format(best_k))

        if self.cluster_method == 'mixture':
            self.clustering.set_params(n_components=best_k)
        else:
            self.clustering.set_params(n_clusters=best_k)

    def predict_labels(self):
        """
        predict labels from training set
        using K-Means algorithm on the node activities,
        using only nodes linked to survival
        """
        print('performing clustering on the omic model with the following key:{0}'.format(
            self.training_omic_list))

        if self.cluster_method == 'kmeans':
            self.clustering = KMeans(n_clusters=self.nb_clusters, n_init=100)

        elif self.cluster_method == 'mixture':
            self.clustering = GaussianMixture(
                n_components=self.nb_clusters,
                covariance_type='spherical',
                max_iter=10000,
                n_init=100)

        if not self.activities_train.any():
            raise Exception('No components linked to survival!'\
                            ' cannot perform clustering')
        self._predict_best_k_for_cluster()

        self.clustering.fit(self.activities_train)
        labels = self.clustering.predict(self.activities_train)

        labels = self._order_labels_according_to_survival(labels)

        print("clustering done, labels ordered according to survival:")
        for key, value in Counter(labels).items():
            print('cluster label: {0}\t number of samples:{1}'.format(key, value))

        print('\n')

        nbdays, isdead = self.dataset.survival.T.tolist()
        pvalue = coxph(labels, isdead, nbdays)
        print('Cox-PH p-value (Log-Rank) for the cluster labels: {0}'.format(pvalue))

        return labels

    def _order_labels_according_to_survival(self, labels):
        """
        Order cluster labels according to survival
        """
        labels_old = labels.copy()

        days, dead = np.asarray(self.dataset.survival).T

        self._label_ordered_dict = {}

        for label in set(labels_old):
            mean = surv_mean(dead[labels_old == label],
                             days[labels_old == label])
            self._label_ordered_dict[label] = mean

        label_ordered = [label for label, mean in
                         sorted(self._label_ordered_dict.items(), key=lambda x:x[1])]

        self._label_ordered_dict = {old_label: new_label
                      for new_label, old_label in enumerate(label_ordered)}

        for old_label in self._label_ordered_dict:
            labels[labels_old == old_label] = self._label_ordered_dict[old_label]

        return labels

    def look_for_survival_nodes(self, keys=None):
        """
        detect nodes from the autoencoder significantly
        linked with survival through coxph regression
        """
        self.training_omic_list = []

        if not keys:
            keys = self.encoder_array.keys()

        for key in keys:
            self.training_omic_list.append(key)

            encoder = self.encoder_array[key]
            matrix_train = self.matrix_array_train[key]

            activities = encoder.predict(matrix_train)

            valid_node_ids = self._look_for_survival_nodes(activities)
            self.valid_node_ids_array[key] = valid_node_ids

            print('number of components linked to survival found:{0} for key {1}'.format(
                len(valid_node_ids), key))

            self.activities_array[key] = activities.T[valid_node_ids].T

        self.activities_train = hstack([self.activities_array[key]
                                        for key in self.activities_array])

    def _look_for_survival_nodes(self, activities):
        """
        """
        nbdays, isdead = self.dataset.survival.T.tolist()
        pvalue_list = []

        for node_id, activity in enumerate(activities.T):
            activity = activity.tolist()
            pvalue = coxph(activity, isdead, nbdays)
            pvalue_list.append((node_id, pvalue))

        pvalue_list = filter(lambda x: not np.isnan(x[1]), pvalue_list)
        pvalue_list.sort(key=lambda x:x[1], reverse=True)

        valid_node_ids = [node_id for node_id, pvalue in pvalue_list
                               if pvalue < self.pvalue_thres]
        return valid_node_ids


if __name__ == "__main__":
    main()
