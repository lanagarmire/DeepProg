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
from simdeep.config import CLASSIFIER_TYPE
from simdeep.config import CLASSIFIER
from simdeep.config import MIXTURE_PARAMS
from simdeep.config import PATH_RESULTS
from simdeep.config import PROJECT_NAME
from simdeep.config import CLASSIFICATION_METHOD

from simdeep.config import CLUSTER_EVAL_METHOD
from simdeep.config import CLUSTER_METHOD
from simdeep.config import NB_THREADS_COXPH
from simdeep.config import NB_SELECTED_FEATURES

from simdeep.survival_utils import _process_parallel_coxph
from simdeep.survival_utils import _process_parallel_feature_importance

from simdeep.survival_utils import select_best_classif_params

from coxph_from_r import coxph
from coxph_from_r import surv_mean

from collections import Counter

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

import numpy as np
from numpy import hstack

from collections import defaultdict

import warnings

from multiprocessing import Pool


################ VARIABLE ############################################
_CLASSIFICATION_METHOD_LIST = ['ALL_FEATURES', 'SURVIVAL_FEATURES']
######################################################################

def main():
    """
    """
    sim_deep = SimDeep()
    sim_deep.load_training_dataset()
    sim_deep.fit()

    sim_deep.load_test_dataset()
    sim_deep.predict_labels_on_test_dataset()
    sim_deep.predict_labels_on_test_fold()
    sim_deep.predict_labels_on_full_dataset()


class SimDeep(DeepBase):
    """ """
    def __init__(self,
                 nb_clusters=NB_CLUSTERS,
                 pvalue_thres=PVALUE_THRESHOLD,
                 cluster_method=CLUSTER_METHOD,
                 cluster_eval_method=CLUSTER_EVAL_METHOD,
                 classifier_type=CLASSIFIER_TYPE,
                 project_name=PROJECT_NAME,
                 path_results=PATH_RESULTS,
                 cluster_array=CLUSTER_ARRAY,
                 nb_selected_features=NB_SELECTED_FEATURES,
                 mixture_params=MIXTURE_PARAMS,
                 nb_threads_coxph=NB_THREADS_COXPH,
                 classification_method=CLASSIFICATION_METHOD,
                 do_KM_plot=True,
                 verbose=True,
                 _isboosting=False,
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
        self.cluster_array = cluster_array
        self.path_results = path_results
        self.mixture_params = mixture_params
        self.project_name = project_name
        self.do_KM_plot = do_KM_plot
        self.nb_threads_coxph = nb_threads_coxph
        self.classification_method = classification_method
        self.nb_selected_features = nb_selected_features

        self.train_pvalue = None
        self.classifier = None
        self.classifier_type = classifier_type
        self._isboosting = _isboosting

        self.valid_node_ids_array = {}
        self.activities_array = {}

        self.activities_train = None
        self.activities_test = None
        self.activities_cv = None

        self.test_labels = None
        self.test_labels_proba = None
        self.training_omic_list = []

        self.feature_scores = defaultdict(list)

        self._label_ordered_dict = {}

        self.cluster_method = cluster_method
        self.cluster_eval_method = cluster_eval_method
        self.verbose = verbose

        DeepBase.__init__(self, verbose=self.verbose, **kwargs)

    def fit(self):
        """
        main function
        construct an autoencoder, predict nodes linked with survival
        and do clustering
        """
        self.construct_autoencoders()
        self.look_for_survival_nodes()
        self.predict_labels()
        self.fit_classification_model()

    def predict_labels_on_test_fold(self):
        """
        """
        if not self.dataset.cross_validation_instance:
            return

        self.dataset.load_matrix_test_fold()

        nbdays, isdead = self.dataset.survival_cv.T.tolist()
        activities_array = []

        for key in self.training_omic_list:

            test_matrix = self.dataset.matrix_cv_array[key]
            encoder = self.encoder_array[key]
            valid_node_ids = self.valid_node_ids_array[key]

            activities_array.append(encoder.predict(test_matrix).T[valid_node_ids].T)

        self.activities_cv = hstack(activities_array)

        self._predict_test_labels(self.activities_cv, self.dataset.matrix_cv_array)

        if self.verbose:
            print('#### report of assigned cluster:')
            for key, value in Counter(self.test_labels).items():
                print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_test_fold', nbdays, isdead)

        self._write_labels(self.dataset.sample_ids_cv, self.test_labels, '{0}_test_fold_labels'.format(
            self.project_name))

        return self.test_labels, pvalue

    def predict_labels_on_full_dataset(self):
        """
        """
        self.dataset.load_matrix_full()

        nbdays, isdead = self.dataset.survival_full.T.tolist()
        activities_array = []

        for key in self.training_omic_list:

            test_matrix = self.dataset.matrix_array_full[key]
            encoder = self.encoder_array[key]
            valid_node_ids = self.valid_node_ids_array[key]

            activities_array.append(encoder.predict(test_matrix).T[valid_node_ids].T)

        self.activities_full = hstack(activities_array)

        self._predict_test_labels(self.activities_full, self.dataset.matrix_array_full)

        if self.verbose:
            print('#### report of assigned cluster for full dataset:')
            for key, value in Counter(self.test_labels).items():
                print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_full', nbdays, isdead)

        self._write_labels(self.dataset.sample_ids_cv, self.test_labels, '{0}_full_labels'.format(
            self.project_name))

        return self.test_labels, pvalue

    def predict_labels_on_test_dataset(self):
        """
        """
        nbdays, isdead = self.dataset.survival_test.T.tolist()

        if set(self.dataset.matrix_test_array.keys()) != set(self.training_omic_list):
            self.look_for_survival_nodes(self.dataset.matrix_test_array.keys())
            self.fit_classification_model()

        activities_array = []

        for key in self.training_omic_list:
            self.dataset.reorder_test_matrix(key)

            test_matrix = self.dataset.matrix_test_array[key]
            encoder = self.encoder_array[key]
            valid_node_ids = self.valid_node_ids_array[key]

            activities_array.append(encoder.predict(test_matrix).T[valid_node_ids].T)

        self.activities_test = hstack(activities_array)

        self._predict_test_labels(self.activities_test, self.dataset.matrix_test_array)

        if self.verbose:
            print('#### report of assigned cluster:')
            for key, value in Counter(self.test_labels).items():
                print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_test', nbdays, isdead)

        self._write_labels(self.dataset.sample_ids_test, self.test_labels, '{0}_test_labels'.format(
            self.project_name))

        return self.test_labels, pvalue

    def _compute_test_coxph(self, fname_base, nbdays, isdead):
        """ """
        pvalue = coxph(
            self.test_labels, isdead, nbdays,
            isfactor=False,
            do_KM_plot=self.do_KM_plot,
            png_path=self.path_results,
            fig_name='{0}_{1}'.format(self.project_name, fname_base))

        if self.verbose:
            print('Cox-PH p-value (Log-Rank) for inferred labels: {0}'.format(pvalue))

        pvalue_proba = coxph(
            self.test_labels_proba.T[0],
            isdead, nbdays,
            isfactor=False,
            do_KM_plot=False,
            png_path=self.path_results,
            fig_name='{0}_{1}_proba'.format(self.project_name, fname_base))

        if self.verbose:
            print('Cox-PH proba p-value (Log-Rank) for inferred labels: {0}'.format(pvalue_proba))

        return pvalue, pvalue_proba

    def compute_feature_scores(self):
        """
        """
        if self.feature_scores:
            return

        if not self._isboosting:
            pool = Pool(self.nb_threads_coxph)
            mapf = pool.map
        else:
            mapf = map

        def generator(labels, feature_list, matrix):
            for i in range(len(feature_list)):
                yield feature_list[i], matrix[i], labels

        for key in self.dataset.matrix_array_train:
            feature_list = self.dataset.feature_train_array[key][:]
            matrix = self.dataset.matrix_array_train[key][:]
            labels = self.labels[:]

            input_list = generator(labels, feature_list, matrix.T)

            features_scored = mapf(_process_parallel_feature_importance, input_list)
            features_scored.sort(key=lambda x:x[1])

            self.feature_scores[key] = features_scored

    def _return_train_matrix_for_classification(self):
        """
        """
        assert (self.classification_method in _CLASSIFICATION_METHOD_LIST)

        if self.verbose:
            print('classification method: {0}'.format(self.classification_method))

        if self.classification_method == 'SURVIVAL_FEATURES':
            assert(self.classifier_type != 'clustering')
            matrix = self.activities_train
        elif self.classification_method == 'ALL_FEATURES':
            matrix = self._reduce_and_stack_matrices(self.matrix_array_train)
        if self.verbose:
            print('number of features for the classifier: {0}'.format(matrix.shape[1]))

        return matrix

    def _reduce_and_stack_matrices(self, matrices):
        """
        """
        if not self.nb_selected_features:
            return hstack(matrices.values())
        else:
            self.compute_feature_scores()

            matrix = []

            for key in self.feature_scores:
                index = [self.dataset.feature_train_index[key][feature]
                         for feature, pvalue in
                         self.feature_scores[key][:self.nb_selected_features]]
                matrix.append(matrices[key].T[index].T)

            return hstack(matrix)

    def fit_classification_model(self):
        """ """
        train_matrix = self._return_train_matrix_for_classification()
        labels = self.labels

        if self.classifier_type == 'clustering':
            if self.verbose:
                print('clustering model defined as the classifier')

            self.classifier = self.clustering
            return

        if self.verbose:
            print('classification analysis...')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classifier_grid.fit(train_matrix, labels)

        self.classifier, params = select_best_classif_params(self.classifier_grid)

        cvs = cross_val_score(self.classifier, train_matrix, labels, cv=5)

        self.classifier.set_params(probability=True)

        self.classifier.fit(train_matrix, labels)

        if self.verbose:
            print('best params:', params)
            print('cross val score: {0}'.format(np.mean(cvs)))
            print('classification score:', self.classifier.score(train_matrix, labels))

    def predict_labels(self):
        """
        predict labels from training set
        using K-Means algorithm on the node activities,
        using only nodes linked to survival
        """
        if self.verbose:
            print('performing clustering on the omic model with the following key:{0}'.format(
                self.training_omic_list))

        if self.cluster_method == 'kmeans':
            self.clustering = KMeans(n_clusters=self.nb_clusters, n_init=100)

        elif self.cluster_method == 'mixture':
            self.clustering = GaussianMixture(
                n_components=self.nb_clusters,
                **self.mixture_params
            )

        if not self.activities_train.any():
            raise Exception('No components linked to survival!'\
                            ' cannot perform clustering')

        if self.cluster_array and len(self.cluster_array) > 1:
            self._predict_best_k_for_cluster()

        self.clustering.fit(self.activities_train)
        labels = self.clustering.predict(self.activities_train)

        labels = self._order_labels_according_to_survival(labels)
        self.labels = labels

        if self.verbose:
            print("clustering done, labels ordered according to survival:")
            for key, value in Counter(labels).items():
                print('cluster label: {0}\t number of samples:{1}'.format(key, value))
            print('\n')

        nbdays, isdead = self.dataset.survival.T.tolist()
        pvalue = coxph(self.labels, isdead, nbdays)

        pvalue = coxph(self.labels, isdead, nbdays,
                       isfactor=False,
                       do_KM_plot=self.do_KM_plot,
                       png_path=self.path_results,
                       fig_name='{0}_KM_plot_training_dataset'.format(self.project_name))

        self._write_labels(self.dataset.sample_ids, self.labels, '{0}_training_set_labels'.format(
            self.project_name))

        if self.verbose:
            print('Cox-PH p-value (Log-Rank) for the cluster labels: {0}'.format(pvalue))

        self.train_pvalue = pvalue

    def _write_labels(self, sample_ids, labels, fname):
        """ """
        f_file = open('{0}/{1}.tsv'.format(self.path_results, fname), 'w')

        for sample, label in zip(sample_ids, labels):
            f_file.write('{0}\t{1}\n'.format(sample, label))

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

            if self.verbose:
                print('number of components linked to survival found:{0} for key {1}'.format(
                    len(valid_node_ids), key))

            self.activities_array[key] = activities.T[valid_node_ids].T

        self.activities_train = hstack([self.activities_array[key]
                                        for key in keys])

    def _return_test_matrix_for_classification(self, activities, matrix_array):
        """
        """
        if self.classification_method == 'SURVIVAL_FEATURES':
            return activities
        elif self.classification_method == 'ALL_FEATURES':
            matrix = self._reduce_and_stack_matrices(matrix_array)
            assert(matrix.shape[1] == self.classifier.shape_fit_[1])
            return matrix

    def _predict_test_labels(self, activities, matrix_array):
        """ """
        matrix_test = self._return_test_matrix_for_classification(
            activities, matrix_array)

        self.test_labels = self.classifier.predict(matrix_test)
        self.test_labels_proba = self.classifier.predict_proba(matrix_test)

    def _predict_best_k_for_cluster(self):
        """ """
        criterion = None
        best_k = None

        for k_cluster in self.cluster_array:
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

            if self.verbose:
                print('obtained {2}: {0} for k = {1}'.format(score, k_cluster,
                                                             self.cluster_eval_method))

            if criterion == None or score < criterion:
                criterion, best_k = score, k_cluster

        if self.verbose:
            print('best k: {0}'.format(best_k))

        if self.cluster_method == 'mixture':
            self.clustering.set_params(n_components=best_k)
        else:
            self.clustering.set_params(n_clusters=best_k)

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

    def _look_for_survival_nodes(self, activities):
        """
        """
        if not self._isboosting:
            pool = Pool(self.nb_threads_coxph)
            mapf = pool.map
        else:
            mapf = map

        nbdays, isdead = self.dataset.survival.T.tolist()
        pvalue_list = []

        input_list = iter((node_id, activity, isdead, nbdays)
                           for node_id, activity in enumerate(activities.T))

        pvalue_list = mapf(_process_parallel_coxph, input_list)

        pvalue_list = filter(lambda x: not np.isnan(x[1]), pvalue_list)
        pvalue_list.sort(key=lambda x:x[1], reverse=True)

        valid_node_ids = [node_id for node_id, pvalue in pvalue_list
                               if pvalue < self.pvalue_thres]
        return valid_node_ids


if __name__ == "__main__":
    main()
