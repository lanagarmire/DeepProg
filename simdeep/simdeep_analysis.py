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
from simdeep.config import CINDEX_THRESHOLD
from simdeep.config import CLASSIFIER_TYPE

from simdeep.config import MIXTURE_PARAMS
from simdeep.config import PATH_RESULTS
from simdeep.config import PROJECT_NAME
from simdeep.config import CLASSIFICATION_METHOD

from simdeep.config import CLUSTER_EVAL_METHOD
from simdeep.config import CLUSTER_METHOD
from simdeep.config import NB_THREADS_COXPH
from simdeep.config import NB_SELECTED_FEATURES
from simdeep.config import SAVE_FITTED_MODELS
from simdeep.config import LOAD_EXISTING_MODELS
from simdeep.config import NODES_SELECTION
from simdeep.config import CLASSIFIER
from simdeep.config import HYPER_PARAMETERS
from simdeep.config import PATH_TO_SAVE_MODEL

from simdeep.survival_utils import _process_parallel_coxph
from simdeep.survival_utils import _process_parallel_cindex
from simdeep.survival_utils import _process_parallel_feature_importance
from simdeep.survival_utils import _process_parallel_feature_importance_per_cluster

from simdeep.survival_utils import select_best_classif_params

from simdeep.coxph_from_r import coxph
from simdeep.coxph_from_r import c_index
from simdeep.coxph_from_r import c_index_multiple

from simdeep.coxph_from_r import surv_median

from collections import Counter

from sklearn.metrics import silhouette_score

try:
    from sklearn.metrics import calinski_harabasz_score as calinski_harabaz_score
except Exception:
    from sklearn.metrics import calinski_harabaz_score

from sklearn.model_selection import GridSearchCV

import numpy as np
from numpy import hstack

from collections import defaultdict

import warnings

from multiprocessing import Pool


################ VARIABLE ############################################
_CLASSIFICATION_METHOD_LIST = ['ALL_FEATURES', 'SURVIVAL_FEATURES']
MODEL_THRES = 0.05
######################################################################


def main():
    """
    """
    from simdeep.config import TEST_TSV
    from simdeep.config import SURVIVAL_TSV_TEST

    sim_deep = SimDeep(seed=3)
    sim_deep.load_training_dataset()
    sim_deep.fit()

    sim_deep.predict_labels_on_test_fold()
    sim_deep.predict_labels_on_full_dataset()

    sim_deep.plot_cluster_labels()

    if SAVE_FITTED_MODELS:
        sim_deep.save_encoders()

    sim_deep.compute_c_indexes_for_test_fold_dataset()
    sim_deep.compute_c_indexes_for_full_dataset()

    sim_deep.look_for_prediction_nodes()

    sim_deep.compute_c_indexes_multiple_for_test_fold_dataset()

    sim_deep.load_new_test_dataset(TEST_TSV,
                                   SURVIVAL_TSV_TEST,
                                   # normalization={'TRAIN_NORM_SCALE': True},
                                   fname_key='dummy')
    sim_deep.compute_c_indexes_multiple_for_test_dataset()
    sim_deep.predict_labels_on_test_dataset()
    sim_deep.compute_c_indexes_for_test_dataset()
    sim_deep.plot_kernel_for_test_sets()


class SimDeep(DeepBase):
    """ """
    def __init__(self,
                 nb_clusters=NB_CLUSTERS,
                 pvalue_thres=PVALUE_THRESHOLD,
                 cindex_thres=CINDEX_THRESHOLD,
                 cluster_method=CLUSTER_METHOD,
                 cluster_eval_method=CLUSTER_EVAL_METHOD,
                 classifier_type=CLASSIFIER_TYPE,
                 project_name=PROJECT_NAME,
                 path_results=PATH_RESULTS,
                 cluster_array=CLUSTER_ARRAY,
                 nb_selected_features=NB_SELECTED_FEATURES,
                 mixture_params=MIXTURE_PARAMS,
                 node_selection=NODES_SELECTION,
                 nb_threads_coxph=NB_THREADS_COXPH,
                 classification_method=CLASSIFICATION_METHOD,
                 load_existing_models=LOAD_EXISTING_MODELS,
                 path_to_save_model=PATH_TO_SAVE_MODEL,
                 do_KM_plot=True,
                 verbose=True,
                 _isboosting=False,
                 dataset=None,
                 deep_model_additional_args={}):
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
            epochs = 10
            loss = 'binary_crossentropy'
            optimizer = 'sgd'
        """
        self.nb_clusters = nb_clusters
        self.pvalue_thres = pvalue_thres
        self.cindex_thres = cindex_thres

        self.classifier_grid = GridSearchCV(CLASSIFIER(), HYPER_PARAMETERS, cv=5)
        self.cluster_array = cluster_array
        self.path_results = path_results
        self.mixture_params = mixture_params

        self.project_name = project_name
        self._project_name = project_name
        self.do_KM_plot = do_KM_plot
        self.nb_threads_coxph = nb_threads_coxph
        self.classification_method = classification_method
        self.nb_selected_features = nb_selected_features
        self.node_selection = node_selection

        self.train_pvalue = None
        self.train_pvalue_proba = None
        self.full_pvalue = None
        self.full_pvalue_proba = None
        self.cv_pvalue = None
        self.cv_pvalue_proba = None
        self.test_pvalue = None
        self.test_pvalue_proba = None

        self.classifier = None
        self.classifier_test = None

        self.classifier_dict = {}

        self.encoder_for_kde_plot_dict = {}
        self._main_kernel = {}

        self.classifier_type = classifier_type

        self.used_normalization = None
        self.test_normalization = None

        self.used_features_for_classif = None

        self._isboosting = _isboosting
        self._is_fitted = False

        self.valid_node_ids_array = {}
        self.activities_array = {}
        self.activities_pred_array = {}
        self.pred_node_ids_array = {}

        self.activities_train = None
        self.activities_test = None
        self.activities_cv = None

        self.activities_for_pred_train = None
        self.activities_for_pred_test = None
        self.activities_for_pred_cv = None

        self.test_labels = None
        self.test_labels_proba = None
        self.cv_labels = None
        self.cv_labels_proba = None
        self.full_labels = None
        self.full_labels_proba = None

        self.training_omic_list = []
        self.test_omic_list = []

        self.feature_scores = defaultdict(list)
        self.feature_scores_per_cluster = {}

        self._label_ordered_dict = {}

        self.clustering_performance = None
        self.bic_score = None
        self.silhouette_score = None
        self.calinski_score = None

        self.cluster_method = cluster_method
        self.cluster_eval_method = cluster_eval_method
        self.verbose = verbose
        self._load_existing_models = load_existing_models
        self._features_scores_changed = False

        self.path_to_save_model = path_to_save_model

        deep_model_additional_args['path_to_save_model'] = self.path_to_save_model

        DeepBase.__init__(self,
                          verbose=self.verbose,
                          dataset=dataset,
                          **deep_model_additional_args)

    def _look_for_nodes(self, key):
        """
        """
        assert(self.node_selection in ['Cox-PH', 'C-index'])

        if self.node_selection == 'Cox-PH':
            return self._look_for_survival_nodes(key)

        if self.node_selection == 'C-index':
            return self._look_for_prediction_nodes(key)

    def load_new_test_dataset(self, tsv_dict, path_survival_file,
                              fname_key=None, normalization=None):
        """
        """
        self.dataset.load_new_test_dataset(tsv_dict,
                                           path_survival_file,
                                           normalization=normalization)

        if normalization is not None:
            self.test_normalization = {
                key: normalization[key]
                for key in normalization
                if normalization[key]}

        else:
            self.test_normalization = {
                key: self.dataset.normalization[key]
                for key in self.dataset.normalization
                if self.dataset.normalization[key]}

        if self.used_normalization != self.test_normalization:
            if self.verbose:
                print('recombuting feature scores...')

            self.feature_scores = {}
            self.compute_feature_scores(use_ref=True)
            self._features_scores_changed = True

        if fname_key:
            self.project_name = '{0}_{1}'.format(self._project_name, fname_key)

    def fit(self):
        """
        main function
        construct an autoencoder, predict nodes linked with survival
        and do clustering
        """
        if self._load_existing_models:
            self.load_encoders()

        if not self.is_model_loaded:
            self.construct_autoencoders()

        self.look_for_survival_nodes()

        self.training_omic_list = list(self.encoder_array.keys())
        self.predict_labels()

        self.used_normalization = {key: self.dataset.normalization[key]
                                   for key in self.dataset.normalization
                                   if self.dataset.normalization[key]}

        self.used_features_for_classif = self.dataset.feature_train_array
        self.fit_classification_model()

    def predict_labels_on_test_fold(self):
        """
        """
        if not self.dataset.cross_validation_instance:
            return

        self.dataset.load_matrix_test_fold()

        nbdays, isdead = self.dataset.survival_cv.T.tolist()
        self.activities_cv = self._predict_survival_nodes(self.dataset.matrix_cv_array)

        self.cv_labels, self.cv_labels_proba = self._predict_labels(
            self.activities_cv, self.dataset.matrix_cv_array)

        if self.verbose:
            print('#### report of test fold cluster:):')
            for key, value in Counter(self.cv_labels).items():
                print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_test_fold',
                                                        nbdays, isdead,
                                                        self.cv_labels, self.cv_labels_proba)
        self.cv_pvalue = pvalue
        self.cv_pvalue_proba = pvalue_proba

        self._write_labels(self.dataset.sample_ids_cv, self.cv_labels,
                           labels_proba=self.cv_labels_proba.T[0],
                           fname='{0}_test_fold_labels'.format(self.project_name))

        return self.cv_labels, pvalue, pvalue_proba

    def predict_labels_on_full_dataset(self):
        """
        """
        self.dataset.load_matrix_full()

        nbdays, isdead = self.dataset.survival_full.T.tolist()

        self.activities_full = self._predict_survival_nodes(self.dataset.matrix_full_array)

        self.full_labels, self.full_labels_proba = self._predict_labels(
            self.activities_full, self.dataset.matrix_full_array)

        if self.verbose:
            print('#### report of assigned cluster for full dataset:')
            for key, value in Counter(self.full_labels).items():
                print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_full',
                                                        nbdays, isdead,
                                                        self.full_labels, self.full_labels_proba)
        self.full_pvalue = pvalue
        self.full_pvalue_proba = pvalue_proba

        self._write_labels(self.dataset.sample_ids_full, self.full_labels,
                           labels_proba=self.full_labels_proba.T[0],
                           fname='{0}_full_labels'.format(self.project_name))

        return self.full_labels, pvalue, pvalue_proba

    def predict_labels_on_test_dataset(self):
        """
        """
        nbdays, isdead = self.dataset.survival_test.T.tolist()

        self.test_omic_list = list(self.dataset.matrix_test_array.keys())
        self.test_omic_list = list(set(self.test_omic_list).intersection(
            self.training_omic_list))

        try:
            assert(len(self.test_omic_list) > 0)
        except AssertionError:
            raise Exception('in predict_labels_on_test_dataset: test_omic_list is empty!'\
                            '\n either no common omic with trining_omic_list or error!')

        self.fit_classification_test_model()

        self.activities_test = self._predict_survival_nodes(self.dataset.matrix_test_array)
        self._predict_test_labels(self.activities_test, self.dataset.matrix_test_array)

        if self.verbose:
            print('#### report of assigned cluster:')
            for key, value in Counter(self.test_labels).items():
                print('class: {0}, number of samples :{1}'.format(key, value))

        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_test',
                                                        nbdays, isdead,
                                                        self.test_labels, self.test_labels_proba)
        self.test_pvalue = pvalue
        self.test_pvalue_proba = pvalue_proba

        self._write_labels(self.dataset.sample_ids_test, self.test_labels,
                           labels_proba=self.test_labels_proba.T[0],
                           fname='{0}_test_labels'.format(self.project_name))

        return self.test_labels, pvalue, pvalue_proba

    def _compute_test_coxph(self, fname_base,
                            nbdays, isdead,
                            labels, labels_proba):
        """ """
        pvalue = coxph(
            labels, isdead, nbdays,
            isfactor=False,
            do_KM_plot=self.do_KM_plot,
            png_path=self.path_results,
            fig_name='{0}_{1}'.format(self.project_name, fname_base))

        if self.verbose:
            print('Cox-PH p-value (Log-Rank) for inferred labels: {0}'.format(pvalue))

        pvalue_proba = coxph(
            labels_proba.T[0],
            isdead, nbdays,
            isfactor=False,
            do_KM_plot=False,
            png_path=self.path_results,
            fig_name='{0}_{1}_proba'.format(self.project_name, fname_base))

        if self.verbose:
            print('Cox-PH proba p-value (Log-Rank) for inferred labels: {0}'.format(pvalue_proba))

        return pvalue, pvalue_proba

    def compute_feature_scores(self, use_ref=False):
        """
        """
        if self.feature_scores:
            return

        pool = None

        if not self._isboosting:
            pool = Pool(self.nb_threads_coxph)
            mapf = pool.map
            mapf = map
        else:
            mapf = map

        def generator(labels, feature_list, matrix):
            for i in range(len(feature_list)):
                yield feature_list[i], matrix[i], labels

        if use_ref:
            key_array = list(self.dataset.matrix_ref_array.keys())
        else:
            key_array = list(self.dataset.matrix_train_array.keys())

        for key in key_array:
            if use_ref:
                feature_list = self.dataset.feature_ref_array[key][:]
                matrix = self.dataset.matrix_ref_array[key][:]
            else:
                feature_list = self.dataset.feature_train_array[key][:]
                matrix = self.dataset.matrix_train_array[key][:]

            labels = self.labels[:]

            input_list = generator(labels, feature_list, matrix.T)

            features_scored = list(mapf(_process_parallel_feature_importance, input_list))
            features_scored.sort(key=lambda x:x[1])

            self.feature_scores[key] = features_scored

        if pool is not None:
            pool.close()
            pool.join()

    def compute_feature_scores_per_cluster(self, use_ref=False):
        """
        """
        print('computing feature importance per cluster...')

        mapf = map

        for label in set(self.labels):
            self.feature_scores_per_cluster[label] = []

        def generator(labels, feature_list, matrix):
            for i in range(len(feature_list)):
                yield feature_list[i], matrix[i], labels

        if use_ref:
            key_array = list(self.dataset.matrix_ref_array.keys())
        else:
            key_array = list(self.dataset.matrix_train_array.keys())

        for key in key_array:
            if use_ref:
                feature_list = self.dataset.feature_ref_array[key][:]
                matrix = self.dataset.matrix_ref_array[key][:]
            else:
                feature_list = self.dataset.feature_train_array[key][:]
                matrix = self.dataset.matrix_train_array[key][:]

            labels = self.labels[:]

            input_list = generator(labels, feature_list, matrix.T)

            features_scored = mapf(_process_parallel_feature_importance_per_cluster, input_list)
            features_scored = [feat for feat_list in features_scored for feat in feat_list]

            for label, feature, median_diff, pvalue in features_scored:
                self.feature_scores_per_cluster[label].append((feature, median_diff, pvalue))

            for label in self.feature_scores_per_cluster:
                self.feature_scores_per_cluster[label].sort(key=lambda x:x[1])

    def write_feature_score_per_cluster(self):
        """
        """
        f_file_name = '{0}/{1}_features_scores_per_clusters.tsv'.format(
            self.path_results, self._project_name)
        f_anti_name = '{0}/{1}_features_anticorrelated_scores_per_clusters.tsv'.format(
            self.path_results, self._project_name)

        f_file = open(f_file_name, 'w')
        f_anti_file = open(f_anti_name, 'w')

        f_file.write('cluster id;feature;median diff;p-value\n')

        for label in self.feature_scores_per_cluster:
            for feature, median_diff, pvalue in self.feature_scores_per_cluster[label]:
                if median_diff > 0:
                    f_to_write = f_file
                else:
                    f_to_write = f_anti_file

                f_to_write.write('{0};{1};{2};{3}\n'.format(label, feature, median_diff, pvalue))

        print('{0} written'.format(f_file_name))
        print('{0} written'.format(f_anti_name))


    def write_feature_scores(self):
        """
        """
        with open('{0}/{1}_features_scores.tsv'.format(
            self.path_results, self.project_name), 'w') as f_file:

            for key in self.feature_scores:
                f_file.write('#### {0} ####\n'.format(key))

                for feature, score in self.feature_scores[key]:
                    f_file.write('{0};{1}\n'.format(feature, score))

            print('{0}/{1}_features_scores.tsv written'.format(
                self.path_results, self.project_name))

    def _return_train_matrix_for_classification(self):
        """
        """
        assert (self.classification_method in _CLASSIFICATION_METHOD_LIST)

        if self.verbose:
            print('classification method: {0}'.format(self.classification_method))

        if self.classification_method == 'SURVIVAL_FEATURES':
            assert(self.classifier_type != 'clustering')
            matrix = self._predict_survival_nodes(self.dataset.matrix_ref_array)
        elif self.classification_method == 'ALL_FEATURES':
            matrix = self._reduce_and_stack_matrices(self.dataset.matrix_ref_array)

        if self.verbose:
            print('number of features for the classifier: {0}'.format(matrix.shape[1]))

        return np.nan_to_num(matrix)

    def _reduce_and_stack_matrices(self, matrices):
        """
        """
        if not self.nb_selected_features:
            return hstack(matrices.values())
        else:
            self.compute_feature_scores()
            matrix = []

            for key in matrices:
                index = [self.dataset.feature_ref_index[key][feature]
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

        self.classifier.set_params(probability=True)
        self.classifier.fit(train_matrix, labels)

        self.classifier_dict[str(self.used_normalization)] = self.classifier

        if self.verbose:
            cvs = cross_val_score(self.classifier, train_matrix, labels, cv=5)
            print('best params:', params)
            print('cross val score: {0}'.format(np.mean(cvs)))
            print('classification score:', self.classifier.score(train_matrix, labels))

    def fit_classification_test_model(self):
        """ """
        is_same_features = self.used_features_for_classif == self.dataset.feature_ref_array
        is_same_normalization = self.used_normalization == self.test_normalization
        is_filled_with_zero = self.dataset.fill_unkown_feature_with_0

        if (is_same_features and is_same_normalization and is_filled_with_zero)\
           or self.classifier_type == 'clustering':
            if self.verbose:
                print('Not rebuilding the test classifier')

            if self.classifier_test is None:
                self.classifier_test = self.classifier
            return

        if self.verbose:
            print('classification for test set analysis...')

        self.used_normalization = self.dataset.normalization_test
        self.used_features_for_classif = self.dataset.feature_ref_array

        train_matrix = self._return_train_matrix_for_classification()
        labels = self.labels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classifier_grid.fit(train_matrix, labels)

        self.classifier_test, params = select_best_classif_params(self.classifier_grid)

        self.classifier_test.set_params(probability=True)
        self.classifier_test.fit(train_matrix, labels)

        if self.verbose:
            cvs = cross_val_score(self.classifier_test, train_matrix, labels, cv=5)
            print('best params:', params)
            print('cross val score: {0}'.format(np.mean(cvs)))
            print('classification score:', self.classifier_test.score(train_matrix, labels))

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

        if self.cluster_method == 'mixture':
            self.labels_proba = self.clustering.predict_proba(self.activities_train)
        else:
            self.labels_proba = np.array([self.labels, self.labels]).T

        if self.labels_proba.shape[1] < self.nb_clusters:
            missing_columns = self.nb_clusters - self.labels_proba.shape[1]

            for i in range(missing_columns):
                self.labels_proba = hstack([
                    self.labels_proba, np.zeros(
                        shape=(self.labels_proba, 1))])

        if self.verbose:
            print("clustering done, labels ordered according to survival:")
            for key, value in Counter(labels).items():
                print('cluster label: {0}\t number of samples:{1}'.format(key, value))
            print('\n')

        nbdays, isdead = self.dataset.survival.T.tolist()

        pvalue = coxph(self.labels, isdead, nbdays,
                       isfactor=False,
                       do_KM_plot=self.do_KM_plot,
                       png_path=self.path_results,
                       fig_name='{0}_KM_plot_training_dataset'.format(self.project_name))

        pvalue_proba = coxph(self.labels_proba.T[0], isdead, nbdays,
                             isfactor=False)

        self._write_labels(self.dataset.sample_ids, self.labels,
                           labels_proba=self.labels_proba.T[0],
                           fname='{0}_training_set_labels'.format(self.project_name))

        if self.verbose:
            print('Cox-PH p-value (Log-Rank) for the cluster labels: {0}'.format(pvalue))

        self.train_pvalue = pvalue
        self.train_pvalue_proba = pvalue_proba

    def evalutate_cluster_performance(self):
        """
        """

        if self.cluster_method == 'mixture':
            self.bic_score = self.clustering.bic(self.activities_train)

        self.silhouette_score = silhouette_score(self.activities_train, self.labels)
        self.calinski_score = calinski_harabaz_score(self.activities_train, self.labels)

        if self.verbose:
            print('silhouette score: {0}'.format(self.silhouette_score))
            print('calinski-harabaz score: {0}'.format(self.calinski_score))
            print('bic score: {0}'.format(self.bic_score))

    def _write_labels(self, sample_ids, labels, fname,
                      labels_proba=None, nbdays=None, isdead=None):
        """ """
        with open('{0}/{1}.tsv'.format(self.path_results, fname), 'w') as f_file:
            for ids, (sample, label) in enumerate(zip(sample_ids, labels)):
                suppl = ''

                if labels_proba is not None:
                    suppl += '\t{0}'.format(labels_proba[ids])
                if nbdays is not None:
                    suppl += '\t{0}'.format(nbdays[ids])
                if isdead is not None:
                    suppl += '\t{0}'.format(isdead[ids])

                f_file.write('{0}\t{1}{2}\n'.format(sample, label, suppl))

    def _predict_survival_nodes(self, matrix_array, keys=None):
        """
        """
        activities_array = {}

        if keys is None:
            keys = list(matrix_array.keys())

        for key in keys:
            matrix = matrix_array[key]

            if self.encoder_input_shape(key)[1] != matrix.shape[1]:
                if self.verbose:
                    print('matrix doesnt have the input dimension of the encoder'\
                          ' returning None')
                return None


            activities = self.encoder_predict(key, matrix)
            activities_array[key] = activities.T[self.valid_node_ids_array[key]].T

        return hstack([activities_array[key]
                       for key in keys])

    def look_for_survival_nodes(self, keys=None):
        """
        detect nodes from the autoencoder significantly
        linked with survival through coxph regression
        """
        if not keys:
            keys = list(self.encoder_array.keys())

        for key in keys:
            valid_node_ids = self._look_for_nodes(key)
            self.valid_node_ids_array[key] = valid_node_ids

            matrix_train = self.matrix_train_array[key]

            activities = self.encoder_predict(key, matrix_train)
            self.activities_array[key] = activities.T[valid_node_ids].T

        self.activities_train = hstack([self.activities_array[key]
                                        for key in keys])

    def look_for_prediction_nodes(self, keys=None):
        """
        detect nodes from the autoencoder that predict a
        high c-index scores using label from the retained test fold
        """
        if not keys:
            keys = list(self.encoder_array.keys())

        for key in keys:
            matrix_train = self.matrix_train_array[key]

            activities = self.encoder_predict(key, matrix_train)

            valid_node_ids = self._look_for_prediction_nodes(key)
            self.pred_node_ids_array[key] = valid_node_ids

            self.activities_pred_array[key] = activities.T[valid_node_ids].T

        self.activities_for_pred_train = hstack([self.activities_pred_array[key]
                                                 for key in keys])

    def compute_c_indexes_multiple_for_test_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.dataset.survival).T
        days_test, dead_test = np.asarray(self.dataset.survival_test).T

        activities_test = {}

        for key in self.dataset.matrix_test_array:
            node_ids = self.pred_node_ids_array[key]
            activities_test[key] = self.encoder_predict(
                key, self.dataset.matrix_test_array[key]).T[node_ids].T

        activities_test = hstack(activities_test.values())
        activities_train = hstack([self.activities_pred_array[key]
                                   for key in self.dataset.matrix_ref_array])

        cindex = c_index_multiple(activities_train, dead, days,
                                  activities_test, dead_test, days_test)

        if self.verbose:
            print('c-index multiple for test dataset:{0}'.format(cindex))

        return cindex

    def compute_c_indexes_multiple_for_test_fold_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.dataset.survival).T
        days_cv, dead_cv = np.asarray(self.dataset.survival_cv).T

        activities_cv = {}

        for key in self.dataset.matrix_cv_array:
            node_ids = self.pred_node_ids_array[key]
            activities_cv[key] = self.encoder_predict(
                key, self.dataset.matrix_cv_array[key]).T[node_ids].T

        activities_cv = hstack(activities_cv.values())
        cindex = c_index_multiple(self.activities_for_pred_train, dead, days,
                                  activities_cv, dead_cv, days_cv)

        if self.verbose:
            print('c-index multiple for test fold dataset:{0}'.format(cindex))

        return cindex

    def _return_test_matrix_for_classification(self, activities, matrix_array):
        """
        """
        if self.classification_method == 'SURVIVAL_FEATURES':
            return activities
        elif self.classification_method == 'ALL_FEATURES':
            matrix = self._reduce_and_stack_matrices(matrix_array)
            return matrix

    def _predict_test_labels(self, activities, matrix_array):
        """ """
        matrix_test = self._return_test_matrix_for_classification(
            activities, matrix_array)

        self.test_labels = self.classifier_test.predict(matrix_test)
        self.test_labels_proba = self.classifier_test.predict_proba(matrix_test)

        if self.test_labels_proba.shape[1] < self.nb_clusters:
            missing_columns = self.nb_clusters - self.test_labels_proba.shape[1]

            for i in range(missing_columns):
                self.test_labels_proba = hstack([
                    self.test_labels_proba, np.zeros(
                        shape=(self.test_labels_proba, 1))])

    def _predict_labels(self, activities, matrix_array):
        """ """
        matrix_test = self._return_test_matrix_for_classification(
            activities, matrix_array)

        labels = self.classifier.predict(matrix_test)
        labels_proba = self.classifier.predict_proba(matrix_test)

        if labels_proba.shape[1] < self.nb_clusters:
            missing_columns = self.nb_clusters - labels_proba.shape[1]

            for i in range(missing_columns):
                labels_proba = hstack([
                    labels_proba, np.zeros(
                        shape=(labels_proba.shape[0], 1))])

        return labels, labels_proba

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
            elif self.cluster_eval_method == 'calinski':
                score = calinski_harabaz_score(
                    self.activities_train,
                    self.clustering.predict(self.activities_train)
                )
            elif self.cluster_eval_method == 'silhouette':
                score = silhouette_score(
                    self.activities_train,
                    self.clustering.predict(self.activities_train)
                )

            if self.verbose:
                print('obtained {2}: {0} for k = {1}'.format(score, k_cluster,
                                                             self.cluster_eval_method))

            if criterion == None or score < criterion:
                criterion, best_k = score, k_cluster

                self.clustering_performance = criterion

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
            mean = surv_median(dead[labels_old == label],
                             days[labels_old == label])
            self._label_ordered_dict[label] = mean

        label_ordered = [label for label, _ in
                         sorted(self._label_ordered_dict.items(), key=lambda x:x[1])]

        self._label_ordered_dict = {old_label: new_label
                      for new_label, old_label in enumerate(label_ordered)}

        for old_label in self._label_ordered_dict:
            labels[labels_old == old_label] = self._label_ordered_dict[old_label]

        return labels

    def _look_for_survival_nodes(self, key=None, activities=None, survival=None):
        """
        """
        pool = None

        if not self._isboosting:
            pool = Pool(self.nb_threads_coxph)
            mapf = pool.map
        else:
            mapf = map

        if key is not None:
            matrix_train = self.matrix_train_array[key]
            activities = np.nan_to_num(self.encoder_predict(key, matrix_train))
        else:
            assert(activities is not None)

        if survival is not None:
            nbdays, isdead = survival.T.tolist()
        else:
            nbdays, isdead = self.dataset.survival.T.tolist()

        pvalue_list = []

        input_list = iter((node_id, activity, isdead, nbdays)
                           for node_id, activity in enumerate(activities.T))

        pvalue_list = mapf(_process_parallel_coxph, input_list)

        pvalue_list = list(filter(lambda x: not np.isnan(x[1]), pvalue_list))
        pvalue_list.sort(key=lambda x:x[1], reverse=True)

        valid_node_ids = [node_id for node_id, pvalue in pvalue_list
                               if pvalue < self.pvalue_thres]

        if self.verbose:
            print('number of components linked to survival found:{0} for key {1}'.format(
                len(valid_node_ids), key))

        if pool is not None:
            pool.close()
            pool.join()

        return valid_node_ids

    def _look_for_prediction_nodes(self, key):
        """
        """
        nbdays, isdead = self.dataset.survival.T.tolist()
        nbdays_cv, isdead_cv = self.dataset.survival_cv.T.tolist()

        matrix_train = self.matrix_train_array[key]
        matrix_cv = self.dataset.matrix_cv_array[key]

        activities_train = self.encoder_predict(key, matrix_train)
        activities_cv = self.encoder_predict(key, matrix_cv)

        input_list = iter((node_id,
                           activities_train.T[node_id], isdead, nbdays,
                           activities_cv.T[node_id], isdead_cv, nbdays_cv)
                           for node_id in range(activities_train.shape[1]))

        score_list = map(_process_parallel_cindex, input_list)

        score_list = filter(lambda x: not np.isnan(x[1]), score_list)
        score_list.sort(key=lambda x:x[1], reverse=True)

        valid_node_ids = [node_id for node_id, pvalue in score_list
                               if pvalue > self.cindex_thres]

        scores = [score for node_id, score in score_list
                  if score > self.cindex_thres]

        if self.verbose:
            print('number of components with a high prediction score:{0} for key {1}'\
                  ' \n\t mean: {2} std: {3}'.format(
                      len(valid_node_ids), key, np.mean(scores), np.std(scores)))

        return valid_node_ids

    def compute_c_indexes_for_full_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.dataset.survival).T
        days_full, dead_full = np.asarray(self.dataset.survival_full).T

        cindex = c_index(self.labels, dead, days,
                       self.full_labels, dead_full, days_full)

        if self.verbose:
            print('c-index for full dataset:{0}'.format(cindex))

        return cindex

    def compute_c_indexes_for_training_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.dataset.survival).T

        cindex = c_index(self.labels, dead, days,
                       self.labels, dead, days)

        if self.verbose:
            print('c-index for training dataset:{0}'.format(cindex))

        return cindex

    def compute_c_indexes_for_test_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.dataset.survival).T
        days_test, dead_test = np.asarray(self.dataset.survival_test).T

        cindex = c_index(self.labels, dead, days,
                         self.test_labels, dead_test, days_test)

        if self.verbose:
            print('c-index for test dataset:{0}'.format(cindex))

        return cindex

    def compute_c_indexes_for_test_fold_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.dataset.survival).T
        days_cv, dead_cv= np.asarray(self.dataset.survival_cv).T

        cindex =  c_index(self.labels, dead, days,
                          self.cv_labels, dead_cv, days_cv)

        if self.verbose:
            print('c-index for test fold dataset:{0}'.format(cindex))

        return cindex

    def predict_nodes_activities(self, matrix_array):
        """
        """
        activities = []

        for key in matrix_array:
            if key not in self.pred_node_ids_array:
                continue

            node_ids = self.pred_node_ids_array[key]

            activities.append(self.encoder_predict(key, matrix_array[key]).T[node_ids].T)

        return hstack(activities)

    def plot_kernel_for_test_sets(self,
                                  dataset=None,
                                  labels=None,
                                  labels_proba=None,
                                  test_labels=None,
                                  test_labels_proba=None,
                                  define_as_main_kernel=False,
                                  use_main_kernel=False,
                                  activities=None,
                                  activities_test=None,
                                  key=''):
        """
        """
        from simdeep.plot_utils import plot_kernel_plots

        if dataset is None:
            dataset = self.dataset

        if labels is None:
            labels = self.labels

        if labels_proba is None:
            labels_proba = self.labels_proba

        if test_labels_proba is None:
            test_labels_proba = self.test_labels_proba

        if test_labels is None:
            test_labels = self.test_labels

        if test_labels_proba is None:
            test_labels_proba = self.test_labels_proba

        test_norm = self.test_normalization
        train_norm = self.dataset.normalization
        train_norm = {key: train_norm[key] for key in train_norm if train_norm[key]}

        is_same_normalization = train_norm == test_norm
        is_filled_with_zero = self.dataset.fill_unkown_feature_with_0

        if activities is None or activities_test is None:
            if not (is_same_normalization and is_filled_with_zero):
                print('\n<><><><> Cannot plot survival KDE plot' \
                      ' Different normalisation used for test set <><><><>\n')
                return

            activities = hstack([self.activities_array[omic]
                                 for omic in self.test_omic_list])
            activities_test = self.activities_test

        if define_as_main_kernel:
            self._main_kernel = {'activities': activities_test.copy(),
                                 'labels': test_labels.copy()}

        if use_main_kernel:
            activities = self._main_kernel['activities']
            labels = self._main_kernel['labels']

        html_name = '{0}/{1}{2}_test_kdeplot.html'.format(
            self.path_results,
            self.project_name,
            key)

        plot_kernel_plots(
            test_labels=test_labels,
            test_labels_proba=test_labels_proba,
            labels=labels,
            activities=activities,
            activities_test=activities_test,
            dataset=self.dataset,
            path_html=html_name)

    def plot_supervised_kernel_for_test_sets(
            self,
            labels=None,
            labels_proba=None,
            dataset=None,
            key='',
            **kwargs):
        """
        """
        if labels is None:
            labels = self.labels

        if labels_proba is None:
            labels_proba = self.labels_proba

        if dataset is None:
            dataset = self.dataset

        activities, activities_test = self._predict_kde_matrix(
            labels_proba, dataset)

        key += '_supervised'

        self.plot_kernel_for_test_sets(labels=labels,
                                       labels_proba=labels_proba,
                                       dataset=dataset,
                                       activities=activities,
                                       activities_test=activities_test,
                                       key=key,
                                       **kwargs)

    def _create_autoencoder_for_kernel_plot(self, labels_proba, dataset):
        """
        """
        autoencoder = DeepBase(dataset=dataset,
                               seed=self.seed,
                               verbose=False,
                               dropout=0.1,
                               epochs=50)

        autoencoder.matrix_train_array = dataset.matrix_ref_array
        autoencoder.construct_supervized_network(labels_proba)

        key = str(self.test_normalization)

        self.encoder_for_kde_plot_dict[key] = autoencoder.encoder_array

    def _predict_kde_matrix(self, labels_proba, dataset):
        """
        """
        matrix_ref_list = []
        matrix_test_list = []

        encoder_key = str(self.test_normalization)

        if encoder_key not in self.encoder_for_kde_plot_dict or \
           not dataset.fill_unkown_feature_with_0:
            self._create_autoencoder_for_kernel_plot(labels_proba, dataset)

        encoder_array = self.encoder_for_kde_plot_dict[encoder_key]

        for key in encoder_array:
            matrix_ref = self.encoder_predict(key, dataset.matrix_ref_array[key])
            matrix_test = self.encoder_predict(key, dataset.matrix_test_array[key])

            survival_node_ids = self._look_for_survival_nodes(
                activities=matrix_ref, survival=dataset.survival)

            if len(survival_node_ids) > 1:
                matrix_ref = matrix_ref.T[survival_node_ids].T
                matrix_test = matrix_test.T[survival_node_ids].T
            else:
                print('not enough survival nodes to construct kernel for key: {0}' \
                      'skipping the {0} matrix'.format(key))
                continue

            matrix_ref_list.append(matrix_ref)
            matrix_test_list.append(matrix_test)

        if not matrix_ref_list:
            print('matrix_ref_list / matrix_test_list empty!' \
                  'take the last OMIC ({0}) matrix as ref'.format(key))
            matrix_ref_list.append(matrix_ref)
            matrix_test_list.append(matrix_test)

        return hstack(matrix_ref_list), hstack(matrix_test_list)


    def _get_probas_for_full_model(self):
        """
        return sample and proba
        """
        return list(zip(self.dataset.sample_ids_full, self.full_labels_proba))

    def _get_survival_full(self):
        """
        """
        return self.dataset.survival_full

    def _get_sample_ids_full(self):
        """
        """
        return self.dataset.sample_ids_full


    def _partial_fit_model_pool(self):
        """
        """
        try:
            self.load_training_dataset()
            self.fit()

            if len(set(self.labels)) < 1:
                raise Exception('only one class!')

            if self.train_pvalue > MODEL_THRES:
                raise Exception('pvalue: {0} not significant!'.format(self.train_pvalue))

        except Exception as e:
            print('model with random state:{1} didn\'t converge:{0}'.format(str(e), self.seed))
            return False

        else:
            print('model with random state:{0} fitted'.format(self.seed))
            self._is_fitted = True

        self.predict_labels_on_test_fold()
        self.predict_labels_on_full_dataset()
        self.evalutate_cluster_performance()

        return self._is_fitted



if __name__ == "__main__":
    main()
