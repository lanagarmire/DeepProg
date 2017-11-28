from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData

from coxph_from_r import coxph
from coxph_from_r import c_index
from coxph_from_r import c_index_multiple

from sklearn.model_selection import KFold

from multiprocessing.pool import Pool

from collections import Counter
from collections import defaultdict
from itertools import combinations

import numpy as np

from scipy.stats import gmean
from sklearn.metrics import adjusted_rand_score

from simdeep.config import PROJECT_NAME
from simdeep.config import PATH_RESULTS
from simdeep.config import NB_THREADS
from simdeep.config import NB_ITER
from simdeep.config import NB_FOLDS
from simdeep.config import CLASS_SELECTION
from simdeep.config import PATH_MODEL
from simdeep.config import NB_CLUSTERS
from simdeep.config import NORMALIZATION
from simdeep.config import NB_EPOCH
from simdeep.config import NEW_DIM
from simdeep.config import NB_SELECTED_FEATURES
from simdeep.config import PVALUE_THRESHOLD
from simdeep.config import CLUSTER_METHOD
from simdeep.config import CLASSIFICATION_METHOD

import simplejson

from distutils.dir_util import mkpath

from os.path import isdir

import cPickle

import gc

from time import time

from numpy import hstack
from simdeep.survival_utils import _process_parallel_feature_importance_per_cluster


################# Variable ################
MODEL_THRES = 0.05
###########################################


def main():
    """ """
    boosting = SimDeepBoosting()
    boosting.fit()

    boosting.predict_labels_on_test_dataset()

    boosting.predict_labels_on_full_dataset()

    boosting.collect_pvalue_on_full_dataset()

    boosting.collect_pvalue_on_training_dataset()
    boosting.collect_pvalue_on_test_fold()

    boosting.collect_number_of_features_per_omic()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_pvalue_on_test_dataset()
    boosting.collect_cindex_for_test_dataset()

    boosting.compute_c_indexes_for_test_dataset()
    boosting.compute_c_indexes_multiple_for_test_dataset()
    boosting.compute_clusters_consistency_for_full_labels()
    boosting.compute_clusters_consistency_for_test_labels()


class SimDeepBoosting():
    """
    """
    def __init__(self,
                 nb_it=NB_ITER,
                 nb_fold=NB_FOLDS,
                 do_KM_plot=True,
                 nb_threads=NB_THREADS,
                 class_selection=CLASS_SELECTION,
                 model_thres=MODEL_THRES,
                 verbose=True,
                 seed=None,
                 project_name='{0}_boosting'.format(PROJECT_NAME),
                 split_n_fold=3,
                 path_results=PATH_RESULTS,
                 nb_clusters=NB_CLUSTERS,
                 nb_epoch=NB_EPOCH,
                 normalization=NORMALIZATION,
                 nb_selected_features=NB_SELECTED_FEATURES,
                 cluster_method=CLUSTER_METHOD,
                 pvalue_thres=PVALUE_THRESHOLD,
                 classification_method=CLASSIFICATION_METHOD,
                 new_dim=NEW_DIM,
                 **kwargs):
        """ """
        assert(class_selection in ['max', 'mean', 'weighted_mean'])

        self._instance_weights = None

        if class_selection == 'max':
            self.class_selection =  _highest_proba
        elif class_selection == 'mean':
            self.class_selection = _mean_proba
        elif class_selection == 'weighted_mean':
            self.class_selection = _weighted_mean

        self._class_selection = class_selection

        self.model_thres = model_thres
        self.models = []
        self.verbose = verbose
        self.nb_threads = nb_threads
        self.do_KM_plot = do_KM_plot
        self.project_name = project_name
        self._project_name = project_name
        self.path_results = '{0}/{1}'.format(path_results, project_name)

        if not isdir(self.path_results):
            try:
                mkpath(self.path_results)
            except Exception as e:
                print('cannot find or create the current result path: {0}'\
                      '\n consider changing it as option' \
                      .format(self.path_results))

        self.test_labels = None
        self.test_labels_proba = None
        self.cv_labels = None
        self.cv_labels_proba = None
        self.full_labels = None
        self.full_labels_dicts = None
        self.full_labels_proba = None
        self.survival_full = None
        self.sample_ids_full = None
        self.feature_scores_per_cluster = {}

        self.log = {}

        self.feature_train_array = None
        self.matrix_full_array = None

        ######## deepprob instance parameters ########
        self.nb_clusters = nb_clusters
        self.normalization = normalization
        self.nb_epoch = nb_epoch
        self.new_dim = new_dim
        self.nb_selected_features = nb_selected_features
        self.pvalue_thres = pvalue_thres
        self.cluster_method = cluster_method
        self.cindex_test_folds = []
        self.classification_method = classification_method
        ##############################################

        self.test_fname_key = ''

        parameters = {
            'nb_clusters': self.nb_clusters,
            'nb_epoch': self.nb_epoch,
            'nb_selected_features': self.nb_selected_features,
            'new_dim': self.new_dim,
            'pvalue_thres': self.pvalue_thres,
            'cluster_method': self.cluster_method,
            'path_results': self.path_results,
            'project_name': self.project_name,
            'classification_method': self.classification_method,
        }

        self.datasets = []
        self.seed = seed

        self.log['seed'] = seed
        self.log['parameters'] = parameters
        self.log['nb_it'] = nb_it
        self.log['normalization'] = normalization
        self.log['nb clusters'] = nb_clusters
        self.log['success'] = False

        if 'survival_tsv' in kwargs:
            self.log['survival_tsv'] = kwargs['survival_tsv']
        if 'path_data' in kwargs:
            self.log['path_data'] = kwargs['path_data']
        if 'training_tsv' in kwargs:
            self.log['training_tsv'] = kwargs['training_tsv']
        if 'fill_unkown_feature_with_0' in kwargs:
            self.log['fill_unkown_feature_with_0'] = kwargs['fill_unkown_feature_with_0']

        if self.seed:
            np.random.seed(seed)

        max_seed = 1000
        min_seed = 0

        if seed > max_seed:
            min_seed = seed - max_seed
            max_seed = seed

        random_states = np.random.randint(min_seed, max_seed, nb_it)

        self.split_n_fold = split_n_fold

        for it in range(nb_it):
            split = KFold(n_splits=split_n_fold,
                          shuffle=True, random_state=random_states[it])

            dataset = LoadData(cross_validation_instance=split,
                               verbose=False,
                               normalization=self.normalization,
                               _parameters=parameters,
                               **kwargs)

            self.datasets.append(dataset)

    def __del__(self):
        """
        """
        for model in self.models:
            del model

        gc.collect()

    def partial_fit(self, debug=False):
        """
        """
        self._fit(_partial_fit_model_pool, debug=debug)

    def fit(self, debug=False):
        """
        """
        self._fit(_fit_model_pool, debug=debug)

    def _fit(self, _fit_func, debug=False):
        """ """
        print('fit models...')
        pool = None
        start_time = time()

        if debug:
            map_func = map
            for dataset in self.datasets:
                dataset.verbose = True
        else:
            pool = Pool(self.nb_threads)
            map_func = pool.map

        try:
            self.models = map_func(_fit_func,  self.datasets)
            self.models = [model for model in self.models if model != None]

            nb_models = len(self.models)

            print('{0} models fitted'.format(nb_models))
            self.log['nb. models fitted'] = nb_models

            assert(nb_models)

            if nb_models > 1:
                assert(len(set([model.train_pvalue for model in self.models])) > 1)

        except Exception as e:
            self.log['failure'] = str(e)
            raise e

        else:
            self.log['success'] = True

        finally:
            if pool is not None:
                pool.close()
                pool.join()
            self.log['fitting time (s)'] = time() - start_time

            if self._class_selection == 'weighted_mean':
                self.collect_cindex_for_test_fold()

    def predict_labels_on_test_dataset(self):
        """
        """
        print('predict labels on test datasets...')
        test_labels_proba = np.asarray([model.test_labels_proba for model in self.models])

        res = self.class_selection(test_labels_proba, weights=self.cindex_test_folds)
        self.test_labels, self.test_labels_proba = res

        print('#### report of assigned cluster:')
        for key, value in Counter(self.test_labels).items():
            print('class: {0}, number of samples :{1}'.format(key, value))

        nbdays, isdead = self.models[0].dataset.survival_test.T.tolist()
        pvalue, pvalue_proba, pvalue_cat = self._compute_test_coxph(
            'KM_plot_boosting_test',
            nbdays, isdead,
            self.test_labels, self.test_labels_proba,
            self.project_name)

        self.log['pvalue test {0}'.format(self.test_fname_key)] = pvalue
        self.log['pvalue proba test {0}'.format(self.test_fname_key)] = pvalue_proba
        self.log['pvalue cat test {0}'.format(self.test_fname_key)] = pvalue_cat

        self.models[0]._write_labels(
            self.models[0].dataset.sample_ids_test,
            self.test_labels,
            '{0}_test_labels'.format(self.project_name),
            labels_proba=self.test_labels_proba.T[0],
            nbdays=nbdays, isdead=isdead)

        return pvalue, pvalue_proba

    def collect_pvalue_on_test_fold(self):
        """
        """
        print('predict labels on test fold datasets...')

        pvalues, pvalues_proba = zip(*[(model.cv_pvalue, model.cv_pvalue_proba)
                                       for model in self.models])

        pvalue_gmean, pvalue_proba_gmean = gmean(pvalues), gmean(pvalues_proba)

        if self.verbose:
            print('geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                pvalue_gmean, pvalue_proba_gmean))

        self.log['pvalue geo mean test fold'] = pvalue_gmean
        self.log['pvalue proba geo mean test fold'] = pvalue_proba_gmean

        return pvalues, pvalues_proba

    def collect_pvalue_on_training_dataset(self):
        """
        """
        print('predict labels on training datasets...')
        pvalues, pvalues_proba = [], []

        for model in self.models:
            pvalues.append(model.train_pvalue)
            pvalues_proba.append(model.train_pvalue_proba)

        pvalue_gmean, pvalue_proba_gmean = gmean(pvalues), gmean(pvalues_proba)

        if self.verbose:
            print('training geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                pvalue_gmean, pvalue_proba_gmean))

        self.log['pvalue geo mean train'] = pvalue_gmean
        self.log['pvalue proba geo mean train'] = pvalue_proba_gmean

        return pvalues, pvalues_proba

    def collect_pvalue_on_test_dataset(self):
        """
        """
        print('collect pvalues on test datasets...')

        pvalues, pvalues_proba = zip(*[(model.test_pvalue, model.test_pvalue_proba)
                                       for model in self.models])
        pvalue_gmean, pvalue_proba_gmean = gmean(pvalues), gmean(pvalues_proba)

        if self.verbose:
            print('test geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                pvalue_gmean, pvalue_proba_gmean))

        self.log['pvalue geo mean test {0}'.format(self.test_fname_key)] = pvalue_gmean
        self.log['pvalue proba geo mean test {0}'.format(
            self.test_fname_key)] = pvalue_proba_gmean

        return pvalues, pvalues_proba

    def collect_pvalue_on_full_dataset(self):
        """
        """
        print('collect pvalues on full datasets...')

        pvalues, pvalues_proba = zip(*[(model.full_pvalue, model.full_pvalue_proba)
                                       for model in self.models])
        pvalue_gmean, pvalue_proba_gmean = gmean(pvalues), gmean(pvalues_proba)

        if self.verbose:
            print('full geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                pvalue_gmean, pvalue_proba_gmean))

        self.log['pvalue geo mean full'] = pvalue_gmean
        self.log['pvalue proba geo mean full'] = pvalue_proba_gmean

        return pvalues, pvalues_proba

    def collect_number_of_features_per_omic(self):
        """
        """
        counter = defaultdict(list)
        self.log['number of features per omics'] = {}

        for model in self.models:
            for key in model.valid_node_ids_array:
                counter[key].append(len(model.valid_node_ids_array[key]))

        if self.verbose:
            for key in counter:
                print('key:{0} mean: {1} std: {2}'.format(
                    key, np.mean(counter[key]), np.std(counter[key])))

                self.log['number of features per omics'][key] = float(np.mean(counter[key]))

        return counter

    def collect_cindex_for_test_fold(self):
        """
        """
        self.cindex_test_folds = []

        for model in self.models:
            model.predict_labels_on_test_fold()
            self.cindex_test_folds.append(model.compute_c_indexes_for_test_fold_dataset())

        if self.verbose:
            print('C-index results for test fold: mean {0} std {1}'.format(
                np.mean(self.cindex_test_folds), np.std(self.cindex_test_folds)))

        self.log['c-indexes test fold (mean)'] = np.mean(self.cindex_test_folds)

        return self.cindex_test_folds

    def collect_cindex_for_full_dataset(self):
        """
        """
        cindexes_list = []

        for model in self.models:
            model.predict_labels_on_test_fold()
            cindexes_list.append(model.compute_c_indexes_for_full_dataset())

        if self.verbose:
            print('c-index results for full dataset: mean {0} std {1}'.format(
                np.mean(cindexes_list), np.std(cindexes_list)))

        self.log['c-indexes full (mean)'] = np.mean(cindexes_list)

        return cindexes_list

    def collect_cindex_for_training_dataset(self):
        """
        """
        cindexes_list = []

        for model in self.models:
            model.predict_labels_on_test_fold()
            cindexes_list.append(model.compute_c_indexes_for_training_dataset())

        if self.verbose:
            print('C-index results for training dataset: mean {0} std {1}'.format(
                np.mean(cindexes_list), np.std(cindexes_list)))

        self.log['c-indexes train (mean)'] = np.mean(cindexes_list)

        return cindexes_list

    def collect_cindex_for_test_dataset(self):
        """
        """
        cindexes_list = []

        for model in self.models:
            model.predict_labels_on_test_fold()
            cindexes_list.append(model.compute_c_indexes_for_test_dataset())

        if self.verbose:
            print('C-index results for test: mean {0} std {1}'.format(
                np.mean(cindexes_list), np.std(cindexes_list)))

        self.log['C-index test {0}'.format(self.test_fname_key)] = np.mean(cindexes_list)

        return cindexes_list

    def predict_labels_on_full_dataset(self):
        """
        """
        print('predict labels on full datasets...')

        self._get_probas_for_full_models()
        self._reorder_survival_full()

        print('#### report of assigned cluster:')
        for key, value in Counter(self.full_labels).items():
            print('class: {0}, number of samples :{1}'.format(key, value))

        nbdays, isdead = self.survival_full.T.tolist()
        pvalue, pvalue_proba, pvalue_cat = self._compute_test_coxph(
            'KM_plot_boosting_full',
            nbdays, isdead,
            self.full_labels, self.full_labels_proba,
            self._project_name)

        self.log['pvalue full'] = pvalue
        self.log['pvalue proba full'] = pvalue_proba
        self.log['pvalue cat full'] = pvalue_cat

        self.models[0]._write_labels(
            self.sample_ids_full,
            self.full_labels,
            '{0}_full_labels'.format(self._project_name),
            labels_proba=self.full_labels_proba.T[0],
            nbdays=nbdays, isdead=isdead)

        return pvalue, pvalue_proba

    def compute_clusters_consistency_for_full_labels(self):
        """
        """
        scores = []

        for model_1, model_2 in combinations(self.models, 2):
            full_labels_1_old = model_1.full_labels
            full_labels_2_old = model_2.full_labels

            full_ids_1 = model_1.dataset.sample_ids_full
            full_ids_2 = model_2.dataset.sample_ids_full

            full_labels_1 = _reorder_labels(full_labels_1_old, full_ids_1)
            full_labels_2 = _reorder_labels(full_labels_2_old, full_ids_2)

            scores.append(adjusted_rand_score(full_labels_1,
                                              full_labels_2))
        print('Adj. Rand scores for full label: mean: {0} std: {1}'.format(
            np.mean(scores), np.std(scores)))

        self.log['Adj. Rand scores'] = np.mean(scores)

        return scores

    def compute_clusters_consistency_for_test_labels(self):
        """
        """
        scores = []

        for model_1, model_2 in combinations(self.models, 2):
            scores.append(adjusted_rand_score(model_1.test_labels,
                                              model_2.test_labels))
        print('Adj. Rand scores for test label: mean: {0} std: {1}'.format(
            np.mean(scores), np.std(scores)))

        self.log['Adj. Rand scores test {0}'.format(self.test_fname_key)] = np.mean(scores)

        return scores

    def _reorder_survival_full(self):
        """
        """
        survival_old = self.models[0].dataset.survival_full
        sample_ids = self.models[0].dataset.sample_ids_full
        surv_dict = {sample: surv for sample, surv in zip(sample_ids, survival_old)}

        self.survival_full = np.asarray([np.asarray(surv_dict[sample])[0]
                                         for sample in self.sample_ids_full])

    def _reorder_matrix_full(self):
        """
        """
        sample_ids = self.models[0].dataset.sample_ids_full
        index_dict = {sample: ids for ids, sample in enumerate(sample_ids)}
        index = [index_dict[sample] for sample in self.sample_ids_full]

        self.feature_train_array = self.models[0].dataset.feature_train_array
        self.matrix_full_array = self.models[0].dataset.matrix_full_array

        for key in self.matrix_full_array:
            self.matrix_full_array[key] = self.matrix_full_array[key][index]

    def _get_probas_for_full_models(self):
        """
        """
        proba_dict = defaultdict(list)

        for model in self.models:
            for sample, proba in zip(model.dataset.sample_ids_full, model.full_labels_proba):
                proba_dict[sample].append([proba.tolist()])

        labels, probas = self.class_selection(hstack(proba_dict.values()),
                                              weights=self.cindex_test_folds)

        self.full_labels = labels
        self.full_labels_proba = probas
        self.sample_ids_full = proba_dict.keys()

    def _compute_test_coxph(self, fname_base, nbdays,
                            isdead, labels, labels_proba,
                            project_name):
        """ """
        pvalue = coxph(
            labels, isdead, nbdays,
            isfactor=False,
            do_KM_plot=self.do_KM_plot,
            png_path=self.path_results,
            fig_name='{0}_{1}'.format(project_name, fname_base))

        if self.verbose:
            print('Cox-PH p-value (Log-Rank) for inferred labels: {0}'.format(pvalue))

        pvalue_proba = coxph(
            labels_proba.T[0],
            isdead, nbdays,
            isfactor=False)

        if self.verbose:
            print('Cox-PH proba p-value (Log-Rank) for inferred labels: {0}'.format(pvalue_proba))

        labels_categorical = self._labels_proba_to_labels(labels_proba)

        pvalue_cat = coxph(
            labels_categorical, isdead, nbdays,
            isfactor=False,
            do_KM_plot=self.do_KM_plot,
            png_path=self.path_results,
            fig_name='{0}_proba_{1}'.format(project_name, fname_base))

        if self.verbose:
            print('Cox-PH categorical p-value (Log-Rank) for inferred labels: {0}'.format(
                pvalue_cat))

        return pvalue, pvalue_proba, pvalue_cat

    def _labels_proba_to_labels(self, labels_proba):
        """
        """
        probas = labels_proba.T[0]
        labels = np.zeros(len(probas))
        nb_clusters = labels_proba.shape[1]

        for cluster in range(nb_clusters):
            percentile = 100 * (1.0 - 1.0 / (cluster + 1.0))
            value = np.percentile(probas, percentile)
            labels[probas >= value] = nb_clusters - cluster

        return labels

    def compute_c_indexes_for_test_dataset(self):
        """
        return c-index using labels as predicat
        """
        days_full, dead_full = np.asarray(self.survival_full).T
        days_test, dead_test = np.asarray(self.models[0].dataset.survival_test).T

        cindex = c_index(self.full_labels, dead_full, days_full,
                         self.test_labels, dead_test, days_test)

        cindex_proba = c_index(self.full_labels_proba.T[0], dead_full, days_full,
                               self.test_labels_proba.T[0], dead_test, days_test)

        if self.verbose:
            print('c-index for boosting test dataset:{0}'.format(cindex))
            print('c-index proba for boosting test dataset:{0}'.format(cindex_proba))

        self.log['c-index test boosting {0}'.format(self.test_fname_key)] = cindex
        self.log['c-index proba test boosting {0}'.format(self.test_fname_key)] = cindex_proba

        return cindex

    def compute_c_indexes_for_full_dataset(self):
        """
        return c-index using labels as predicat
        """
        days_full, dead_full = np.asarray(self.survival_full).T

        cindex = c_index(self.full_labels, dead_full, days_full,
                         self.full_labels, dead_full, days_full)

        cindex_proba = c_index(self.full_labels_proba.T[0], dead_full, days_full,
                               self.full_labels_proba.T[0], dead_full, days_full)

        if self.verbose:
            print('c-index for boosting full dataset:{0}'.format(cindex))
            print('c-index proba for boosting full dataset:{0}'.format(cindex_proba))

        self.log['c-index full boosting {0}'.format(self.test_fname_key)] = cindex
        self.log['c-index proba full boosting {0}'.format(self.test_fname_key)] = cindex_proba

        return cindex

    def compute_c_indexes_multiple_for_test_dataset(self):
        """
        """
        matrix_array_train = self.models[0].dataset.matrix_ref_array
        matrix_array_test = self.models[0].dataset.matrix_test_array

        nbdays, isdead = self.models[0].dataset.survival.T.tolist()
        nbdays_test, isdead_test = self.models[0].dataset.survival_test.T.tolist()

        activities_train, activities_test = [], []

        for model in self.models:
            activities_train.append(model.predict_nodes_activities(matrix_array_train))
            activities_test.append(model.predict_nodes_activities(matrix_array_test))

        activities_train = hstack(activities_train)
        activities_test = hstack(activities_test)

        cindex = c_index_multiple(activities_train, isdead, nbdays,
                                   activities_test, isdead_test, nbdays_test)

        print('total number of survival features: {0}'.format(activities_train.shape[1]))
        print('cindex multiple for test set: {0}:'.format(cindex))

        self.log['c-index multiple test {0}'.format(self.test_fname_key)] = cindex
        self.log['Number of survival features {0}'.format(
            self.test_fname_key)] = activities_train.shape[1]

        return cindex

    def plot_predicted_labels_for_test_sets(self):
        """
        """
        # self.models[0].plot_predicted_labels_for_test_sets(
        #     self.test_labels, key=self.test_fname_key)
        self.models[0].plot_kernel_for_test_sets(
            self.test_labels, key=self.test_fname_key)

    def load_new_test_dataset(self, tsv_dict,
                              path_survival_file,
                              fname_key=None,
                              normalization=None,
                              debug=False):
        """
        """
        pool = None

        if debug:
            map_func = map
            for model in self.models:
                model.verbose = True
                model.dataset.verbose = True
        else:
            pool = Pool(self.nb_threads)
            map_func = pool.map

        self.test_fname_key = fname_key

        inputs = [(model, tsv_dict, path_survival_file, normalization)
                  for model in self.models]
        self.models = map_func(_predict_new_dataset, inputs)

        if fname_key:
            self.project_name = '{0}_{1}'.format(self._project_name, fname_key)

        if pool is not None:
            pool.close()
            pool.join()

    def compute_feature_scores_per_cluster(self):
        """
        """
        print('computing feature importance per cluster...')

        self._reorder_matrix_full()

        mapf = map

        for label in set(self.full_labels):
            self.feature_scores_per_cluster[label] = []

        def generator(labels, feature_list, matrix):
            for i in range(len(feature_list)):
                yield feature_list[i], matrix[i], labels

        for key in self.matrix_full_array:
            feature_list = self.feature_train_array[key][:]
            matrix = self.matrix_full_array[key][:]
            labels = self.full_labels[:]

            input_list = generator(labels, feature_list, matrix.T)

            features_scored = mapf(_process_parallel_feature_importance_per_cluster, input_list)
            features_scored = [feat for feat_list in features_scored for feat in feat_list]

            for label, feature, pvalue in features_scored:
                self.feature_scores_per_cluster[label].append((feature, pvalue))

            for label in self.feature_scores_per_cluster:
                self.feature_scores_per_cluster[label].sort(key=lambda x:x[1])

    def write_feature_score_per_cluster(self):
        """
        """
        with open('{0}/{1}_features_scores_per_clusters.tsv'.format(
            self.path_results, self._project_name), 'w') as f_file:

            f_file.write('cluster id;feature;p-value\n')

            for label in self.feature_scores_per_cluster:
                for feature, pvalue in self.feature_scores_per_cluster[label]:
                    f_file.write('{0};{1};{2}\n'.format(label, feature, pvalue))

            print('{0}/{1}_features_scores_per_clusters.tsv written'.format(
                self.path_results, self._project_name))

    def evalutate_cluster_performance(self):
        """
        """
        bic_scores = np.array([model.bic_score for model in self.models])

        if bic_scores[0] is not None:
            bic = np.nanmean(bic_scores)
            print('bic score: mean: {0} std :{1}'.format(bic_scores.mean(), bic_scores.std()
            ))
            self.log['bic'] = bic
        else:
            bic = np.nan

        silhouette_scores = np.array([model.silhouette_score for model in self.models])
        silhouette = silhouette_scores.mean()
        print('silhouette score: mean: {0} std :{1}'.format(silhouette,
                                                            silhouette_scores.std()
        ))
        self.log['silhouette'] = silhouette

        calinski_scores = np.array([model.calinski_score for model in self.models])
        calinski = calinski_scores.mean()
        print('calinski harabasz score: mean: {0} std :{1}'.format(calinski_scores.mean(),
                                                                   calinski_scores.std()
        ))
        self.log['calinski'] = calinski

        return bic, silhouette, calinski

    def write_logs(self):
        """
        """
        for key in self.log:
            if isinstance(self.log[key], np.float32):
                self.log[key] = float(self.log[key])

        with open('{0}/{1}.log.json'.format(self.path_results, self._project_name), 'w') as f:
            f.write(simplejson.dumps(self.log, indent=2))


def save_class(boosting):
    """ """
    assert(isdir(PATH_MODEL))

    t = time()

    with open('{0}/{1}.pickle'.format(PATH_MODEL, boosting._project_name), 'w') as f_pick:
        cPickle.dump(boosting, f_pick)

    print('model saved in %2.1f s' % (time() - t))

def load_class(project_name=PROJECT_NAME + '_boosting'):
    """ """
    assert(isdir(PATH_MODEL))

    t = time()

    with open('{0}/{1}.pickle'.format(PATH_MODEL, project_name), 'r') as f_pick:
        boosting = cPickle.load(f_pick)

    print('model loaded in %2.1f s' % (time() - t))

    return boosting


def _highest_proba(proba, **kwargs):
    """
    """
    res = []
    labels = []
    probas = []

    clusters = range(proba.shape[2])
    samples = range(proba.shape[1])

    for sample in samples:
        label = max([(cluster, proba.T[cluster][sample].max()) for cluster in clusters],
                    key=lambda x:x[1])
        res.append(label)

    for label, proba in res:
        if not label:
            probas.append([proba, 1.0 - proba])
        else:
            probas.append([1.0 - proba, proba])
        labels.append(label)

    return labels, np.asarray(probas)

def _mean_proba(proba, **kwargs):
    """
    """
    res = []
    labels = []
    probas = []

    clusters = range(proba.shape[2])
    samples = range(proba.shape[1])

    for sample in samples:
        label = max([(cluster, proba.T[cluster][sample].mean()) for cluster in clusters],
                    key=lambda x:x[1])
        res.append(label)

    for label, proba in res:
        if not label:
            probas.append([proba, 1.0 - proba])
        else:
            probas.append([1.0 - proba, proba])
        labels.append(label)

    return labels, np.asarray(probas)

def _weighted_mean(proba, weights):
    """
    """
    res = []
    labels = []
    probas = []
    weights = np.array(weights)
    weights[weights < 0.50] = 0.0
    weights = np.power(weights, 4)

    if weights.sum() == 0:
        weights[:] = 1.0

    clusters = range(proba.shape[2])
    samples = range(proba.shape[1])

    for sample in samples:
        label = max([(cluster, np.average(proba.T[cluster][sample], weights=weights))
                     for cluster in clusters],
                    key=lambda x:x[1])
        res.append(label)

    for label, proba in res:
        if not label:
            probas.append([proba, 1.0 - proba])
        else:
            probas.append([1.0 - proba, proba])
        labels.append(label)

    return labels, np.asarray(probas)

def _fit_model_pool(dataset):
    """ """
    parameters = dataset._parameters

    model = SimDeep(dataset=dataset,
                    load_existing_models=False,
                    verbose=dataset.verbose,
                    _isboosting=True,
                    seed=dataset.cross_validation_instance.random_state,
                    do_KM_plot=False,
                    **parameters)

    before = model.dataset.cross_validation_instance.random_state
    try:
        model.load_training_dataset()
        model.fit()

        if len(set(model.labels)) < 1:
            raise Exception('only one class!')

        if model.train_pvalue > MODEL_THRES:
            raise Exception('pvalue: {0} not significant!'.format(model.train_pvalue))

    except Exception as e:
        print('model with random state:{1} didn\'t converge:{0}'.format(e, before))
        return None

    else:
        print('model with random state:{0} fitted'.format(before))

    model.predict_labels_on_test_fold()
    model.predict_labels_on_full_dataset()

    model.load_test_dataset()
    print('test dataset loaded for model: {0}'.format(before))
    model.predict_labels_on_test_dataset()

    model.look_for_prediction_nodes()
    model.evalutate_cluster_performance()

    return model

def _partial_fit_model_pool(dataset):
    """ """
    parameters = dataset._parameters

    model = SimDeep(dataset=dataset,
                    load_existing_models=False,
                    verbose=dataset.verbose,
                    _isboosting=True,
                    seed=dataset.cross_validation_instance.random_state,
                    do_KM_plot=False,
                    **parameters)

    before = model.dataset.cross_validation_instance.random_state

    try:
        model.load_training_dataset()
        model.fit()

        if len(set(model.labels)) < 1:
            raise Exception('only one class!')

        if model.train_pvalue > MODEL_THRES:
            raise Exception('pvalue: {0} not significant!'.format(model.train_pvalue))

    except Exception as e:
        print('model with random state:{1} didn\'t converge:{0}'.format(str(e), before))
        return None

    else:
        print('model with random state:{0} fitted'.format(before))

    model.predict_labels_on_test_fold()
    model.predict_labels_on_full_dataset()
    model.evalutate_cluster_performance()

    return model

def _predict_new_dataset(inputs):
    """
    """
    model, tsv_dict, path_survival_file, normalization = inputs

    model.load_new_test_dataset(tsv_dict, path_survival_file,
                                        normalization=normalization)
    model.predict_labels_on_test_dataset()

    return model

def _reorder_labels(labels, sample_ids):
    """
    """
    sample_dict = {sample: id for id, sample in enumerate(sample_ids)}
    sample_ordered = set(sample_ids)

    index = [sample_dict[sample] for sample in sample_ordered]

    return labels[index]


if __name__ == '__main__':
    main()
