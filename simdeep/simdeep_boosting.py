from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData

from coxph_from_r import coxph
from coxph_from_r import c_index

from sklearn.model_selection import KFold

from multiprocessing.pool import Pool

from collections import Counter
from collections import defaultdict

import numpy as np

from scipy.stats import gmean

from simdeep.config import PROJECT_NAME
from simdeep.config import PATH_RESULTS
from simdeep.config import NB_THREADS
from simdeep.config import NB_ITER
from simdeep.config import NB_FOLDS
from simdeep.config import CLASS_SELECTION
from simdeep.config import PATH_MODEL

from os.path import isdir

import cPickle

from time import time


################# Variable ################
MODEL_THRES = 0.05
###########################################


def main():
    """ """
    boosting = SimDeepBoosting()
    boosting.load_training_dataset()
    boosting.fit()

    boosting.load_test_dataset()

    boosting.predict_labels_on_test_dataset()
    # boosting.predict_labels_on_full_dataset()

    boosting.collect_pvalue_on_training_dataset()
    boosting.collect_pvalue_on_test_fold()
    boosting.collect_number_of_features_per_omic()
    boosting.collect_cindex_for_test_fold()
    boosting.collect_pvalue_on_test_dataset()
    boosting.collect_cindex_for_test_dataset()

    # boosting.compute_c_indexes_for_test_dataset()
    # boosting.compute_c_indexes_for_full_dataset()

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
                 project_name='{0}_boosting'.format(PROJECT_NAME),
                 path_results=PATH_RESULTS):
        """ """
        assert(class_selection in ['max', 'mean'])

        self.model_thres = model_thres
        self.class_selection = class_selection
        self.models = []
        self.verbose = verbose
        self.nb_threads = nb_threads
        self.do_KM_plot = do_KM_plot
        self.project_name = project_name
        self.path_results = path_results

        self.test_labels = None
        self.test_labels_proba = None
        self.cv_labels = None
        self.cv_labels_proba = None
        self.full_labels = None
        self.full_labels_proba = None

        for it in range(nb_it):
            split = KFold(n_splits=3, shuffle=True, random_state=np.random.randint(0, 1000))
            dataset = LoadData(cross_validation_instance=split, verbose=False)
            self.models.append(SimDeep(dataset=dataset,
                                       load_existing_models=False,
                                       verbose=False,
                                       _isboosting=True,
                                       do_KM_plot=False))

    def load_training_dataset(self):
        """ """
        print('load training sets...')
        pool = Pool(self.nb_threads)

        self.models = pool.map(_load_training_dataset_pool, self.models)

    def fit(self):
        """ """
        print('fit models...')
        pool = Pool(self.nb_threads)
        self.models = pool.map(_fit_model_pool,  self.models)

        self.models = [model for model in self.models if model != None]

        nb_models = len(self.models)

        print('{0} models fitted'.format(nb_models))

        assert(nb_models)

        if nb_models > 1:
            assert(len(set([model.train_pvalue for model in self.models])) > 1)

    def load_test_dataset(self):
        """
        """
        print('load test datasets for the fitted models...')
        pool = Pool(self.nb_threads)
        self.models = pool.map(_load_test_dataset, self.models)

    def load_cv_dataset(self):
        """
        """
        print('load test datasets for the fitted models...')
        pool = Pool(self.nb_threads)
        self.models = pool.map(_load_test_dataset, self.models)

    def predict_labels_on_test_dataset(self):
        """
        """
        print('predict labels on test datasets...')
        pool = Pool(self.nb_threads)
        self.models = pool.map(_predict_labels_on_test_dataset, self.models)

        test_labels_proba = np.asarray([model.test_labels_proba for model in self.models])

        if self.class_selection == 'max':
            res = _highest_proba(test_labels_proba)
        elif self.class_selection == 'mean':
            res = _mean_proba(test_labels_proba)

        self.test_labels, self.test_labels_proba = res

        print('#### report of assigned cluster:')
        for key, value in Counter(self.test_labels).items():
            print('class: {0}, number of samples :{1}'.format(key, value))

        nbdays, isdead = self.models[0].dataset.survival_test.T.tolist()
        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_boosting_test',
                                                        nbdays, isdead,
                                                        self.test_labels, self.test_labels_proba)

        self.models[0]._write_labels(
            self.models[0].dataset.sample_ids_test, self.test_labels, '{0}_test_labels'.format(
            self.project_name))

    def collect_pvalue_on_test_fold(self):
        """
        """
        print('predict labels on test datasets...')
        pool = Pool(self.nb_threads)
        res = pool.map(_predict_labels_on_test_fold, self.models)

        labels, pvalues, pvalues_proba = zip(*res)

        if self.verbose:
            print('geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                gmean(pvalues), gmean(pvalues_proba)))

        return labels, pvalues, pvalues_proba

    def collect_pvalue_on_training_dataset(self):
        """
        """
        print('predict labels on test datasets...')
        pvalues, pvalues_proba = [], []

        for model in self.models:
            pvalues.append(model.train_pvalue)
            pvalues_proba.append(model.train_pvalue_proba)

        if self.verbose:
            print('training geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                gmean(pvalues), gmean(pvalues_proba)))

        return pvalues, pvalues_proba

    def collect_pvalue_on_test_dataset(self):
        """
        """
        print('predict labels on test datasets...')
        res = []

        for model in self.models:
            res.append(model.predict_labels_on_test_dataset())

        labels, pvalues, pvalues_proba = zip(*res)

        if self.verbose:
            print('test geo mean pvalues: {0} geo mean pvalues probas: {1}'.format(
                gmean(pvalues), gmean(pvalues_proba)))

        return labels, pvalues, pvalues_proba

    def collect_number_of_features_per_omic(self):
        """
        """
        counter = defaultdict(list)

        for model in self.models:
            for key in model.valid_node_ids_array:
                counter[key].append(len(model.valid_node_ids_array[key]))

        if self.verbose:
            for key in counter:
                print('key:{0} mean: {1} std: {2}'.format(
                    key, np.mean(counter[key]), np.std(counter[key])))

        return counter

    def collect_cindex_for_test_fold(self):
        """
        """
        cindexes_list = []

        for model in self.models:
            model.predict_labels_on_test_fold()
            cindexes_list.append(model.compute_c_indexes_for_test_fold_dataset())

        cindexes, cindexes_proba, cindexes_act = zip(*cindexes_list)

        if self.verbose:
            print('C-index results for test fold: mean {0} std {1}'.format(
                np.mean(cindexes_list), np.std(cindexes_list)))

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

        return cindexes_list

    def predict_labels_on_full_dataset(self):
        """
        """
        print('predict labels on full datasets...')
        pool = Pool(self.nb_threads)

        self.models = pool.map(_predict_labels_on_full_dataset, self.models)

        test_labels_proba = np.asarray([model.full_labels_proba for model in self.models])

        if self.class_selection == 'max':
            res = _highest_proba(test_labels_proba)
        elif self.class_selection == 'mean':
            res = _mean_proba(test_labels_proba)

        self.full_labels, self.full_labels_proba = res

        print('#### report of assigned cluster:')
        for key, value in Counter(self.full_labels).items():
            print('class: {0}, number of samples :{1}'.format(key, value))

        nbdays, isdead = self.models[0].dataset.survival_full.T.tolist()
        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_boosting_full',
                                                        nbdays, isdead,
                                                        self.full_labels, self.full_labels_proba)

        self.models[0]._write_labels(
            self.models[0].dataset.sample_ids_full, self.full_labels, '{0}_full_labels'.format(
            self.project_name))

    def _compute_test_coxph(self, fname_base, nbdays, isdead, labels, labels_proba):
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
            do_KM_plot=False)

        if self.verbose:
            print('Cox-PH proba p-value (Log-Rank) for inferred labels: {0}'.format(pvalue_proba))

        return pvalue, pvalue_proba

    def compute_c_indexes_for_test_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.models[0].dataset.survival).T
        days_test, dead_test = np.asarray(self.models[0].dataset.survival_test).T

        cindex = c_index(self.labels, dead, days,
                         self.test_labels, dead_test, days_test)

        if self.verbose:
            print('c-index for boosting test dataset:{0}'.format(cindex))

        return cindex

    def compute_c_indexes_for_full_dataset(self):
        """
        return c-index using labels as predicat
        """
        days, dead = np.asarray(self.models[0].dataset.survival).T
        days_full, dead_full = np.asarray(self.models[0].self.dataset.survival_full).T

        cindex = c_index(self.labels, dead, days,
                         self.full_labels, dead_full, days_full)

        if self.verbose:
            print('c-index for boosting full dataset:{0}'.format(cindex))

        return cindex

def save_class(boosting):
    """ """
    assert(isdir(PATH_MODEL))

    t = time()

    with open('{0}/{1}.pickle'.format(PATH_MODEL, boosting.project_name), 'w') as f_pick:
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


def _highest_proba(proba):
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

def _mean_proba(proba):
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

def _predict_labels_on_test_dataset(model):
    """
    """
    model.predict_labels_on_test_dataset()
    return model

def _predict_labels_on_test_fold(model):
    """
    """
    return model.predict_labels_on_test_fold()

def _predict_labels_on_full_dataset(model):
    """
    """
    model.predict_labels_on_full_dataset()
    return model

def _load_test_dataset(model):
    """
    """
    model.load_test_dataset()
    return model

def _load_training_dataset_pool(model):
    """ """
    model.load_training_dataset()
    return model

def _fit_model_pool(model):
    """ """
    before = model.dataset.cross_validation_instance.random_state
    try:
        model.fit()
        assert(len(set(model.labels)) > 1)
        assert(model.train_pvalue < MODEL_THRES)

        after = model.dataset.cross_validation_instance.random_state
        assert(before == after)

    except Exception as e:
        print('model with random state:{1} didn\'t converge:{0}'.format(e, before))
        return None

    else:
        print('model with random state:{0} fitted'.format(before))
        return model


if __name__ == '__main__':
    main()
