from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData

from coxph_from_r import coxph

from sklearn.model_selection import KFold

from multiprocessing.pool import Pool

from collections import Counter

import numpy as np

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

        for it in range(nb_it):
            split = KFold(n_splits=3, shuffle=True, random_state=np.random.randint(0,1000))
            dataset = LoadData(cross_validation_instance=split, verbose=False)
            self.models.append(SimDeep(dataset=dataset, verbose=False))

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
        pvalue, pvalue_proba = self._compute_test_coxph('KM_plot_boosting_test', nbdays, isdead)

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
            do_KM_plot=False)

        if self.verbose:
            print('Cox-PH proba p-value (Log-Rank) for inferred labels: {0}'.format(pvalue_proba))

        return pvalue, pvalue_proba

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
