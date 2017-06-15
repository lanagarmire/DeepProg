"""
"""
from simdeep.config import PATH_DATA

import  numpy as np

from scipy.stats import rankdata

from sklearn.metrics import pairwise_distances


class MadScaler():
    def __init__(self):
        pass
    def fit_transform(self, X):
        """ """
        X = np.asarray(X)

        for i in xrange(len(X)):
            med = np.median(X[i])
            mad = np.median(np.abs(X[i] - med))
            X[i] = (X[i] - med) / mad

        return np.matrix(X)

class RankNorm():
    def __init__(self):
        pass
    def fit_transform(self, X):
        """ """
        X = np.asarray(X)
        shape = map(float, X.shape)

        for i in xrange(len(X)):
            X[i] = rankdata(X[i]) / shape[1]

        return np.matrix(X)


class CorrelationReducer():
    def __init__(self):
        self.dataset = None

    def fit(self, dataset):
        """ """
        self.dataset = dataset

    def transform(self, dataset):
        """ """
        return 1.0 - pairwise_distances(dataset,
                                        self.dataset,
                                        'correlation')
    def fit_transform(self, dataset):
        """ """
        self.fit(dataset)
        return self.transform(dataset)


def load_survival_file(f_name, path_data=PATH_DATA, sep='\t'):
    """ """
    survival = {}
    f_surv = open(path_data + f_name, 'r')
    first_line = f_surv.readline()
    first_split = first_line.split(sep)[2]

    if first_split[2].isdigit() or first_split[1].isdigit():
        print('warning! the first line of the survival' \
              'file might be not a header!:\nfirst line:{0}'.format(first_line))

    for line in f_surv:
        ids, ndays, isdead = line.split(sep)[:3]
        survival[ids] = (float(ndays), float(isdead))

    return survival

def load_data_from_tsv(f_name,
                       sep='\t',
                       path_data=PATH_DATA,
                       f_type=float,
                       nan_to_num=True):
    """ """
    f_short = f_name[:3]
    f_tsv = open(path_data + f_name)
    header = f_tsv.readline().strip(sep + '\n').split(sep)

    feature_ids = ['{0}_{1}'.format(f_short, head)
                   for head in header]
    sample_ids = []
    f_matrix = []

    for line in f_tsv:
        line = line.strip(sep + '\n').split(sep)
        sample_ids.append(line[0])

        if nan_to_num:
            line[1:] = [0 if (l.isalpha() or not l) else l
                        for l in line[1:]]

        f_matrix.append(map(f_type, line[1:]))

    return sample_ids, feature_ids, np.array(f_matrix)

def select_best_classif_params(clf):
    """
    select best classifier parameters based uniquely
    on test errors
    """
    arr = []

    for fold in range(clf.cv):
        arr.append(clf.cv_results_[
            'split{0}_test_score'.format(fold)])

    score = [ar.max() for ar in np.array(arr).T]
    index = score.index(max(score))

    params = clf.cv_results_['params'][index]

    clf = clf.estimator.set_params(**params)

    return clf, params
