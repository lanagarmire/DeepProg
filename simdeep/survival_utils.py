"""
"""
import re

from simdeep.config import PATH_DATA
from simdeep.config import SURVIVAL_FLAG
from simdeep.config import SEPARATOR

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

def load_survival_file(f_name, path_data=PATH_DATA, sep=','):
    """ """
    survival = {}
    f_surv = open(path_data + f_name, 'r')

    first_line = f_surv.readline().strip(' \n\r\t').split(sep)

    for field in SURVIVAL_FLAG.values():
        try:
            assert(field in first_line)
        except Exception as e:
            raise Exception("{0} not in {1}".format(
                field, first_line))

    patient_id = first_line.index(SURVIVAL_FLAG['patient_id'])
    surv_id = first_line.index(SURVIVAL_FLAG['survival'])
    event_id = first_line.index(SURVIVAL_FLAG['event'])

    for line in f_surv:
        line = line.split(sep)
        ids  = line[patient_id].strip('"')
        ndays = line[surv_id].strip('"')
        isdead = line[event_id].strip('"')

        survival[ids] = (float(ndays), float(isdead))

    return survival

def load_data_from_tsv(
        f_name,
        key,
        path_data=PATH_DATA,
        f_type=float,
        nan_to_num=True):
    """ """
    f_short = key
    sep = SEPARATOR[f_name]

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

def format_sample_name(sample_ids):
    """
    """
    regex = re.compile('_1_[A-Z][A-Z]')

    sample_ids = [regex.sub('', sample.strip('"')) for sample in sample_ids]
    return sample_ids

def load_data_from_tsv_transposee(
        f_name,
        key,
        path_data=PATH_DATA,
        f_type=float,
        nan_to_num=True):
    """ """
    sep = SEPARATOR[f_name]

    f_tsv = open(path_data + f_name)
    header = f_tsv.readline().strip(sep + '\n').split(sep)

    sample_ids = header[1:]

    sample_ids = format_sample_name(sample_ids)

    feature_ids = []
    f_matrix = []

    for line in f_tsv:
        line = line.strip(sep + '\n').split(sep)
        feature_ids.append('{0}_{1}'.format(key, line[0]))

        if nan_to_num:
            line[1:] = [0 if (l.isalpha() or not l) else l
                        for l in line[1:]]

        f_matrix.append(map(f_type, line[1:]))

    return sample_ids, feature_ids, np.array(f_matrix).T

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
