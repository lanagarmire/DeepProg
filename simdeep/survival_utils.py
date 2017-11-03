"""
"""
import re

from simdeep.config import PATH_DATA
from simdeep.config import SURVIVAL_FLAG
from simdeep.config import SEPARATOR
from simdeep.config import ENTREZ_TO_ENSG_FILE
from simdeep.config import USE_INPUT_TRANSPOSE
from simdeep.config import DEFAULTSEP
from simdeep.config import CLASSIFIER

import  numpy as np

from scipy.stats import rankdata

from sklearn.metrics import pairwise_distances

from collections import defaultdict
from coxph_from_r import coxph

from coxph_from_r import c_index

from scipy.stats import kruskal
from scipy.stats import ranksums


################ DEBUG ################
# supposed to be None for normal usage
MAX_FEATURE = None
#######################################


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

        return np.nan_to_num(np.matrix(X))

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

def load_survival_file(f_name, path_data=PATH_DATA, sep=DEFAULTSEP):
    """ """
    if f_name in SEPARATOR:
        sep = SEPARATOR[f_name]

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
        line = line.strip('\n').split(sep)
        ids  = line[patient_id].strip('"')
        ndays = line[surv_id].strip('"')
        isdead = line[event_id].strip('"')

        survival[ids] = (float(ndays), float(isdead))

    return survival

def load_data_from_tsv(use_transpose=USE_INPUT_TRANSPOSE, **kwargs):
    """
    """
    if use_transpose:
        return _load_data_from_tsv_transposee(**kwargs)
    else:
        return _load_data_from_tsv(**kwargs)

def _load_data_from_tsv(
        f_name,
        key,
        path_data=PATH_DATA,
        f_type=float,
        sep=DEFAULTSEP,
        nan_to_num=True):
    """ """
    f_short = key

    if f_name in SEPARATOR:
        sep = SEPARATOR[f_name]

    f_tsv = open(path_data + f_name)
    header = f_tsv.readline().strip(sep + '\n').split(sep)

    feature_ids = ['{0}_{1}'.format(f_short, head)
                   for head in header][:MAX_FEATURE]
    sample_ids = []
    f_matrix = []

    for line in f_tsv:
        line = line.strip(sep + '\n').split(sep)
        sample_ids.append(line[0])

        if nan_to_num:
            line[1:] = [0 if (l.isalpha() or not l) else l
                        for l in line[1:MAX_FEATURE]]

        f_matrix.append(map(f_type, line[1:MAX_FEATURE]))

    f_matrix = np.array(f_matrix)

    if f_matrix.shape[1] == len(feature_ids) - 1:
        feature_ids = feature_ids[1:]

    assert(f_matrix.shape[1] == len(feature_ids))
    assert(f_matrix.shape[0] == len(sample_ids))

    return sample_ids, feature_ids, f_matrix

def _format_sample_name(sample_ids):
    """
    """
    regex = re.compile('_1_[A-Z][A-Z]')

    sample_ids = [regex.sub('', sample.strip('"')) for sample in sample_ids]
    return sample_ids

def _load_data_from_tsv_transposee(
        f_name,
        key,
        path_data=PATH_DATA,
        f_type=float,
        sep=DEFAULTSEP,
        nan_to_num=True):
    """ """
    if f_name in SEPARATOR:
        sep = SEPARATOR[f_name]

    f_tsv = open(path_data + f_name)
    header = f_tsv.readline().strip(sep + '\n').split(sep)

    sample_ids = header[1:]

    sample_ids = _format_sample_name(sample_ids)

    feature_ids = []
    f_matrix = []

    if f_name.lower().count('entrez'):
        ensg_dict = load_entrezID_to_ensg()
        use_ensg = True
    else:
        use_ensg = False

    for line in f_tsv:
        line = line.strip(sep + '\n').split(sep)
        feature = line[0].strip('"')

        if nan_to_num:
            line[1:] = [0 if (l.isalpha() or not l) else l
                        for l in line[1:]]

        if use_ensg and feature in ensg_dict:
            features = ensg_dict[feature]
        else:
            features = [feature]

        for feature in features:
            feature_ids.append('{0}_{1}'.format(key, feature))
            f_matrix.append(map(f_type, line[1:]))


    f_matrix = np.array(f_matrix).T

    assert(f_matrix.shape[1] == len(feature_ids))
    assert(f_matrix.shape[0] == len(sample_ids))

    return sample_ids, feature_ids, f_matrix

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

    clf = CLASSIFIER(**params)

    return clf, params

def load_entrezID_to_ensg():
    """
    """
    entrez_dict = {}

    for line in open(ENTREZ_TO_ENSG_FILE):
        line = line.split()
        entrez_dict[line[0]] = line[1:]

    return entrez_dict

def _process_parallel_coxph(inp):
    """
    """
    node_id, activity, isdead, nbdays = inp
    pvalue = coxph(activity, isdead, nbdays)

    return node_id, pvalue

def _process_parallel_cindex(inp):
    """
    """
    (node_id,
     act_ref, isdead_ref, nbdays_ref,
     act_test, isdead_test, nbdays_test) = inp

    score = c_index(act_ref, isdead_ref, nbdays_ref,
                    act_test, isdead_test, nbdays_test)

    return node_id, score

def _process_parallel_feature_importance(inp):
    """
    """
    arrays = defaultdict(list)
    feature, array, labels = inp

    for label, value in zip(labels, np.array(array).reshape(-1)):
        arrays[label].append(value)
    try:
        score, pvalue = kruskal(*arrays.values())
    except Exception:
        return feature, 1.0

    return feature, pvalue

def _process_parallel_feature_importance_per_cluster(inp):
    """
    """
    arrays = defaultdict(list)
    results = []

    feature, array, labels = inp

    for label, value in zip(labels, np.array(array).reshape(-1)):
        arrays[label].append(value)

    for cluster in arrays:
        array = np.array(arrays[cluster])
        array_comp = np.array([a for comp in arrays for a in arrays[comp]
                      if comp != cluster])

        score, pvalue = ranksums(array, array_comp)

        if pvalue < 0.05 and np.median(array) > np.median(array_comp):
            results.append((cluster, feature, pvalue))

    return results
