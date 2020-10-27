"""
"""
import re

import pandas as pd

from simdeep.config import PATH_DATA
from simdeep.config import SURVIVAL_FLAG
from simdeep.config import SEPARATOR
from simdeep.config import ENTREZ_TO_ENSG_FILE
from simdeep.config import USE_INPUT_TRANSPOSE
from simdeep.config import DEFAULTSEP
from simdeep.config import CLASSIFIER

import  numpy as np

from scipy.stats import rankdata

from numpy import hstack

from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import RobustScaler

from collections import defaultdict

from simdeep.coxph_from_r import coxph
from simdeep.coxph_from_r import c_index

from scipy.stats import kruskal
from scipy.stats import ranksums

from os.path import isdir
from os import mkdir


################ DEBUG ################
# supposed to be None for normal usage
MAX_FEATURE = None
#######################################


class MadScaler():
    def __init__(self):
        """
        """
        pass
    def fit_transform(self, X):
        """ """
        X = np.asarray(X)

        for i in range(len(X)):
            med = np.median(X[i])
            mad = np.median(np.abs(X[i] - med))
            X[i] = (X[i] - med) / mad

        return np.nan_to_num(np.matrix(X))

class RankNorm():
    """
    """
    def __init__(self):
        """
        """
        pass

    def fit_transform(self, X):
        """ """
        X = np.asarray(X)
        shape = list(map(float, X.shape))

        for i in range(len(X)):
            X[i] = rankdata(X[i]) / shape[1]

        return np.matrix(X)

class SampleReducer():
    """
    """
    def __init__(self, perc_sample_to_keep=0.90):
        """
        """
        assert(isinstance(perc_sample_to_keep, float))
        assert(0.0 < perc_sample_to_keep < 1.0)
        self.perc_sample_to_keep = perc_sample_to_keep

    def sample_to_keep(self, datasets, index=None):
        """
        """
        nb_samples = len(datasets.values()[0][index])
        scores = np.zeros(nb_samples)
        threshold = int(nb_samples * self.perc_sample_to_keep)

        for key in datasets:
            scores_array = np.array([vector.sum() for vector in datasets[key][index]])
            scores = scores + scores_array

        scores = [(pos, score) for pos, score in enumerate(scores)]

        scores.sort(key=lambda x:x[1], reverse=True)
        to_keep = [pos for pos, score in scores[:threshold]]
        to_remove = [pos for pos, score in scores[threshold:]]

        return to_keep, to_remove

class VarianceReducer():
    """
    """
    def __init__(self, nb_features=200):
        """
        """
        self.nb_features = nb_features
        self.index_to_keep = []

    def fit(self, dataset):
        """
        """
        if self.nb_features > dataset.shape[1]:
            self.nb_features = dataset.shape[1]

        variances = [np.var(array) for array in dataset.T]
        threshold = sorted(enumerate(variances),
                           reverse=True,
                           key=lambda x:x[1],
        )
        self.index_to_keep = [pos for pos, var in threshold[:self.nb_features]]

    def transform(self, dataset):
        """
        """
        return dataset.T[self.index_to_keep].T

    def fit_transform(self, dataset):
        """
        """
        self.fit(dataset)
        return self.transform(dataset)

class CorrelationReducer():
    """
    """
    def __init__(self, distance='correlation', threshold=None):
        """
        """
        self.distance = distance
        self.dataset = None
        self.threshold = threshold

    def fit(self, dataset):
        """ """
        self.dataset = dataset

        if self.threshold:
            self.dataset[self.dataset < self.threshold] = 0

    def transform(self, dataset):
        """ """
        if self.threshold:
            dataset[dataset < self.threshold] = 0

        return 1.0 - pairwise_distances(dataset,
                                        self.dataset,
                                        self.distance)

    def fit_transform(self, dataset):
        """ """
        self.fit(dataset)
        return self.transform(dataset)


class RankCorrNorm():
    """
    """
    def __init__(self, dataset):
        """
        """
        self.dataset = dataset


def load_survival_file(f_name,
                       path_data=PATH_DATA,
                       sep=DEFAULTSEP,
                       survival_flag=SURVIVAL_FLAG):
    """ """
    if f_name in SEPARATOR:
        sep = SEPARATOR[f_name]

    survival = {}

    with open(path_data + f_name, 'r') as f_surv:
        first_line = f_surv.readline().strip(' \n\r\t').split(sep)
        for field in survival_flag.values():
            try:
                assert(field in first_line)
            except Exception:
                raise Exception("{0} not in {1}".format(
                    field, first_line))

        patient_id = first_line.index(survival_flag['patient_id'])
        surv_id = first_line.index(survival_flag['survival'])
        event_id = first_line.index(survival_flag['event'])

        for line in f_surv:
            line = line.strip('\n').split(sep)
            ids  = line[patient_id].strip('"')
            ndays = line[surv_id].strip('"')
            isdead = line[event_id].strip('"')

            survival[ids] = (float(ndays), float(isdead))

    return survival


def translate_index(original_ids, new_ids):
    """ """
    index1d = {ids: pos for pos, ids in enumerate(original_ids)}

    return np.asarray([index1d[sample] for sample in new_ids])


def return_intersection_indexes(ids_1, ids_2):
    """ """
    index1d = {ids: pos for pos, ids in enumerate(ids_1)}
    index2d = {ids: pos for pos, ids in enumerate(ids_2)}

    inter = set(ids_1).intersection(ids_2)

    if len(inter) == 0:
        raise(Exception("Error! No common sample index between: {0}... and {1}...".format(
            ids_1[:2], ids_2[:2])))

    index1 = np.asarray([index1d[sample] for sample in inter])
    index2 = np.asarray([index2d[sample] for sample in inter])

    return index1, index2, list(inter)


def convert_metadata_frame_to_matrix(frame):
    """ """
    lbl = LabelBinarizer()

    normed_matrix = np.zeros((frame.shape[0], 0))
    keys = []

    for key in frame.keys():
        if str(frame[key].dtype) == 'object' or str(frame[key].dtype) == 'string':
            matrix = lbl.fit_transform(frame[key].astype('string'))
            if lbl.y_type_ == "binary":
                keys += list(["{0}_{1}".format(key, lbl.classes_[lbl.pos_label])])
            else:
                keys += ["{0}_{1}".format(key, k) for k in lbl.classes_]
        else:
            rbs = RobustScaler()
            matrix = np.asarray(frame[key]).reshape((frame.shape[0], 1))
            matrix = rbs.fit_transform(matrix)
            keys.append(key)

        normed_matrix = hstack([normed_matrix, matrix])

    return pd.DataFrame(normed_matrix, columns=keys)


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

        f_matrix.append(list(map(f_type, line[1:MAX_FEATURE])))

    f_matrix = np.array(f_matrix)

    if f_matrix.shape[1] == len(feature_ids) - 1:
        feature_ids = feature_ids[1:]

    assert(f_matrix.shape[1] == len(feature_ids))
    assert(f_matrix.shape[0] == len(sample_ids))

    f_tsv.close()

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
            f_matrix.append(list(map(f_type, line[1:])))


    f_matrix = np.array(f_matrix).T

    assert(f_matrix.shape[1] == len(feature_ids))
    assert(f_matrix.shape[0] == len(sample_ids))

    f_tsv.close()

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
    node_id, activity, isdead, nbdays, seed, metadata_mat = inp
    pvalue = coxph(activity,
                   isdead,
                   nbdays,
                   seed=seed,
                   metadata_mat=metadata_mat)

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
        median_diff = np.median(array) - np.median(array_comp)

        if pvalue < 0.001:
            results.append((cluster, feature, median_diff, pvalue))

    return results

def save_matrix(matrix, feature_array, sample_array,
                path_folder, project_name, key='', sep='\t'):
    """
    """
    if not isdir(path_folder):
        mkdir(path_folder)

    if key:
        key = '_' + key

    f_csv = open('{0}/{1}{2}.tsv'.format(path_folder, project_name, key), 'w')

    f_csv.write(sep + sep.join(map(lambda x:x.split('_', 1)[-1], feature_array)) + '\n')

    for sample, vector in zip(sample_array, matrix):
        vector = np.asarray(vector).reshape(-1)
        f_csv.write('{0}{1}'.format(sample, sep) + sep.join(map(str, vector)) + '\n')

    print('{0}/{1}{2}.tsv saved'.format(path_folder, project_name, key))
