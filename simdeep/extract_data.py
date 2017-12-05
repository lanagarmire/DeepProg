""" """
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import quantile_transform

from simdeep.config import TRAINING_TSV
from simdeep.config import SURVIVAL_TSV

from simdeep.config import PATH_DATA
from simdeep.config import STACK_MULTI_OMIC

from simdeep.config import NORMALIZATION

from simdeep.config import FILL_UNKOWN_FEATURE_WITH_0

from simdeep.config import CROSS_VALIDATION_INSTANCE
from simdeep.config import TEST_FOLD

from simdeep.survival_utils import load_data_from_tsv
from simdeep.survival_utils import load_survival_file
from simdeep.survival_utils import MadScaler
from simdeep.survival_utils import RankNorm
from simdeep.survival_utils import CorrelationReducer
from simdeep.survival_utils import VarianceReducer

from collections import defaultdict

from time import time

import numpy as np

from numpy import hstack
from numpy import vstack

######################## VARIABLE ############################
QUANTILE_OPTION = {'n_quantiles': 100,
                   'output_distribution':'normal'}
###############################################################


def main():
    """ """
    load_data = LoadData(normalization={'TRAIN_NORM_SCALE': True,})
    load_data.load_array()
    load_data.load_survival()
    load_data.create_a_cv_split()

    load_data.normalize_training_array()
    load_data.load_matrix_test()

    load_data.load_matrix_test_fold()
    load_data.load_matrix_full()

    load_data.load_new_test_dataset(
        {'METH': '../../../../data/survival_analysis_multiple/meth_validation.tsv'},
        '../../../../data/survival_analysis_multiple/survival_event_meth.txt',
        normalization={'TRAIN_CORR_REDUCTION': True})


class LoadData():
    """
    """

    def __init__(
            self,
            path_data=PATH_DATA,
            training_tsv=TRAINING_TSV,
            survival_tsv=SURVIVAL_TSV,
            cross_validation_instance=CROSS_VALIDATION_INSTANCE,
            test_fold=TEST_FOLD,
            stack_multi_omic=STACK_MULTI_OMIC,
            fill_unkown_feature_with_0=FILL_UNKOWN_FEATURE_WITH_0,
            normalization=NORMALIZATION,
            _parameters={},
            verbose=True,
    ):
        """
        class to extract data
        :training_matrices: dict(matrice_type, path to the tsv file)

        :path_data: str    path to the folder containing the data
        :training_tsv: dict    dict('data type', 'name of the tsv file')
        :survival_tsv: str    name of the tsv file containing the survival data
                              of the training set
        :survival_tsv_test: str    name of the tsv file containing the survival data
                                   of the test set
        :tsv_test: str    name of the file containing the test dataset
        :data_type_test: str    name of the data type of the test set
                                must match a key existing in training_tsv
        """

        self.verbose = verbose
        self.do_stack_multi_omic = stack_multi_omic
        self.path_data = path_data
        self.survival_tsv = survival_tsv
        self.training_tsv = training_tsv
        self.fill_unkown_feature_with_0 = fill_unkown_feature_with_0
        self.feature_array = {}
        self.matrix_array = {}

        self.test_tsv = None
        self.matrix_train_array = {}

        self.sample_ids = []
        self.data_type = training_tsv.keys()

        self.survival = None
        self.survival_tsv_test = None

        self.matrix_full_array = {}
        self.sample_ids_full = []
        self.survival_full = None

        self.feature_test_array = {}
        self.matrix_test_array = {}

        self.sample_ids_cv = []
        self.matrix_cv_array = {}
        self.survival_cv = None

        self._cv_loaded = False
        self._full_loaded = False

        self.matrix_ref_array = {}
        self.feature_ref_array = {}
        self.feature_ref_index = {}
        self.feature_train_array = {}
        self.feature_train_index = {}

        self.survival_test = None
        self.sample_ids_test = None

        self.cross_validation_instance = cross_validation_instance
        self.test_fold = test_fold

        self.do_feature_reduction = None

        self.normalizer = Normalizer()
        self.mad_scaler = MadScaler()
        self.robust_scaler = RobustScaler()
        self.min_max_scaler = MinMaxScaler()
        self.dim_reducer = CorrelationReducer()
        self.variance_reducer = VarianceReducer()

        self._parameters = _parameters
        self.normalization = defaultdict(bool, normalization)
        self.normalization_test = None

    def __del__(self):
        """
        """
        import gc
        gc.collect()

    def _stack_multiomics(self, arrays=None, features=None):
        """
        """
        if not self.do_stack_multi_omic:
            return

        if arrays is not None and len(arrays) > 1:
            arrays['STACKED'] = hstack(
                arrays.values())

            for key in arrays.keys():
                arrays.pop(key) if key != 'STACKED' else True

        if not features:
            return

        if len(features) > 1:
            features['STACKED'] = [feat for key in features
                                   for feat in features[key]]
            for key in features.keys():
                features.pop(key) if key != 'STACKED' else True

            self.feature_ref_index['STACKED'] = {feature: pos for pos, feature
                                                 in enumerate(features['STACKED'])}

    def load_matrix_test_fold(self):
        """ """
        if not self.cross_validation_instance or self._cv_loaded:
            return

        for key in self.matrix_array:

            matrix_test = self.matrix_cv_array[key].copy()
            matrix_ref = self.matrix_array[key].copy()

            matrix_ref, matrix_test = self.transform_matrices(
                matrix_ref, matrix_test, key,
            )

            self.matrix_cv_array[key] = matrix_test

        self._stack_multiomics(self.matrix_cv_array)
        self._cv_loaded = True

    def load_matrix_test(self, normalization=None):
        """ """
        if normalization is not None:
            self.normalization_test = normalization
        else:
            self.normalization_test = self.normalization

        for key in self.test_tsv:
            sample_ids, feature_ids, matrix = load_data_from_tsv(
                f_name=self.test_tsv[key],
                key=key,
                path_data=self.path_data)

            feature_ids_ref = self.feature_array[key]
            matrix_ref = self.matrix_array[key].copy()

            common_features = set(feature_ids).intersection(feature_ids_ref)

            if self.verbose:
                print('nb common features for the test set:{0}'.format(len(common_features)))

            feature_ids_dict = {feat: i for i,feat in enumerate(feature_ids)}
            feature_ids_ref_dict = {feat: i for i,feat in enumerate(feature_ids_ref)}

            if len(common_features) < len(feature_ids_ref) and self.fill_unkown_feature_with_0:
                missing_features = set(feature_ids_ref).difference(common_features)

                if self.verbose:
                    print('filling {0} with 0 for {1} additional features'.format(
                        key, len(missing_features)))

                matrix = hstack([matrix, np.zeros((len(sample_ids), len(missing_features)))])

                for i, feat in enumerate(missing_features):
                    feature_ids_dict[feat] = i + len(feature_ids)

                common_features = feature_ids_ref

            feature_index = [feature_ids_dict[feature] for feature in common_features]
            feature_ref_index = [feature_ids_ref_dict[feature] for feature in common_features]

            matrix_test = np.nan_to_num(matrix.T[feature_index].T)
            matrix_ref = np.nan_to_num(matrix_ref.T[feature_ref_index].T)

            self.feature_test_array[key] = list(common_features)

            if not isinstance(self.sample_ids_test, type(None)):
                try:
                    assert(self.sample_ids_test == sample_ids)
                except Exception as e:
                    raise Exception('Assertion error when loading test sample ids!')
            else:
                self.sample_ids_test = sample_ids

            matrix_ref, matrix_test = self.transform_matrices(
                matrix_ref, matrix_test, key, normalization=normalization)

            self._define_test_features(key, normalization)

            self.matrix_test_array[key] = matrix_test
            self.matrix_ref_array[key] = matrix_ref
            self.feature_ref_array[key] = self.feature_test_array[key]
            self.feature_ref_index[key] = {feat: pos for pos, feat in enumerate(common_features)}

            self._define_ref_features(key, normalization)

        self._stack_multiomics(self.matrix_test_array,
                               self.feature_test_array)
        self._stack_multiomics(self.matrix_ref_array,
                               self.feature_ref_array)

    def load_new_test_dataset(self, tsv_dict, path_survival_file, normalization=None):
        """
        """
        if normalization is not None:
            normalization = defaultdict(bool, normalization)

        self.test_tsv = tsv_dict
        self.survival_test = None
        self.sample_ids_test = None
        self.survival_tsv_test = path_survival_file

        self.matrix_test_array = {}
        self.matrix_ref_array = {}
        self.feature_test_array = {}
        self.feature_ref_array = {}
        self.feature_ref_index = {}

        self.load_matrix_test(normalization)
        self.load_survival_test()

    def _create_ref_matrix(self, key):
        """ """
        features_test = self.feature_test_array[key]

        features_train = self.feature_train_array[key]
        matrix_train = self.matrix_ref_array[key]

        test_dict = {feat: pos for pos, feat in enumerate(features_test)}
        train_dict = {feat: pos for pos, feat in enumerate(features_train)}

        index = [train_dict[feat] for feat in features_test]

        self.feature_ref_array[key] = self.feature_test_array[key]
        self.matrix_ref_array[key] = np.nan_to_num(matrix_train.T[index].T)

        self.feature_ref_index[key] = test_dict

    def load_array(self):
        """ """
        if self.verbose:
            print('loading data...')

        t = time()

        self.feature_array = {}
        self.matrix_array = {}

        data = self.data_type[0]
        f_name = self.training_tsv[data]

        self.sample_ids, feature_ids, matrix = load_data_from_tsv(
            f_name=f_name,
            key=data,
            path_data=self.path_data)

        if self.verbose:
            print('{0} loaded of dim:{1}'.format(f_name, matrix.shape))

        self.feature_array[data] = feature_ids
        self.matrix_array[data] = matrix

        for data in self.data_type[1:]:
            f_name = self.training_tsv[data]
            sample_ids, feature_ids, matrix = load_data_from_tsv(
                f_name=f_name,
                key=data,
                path_data=self.path_data)
            try:
                assert(self.sample_ids == sample_ids)
            except Exception as e:
                raise Exception('Assertion error: {0} when loading'\
                                ' the sample from the training set!'.format(e))

            self.feature_array[data] = feature_ids
            self.matrix_array[data] = matrix

            if self.verbose:
                print('{0} loaded of dim:{1}'.format(f_name, matrix.shape))

        if self.verbose:
            print('data loaded in {0} s'.format(time() - t))

    def create_a_cv_split(self):
        """ """
        if not self.cross_validation_instance:
            return

        cv = self.cross_validation_instance
        train, test = [(tn, tt) for tn, tt in cv.split(self.sample_ids)][self.test_fold]

        for key in self.matrix_array:
            self.matrix_cv_array[key] = self.matrix_array[key][test]
            self.matrix_array[key] = self.matrix_array[key][train]

        self.survival_cv = self.survival.copy()[test]
        self.survival = self.survival[train]

        self.sample_ids_cv = np.asarray(self.sample_ids)[test].tolist()
        self.sample_ids = np.asarray(self.sample_ids)[train].tolist()

    def load_matrix_full(self):
        """
        """
        if self._full_loaded:
            return

        if not self.cross_validation_instance:
            self.matrix_full_array = self.matrix_train_array
            self.sample_ids_full = self.sample_ids
            self.survival_full = self.survival
            return

        if not self._cv_loaded:
            self.load_matrix_test_fold()

        for key in self.matrix_train_array:
            self.matrix_full_array[key] = vstack([self.matrix_train_array[key],
                                                  self.matrix_cv_array[key]])

        self.sample_ids_full = self.sample_ids[:] + self.sample_ids_cv[:]
        self.survival_full = vstack([self.survival, self.survival_cv])

        self._full_loaded = True

    def load_survival(self):
        """ """
        survival = load_survival_file(self.survival_tsv, path_data=self.path_data)
        matrix = []

        retained_samples = []
        sample_removed = 0

        for ids, sample in enumerate(self.sample_ids):
            if sample not in survival:
                sample_removed += 1
                continue

            retained_samples.append(ids)
            matrix.append(survival[sample])

        self.survival = np.asmatrix(matrix)

        if sample_removed:
            for key in self.matrix_array:
                self.matrix_array[key] = self.matrix_array[key][retained_samples]

            self.sample_ids = np.asarray(self.sample_ids)[retained_samples]

            if self.verbose:
                print('{0} samples without survival removed'.format(sample_removed))

    def load_survival_test(self):
        """ """
        survival = load_survival_file(self.survival_tsv_test, path_data=self.path_data)
        matrix = []

        retained_samples = []
        sample_removed = 0

        for ids, sample in enumerate(self.sample_ids_test):
            if sample not in survival:
                sample_removed += 1
                continue

            retained_samples.append(ids)
            matrix.append(survival[sample])

        self.survival_test = np.asmatrix(matrix)

        if sample_removed:
            for key in self.matrix_test_array:
                self.matrix_test_array[key] = self.matrix_test_array[key][retained_samples]

            self.sample_ids_test = np.asarray(self.sample_ids_test)[retained_samples]

            if self.verbose:
                print('{0} samples without survival removed'.format(sample_removed))

    def _define_train_features(self, key):
        """ """
        self.feature_train_array[key] = self.feature_array[key][:]

        if self.normalization['TRAIN_CORR_REDUCTION']:
            self.feature_train_array[key] = ['{0}_{1}'.format(key, sample)
                                             for sample in self.sample_ids]
        elif self.normalization['NB_FEATURES_TO_KEEP']:
            self.feature_train_array[key] = np.array(self.feature_train_array[key])[
                self.variance_reducer.index_to_keep].tolist()

        self.feature_ref_array[key] = self.feature_train_array[key]

        self.feature_train_index[key] = {key: id for id, key in enumerate(
            self.feature_train_array[key])}
        self.feature_ref_index[key] = self.feature_train_index[key]

    def _define_test_features(self, key, normalization=None):
        """ """
        if normalization is None:
            normalization = self.normalization

        if normalization['TRAIN_CORR_REDUCTION']:
            self.feature_test_array[key] = ['{0}_{1}'.format(key, sample)
                                             for sample in self.sample_ids]

        elif normalization['NB_FEATURES_TO_KEEP']:
            self.feature_test_array[key] = np.array(self.feature_test_array[key])[
                self.variance_reducer.index_to_keep].tolist()

    def _define_ref_features(self, key, normalization=None):
        """ """
        if normalization is None:
            normalization = self.normalization

        if normalization['TRAIN_CORR_REDUCTION']:
            self.feature_ref_array[key] = ['{0}_{1}'.format(key, sample)
                                           for sample in self.sample_ids]

            self.feature_ref_index[key] = {feat:pos for pos, feat in
                                           enumerate(self.feature_ref_array[key])}

        elif normalization['NB_FEATURES_TO_KEEP']:
            self.feature_ref_index[key] = {feat: pos for pos, feat in
                                           enumerate(self.feature_ref_array[key])}

    def normalize_training_array(self):
        """ """
        for key in self.matrix_array:
            matrix = self.matrix_array[key].copy()
            matrix = self._normalize(matrix, key)

            self.matrix_train_array[key] = matrix
            self.matrix_ref_array[key] = self.matrix_train_array[key]
            self._define_train_features(key)

        self._stack_multiomics(self.matrix_train_array, self.feature_train_array)
        self._stack_multiomics(self.matrix_ref_array, self.feature_ref_array)
        self._stack_index()

    def _stack_index(self):
        """
        """
        if not self.do_stack_multi_omic:
            return

        index = {'STACKED':{}}
        count = 0

        for key in self.feature_train_index:
            for feature in self.feature_train_index[key]:
                index['STACKED'][feature] = count + self.feature_train_index[key][feature]

            count += len(self.feature_train_index[key])

        self.feature_train_index = index
        self.feature_ref_index = self.feature_train_index

    def _normalize(self, matrix, key):
        """ """
        if self.verbose:
            print('normalizing for {0}...'.format(key))

        if self.normalization['NB_FEATURES_TO_KEEP']:
            self.variance_reducer.nb_features = self.normalization['NB_FEATURES_TO_KEEP']
            matrix = self.variance_reducer.fit_transform(matrix)

        if self.normalization['TRAIN_MIN_MAX']:
            matrix = MinMaxScaler().fit_transform(matrix.T).T

        if self.normalization['TRAIN_MAD_SCALE']:
            matrix = self.mad_scaler.fit_transform(matrix.T).T

        if self.normalization['TRAIN_ROBUST_SCALE'] or\
           self.normalization['TRAIN_ROBUST_SCALE_TWO_WAY']:
            matrix = self.robust_scaler.fit_transform(matrix)

        if self.normalization['TRAIN_NORM_SCALE']:
            matrix = self.normalizer.fit_transform(matrix)

        if self.normalization['TRAIN_QUANTILE_TRANSFORM']:
            matrix = quantile_transform(matrix, **QUANTILE_OPTION)

        if self.normalization['TRAIN_RANK_NORM']:
            matrix = RankNorm().fit_transform(
                matrix)

        if self.normalization['TRAIN_CORR_REDUCTION']:
            args = self.normalization['TRAIN_CORR_REDUCTION']
            if args == True:
                args = {}

            if self.verbose:
                print('dim reduction for {0}...'.format(key))

            reducer = CorrelationReducer(**args)
            matrix = reducer.fit_transform(
                matrix)

            if self.normalization['TRAIN_CORR_RANK_NORM']:
                matrix = RankNorm().fit_transform(
                    matrix)

            if self.normalization['TRAIN_CORR_QUANTILE_NORM']:
                matrix = quantile_transform(matrix, **QUANTILE_OPTION)

            if self.normalization['TRAIN_CORR_NORM_SCALE']:
                matrix = self.normalizer.fit_transform(matrix)

        return np.nan_to_num(matrix)

    def transform_matrices(self, matrix_ref, matrix, key, normalization=None):
        """ """
        if normalization is None:
            normalization = self.normalization

        if self.verbose:
            print('Scaling/Normalising dataset...')

        if normalization['LOG_REF_MATRIX']:
            matrix_ref = np.log2(1.0 + matrix_ref)

        if normalization['LOG_TEST_MATRIX']:
            matrix = np.log2(1.0 +  matrix)

        if normalization['NB_FEATURES_TO_KEEP']:
            self.variance_reducer.nb_features = normalization['NB_FEATURES_TO_KEEP']
            matrix_ref = self.variance_reducer.fit_transform(matrix_ref)
            matrix = self.variance_reducer.transform(matrix)

        if normalization['TRAIN_MIN_MAX']:
            matrix_ref = self.min_max_scaler.fit_transform(matrix_ref.T).T
            matrix = self.min_max_scaler.fit_transform(matrix.T).T

        if normalization['TRAIN_MAD_SCALE']:
            matrix_ref = self.mad_scaler.fit_transform(matrix_ref.T).T
            matrix = self.mad_scaler.fit_transform(matrix.T).T

        if normalization['TRAIN_ROBUST_SCALE']:
            matrix_ref = self.robust_scaler.fit_transform(matrix_ref)
            matrix = self.robust_scaler.transform(matrix)

        if normalization['TRAIN_ROBUST_SCALE_TWO_WAY']:
            matrix_ref = self.robust_scaler.fit_transform(matrix_ref)
            matrix = self.robust_scaler.transform(matrix)

        if normalization['TRAIN_NORM_SCALE']:
            matrix_ref = self.normalizer.fit_transform(matrix_ref)
            matrix = self.normalizer.transform(matrix)

        if self.normalization['TRAIN_QUANTILE_TRANSFORM']:
            matrix_ref = quantile_transform(matrix_ref, **QUANTILE_OPTION)
            matrix = quantile_transform(matrix, **QUANTILE_OPTION)

        if normalization['TRAIN_RANK_NORM']:
            matrix_ref = RankNorm().fit_transform(matrix_ref)
            matrix = RankNorm().fit_transform(matrix)

        if normalization['TRAIN_CORR_REDUCTION']:
            args = normalization['TRAIN_CORR_REDUCTION']

            if args == True:
                args = {}

            reducer = CorrelationReducer(**args)
            matrix_ref = reducer.fit_transform(matrix_ref)
            matrix = reducer.transform(matrix)

            if normalization['TRAIN_CORR_RANK_NORM']:
                matrix_ref = RankNorm().fit_transform(matrix_ref)
                matrix = RankNorm().fit_transform(matrix)

            if self.normalization['TRAIN_CORR_QUANTILE_TRANSFORM']:
                matrix_ref = quantile_transform(matrix_ref, **QUANTILE_OPTION)
                matrix = quantile_transform(matrix, **QUANTILE_OPTION)

            if self.normalization['TRAIN_CORR_NORM_SCALE']:
                matrix_ref = self.normalizer.fit_transform(matrix_ref)
                matrix = self.normalizer.fit_transform(matrix)

        return np.nan_to_num(matrix_ref), np.nan_to_num(matrix)

    def save_dataset(self):
        """
        """
        pass


if __name__ == '__main__':
    main()
