""" """
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from simdeep.config import TRAINING_TSV
from simdeep.config import TEST_TSV
from simdeep.config import SURVIVAL_TSV
from simdeep.config import SURVIVAL_TSV_TEST
from simdeep.config import PATH_DATA
from simdeep.config import STACK_MULTI_OMIC

from simdeep.config import TRAIN_MIN_MAX
from simdeep.config import TRAIN_NORM_SCALE
from simdeep.config import TRAIN_CORR_REDUCTION
from simdeep.config import TRAIN_RANK_NORM
from simdeep.config import TRAIN_MAD_SCALE
from simdeep.config import TRAIN_ROBUST_SCALE
from simdeep.config import TRAIN_ROBUST_SCALE_TWO_WAY
from simdeep.config import TRAIN_CORR_RANK_NORM
from simdeep.config import FILL_UNKOWN_FEATURE_WITH_0

from simdeep.config import CROSS_VALIDATION_INSTANCE
from simdeep.config import TEST_FOLD

from simdeep.survival_utils import load_data_from_tsv
from simdeep.survival_utils import load_survival_file
from simdeep.survival_utils import MadScaler
from simdeep.survival_utils import RankNorm
from simdeep.survival_utils import CorrelationReducer

from time import time

import numpy as np

from numpy import hstack
from numpy import vstack


def main():
    """ """
    load_data = LoadData()
    load_data.load_array()
    load_data.load_survival()
    load_data.create_a_cv_split()

    load_data.normalize_training_array()
    load_data.load_matrix_test()

    load_data.load_matrix_test_fold()
    load_data.load_survival_test()
    load_data.load_matrix_full()


class LoadData():
    def __init__(
            self,
            path_data=PATH_DATA,
            training_tsv=TRAINING_TSV,
            test_tsv=TEST_TSV,
            survival_tsv=SURVIVAL_TSV,
            survival_tsv_test=SURVIVAL_TSV_TEST,
            cross_validation_instance=CROSS_VALIDATION_INSTANCE,
            test_fold=TEST_FOLD,
            stack_multi_omic=STACK_MULTI_OMIC,
            fill_unkown_feature_with_0=FILL_UNKOWN_FEATURE_WITH_0,
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

        self.test_tsv = test_tsv
        self.matrix_train_array = {}

        self.sample_ids = []
        self.data_type = training_tsv.keys()

        self._correlation_red_used = False

        self.survival = None

        self.survival_tsv_test = survival_tsv_test

        self.matrix_full_array = {}
        self.sample_ids_full = []
        self.survival_full = None

        self.feature_test_array = {}
        self.matrix_test_array = {}

        self.sample_ids_cv = []
        self.matrix_cv_array = {}
        self.survival_cv = None

        self._cv_loaded = False
        self._test_loaded = False
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

    def load_matrix_test(self):
        """ """
        if self._test_loaded:
            return

        self.matrix_ref_array = {}

        for key in self.test_tsv:
            sample_ids, feature_ids, matrix = load_data_from_tsv(
                f_name=self.test_tsv[key],
                key=key,
                path_data=self.path_data)

            feature_ids_ref = self.feature_array[key]
            matrix_ref = self.matrix_array[key]

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
                assert(self.sample_ids_test == sample_ids)
            else:
                self.sample_ids_test = sample_ids

            matrix_ref, matrix_test = self.transform_matrices(
                matrix_ref, matrix_test, key)

            self._define_test_features(key)

            self.matrix_test_array[key] = matrix_test
            self.matrix_ref_array[key] = matrix_ref
            self.feature_ref_array[key] = self.feature_test_array[key]

            if not self.do_stack_multi_omic:
                self._create_ref_matrix(key)

        self._stack_multiomics(self.matrix_test_array,
                               self.feature_test_array)
        self._stack_multiomics(self.matrix_ref_array,
                               self.feature_ref_array)

        if self.do_stack_multi_omic:
            self._create_ref_matrix('STACKED')

        self._test_loaded = True

    def load_new_test_dataset(self, tsv_dict, path_survival_file):
        """
        """
        self._test_loaded = False
        self.test_tsv = tsv_dict
        self.survival_test = None
        self.sample_ids_test = None
        self.survival_tsv_test = path_survival_file

        self.load_matrix_test()
        self.load_survival_test()

    def _create_ref_matrix(self, key):
        """ """
        features_test = self.feature_test_array[key]
        features_train = self.feature_train_array[key]

        test_dict = {feat: pos for pos, feat in enumerate(features_test)}
        train_dict = {feat: pos for pos, feat in enumerate(features_train)}

        index = [train_dict[feat] for feat in features_test]

        self.feature_ref_array[key] = self.feature_test_array[key]
        self.matrix_ref_array[key] = self.matrix_train_array[key].T[index].T

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
            assert(self.sample_ids == sample_ids)

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

        if self._correlation_red_used:
            self.feature_train_array[key] = ['{0}_{1}'.format(key, sample)
                                             for sample in self.sample_ids]

        self.feature_ref_array[key] = self.feature_train_array[key]

        self.feature_train_index[key] = {key: id for id, key in enumerate(
            self.feature_train_array[key])}
        self.feature_ref_index[key] = self.feature_train_index[key]

    def _define_test_features(self, key):
        """ """
        if self._correlation_red_used:
            self.feature_test_array[key] = ['{0}_{1}'.format(key, sample)
                                             for sample in self.sample_ids]

    def normalize_training_array(self):
        """ """
        for key in self.matrix_array:
            matrix = self.matrix_array[key]
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

    def _normalize(self,
                   matrix,
                   key,
                   mad_scale=TRAIN_MAD_SCALE,
                   robust_scale=TRAIN_ROBUST_SCALE,
                   robust_scale_two_way=TRAIN_ROBUST_SCALE_TWO_WAY,
                   min_max=TRAIN_MIN_MAX,
                   norm_scale=TRAIN_NORM_SCALE,
                   rank_scale=TRAIN_RANK_NORM,
                   corr_rank_scale=TRAIN_CORR_RANK_NORM,
                   dim_reduction=TRAIN_CORR_REDUCTION):
        """ """
        if self.verbose:
            print('normalizing for {0}...'.format(key))

        if min_max:
            matrix = MinMaxScaler().fit_transform(
                matrix.T).T

        if mad_scale:
            matrix = self.mad_scaler.fit_transform(matrix.T).T

        if robust_scale or robust_scale_two_way:
            matrix = self.robust_scaler.fit_transform(matrix)

        if norm_scale:
            matrix = self.normalizer.fit_transform(
                matrix)

        if rank_scale:
            matrix = RankNorm().fit_transform(
                matrix)

        if dim_reduction:
            if self.verbose:
                print('dim reduction for {0}...'.format(key))

            reducer = CorrelationReducer()
            matrix = reducer.fit_transform(
                matrix)
            self._correlation_red_used = True

            if corr_rank_scale:
                matrix = RankNorm().fit_transform(
                    matrix)

        return matrix

    def transform_matrices(self,
                           matrix_ref, matrix, key,
                           mad_scale=TRAIN_MAD_SCALE,
                           robust_scale=TRAIN_ROBUST_SCALE,
                           robust_scale_two_way=TRAIN_ROBUST_SCALE_TWO_WAY,
                           min_max_scale=TRAIN_MIN_MAX,
                           rank_scale=TRAIN_RANK_NORM,
                           correlation_reducer=TRAIN_CORR_REDUCTION,
                           corr_rank_scale=TRAIN_CORR_RANK_NORM,
                           unit_norm=TRAIN_NORM_SCALE):
        """ """
        if self.verbose:
            print('Scaling/Normalising dataset...')

        if min_max_scale:
            matrix_ref = self.min_max_scaler.fit_transform(matrix_ref.T).T
            matrix = self.min_max_scaler.fit_transform(matrix.T).T

        if mad_scale:
            matrix_ref = self.mad_scaler.fit_transform(matrix_ref.T).T
            matrix = self.mad_scaler.fit_transform(matrix.T).T

        if robust_scale:
            matrix_ref = self.robust_scaler.fit_transform(matrix_ref)
            matrix = self.robust_scaler.transform(matrix)

        if robust_scale_two_way:
            matrix_ref = self.robust_scaler.fit_transform(matrix_ref)
            matrix = self.robust_scaler.transform(matrix)

        if unit_norm:
            matrix_ref = self.normalizer.fit_transform(matrix_ref)
            matrix = self.normalizer.transform(matrix)

        if rank_scale:
            matrix_ref = RankNorm().fit_transform(matrix_ref)
            matrix = RankNorm().fit_transform(matrix)

        if correlation_reducer:
            reducer = CorrelationReducer()
            matrix_ref = reducer.fit_transform(matrix_ref)
            matrix = reducer.transform(matrix)

            if corr_rank_scale:
                matrix_ref = RankNorm().fit_transform(matrix_ref)
                matrix = RankNorm().fit_transform(matrix)

        return matrix_ref, matrix

    def save_dataset(self):
        """
        """
        pass


if __name__ == '__main__':
    main()
