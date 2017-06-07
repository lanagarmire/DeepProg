""" """
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from simdeep.config import TRAINING_TSV
from simdeep.config import TEST_TSV
from simdeep.config import SURVIVAL_TSV
from simdeep.config import SURVIVAL_TSV_TEST
from simdeep.config import PATH_DATA

from simdeep.config import MAD_SCALE
from simdeep.config import ROBUST_SCALE
from simdeep.config import MIN_MAX_SCALE
from simdeep.config import UNIT_NORM
from simdeep.config import RANK_SCALE
from simdeep.config import CORRELATION_REDUCER

from simdeep.config import TRAIN_MIN_MAX
from simdeep.config import TRAIN_NORM_SCALE
from simdeep.config import TRAIN_DIM_REDUCTION
from simdeep.config import TRAIN_RANK_NORM

from simdeep.survival_utils import load_data_from_tsv
from simdeep.survival_utils import load_survival_file
from simdeep.survival_utils import MadScaler
from simdeep.survival_utils import RankNorm
from simdeep.survival_utils import CorrelationReducer

from time import time

from numpy import hstack

import numpy as np


def main():
    """ """
    load_data = LoadData()
    load_data.load_array()
    load_data.normalize_training_array()

    load_data.load_survival()

    load_data.load_matrix_test()
    # load_data.reorder_test_matrix('SIJIA')

    load_data.load_survival_test()

class LoadData():
    def __init__(
            self,
            path_data=PATH_DATA,
            training_tsv=TRAINING_TSV,
            test_tsv=TEST_TSV,
            survival_tsv=SURVIVAL_TSV,
            survival_tsv_test=SURVIVAL_TSV_TEST):
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

        self.path_data = path_data
        self.survival_tsv = survival_tsv
        self.training_tsv = training_tsv
        self.feature_array = {}
        self.matrix_array = {}

        self.test_tsv = test_tsv
        self.matrix_array_train = {}

        self.sample_ids = []
        self.data_type = training_tsv.keys()

        self.matrix_stacked = None
        self.features_stacked = None
        self.survival = None

        self.survival_tsv_test = survival_tsv_test

        self.feature_test_array = {}
        self.matrix_test_array = {}
        self.matrix_ref_array = {}
        self.survival_test = None
        self.sample_ids_test = None

        self.do_feature_reduction = None

        self.normalizer = Normalizer()
        self.mad_scaler = MadScaler()
        self.robust_scaler = RobustScaler()
        self.min_max_scaler = MinMaxScaler()
        self.dim_reducer = CorrelationReducer()

    def load_matrix_test(self):
        """ """
        for key in self.test_tsv:
            sample_ids, feature_ids, matrix = load_data_from_tsv(self.test_tsv[key],
                                                             path_data=self.path_data)

            feature_ids_ref = self.feature_array[key]
            matrix_ref = self.matrix_array[key]

            common_features = set(feature_ids).intersection(feature_ids_ref)

            feature_ids_dict = {feat: i for i,feat in enumerate(feature_ids)}
            feature_ids_ref_dict = {feat: i for i,feat in enumerate(feature_ids_ref)}

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
                matrix_ref, matrix_test, key,
                unit_norm=TRAIN_NORM_SCALE,
                rank_scale=TRAIN_RANK_NORM,
                min_max_scale=TRAIN_MIN_MAX,
                correlation_reducer=TRAIN_DIM_REDUCTION,
            )

            self.matrix_test_array[key] = matrix_test
            self.matrix_ref_array[key] = matrix_ref

    def reorder_test_matrix(self, key):
        """ """
        features_test = self.feature_test_array[key]
        features_ref = self.feature_array[key]

        ref_dict = {feat: pos for pos, feat in enumerate(features_test)}
        index = [ref_dict[feat] for feat in features_ref]

        self.feature_test_array[key] = features_ref[:]

        self.matrix_test_array[key] = self.matrix_test_array[key].T[index].T
        self.matrix_ref_array[key] = self.matrix_ref_array[key].T[index].T

    def load_array(self):
        """ """
        print('loading data...')
        t = time()

        self.feature_array = {}
        self.matrix_array = {}

        data = self.data_type[0]
        f_name = self.training_tsv[data]

        self.sample_ids, feature_ids, matrix = load_data_from_tsv(f_name,
                                                                  path_data=self.path_data)
        print('{0} loaded of dim:{1}'.format(f_name, matrix.shape))

        self.feature_array[data] = feature_ids
        self.matrix_array[data] = matrix

        for data in self.data_type[1:]:
            f_name = self.training_tsv[data]
            sample_ids, feature_ids, matrix = load_data_from_tsv(f_name,
                                                                 path_data=self.path_data)
            assert(self.sample_ids == sample_ids)

            self.feature_array[data] = feature_ids
            self.matrix_array[data] = matrix

            print('{0} loaded of dim:{1}'.format(f_name, matrix.shape))

        self._stack_matrices()
        print('data loaded in {0} s'.format(time() - t))

    def _stack_matrices(self):
        """ """
        self.matrix_stacked = hstack([self.matrix_array[data]
                                      for data in self.data_type])
        self.features_stacked = [feat for data in self.data_type
                                 for feat in self.feature_array[data]]

    def load_survival(self):
        """ """
        survival = load_survival_file(self.survival_tsv, path_data=self.path_data)
        matrix = []

        for sample in self.sample_ids:
            assert(sample in survival)
            matrix.append(survival[sample])

        self.survival = np.asmatrix(matrix)

    def load_survival_test(self):
        """ """
        survival = load_survival_file(self.survival_tsv_test, path_data=self.path_data)
        matrix = []

        for sample in self.sample_ids_test:
            assert(sample in survival)
            matrix.append(survival[sample])

        self.survival_test = np.asmatrix(matrix)

    def normalize_training_array(self):
        """ """
        for key in self.matrix_array:
            matrix = self.matrix_array[key]
            matrix = self._normalize(matrix, key)
            self.matrix_array_train[key] = matrix

    def _normalize(self,
                   matrix,
                   key,
                   do_min_max=TRAIN_MIN_MAX,
                   do_norm_scale=TRAIN_NORM_SCALE,
                   do_rank_scale=TRAIN_RANK_NORM,
                   do_dim_reduction=TRAIN_DIM_REDUCTION):
        """ """
        print('normalizing...')

        if do_min_max:
            matrix = MinMaxScaler().fit_transform(
                matrix)

        if do_rank_scale and not do_min_max:
            matrix = RankNorm().fit_transform(
                matrix)

        if do_dim_reduction:
            print('dim reduction...')
            reducer = CorrelationReducer()
            matrix = reducer.fit_transform(
                matrix)

            if do_rank_scale:
                matrix = RankNorm().fit_transform(
                    matrix)

        if do_norm_scale:
            matrix = self.normalizer.fit_transform(
                matrix)

        return matrix

    def transform_matrices(self,
                           matrix_ref, matrix, key,
                           mad_scale=MAD_SCALE,
                           robust_scale=ROBUST_SCALE,
                           min_max_scale=MIN_MAX_SCALE,
                           rank_scale=RANK_SCALE,
                           correlation_reducer=CORRELATION_REDUCER,
                           unit_norm=UNIT_NORM):
        """ """
        print('Scaling/Normalising dataset...')
        if min_max_scale:
            matrix_ref = self.min_max_scaler.fit_transform(matrix_ref)
            matrix = self.min_max_scaler.fit_transform(matrix)

        if rank_scale and not min_max_scale:
            matrix_ref = RankNorm().fit_transform(matrix_ref)
            matrix = RankNorm().fit_transform(matrix)

        if correlation_reducer:
            reducer = CorrelationReducer()
            matrix_ref = reducer.fit_transform(matrix_ref)
            matrix = reducer.transform(matrix)

            self.feature_test_array[key] = self.sample_ids
            self.feature_array[key] = self.sample_ids

            if rank_scale:
                matrix_ref = RankNorm().fit_transform(matrix_ref)
                matrix = RankNorm().fit_transform(matrix)

        if mad_scale:
            matrix_ref = self.mad_scaler.fit_transform(matrix_ref.T).T
            matrix = self.mad_scaler.fit_transform(matrix.T).T

        if robust_scale:
            matrix_ref = self.robust_scaler.fit_transform(matrix_ref)
            matrix = self.robust_scaler.transform(matrix)

        if unit_norm:
            matrix_ref = self.normalizer.fit_transform(matrix_ref)
            matrix = self.normalizer.transform(matrix)

        return matrix_ref, matrix


if __name__ == '__main__':
    main()
