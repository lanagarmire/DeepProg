""" """
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

from simdeep.config import TRAINING_TSV
from simdeep.config import SURVIVAL_TSV
from simdeep.config import TSV_TEST
from simdeep.config import DATA_TYPE_TEST
from simdeep.config import SURVIVAL_TSV_TEST
from simdeep.config import PATH_DATA

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
    load_data.load_survival()

    load_data.load_matrix_test_v2()
    load_data.load_survival_test()

class LoadData():
    def __init__(
            self,
            path_data=PATH_DATA,
            training_tsv=TRAINING_TSV,
            survival_tsv=SURVIVAL_TSV,
            tsv_test=TSV_TEST,
            survival_tsv_test=SURVIVAL_TSV_TEST,
            data_type_test=DATA_TYPE_TEST):
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
        self.sample_ids = []
        self.data_type = training_tsv.keys()

        self.matrix_stacked = None
        self.features_stacked = None
        self.survival = None

        self.tsv_test = tsv_test
        self.survival_tsv_test = survival_tsv_test
        self.data_type_test = data_type_test

        self.matrix_test = None
        self.matrix_ref = None
        self.features_test = None
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
        sample_ids, feature_ids, matrix = load_data_from_tsv(self.tsv_test,
                                                             path_data=self.path_data)

        feature_ids_ref = self.feature_array[self.data_type_test]
        matrix_ref = self.matrix_array[self.data_type_test]

        common_features = set(feature_ids).intersection(feature_ids_ref)

        feature_ids_dict = {feat: i for i,feat in enumerate(feature_ids)}
        feature_ids_ref_dict = {feat: i for i,feat in enumerate(feature_ids_ref)}

        feature_index = [feature_ids_dict[feature] for feature in common_features]
        feature_ref_index = [feature_ids_ref_dict[feature] for feature in common_features]

        self.matrix_test = np.nan_to_num(matrix.T[feature_index].T)
        self.matrix_ref = np.nan_to_num(matrix_ref.T[feature_ref_index].T)
        self.sample_ids_test = sample_ids
        self.features_test = common_features

    def load_matrix_test_v2(self):
        """ """
        sample_ids, feature_ids, matrix = load_data_from_tsv(self.tsv_test,
                                                             path_data=self.path_data)

        feature_ids_ref = self.feature_array[self.data_type_test]
        matrix_ref = self.matrix_array[self.data_type_test]

        common_features = set(feature_ids).intersection(feature_ids_ref)

        feature_ids_dict = {feat: i for i,feat in enumerate(feature_ids)}
        feature_ids_ref_dict = {feat: i for i,feat in enumerate(feature_ids_ref)}

        feature_index = [feature_ids_dict[feature] for feature in common_features]
        feature_ref_index = [feature_ids_ref_dict[feature] for feature in common_features]

        matrix = np.nan_to_num(matrix.T[feature_index].T)
        matrix_ref = np.nan_to_num(matrix_ref.T[feature_ref_index].T)

        matrix_ref, matrix = self.transform_matrices(
            matrix_ref,
            matrix,
            mad_scale=False,
            robust_scale=False,
            min_max_scale=False,
            unit_norm=False,
            rank_scale=True,
            feature_reduction=True
        )

        self.matrix_test = matrix
        self.matrix_ref = matrix_ref

        self.sample_ids_test = sample_ids

    def load_array(self):
        """ """
        print 'loading data...'
        t = time()

        self.feature_array = {}
        self.matrix_array = {}

        data = self.data_type[0]
        f_name = self.training_tsv[data]

        self.sample_ids, feature_ids, matrix = load_data_from_tsv(f_name,
                                                                  path_data=self.path_data)
        print '{0} loaded of dim:{1}'.format(f_name, matrix.shape)

        self.feature_array[data] = feature_ids
        self.matrix_array[data] = matrix

        for data in self.data_type[1:]:
            f_name = self.training_tsv[data]
            sample_ids, feature_ids, matrix = load_data_from_tsv(f_name,
                                                                 path_data=self.path_data)
            assert(self.sample_ids == sample_ids)

            self.feature_array[data] = feature_ids
            self.matrix_array[data] = matrix

            print '{0} loaded of dim:{1}'.format(f_name, matrix.shape)

        self._stack_matrices()
        print 'data loaded in {0} s'.format(time() - t)

    def _stack_matrices(self):
        """ """
        self.matrix_stacked = hstack([self.matrix_array[data]
                                      for data in self.data_type])
        self.features_stacked = []
        self.stacked_features = [feat for data in self.data_type
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

    def normalize(self,
                  do_norm_scale=False,
                  do_rank_scale=True,
                  do_dim_reduction=True):
        """ """
        print 'normalizing...'

        if do_rank_scale:
            self.matrix_stacked = RankNorm().fit_transform(
                self.matrix_stacked)

        if do_dim_reduction:
            print 'dim reduction...'
            self.matrix_stacked = self.dim_reducer.fit_transform(
                self.matrix_stacked)
            self.do_feature_reduction = True

            if do_rank_scale:
                self.matrix_stacked = RankNorm().fit_transform(
                    self.matrix_stacked)

        if do_norm_scale:
            self.matrix_stacked = self.normalizer.fit_transform(
                self.matrix_stacked)

    def transform_matrices(self, matrix_ref, matrix,
                           mad_scale=False,
                           robust_scale=False,
                           min_max_scale=False,
                           rank_scale=True,
                           unit_norm=False,
                           feature_reduction=True):
        """ """
        print 'Scaling/Normalising dataset...'
        if rank_scale:
            matrix_ref = RankNorm().fit_transform(matrix_ref)
            matrix = RankNorm().fit_transform(matrix)

        if feature_reduction:
            reducer = CorrelationReducer()
            matrix_ref = reducer.fit_transform(matrix_ref)
            matrix = reducer.transform(matrix)

            if rank_scale:
                matrix_ref = RankNorm().fit_transform(matrix_ref)
                matrix = RankNorm().fit_transform(matrix)

        if mad_scale:
            matrix_ref = self.mad_scaler.fit_transform(matrix_ref.T).T
            matrix = self.mad_scaler.fit_transform(matrix.T).T

        if robust_scale:
            matrix_ref = self.robust_scaler.fit_transform(matrix_ref)
            matrix = self.robust_scaler.transform(matrix)

        if min_max_scale:
            matrix_ref = self.min_max_scaler.fit_transform(matrix_ref)
            matrix = self.min_max_scaler.fit_transform(matrix)

        if unit_norm:
            matrix_ref = self.normalizer.fit_transform(matrix_ref)
            matrix = self.normalizer.transform(matrix)

        return matrix_ref, matrix


if __name__ == '__main__':
    main()
