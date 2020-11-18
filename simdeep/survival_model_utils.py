from simdeep.coxph_from_r import predict_with_coxph_glmnet

import numpy as np

from numpy import hstack

import warnings

from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

def test():
    """
    """
    #### Compare glmnet with sksurv CoxnetSurvivalAnalysis
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    ######################################################

    ################ DUMMY DATA ##########################
    isdead = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
    nbdays = [24, 10, 25, 50, 14, 10 ,100, 10, 50, 10]
    matrix = np.array([
        [0, 1, 1, 0 , 1, 2, 0, 1, 0, 0],
        [0, 1, 1, 0 , 1, 3, 0, 1, 0, 0]]).T
    ######################################################

    res = predict_with_coxph_glmnet(
        matrix, isdead, nbdays, matrix)

    coxph = CoxnetSurvivalAnalysis()
    Y = np.asarray([(bool(a), b) for a, b in zip(isdead, nbdays)],
                   dtype=[("event", np.bool), ("time", np.int)])

    coxph.fit(matrix, Y)


class ClusterWithSurvival(object):
    """
    """

    def __init__(self,
                 isdead, nbdays,
                 n_clusters=2,
                 metadata_mat = None,
                 use_gaussian_to_dichotomize=False,
                 use_sksurv=True):
        "docstring"

        self.use_sksurv = use_sksurv
        self.coxph_python = None
        self.isdead = isdead
        self.nbdays = nbdays
        self.n_clusters = n_clusters
        self.metadata_mat = metadata_mat
        self.matrix = None
        self._glm = None
        self._labels = None
        self._use_gaussian_to_dichotomize = use_gaussian_to_dichotomize

    def get_nonzero_features(self, matrix):
        """
        Get non zero features using lasso coxPH
        """
        if self.metadata_mat is not None:
            self.matrix = hstack([matrix, self.metadata_mat])
            rbs = RobustScaler()
            self.matrix = rbs.fit_transform(self.matrix)

        else:
            self.matrix = matrix

        return self._fit_with_python(self.matrix,
                                     l1_ratio=1.0,
                                     return_nonzero_features=True)


    def fit(self, matrix):
        """
        """
        self.matrix = matrix

    def predict(self, matrix_test):
        """
        """
        if self.use_sksurv:
            return self._fit_with_python(matrix_test)
        else:
            return self._fit_with_glm(matrix_test)

    def predict_proba(self, matrix_test):
        """
        """
        if self.use_sksurv:
            return self._fit_with_python(matrix_test, get_proba=True)
        else:
            return self._fit_with_glm(matrix_test, get_proba=True)

    def _fit_with_python(self, matrix_test,
                         get_proba=False,
                         return_nonzero_features=False,
                         l1_ratio=0.5):
        """
        """
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        Y = np.asarray([(bool(a), b) for a, b in zip(
            self.isdead, self.nbdays)],
                       dtype=[("event", np.bool), ("time", np.int)])

        self.coxph_python = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            fit_baseline_model=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.coxph_python.fit(self.matrix, Y)

        predictions = self.coxph_python.predict(matrix_test)

        if get_proba:
            return self._get_proba_from_prediction(predictions)

        if return_nonzero_features:
            for coef in self.coxph_python.coef_.T:
                if coef.sum() != 0:
                    break
            if coef.sum() == 0:
                raise(Exception("All features Coefficient are 0!"))

            if self.metadata_mat is not None:
                if coef[:-self.metadata_mat.shape[1]].sum() == 0:
                    raise(Exception("Only metadata features are non zero"))

                return np.nonzero(coef[:-self.metadata_mat.shape[1]])
            else:
                return np.nonzero(coef)

        return self._fit_and_dichotomise(
            predictions,
            n_clusters=self.n_clusters)

    def _fit_with_glm(self, matrix_test, get_proba=False):
        """
        """
        predictions = predict_with_coxph_glmnet(
            self.matrix, self.isdead, self.nbdays, matrix_test)

        if get_proba:
            return self._get_proba_from_prediction(predictions)

        return self._fit_and_dichotomise(
            predictions,
            n_clusters=self.n_clusters)

    def _fit_and_dichotomise(self, predicted_time, n_clusters=2):
        """
        """
        labels = np.zeros(predicted_time.shape)
        predicted_time[predicted_time == 0] = np.inf

        if self._use_gaussian_to_dichotomize:
            glm = GaussianMixture(n_components=n_clusters)
            self._labels = glm.fit_predict(predicted_time.reshape(1, -1).T)
            self._glm = glm

            return self._labels

        for cluster in range(n_clusters):
            percentile = 100 * (1.0 - 1.0 / (cluster + 1.0))
            value = np.percentile(predicted_time, percentile)
            labels[predicted_time >= value] = n_clusters - cluster

        return labels

    def _get_proba_from_prediction(self, predicted_time, time_of_following=None):
        """
        time_of_following is used to compute the probability of the even happening
        using the predicted values as referendce => proba = time_predicted / time_of_following
        if None, time_of_following is computed using the std of time_predicted for all non zero
        """
        if self._glm is not None:
            return self._glm.predict_proba(predicted_time.reshape(1, -1).T)

        predicted_time = predicted_time.astype("float32")

        if not time_of_following:
            time_of_following = np.max(predicted_time[predicted_time != 0]) + \
                np.std(predicted_time[predicted_time != 0])

        predicted_time[predicted_time == 0] = time_of_following

        return predicted_time / time_of_following


if __name__ == '__main__':
    test()
