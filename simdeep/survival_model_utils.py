from simdeep.coxph_from_r import predict_with_coxph_glmnet

import numpy as np

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

    print(fit_and_dichotomise(res))
    print(fit_and_dichotomise(coxph.predict(matrix)))


class ClusterWithSurvival(object):
    """
    """

    def __init__(self,
                 isdead, nbdays,
                 n_clusters=2,
                 use_sksurv=True):
        "docstring"

        self.use_sksurv = use_sksurv
        self.coxph_python = None
        self.isdead = isdead
        self.nbdays = nbdays
        self.n_clusters = n_clusters

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

    def _fit_with_python(self, matrix_test, get_proba=False):
        """
        """
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        Y = np.asarray([(bool(a), b) for a, b in zip(
            self.isdead, self.nbdays)],
                       dtype=[("event", np.bool), ("time", np.int)])

        self.coxph_python = CoxnetSurvivalAnalysis(l1_ratio=0.5)
        self.coxph_python.fit(self.matrix, Y)

        predictions = self.coxph_python.predict(matrix_test)

        if get_proba:
            return get_proba_from_prediction(predictions)

        return fit_and_dichotomise(
            predictions,
            n_clusters=self.n_clusters)

    def _fit_with_glm(self, matrix_test, get_proba=False):
        """
        """
        predictions = predict_with_coxph_glmnet(
            self.matrix, self.isdead, self.nbdays, matrix_test)

        if get_proba:
            return get_proba_from_prediction(predictions)

        return fit_and_dichotomise(predictions,
                                   n_clusters=self.n_clusters)


def fit_and_dichotomise(predicted_time, n_clusters=2):
    """
    """
    labels = np.zeros(predicted_time.shape)
    predicted_time[predicted_time == 0] = np.inf

    for cluster in range(n_clusters):
        percentile = 100 * (1.0 - 1.0 / (cluster + 1.0))
        value = np.percentile(predicted_time, percentile)
        labels[predicted_time >= value] = n_clusters - cluster

    return labels

def get_proba_from_prediction(predicted_time, time_of_following=None):
    """
    time_of_following is used to compute the probability of the even happening
    using the predicted values as referendce => proba = time_predicted / time_of_following
    if None, time_of_following is computed using the std of time_predicted for all non zero
    """
    predicted_time = predicted_time.astype("float32")

    if not time_of_following:
        time_of_following = np.max(predicted_time[predicted_time != 0]) + \
            np.std(predicted_time[predicted_time != 0])

    predicted_time[predicted_time == 0] = time_of_following

    return predicted_time / time_of_following


if __name__ == '__main__':
    test()
