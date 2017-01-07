"""
SimDeep main class
"""

from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score

from simdeep.deepmodel_base import DeepBase

from simdeep.config import NB_CLUSTERS
from simdeep.config import PVALUE_THRESHOLD
from simdeep.config import SELECT_FEATURES_METHOD
from simdeep.config import CLASSIFIER

from simdeep.survival_utils import select_best_classif_params

from coxph_from_r import coxph
from coxph_from_r import surv_mean

from collections import Counter

import numpy as np


def main():
    """
    DEBUG function
    """
    sim_deep = SimDeep()
    sim_deep.load_training_dataset()
    sim_deep.fit()
    # sim_deep.load_encoder('encoder_seed_s0_full.h5')
    # sim_deep.look_for_survival_nodes()
    # sim_deep.predict_labels()
    sim_deep.load_test_dataset()
    sim_deep.predict_labels_on_test_dataset()


class SimDeep(DeepBase):
    """ """
    def __init__(self,
                 nb_clusters=NB_CLUSTERS,
                 pvalue_thres=PVALUE_THRESHOLD,
                 **kwargs):
        """
        ### AUTOENCODER PARAMETERS ###:
            dataset=None      ExtractData instance (load the dataset),
            f_matrix_train=None
            f_matrix_test=None
            f_matrix_train_out=None
            f_matrix_test_out=None

            level_dims = [500]
            level_dnn = [500]
            new_dim = 100
            dropout = 0.5
            act_reg = 0.0001
            w_reg = 0.001
            data_split = 0.2
            activation = 'tanh'
            nb_epoch = 10
            loss = 'binary_crossentropy'
            optimizer = 'sgd'
        """
        self.nb_clusters = nb_clusters
        self.pvalue_thres = pvalue_thres
        self.classifier_grid = CLASSIFIER
        self.classifier = None

        self.valid_node_ids = None
        self.activities = None

        self.test_labels = None
        self.test_labels_proba = None

        DeepBase.__init__(self, **kwargs)

    def fit(self):
        """
        main function
        construct an autoencoder, predict nodes linked with survival
        and do clustering
        """
        self.construct_autoencoder()
        self.look_for_survival_nodes()
        self.predict_labels()

    def predict_labels_on_test_dataset(
            self,
            return_proba=False,
            mad_scale=True,
            robust_scale=True,
            min_max_scale=False,
            unit_norm=False):
        """
        predict labels of the test set

        argument:
            :return_proba: Bool    if true return class probabilities instead
            :mad_scale: Bool (default True)    use the mad scaler normalisation
            robust_scale: Bool (default True)    use the robust scale transformation
            min_max_scale: Bool (default False)    use min max transformation
            unit_norm: Bool (default False)    use the unit norm normalization

        return:
            class labels (array), Cox-PH P-value
        """
        test_matrix = self.dataset.matrix_test
        ref_matrix = self.dataset.matrix_ref

        ref_matrix, test_matrix = self.dataset.transform_matrices(
            ref_matrix,
            test_matrix,
            mad_scale=mad_scale,
            robust_scale=robust_scale,
            min_max_scale=min_max_scale,
            unit_norm=unit_norm)

        data_type = self.dataset.data_type_test
        nbdays, isdead = self.dataset.survival_test.T.tolist()

        print 'feature selection...'
        embbed = SELECT_FEATURES_METHOD[data_type]

        train_matrix = embbed.fit_transform(ref_matrix, self.labels)
        test_matrix = embbed.transform(test_matrix)

        self._do_classification(train_matrix, test_matrix, self.labels)

        pvalue = coxph(self.test_labels, isdead, nbdays)

        print 'Cox-PH p-value (Log-Rank) for inferred labels: {0}'.format(pvalue)

        labels = self.test_labels_proba if return_proba else self.test_labels

        return labels, pvalue

    def _do_classification(self, train_matrix, test_matrix, labels):
        """ """
        print 'classification analysis...'

        self.classifier_grid.fit(train_matrix, labels)
        self.classifier, params = select_best_classif_params(self.classifier_grid)
        print 'best params:', params

        cvs = cross_val_score(self.classifier, train_matrix, labels, cv=5)
        print 'cross val score: {0}'.format(np.mean(cvs))
        self.classifier.set_params(probability=True)
        self.classifier.fit(train_matrix, labels)

        print 'classification score:', self.classifier.score(train_matrix, labels)

        self.test_labels = self.classifier.predict(test_matrix)
        self.test_labels_proba = self.classifier.predict_proba(test_matrix)

        print '#### report of assigned cluster:'
        for key, value in Counter(self.test_labels).items():
            print 'class: {0}, number of samples :{1}'.format(key, value)

    def predict_activities(self, input_matrix=None):
        """
        predict activities of the nodes linked to survial
        if an input matrix is given, the activities are computed
        using this matrix instead of the train matrix

            :input_matrix: numpy matrix (must have the same dim that the train matrix)
        """
        if not input_matrix:
            input_matrix = self.matrix_train

        return self.encoder.predict(input_matrix).T[self.valid_node_ids].T

    def predict_labels(self):
        """
        predict labels from training set
        using K-Means algorithm on the node activities,
        using only nodes linked to survival
        """
        clustering = KMeans(n_clusters=self.nb_clusters, n_init=100)

        if not self.activities.any():
            raise Exception('No components linked to survial!'\
                            ' cannot perform clustering')

        clustering.fit(self.activities)
        self.labels = clustering.labels_

        self._order_labels_according_to_survival()

        print "clustering done, labels ordered according to survival:"
        for key, value in Counter(self.labels).items():
            print 'cluster label: {0}\t number of samples:{1}'.format(key, value)

        print '\n'

        return self.labels

    def _order_labels_according_to_survival(self):
        """
        Order cluster labels according to survival
        """
        days, dead = np.asarray(self.dataset.survival).T

        label_dict = {}

        for label in set(self.labels):
            mean = surv_mean(dead[self.labels == label],
                             days[self.labels == label])
            label_dict[label] = mean

        label_ordered = [label for label, mean in
                         sorted(label_dict.items(), key=lambda x:x[1])]

        label_dict = {old_label: new_label
                      for new_label, old_label in enumerate(label_ordered)}

        labels = self.labels.copy()

        for old_label in label_dict:
            labels[self.labels == old_label] = label_dict[old_label]

        self.labels = labels

    def look_for_survival_nodes(self):
        """
        detect nodes from the autoencoder significantly
        linked with survival through coxph regression
        """
        self.activities = self.encoder.predict(self.matrix_train)

        self._look_for_survival_nodes()

    def _look_for_survival_nodes(self):
        """
        """
        nbdays, isdead = self.dataset.survival.T.tolist()
        pvalue_list = []

        for node_id, activity in enumerate(self.activities.T):
            activity = activity.tolist()
            pvalue = coxph(activity, isdead, nbdays)
            pvalue_list.append((node_id, pvalue))

        pvalue_list = filter(lambda x: not np.isnan(x[1]), pvalue_list)
        pvalue_list.sort(key=lambda x:x[1], reverse=True)

        self.valid_node_ids = [node_id for node_id, pvalue in pvalue_list
                               if pvalue < self.pvalue_thres]

        print 'number of components linked to survival found:', len(self.valid_node_ids)

        self.activities = self.activities.T[self.valid_node_ids].T


if __name__ == "__main__":
    main()
