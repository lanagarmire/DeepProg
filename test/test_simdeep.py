import unittest
import warnings

import numpy as np

from simdeep.config import ACTIVATION
from simdeep.config import OPTIMIZER
from simdeep.config import LOSS

from os.path import abspath
from os.path import split

from os.path import isfile
from os.path import isdir

from os import remove
from shutil import rmtree


class TestPackage(unittest.TestCase):
    """ """
    def test_1_rpy2_and_r(self):
        """test that rpy2 is installed and linked to R"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rpy2 import robjects as rob

        var = rob.r('1')[0]

        self.assertTrue(var == 1)

    def test_2_survival_package_from_r(self):
        """test if the survival package is installed in R """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rpy2.robjects.packages import importr
            importr('survival')

    def test_3_coxph_function(self):
        """test if the coxph function works """
        from simdeep.coxph_from_r import coxph

        isdead = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
        nbdays = [24, 10, 25, 50, 14, 10 ,100, 10, 50, 10]
        values = [0, 1, 1, 0 , 1, 2, 0, 1, 0, 0]


        pvalue = coxph(values, isdead, nbdays, isfactor=True)

        self.assertTrue(isinstance(pvalue, float))
        self.assertTrue(pvalue < 0.05)

    def test_4_keras_model_instantiation(self):
        """
        test if keras can be loaded and if that a model
        can be instanciated
        """
        from keras.models import Sequential
        from keras.layers import Dense

        dummy_model = Sequential()
        dummy_model.add(Dense(10, input_dim=20,
                                   activation=ACTIVATION))

        dummy_model.compile(
            optimizer=OPTIMIZER, loss=LOSS)

        Xmat = np.random.random((50,20))
        Ymat = np.random.random((50,10))

        dummy_model.fit(
            x=Xmat,
            y=Ymat)

    def test_5_one_simdeep_instance(self):
        """
        test one simdeep instance
        """
        from simdeep.simdeep_analysis import SimDeep
        from simdeep.extract_data import LoadData

        PATH_DATA = '{0}/../examples/data/'.format(split(abspath(__file__))[0])

        TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv', 'MIR': 'mir_dummy.tsv'}
        SURVIVAL_TSV = 'survival_dummy.tsv'

        PROJECT_NAME = 'TestProject'
        EPOCHS = 3

        deep_model_additional_args = {
        "epochs":EPOCHS, "seed":4}

        dataset = LoadData(path_data=PATH_DATA,
                       survival_tsv=SURVIVAL_TSV,
                       training_tsv=TRAINING_TSV)

        simdeep = SimDeep(dataset=dataset,
                          project_name=PROJECT_NAME,
                          path_results=PATH_DATA,
                          deep_model_additional_args=deep_model_additional_args,
        )
        simdeep.load_training_dataset()
        simdeep.fit()
        simdeep.predict_labels_on_full_dataset()
        simdeep.predict_labels_on_test_fold()

        simdeep.load_new_test_dataset(
            {'RNA': 'rna_test_dummy.tsv'},
            'survival_test_dummy.tsv',
            'dummy')

        simdeep.predict_labels_on_test_dataset()

        path_fig = '{0}/{1}_KM_plot_training_dataset.png'.format(PATH_DATA, PROJECT_NAME)

        self.assertTrue(isfile(path_fig))

        from glob import glob

        for fil in glob('{0}/{1}*'.format(PATH_DATA, PROJECT_NAME)):
            if isfile(fil):
                remove(fil)
            elif isdir(fil):
                rmtree(fil)

    def test_6_simdeep_boosting(self):
        """
        test simdeep boosting
        """
        from simdeep.simdeep_boosting import SimDeepBoosting

        PATH_DATA = '{0}/../examples/data/'.format(split(abspath(__file__))[0])

        TRAINING_TSV = {'RNA': 'rna_dummy.tsv', 'METH': 'meth_dummy.tsv', 'MIR': 'mir_dummy.tsv'}
        SURVIVAL_TSV = 'survival_dummy.tsv'

        PROJECT_NAME = 'TestProject'
        EPOCHS = 3
        SEED = 3
        nb_it = 3
        nb_threads = 2

        boosting = SimDeepBoosting(
            nb_threads=nb_threads,
            nb_it=nb_it,
            survival_tsv=SURVIVAL_TSV,
            training_tsv=TRAINING_TSV,
            path_data=PATH_DATA,
            project_name=PROJECT_NAME,
            path_results=PATH_DATA,
            epochs=EPOCHS,
            normalization={'TRAIN_CORR_REDUCTION':True},
            seed=SEED)

        boosting.partial_fit()
        boosting.predict_labels_on_full_dataset()
        boosting.compute_clusters_consistency_for_full_labels()
        boosting.evalutate_cluster_performance()
        boosting.collect_cindex_for_test_fold()
        boosting.collect_cindex_for_full_dataset()

        boosting.load_new_test_dataset(
            {'RNA': 'rna_test_dummy.tsv'},
            'survival_test_dummy.tsv',
            'dummy',
            normalization={'TRAIN_NORM_SCALE':True},
        )

        boosting.predict_labels_on_test_dataset()
        boosting.predict_labels_on_test_dataset()
        boosting.compute_c_indexes_for_test_dataset()
        boosting.compute_clusters_consistency_for_test_labels()

        from glob import glob

        for fil in glob('{0}/{1}*'.format(PATH_DATA, PROJECT_NAME)):
            if isfile(fil):
                remove(fil)
            elif isdir(fil):
                rmtree(fil)


if __name__ == "__main__":
    unittest.main()
