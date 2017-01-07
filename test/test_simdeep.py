import unittest
import warnings

import numpy as np

from simdeep.config import ACTIVATION
from simdeep.config import OPTIMIZER
from simdeep.config import LOSS


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
            survival = importr('survival')

    def test_3_coxph_function(self):
        """test if the coxph function works """
        from simdeep.coxph_from_r import coxph
        pvalue = coxph(
            np.random.randint(0,2,50),
            np.random.randint(0,2,50),
            np.random.randint(50,365,50),
        )
        self.assertTrue(isinstance(pvalue, float))

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

    def test_5_simdeep_load_full_model(self):
        """
        test if simdeep can load the full model
        """
        from simdeep.simdeep_analysis import SimDeep

        simDeep = SimDeep()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simDeep.load_encoder('encoder_seed_s0_full.h5')

        self.assertTrue(not isinstance(simDeep.encoder, type(None)))


if __name__ == "__main__":
    unittest.main()
