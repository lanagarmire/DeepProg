import numpy as np

from simdeep.config import SEED

if SEED:
    np.random.seed(SEED)

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input

from keras.models import Sequential
from keras.models import load_model
from keras.models import Model

from keras.regularizers import ActivityRegularizer
from keras.regularizers import WeightRegularizer

from simdeep.extract_data import LoadData

from time import time

from simdeep.config import NB_EPOCH
from simdeep.config import LEVEL_DIMS
from simdeep.config import NEW_DIM
from simdeep.config import LOSS
from simdeep.config import OPTIMIZER
from simdeep.config import ACT_REG
from simdeep.config import W_REG
from simdeep.config import DROPOUT
from simdeep.config import ACTIVATION
from simdeep.config import PATH_MODEL
from simdeep.config import DATA_SPLIT

from os.path import isfile


def main():
    """ """
    simdeep = DeepBase()
    simdeep.load_training_dataset()
    simdeep.construct_autoencoders()


class DeepBase():
    """ """
    def __init__(self,
                 dataset=LoadData(),
                 nb_epoch=NB_EPOCH,
                 level_dims=LEVEL_DIMS,
                 new_dim=NEW_DIM,
                 loss=LOSS,
                 optimizer=OPTIMIZER,
                 act_reg=ACT_REG,
                 w_reg=W_REG,
                 dropout=DROPOUT,
                 data_split=DATA_SPLIT,
                 activation=ACTIVATION,
                 path_model=PATH_MODEL):
        """
        ### DEFAULT PARAMETER ###:
            dataset=None      ExtractData instance (load the dataset),
            level_dims = [500]
            new_dim = 100
            dropout = 0.5
            act_reg = 0.0001
            w_reg = 0.001
            data_split = 0.2
            activation = 'tanh'
            nb_epoch = 10
            loss = 'binary_crossentropy'
            optimizer = 'sgd'
            path_model where to save/load the models
        """
        self.dataset = dataset

        self.matrix_array_train = {}

        self.nb_epoch = nb_epoch
        self.level_dims = level_dims
        self.new_dim = new_dim
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout
        self.path_model = path_model
        self.activation = activation
        self.data_split = data_split

        self.W_l1_constant = w_reg
        self.A_l2_constant = act_reg

        self.encoder_array = {}
        self.model_array = {}

    def construct_autoencoders(self):
        """
        main class to create the autoencoder
        """
        self.create_autoencoders()
        self.compile_models()
        self.fit_autoencoders()

    def load_training_dataset(self):
        """
        load training dataset and surival
        """
        self.dataset.load_array()
        self.dataset.load_survival()
        self.dataset.normalize_training_array()

        self.matrix_array_train = self.dataset.matrix_array_train

    def load_test_dataset(self):
        """
        load test dataset and test surival
        """
        self.dataset.load_matrix_test()
        self.dataset.load_survival_test()

    def create_autoencoders(self):
        """ """
        for key in self.matrix_array_train:
            self._create_autoencoder(self.matrix_array_train[key], key)

    def _create_autoencoder(self, matrix_train, key):
        """
        Instantiate the  autoencoder architecture
        """
        print('creating autoencoder...')
        t = time()

        model = Sequential()

        X_shape = matrix_train.shape

        nb_hidden = 0

        for dim in self.level_dims:
            nb_hidden += 1
            model = self._add_dense_layer(
                model,
                X_shape,
                dim,
                name='hidden layer nb:{0}'.format(nb_hidden))

            if self.dropout:
                model.add(Dropout(self.dropout))

        model = self._add_dense_layer(
                model,
                X_shape,
                dim,
            name='new dim')

        if self.dropout:
            model.add(Dropout(self.dropout))

        for dim in reversed(self.level_dims):
            nb_hidden += 1
            model = self._add_dense_layer(
                model,
                X_shape,
                dim,
                name='hidden layer nb:{0}'.format(nb_hidden))

            if self.dropout:
                model.add(Dropout(self.dropout))

        model = self._add_dense_layer(
            model,
            X_shape,
            X_shape[1],
            name='final layer')

        self.model_array[key] = model

        print('model for {1} created in {0}s !'.format(time() - t, key))

    def _add_dense_layer(self, model, shape, dim, name=None):
        """
        private function to add one layer
        """
        input_dim = None

        if not model.layers:
            input_dim = shape[1]

        model.add(Dense(dim,
                        activity_regularizer=ActivityRegularizer(
                        l2=self.A_l2_constant),
                        W_regularizer=WeightRegularizer(
                            l1=self.W_l1_constant),
                        name=name,
                        activation=self.activation,
                        input_dim=input_dim))
        return model

    def compile_models(self):
        """
        define the optimizer and the loss function
        compile the model and ready to fit the data!
        """
        for key in self.model_array:
            model = self.model_array[key]
            print('compiling deep model...')
            model.compile(optimizer=self.optimizer, loss=self.loss)

            print('compilation done for key {0}!'.format(key))

    def fit_autoencoders(self):
        """
        fit the autoencoder using the training matrix
        """
        for key in self.model_array:
            model = self.model_array[key]
            matrix_train = self.matrix_array_train[key]

            model.fit(x=matrix_train,
                       y=matrix_train,
                       verbose=2,
                       nb_epoch=self.nb_epoch,
                       validation_split=self.data_split,
                       shuffle=True)
            print('fitting done for model {0}!'.format(key))

        self._define_encoders()

    def _define_encoders(self):
        """ """
        for key in self.model_array:
            model = self.model_array[key]
            matrix_train = self.matrix_array_train[key]

            X_shape = matrix_train.shape

            inp = Input(shape=(X_shape[1],))

            encoder = model.layers[0](inp)

            for layer in model.layers[1:]:
                encoder = layer(encoder)
                if layer.name == 'new dim':
                    break

            encoder = Model(inp, encoder)
            self.encoder_array[key] = encoder

    def save_encoders(self, fname='encoder.h5'):
        """
        Save a keras model in the self.path_model directory
        :fname: str    the name of the file to save the model
        """
        for key in self.encoder_array:
            encoder = self.encoder_array[key]
            encoder.save('{0}/{1}_{2}'.format(self.path_model, key, fname))
            print('model saved for key:{0}!'.format(key))

    def load_encoders(self, fname='encoder.h5'):
        """
        Load a keras model from the self.path_model directory
        :fname: str    the name of the file to load
        """
        for key in self.matrix_array_train:
            file_path = '{0}/{1}_{2}'.format(self.path_model, key, fname)
            assert(isfile(file_path))
            t = time()
            encoder = load_model(file_path)
            print('model {1} loaded in {0} s!'.format(time() - t, key))
            self.encoder_array[key] = encoder


if __name__ == "__main__":
    main()
