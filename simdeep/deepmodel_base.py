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


############ Variable ################
LEVEL_DIMS = [500, 500, 500]
LEVEL_DNN = [500]
NEW_DIM = 100
DROPOUT = 0.5
ACT_REG = 0.0001
W_REG = 0.001
DATA_SPLIT = 0.2
ACTIVATION = 'tanh' # The activation function of the autoencoder

NB_EPOCH = 10
LOSS = 'binary_crossentropy'
# LOSS = 'categorical_crossentropy'
OPTIMIZER = 'sgd'
######################################


def main():
    """ """
    simdeep = DeepBase()
    simdeep.load_training_dataset()
    simdeep.fit()


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
                 activation=ACTIVATION):
        """
        ### DEFAULT PARAMETER ###:
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
            activation_enc = 'tanh'
            nb_epoch = 10
            loss = 'binary_crossentropy'
            optimizer = 'sgd'
        """
        self.dataset = dataset

        self.matrix_train = None

        self.nb_epoch = nb_epoch
        self.level_dims = level_dims
        self.new_dim = new_dim
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout

        self.activation = activation

        self.W_l1_constant = w_reg
        self.A_l2_constant = act_reg

        self.encoder = None
        self.model = None

    def construct_autoencoder(self):
        """
        main class to create the autoencoder
        """
        self.create_autoencoder()
        self.compile_model()
        self.fit_autoencoder()

    def load_training_dataset(self):
        """
        load training dataset and surival
        """
        self.dataset.load_array()
        self.dataset.load_survival()
        self.dataset.normalize()

        self.matrix_train = self.dataset.matrix_stacked

    def load_test_dataset(self):
        """
        load test dataset and test surival
        """
        self.dataset.load_matrix_test()
        self.dataset.load_survival_test()

    def create_autoencoder(self):
        """
        Instantiate the  autoencoder architecture
        """
        print 'creating autoencoder...'
        t = time()

        self.model = Sequential()

        X_shape = self.matrix_train.shape

        for dim in self.level_dims:
            self._add_dense_layer(dim, self.activation)

            if self.dropout:
                self.model.add(Dropout(self.dropout))

        self._add_dense_layer(self.new_dim,
                              self.activation,
                              name='new dim')

        if self.dropout:
            self.model.add(Dropout(self.dropout))

        for dim in reversed(self.level_dims):
            self._add_dense_layer(dim, self.activation)

            if self.dropout:
                self.model.add(Dropout(self.dropout))

        self._add_dense_layer(X_shape[1], self.activation)

        print 'model created in {0}s !'.format(time() - t)

    def _add_dense_layer(self, dim, activation, name=None):
        """
        private function to add one layer
        """
        input_dim = None

        if not self.model.layers:
            X_shape = self.matrix_train.shape
            input_dim = X_shape[1]

        self.model.add(Dense(dim,
                        activity_regularizer=ActivityRegularizer(
                            l2=self.A_l2_constant),
                             W_regularizer=WeightRegularizer(
                                 l1=self.W_l1_constant),
                             name=name,
                             activation=activation,
                             input_dim=input_dim))

    def compile_model(self):
        """
        define the optimizer and the loss function
        compile the model and ready to fit the data!
        """
        print 'compiling deep model...'
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        print 'compilation done!\n summary of the model:\n',self.model.summary()

    def fit_autoencoder(self):
        """
        fit the autoencoder using the training matrix
        """
        self.model.fit(x=self.matrix_train,
                       y=self.matrix_train,
                       verbose=2,
                       nb_epoch=self.nb_epoch,
                       shuffle=True)
        print 'fitting done!'

        self._define_encoder()

    def _define_encoder(self):
        """ """
        X_shape = self.matrix_train.shape

        inp = Input(shape=(X_shape[1],))

        encoder = self.model.layers[0](inp)

        for layer in self.model.layers[1:]:
            encoder = layer(encoder)
            if layer.name == 'new dim':
                break

        self.encoder = Model(inp, encoder)


if __name__ == "__main__":
    main()
