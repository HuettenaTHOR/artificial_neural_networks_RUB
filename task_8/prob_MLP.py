import numpy as np
import matplotlib.pyplot as plt


class MLP():

    def __init__(self, input_dim: int = 2, hidden_layer_units: int = 64, eta: float = 0.005, xavier: bool = False):
        '''
        This class implements a simple multilayer perceptron (MLP) with one hidden layer.
        The MLP is trained with true SGD.

        Parameters
        ----------
        input_dim :                         Number of input units.\n
        hidden_layer_units :                Size of hidden layer.\n
        eta :                               The learning rate.\n
        xavier :                            A flag indicating whether Xavier initialization should be used.\n

        Returns
        ----------
        None\n
        '''
        # input dimensions
        self.input_dim = input_dim
        # number of units in the hidden layer
        self.hidden_layer_units = hidden_layer_units
        # learning rate
        self.eta = eta
        # initialize weights
        if xavier:
            pass
        else:
            self.weights_input_hidden = np.random.normal(0, 1, (self.hidden_layer_units, input_dim + 1))
            self.weights_hidden_output = np.random.normal(0, 1, (1, self.hidden_layer_units + 1))

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 100):
        '''
        Train function of the MLP class.
        This functions trains a MLP using true SGD with a constant learning rate.

        Parameters
        ----------
        X :                                 Training examples.\n
        Y :                                 Ground truth labels.\n
        epochs :                            Number of epochs the MLP will be trained.\n

        Returns
        ----------
        None\n
        '''
        X_full = np.hstack((np.ones((X.shape[0], 1)), X))
        for epoch in epochs:
            pass


    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> float:
        '''
        Test function of the MLP class. This function computes the MSE for all examples in X.

        Parameters
        ----------
        X :                                 Training examples.\n
        Y :                                 Ground truth labels.\n

        Returns
        ----------
        loss :                              The MSE for the given training examples.\n
        '''
        pass


if __name__ == '__main__':
    '''
    Train MLPs as defined in the assignments.
    '''
    # load dataset
    dataset = np.load('xor.npy')
    # prepare training data and labels
    X, y = dataset[:, :2], dataset[:, 2]
    # split into training and test
    train_size = 4000  # out of 5000
    X_train, X_test, Y_train, Y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

