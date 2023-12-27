# basic imports
import numpy as np


class Perceptron:
    '''
    This class implements the Perceptron algorithm.
    '''

    def __init__(self):
        self.weights = None
        self.current_training_error = None
        self.training_errors = None

    def get_count_miss_classification(self, X, Y):
        wrongs = 0
        for x, y in zip(X, Y):
            y_hat = -1 if np.dot(self.weights, x) <= 0 else 1
            if y * y_hat == -1:
                wrongs += 1
        return wrongs

    def train(self, X, Y, learning_rate, max_epochs, max_error):
        '''
        This function trains the Perceptron's weights until it either converges or the maximum number of epochs is exceeded.

        | **Args**
        | X:                            The data samples.
        | Y:                            The data labels/targets.
        | learning_rate:                The learning rate with which the Perceptron's weights will be updated.
        | max_epochs:                   The maximum number of epochs that the Perceptrin will be trained.
        | max_error:                    The error threshold below which the Perceptron will be considered to have converged.
        '''
        # !!! implement your solution here !!!
        self.weights = np.zeros(X.shape[1])
        self.training_errors = []
        self.current_training_error = self.get_count_miss_classification(X, Y)

        for epoch in range(max_epochs):
            tmp = list(zip(X, Y))
            np.random.shuffle(tmp)
            X, Y = zip(*tmp)
            X, Y = list(X), list(Y)
            for x, y in zip(X, Y):
                y_hat = -1 if np.dot(self.weights, x) <= 0 else 1
                if y * y_hat == -1:
                    self.weights = self.weights + 2 * learning_rate * x * y
                    self.current_training_error = self.get_count_miss_classification(X, Y)
                    self.training_errors.append(self.current_training_error)
                if self.current_training_error <= max_error:
                    print(f"solution found: {self.weights}")
                    return


class PocketPerceptron(Perceptron):  # new class inherits from Perceptron
    '''
    This class implements the Pocket Perceptron algorithm.
    '''

    def __init__(self):
        super().__init__()  # call the initialization function of the superclass Perceptron
        self.weights_in_pocket = None
        self.smallest_training_error = None
        self.smallest_training_errors = []

    def train(self, X, Y, learning_rate, max_epochs, max_error):
        '''
        This function trains the Perceptron's weights until it either converges or the maximum number of epochs is exceeded.

        | **Args**
        | X:                            The data samples.
        | Y:                            The data labels/targets.
        | learning_rate:                The learning rate with which the Perceptron's weights will be updated.
        | max_epochs:                   The maximum number of epochs that the Perceptrin will be trained.
        | max_error:                    The error threshold below which the Perceptron will be considered to have converged.
        '''
        self.weights = np.zeros(X.shape[1])
        self.training_errors = []
        self.current_training_error = self.get_count_miss_classification(X, Y)
        self.smallest_training_error = np.inf

        for epoch in range(max_epochs):
            tmp = list(zip(X, Y))
            np.random.shuffle(tmp)
            X, Y = zip(*tmp)
            X, Y = list(X), list(Y)
            for x, y in zip(X, Y):
                y_hat = -1 if np.dot(self.weights, x) <= 0 else 1
                if y * y_hat == -1:
                    self.weights = self.weights + 2 * learning_rate * x * y
                    self.current_training_error = self.get_count_miss_classification(X, Y)
                    self.training_errors.append(self.current_training_error)
                    if self.current_training_error < self.smallest_training_error:
                        self.smallest_training_error = self.current_training_error
                        self.weights_in_pocket = self.weights
                    self.smallest_training_errors.append(self.smallest_training_error)
                if self.current_training_error <= max_error:
                    print(f"solution found: {self.weights}")
                    return self.weights_in_pocket

