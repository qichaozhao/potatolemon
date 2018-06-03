"""
The Neuron class.

Holds its own weights and contains getters and setters for it.

Also contains the forward function, used to "activate" the neuron.
"""

import numpy as np

from .activations import sigmoid
from .optimisers import sgd


class Neuron(object):
    def __init__(self, num_inputs, activation=sigmoid, optimiser=sgd, learning_rate=0.01):
        self.num_inputs = num_inputs
        self.activation = activation
        self.optimiser = optimiser
        self.learning_rate = learning_rate

        self.weights = np.random.uniform(0, 1, (1, num_inputs))
        self.bias = np.zeros((1, 1))
        self.input = None
        self.z = None

        # For weight updation
        self.dw = None
        self.db = None

    def forward(self, input):
        """
        In this function we implement the equation A = f(W.X + b)

        :param input: a column vector of length (i, t)
            i: the input size (i.e. number of weights, or num_inputs)
            t: num training examples
        :return: a vector of shape (1, t) where m is the number of training examples
        """

        # We save the neuron input and activation function input to the neuron (X) for backpropagation
        self.input = input
        self.z = np.dot(self.weights, self.input) + self.bias
        return self.activation(self.z)

    def backward(self, da):
        """
        In this function we implement the backwards propagation step for the neuron.

        We will calculate the following equations:

        1. dz = dJ/dz - using our sigmoid backwards function
        2. dw = dJ/dz = 1 / m * (dz . A_prev.T)
        3. db = dJ/db
        4. dp = dJ/da for the neuron connections in the layer l-1

        Afterwards, we will do the weight update (via our optimisation method).

        :param da: This is the value dJ/dA passed in from layer l+1 (shape: (1, t))
        :return: dp, to be used in the downstream backpropagation step (shape: (m, t)) where m is the number of inputs
        """

        # Number of training examples
        t = self.input.shape[1]

        # Calculate our update equations
        dz = self.activation(self.z, direction='backward', dp=da)
        self.dw = 1 / t * np.dot(dz, self.input.T)
        self.db = 1 / t * np.sum(dz)
        dp = np.dot(self.weights.T, dz)

        # Do the updates
        self.weights = self.optimiser(self.weights, self.dw, self.learning_rate)
        self.bias = self.optimiser(self.bias, self.db, self.learning_rate)

        # Pass the backpropagation further downstream
        return dp

    def get_weights(self):
        """
        Return the weights

        :return: a weight matrix of size (1, num_inputs)
        """
        return self.weights

    def set_weights(self, weights):
        """
        Update the weights

        :return:
        """
        self.weights = weights.reshape(1, self.num_inputs)
