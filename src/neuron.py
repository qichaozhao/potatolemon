"""
The Neuron class.

Holds its own weights and contains getters and setters for it.

Also contains the forward function, used to "activate" the neuron.
"""

import numpy as np

from .activations import sigmoid


class Neuron(object):


    def __init__(self, num_inputs, activation=sigmoid):

        self.activation = activation
        self.num_inputs = num_inputs
        self.weights = np.random.randn(1, num_inputs) * 0.01
        self.bias = 0

    def forward(self, input):
        """
        In this function we implement the equation y = f(W.X + b)

        :param input: a column vector of length (i, m)
            i: num weights, or rather the number of input nodes
            m: num training examples
        :return: a vector of shape (1, m) where m is the number of training examples
        """
        return self.activation(np.dot(self.weights, input) + self.bias)

    def get_weights(self):
        """
        Return the weights

        :return: a weight vector of length num_inputs
        """
        return self.weights

    def set_weights(self, weights):
        """
        Update the weights

        :return:
        """
        self.weights = weights.reshape(1, self.num_inputs)