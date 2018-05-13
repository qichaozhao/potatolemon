"""
A layer is a collection of Neurons
"""
import numpy as np

from .neuron import Neuron
from .activations import sigmoid

class Layer(object):

    def __init__(self, size, num_inputs, type=Neuron, activation=sigmoid):
        """
        :param size: The number of neurons in the layer
        :param num_inputs: The number of neurons or inputs in the previous layer
        :type: The type of neuron to use
        """

        self.size = size
        self.num_inputs = num_inputs
        self.neuron_type = type
        self.activation = activation

        # Create all the neurons in this layer
        self.neurons = []
        for i in range(size):
            self.neurons.append(Neuron(self.num_inputs, activation=self.activation))

    def get_weights(self):
        """
        We have a list of neuron objects with their associated weights.

        For each item in the list, the shape will be a vector of length self.num_inputs

        Therefore, we should concatenate these weights together, so that the layer weights will be (self.num_inputs, self.size)

        :return: A matrix of shape (self.num_inputs, self.size)
        """

        weights = np.zeros((self.num_inputs, self.size))

        for idx, neuron in enumerate(self.neurons):
            weights[:, idx] = neuron.get_weights()

        return weights

    def set_weights(self, weights):
        """
        Decomposes the weights matrix into a list of vectors to store into the Neuron weights.

        :param weights: A matrix of shape (self.num_inputs, self.size)
        :return:
        """

        for idx, neuron in enumerate(self.neurons):
            neuron.set_weights(weights[:, idx])

    def forward(self, input):
        """
        Performs a forward pass step, calculating the result of all neurons.

        :param input: A vector of length self.num_inputs (from the previous layer or the overall input)
        :return: A vector of length self.size (i.e. the result of the equation sigmoid(W^T.x + b))
        """

        # In a more performant network, we should do a direct matrix multiplication for all Neurons
        # But in our slower version we rely on the per neuron forward function to retrieve our forward propagation result
        res = []
        for idx, neuron in enumerate(self.neurons):
            res.append(neuron.forward(input))

        return np.asarray(res)



