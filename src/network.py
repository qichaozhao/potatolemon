"""
The top most class that we use to build a Neural Network
"""
import numpy as np

from .loss import *
from .layer import Layer
from .activations import sigmoid
from .neuron import Neuron

class Network(object):

    def __init__(self, input_dim, hidden_layer_dim, optimiser=None, neuron_type=Neuron, activation=sigmoid):

        self.input_dim = input_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.optimiser = optimiser
        self.neuron_type = neuron_type
        self.activation = activation

        self.layers = []
        for idx, dim in enumerate(hidden_layer_dim):

            # For the first hidden layer, the input dimension should be the overall input dimension
            if idx == 0:
                self.layers.append(Layer(dim, self.input_dim, type=self.neuron_type, activation=self.activation))

            # For all other hidden layers, the layer size will use the previous layer size as input size, and the layer size specified in the config
            else:
                self.layers.append(Layer(dim, hidden_layer_dim[idx - 1], type=self.neuron_type, activation=self.activation))

    def forward(self, input):
        """
        The forward function which runs a forward pass on the entire network

        :param input: A column vector of length input_dim
        :return: A column vector representing the output of the final layer
        """

        res = input
        for layer in self.layers:
            res = layer.forward(res)

        return res

    def backward(self, loss):
        """
        The backward function that runs one backward propagation pass on the entire network

        :param loss:
        :return:
        """
        raise NotImplementedError

    def predict(self, input):
        """
        Relies on the forward function to make inference

        :param input: A column vector of length input_dim
        :return: A column vector representing the output of the final layer
        """
        return self.forward(input)

    def fit(self, input, target):
        """
        Trains the neural network given a set of data

        :param input: A column vector of length input_dim
        :param target: A target vector of length input_dim
        :return:
        """
        raise NotImplementedError