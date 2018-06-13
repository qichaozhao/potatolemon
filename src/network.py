"""
The top most class that we use to build a Neural Network
"""
import numpy as np

from .loss import *
from .layer import Layer
from .activations import sigmoid, softmax, tanh, relu
from .neuron import Neuron
from .optimisers import sgd


class Network(object):
    def __init__(self, input_dim, hidden_layer_dim, num_classes=2, optimiser=sgd, neuron_type=Neuron,
                 activation=sigmoid, loss=binary_crossentropy, learning_rate=0.01):

        self.input_dim = input_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.optimiser = optimiser
        self.num_classes = num_classes
        self.neuron_type = neuron_type
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate

        self.layers = []
        for idx, dim in enumerate(hidden_layer_dim):

            # For the first hidden layer, the input dimension should be the overall input dimension
            if idx == 0:
                self.layers.append(Layer(dim, self.input_dim, type=self.neuron_type, activation=self.activation,
                                         learning_rate=self.learning_rate, optimiser=self.optimiser))

            # For all other hidden layers, the layer size will use the previous layer size as input size, and the layer size specified in the config
            else:
                self.layers.append(
                    Layer(dim, hidden_layer_dim[idx - 1], type=self.neuron_type, activation=self.activation,
                          learning_rate=self.learning_rate, optimiser=self.optimiser))

        if self.num_classes <= 2:
            # We add an extra logistic layer with one neuron.
            self.layers.append(
                Layer(1, hidden_layer_dim[-1], type=self.neuron_type, activation=sigmoid,
                      learning_rate=self.learning_rate, optimiser=self.optimiser))
        else:
            # We add an extra logistic layer with neurons for the number of classes
            self.layers.append(
                Layer(self.num_classes, hidden_layer_dim[-1], type=self.neuron_type, activation=sigmoid,
                      learning_rate=self.learning_rate, optimiser=self.optimiser))

    def forward(self, input):
        """
        The forward function which runs a forward pass on the entire network

        :param input: A matrix of shape (input_dim, training_examples) that are the features
        :return: A column vector representing the output of the final layer
        """

        res = input
        for layer in self.layers:
            res = layer.forward(res)

        return res

    def backward(self, pred, target):
        """
        The backward function that runs one backward propagation pass on the entire network

        :param target: The ground truth
        :param pred: The predicted value
        :return:
        """

        # First we compute the backward pass on the loss
        dp = self.loss(pred, target, direction='backward')

        # Now we can propagate this through all of our layers
        for layer in reversed(self.layers):
            dp = layer.backward(dp)

    def predict(self, input, proba=False, proba_thresh=0.5):
        """
        Relies on the forward function to make inference. If we have more than two classes, softmax the prediction

        :param input: A matrix of shape (num_features, num_examples)
        :param proba: If True then return probabilities, else just return labels (0 to 1)
        :param proba_thresh: Threshold for probability, by default 0.5.
        :return: A matrix representing the output of the final layer (c, num_examples), where c is the class label
        """

        fwd = self.forward(input)

        if self.num_classes > 2:

            s = softmax(fwd)
            if not proba:
                return np.argmax(s, axis=0)

            else:
                return s
        else:
            if not proba:
                fwd[fwd >= proba_thresh] = 1
                fwd[fwd < proba_thresh] = 0

                return fwd

            else:
                return fwd

    def fit(self, input, target, epochs=100, batch_size=None, verbose=False):
        """
        Trains the neural network given a set of data

        :param input: A matrix vector of shape (num_features, num_examples)
        :param target: A matrix vector of shape (1, num_examples) for binary classification,
            and (num_classes, num examples) for multiclass classification.
        :param num_epochs: How many times to iterate over the training data set
        :param batch_size: Whether to use minibatches, if set to None will use the whole training set
        :return:
        """

        losses = []

        if batch_size is not None:
            raise NotImplementedError('We cannot do mini-batch training right now!')

        for i in range(0, epochs):
            # Forward propagation and calculate losses
            pred = self.forward(input)
            loss = self.loss(pred, target)

            # Save losses for printing later
            losses.append(loss)

            # Backward propagate to update weights
            self.backward(pred, target)

            if verbose:
                print('Loss after epoch {}: {}'.format(i, loss))

        return losses
