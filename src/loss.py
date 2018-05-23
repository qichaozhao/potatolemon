"""
This file contains all of our cost functions.
"""

import numpy as np


def binary_crossentropy(pred, actual, epsilon=1e-15):
    """
    Calculates the cross entropy loss for a binary classification output (e.g. one output node).

    :param pred: A vector value that is the output from the network of shape (1, training_examples)
    :param actual: A vector value that is the ground truth of shape (1, training_examples)
    :return: A vector value representing the binary_crossentropy loss of shape (1, training_examples)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    m = pred.shape[1]
    J = -1 / m * np.sum(np.dot(actual, np.log(pred.T)) + np.dot((1 - actual), np.log((1 - pred).T)))

    return J


def categorical_crossentropy(pred, actual, epsilon=1e-15):
    """
    Calculates the average entropy loss for a multi class classification output (e.g. multiple output node).

    :param pred: A vector value that is the output from the network of shape (num_classes, training_examples)
    :param actual: A vector value that is the ground truth of shape (num_classes, training_examples)
    :return: A vector value representing the binary_crossentropy loss of shape (1, training_examples)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    i, m = pred.shape

    # We loop through each class and calculate our final cost
    # This could probably be done in a quicker fashion via a matrix multiplication but for simplicity's sake we use a loop
    J = 0
    for cls in range(i):

        pred_cls = pred[cls, :].T
        actual_cls = actual[cls, :]

        J = J + (- 1 / m * np.sum(np.dot(actual_cls, np.log(pred_cls))))

    return J