"""
This file contains all of our cost functions.
"""

import numpy as np


def binary_crossentropy(pred, actual, epsilon=1e-15):
    """
    Calculates the cross entropy loss for a binary classification output (e.g. one output node).

    :param pred: A vector value that is the output from the network of shape (training_examples, 1)
    :param actual: A vector value that is the ground truth of shape (training_examples, 1)
    :return: A vector value representing the binary_crossentropy loss of shape (training_examples, 1)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    m = pred.shape[0]
    J = -1 / m * np.sum(actual * np.log(pred) + (1 - actual) * np.log(1 - pred))

    return J


def categorical_crossentropy(pred, actual, epsilon=1e-15):
    """
    Calculates the average entropy loss for a multi class classification output (e.g. multiple output node).

    :param pred: A vector value that is the output from the network of shape (training_examples, num_classes)
    :param actual: A vector value that is the ground truth of shape (training_examples, num_classes)
    :return: A vector value representing the binary_crossentropy loss of shape (training_examples, 1)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    m, i = pred.shape

    # We loop through each class and calculate our final cost
    # This could probably be done in a quicker fashion via a matrix multiplication but for simplicity's sake we use a loop
    J = 0
    for cls in range(i):

        pred_cls = pred[:, cls]
        actual_cls = actual[:, cls]

        J = J + (- 1 / m * np.sum(actual_cls * np.log(pred_cls)))

    return J