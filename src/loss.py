"""
This file contains all of our cost functions.
"""

import numpy as np

from .activations import softmax


def binary_crossentropy(pred, actual, epsilon=1e-15, direction='forward'):
    """
    Calculates the cross entropy loss for a binary classification output (e.g. one output node).

    if direction == 'forward'
        :param pred: A vector value that is the output from the network of shape (1, t) where t is the number of training examples
        :param actual: A vector value that is the ground truth of shape (1, t) where t is the number of training examples
        :return: A vector value representing the binary_crossentropy loss of shape (1, t) where m is the number of training examples

    if direction == 'backward'
        :param pred: A vector value that is the output from the network of shape (1, t) where t is the number of training examples
        :param actual: A vector value that is the ground truth of shape (1, t) where t is the number of training examples
        :return: A vector value representing the backpropagation from the cost (dJ/d (yhat)), shape (1, t)
    """

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    # m is the number of training examples in this case
    t = pred.shape[1]

    if direction == 'forward':
        J = -1 / t * np.sum(np.dot(actual, np.log(pred.T) + np.dot((1 - actual), np.log((1 - pred).T))))
        return J

    elif direction == 'backward':
        dy = -1 * np.divide(actual - pred, np.multiply(pred, 1 - pred))
        return dy

    else:
        raise Exception('Parameter direction must be only of values { "backward", "forward" }')


def categorical_crossentropy(pred, actual, epsilon=1e-15, direction='forward', use_softmax=True):
    """
    Calculates the average entropy loss for a multi class classification output (e.g. multiple output node).

    This function also by default puts the input through a softmax activation before calculating the loss,
    as this is best practice for multi class classification problems.

    :param epsilon: A small number to avoid zero or 1 errors
    :param use_softmax: Softmax the incoming activation before calculating loss (generally yes)

    if direction == 'forward'
        :param pred: A vector value that is the output from the network of shape (i, t) where i is the number of
            classes, and t is the number of training examples
        :param actual: A vector value that is the ground truth of shape (i, t) where i is the number of classes,
            and t is the number of training examples
        :return: A vector value representing the categorical crossentropy loss of shape (i, t) where i is the
            number of classes, and t is the number of training examples

    if direction == 'backward'
        :param pred: A vector value that is the output from the network of shape (i, t) where i is the number of
            classes, and t is the number of training examples
        :param actual: A vector value that is the ground truth of shape (i, t) where i is the number of classes,
            and t is the number of training examples
        :return: A vector value representing the backpropped loss of shape (i, t) where i is the number of classes,
            and t is the number of training examples
    """

    if use_softmax:
        pred = softmax(pred)

    # Here we offset zero and one values to avoid infinity when we take logs.
    pred[pred == 0] = epsilon
    pred[pred == 1] = 1 - epsilon

    # Get the shape of the predictions, useful later
    i, t = pred.shape

    if direction == 'forward':

        # We loop through each class and calculate our final cost
        # This could probably be done in a quicker fashion via a matrix multiplication but for simplicity's sake
        # we use a loop
        J = 0
        for cls in range(i):
            pred_cls = pred[cls, :].T
            actual_cls = actual[cls, :]

            J = J + (- 1 / t * np.sum(np.dot(actual_cls, np.log(pred_cls))))

        return J

    elif direction == 'backward':

        # If we used a softmax activation function the derivative becomes slightly different as we are calculating
        # the derivative over the softmax too.
        if use_softmax:
            dy = pred - actual

        else:
            dy = -1 * np.divide(actual, pred)

        return dy

    else:
        raise Exception('Parameter direction must be only of values { "backward", "forward" }')
