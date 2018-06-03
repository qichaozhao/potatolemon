"""
This file contains our optimiser equations
"""

def sgd(val, update, learning_rate):
    """
    This performs a stochastic gradient descent update on the passed in values with the passed in learning rate

    :param val: A vector or matrix (e.g. of weights, or biases)
    :param update: A delta calculated from backpropagation
    :param learning_rate: A multiplier on the update value

    :return: A vector or matrix of updated weights or biases
    """

    return val - update * learning_rate