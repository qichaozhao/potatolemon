"""
This file contains all of our activation functions.
"""
import numpy as np

def sigmoid(z):
    """
    This function implements the logistic function and returns the result. It can operate on vectors.

    :param z: A vector of dimension (m, 1)
    :return: sigmoid(input)
    """

    return 1 / (1 + np.exp(-z))
