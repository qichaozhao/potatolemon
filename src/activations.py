"""
This file contains all of our activation functions.
"""
import numpy as np

def tanh(z, direction='forward', dp=None):
    """
    This function implements the tanh function and returns the result. It can operate on matrixes.

    :param z:
    :param direction:
    :param dp:
    :return:
    """

    a = np.tanh(z)

    if direction == 'forward':
        return a

    elif direction == 'backward':
        return (1 - np.square(a)) * dp

    else:
        raise Exception('Parameter "direction" must take values of { "forward", "backward" }')


def sigmoid(z, direction='forward', dp=None):
    """
    This function implements the logistic function and returns the result. It can operate on matrixes.

    if direction == 'forward' (forward propagation)
        :param z: a matrix of dimension (n, t)
            - n is the number of nodes (usually 1 in our case)
            - t is the number of training examples

        :return: sigmoid(input)

    if direction == 'backward' (backward propagation)
        :param z: a matrix of dimension (i, t)
        :param dp: a matrix of dimension (1, t)
    """

    a = 1 / (1 + np.exp(-z))

    if direction == 'forward':
        return a

    elif direction == 'backward':
        return a * (1 - a) * dp

    else:
        raise Exception('Parameter "direction" must take values of { "forward", "backward" }')


def softmax(z, direction='forward', dp=None):
    """
    Compute the softmax of an output.

    NOTE: The backwards should not really need to be used, as the softmax is mainly used with the cross entropy activation
    function and the backwards step already backpropagates across softmax. Also, the backprop over softmax is not tested.

    if direction == 'forward' (forward propagation)
        :param z: a matrix of dimension (m, t)
            - m is the number of classes
            - t is the number of training examples

        :return: softmax(input)

    if direction == 'backward' (backward propagation)
        :param z: a matrix of dimension (m, t)
        :param dp: a matrix of dimension (m, t) where m is the number of classes

        :return: the backpropagated softmax
    """

    e = np.exp(z - np.max(z))
    a = e / np.sum(e, axis=0)

    if direction == 'forward':
        return a

    elif direction == 'backward':

        num_cls, num_train = dp.shape

        res = np.zeros_like(dp)

        # We need to calculate for each training example separately
        for t_idx in range(num_train):
            # make the matrix whose size is n^2.
            a_tmp = a[:, t_idx]
            jacobian_m = np.diag(a_tmp)

            for i in range(len(jacobian_m)):
                for j in range(len(jacobian_m)):
                    if i == j:
                        jacobian_m[i][j] = a_tmp[i] * (1 - a_tmp[i])
                    else:
                        jacobian_m[i][j] = -a_tmp[i] * a_tmp[j]

            # Now we have the jacobian, we can successfully create our backpropagated output
            # Remember the formula is dJ/dZi = sum over k classes (dJ / dSk * dSk / dSi)
            # First we build the output for the case where i == j
            # Take the diagonal from the jacobian and multiply with our backpropagated cost function
            res[:, t_idx] = np.multiply(dp[:, t_idx].reshape((num_cls, 1)), np.diag(jacobian_m).reshape((num_cls, 1)))

            # Now we calculate the second part of the derivative and add it on
            for i in num_cls:
                for j in num_cls:

                    if i == j:
                        pass

                    else:
                        res[i, t_idx] = res[i, t_idx] + dp[j, t_idx]

        # We should be left with a matrix of size (m, t) which we return
        return res


    else:
        raise Exception('Parameter "direction" must take values of { "forward", "backward" }')
