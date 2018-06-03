"""
Test each of our functions individually to make sure they work!
"""

import sys

sys.path.insert(0, '../')

from src import activations
from src import layer
from src import network
from src import neuron
from src import loss

import numpy as np

np.random.seed(1)


def test_sigmoid_activation():
    """
    Test our sigmoid function

    :return:
    """

    z = np.array([[-np.inf, 0, np.inf],
                  [np.inf, 0, -np.inf]])

    np.testing.assert_array_equal(activations.sigmoid(z), np.array([[0.0, 0.5, 1.0],
                                                                    [1.0, 0.5, 0.0]]))


def test_softmax_activation():
    """
    Test our softmax activation

    :return:
    """
    z = np.array([[1, 2, 3], [1, 2, 3]])

    np.testing.assert_array_equal(activations.softmax(z), np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))


def test_neuron_forward():
    """
    Test our basic neuron with a sigmoid activation function

    :return:
    """

    nron = neuron.Neuron(5)

    # Test get weights
    assert nron.get_weights().shape == (1, 5)

    # Test set weights
    weights = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
    nron.set_weights(weights)
    np.testing.assert_array_equal(weights, nron.weights)

    # Test forward prop step
    ipt = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    np.testing.assert_array_equal(activations.sigmoid(np.dot(weights, ipt)), nron.forward(ipt))


def test_basic_layer_forward():
    """
    Test our basic layer with basic neurons

    :return:
    """

    lyr = layer.Layer(2, 5)

    # Test that the neurons have been created properly
    assert len(lyr.neurons) == 2

    # Test that we can get the right weight matrix out of the neurons
    assert lyr.get_weights().shape == (2, 5)

    # Test that we can set the right weight matrix
    weights = np.ones((2, 5))
    lyr.set_weights(weights)
    np.testing.assert_array_equal(weights, lyr.get_weights())

    # Test that the forward propagation step works
    ipt = np.array([[1], [1], [1], [1], [1]])
    np.testing.assert_array_equal(activations.sigmoid(np.dot(weights, ipt) + 0), lyr.forward(ipt))


def test_basic_network_forward():
    """
    Test our basic network comprised of basic layers.

    We use a one hidden layer network for this test. In this case, the network will have architecture (5, 1)

    :return:
    """

    nn = network.Network(5, [5, 1])

    # Test that three layers are created (two hidden layers + 1 output layer)
    assert len(nn.layers) == 3

    # Test that the first layer has five neurons
    assert len(nn.layers[0].neurons) == 5

    # Test that the second layer has 1 neuron
    assert len(nn.layers[1].neurons) == 1

    # Test that the output layer has 1 neuron
    assert len(nn.layers[2].neurons) == 1

    # Test the forward propagation function is equivalent to manually propagating through the network
    ipt = np.array([1, 1, 1, 1, 1])
    nn_fwd = nn.forward(ipt)
    man_fwd = nn.layers[2].forward(nn.layers[1].forward(nn.layers[0].forward(ipt)))
    np.testing.assert_array_equal(nn_fwd, man_fwd)

    # Test that the predict function performs forward propagation as expected
    np.testing.assert_array_equal(nn.forward(ipt), nn.predict(ipt))


def test_binary_crossentropy():
    """
    Test the binary cross entropy loss function. We pass a number of prediction, actual pairs and check the results.

    :return:
    """

    # Set up our inputs
    pred = np.array([0.0, 0.5, 1.0]).reshape((1, 3))
    actual = np.array([1.0, 1.0, 1.0]).reshape((1, 3))

    # Manually calculate an expected result
    pred_adj = np.array([1e-15, 0.5, (1 - 1e-15)]).reshape((1, 3))
    expected_results = -1 / 3 * np.sum(actual * np.log(pred_adj))

    # Assert equals
    assert loss.binary_crossentropy(pred, actual) == expected_results


def test_categorical_crossentropy():
    """
    Test the categorical cross entropy loss function. We pass a number of prediction, actual pairs and check the results.

    :return:
    """

    # Set up our inputs
    pred = np.array([0.0, 0.5, 1.0]).reshape((3, 1))
    actual = np.array([1.0, 1.0, 1.0]).reshape((3, 1))

    # Manually calculate an expected result
    pred_adj = activations.softmax(pred)
    expected_results = -1 * np.sum(actual * np.log(pred_adj))

    # Assert equals
    assert loss.categorical_crossentropy(pred, actual) == expected_results


def test_sigmoid_backward():
    """
    Test backward propagation on our sigmoid function

    :return:
    """
    z = np.array([[-np.inf, 0, np.inf], [np.inf, 0, -np.inf]])
    dp = np.array([[0.3, 0.3, 0.3], [0.6, 0.6, 0.6]])

    backwards = activations.sigmoid(z) * (1 - activations.sigmoid(z)) * dp
    np.testing.assert_array_equal(activations.sigmoid(z, direction='backward', dp=dp), backwards)


def test_neuron_backward():
    """
    Test backward propagation on our neuron

    :return:
    """
    nron = neuron.Neuron(5)

    # Forward Prop
    ipt = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    pred = nron.forward(ipt)

    # Backward Prop
    da = loss.binary_crossentropy(pred, np.array([[1, 1]]), direction='backward')
    dp = nron.backward(da)

    # Assert
    assert dp.shape == ipt.shape


def test_layer_backward():
    """
    Test backward propagation on our layer

    :return:
    """
    lyr1 = layer.Layer(2, 5)
    lyr2 = layer.Layer(1, 2)

    # Forward prop
    ipt = np.array([[1], [1], [1], [1], [1]])
    fwd1 = lyr1.forward(ipt)
    fwd2 = lyr2.forward(fwd1)

    # Back prop
    da = loss.binary_crossentropy(fwd2, np.array([[1]]), direction='backward')
    dp2 = lyr2.backward(da)
    dp1 = lyr1.backward(dp2)

    assert dp2.shape == fwd1.shape
    assert dp1.shape == ipt.shape


def binary_crossentropy_backward():
    """
    Test backward propagation on binary crossentropy cost function

    :return:
    """
    actual = np.array([1, 0, 0, 1, 0]).reshape(1, 5)
    pred = np.array([0.9, 0.3, 0.6, 0.9, 0.1]).reshape(1, 5)

    assert loss.binary_crossentropy(pred, actual, direction='backward').shape == actual.shape


def categorical_crossentropy_backward():
    """
    Test backward propagation on categorical crossentropy cost function

    :return:
    """
    actual = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 1]]).reshape(2, 5)
    pred = np.array([[0.9, 0.3, 0.6, 0.9, 0.1], [0.9, 0.3, 0.6, 0.9, 0.1]]).reshape(2, 5)

    assert loss.categorical_crossentropy(pred, actual, direction='backward').shape == actual.shape


def test_network_backward():
    """
    Test backward propagation in the network

    :return:
    """

    #  One forward and backprop cycle (binary crossentropy), with high learning rate
    nn = network.Network(5, [5, 1], learning_rate=100)

    # Set up our inputs (num features x num_examples)
    ipt = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]]).reshape((5, 2))

    #  Set up our ground truths for these training examples (1, num_examples)
    y = np.array([[0, 1]])

    # Make a prediction from the network and check the loss
    pred_1 = nn.predict(ipt)
    loss_1 = loss.binary_crossentropy(pred_1, y)

    # Note the weights
    wts = []
    for lyr in nn.layers:
        wts.append(lyr.get_weights())

    # Do one loop of training
    nn_fwd = nn.forward(ipt)
    nn.backward(pred_1, y)

    # Note the weights
    wts_1 = []
    for lyr in nn.layers:
        wts_1.append(lyr.get_weights())

    # Predict again
    pred_2 = nn.predict(ipt)
    loss_2 = loss.binary_crossentropy(pred_2, y)

    # Assert weights are different
    assert not np.array_equal(wts, wts_1)

    # Assert predictions are different and the losses are different
    assert not np.array_equal(pred_2, pred_1)
    assert loss_2 != loss_1