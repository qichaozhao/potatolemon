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

    # Test that two layers are created
    assert len(nn.layers) == 2

    # Test that the first layer has five neurons
    assert len(nn.layers[0].neurons) == 5

    # Test that the second layer has 1 neuron
    assert len(nn.layers[1].neurons) == 1

    # Test the forward propagation function is equivalent to manually propagating through the network
    ipt = np.array([1, 1, 1, 1, 1])
    nn_fwd = nn.forward(ipt)
    man_fwd = nn.layers[1].forward(nn.layers[0].forward(ipt))
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
    pred = np.array([0.0, 0.5, 1.0]).reshape((1, 3))
    actual = np.array([1.0, 1.0, 1.0]).reshape((1, 3))

    # Manually calculate an expected result
    pred_adj = np.array([1e-15, 0.5, (1 - 1e-15)]).reshape((1, 3))
    expected_results = -1 / 3 * np.sum(actual * np.log(pred_adj))

    # Assert equals
    assert loss.categorical_crossentropy(pred, actual) == expected_results