""" Implementation of a multi-layer perceptron, the standard feedforward neural network.

@author: Gabriel G. Nogueira (Talendar)
"""

import numpy as np
import activation_functions


class MultilayerPerceptron:
    """ Implements a multi-layer perceptron, the standard feedforward neural network.

    Attributes:
        _input_size: the length of the input vectors (each of which represents a sample).
        _layers: list of NeuralLayers, i.e., the MLP's layers of neurons; the input layer is not considered in this list.
    """

    def __init__(self, input_size, layers_size=None, layers_activation="sigmoid"):
        """ Builds a new MLP.

        :param input_size: the length of the input vectors (each of which represents a sample).
        :param layers_size: list containing integers, each one specifying the amount of neurons that the respective
        layer will have; alternatively, the layers can be added later, one by one.
        :param layers_activation: the activation function used by all the neurons in the specified layers.
        """
        self._input_size = input_size
        self._layers = []

        if layers_size is not None:
            self._layers.append( NeuralLayer(layers_size[0], self._input_size,
                                             activation_functions.create_by_name(layers_activation)) )

            for i in range(1, len(layers_size)):
                self._layers.append( NeuralLayer(layers_size[i], layers_size[i - 1],
                                                 activation_functions.create_by_name(layers_activation)) )

    def add_layer(self, neural_layer):
        """ Adds the given neural layer to the network.

        :param neural_layer: the layer to be added; an instance of NeuralLayer.
        :raises ValueError: if the layer's input size is incompatible with the network's last layer.
        """
        if neural_layer.input_size() != self._layers[-1].num_neurons():
            raise ValueError("The given layer must expect an input size equal to the number of neurons in the network's"
                             " current last layer (%d)." % self._layers[-1].num_neurons())
        self._layers.append(neural_layer)

    def _feedforward(self, X):
        """ Feeds the given input to the network.

        :param X: numpy 2D-array in which each ROW represents an input sample vector; if you're feeding just one sample,
        pass a row vector representation of it.
        :return: a tuple containing the activations (index 0) and the weighted input (index 1) of all the network's neurons.
        """
        activations = []
        zs = []

        a = X.T
        for layer in self._layers:
            a, z = layer.process(a)
            activations.append(a)
            zs.append(z)

        return activations, zs

    def predict(self, X):
        """ Feeds the given input to the network. Wrapper for the _feedforward method.

        :param X: numpy 2D-array in which each ROW represents an input sample vector; if you're feeding just one sample,
        pass a row vector representation of it.
        :return: the output of the network's last layer when the input is X.
        """
        return self._feedforward(X)[0][-1]

    def _backpropagate(self, X, Y, cost):
        """ Calculates the derivatives of the cost function with respect to the network's weights and biases.

        :param X: matrix with the training samples.
        :param Y: matrix with the labels for the samples.
        :param cost: cost function.
        :return: a tuple (dW, db), where dW and db are lists containing Jacobian matrices with the derivatives of the
        cost function with respect to each of the network's weights (dW) and biases (db).
        """
        # forward pass
        A, Z = self._feedforward(X)
        assert A[-1].shape == (self._layers[-1].num_neurons(), len(X)) == Z[-1].shape

        # backward pass (calculating errors)
        errors = [  # list with the errors δ of each layer; starts with the δ of the output layer
            np.multiply(cost.unit(A[-1], Y, derivative=True), self._layers[-1].activation(Z[-1], derivative=True))
        ]

        for l in range(len(self._layers) - 2, -1, -1):
            error_lp1 = errors[0]
            W_lp1_T = self._layers[l + 1].weights.T
            der_actv_zl = self._layers[l].activation(Z[l], derivative=True)

            errors.insert(0,
                    np.multiply( np.matmul(W_lp1_T, error_lp1), der_actv_zl ))
            assert errors[0].shape == (self._layers[l].num_neurons(), len(X))

        # calculating derivatives (based on the errors)
        dW = [ np.matmul(errors[0], X) / len(X)]
        db = [ np.sum(errors[0], axis=1, keepdims=True) / len(X)]

        for l in range(1, len(errors)):
            e_l, a_lm1_T = errors[l], A[l-1].T
            dW.append(np.matmul(e_l, a_lm1_T) / len(X))
            db.append(np.sum(e_l, axis=1, keepdims=True) / len(X))

        for l in range(len(self._layers)):
            assert self._layers[l].weights.shape == dW[l].shape
            assert self._layers[l].bias.shape == db[l].shape

        return dW, db

    @staticmethod
    def _make_batches(X, Y, size):
        """ Makes training batches from the data.

        :param X: matrix with the training samples.
        :param Y: matrix with the labels of the samples.
        :param size: number of samples in each batch.
        :return: a list containing the training batches.
        """
        indices = list(range(len(X)))
        np.random.shuffle(indices)

        return [ ( X[ indices[i:(i + size)] ],
                   Y[ indices[i:(i + size)] ] )
                 for i in range(0, len(X), size) ]

    def _sgd(self, data, labels, cost, epochs, lr, batch_size, gradient_checking, mt):
        """ Runs the stochastic gradient descent optimization algorithm on the given data.
        
        :param data: matrix with the training samples.
        :param labels: matrix with the labels of the samples.
        :param cost: cost function.
        :param epochs: number of training epochs.
        :param lr: learning rate.
        :param batch_size: number of samples in the mini-batches.
        :param mt: momentum term.
        """""
        print("Initial cost: %.5f\n" % cost(self.predict(data), labels))
        last_deltaW = [None] * len(self._layers)    # stores the last changes made to the weights
        for e in range(epochs):
            batches = self._make_batches(data, labels, batch_size)

            for batch in batches:
                # backpropagation
                X, Y = batch
                dW, db = self._backpropagate(X, Y, cost)

                # gradient checking
                if gradient_checking:
                    dW_check, db_check = self._numerical_gradient(X, Y, cost)
                    print("[W]", end="")
                    self._gradient_checking(dW, dW_check)
                    print("[b]", end="")
                    self._gradient_checking(db, db_check)
                    print()

                # updating weights
                for l in range(len(self._layers)):
                    layer = self._layers[l]
                    deltaW = -lr * dW[l] + (0 if last_deltaW[l] is None else mt*last_deltaW[l])
                    last_deltaW[l] = deltaW

                    layer.weights += deltaW
                    layer.bias -= lr * db[l]

            print("[Epoch %d/%d] Cost: %.5f" % (e+1, epochs, cost(self.predict(data), labels)))

    def fit(self, data, labels, cost_function, epochs, learning_rate, batch_size=32, gradient_checking=False, momentum_term=0):
        """ Fits the model to the given data.

        :param data: numpy 2D-array in which each ROW represents an input sample vector.
        :param labels: numpy 2D-array in which each ROW represents a vector with the samples' labels.
        :param cost_function: cost function to be minimized.
        :param epochs: number of training epochs (iterations).
        :param learning_rate: learning rate of the model.
        :param batch_size: number of samples in each training batch.
        :param gradient_checking: if True, the derivatives of the cost function will also be calculated numerically, in
        order to compare them with the ones obtained through backpropagation; used for tests only.
        :param momentum_term: value for the momentum term (used to speed up the SGD convergence).
        """
        self._sgd(
            data, labels,
            cost=cost_function,
            epochs=epochs,
            lr=learning_rate,
            batch_size=batch_size,
            gradient_checking=gradient_checking,
            mt=momentum_term
        )

    def _numerical_gradient(self, X, Y, cost):
        """ Estimates the network's gradient. Used for tests (gradient checking) only.

        Uses the symmetric derivative definition to estimate the derivatives of the cost function with respect to the
        network's weights.
        """
        h = 1e-5
        dW, db = [], []

        for layer in self._layers:
            grad_W = np.zeros(shape=layer.weights.shape)
            grad_b = np.zeros(shape=layer.bias.shape)

            for i in range(0, len(layer.weights)):
                # biases
                b = layer.bias[i]

                layer.bias[i] = b + h
                inc_cost = cost(self.predict(X), Y)

                layer.bias[i] = b - h
                dec_cost = cost(self.predict(X), Y)

                grad_b[i] = (inc_cost - dec_cost) / (2 * h)
                layer.bias[i] = b

                for j in range(0, len(layer.weights[i])):
                    # weights
                    w = layer.weights[i][j]

                    layer.weights[i][j] = w + h
                    inc_cost = cost(self.predict(X), Y)

                    layer.weights[i][j] = w - h
                    dec_cost = cost(self.predict(X), Y)

                    grad_W[i][j] = (inc_cost - dec_cost) / (2 * h)
                    layer.weights[i][j] = w

            dW.append(grad_W)
            db.append(grad_b)

        return dW, db

    @staticmethod
    def _gradient_checking(dW, dW_check):
        """ Compares the derivatives obtained through backpropagation with the ones obtained numerically. """
        abs_error = weights_count = relative_error = 0
        for gw1, gw2 in zip(dW, dW_check):
            assert gw1.shape == gw2.shape
            weights_count += gw1.size
            abs_error += np.sum(np.abs(gw1 - gw2))
            relative_error += np.linalg.norm(gw1 - gw2) / (np.linalg.norm(gw1) + np.linalg.norm(gw2))

        mean_error = abs_error / weights_count
        relative_error /= len(dW)
        print("[Gradient Checking] Absolute error: %.6f  |  Mean error: %.6f  |  Relative error: %.6f%%"
              % (abs_error, mean_error, 100 * relative_error))


class NeuralLayer:
    """ Defines a layer of neurons in a feedforward neural network.

    Attributes:
        weights: matrix with the weights of the connections with the previous layer; the index (i, j) stores the weight
                of the connection between the neuron i of the current layer and the neuron j of the previous layer.
        bias: vector with the biases of the layer's neurons.
        activation: an instance of the activation function used by the layer's neurons.
    """

    def __init__(self, num_neurons, input_size, activation):
        """ Creates a new layer of neurons.

        :param num_neurons: number of neurons in the layer.
        :param input_size: expected size of the input to be received by the layer.
        :param activation: an instance of the activation function used by the layer's neurons; an instance of a subclass
        of activation_functions.ActivationFunction.
        """
        self.weights = np.random.uniform(low=-1, high=1, size=(num_neurons, input_size))
        self.bias = np.random.uniform(low=-1, high=1, size=(num_neurons, 1))
        self.activation = activation

    def process(self, X):
        """ Processes the given input.

        :param X: expects the transpose of the input matrix or the output of the previous layer.
        :return: a tuple containing the layer's output (index 0) and weighted input z (index 1).
        """
        z = np.matmul(self.weights, X) + self.bias
        return self.activation(z), z

    def num_neurons(self):
        """ Returns the number of neurons in the layer. """
        return self.weights.shape[0]

    def input_size(self):
        """ Returns the size of the input vector expected by the layer. """
        return self.weights.shape[1]
