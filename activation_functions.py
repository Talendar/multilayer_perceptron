""" Implements the activation functions used by artificial neurons.

@author: Gabriel G. Nogueira (Talendar)
"""

import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """ Abstract class defining the basic interface of an activation function. """

    @abstractmethod
    def __call__(self, z, derivative=False):
        """ Computes the function. """
        raise NotImplementedError("This method wasn't implemented!")


class SigmoidActivation(ActivationFunction):
    """ Sigmoid activation function. """

    def __call__(self, z, derivative=False):
        """ Computes the function. """
        if not derivative:
            return 1 / (1 + np.exp(-z))
        return np.exp(-z) / ( (1 + np.exp(-z))**2 )


class ReluActivation(ActivationFunction):
    """ Rectifier activation function, used by ReLU (rectified linear unit) neurons. """

    def __call__(self, z, derivative=False):
        """ Computes the function. """
        if not derivative:
            return np.maximum(0, z)
        return np.ceil(np.clip(z, 0, 1))


class LinearActivation(ActivationFunction):
    """ Linear activation function. """

    def __call__(self, z, derivative=False):
        """ Computes the function. """
        if not derivative:
            return z
        return 1


def create_by_name(name):
    """ Creates an instance of the cost function with the given name. """
    name = name.lower()
    if name == "sigmoid":
        return SigmoidActivation()
    if name == "relu":
        return ReluActivation()
    if name == "linear":
        return LinearActivation()

    raise NameError("Activation function with name \"" + name + "\" not found!")
