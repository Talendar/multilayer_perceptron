""" Implements the cost functions used by the models.

@author: Gabriel G. Nogueira (Talendar)
"""

from abc import ABC, abstractmethod
import numpy as np


class CostFunction(ABC):
    """ Abstract class defining a cost function and its operations. """

    @abstractmethod
    def __call__(self, predictions, labels):
        """ Computes the mean cost. """
        raise NotImplementedError("This method wasn't implemented!")

    @abstractmethod
    def unit(self, prediction, label, derivative=False):
        """ Computes the total cost. """
        raise NotImplementedError("This method wasn't implemented!")


class MeanSquaredError(CostFunction):
    """ Mean Squared Error (MSE) cost function. """

    def __call__(self, predictions, labels):
        """ Computes the mean cost. """
        m = len(predictions)
        return np.sum( (labels.T - predictions) ** 2 ) / (2 * m)

    def unit(self, prediction, label, derivative=False):
        """ Computes the total cost. """
        if not derivative:
            return ((label.T - prediction) ** 2) / 2

        return prediction - label.T
