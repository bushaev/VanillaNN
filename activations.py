import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def evaluate(z):
        pass

    @abstractmethod
    def diff(z):
        pass


class Sigmoid(Activation):
    @staticmethod
    def evaluate(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def diff(z):
        return Sigmoid.evaluate(z) * (1 - Sigmoid.evaluate(z))


# TODO: implement softmax, relu, tanh activation functions
