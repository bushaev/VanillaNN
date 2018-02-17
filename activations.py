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


class Relu(Activation):
    @staticmethod
    def evaluate(z):
        return np.maximum(0, z)

    @staticmethod
    def diff(z):
        return np.asarray(z > 0, dtype=np.uint8)

class Softmax(Activation):
    @staticmethod
    def evaluate(z):
        z = np.exp(z)
        return z / np.sum(z, axis=0)

    @staticmethod
    def diff(z):
        pass