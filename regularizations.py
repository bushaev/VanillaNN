import numpy as np
from abc import ABC, abstractmethod


class Regularization(ABC):
    @abstractmethod
    def fn(self, weights, n_data):
        pass


class L2(Regularization):
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def fn(self, weights, n_data):
        return (self.lmbda / (2 * n_data)) * sum(
            np.linalg.norm(w) ** 2 for w in weights)
