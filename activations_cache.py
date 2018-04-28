import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def evaluate(self, z):
        pass

    @abstractmethod
    def diff(self):
        pass


class Sigmoid(Activation):
    def evaluate(self, z):
        if z >= 0:
            x = np.exp(-z)
            self.z = 1 / (1 + x)
        else:
            x = np.exp(z)
            self.z = x / (1 + x)

        return self.z

    def diff(self):
        return self.z * (1 - self.z)


class Relu(Activation):
    def evaluate(self, z):
        self.z = np.maximum(0, z)
        return self.z

    def diff(self):
        return np.asarray(self.z > 0, dtype=np.uint8)

class Softmax(Activation):
    def evaluate(self, z):
        self.z = z - z.max(axis=0)
        self.z = np.exp(self.z)
        self.z = self.z / np.sum(self.z, axis=0)
        return self.z

    #TODO: Implement diff. Not important as often used as last layer activation.
    def diff(self):
        return None