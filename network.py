import numpy as np


class Network:
    def __init__(self, layers, activation, loss):
        self.layers = layers
        self.activation = activation
        self.loss = loss
        self.num_layers = len(layers)

        self.biases  = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.layers[:-1], self.layers[1:])]

    @staticmethod
    def from_file(filename):
        pass

    # TODO: implement loading weight from file
    def load(self, file):
        pass

    # TODO: implement saving weights to file
    def save(self, file):
        pass

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation.evaluate(np.dot(w, a) + b)
        return a
