from layer import Layer
import numpy as np
from activations_cache import Sigmoid

class Dense(Layer):
    def __init__(self, n_input, n_output, activation=Sigmoid):
        self.activation = activation()
        self.weight = np.random.rand(n_output, n_input) / np.sqrt(n_input)
        self.bias = np.zeros(shape=(n_output, 1))

    def forward(self, x):
        return self.activation.evaluate(np.dot(self.weight, x) + self.bias)

    def backward(self, *args, **kwargs):
        if 'is_final' in kwargs:
            is_final = kwargs['is_final']
        else:
            is_final = False

        if is_final:
            delta = args[0]
        else:
            dA = args[0]
            delta = dA * self.activation.diff()

        db = np.sum(delta, axis=1, keepdims=True)
        dW = np.dot(delta, kwargs['A_prev'].T)

        dA_prev = np.dot(self.weight.T, delta)

        return (db, dW), dA_prev

    def update(self, *args, **kwargs):
        db, dW = args
        lr = kwargs['lr']
        m = kwargs['m']

        self.weight = self.weight - (lr / m) * dW
        self.bias = self.bias - (lr / m) * db



