from layer import Layer
import numpy as np
from activations_cache import Sigmoid

class Dense(Layer):
    def __init__(self, n_input, n_output, activation=Sigmoid):
        self.activation = activation()
        self.weight = np.random.rand(n_output, n_input) / np.sqrt(n_input)
        self.bias = np.zeros(shape=(n_output, 1))
        self.db = None
        self.dW = None

    def forward(self, x):
        return self.activation.evaluate(np.dot(self.weight, x) + self.bias)

    # TODO: cache A_prev from forward pass
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

        m = delta.shape[1]
        self.db = np.sum(delta, axis=1, keepdims=True) / m
        self.dW = np.dot(delta, kwargs['A_prev'].T) / m

        dA_prev = np.dot(self.weight.T, delta)

        return (self.db, self.dW), dA_prev

    def update(self, optim):
        if self.db is not None and self.dW is not None:
            self.weight = optim.update_weights(self.weight, self.dW)
            self.bias = optim.update_weights(self.bias, self.db)



