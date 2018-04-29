from layer import Layer
import numpy as np
from activations_cache import Sigmoid
from parameter import Parameter

class Dense(Layer):
    def __init__(self, n_input, n_output, activation=Sigmoid):
        self.activation = activation()
        self.weight = Parameter(np.random.rand(n_output, n_input) / np.sqrt(n_input))
        self.bias = Parameter(np.zeros(shape=(n_output, 1)))
        self.db = None
        self.dW = None

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        return self.activation.evaluate(np.dot(self.weight.get(), x) + self.bias.get())

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
        self.bias.set_grad(np.sum(delta, axis=1, keepdims=True) / m)
        self.weight.set_grad(np.dot(delta, kwargs['A_prev'].T) / m)

        dA_prev = np.dot(self.weight.get().T, delta)

        return (self.bias.grad(), self.weight.grad()), dA_prev


