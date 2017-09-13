# TODO: relocate optimization code in here
from abc import ABC, abstractclassmethod
import numpy as np

from activations import Sigmoid
from cost_functions import QuadraticCost


class Optimizer(ABC):
    def __init__(self, regularization, weights, biases,
                 cost=QuadraticCost, activation=Sigmoid):
        self.regularization = regularization
        self.weights = weights
        self.biases = biases
        self.cost = cost
        self.activation = activation
        self.num_layers = biases.shape[0]

    def backprop(self, x, y):
        nabla_w = np.zeros(self.weights.shape)
        nabla_b = np.zeros(self.biases.shape)
        activations = [x]
        activation = x
        zs = []

        for b, w in zip(self.weights, self.biases):
            zs.append(np.dot(activation, w) + b)
            activation = self.activation.evaluate(zs[-1])
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activation, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)

        for l in range(self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.activation.diff(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(activation[-l - 1].transpose(), delta)

        return nabla_b, nabla_w

    @abstractclassmethod
    def optimize(self, tr_data, lr, batch_size):
        pass


class SGD(Optimizer):
    def __init__(self, regularization, weights, biases,
                 cost=QuadraticCost, activation=Sigmoid):
        super(SGD, self).__init__(regularization, weights, biases,
                                  cost, activation)

    def update_batch(self, batch, lr):
        nabla_b = np.zeros(self.biases.shape)
        nabla_w = np.zeros(self.weights.shape)
        m = len(batch)

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # TODO: add regularization support
        self.weights = [w - (lr / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def optimize(self, tr_data, lr, batch_size):
        np.random.shuffle(tr_data)
        batches = [tr_data[k:k + batch_size] for k in range(0, len(tr_data), batch_size)]

        for mini_batch in batches:
            self.update_batch(mini_batch, lr)
