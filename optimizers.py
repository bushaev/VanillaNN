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
        self.num_layers = biases.shape[0] + 1

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations = [x]
        activation = x
        zs = []

        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activation) + b)
            activation = self.activation.evaluate(zs[-1])
            activations.append(activation)

        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.activation.diff(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

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
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m = len(batch)

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # TODO: add regularization support
        self.weights = np.array([w - (lr / m) * nw for w, nw in zip(self.weights, nabla_w)])
        self.biases = np.array([b - (lr / m) * nb for b, nb in zip(self.biases, nabla_b)])

    def optimize(self, tr_data, lr, batch_size):
        np.random.shuffle(tr_data)
        batches = [tr_data[k:k + batch_size] for k in range(0, len(tr_data), batch_size)]

        for mini_batch in batches:
            self.update_batch(mini_batch, lr)

        return self.biases, self.weights
