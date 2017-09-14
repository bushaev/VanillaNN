from activations import *
from optimizers import *


class Network:
    def __init__(self, layers, activation=Sigmoid, cost=QuadraticCost(), regularization=None):
        self.layers = layers
        self.activation = activation
        self.cost = cost
        self.num_layers = len(layers)
        self.regularization = regularization

        self.biases = np.array([np.random.randn(y, 1) for y in self.layers[1:]])
        self.weights = np.array([np.random.randn(y, x) / np.sqrt(x)
                                 for x, y in zip(self.layers[:-1], self.layers[1:])])

    # TODO: implement loading a network from a file
    @staticmethod
    def from_file(filename):
        pass

    # TODO: implement saving weights to file
    def save(self, file):
        pass

    def forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation.evaluate(np.dot(w, a) + b)
        return a

    def optimize(self, tr_data, lr, batch_size=10, optimizer=SGD, nb_epoch=1):
        op = optimizer(str(self.regularization), self.weights, self.biases, self.cost, self.activation)
        op.optimize(tr_data, lr, batch_size)


class ClassificationNetwork(Network):
    def optimize(self, tr_data, lr, batch_size=10, optimizer=SGD, nb_epoch=1):

        for j in range(nb_epoch):
            super(ClassificationNetwork, self).optimize(tr_data, lr, batch_size, optimizer, nb_epoch=1)
            print(f"Accuracy after {j + 1} epochs ", str(self.accuracy(tr_data)), "%")

    def accuracy(self, data):
        n_data = len(data)
        accurate = 0
        for x, y in data:
            if np.argmax(y) == np.argmax(self.forward(x)):
                accurate += 1

        return int(accurate * 100 / n_data)

    def predict(self, x):
        return np.argmax(self.forward(x))
