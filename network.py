from activations import *
from optimizers import *
from utils import *


class Network:
    def __init__(self, layers, activation=Sigmoid, cost=QuadraticCost(), regularization=None):
        self.layers = layers
        self.activation = activation
        self.cost = cost
        self.num_layers = len(layers)
        self.regularization = regularization
        self.sc = Scaler()
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

    def forward(self, X):
        a = X.T
        for b, w in zip(self.biases, self.weights):
            a = self.activation.evaluate(np.dot(w, a) + b)
        return a

    def optimize(self, X, y, lr, batch_size=10, optimizer=SGD):
        op = optimizer(str(self.regularization), self, self.cost, self.activation)
        op.optimize(X, y, lr, batch_size)

    def backprop(self, X, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        As = [X.T]
        activation = X.T
        Zs = []
        m = X.shape[0]

        for b, w in zip(self.biases, self.weights):
            Zs.append(np.dot(w, activation) + b)
            activation = self.activation.evaluate(Zs[-1])
            As.append(activation)

        delta = self.cost.delta(Zs[-1], As[-1], y)
        nabla_b[-1] =  1 / m * np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] =  1 / m * np.dot(delta, As[-2].T)

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation.diff(Zs[-l])
            nabla_b[-l] = 1 / m * np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = 1 / m * np.dot(delta, As[-l - 1].T)

        return nabla_b, nabla_w


class ClassificationNetwork(Network):
    def optimize(self, X, y, lr, batch_size=10, optimizer=SGD, nb_epoch=1):
        for j in range(nb_epoch):
            X, y = shuffle(X, y)
            super(ClassificationNetwork, self).optimize(X, y, lr, batch_size, optimizer)
            print(f"Accuracy after {j + 1} epochs ", str(self.accuracy(X, y)), "%")

    def accuracy(self, X, y):
        pred = np.array([])

        for k in range(0, len(X), 60):
            pred = np.concatenate([pred, self.predict(X[k:k+60])])

        return (pred == y).mean() * 100

    def predict(self, X):
        return np.argmax(self.forward(X), axis=0)
