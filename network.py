from activations import *
from optimizers import *
from utils import *

class LayerNetwork():
    def __init__(self, layers, cost=QuadraticCost()):
        self.layers = layers
        self.cost = cost

    def forward(self, X):
        a = X.T
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def optimize(self, X, y, lr, batch_size):
        for k in range(0, len(X), batch_size):
            X_batch = X[k:k+batch_size]
            y_batch = y[k:k+batch_size]

            a = X_batch.T
            As = [a]
            for layer in self.layers:
                As.append(layer.forward(a))
                a = As[-1]

            delta = self.cost.delta(None, As[-1], y_batch)
            grads, error = self.layers[-1].backward(delta, is_final=True, A_prev=As[-2])
            self.layers[-1].update(*grads, lr=lr, m=batch_size)

            for l in range(2, len(self.layers)):
                grads, error = self.layers[-l].backward(error, A_prev=As[-l - 1])
                self.layers[-l].update(*grads, lr=lr, m=batch_size)



class Network:
    def __init__(self, layers, activation=Sigmoid, cost=QuadraticCost(), regularization=None):
        self.layers = layers
        self.num_layers = len(layers)
        if type(activation) == list:
            self.activations = [activation[0] for _ in range(self.num_layers - 2)]
            self.activations.append(activation[1])
        else:
            self.activations = [activation for _ in range(self.num_layers)]
        self.cost = cost
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

    def forward(self, X):
        a = X.T
        for b, w, act in zip(self.biases, self.weights, self.activations):
            a = act.evaluate(np.dot(w, a) + b)
        return a

    def optimize(self, X, y, lr, batch_size=10, optimizer=SGD):
        op = optimizer(self)
        op.optimize(X, y, lr, batch_size)

    def backprop(self, X, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        As = [X.T]
        activation = X.T
        Zs = []
        m = X.shape[0]

        for b, w, a in zip(self.biases, self.weights, self.activations):
            Zs.append(np.dot(w, activation) + b)
            activation = a.evaluate(Zs[-1])
            As.append(activation)

        delta = self.cost.delta(Zs[-1], As[-1], y)
        nabla_b[-1] =  1 / m * np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] =  1 / m * np.dot(delta, As[-2].T)

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activations[-l].diff(Zs[-l])
            nabla_b[-l] = 1 / m * np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = 1 / m * np.dot(delta, As[-l - 1].T)

        return nabla_b, nabla_w


class ClassificationNetwork(LayerNetwork):
    def optimize(self, X, y, lr, batch_size=10, optimizer=SGD, nb_epoch=1):
        for j in range(nb_epoch):
            X, y = shuffle(X, y)
            super(ClassificationNetwork, self).optimize(X, y, lr, batch_size)
            print(f"Accuracy after {j + 1} epochs ", str(self.accuracy(X, y)), "%")

    def accuracy(self, X, y):
        pred = np.array([])

        for k in range(0, len(X), 60):
            pred = np.concatenate([pred, self.predict(X[k:k+60])])

        return (pred == y).mean() * 100

    def predict(self, X):
        return np.argmax(self.forward(X), axis=0)
