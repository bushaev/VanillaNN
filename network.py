from activations import *
from optimizers import *
from utils import *

class LayerNetwork():
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def forward(self, X):
        a = X.T
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def gradients(self, X, y):
        grads = []
        As = [X.T]
        for layer in self.layers:
            As.append(layer.forward(As[-1]))

        delta = self.cost.delta(None, As[-1], y)
        g, e = self.layers[-1].backward(delta, is_final=True, A_prev=As[-2])
        grads.append(g)

        for l in range(2, len(self.layers) + 1):
            g, e = self.layers[-l].backward(e, A_prev=As[-l - 1])
            grads.append(g)

        return grads[::-1]

    def optimize(self, X, y, batch_size, optim):
        for k in range(0, len(X), batch_size):
            X_batch = X[k:k+batch_size]
            y_batch = y[k:k+batch_size]

            As = [X_batch.T]
            for layer in self.layers:
                As.append(layer.forward(As[-1]))

            delta = self.cost.delta(None, As[-1], y_batch)
            grads, error = self.layers[-1].backward(delta, is_final=True, A_prev=As[-2])
            self.layers[-1].update(optim)

            for l in range(2, len(self.layers) + 1):
                grads, error = self.layers[-l].backward(error, A_prev=As[-l - 1])
                self.layers[-l].update(optim)


class ClassificationNetwork(LayerNetwork):
    def optimize(self, X, y, optim, batch_size=10, nb_epoch=1):
        for j in range(nb_epoch):
            X, y = shuffle(X, y)
            super(ClassificationNetwork, self).optimize(X, y, batch_size, optim)
            print(f"Accuracy after {j + 1} epochs ", str(self.accuracy(X, y)), "%")

    def accuracy(self, X, y):
        pred = np.array([])

        #TODO: WTF is this??
        for k in range(0, len(X), 60):
            pred = np.concatenate([pred, self.predict(X[k:k+60])])

        return (pred == y).mean() * 100

    def predict(self, X):
        return np.argmax(self.forward(X), axis=0)
