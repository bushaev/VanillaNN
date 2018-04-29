from activations import *
from optimizers import *
from utils import *

class LayerNetwork():
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def parameters(self):
        return np.concatenate([l.parameters() for l in self.layers])

    def forward(self, X):
        a = X.T
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def gradients(self, X, y):
        As = [X.T]
        for layer in self.layers:
            As.append(layer.forward(As[-1]))

        delta = self.cost.delta(None, As[-1], y)
        dA_prev = self.layers[-1].backward(delta, is_final=True)

        for l in range(2, len(self.layers) + 1):
            dA_prev = self.layers[-l].backward(dA_prev)

        grads = []
        for l in self.layers:
            grads.append([p.grad() for p in l.parameters()])
        return grads

    def optimize(self, X, y, batch_size, optim):
        for k in range(0, len(X), batch_size):
            X_batch = X[k:k+batch_size]
            y_batch = y[k:k+batch_size]

            As = [X_batch.T]
            for layer in self.layers:
                As.append(layer.forward(As[-1]))

            delta = self.cost.delta(None, As[-1], y_batch)
            dA_prev = self.layers[-1].backward(delta, is_final=True)

            for l in range(2, len(self.layers) + 1):
                dA_prev = self.layers[-l].backward(dA_prev)

            optim.update()


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
