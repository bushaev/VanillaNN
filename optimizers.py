from abc import ABC, abstractclassmethod
import numpy as np

class Optimizer(ABC):
    def __init__(self, model):
        self.model = model
        self.cost = model.cost

    @abstractclassmethod
    def optimize(self, X, y, lr, batch_size):
        pass


#TODO: update this class
#TODO: add momentum
class SGD(Optimizer):
    def __init__(self, model):
        super(SGD, self).__init__(model)

    def update_batch(self, X, y, lr):
        nabla_b, nabla_w = self.model.backprop(X, y)

        # TODO: add regularization support
        self.model.weights = np.array([w - lr * nw for w, nw in zip(self.model.weights, nabla_w)])
        self.model.biases = np.array([b - lr * nb for b, nb in zip(self.model.biases, nabla_b)])

    def optimize(self, X, y, lr, batch_size):
        cs = []
        for k in range(0, len(X), batch_size):
            self.update_batch(X[k:k+batch_size], y[k:k+batch_size], lr)
            a = self.model.forward(X[k:k+batch_size])
            cs.append(self.cost.evaluate(a, y[k:k+batch_size]))

#TODO: add RMSprop, Adam
#TODO: add learning rate schedule (exponential anealing, sgdr, cyclic learning rate, one cycle learning rate)