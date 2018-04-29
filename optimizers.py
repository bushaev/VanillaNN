from abc import ABC, abstractclassmethod
import numpy as np

class Optimizer(ABC):
    def __init__(self):
        self.params = None

    def set_params(self, params):
        self.params = params

    @abstractclassmethod
    def update(self):
        pass


class SGD(Optimizer):
    def __init__(self, lr, params=None):
        super(SGD, self).__init__()
        self.lr = lr

        if params is not None:
            self.set_params(params)

    def update(self):
        for p in self.params:
            p.minus_(self.lr * p.grad())


class SGDM(Optimizer):
    def __init__(self, lr, params=None, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.moving_average = None

        if params is not None:
            self.set_params(params)
            self.moving_average = []

            for p in self.params:
                self.moving_average.append(np.zeros(shape=p.get().shape))

    def update(self):
        for ind, p in enumerate(self.params):
            self.moving_average[ind] = self.momentum * self.moving_average[ind] +\
                                       (1 - self.momentum) * p.grad()
            p.minus_(self.lr * self.moving_average[ind])


#TODO: add RMSprop, Adam
#TODO: add learning rate schedule (exponential anealing, sgdr, cyclic learning rate, one cycle learning rate)