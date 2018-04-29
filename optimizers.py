from abc import ABC, abstractclassmethod
import numpy as np

class Optimizer(ABC):
    @abstractclassmethod
    def update_weights(self, weight, grads):
        pass


class SGD(Optimizer):
    def __init__(self, lr):
        super(SGD, self).__init__()
        self.lr = lr

    def update_weights(self, weight, grads):
        return weight - self.lr * grads


class SGDM(Optimizer):
    def __init__(self, lr, momentum=0.9):
        super(SGDM, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.average = None

    def update_weights(self, weight, grads, *args, **kwargs):
        if self.average is None:
            self.average = np.zeros(shape=grads.shape)

        self.average = self.momentum * self.average + (1 - self.momentum) * grads
        return weight - self.lr * self.average


#TODO: add RMSprop, Adam
#TODO: add learning rate schedule (exponential anealing, sgdr, cyclic learning rate, one cycle learning rate)