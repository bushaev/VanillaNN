from abc import abstractmethod, ABC

class Layer(object):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass