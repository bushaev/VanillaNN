from activations import *


class CostFunction(ABC):
    @abstractmethod
    def evaluate(self, a, y):
        pass

    @abstractmethod
    def delta(self, z, a, y):
        pass


class QuadraticCost(CostFunction):
    def __init__(self, activation=Sigmoid):
        self.activation = activation

    def evaluate(self, a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    def delta(self, z, a, y):
        return (a - y) * self.activation.diff(z)

# TODO: implement cross-entrapy loss function
