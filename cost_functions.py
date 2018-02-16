from activations import *
from utils import *


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
        norm = (1 / a.shape[1])
        return 0.5 * norm * np.linalg.norm(a - one_hot(y, 10).T) ** 2

    def delta(self, z, a, y):
        return (a - one_hot(y, 10).T) * self.activation.diff(z)

# TODO: implement cross-entrapy loss function
