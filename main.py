import numpy as np
from network import ClassificationNetwork
from utils import data

model = ClassificationNetwork([784, 100, 10])
X_tr, y_tr, X_val, y_val = data('mnist')

model.optimize(X_tr, y_tr, lr=0.1, batch_size=32, nb_epoch=5)
print (model.accuracy(X_val, y_val))