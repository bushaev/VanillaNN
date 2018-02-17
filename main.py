import numpy as np
from network import ClassificationNetwork
from utils import data, Scaler
from activations import *

scaler = Scaler()

model = ClassificationNetwork([784, 100, 10], activation=[Relu, Softmax])
X_tr, y_tr, X_val, y_val = data('mnist')
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

model.optimize(X_tr, y_tr, lr=0.5, batch_size=64, nb_epoch=5)
print (model.accuracy(X_val, y_val))