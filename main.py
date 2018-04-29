import numpy as np
from network import *
from utils import data, Scaler
from activations_cache import *
from fully_connected import Dense
from cost_functions import NLL
scaler = Scaler()

# model = ClassificationNetwork([784, 100, 10], activation=[Relu, Softmax])
model = ClassificationNetwork([
    Dense(784, 100, activation=Relu),
    Dense(100, 10, activation=Softmax),
], cost=NLL())
X_tr, y_tr, X_val, y_val = data('mnist')
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

model.optimize(X_tr, y_tr, SGDM(lr=0.1), batch_size=64, nb_epoch=4)
# model.optimize(X_tr, y_tr, lr=0.01, batch_size=64, nb_epoch=4)
print (model.accuracy(X_val, y_val))