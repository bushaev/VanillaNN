import numpy as np
from mnist import MNIST
import bcolz

def one_hot(a, n):
   return np.eye(n)[a]

def shuffle(X, y):
    assert (len(X) == len(y))
    p = np.random.permutation(len(X))
    return X[p], y[p]

def data(path, tr_len=50000):
    data = MNIST('mnist')
    images, labels = data.load_training()
    images = np.array(images)
    labels = np.array(labels)

    X_tr, X_val = images[:tr_len], images[tr_len:]
    y_tr, y_val = labels[:tr_len], labels[tr_len:]

    return (X_tr, y_tr, X_val, y_val)

class Scaler():
    def __init__(self, X=None):
        if X is not None:
            self.fit(X)

    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
