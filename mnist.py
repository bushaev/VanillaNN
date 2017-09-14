from network import ClassificationNetwork
import numpy as np

model = ClassificationNetwork([1, 30, 2])


def vectorized_result(x, n=2):
    r = np.zeros((n, 1))
    r[x] = 1
    return r


tr_x = np.random.randint(0, 1000000, size=10000)
tr_y = [vectorized_result(int(x % 2 == 0)) for x in tr_x]

print (tr_x[:2])
print (tr_y[:2])

model.optimize(tr_data=list(zip(tr_x, tr_y)), lr=0.01, nb_epoch=4)
