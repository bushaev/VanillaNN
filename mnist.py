from mnist_loader import *
from network import ClassificationNetwork

model = ClassificationNetwork([784, 30, 10])
training_data, validation_data, test_data = load_data_wrapper()

model.optimize(tr_data=list(training_data), lr=3, batch_size=10, nb_epoch=4)
