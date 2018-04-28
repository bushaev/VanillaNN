import numpy as np
from fully_connected import Dense
from activations_cache import *
from network import ClassificationNetwork
from cost_functions import NLL

def evaluate_cost(net, X, y):
    a = net.forward(X)
    return net.cost.evaluate(a, y)

def check_grad(grad, gradapprox, error_epsilon=1e-7):
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    return difference < error_epsilon

def check_dense(epsilon=1e-9, error_epsilon=1e-6):
    print ("Running gradient checking for dense layers!")
    net = ClassificationNetwork([
        Dense(64, 64, activation=Relu),
        Dense(64, 10, activation=Softmax),
    ], cost=NLL())
    X = np.random.randn(1, 64)
    y = [4]

    grads = net.gradients(X, y)
    for lind in range(2):
        db, dW = grads[lind]
        for ind in range(net.layers[lind].bias.shape[0]):
            net.layers[lind].bias[ind, 0] += epsilon
            cost_plus = evaluate_cost(net, X, y)

            net.layers[lind].bias[ind, 0] -= 2 * epsilon
            cost_minus = evaluate_cost(net, X, y)

            grad = (cost_plus - cost_minus) / (2 * epsilon)
            if np.fabs(grad - db[ind, 0]) > error_epsilon:
                print("Wrong gradient for bias: " + str(grad) + " ," + str(db[ind, 0]))

        for i, j in zip(range(net.layers[lind].weight.shape[0]),
                        range(net.layers[lind].weight.shape[1])):
            net.layers[lind].weight[i, j] += epsilon
            cost_plus = evaluate_cost(net, X, y)

            net.layers[lind].weight[i, j] -= 2 * epsilon
            cost_minus = evaluate_cost(net, X, y)

            grad = (cost_plus - cost_minus) / (2 * epsilon)
            if np.fabs(grad - dW[i, j]) > error_epsilon:
                print("Wrong gradient for weights: " + str(grad) + " ," + str(dW[i, j]))

    print ("Gradients for dense layers checked!")


if __name__ == '__main__':
    check_dense()