import numpy as np
from sigmoid import Sigmoid

class UnitGradientDescent():
    def __init__(self):
        self.sigmoid = Sigmoid()

    def calc_gradient(self, x_train, y_train, weights, lr, epochs):
        n_sample = len(y_train)

        for i in range(epochs):
            y_pred = self.sigmoid.calc_sigmoid(np.dot(x_train, weights))

            dw = np.dot(x_train.T, (y_pred - y_train)) / n_sample

            weights -= lr * dw

        return weights
