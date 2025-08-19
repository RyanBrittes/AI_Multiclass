import numpy as np
from sigmoid import Sigmoid

class GradientDescent():
    def __init__(self):
        self.sigmoid = Sigmoid()

    def calc_gradient(self, x_train, y_train, weights, lr, epochs):
        n_sample = len(y_train)

        for i in range(epochs):
            z_value = np.array(x_train @ weights).reshape(-1, 1)
            y_pred = self.sigmoid.calc_sigmoid(z_value)
            simple_loss = (y_pred - y_train)

            dw = np.array((x_train.T @ simple_loss) / n_sample).flatten()

            weights -= lr * dw

        return weights
