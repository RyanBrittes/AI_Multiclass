import numpy as np
from multiclassClassification import MulticlassClassification
from sigmoid import Sigmoid

class CompairClasses():
    def __init__(self):
        self.trained_model = MulticlassClassification()
        self.sigmoid = Sigmoid()

    def compair_one_vs_all(self):
        trained_params = self.trained_model.train_model()
        weights = trained_params[0]
        x_test = trained_params[1]
        y_test = trained_params[2]

        x_test = np.insert(x_test, 0, 1, axis=1)
        y_pred = self.sigmoid.calc_sigmoid(np.dot(x_test, weights.T))

        return [np.argmax(y_pred, axis=1), y_test, weights]