import numpy as np

class Sigmoid():
    def calc_sigmoid(self, z_value):
        return 1/(1 + np.exp(-z_value))
    