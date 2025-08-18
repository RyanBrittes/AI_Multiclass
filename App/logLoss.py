import numpy as np

class LogLoss():
    def calc_log_loss(self, y_pred, y_true):
        alpha = 1e-15
        return np.mean(-y_true * np.log(y_pred + alpha) - (1 - y_true) * np.log(1 - y_pred + alpha))
    
    def calc_simple_loss(self, y_pred, y_true):
        return y_pred - y_true