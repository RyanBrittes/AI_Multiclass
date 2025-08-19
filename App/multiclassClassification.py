from sigmoid import Sigmoid
from logLoss import LogLoss
from loadData import LoadData
from unitGradientDescent import UnitGradientDescent
import numpy as np

class MulticlassClassification():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.loss = LogLoss()
        self.data = LoadData()
        self.gradient = UnitGradientDescent()
        self.rate_test = 0.2
        self.rate_validation = 0
        self.shuffled_data = self.data.get_shuffle_separe_train_validation_test(self.rate_test, self.rate_validation)
        self.x_train = self.shuffled_data[0]
        self.y_train = self.shuffled_data[1]
        self.x_validation = self.shuffled_data[2]
        self.y_validation = self.shuffled_data[3]
        self.x_test = self.shuffled_data[4]
        self.y_test = self.shuffled_data[5]
        self.len_sample = self.shuffled_data[6]
        self.n_classes = self.shuffled_data[7]
        self.weights = np.zeros(self.shuffled_data[0].shape[1])
        self.lr = 0.0001
        self.epochs = 8000
        self.losses = []
        self.batch_size = 50
    
    def train_model(self):
        line_sample, col_sample = self.x_train.shape
        self.x_train = np.insert(self.x_train, 0, 1, axis=1)
        all_weights = np.zeros((self.n_classes, col_sample + 1))

        for i in range(self.n_classes):
            y_compair = np.where(self.y_train == i, 1, 0)
            weights = np.zeros(col_sample + 1)
            weights = self.gradient.calc_gradient(self.x_train, y_compair, weights, self.lr, self.epochs)
            all_weights[i] = weights
        
        return [all_weights, self.x_test]
