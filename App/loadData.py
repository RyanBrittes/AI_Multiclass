import pandas as pd
import numpy as np
from normalizeData import NormalizeData

class LoadData():
    def __init__(self):
        self.normalize = NormalizeData()
        self.__data = pd.read_csv('files/diabetes.csv')
        self.__y_true = self.__data[['Outcome']].values
        self.__x_true = self.__data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values

    def get_dataset(self):
        return self.__data

    def get_x_value(self):
        return self.__x_true
    
    def get_y_value(self):
        return self.__y_true
    
    def get_score_Z(self, dataset):
        return self.normalize.calc_score_Z(dataset)
    
    def get_log(self, dataset):
        return self.normalize.calc_log(dataset)
    
    def get_shuffle_separe_train_validation_test(self, rate_test, rate_validation):
        x_values = self.get_log(self.__x_true)
        y_values = self.__y_true

        np.random.seed(42)
        indexShuffled = np.random.permutation(len(y_values))

        y_shuffled = y_values[indexShuffled]
        x_shuffled = x_values[indexShuffled]
        
        rate_train = 1 - rate_test - rate_validation

        len_sample = len(y_values)
        n_classes = len(np.unique(y_values))
        len_train = np.floor(len_sample * rate_train).astype(int)
        len_validation = np.round(len_sample * rate_validation).astype(int)

        x_train = x_shuffled[0:len_train]
        y_train = y_shuffled[0:len_train]
        x_validation = x_shuffled[len_train:(len_train + len_validation)]
        y_validation = y_shuffled[len_train:(len_train + len_validation)]
        x_test = x_shuffled[(len_train + len_validation):len_sample]
        y_test = y_shuffled[(len_train + len_validation):len_sample]
        
        return [x_train, y_train, x_validation, y_validation, x_test, y_test, len_sample, n_classes]
    