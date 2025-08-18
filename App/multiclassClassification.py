from sigmoid import Sigmoid
from logLoss import LogLoss
from loadData import LoadData
import numpy as np

class MulticlassClassification():
    def __init__(self):
        self.sigmoid = Sigmoid()
        self.loss = LogLoss()
        self.data = LoadData()
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
        self.weights = np.zeros(self.shuffled_data[0].shape[1])
        self.bias = 0
        self.lr = 0.01
        self.epochs = 8000
        self.losses = []
        self.batch_size = 50
    
    def train_model(self):
        for epoch in range(self.epochs):

            for i in range(0, self.len_sample, self.batch_size):
                x_batch = self.x_train[i:i+self.batch_size]
                y_batch = self.y_train[i:i+self.batch_size]

                z_value = np.array(x_batch @ self.weights + self.bias).reshape(-1, 1)
                y_pred = self.sigmoid.calc_sigmoid(z_value)
                simple_loss = self.loss.calc_simple_loss(y_pred, y_batch)

                dw = np.array((x_batch.T @ simple_loss) / self.batch_size).flatten()
                db = np.sum(simple_loss) / self.batch_size

                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
            z_train_predict = np.array(self.x_train @ self.weights + self.bias).reshape(-1, 1)
            y_train_predict = self.sigmoid.calc_sigmoid(z_train_predict)
            loss = self.loss.calc_log_loss(y_train_predict, self.y_train)
            self.losses.append(loss)
            
            #print(f"Epoch: {epoch}\nLoss: {loss:.4f}\n------")
        
        return [self.weights, self.bias, self.losses, self.x_test, self.y_test]