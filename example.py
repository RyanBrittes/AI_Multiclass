import numpy as np
import pandas as pd

# Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função de custo (log loss)
def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5  # para evitar log(0)
    cost = -(1/m) * np.sum(y*np.log(h+epsilon) + (1-y)*np.log(1-h+epsilon))
    return cost

# Gradiente descendente para um classificador binário
def gradient_descent(X, y, weights, lr, epochs):
    m = len(y)
    for _ in range(epochs):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= lr * gradient
    return weights

# Treinamento One-vs-All
def train_one_vs_all(X, y, num_classes, lr=0.1, epochs=1000):
    m, n = X.shape
    X = np.insert(X, 0, 1, axis=1)  # bias
    all_weights = np.zeros((num_classes, n+1))

    for c in range(num_classes):
        y_binary = np.where(y == c, 1, 0)  # classe c vs resto
        weights = np.zeros(n+1)
        weights = gradient_descent(X, y_binary, weights, lr, epochs)
        all_weights[c] = weights
    return all_weights

# Predição
def predict_one_vs_all(X, all_weights):
    X = np.insert(X, 0, 1, axis=1)  # bias
    probs = sigmoid(np.dot(X, all_weights.T))  # (m, num_classes)
    return np.argmax(probs, axis=1)

# ----------------------------------------------
# Exemplo de uso com dataset fictício
# ----------------------------------------------

# Criando dataset simples
data = {
    "feature1": [1, 2, 3, 6, 7, 8, 1, 2, 3],
    "feature2": [1, 2, 1, 6, 7, 8, 7, 8, 9],
    "class":    [0, 0, 0, 1, 1, 1, 2, 2, 2]
}

df = pd.DataFrame(data)

X = df[["feature1", "feature2"]].values
y = df["class"].values

# Treinar
num_classes = len(np.unique(y))
weights = train_one_vs_all(X, y, num_classes, lr=0.1, epochs=5000)

# Prever
preds = predict_one_vs_all(X, weights)

print("Pesos treinados:\n", weights)
print("Verdadeiro:", y)
print("Previsto:  ", preds)
print("Acurácia:", np.mean(preds == y))
