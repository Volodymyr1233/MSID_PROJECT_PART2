import numpy as np


class LogisticRegrGradient:
    def __init__(self, lr=0.01, epochs=100, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.wages = None

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        m, n = x.shape
        x_bias = np.hstack([np.ones((m, 1)), x])
        self.wages = np.zeros(x_bias.shape[1])

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = x_bias[indices]
            y_shuffled = y[indices]

            for start in range(0, m, self.batch_size):
                end = start + self.batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                preds = self.__sigmoid(xb @ self.wages)
                error = preds - yb
                gradient = xb.T @ error / len(yb)
                self.wages -= self.lr * gradient

    def predict(self, x):
        x_bias = np.hstack([np.ones((x.shape[0], 1)), x])
        probs = self.__sigmoid(x_bias @ self.wages)
        return (probs >= 0.5).astype(int), probs

    def compute_cross_entropy(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))