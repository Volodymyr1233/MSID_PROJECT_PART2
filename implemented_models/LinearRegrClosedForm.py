import numpy as np

class LinearRegrClosedForm:
    def __init__(self):
        self.wages = None

    def fit(self, x, y):
        X_bias = np.hstack([np.ones((x.shape[0], 1)), x])
        self.wages = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y.values

    def predict(self, x):
        x_bias_correct = np.hstack([np.ones((x.shape[0], 1)), x])
        return x_bias_correct @ self.wages