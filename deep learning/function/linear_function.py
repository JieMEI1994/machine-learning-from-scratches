import numpy as np

class linear:
    def forward(self, X, W, b):
        Z = np.dot(X, W) + b
        return Z

    def backward(self, dZ, X, W, b, m):
        dW = (1.0 / m) * np.dot(X.T, dZ)
        db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, W.T)
        return dX, dW, db