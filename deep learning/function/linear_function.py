import numpy as np

class linear:
    def forward(self, X, W, b):
        Z = np.dot(X, W) + b
        return Z

    def backward(self, dZ, X, W, b):
        dX = np.dot(dZ, W.T)
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0)
        return dX, dW, db