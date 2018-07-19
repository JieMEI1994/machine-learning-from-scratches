import numpy as np

class dropout:
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def forward(self, A):
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < self.keep_prob
        A = np.multiply(A, D)
        A /= self.keep_prob
        return A, D

    def backward(self, dA, D):
        dA *= D
        dA /= self.keep_prob
        return dA