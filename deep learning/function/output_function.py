import numpy as np

class softmax:
    def forward(self, A):
        A = np.exp(A - np.max(A, axis=1, keepdims=True))
        Y_het = A / np.sum(A, axis=1, keepdims=True)
        return Y_het

    def loss(self, Y, Y_het):
        m = Y.shape[0]
        np.clip(Y_het, 1e-100, 1-(1e-100))
        log_likelihood = -np.log(Y_het[range(m), Y])
        loss = (1.0 / m) * np.sum(log_likelihood)
        loss = np.squeeze(loss)
        return loss

    def backward(self, Y, Y_het):
        m = Y.shape[0]
        Y_het[range(m), Y] -= 1
        dA = (1.0 / m) * Y_het
        return dA