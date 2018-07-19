import numpy as np

class l2:
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self):
        pass

    def loss(self, W, m):
        loss = 0
        for l in range(len(W)):
            loss += self.lambd * (np.sum(np.square(W[l]))) / (2 * m)
        loss = np.squeeze(loss)
        return loss

    def backward(self, W, m):
        dW = (1.0 / m) * self.lambd * W
        return dW