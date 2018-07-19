import numpy as np

class mse:
    def forward(self, A):
        Y_het = A
        return Y_het

    def loss(self, Y, Y_het):
        loss = np.mean(np.power((Y - Y_het), 2))
        loss = np.squeeze(loss)
        return loss

    def backward(self, Y, Y_het):
        dA = Y - Y_het
        return dA

class rmse:
    def forward(self, A):
        Y_het = A
        return Y_het

    def loss(self, Y, Y_het):
        loss = np.sqrt(np.mean(np.power((Y - Y_het), 2)))
        loss = np.squeeze(loss)
        return loss

    def backward(self, Y, Y_het):
        dA = Y - Y_het
        return dA

class cross_entropy:
    def forward(self, A):
        Y_het = A
        return Y_het

    def loss(self, Y, Y_het):
        m = Y.shape[0]
        log_likelihood = np.multiply(np.log(Y_het), Y) + np.multiply((1 - Y), np.log(1 - Y_het))
        loss = - np.sum(logprobs) / m
        loss = np.squeeze(loss)
        return loss

    def backward(self, Y, Y_het):
        dA = - (np.divide(Y, Y_het) - np.divide(1 - Y, 1 - Y_het))
        return dA