import numpy as np

class vanilla:
    def __init__(self,  learning_rate):
        self.lr = learning_rate

    def update(self, parameters, gradients):
        [self.W, self.b] = parameters
        [self.dW, self.db] = gradients
        self.W = self.W - (self.lr * self.dW)
        self.b = self.b - (self.lr * self.db)
        return self.W, self.b

class momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.beta = momentum

    def update(self, parameters, gradients):
        [self.W, self.b] = parameters
        [self.dW, self.db] = gradients
        self.velocity_W = np.zeros_like(self.W)
        self.velocity_b = np.zeros_like(self.b)
        self.velocity_W = self.beta * self.velocity_W + (1 - self.beta) * self.dW
        self.velocity_b = self.beta * self.velocity_b + (1 - self.beta) * self.db
        self.W = self.W - (self.lr * self.velocity_W)
        self.b = self.b - (self.lr * self.velocity_b)
        return self.W, self.b


class rmsprop:
    def __init__(self, learning_rate, momentum=0.9):
        self.lr = learning_rate
        self.beta = momentum

    def update(self, parameters, gradients):
        [self.W, self.b] = parameters
        [self.dW, self.db] = gradients
        self.squared_gradient_W = np.zeros_like(self.W)
        self.squared_gradient_b = np.zeros_like(self.b)
        self.gradient_W = self.beta * self.squared_gradient_W + (1 - self.beta) * np.power(self.dW, 2)
        self.gradient_b = self.beta * self.squared_gradient_b + (1 - self.beta) * np.power(self.db, 2)
        self.W = self.W - (self.lr * self.dW/np.sqrt(self.gradient_W))
        self.b = self.b - (self.lr * self.db/np.sqrt(self.gradient_b))
        return self.W, self.b

class adam:
    def __init__(self, learning_rate, momentum1=0.9, momentum2=0.9, epsilon = 1e-8):
        self.lr = learning_rate
        self.beta1 = momentum1
        self.beta2 = momentum2
        self.epsilon = epsilon

    def update(self, parameters, gradients, counter):
        [self.W, self.b] = parameters
        [self.dW, self.db] = gradients
        self.t = counter
        self.velocity_W = np.zeros_like(self.W)
        self.velocity_b = np.zeros_like(self.b)
        self.squared_gradient_W = np.zeros_like(self.W)
        self.squared_gradient_b = np.zeros_like(self.b)
        self.velocity_W = self.beta1 * self.velocity_W + (1 - self.beta1) * self.dW
        self.velocity_b = self.beta1 * self.velocity_b + (1 - self.beta1) * self.db
        self.squared_gradient_W = self.beta2 * self.squared_gradient_W + (1 - self.beta2) * np.power(self.dW, 2)
        self.squared_gradient_b = self.beta2 * self.squared_gradient_b + (1 - self.beta2) * np.power(self.db, 2)
        self.velocity_W_corrected = self.velocity_W /(1-np.power(self.beta1, counter))
        self.velocity_b_corrected = self.velocity_b / (1 - np.power(self.beta1, counter))
        self.squared_gradient_W_corrected = self.squared_gradient_W /(1-np.power(self.beta2, counter))
        self.squared_gradient_b_corrected = self.squared_gradient_b / (1 - np.power(self.beta2, counter))
        self.W = self.W - (self.lr * self.velocity_W_corrected/(np.sqrt(self.squared_gradient_W)+self.epsilon))
        self.b = self.b - (self.lr * self.velocity_b_corrected/(np.sqrt(self.squared_gradient_b)+self.epsilon))
        return self.W, self.b