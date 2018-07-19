import numpy as np
from initialization_function import initialization
from linear_function import linear
from activation_function import relu
from output_function import softmax
from optimization_function import adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
sys.path.append("C:\\Users\\jmei\\Documents\\Code\\machine learning from scratch\\utili")
from preprocess import one_hot_vector

class vanilla_nural_network:
    def __init__(self, hidden_layer_dims):
        self.layer_dims = hidden_layer_dims
        self.W = []
        self.b = []
        self.loss = []
        self.accuracy = []

    def train(self, X, Y, iteration, learning_rate):
        init = initialization(self.layer_dims)
        self.W, self.b = init.he()
        lin = linear()
        act = relu()
        clas = softmax()
        optimizer = adam(learning_rate)
        counter = 0
        train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)
        for i in range(iteration):
            # forward
            A = train_X
            cache = [[None, A]]
            for l in range(len(self.layer_dims)-1):
                Z = lin.forward(A, self.W[l], self.b[l])
                A = act.forward(Z)
                cache.append([Z, A])
            # loss/accuracy
            prob = clas.forward(A)
            loss_tmp = clas.loss(train_Y, prob)
            self.loss.append(loss_tmp)
            pred = self.predict(validation_X)
            pred = one_hot_vector.decoder(pred)
            acc_tmp = np.mean(validation_Y == pred)
            self.accuracy.append(acc_tmp)
            if i % 1000 == 0:
                print("Iteration %i, Loss: %f, Accuracy: %.f%%" %(i, loss_tmp, acc_tmp*100))
            # backward
            dA = clas.backward(train_Y, prob)
            for l in range(len(self.layer_dims)-1, 1, -1):
                dZ = act.backward(dA, cache[l][0])
                dA, dW, db = lin.backward(dZ, cache[l-1][1], self.W[l-1], self.b[l-1])
                # update
                counter += 1
                self.W[l-1], self.b[l-1] = optimizer.update([self.W[l-1], self.b[l-1]], [dW, db], counter)

    def predict(self, X):
        lin = linear()
        act = relu()
        clas = softmax()
        A = X
        for l in range(len(self.layer_dims)-1):
            Z = lin.forward(A, self.W[l], self.b[l])
            A = act.forward(Z)
        Y_het = clas.forward(A)
        return Y_het
    
    def plot(self):
        plt.figure()
        plt.grid()
        plt.plot(self.loss)
