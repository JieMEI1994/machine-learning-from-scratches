import numpy as np
from initialization_function import initialization
from linear_function import linear
from activation_function import relu
from dropout_function import dropout
from output_function import softmax
from regularization_function import l2
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

    def train(self, X, Y, iteration, learning_rate, lambd = 0, keep_prob = 1, print_loss = True):
        # import function
        init = initialization(self.layer_dims)
        lin = linear()
        act = relu()
        drop = dropout(keep_prob)
        classifier = softmax()
        regulator = l2(lambd)
        optimizer = adam(learning_rate)
        # initialization
        counter = 0
        train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)
        self.W, self.b = init.he()
        # iteration
        for i in range(iteration):
            m = Y.shape[0]
            # forward
            A = train_X
            cache = [[None, A, None]]
            for l in range(len(self.layer_dims)-1):
                Z = lin.forward(A, self.W[l], self.b[l])
                A = act.forward(Z)
                A, D = drop.forward(A)
                cache.append([Z, A, D])
            # loss
            prob = classifier.forward(A)
            loss_tmp1 = classifier.loss(train_Y, prob)
            loss_tmp2 = regulator.loss(self.W, m)
            loss_tmp = loss_tmp1 + loss_tmp2
            self.loss.append(loss_tmp)
            # validation accuracy
            pred = self.predict(validation_X)
            pred = one_hot_vector.decoder(pred)
            acc_tmp = np.mean(validation_Y == pred)
            self.accuracy.append(acc_tmp)
            # print
            if print_loss and i % 1000 == 0:
                print("Iteration %i, Loss: %f, Accuracy: %.f%%" %(i, loss_tmp, acc_tmp*100))
            # backward
            dA = classifier.backward(train_Y, prob)
            for l in range(len(self.layer_dims)-1, 1, -1):
                dA = drop.backward(dA, cache[l][2])
                dZ = act.backward(dA, cache[l][0])
                dA, dW, db = lin.backward(dZ, cache[l-1][1], self.W[l-1], self.b[l-1], m)
                dW += regulator.backward(self.W[l-1], m)
                # update
                counter += 1
                self.W[l-1], self.b[l-1] = optimizer.update([self.W[l-1], self.b[l-1]], [dW, db], counter)

    def predict(self, X):
        lin = linear()
        act = relu()
        classifier = softmax()
        A = X
        for l in range(len(self.layer_dims)-1):
            Z = lin.forward(A, self.W[l], self.b[l])
            A = act.forward(Z)
        Y_het = classifier.forward(A)
        return Y_het
    
    def plot(self):
        plt.figure()
        plt.grid()
        plt.plot(self.loss)