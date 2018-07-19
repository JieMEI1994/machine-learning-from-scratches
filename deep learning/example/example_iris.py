#-*- coding: utf-8 -*
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
import numpy as np

import sys
sys.path.append("C:\\Users\\jmei\\Documents\\Code\\machine learning from scratch\\deep learning\\function")
import model

np.random.seed(20180718)
random.seed(20180718)
  
iris = datasets.load_iris()

data = iris["data"]
label = iris["target"]
num_label = len(np.unique(label))

[train_data, test_data, train_label, test_label] = train_test_split(data, label, test_size=0.2, shuffle=True)

train_data = np.stack(train_data)
test_data = np.stack(test_data)
train_label = np.stack(train_label)
test_label = np.stack(test_label)

layers_dim = [4,4,4]

model = model.vanilla_nural_network(layers_dim)
model.train(train_data, train_label,
            iteration=200000,
            learning_rate=0.001,
            lambd = 0,
            keep_prob = 1,
            interrupt_threshold = 0.1,
            print_loss = True)
model.plot()
prob = model.predict(test_data)
predicet_label = np.argmax(prob,  axis=1)
print("Test Accuracy: %.f%%" % (np.mean(test_label == predicet_label) * 100))
