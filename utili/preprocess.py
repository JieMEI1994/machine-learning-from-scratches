import numpy as np

class one_hot_vector:
    def encoder(Y, num_label):
        m = Y.shape[0]
        vector = np.zeros((m, num_label), dtype="int32")
        for i in range(m):
            idx = Y[i]
            vector[i][idx] = 1
        return vector

    def decoder(Y_het):
        label = np.argmax(Y_het, axis=1)
        return label