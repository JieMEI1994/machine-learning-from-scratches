import numpy as np

class max:
    def __init__(self):
        pass

    def forward(self, X, n_f, n_s):
        (m, n_H_prev, n_W_prev, n_C_prev) = X.shape
        n_H_new = np.int((n_H_prev - n_f) / n_s + 1)
        n_W_new = np.int((n_W_prev - n_f) / n_s + 1)
        n_C_new = n_C_prev
        output = np.zeros((m, n_H_new, n_W_new, n_C_new))
        for i in range(m):
            sample = X[i]
            for h in range(n_H_new):
                for w in range(n_W_new):
                    for c in range(n_C_new):
                        h_start = h * n_s
                        h_end = h_start + n_f
                        w_start = w * n_s
                        w_end = w_start + n_f
                        input = sample[h_start:h_end, w_start:w_end, c]
                        output[i, h, w, c] = np.max(input)
        return output

    def backward(self, dZ, X, n_f, n_s):
        (m, n_H_prev, n_W_prev, n_C_prev) = X.shape
        (m, n_H_new, n_W_new, n_C_new) = dZ.shape
        dX = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        for i in range(m):
            sample_X = X[i]
            for h in range(n_H_new):
                for w in range(n_W_new):
                    for c in range(n_C_new):
                        h_start = h * n_s
                        h_end = h_start + n_f
                        w_start = w * n_s
                        w_end = w_start + n_f
                        input = sample_X[h_start:h_end, w_start:w_end, c]
                        output = dX
                        mask = (input == np.max(input))
                        output[i, h_start:h_end, w_start:w_end, c] += np.multiply(mask, dZ[i, h_start:h_end, w_start:w_end, c])
        return output


