import numpy as np
from padding_function import padding
from linear_function import linear

class convolution:
    def __init__(self):
        pass

    def forward(self, X, W, b, n_s, n_p):
        """
        :param X: input as numpy array of shape (m, height_size, width_size, color_channel_size)
        :param W: wights as numpy array of shape (filter_size, filter_size, color_channel_size, num_filters/new_color_channel_size)
        :param b: bias as numpy array of shape (1, 1, 1,num_filters/new_color_channel_size)
        :param n_s: stride parameters as integer of the moving length through filtering
        :param n_p: padding parameters as integer of the padding number around X
        :return: output as numpy array of shape (m, new_height_size, new_width_size, new_color_channel_size)
        """
        (m, n_H_prev, n_W_prev, n_C_prev) = X.shape
        (n_f, n_f, n_C_prev, n_C_new) = W.shape
        n_H_new = np.int((n_H_prev - n_f + 2 * n_p) / n_s + 1)
        n_W_new = np.int((n_W_prev - n_f + 2 * n_p) / n_s + 1)
        Z = np.zeros((m, n_H_new, n_W_new, n_C_new))
        pad = padding()
        X_pad = pad.zero(X, n_p)
        for i in range(m):
            sample = X_pad[i]
            for h in range(n_H_new):
                for w in range(n_W_new):
                    for c in range(n_C_new):
                        h_start = h * n_s
                        h_end = h_start + n_f
                        w_start = w * n_s
                        w_end = w_start + n_f
                        input = sample[h_start:h_end, w_start:w_end, :]
                        Z_tmp = np.multiply(input, W[:,:,:,c]) + b[:,:,:,c]
                        Z[i, h, w, c] = np.sum(np.sum(Z_tmp))
        return Z

    def backward(self, dZ, X, W, n_p):
        """
        :param dZ: input gradient as numpy array of shape (m, new_height_size, new_width_size, new_color_channel_size)
        :param X: input as numpy array of shape (m, height_size, width_size, color_channel_size)
        :param W: wights as numpy array of shape (filter_size, filter_size, color_channel_size, num_filters/new_color_channel_size)
        :param b: bias as numpy array of shape (1, 1, 1,num_filters/new_color_channel_size)
        :param stride: stride parameters as integer of the moving length through filtering
        :param pad: padding parameters as integer of the padding number around X
        :return: dX: output gradient as numpy array of same shape of X
        :return: dW: output gradient as numpy array of same shape of W
        :return: db: output gradient as numpy array of same shape of b
        """
        (m, n_H_prev, n_W_prev, n_C_prev) = X.shape
        (n_f, n_f, n_C_prev, n_C_new) = W.shape
        (m, n_H_new, n_W_new, n_C_new) = dZ.shape
        dX = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((n_f, n_f, n_C_prev, n_C_new))
        db = np.zeros((1, 1, 1, n_C_new))
        pad = padding()
        X_pad = pad.zero(X, n_p)
        dX_pad = pad.zero(dX, n_p)
        for i in range(m):
            sample_X = X_pad[i]
            sample_dX = dX_pad[i]
            for h in range(n_H_new):
                for w in range(n_W_new):
                    for c in range(n_C_new):
                        h_start = h * n_s
                        h_end = h_start + n_f
                        w_start = w * n_s
                        w_end = w_start + n_f
                        input = sample_X[h_start:h_end, w_start:w_end, :]
                        output = sample_dX
                        output[h_start:h_end, w_start:w_end, :] += (W[:,:,:,c] * dZ[i, h, w, c])
                        dW[:,:,:,c] += input * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
            dX[i] = output[n_p:-n_p, n_p:-n_p, :]
        return dX, dW, db