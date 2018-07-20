import numpy as np

class padding:
    def __init__(self):
        pass

    def zero(self, X, pad):
        """
        :return: numpy array of shape (m, n_H+2*pad, n_W+2*pad, n_C)
        """
        X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values = 0)
        return X_pad

    def constant(self, X, pad,const):
        """
        :param const: integer of the value of pad
        :return: numpy array of shape (m, n_H+2*pad, n_W+2*pad, n_C)
        """
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=const)
        return X_pad