import numpy as np

class sigmoid:
    def forward(self, Z):
        A = 1.0 / (1.0 + np.exp(-Z))
        return A

    def backward(self, dA, A):
        dZ = A * (1 - A) * dA
        return dZ

class tanh:
    def forward(self, Z):
        A = np.tanh(Z)
        return A

    def backward(self, dA, A):
        dZ  = (1.0 - np.square(A)) * dA
        return dZ

class relu:
    def forward(self, Z):
        A = np.maximum(Z, 0)
        return A

    def backward(self, dA, Z):
        dZ = dA
        dZ[Z < 0] = 0
        return dZ