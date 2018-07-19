import numpy as np

class initialization:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.W = []
        self.b = []

    def zeros(self):
        for l in range(len(self.layer_dims)-1):
            self.W.append(np.zeros(self.layer_dims[l], self.layer_dims[l+1]))
            self.b.append(np.zeros((1, self.layer_dims[l+1])))
        return self.W, self.b

    def random(self):
        for l in range(len(self.layer_dims)-1):
            self.W.append(np.random.rand(self.layer_dims[l], self.layer_dims[l+1]))
            self.b.append(np.zeros((1, self.layer_dims[l+1])))
        return self.W, self.b

    def he(self):
        for l in range(len(self.layer_dims)-1):
            self.W.append(np.random.rand(self.layer_dims[l], self.layer_dims[l+1]) / (np.sqrt(2. / self.layer_dims[l])))
            self.b.append(np.zeros((1, self.layer_dims[l+1])))
        return self.W, self.b