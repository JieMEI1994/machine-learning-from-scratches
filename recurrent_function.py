import numpz as np
from initialization_function import initialization
from linear_function import linear
from avtivation_function import tanh, sigmoid
from classification_function import softmax

class vanilla:
    def __int__(self):
        self.initialization = initialization.zero()
        self.linear = linear()
        self.tanh = tanh()
        self.sigmoid = sigmoid()

    def forward(self, X, A0, W, b, output_Y = False):
        """
        Elman network
        :param X:
        :param A0:
        :param W:
        :param b:
        :param output_Y:
        :return:
        """
        [Wxa, Waa, Way] = W
        [ba, by] = b
        [input_dims, batch_size, time_step] = X.shape
        [output_dims, state_dims] = Way.shape
        # initialization
        A_pre = self.initialization(state_dims, batch_size, time_step)
        A_next = self.initialization(state_dims, batch_size, time_step)
        Y = self.initialization(output_dims, batch_size, time_step)
        A_pre_tmp = A0
        for t in rang(time_step):
            A_pre[:,:,t] = A_pre_tmp
            A_next_tmp = self.tanh(self.linear(Waa, A_pre_tmp, ba) + self.linear(X[:,:,t], Wxa, ba))
            Y_tmp = self.linear(A_next_tmp, Way, by)
            if output_Y:
                classifier = softmax()
                Y = classifier.forward(Y)
            else:
                pass
            A_next[:,:,t] = A_next_tmp
            Y[:,:,t] = Y_tmp
            A_pre = A_next
        return A_pre, A_next, Y

    def backward(self, dA, X, A_pre, A_next, W, b):
        [Wax, Waa, Wya] = W
        [ba, by] = b
        [state_dims, batch_size, time_step] = dA.shape
        [input_dims, batch_size, time_step] = X.shape
        dX = self.initialization(X.shape)
        dWax = self.initialization(Wax.shape)
        dWaa = self.initialization(Waa.shape)
        dba = self.initialization(ba.shape)
        dA_Pre = self.initialization(A_pre.shape)
        for t in range(time_step):
            dA_next_tmp = dA[:,:,t] + dA_pre[:,:,t]
            dtanh = (1 - A_next[:,:,t] ** 2) * dA_next_tmp
            dX[:,:,t] = np.dot(Wax.T, dtanh)
            dWax_tmp = np.dot(dtanh, X[:,:,t].T)
            dA_Pre[:,:,t] = np.dot(Waa.T, dtanh)
            dWaa_tmp = np.dot(dtanh, A_pre[:,:,t].T)
            dba_tmp = np.sum(dtanh, axis=1, keepdims=1)
            dWax += dWax_tmp
            dWaa += dWaa_tmp
            dba += dba_tmp
        return dX, dA_Pre, dWax, dWaa, dba

class lstm:
    def __init__(self):
        self.zero = initialization.zero()
        self.linear = linear()
        self.tanh = tanh()
        self.sigmoid = sigmoid()

    def forward(self, X, A0, W, b, output_Y = False):
        [Wf, Wu, Wc, Wo, Wy] = W
        [bf, bu, bc, bo, by] = b
        [input_dims, batch_size, time_step] = X.shape
        [output_dims, state_dims] = Way.shape
        A_pre = self.zero(state_dims, batch_size, time_step)
        A_next = self.zero(state_dims, batch_size, time_step)
        C_pre = self.zero(state_dims, batch_size, time_step)
        C_next = self.zero(state_dims, batch_size, time_step)
        forget_gate = []
        update_gate =[]
        update_cell = []
        output_gate = []
        Y = self.zero(output_dims, batch_size, time_step)
        A_pre_tmp = A0
        C_pre_tmp = self.zero(A0.shape)
        for t in rang(time_step):
            A_pre[:,:,t] = A_pre_tmp
            C_pre[:,:,t] = C_pre_tmp
            input = np.concatenate((A_pre_tmp, X[:,:,t]), axis=0)
            forget_gate.append(self.sigmoid(self.linear(Wf, input, bf)))
            update_gate.append(self.sigmoid(self.linear(Wu, input, bu)))
            update_cell.append(self.tanh(self.linear(Wc, input, bc)))
            C_next_tmp = forget_gate[t] * C_pre_tmp + update_gate[t] * update_cell[t]
            output.append(self.sigmoid(self.linear(Wo, input, bo)))
            A_next_tmp = output_gate[t] * self.tanh(C_next_tmp)
            Y_tmp = self.linear(A_next_tmp, Wy, by)
            if output_Y:
                classifier = softmax()
                Y = classifier.forward(Y)
            A_pre_tmp = A_next_tmp
            C_pre_tmp = C_next_tmp
            A_next[:,:,t] = A_next_tmp
            C_next[:,:,t] = C_next_tmp
            Y[:,:,t] = Y_tmp
        return A_pre, A_next, C_pre, C_next, Y, forget_gate, update_gate, update_cell, output_gate

    def backward(self, dA, X, A_pre, A_next, C_pre, C_next, forget_gate, update_gate, update_cell, output_gate, W, b):
        [Wf, Wu, Wc, Wo, Wy] = W
        [bf, bu, bc, bo, by] = b
        [input_dims, batch_size, time_step] = X.shape
        [output_dims, state_dims] = Way.shape
        dX = self.zero((X.shape))
        dA_pre = self.zero((A_pre.shape))
        dC_pre = self.zero((C_pre.shape))
        dWf = self.zero((Wf.shape))
        dWu = np.zeros((Wu.shape))
        dWc = np.zeros((Wc.shape))
        dWo = np.zeros((Wo.shape))
        dbf = np.zeros((bf.shape))
        dbu = np.zeros((bu.shape))
        dbc = np.zeros((bc.shape))
        dbo = np.zeros((bo.shape))
        for t in range(time_step):
            dA_next_tmp = dA[:, :, t] + dA_pre[:, :, t]
            doutput_gate_tmp = dA_next_tmp * tanh(C_next[:,:,t]) * output_gate[t] * (1-output_gate[t])
            dupdate_cell_tmp = dA_next_tmp * doutput_gate_tmp * (1 - tanh(C_next[:,:,t]) ** 2 + dC_pre[:,:,t])* \
                               update_gate[t] * (1 - update_cell[t] ** 2)
            dupdate_gate_tmp = dA_next_tmp * doutput_gate_tmp * (1 - tanh(C_next[:,:,t]) ** 2 + dC_pre[:,:,t])* \
                               update_cell[t] * (1 - update_gate[t] * update_gate[t])
            dforget_gate_tmp = dA_next_tmp * doutput_gate_tmp * (1 - tanh(C_next[:,:,t]) ** 2 + dC_pre[:,:,t])* \
                               C_pre[:,:,t] * forget_gate[t] (1 - forget_gate[t])
            dWf_tmp = np.dot(dforget_gate_tmp, np.hstack([A_pre[:,:,t].T, X[:,:,t].T]))
            dWu_tmp = np.dot(dupdate_gate_tmp, np.hstack([A_pre[:,:,t].T, X[:,:,t].T]))
            dWc_tmp = np.dot(dupdate_cell_tmp, np.hstack([A_pre[:,:,t].T, X[:,:,t].T]))
            dWo_tmp = np.dot(doutput_gate_tmp, np.hstack([A_pre[:,:,t].T, X[:,:,t].T]))
            dbf_tmp = np.sum(dforget_gate_tmp, axis=1, keepdims=True)
            dbu_tmp = np.sum(dupdate_gate_tmp, axis=1, keepdims=True)
            dbc_tmp = np.sum(dupdate_cell_tmp, axis=1, keepdims=True)
            dbo_tmp = np.sum(doutput_gate_tmp, axis=1, keepdims=True)
            dA_pre[:,:,t] = np.dot(Wf[:, :state_dims].T, dforget_gate_tmp) + \
                            np.dot(Wc[:, :state_dims].T, dupdate_cell_tmp) + \
                            np.dot(Wu[:, :state_dims].T, dupdate_gate_tmp) + \
                            np.dot(Wo[:, :state_dims].T, doutput_gate_tmp)
            dC_pre[:,:,t] = dA_next_tmp * output_gate[t] * (1-tanh(C_next[:,:,t])**2 + dC_pre[:,:,t]) * forget_gate[t]
            dA_pre[:,:,t] = np.dot(Wf[:, state_dims:].T, dforget_gate_tmp) + \
                            np.dot(Wc[:, state_dims:].T, dupdate_cell_tmp) + \
                            np.dot(Wu[:, state_dims:].T, dupdate_gate_tmp) + \
                            np.dot(Wo[:, state_dims:].T, doutput_gate_tmp)
            dWf += dWf_tmp
            dWu += dWu_tmp
            dWc += dWc_tmp
            dWo += dWo_tmp
            dbf += dbf_tmp
            dbu += dbu_tmp
            dbc += dbc_tmp
            dbo += dbo_tmp
        return dX, dA_pre, dWf, dWu, dWo, dbf, dbu, dbo









