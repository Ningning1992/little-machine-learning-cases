import numpy as np


class NeuralNetwork(object):

    def __init__(self, n_inputlayer=784, n_hiddenlayer=200,
                 n_outputlayer=26, mu=0, sd=0.01):
        self.mu = mu
        self.sd = sd
        self.n_inputlayer = n_inputlayer
        self.n_hiddenlayer = n_hiddenlayer
        self.n_outputlayer = n_outputlayer
        self.v = self.initial_weights(self.n_hiddenlayer,
                                      self.n_inputlayer+1)
        self.w = self.initial_weights(self.n_outputlayer,
                                      self.n_hiddenlayer+1)

    def initial_weights(self, row, col):
        return np.random.normal(self.mu, self.sd, (row, col))

    def sigmoid(self, r):
        return 1 / (1 + np.exp(-r))

    def forward(self, x):
        row = x.shape[0]

        z2 = np.dot(self.v, x.T)

        h = np.tanh(z2)
        one_h = np.ones(row, dtype=float)
        one_h = one_h.reshape((1, len(one_h)))
        h = np.concatenate((h, one_h), axis=0)

        z3 = np.dot(self.w, h)
        z = self.sigmoid(z3)

        return h, z

    def each_forward(self, x_i):
        row = 1

        z2 = np.dot(self.v, x_i.T)

        h = np.tanh(z2)
        one_h = np.ones(row, dtype=float)
        h = np.concatenate((h, one_h))

        z3 = np.dot(self.w, h)
        z = self.sigmoid(z3)

        return h, z

    def cross_entropy(self, y, z):
        l = -np.sum(y * np.log(z) + (1 - y) * np.log(1 - z))
        return l

    def tanh_prime(self, a):
        g = 1 - np.tanh(a) ** 2
        return g

    def back_propagation(self, each_x, each_y, each_h, each_z):
        diff = (each_z - each_y).reshape(self.n_outputlayer, 1)
        each_h = each_h.reshape(1, self.n_hiddenlayer + 1)
        dldw = np.dot(diff, each_h)

        dldh = np.dot(self.w.T, diff)

        b = self.tanh_prime(each_h)
        b[:, each_h.shape[1] - 1] = 0
        b = b.reshape(self.n_hiddenlayer + 1, 1)

        mul = np.multiply(dldh, b)
        each_x = each_x.reshape(1, self.n_inputlayer + 1)

        dldv = np.dot(mul, each_x)
        dldv = dldv[:-1, :]

        return dldw, dldv

    def train(self, x, y, epoch, alpha1, alpha2):
        # loss = []
        m = x.shape[0]
        self.v = self.v.astype(float)
        self.w = self.w.astype(float)
        de_v_pre, de_w_pre = 0, 0

        for n in range(epoch):
            for j in range(m):
                x_i = x[j]
                y_i = y[:, j]

                h_i, z_i = self.each_forward(x_i)
                dldw, dldv = self.back_propagation(x_i, y_i, h_i, z_i)

                de_v = alpha1 * dldv
                de_w = alpha1 * dldw
                self.v -= de_v + (alpha2 * de_v_pre)
                self.w -= de_w + (alpha2 * de_w_pre)
                de_v_pre = de_v
                de_w_pre = de_w

                # if j >= 100 and j % 100 == 0:
                    # h_all, z_all = self.forward(x)
                    # loss.append(self.cross_entropy(y, z_all))

        return self.w, self.v #, loss

    def predict(self, test):
        h_all, z_all = self.forward(test)
        y_hat = np.argmax(z_all, axis=0) + 1
        return y_hat

    def accuracy(self, y_hat, y):
        y_hat = y_hat.astype(float)
        y = y.reshape((1, len(y)))
        y = y[0]
        correct_rate = sum(y == y_hat) / len(y)
        return correct_rate

