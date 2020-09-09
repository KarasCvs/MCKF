import numpy as np


class KF():
    def __init__(self):
        pass

    def read_data(self, x_dimension, y_dimension, data):
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.N = data.shape[1]          # 输入数据长度.
        self.X_prior = np.matrix(np.zeros((self.x_dimension, self.N)))
        self.X_posterior = np.matrix(np.zeros((self.x_dimension, self.N)))
        self.KFObservation = np.matrix(np.zeros((self.y_dimension, self.N)))
        self.P_prior = np.zeros((self.x_dimension, self.x_dimension, self.N))
        self.P_posterior = np.zeros((self.x_dimension, self.x_dimension, self.N))
        self.sensor = data

    def ss(self, A, b, c, d, initial=None, input=None, Ts=0.01):
        self.A = A
        self.b = b
        self.c = c
        self.d = d
        if initial is not None:
            self.x_initial = initial
        else:
            self.x_initial = np.matrix(np.zeros((self.x_dimension, 1)))
        if input is not None:
            self.input = input
        else:
            self.input = np.matrix(np.zeros((self.y_dimension, self.N)))
        self.Ts = Ts

    def parmet(self, q=3, r=3):
        self.noise_q = q
        self.noise_r = np.matrix(np.multiply(r, np.eye(self.y_dimension, self.y_dimension)))    # 观测噪音的方差的角矩阵

    def calculation(self):
        for i in range(self.N):
            if i == 0:
                self.X_posterior[:, i] = self.x_initial
                self.P_posterior[:, :, i] = np.matrix(np.eye(self.x_dimension, self.x_dimension))
                self.KFObservation[:, i] = self.c*self.x_initial
            else:
                # Euler method
                # self.X_prior[:, i] = (self.A*self.X_posterior[:, i-1] + self.b*self.input[:, i-1]) *\
                #     self.Ts + self.X_posterior[:, i-1]   # 先验估计 X(x|x-1)
                # self.P_prior[:, :, i] = np.matrix((self.A*self.P_posterior[:, :, i-1] * self.A.T +
                #                                   self.b*self.input[:, i-1] + self.noise_q *
                #                                   np.matrix(np.eye(self.x_dimension, self.x_dimension))) * self.Ts +
                #                                   self.P_posterior[:, :, i-1])        # 先验增益协方差 P(k|k-1)

                # Modified Euler Scheme
                X_prior_temp = (self.A*self.X_posterior[:, i-1] + self.b*self.input[:, i-1]) * self.Ts + self.X_posterior[:, i-1]
                P_prior_temp = np.matrix((self.A*self.P_posterior[:, :, i-1]*self.A.T +
                                          self.b*self.input[:, i-1]) * self.Ts + self.P_posterior[:, :, i-1])
                self.X_prior[:, i] = (self.A*self.X_posterior[:, i-1] + self.b*self.input[:, i-1] + X_prior_temp) *\
                    self.Ts/2 + self.X_posterior[:, i-1]   # 先验估计 X(x|x-1)
                self.P_prior[:, :, i] = np.matrix((self.A*self.P_posterior[:, :, i-1]*self.A.T +
                                                  self.b*self.input[:, i-1] + P_prior_temp)*self.Ts/2 +
                                                  self.P_posterior[:, :, i-1] +
                                                  self.noise_q*np.matrix(np.eye(self.x_dimension, self.x_dimension)))        # 先验增益 P(k|k-1)
                K = self.P_prior[:, :, i] * self.c.T * np.linalg.inv(self.c*self.P_prior[:, :, i]*self.c.T + self.noise_r)
                self.X_posterior[:, i] = self.X_prior[:, i] + K*(self.sensor[:, i] - self.c*self.X_prior[:, i])
                self.P_posterior[:, :, i] = (np.eye(self.x_dimension) - K*self.c) * self.P_prior[:, :, i]
            self.KFObservation[:, i] = self.c * self.X_posterior[:, i]
        return(self.KFObservation)
