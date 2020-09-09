import numpy as np
from numpy.random import randn
from scipy.linalg import ldl


# MCKF
class MCKF():
    def __init__(self):
        self.sig = 2
        self.eps = 0.002

    def read_data(self, x_dimension, y_dimension, data):
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.N = data.shape[1]          # 输入数据长度.
        self.X_prior = np.matrix(np.zeros((self.x_dimension, self.N)))
        self.X_posterior = np.matrix(np.zeros((self.x_dimension, self.N)))
        self.g = np.matrix(np.zeros((self.x_dimension, self.N)))
        self.MCKFObservation = np.matrix(np.zeros((self.y_dimension, self.N)))
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

    def parmet(self, sig, eps, q=3, r=3):
        self.sig = sig
        self.eps = eps
        self.noise_q = q
        self.noise_r = np.matrix(np.multiply(r, np.eye(self.y_dimension, self.y_dimension)))    # 观测噪音的方差的角矩阵

    def G(self, sig, e):          # Correntropy calculation
        result = float(np.exp(-((float(e)**2)/(2*(float(sig)**2)))))
        if result < 1.0e-6:
            result = 1.0e-6
        return result

    def calculation(self):  # H 就是c.T观测矩阵
        # self.noise_state = np.sqrt(self.noise_q) * randn(self.x_dimension, self.N)
        # self.noise_observation = np.sqrt(self.noise_r) * randn(self.y_dimension, self.N)
        for i in range(self.N):
            if i < 2:
                self.X_posterior[:, i] = self.x_initial
                self.P_posterior[:, :, i] = np.matrix(np.eye(self.x_dimension, self.x_dimension))
                self.MCKFObservation[:, i] = self.c*self.x_initial
            else:
                # Euler method
                # self.X_prior[:, i] = (self.A*self.X_posterior[:, i-1] + self.b*self.input[:, i-1]) * self.Ts + self.X_posterior[:, i-1]   # 先验估计 X(x|x-1)
                # self.P_prior[:, :, i] = np.matrix((self.A*self.P_posterior[:, :, i-1]*self.A.T +
                #                                   self.b*self.input[:, i-1] + self.noise_q *
                #                                   np.matrix(np.eye(self.x_dimension, self.x_dimension))) * self.Ts +
                #                                   self.P_posterior[:, :, i-1])        # 先验增益 P(k|k-1)

                # Bilinear transform  公式有问题, 明天再说吧..
                # self.X_prior[:, i] = np.linalg.inv(np.eye(self.x_dimension, self.x_dimension) -
                #                                    self.Ts/2*self.A) *\
                #                                   (self.Ts/2*(self.A*self.X_prior[:, i-1] +
                #                                    self.b*(self.input[:, i-1]+self.input[:, i])) + self.X_prior[:, i-1])
                # self.P_prior[:, :, i] = self.Ts/2 * (self.A*self.P_posterior[:, :, i-1]*self.A.T +
                #                                      self.A*self.P_posterior[:, :, i-2]*self.A.T +
                #                                      self.b*(self.input[:, i]+self.input[:, i-1]) +
                #                                      2*self.noise_q*np.eye(self.x_dimension, self.x_dimension)) +\
                #     np.eye(self.x_dimension, self.x_dimension)

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
                try:
                    B_p = np.linalg.cholesky(self.P_prior[:, :, i])
                    B_r = np.linalg.cholesky(self.noise_r)
                except np.linalg.LinAlgError:       # choleskey分解失败的话用ldl代替
                    try:
                        P_ldl = ldl(self.P_prior[:, :, i])
                        R_ldl = ldl(self.noise_r)
                        B_p = P_ldl[0] * np.linalg.cholesky(P_ldl[1])              # B_p矩阵, 对先验增益的cholesky分解. 这里我是左右两部分进行cholesky分解之后, 再拼起来的.
                        B_r = R_ldl[0] * np.linalg.cholesky(R_ldl[1])
                    except np.linalg.LinAlgError:
                        B_p = 0
                        B_r = 0
                B = np.hstack((np.vstack((B_p, np.zeros((self.y_dimension, self.x_dimension)))),
                               np.vstack((np.zeros((self.x_dimension, self.y_dimension)), B_r))))  # 这里构建B矩阵
                try:
                    D = B.I * np.vstack((self.X_prior[:, i], self.sensor[:, i]))        # 简化公式
                    W = B.I * np.vstack((np.ones((self.x_dimension, self.x_dimension)), self.c))            # 简化公式
                except np.linalg.LinAlgError:     # 若矩阵B不可逆, 则用广义逆代替
                    D = np.linalg.pinv(B) * np.vstack((self.X_prior[:, i], self.sensor[:, i]))
                    W = np.linalg.pinv(B) * np.vstack((np.ones((self.x_dimension, self.x_dimension)), self.c))
                X_posterior_temp_last = self.X_prior[:, i]        # 初始化X_t(0), 令其等于先验
                EstimateRate = 1
                while EstimateRate > self.eps:
                    e = D - W*X_posterior_temp_last                                           # 退出条件, 后验与上一时刻后验之比大于ep
                    entropy_x = []
                    entropy_y = []
                    for k in range(self.x_dimension):                           # 计算熵和C_x,C_y矩阵  问题出在这, D如果是3+1维的话, L=4, 没有D(4)来计算, 也就没有C_y
                        entropy_x.append(self.G(self.sig, e[k]))
                    C_x = np.matrix(np.diag(entropy_x))
                    if self.y_dimension == 1:
                        C_y = np.matrix(self.G(self.sig, e[self.x_dimension]))
                    else:
                        for k in range(self.x_dimension, self.x_dimension + self.y_dimension):  # L = x维度+y维度
                            entropy_y.append(self.G(self.sig, e[k]))
                        C_y = np.matrix(np.diag(entropy_y))
                    try:
                        R = B_r*C_y.I*B_r.T
                        P = B_p*C_x.I*B_p.T
                    except np.linalg.LinAlgError:
                        R = B_r*np.linalg.pinv(C_y)*B_r.T
                        P = B_p*np.linalg.pinv(C_x)*B_p.T
                    K = P*self.c.T*np.linalg.inv(self.c*P*self.c.T+R)
                    X_posterior_temp_now = np.matrix(self.X_prior[:, i] + K*(self.sensor[:, i]-self.c*self.X_prior[:, i]))
                    EstimateRate = np.linalg.norm(X_posterior_temp_now-X_posterior_temp_last) /\
                        np.linalg.norm(X_posterior_temp_last)
                    X_posterior_temp_last = X_posterior_temp_now
                self.X_posterior[:, i] = X_posterior_temp_now
                if R.any() == np.inf:
                    self.P_posterior[:, :, i] = (np.eye(self.x_dimension, self.x_dimension)-K*self.c) *\
                        self.P_prior[:, :, i]*(np.eye(self.x_dimension, self.x_dimension)-K*self.c).T         # 重定义R, 防止R=inf时造成计算出错. 因为R=inf的时候K=0
                else:
                    self.P_posterior[:, :, i] = (np.eye(self.x_dimension, self.x_dimension)-K*self.c) *\
                        self.P_prior[:, :, i]*(np.eye(self.x_dimension, self.x_dimension)-K*self.c).T + K*R*K.T
                self.MCKFObservation[:, i] = self.c*self.X_posterior[:, i]
        return self.MCKFObservation
