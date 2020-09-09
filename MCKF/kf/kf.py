import numpy as np


class UKF():
    def __init__(self):
        pass

    def ss(self, A, c, Ts=0.01):
        self.A = A
        self.c = c
        self.Ts = Ts

    def init(self, x_dimension, y_dimension, q=0, r=3):
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.noise_q = q * np.identity(self.x_dimension)
        self.noise_r = r * np.identity(self.y_dimension)

    def estimate(self, X_prior, data, P):
        sensor = data
        P_prior = self.A*P*self.A.T + self.noise_q    # 先验增益协方差 P(k|k-1)
        K = P_prior * self.c.T * np.linalg.inv(self.c*P_prior*self.c.T + self.noise_r)
        X_posterior = X_prior + K*(sensor - self.c*X_prior)
        P_posterior = (np.eye(self.x_dimension) - K*self.c) * P_prior
        return(X_posterior, P_posterior)
