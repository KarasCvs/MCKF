import numpy as np


class UKF():
    def __init__(self):
        pass

    def state_func(self, F, H, Ts=0.1):
        self.F = F
        self.H = H
        self.Ts = Ts

    def filter_init(self, states_dimension, obs_dimension, q=0, r=3):
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.noise_Q = q*q * np.identity(self.states_dimension)
        self.noise_R = r*r * np.identity(self.obs_dimension)

    def ut_init(self, alpha=1e-3, beta=2, ki=0):
        self.alpha = alpha
        self.beta = beta
        self.ki = ki
        self.lambda_ = self.alpha*self.alpha*(self.states_dimension+self.ki) - self.states_dimension
        self.c_ = self.lambda_ + self.states_dimension                                      # scaling factor
        self.W_mean = (np.hstack(((np.matrix(self.lambda_/self.c_)),
                       0.5/self.c_+np.zeros((1, 2*self.states_dimension))))).A.reshape(self.states_dimension*2+1,)
        self.W_cov = self.W_mean
        self.W_cov[0] = self.W_mean[0] + (1 - self.alpha*self.alpha + self.beta)                # only the first one is different

    def sigma_points(self, x_prior, P):
        sigma_A_ = np.linalg.cholesky((self.c_) * P)
        sigma_X_ = x_prior * np.ones((1, self.states_dimension))
        X_sigmas = np.hstack((x_prior, sigma_X_+sigma_A_, sigma_X_-sigma_A_))
        return X_sigmas

    def ut(self, transfunc, ut_input, Noise_cov, dimension, k):
        cols = ut_input.shape[1]
        trans_mean = np.mat(np.zeros((dimension, 1)))
        trans_points = np.mat(np.zeros((dimension, cols)))
        for i in range(cols):
            trans_points[:, i] = transfunc(ut_input[:, i], k, self.Ts)
            trans_mean = trans_mean + self.W_mean[i] * trans_points[:, i]
        trans_dev = trans_points - trans_mean*np.ones((1, cols))
        trans_cov = trans_dev*np.diag(self.W_cov)*trans_dev.T + Noise_cov
        return trans_mean, trans_points, trans_cov, trans_dev

    def estimate(self, x_prior, sensor_data, P, k):
        X_sigmas = self.sigma_points(x_prior, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas, self.noise_Q, self.states_dimension, k)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, x_points, self.noise_R, self.obs_dimension, k)

        P_xz = x_dev*np.diag(self.W_cov)*z_dev.T
        K = P_xz * np.linalg.inv(P_zz)
        x_posterior = x_mean + K*(sensor_data - obs_mean)
        P_posterior = P_xx - K*P_zz*K.T
        return(x_posterior, P_posterior)
