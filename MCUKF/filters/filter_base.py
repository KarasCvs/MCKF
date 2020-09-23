 # There are atleast 3 ways to calculate W under ut_init,
# No.1: The original one and Katayama's book which will is most stable. Set lambda as a conste.
# No.2: <The Scaled Unscented Transformation> used in robot_localization too,
# but it's not stable enough, could cause the covariance matrix negative definite
# No.3: I tried to use the same way that No.2 to calculate W_m, but make W_c exactly equal
# with W_m. That mean W_m(0) will not be specialization, this is work able but still, not
# stable enough.
import numpy as np


class Filter():
    def __init__(self):
        pass

    def state_func(self, F, H, Ts):
        self.F = F
        self.H = H
        self.Ts = Ts

    def filter_init(self, states_dimension, obs_dimension, q=0, r=3):
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.noise_Q = q*q * np.identity(self.states_dimension)
        self.noise_R = r*r * np.identity(self.obs_dimension)

    def ut_init(self, alpha=1e-3, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        # actually he just have use a constant as lambda, but this is apparently better.
        # self.lambda_ = self.alpha*self.alpha*(self.states_dimension+self.kappa) - self.states_dimension   # No.2
        self.lambda_ = self.states_dimension    # No.1
        self.c_ = self.lambda_ + self.states_dimension                                      # scaling factor
        self.W_mean = (np.hstack(((np.matrix(self.lambda_/self.c_)),
                       0.5/self.c_+np.zeros((1, 2*self.states_dimension))))).A.reshape(self.states_dimension*2+1,)
        self.W_cov = self.W_mean               # No.1 and No.3
        # self.W_cov[0] = self.W_cov[0] + (1-self.alpha*self.alpha+self.beta)   # No.2

    def sigma_points(self, x_prior, P):
        sigma_A_ = np.linalg.cholesky((self.c_) * P)
        sigma_X_ = x_prior * np.ones((1, self.states_dimension))
        X_sigmas = np.hstack((x_prior, sigma_X_+sigma_A_, sigma_X_-sigma_A_))
        return X_sigmas

    def ut(self, transfunc, ut_input, dimension, Noise_cov):
        cols = ut_input.shape[1]
        trans_mean = np.mat(np.zeros((dimension, 1)))
        trans_points = np.mat(np.zeros((dimension, cols)))
        for i in range(cols):
            trans_points[:, i] = transfunc(ut_input[:, i], self.Ts, self.k)
            trans_mean = trans_mean + self.W_mean[i] * trans_points[:, i]
        trans_dev = trans_points - trans_mean*np.ones((1, cols))
        trans_cov = trans_dev*np.diag(self.W_cov)*trans_dev.T + Noise_cov
        return trans_mean, trans_points, trans_cov, trans_dev
