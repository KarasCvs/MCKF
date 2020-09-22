import numpy as np
from filters.filter_base import Filter


class Mcukf(Filter):
    def __init__(self):
        Filter.__init__(self)

    def mc_init(self, sigma, eps):
        self.kernel_G = lambda x: np.exp(-((x*x)/(2*sigma*sigma)))
        self.eps = eps

    def estimate(self, x_prior, sensor_data, P, k):
        # priori
        self.k = k
        X_sigmas = self.sigma_points(x_prior, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas, self.states_dimension, self.noise_Q)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, x_points, self.noise_R, self.obs_dimension)
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas, self.obs_dimension, self.noise_R)
        # posterior
        P_xz = x_dev*np.diag(self.W_cov)*z_dev.T
        L = self.kernel_G(np.linalg.norm((sensor_data - obs_mean))*np.linalg.inv(self.noise_R)) / \
            self.kernel_G(np.linalg.norm((x_mean-self.F(x_prior)))*np.linalg.inv(P_xx))
        K = (np.linalg.inv(P_xx) + L*)
        x_posterior = x_mean + K*(sensor_data - obs_mean)
        P_posterior = P_xx - K*P_zz*K.T
        return x_posterior, P_posterior

    def mc(self, E):
        entropy_x = np.zeros(self.states_dimension)
        entropy_y = np.zeros(self.obs_dimension)
        for i in range(self.states_dimension):
            entropy_x[i] = self.kernel_G(E[i])
            if entropy_x[i] < 1e-9:
                entropy_x[i] = 1e-9
        for i in range(self.obs_dimension):
            entropy_y[i] = self.kernel_G(E[i+self.states_dimension])
            if entropy_y[i] < 1e-9:
                entropy_y[i] = 1e-9
        return np.diag(entropy_x), np.diag(entropy_y)
