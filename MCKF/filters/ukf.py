import numpy as np
from filters.filter_base import Filter


class Ukf(Filter):
    def __init__(self):
        Filter.__init__(self)

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
        K = P_xz * np.linalg.inv(P_zz)
        x_posterior = x_mean + K*(sensor_data - obs_mean)
        P_posterior = P_xx - K*P_zz*K.T
        return x_posterior, P_posterior
