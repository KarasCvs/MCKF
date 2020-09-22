# This is a mcukf based on the paper oldest references.
# Like the original one and Katayama's book
# Tis method use a self-definition lambda, I set it equal to states dimension
import numpy as np
from numpy.linalg import cholesky
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
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas, self.obs_dimension, self.noise_R)
        # posterior
        P_xz = x_dev*np.diag(self.W_cov)*z_dev.T
        # MC part
        Sp_mc = cholesky(P_xx)
        Sr_mc = cholesky(self.noise_R)    # 这里是不是P_zz?
        S_mc = np.hstack((np.vstack((Sp_mc, np.zeros((self.obs_dimension, self.states_dimension)))),
                          np.vstack((np.zeros((self.states_dimension, self.obs_dimension)), Sr_mc))))
        S_mc_inv = np.linalg.inv(S_mc)
        H_mc = (np.linalg.inv(P_xx) * P_xz).T
        W_mc = S_mc_inv*np.vstack((np.identity(self.states_dimension), H_mc))
        D_mc = S_mc_inv*np.vstack((x_mean, sensor_data - obs_mean + H_mc*x_mean))
        x_init_mc = np.linalg.inv(W_mc.T*W_mc)*W_mc.T*D_mc
        # x_init_mc = x_mean
        Evaluation = 1
        count = 0
        x_old_mc = x_init_mc
        while Evaluation > self.eps:
            E_mc = D_mc - W_mc*x_old_mc
            Cx_mc, Cy_mc = self.mc(E_mc)
            P_mc = Sp_mc*np.linalg.inv(Cx_mc)*Sp_mc.T
            R_mc = Sr_mc*np.linalg.inv(Cy_mc)*Sr_mc.T
            K = P_mc*H_mc.T*np.linalg.inv(H_mc*P_mc*H_mc.T+R_mc)
            x_new_mc = x_mean + K*(sensor_data-obs_mean)
            Evaluation = np.linalg.norm(x_new_mc - x_old_mc)/np.linalg.norm(x_old_mc)
            x_old_mc = x_new_mc
            count += 1
        x_posterior = x_new_mc
        P_posterior = (np.eye(self.states_dimension)-K*H_mc)*P_xx*(np.eye(self.states_dimension)-K*H_mc).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, count

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
