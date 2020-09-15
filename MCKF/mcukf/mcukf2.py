# This is a ukf based on the robot_localization and
# <Major development under Gaussian filtering since unscented Kalman filter> has used somthing likely
# The biggest different between this and Jeffrey Uhlmann is
# the calculation of covariance weights.
# This method use alpha beta and kappa to calculate lambda which is little bigger than -1*states_dimension
# This method should have a better result but could cause P_xx negative_definite when
# Ts is not small enough.
import numpy as np
from numpy.linalg import cholesky


class MCUKF():
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

    def ut_init(self, alpha=1e-3, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        # actually he just have use a constant as lambda, but this is apparently better.
        self.lambda_ = self.alpha*self.alpha*(self.states_dimension+self.kappa) - self.states_dimension
        self.c_ = self.lambda_ + self.states_dimension                                      # scaling factor
        self.W_mean = (np.hstack(((np.matrix(self.lambda_/self.c_)),
                       0.5/self.c_+np.zeros((1, 2*self.states_dimension))))).A.reshape(self.states_dimension*2+1,)
        self.W_cov = self.W_mean    # Different with robot_localization
        self.W_cov[0] = self.W_cov[0] + (1-self.alpha*self.alpha+self.beta)

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
        return(x_posterior, P_posterior)

    def sigma_points(self, x_prior, P):
        sigma_A_ = cholesky((self.c_) * P)
        sigma_X_ = x_prior * np.ones((1, self.states_dimension))
        X_sigmas = np.hstack((x_prior, sigma_X_+sigma_A_, sigma_X_-sigma_A_))
        return X_sigmas

    def ut(self, transfunc, ut_input, dimension, Noise_cov=0):
        cols = ut_input.shape[1]
        trans_mean = np.mat(np.zeros((dimension, 1)))
        trans_points = np.mat(np.zeros((dimension, cols)))
        for i in range(cols):
            trans_points[:, i] = transfunc(ut_input[:, i], self.Ts, self.k)
            trans_mean = trans_mean + self.W_mean[i] * trans_points[:, i]
        trans_dev = trans_points - trans_mean*np.ones((1, cols))
        trans_cov = trans_dev*np.diag(self.W_cov)*trans_dev.T + Noise_cov
        try:
            cholesky(trans_cov)
        except:
            print("here")
        return trans_mean, trans_points, trans_cov, trans_dev

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
