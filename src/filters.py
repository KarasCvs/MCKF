# There are atleast 3 ways to calculate W under ut_init,
# No.1: The original one and Katayama's book which will is most stable. Set lambda as a conste.
# No.2: <The Scaled Unscented Transformation> used in robot_localization too,
# but it's not stable enough, could cause the covariance matrix negative definite
# No.3: I tried to use the same way that No.2 to calculate W_m, but make W_c exactly equal
# with W_m. That mean W_m(0) will not be specialization, this is work able but still, not
# stable enough.
import time
import sympy as sy
import numpy as np
from numpy.random import randn
from numpy.linalg import inv, cholesky
from functions import NonLinearFunc as Func
# from functions import MoveSim as Func
import matplotlib.pyplot as plt


class LinearSys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
        self.func = Func()
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.real_obs = np.mat(np.zeros((obs_dimension, self.N)))
        self.state_noise = np.mat(q_ * randn(self.states_dimension, self.N))
        self.noise_r = r_

    # Generate noise lists.
    def noise_init(self, repeat=1, additional_noise=0):
        self.obs_noise = [np.mat(self.noise_r*randn(self.obs_dimension, self.N)
                                 + additional_noise) for _ in range(repeat)]
        return self.obs_noise

    def states_init(self, X0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.func.obs_matrix(self.states[:, 0])
        self.real_obs[:, 0] = self.func.observation_func(self.states[:, 0])

    def run(self):
        for i in range(1, self.N):
            self.func.state_matrix(self.states[:, i-1], self.Ts, i)
            self.states[:, i] = self.func.state_func(self.states[:, i-1], self.Ts, i) + self.state_noise[:, i]
            self.func.obs_matrix(self.states[:, i])
            self.real_obs[:, i] = self.func.observation_func(self.states[:, i])
        return self.time_line, self.states, self.real_obs

    def plot(self):
        for i in range(self.states_dimension):
            plt.subplot(100*self.states_dimension+11+i)
            plt.plot(self.time_line, np.array(self.states)[i, :].reshape(self.N,), linewidth=1, linestyle="-", label="system states")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(f"States {i}")
        plt.show()


class NonlinearSys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
        self.func = Func()
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.real_obs = np.mat(np.zeros((obs_dimension, self.N)))
        self.state_noise = np.mat(q_ * randn(self.states_dimension, self.N))
        self.noise_r = r_

    # Generate noise lists.
    def noise_init(self, repeat=1, additional_noise=0):
        self.obs_noise = [np.mat(self.noise_r*randn(self.obs_dimension, self.N)
                                 + additional_noise[i]) for i in range(repeat)]
        return self.obs_noise

    def states_init(self, X0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.real_obs[:, 0] = self.func.observation_func(self.states[:, 0])

    def run(self):
        for i in range(1, self.N):
            self.states[:, i] = self.func.state_func(self.states[:, i-1], self.Ts, i) + self.state_noise[:, i]
            self.real_obs[:, i] = self.func.observation_func(self.states[:, i])
        return self.time_line, self.states, self.real_obs

    def plot(self):
        for i in range(self.states_dimension):
            plt.subplot(100*self.states_dimension+11+i)
            plt.plot(self.time_line, np.array(self.states)[i, :].reshape(self.N,), linewidth=1, linestyle="-", label="system states")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(f"States {i}")
        plt.show()


class Filter():
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
        self.func = Func()
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.mse1 = np.mat(np.zeros((states_dimension, self.N)))
        self.noise_q = q_*np.ones((self.states_dimension, 1))
        self.noise_r = r_*np.ones((self.obs_dimension, 1))
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.noise_Q = self.noise_q**2 * np.diag(np.ones((self.states_dimension)))
        self.noise_R = self.noise_r**2 * np.diag(np.ones((self.obs_dimension)))
        self.F = self.func.state_func
        self.H = self.func.observation_func

    def states_init(self, init_parameters):
        filter0, P0 = init_parameters
        self.states[:, 0] = np.array(filter0).reshape(self.states_dimension, 1)
        self.P = np.diag(P0)

    def read_data(self, states, obs):
        self.real_states = states
        self.obs = obs

    def MSE(self):
        for i in range(1, self.N):
            self.mse1[:, i] = self.real_states[:, i] - self.states[:, i]
            self.mse1[:, i] = np.power(self.mse1[:, i], 2)
        return self.mse1

    def mc_init(self, sigma, eps=1e-6):
        self.sigma = sigma
        self.eps = eps

    def kernel_G(self, x):
        res = np.exp(-(np.linalg.norm(x)/(2*(self.sigma**2))))
        if res < 1e-4:
            res = 1e-4
        return res

    def ut_init(self, alpha=1e-3, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        # lambda can be calculated by No.2 or just let it to be a const as No.1
        # self.lambda_ = self.alpha**2*(self.states_dimension+self.kappa) - self.states_dimension   # No.2
        self.lambda_ = 6   # No.1
        self.c_ = self.lambda_ + self.states_dimension                                      # scaling factor
        self.W_mean = (np.hstack(((np.matrix(self.lambda_/self.c_)),
                       1/(2*self.c_) + np.zeros((1, 2*self.states_dimension))
                       ))).A.reshape(self.states_dimension*2+1,)
        # self.W_mean = (np.hstack(((np.matrix(1-self.states_dimension-self.alpha**2*self.kappa)),
        #                1/(2*(self.alpha**2*self.kappa)) + np.zeros((1, 2*self.states_dimension))
        #                ))).A.reshape(self.states_dimension*2+1,)
        self.W_cov = self.W_mean               # No.1 and No.3
        # self.W_cov[0] = self.W_mean[0] + (1-self.alpha**2+self.beta)   # No.2

    def sigma_points(self, x_previous, P):
        sigma_A_ = np.linalg.cholesky((self.c_) * P)
        # sigma_A_ = self.alpha*self.kappa*np.linalg.cholesky(P)
        sigma_X_ = x_previous * np.ones((1, self.states_dimension))
        X_sigmas = np.hstack((x_previous, sigma_X_+sigma_A_, sigma_X_-sigma_A_))
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

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        mc_count = 0
        states_mean = 0
        mse1 = 0
        start = time.clock()
        for j in range(repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P, count = \
                    self.estimate(self.states[:, i-1],
                                  self.obs[:, i]+obs_noise[j][:, i],
                                  self.P, i)
                mc_count += count
            states_mean += self.states
            mse1 += self.MSE()
        end = time.clock()
        states_mean /= repeat
        mse1 /= repeat
        mse = mse1.sum(axis=1)/self.N
        mc_count /= self.N*repeat
        self.run_time = end - start
        return states_mean, mse1, mse, mc_count, self.time_line


class Mcukf(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.ut_init(alpha, beta, kappa)
        self.mc_init(sigma, eps)

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        X_sigmas = self.sigma_points(x_previous, P)
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


# This is a version that I'm trying to use Dan.S's method on Ukf.
class Mcukf2(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.ut_init(alpha, beta, kappa)
        self.mc_init(sigma, eps)

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        X_sigmas = self.sigma_points(x_previous, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas, self.states_dimension, self.noise_Q)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, x_points, self.obs_dimension, self.noise_R)
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas, self.obs_dimension, self.noise_R)
        # posterior
        P_xz = x_dev*np.diag(self.W_cov)*z_dev.T
        H = inv(P_xx)*P_xz
        # 先测试用obs_mean和p_zz的组合计算L
        L = self.kernel_G(np.linalg.norm((sensor_data - obs_mean))*inv(self.noise_R)) / \
            self.kernel_G(np.linalg.norm((x_mean-self.F(x_previous, self.Ts)))*inv(P_xx))
        K = inv(P_xx+(P_zz-L*H*self.noise_R*H.T))*H*inv(self.noise_R)
        x_posterior = x_mean + K*(sensor_data - obs_mean)
        P_posterior = (np.eye(self.states_dimension)-K*H.T)*P_xx*(np.eye(self.states_dimension)-K*H.T).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, 0

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


class Ukf(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.ut_init(alpha, beta, kappa)

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        X_sigmas = self.sigma_points(x_previous, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas, self.states_dimension, self.noise_Q)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, x_points, self.obs_dimension, self.noise_R)
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas, self.obs_dimension, self.noise_R)
        # posterior
        P_xz = x_dev*np.diag(self.W_cov)*z_dev.T
        K = P_xz * np.linalg.inv(P_zz)
        x_posterior = x_mean + K*(sensor_data - obs_mean)
        P_posterior = P_xx - K*P_zz*K.T
        return x_posterior, P_posterior, 0


# Fixed Point Iteration
class Mckf1(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, sigma, eps):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mc_init(sigma, eps)

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        F = self.func.state_matrix(x_previous, self.Ts)
        x_prior = F * x_previous
        P = F * P * F.T + self.noise_Q
        H = self.func.obs_matrix(x_prior)
        # posterior
        P_sqrt = cholesky(P)
        R_sqrt = cholesky(self.noise_R)
        B = np.hstack((np.vstack((P_sqrt, np.zeros((self.obs_dimension, self.states_dimension)))),
                       np.vstack((np.zeros((self.states_dimension, self.obs_dimension)), R_sqrt))))
        B_inv = inv(B)
        W = B_inv*np.vstack((np.identity(self.states_dimension), H))
        D = B_inv*np.vstack((x_prior, sensor_data))
        X_temp = x_prior
        Evaluation = 1
        mc_count = 0
        while Evaluation > self.eps:
            E = D - W*X_temp
            Cx, Cy = self.mc(E)
            P_mc = P_sqrt*inv(Cx)*P_sqrt
            R_mc = R_sqrt*inv(Cy)*R_sqrt
            K = P_sqrt*H.T*inv(H*P_mc*H.T+R_mc)
            x_posterior = x_prior + K*(sensor_data - H*x_prior)
            Evaluation = np.linalg.norm(x_posterior - X_temp)/np.linalg.norm(X_temp)
            X_temp = x_posterior
            mc_count += 1
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, mc_count

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


# Dan.S method, this works.
class Mckf2(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, sigma, eps):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mc_init(sigma, eps)

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        F = self.func.state_matrix(x_previous, self.Ts)
        x_prior = F * x_previous
        # For time-variant system
        P = F * P * F.T + self.noise_Q
        # posterior
        H = self.func.obs_matrix(x_prior)
        L = self.kernel_G(np.linalg.norm((x_prior - F*x_previous))*inv(P)) / \
            self.kernel_G(np.linalg.norm((sensor_data - H*x_prior))*inv(self.noise_R))
        K = inv(L*inv(P) + (H.T*inv(self.noise_R)*H))*H.T*inv(self.noise_R)
        x_posterior = x_prior + K*(sensor_data - H*x_prior)
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, 0


class Ekf(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)

    # Calculate Jacobian matrix, for EKF. I think this is a correct version
    # compared the calculate by hand one in <functions>.
    def jacobian(self, function, length):
        args = []
        for i in range(length):
            exec(f"x{i} = sy.symbols(f'x{i}')")
            exec(f"args.append(x{i})")
        variables = sy.Matrix(args)
        function = sy.Matrix(function(args, self.Ts))
        jacobian = function.jacobian(variables)
        return jacobian

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts)
        H = self.func.obs_jacobian(x_prior)
        # For time-variant system
        P = F * P * F.T + self.noise_Q
        # posterior
        K = P*H.T*inv(H*P*H.T + self.noise_R)
        x_posterior = x_prior + K*(sensor_data - self.func.observation_func(x_prior))
        P_posterior = (np.eye(self.states_dimension)-K*H)*P
        return x_posterior, P_posterior, 0


# Fixed Point Iteration
class Mcekf1(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, sigma, eps=0):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mc_init(sigma, eps)

    # Calculate Jacobian matrix, for EKF. I think this is a correct version
    # compared the calculate by hand one in <functions>.
    def jacobian(self, function, length):
        args = []
        for i in range(length):
            exec(f"x{i} = sy.symbols(f'x{i}')")
            exec(f"args.append(x{i})")
        variables = sy.Matrix(args)
        function = sy.Matrix(function(args, self.Ts))
        jacobian = function.jacobian(variables)
        return jacobian

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts)
        obs = self.func.observation_func(x_prior)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts)
        H = self.func.obs_jacobian(x_prior)
        # For time-variant system
        P = F * P * F.T + self.noise_Q
        # posterior
        P_sqrt = cholesky(P)
        R_sqrt = cholesky(self.noise_R)
        B = np.hstack((np.vstack((P_sqrt, np.zeros((self.obs_dimension, self.states_dimension)))),
                       np.vstack((np.zeros((self.states_dimension, self.obs_dimension)), R_sqrt))))
        B_inv = inv(B)
        W = B_inv*np.vstack((np.identity(self.states_dimension), H))
        D = B_inv*np.vstack((x_prior, sensor_data))
        X_temp = x_prior
        Evaluation = 1
        mc_count = 0
        while Evaluation > self.eps:
            E = D - W*X_temp
            Cx, Cy = self.mc(E)
            P_mc = P_sqrt*inv(Cx)*P_sqrt
            R_mc = R_sqrt*inv(Cy)*R_sqrt
            K = P_sqrt*H.T*inv(H*P_mc*H.T+R_mc)
            x_posterior = x_prior + K*(sensor_data - obs)
            Evaluation = np.linalg.norm(x_posterior - X_temp)/np.linalg.norm(X_temp)
            X_temp = x_posterior
            mc_count += 1
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, mc_count

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


# Dan.S's method
class Mcekf2(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, sigma, eps=0):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mc_init(sigma)

    # Calculate Jacobian matrix, for EKF. I think this is a correct version
    # compared the calculate by hand one in <functions>.
    def jacobian(self, function, length):
        args = []
        for i in range(length):
            exec(f"x{i} = sy.symbols(f'x{i}')")
            exec(f"args.append(x{i})")
        variables = sy.Matrix(args)
        function = sy.Matrix(function(args, self.Ts))
        jacobian = function.jacobian(variables)
        return jacobian

    def estimate(self, x_previous, sensor_data, P, k):
        # priori
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts)
        obs = self.func.observation_func(x_prior)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts)
        H = self.func.obs_jacobian(x_prior)
        P = F * P * F.T + self.noise_Q
        # posterior
        # The calculation of L, denominator should be the error of states which can be instead with Q.
        L = self.kernel_G(np.linalg.norm(sensor_data - obs)*inv(cholesky(self.noise_R))) / \
            self.kernel_G(np.linalg.norm(F*self.noise_q)*inv(cholesky(P)))  # x_prior - self.func.state_func(x_previous, self.Ts)
        K = inv(inv(P) + (L*H.T*inv(self.noise_R)*H))*L*H.T*inv(self.noise_R)
        x_posterior = x_prior + K*(sensor_data - obs)
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, 0
