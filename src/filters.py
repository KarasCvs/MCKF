# There are atleast 3 ways to calculate W under ut_init,
# No.1: The original one and Katayama's book which will is most stable. Set lambda as a conste.
# No.2: <The Scaled Unscented Transformation> used in robot_localization too,
# but it's not stable enough, could cause the covariance matrix negative definite
# No.3: I tried to use the same way that No.2 to calculate W_m, but make W_c exactly equal
# with W_m. That mean W_m(0) will not be specialization, this is work able but still, not
# stable enough.
import time
import numpy as np
from numpy.random import randn
from numpy.linalg import inv, cholesky
from functions import NonLinearFunc1 as Func
# from functions import MoveSim as Func
import matplotlib.pyplot as plt
import math


def in_log_dec(keyword):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            name = self.tag + ', ' + keyword
            if name not in self.data_dic["others"]:
                defult = eval(f"{keyword}")
                self.data_dic['others'][name] = [[defult] for _ in range(self.repeat)]
            exec(f"self.data_dic['others'][name][self.repeat_count].append({keyword})")
            return(res)
        return wrapper
    return decorator


class LinearSys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
        self.func = Func()
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t / self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.real_obs = np.mat(np.zeros((obs_dimension, self.N)))
        self.state_noise = np.mat(q_ * randn(self.states_dimension, self.N))
        self.std_R = r_

    # Generate noise lists.
    def noise_init(self, repeat, additional_obs_noise=0):
        self.obs_noise = [
            np.mat(self.std_R * randn(self.obs_dimension, self.N) +
                   additional_obs_noise) for _ in range(repeat)
        ]
        return self.obs_noise

    def states_init(self, X0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.func.obs_matrix(self.states[:, 0])
        self.real_obs[:, 0] = self.func.observation_func(self.states[:, 0])

    def run(self):
        for i in range(1, self.N):
            self.func.state_matrix(self.states[:, i - 1], self.Ts, i)
            self.states[:, i] = self.func.state_func(
                self.states[:, i - 1], self.Ts, i) + self.state_noise[:, i]
            self.func.obs_matrix(self.states[:, i])
            self.real_obs[:, i] = self.func.observation_func(self.states[:, i])
        return self.time_line, self.states, self.real_obs

    def plot(self):
        for i in range(self.states_dimension):
            plt.subplot(100 * self.states_dimension + 11 + i)
            plt.plot(self.time_line,
                     np.array(self.states)[i, :].reshape(self.N, ),
                     linewidth=1,
                     linestyle="-",
                     label="system states")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(f"States {i}")
        plt.show()


class NonlinearSys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, input, repeat=1):
        self.func = Func(input)
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t / self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.real_states = []  # np.mat(np.zeros((states_dimension, self.N)))
        self.real_obs = []  # np.mat(np.zeros((obs_dimension, self.N)))
        self.sensor = []
        self.std_Q = q_  # np.mat(np.dot(q_, randn(self.states_dimension, self.N)))
        self.std_R = r_
        self.repeat = repeat

    # Generate noise lists.
    def noise_init(self, additional_sys_noise=0, additional_obs_noise=0):
        self.sys_noise = [
            np.mat(np.dot(self.std_Q, randn(self.states_dimension, self.N)) +
                   additional_sys_noise[i]) for i in range(self.repeat)
        ]
        self.obs_noise = [
            np.mat(self.std_R * randn(self.obs_dimension, self.N) +
                   additional_obs_noise[i]) for i in range(self.repeat)
        ]
        return self.sys_noise, self.obs_noise

    def states_init(self, X0):
        for j in range(self.repeat):
            # Build list
            self.real_states.append(np.asmatrix(np.zeros((self.states_dimension, self.N))))
            self.real_obs.append(np.asmatrix(np.zeros((self.obs_dimension, self.N))))
            # Initial the first object.
            self.real_states[j][:, 0] = np.matrix(X0).reshape(self.states_dimension, 1)
            self.real_obs[j][:, 0] = self.func.observation_func(self.real_states[j][:, 0])

    def run(self):
        for j in range(self.repeat):
            for i in range(1, self.N):
                self.real_states[j][:, i] = self.func.state_func(
                    self.real_states[j][:, i - 1]+self.sys_noise[j][:, i], self.Ts, i)
                self.real_obs[j][:, i] = self.func.observation_func(self.real_states[j][:, i])
            self.sensor.append(self.real_obs[j] + self.obs_noise[j])
        return self.time_line, self.real_states, self.real_obs, self.sensor

    def plot(self):
        plt.figure(1)
        for i in range(self.states_dimension):
            plt.subplot(100 * self.states_dimension + 11 + i)
            plt.plot(self.time_line,
                     np.array(self.real_states[0])[i, :].reshape(self.N,),
                     linewidth=1,
                     linestyle="-",
                     label="system states")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title(f"X{i+1}")
        plt.figure(2)
        plt.plot(self.time_line, np.array(self.sensor[0]).reshape(self.N,))
        plt.show()


class Filter():
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input):
        self.func = Func(input)
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t / self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.mse = np.mat(np.zeros((states_dimension, self.N)))
        self.std_Q = q_
        self.std_R = np.array(r_)
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.cov_Q = np.matrix(self.std_Q*self.std_Q.T)
        self.cov_R = np.matrix(self.std_R*self.std_R.T)
        self.F = self.func.state_func
        self.H = self.func.observation_func
        self.alpha = None
        self.beta = None
        self.eps = None
        self.kappa = None
        self.tag = "filter"
        self.repeat = repeat
        self.data_dic = data_dic

    def states_init(self, init_parameters):
        filter0, P0 = init_parameters
        self.states[:, 0] = np.array(filter0).reshape(self.states_dimension, 1)
        self.P = np.diag(P0)
        if self.shift_bandwidth:
            self.sigma_square = 400
            self.sigma_square_R = 100
            self.sigma_square_Q = 100

    def summarizes(self, states_mean, mse, ta_mse, run_time):
        self.data_dic['states'][self.tag+' states'] = states_mean.tolist()
        self.data_dic['ta_mse'][self.tag+' ta_mse'] = ta_mse.tolist()
        self.data_dic['mse'][self.tag+' mse'] = mse.tolist()
        self.data_dic['parameters'][self.tag+' parameters'] = {'alpha': self.alpha, 'beta': self.beta, 'kappa': self.kappa}
        self.data_dic['run time'][self.tag+' run time'] = run_time

    def run(self, init_parameters, real_states, sensor, shift_bandwidth=0):
        # --------------------------------main procedure---------------------------------- #
        self.shift_bandwidth = shift_bandwidth
        mc_count = 0
        states_mean = 0
        mse = 0
        start = time.clock()
        print(f"{self.tag} running.")
        for j in range(self.repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P, count = \
                    self.estimate(self.states[:, i-1],
                                  sensor[j][:, i],
                                  self.P, i, j)
                mc_count += count
            states_mean += self.states
            mse += self.MSE(real_states[j])
        end = time.clock()
        states_mean /= self.repeat
        mse /= self.repeat
        ta_mse = mse.sum(axis=1) / self.N
        mc_count /= self.N * self.repeat
        run_time = end - start
        self.summarizes(states_mean, mse, ta_mse, run_time)
        return self.data_dic

    def MSE(self, real_states):
        for i in range(1, self.N):
            self.mse[:, i] = real_states[:, i] - self.states[:, i]
            self.mse[:, i] = np.power(self.mse[:, i], 2)
        return self.mse

    def mc_init(self, sigma, eps=1e-6):
        self.sigma_square = sigma**2
        self.sigma_square_R = self.sigma_square
        self.sigma_square_Q = (sigma)**2
        self.eps = eps

    def kernel_G(self, e, error=0, u=0):
        if self.shift_bandwidth:
            self.shift_sigma(self.sigma_square, e, u)
        res = np.asscalar(np.exp(-((e**2) / (2*self.sigma_square))))
        return res

    def kernel_G_R(self, e, error=0, u=0, cov=0):
        if self.shift_bandwidth:
            self.sigma_square_R = self.shift_sigma(self.sigma_square_R, e, u, cov)
        res = np.asscalar(np.exp(-((e**2) / (2*self.sigma_square_R))))
        self.in_log_func(e, 'e')
        return res

    def kernel_G_Q(self, e, error=0, u=0, cov=0):
        if self.shift_bandwidth:
            self.sigma_square_Q = self.shift_sigma(self.sigma_square_Q, error, u, cov)
        res = np.asscalar(np.exp(-((e**2) / (2*self.sigma_square_Q))))
        if res < 1e-8:
            res = 1e-8
        return res

#   可变bandwidth MCC
    def shift_sigma(self, sigma_square, error, u, cov, alpha_=0.9):
        e_square = np.dot(error.T, error)
        sigma_X = (e_square - cov) / (1e-3*np.linalg.norm(u)**2 * e_square)  # 常数是步长
        if math.isnan(sigma_X):   # 矩阵维度有问题, 如何让矩阵全部变成一维?
            pass
        if 0 <= sigma_X and sigma_X < 1:
            sigma_main = -e_square / (2 * np.log(sigma_X))  # 如何对矩阵求对数? 如果e是矩阵那么sigma_X自然也是矩阵. 按照论文, 噪音也应当是矩阵.
            sigma_temp = alpha_ * sigma_square + (1 - alpha_) * min(sigma_main, sigma_square)
            if type(sigma_temp) is np.matrix:
                sigma_temp = np.asscalar(sigma_temp)
            sigma_square = sigma_temp
        return sigma_square

    def mc(self, E, u=0):
        entropy_x = np.zeros(self.states_dimension)
        entropy_y = np.zeros(self.obs_dimension)
        for i in range(self.states_dimension):
            entropy_x[i] = self.kernel_G(E[i], u)
            if entropy_x[i] < 1e-9:
                entropy_x[i] = 1e-9
        for i in range(self.obs_dimension):
            entropy_y[i] = self.kernel_G(E[i + self.states_dimension], u)
            if entropy_y[i] < 1e-9:
                entropy_y[i] = 1e-9
        return np.diag(entropy_x), np.diag(entropy_y)

    def ut_init(self, alpha=1e-3, beta=2, kappa=0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        # lambda can be calculated by No.2 or just let it to be a const as No.1
        # self.lambda_ = self.alpha**2*(self.states_dimension+self.kappa) - self.states_dimension   # No.2
        self.lambda_ = 3  # No.1
        self.c_ = self.lambda_ + self.states_dimension  # scaling factor
        self.W_mean = (np.hstack(
            ((np.matrix(self.lambda_ / self.c_)), 1 / (2 * self.c_) + np.zeros(
                (1, 2 * self.states_dimension))))).A.reshape(
                    self.states_dimension * 2 + 1, )
        # self.W_mean = (np.hstack(((np.matrix(1-self.states_dimension-self.alpha**2*self.kappa)),
        #                1/(2*(self.alpha**2*self.kappa)) + np.zeros((1, 2*self.states_dimension))
        #                ))).A.reshape(self.states_dimension*2+1,)
        self.W_cov = self.W_mean  # No.1 and No.3
        # self.W_cov[0] = self.W_mean[0] + (1-self.alpha**2+self.beta)   # No.2

    def sigma_points(self, x_previous, P):
        sigma_A_ = np.linalg.cholesky((self.c_) * P)
        # sigma_A_ = self.alpha*self.kappa*np.linalg.cholesky(P)
        sigma_X_ = x_previous * np.ones((1, self.states_dimension))
        X_sigmas = np.hstack(
            (x_previous, sigma_X_ + sigma_A_, sigma_X_ - sigma_A_))
        return X_sigmas

    def ut(self, transfunc, ut_input, dimension, Noise_cov):
        cols = ut_input.shape[1]
        trans_mean = np.mat(np.zeros((dimension, 1)))
        trans_points = np.mat(np.zeros((dimension, cols)))
        for i in range(cols):
            trans_points[:, i] = transfunc(ut_input[:, i], self.Ts, self.k)
            trans_mean = trans_mean + self.W_mean[i] * trans_points[:, i]
        trans_dev = trans_points - trans_mean * np.ones((1, cols))
        trans_cov = trans_dev * np.diag(self.W_cov) * trans_dev.T + Noise_cov
        return trans_mean, trans_points, trans_cov, trans_dev

    def in_log_func(self, ver, keyword=''):
        name = self.tag + ', ' + keyword
        if name not in self.data_dic["others"]:
            self.data_dic['others'][name] = [[ver] for _ in range(self.repeat)]
        exec(f"self.data_dic['others'][name][self.repeat_count].append(ver)")


##########################################################################################################
class EKF(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.tag = "EKF"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts, k)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts, k)
        H = self.func.obs_jacobian(x_prior)
        # For time-variant system
        P = F * P * F.T + self.cov_Q
        # posterior
        K = P * H.T * inv(H * P * H.T + self.cov_R)
        self.in_log_func(K, 'K')
        x_posterior = x_prior + K * (sensor_data -
                                     self.func.observation_func(x_prior))
        P_posterior = (np.eye(self.states_dimension) - K * H) * P
        return x_posterior, P_posterior, 0


##########################################################################################################
# Yoh's method
class IMCEKF(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.mc_init(sigma, eps)
        self.tag = "IMCEKF"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts, k)
        measuring_error = sensor_data - self.func.observation_func(x_prior)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts, k)
        H = self.func.obs_jacobian(x_prior)
        P = F * P * F.T + self.cov_Q
        P_inv = inv(P + np.eye(self.states_dimension)*1e-6)
        # posterior
        x_posterior_temp = x_prior
        evaluation = 1
        mc_count = 0
        while evaluation > self.eps:
            mc_count += 1
            states_error = x_posterior_temp - x_prior
            L = self.kernel_G_R(measuring_error.T*inv(self.cov_R)*measuring_error) / \
                self.kernel_G_Q((states_error.T*P_inv*states_error))
            # # Logs
            # self.in_log_func(self.kernel_G_Q((states_error.T*P_inv*states_error)), 'G(Q)')
            K = inv(P_inv + (L * H.T * inv(self.cov_R) * H)) * L * H.T * inv(self.cov_R)
            x_posterior = x_prior + K * measuring_error
            if np.linalg.norm(x_posterior_temp) == 0:
                evaluation = 0
            else:
                evaluation = (np.linalg.norm(x_posterior-x_posterior_temp))/np.linalg.norm(x_posterior_temp)
            x_posterior_temp = x_posterior
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.cov_R*K.T
        #  Logs
        self.in_log_func(K, 'K')
        # self.in_log_func(self.kernel_G_Q((states_error.T*P_inv*states_error)), 'G(Q)')
        return x_posterior, P_posterior, 0


# Yoh's method, for variable bandwidth.
class IMCEKF2(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.mc_init(sigma, eps)
        self.tag = "IMCEKF2"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts, k)
        measuring_error = sensor_data - self.func.observation_func(x_prior)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts, k)
        H = self.func.obs_jacobian(x_prior)
        P = F * P * F.T + self.cov_Q
        P_inv = inv(P + np.eye(self.states_dimension)*1e-6)
        # posterior
        x_posterior_temp = x_prior
        evaluation = 1
        mc_count = 0
        while evaluation > self.eps:
            mc_count += 1
            states_error = x_posterior_temp - x_prior
            L = self.kernel_G_R(measuring_error.T*inv(self.cov_R)*measuring_error, measuring_error, x_prior, self.cov_R) / \
                self.kernel_G_Q(states_error.T*P_inv*states_error, states_error, x_prior, P)
            K = inv(P_inv + (L * H.T * inv(self.cov_R) * H)) * L * H.T * inv(self.cov_R)
            x_posterior = x_prior + K * measuring_error
            if np.linalg.norm(x_posterior_temp) == 0:
                evaluation = 0
            else:
                evaluation = (np.linalg.norm(x_posterior-x_posterior_temp))/np.linalg.norm(x_posterior_temp)
            x_posterior_temp = x_posterior
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.cov_R*K.T
        # Logs
        self.in_log_func(self.sigma_square_R, 'sigma_square_R')
        self.in_log_func(self.sigma_square_Q, 'sigma_square_Q')
        return x_posterior, P_posterior, 0


##########################################################################################################
# Dan.S's method
class MCEKF2(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.mc_init(sigma)
        self.tag = "MCEKF_DS"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts, k)
        measuring_error = sensor_data - self.func.observation_func(x_prior)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts, k)
        H = self.func.obs_jacobian(x_prior)
        P = F * P * F.T + self.cov_Q
        P_inv = inv(P + np.eye(self.states_dimension)*1e-6)
        # posterior
        # The calculation of L, denominator should be the error of states which can be instead with Q.
        L = self.kernel_G(measuring_error.T*inv(self.cov_R)*measuring_error)
        K = inv(P_inv + (L * H.T * inv(self.cov_R) * H)) * L * H.T * inv(self.cov_R)
        x_posterior = x_prior + K * measuring_error
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.cov_R*K.T
        # Logs
        self.in_log_func(L, 'L')
        return x_posterior, P_posterior, 0


##########################################################################################################
# Fixed Point Iteration
class MCEKF1(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.mc_init(sigma, eps)
        self.tag = "MCEKF_FPI"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        x_prior = self.func.state_func(x_previous, self.Ts, k)
        obs = self.func.observation_func(x_prior)
        # Calculate jacobin
        F = self.func.states_jacobian(x_previous, self.Ts, k)
        H = self.func.obs_jacobian(x_prior)
        # For time-variant system
        P = F * P * F.T + self.cov_Q
        # posterior
        P_sqrt = cholesky(P)
        R_sqrt = cholesky(self.cov_R)
        B = np.hstack(
            (np.vstack(
                (P_sqrt, np.zeros(
                    (self.obs_dimension, self.states_dimension)))),
             np.vstack((np.zeros(
                 (self.states_dimension, self.obs_dimension)), R_sqrt))))
        B_inv = inv(B)
        W = B_inv * np.vstack((np.identity(self.states_dimension), H))
        D = B_inv * np.vstack((x_prior, sensor_data))
        X_temp = x_prior
        Evaluation = 1
        mc_count = 0
        while Evaluation > self.eps:
            E = D - W * X_temp
            Cx, Cy = self.mc(E, X_temp)
            P_mc = P_sqrt * inv(Cx) * P_sqrt
            R_mc = R_sqrt * inv(Cy) * R_sqrt
            K = P_sqrt * H.T * inv(H * P_mc * H.T + R_mc)
            x_posterior = x_prior + K * (sensor_data - obs)
            Evaluation = np.linalg.norm(x_posterior -
                                        X_temp) / np.linalg.norm(X_temp)
            X_temp = x_posterior
            mc_count += 1
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.cov_R*K.T
        return x_posterior, P_posterior, mc_count


##########################################################################################################
class MCUKF1(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.ut_init(alpha, beta, kappa)
        self.mc_init(sigma, eps)
        self.tag = "MCUKF1"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        X_sigmas = self.sigma_points(x_previous, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas,
                                                self.states_dimension,
                                                self.cov_Q)
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas,
                                                    self.obs_dimension,
                                                    self.cov_R)
        # posterior
        P_xz = x_dev * np.diag(self.W_cov) * z_dev.T
        # MC part
        Sp_mc = cholesky(P_xx)
        Sr_mc = cholesky(self.cov_R)  # 这里是不是P_zz?
        S_mc = np.hstack(
            (np.vstack(
                (Sp_mc, np.zeros(
                    (self.obs_dimension, self.states_dimension)))),
             np.vstack((np.zeros(
                 (self.states_dimension, self.obs_dimension)), Sr_mc))))
        S_mc_inv = np.linalg.inv(S_mc)
        H_mc = (np.linalg.inv(P_xx) * P_xz).T
        W_mc = S_mc_inv * np.vstack((np.identity(self.states_dimension), H_mc))
        D_mc = S_mc_inv * np.vstack(
            (x_mean, sensor_data - obs_mean + H_mc * x_mean))
        x_init_mc = np.linalg.inv(W_mc.T * W_mc) * W_mc.T * D_mc
        # x_init_mc = x_mean
        Evaluation = 1
        count = 0
        x_old_mc = x_init_mc
        while Evaluation > self.eps:
            E_mc = D_mc - W_mc * x_old_mc
            Cx_mc, Cy_mc = self.mc(E_mc, x_old_mc)
            P_mc = Sp_mc * np.linalg.inv(Cx_mc) * Sp_mc.T
            R_mc = Sr_mc * np.linalg.inv(Cy_mc) * Sr_mc.T
            K = P_mc * H_mc.T * np.linalg.inv(H_mc * P_mc * H_mc.T + R_mc)
            x_new_mc = x_mean + K * (sensor_data - obs_mean)
            Evaluation = np.linalg.norm(x_new_mc -
                                        x_old_mc) / np.linalg.norm(x_old_mc)
            x_old_mc = x_new_mc
            count += 1
        x_posterior = x_new_mc
        P_posterior = (np.eye(self.states_dimension)-K*H_mc)*P_xx*(np.eye(self.states_dimension)-K*H_mc).T \
            + K*self.cov_R*K.T
        return x_posterior, P_posterior, count


##########################################################################################################
# This is a version that I'm trying to use Dan.S's method on UKF.
class MCUKF2(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.ut_init(alpha, beta, kappa)
        self.mc_init(sigma, eps)
        self.tag = "MCUKF2"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        X_sigmas = self.sigma_points(x_previous, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas,
                                                self.states_dimension,
                                                self.cov_Q)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, x_points,
                                                    self.obs_dimension,
                                                    self.cov_R)
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas,
                                                    self.obs_dimension,
                                                    self.cov_R)
        # posterior
        P_xz = x_dev * np.diag(self.W_cov) * z_dev.T
        H = inv(P_xx) * P_xz
        # 先测试用obs_mean和p_zz的组合计算L
        L = self.kernel_G(np.linalg.norm((sensor_data - obs_mean))*inv(self.cov_R)) / \
            self.kernel_G(np.linalg.norm((x_mean-self.F(x_previous, self.Ts)))*inv(P_xx))
        K = inv(P_xx +
                (P_zz - L * H * self.cov_R * H.T)) * H * inv(self.cov_R)
        x_posterior = x_mean + K * (sensor_data - obs_mean)
        P_posterior = (np.eye(self.states_dimension)-K*H.T)*P_xx*(np.eye(self.states_dimension)-K*H.T).T \
            + K*self.cov_R*K.T
        return x_posterior, P_posterior, 0


class UKF(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.ut_init(alpha, beta, kappa)
        self.tag = "UKF"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        X_sigmas = self.sigma_points(x_previous, P)
        x_mean, x_points, P_xx, x_dev = self.ut(self.F, X_sigmas,
                                                self.states_dimension,
                                                self.cov_Q)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, x_points,
                                                    self.obs_dimension,
                                                    self.cov_R)
        Z_sigmas = self.sigma_points(x_mean, P_xx)
        obs_mean, obs_points, P_zz, z_dev = self.ut(self.H, Z_sigmas,
                                                    self.obs_dimension,
                                                    self.cov_R)
        # posterior
        P_xz = x_dev * np.diag(self.W_cov) * z_dev.T
        K = P_xz * np.linalg.inv(P_zz)
        x_posterior = x_mean + K * (sensor_data - obs_mean)
        P_posterior = P_xx - K * P_zz * K.T
        return x_posterior, P_posterior, 0


##########################################################################################################
# Fixed Point Iteration
class MCKF1(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.mc_init(sigma, eps)
        self.tag = "MCKF1"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        F = self.func.state_matrix(x_previous, self.Ts)
        x_prior = F * x_previous
        P = F * P * F.T + self.cov_Q
        H = self.func.obs_matrix(x_prior)
        # posterior
        P_sqrt = cholesky(P)
        R_sqrt = cholesky(self.cov_R)
        B = np.hstack(
            (np.vstack(
                (P_sqrt, np.zeros(
                    (self.obs_dimension, self.states_dimension)))),
             np.vstack((np.zeros(
                 (self.states_dimension, self.obs_dimension)), R_sqrt))))
        B_inv = inv(B)
        W = B_inv * np.vstack((np.identity(self.states_dimension), H))
        D = B_inv * np.vstack((x_prior, sensor_data))
        X_temp = x_prior
        Evaluation = 1
        mc_count = 0
        while Evaluation > self.eps:
            E = D - W * X_temp
            Cx, Cy = self.mc(E, X_temp)
            P_mc = P_sqrt * inv(Cx) * P_sqrt
            R_mc = R_sqrt * inv(Cy) * R_sqrt
            K = P_sqrt * H.T * inv(H * P_mc * H.T + R_mc)
            x_posterior = x_prior + K * (sensor_data - H * x_prior)
            Evaluation = np.linalg.norm(x_posterior -
                                        X_temp) / np.linalg.norm(X_temp)
            X_temp = x_posterior
            mc_count += 1
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.cov_R*K.T
        return x_posterior, P_posterior, mc_count


##########################################################################################################
# Dan.S method, this works.
class MCKF2(Filter):
    def __init__(self, parameters):
        states_dimension, obs_dimension, t, Ts, q_, r_, alpha, beta, kappa, sigma, eps, data_dic, repeat, input = parameters
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, data_dic, repeat, input)
        self.mc_init(sigma, eps)
        self.tag = "MCKF2"

    def estimate(self, x_previous, sensor_data, P, k, repeat_count=0):
        # priori
        self.repeat_count = repeat_count
        self.k = k
        F = self.func.state_matrix(x_previous, self.Ts)
        x_prior = F * x_previous
        # For time-variant system
        P = F * P * F.T + self.cov_Q
        # posterior
        H = self.func.obs_matrix(x_prior)
        L = self.kernel_G(np.linalg.norm((x_prior - F*x_previous))*P_inv) / \
            self.kernel_G(np.linalg.norm(self.std_Q)*inv(self.cov_R))
        K = inv(L * P_inv +
                (H.T * inv(self.cov_R) * H)) * H.T * inv(self.cov_R)
        x_posterior = x_prior + K * (sensor_data - H * x_prior)
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.cov_R*K.T
        return x_posterior, P_posterior, 0
