# There are atleast 3 ways to calculate W under ut_init,
# No.1: The original one and Katayama's book which will is most stable. Set lambda as a conste.
# No.2: <The Scaled Unscented Transformation> used in robot_localization too,
# but it's not stable enough, could cause the covariance matrix negative definite
# No.3: I tried to use the same way that No.2 to calculate W_m, but make W_c exactly equal
# with W_m. That mean W_m(0) will not be specialization, this is work able but still, not
# stable enough.
import numpy as np
from numpy.random import randn
from numpy.linalg import inv, cholesky
from functions import NonLinearFunc as Func
import matplotlib.pyplot as plt


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
        self.noise_q = q_
        self.noise_r = r_
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.noise_Q = self.noise_q**2 * np.identity(self.states_dimension)
        self.noise_R = self.noise_r**2 * np.identity(self.obs_dimension)

    def states_init(self, init_parameters):
        ukf0, P0 = init_parameters
        self.states[:, 0] = np.array(ukf0).reshape(self.states_dimension, 1)
        self.P = np.diag(P0)

    def read_data(self, states, obs):
        self.real_states = states
        self.obs = obs

    def MSE(self):
        for i in range(1, self.N):
            self.mse1[:, i] = self.real_states[:, i] - self.states[:, i]
            self.mse1[:, i] = np.power(self.mse1[:, i], 2)
        return self.mse1

    def mc_init(self, sigma, eps):
        self.kernel_G = lambda x: np.exp(-(pow(np.linalg.norm(x), 2)/(2*sigma*sigma)))
        self.eps = eps

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

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        mc_count = 0
        states_mean = 0
        mse1 = 0
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
        states_mean /= repeat
        mse1 /= repeat
        mse = mse1.sum(axis=1)/self.N
        mc_count /= self.N*repeat
        return self.time_line, states_mean, mse1, mse, mc_count

# Fixed Point Iteration
class Mckf1(Filter):
    def __init__(self):
        Filter.__init__(self)

    def estimate(self, x_prior, sensor_data, P, k):
        # priori
        self.k = k
        H = self.H
        x_posterior = self.F * x_prior
        P = self.F * P * self.F.T + self.noise_Q
        # posterior
        P_sqrt = cholesky(P)
        R_sqrt = cholesky(self.noise_R)
        B = np.hstack((np.vstack((P_sqrt, np.zeros((self.obs_dimension, self.states_dimension)))),
                       np.vstack((np.zeros((self.states_dimension, self.obs_dimension)), R_sqrt))))
        B_inv = inv(B)
        W = B_inv*np.vstack((np.identity(self.states_dimension), H))
        D = B_inv*np.vstack((x_posterior, sensor_data))
        X_temp = x_posterior
        Evaluation = 1
        mc_count = 0
        while Evaluation > self.eps:
            E = D - W*X_temp
            Cx, Cy = self.mc(E)
            P_mc = P_sqrt*inv(Cx)*P_sqrt
            R_mc = R_sqrt*inv(Cy)*R_sqrt
            K = P_sqrt*H.T*inv(H*P_mc*H.T+R_mc)
            x_posterior = x_posterior + K*(sensor_data - H*x_posterior)
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


class Mckf2(Filter):
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, sigma, eps):
        Filter.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mc_init(sigma, eps)

    def estimate(self, x_prior, sensor_data, P, k):
        # priori
        self.k = k
        F = self.func.state_matrix(x_prior, self.Ts)
        x_posterior = F * x_prior
        P = F * P * F.T + self.noise_Q
        # posterior
        H = self.func.obs_matrix(x_posterior)
        L = self.kernel_G(np.linalg.norm((sensor_data - H*x_posterior))*inv(self.noise_R)) / \
            self.kernel_G(np.linalg.norm((x_posterior - F*x_prior))*inv(P))
        K = inv(inv(P) + (L*H.T*self.noise_R*H))*L*H.T*inv(self.noise_R)
        x_posterior = x_posterior + K*(sensor_data - H*x_posterior)
        P_posterior = (np.eye(self.states_dimension)-K*H)*P*(np.eye(self.states_dimension)-K*H).T \
            + K*self.noise_R*K.T
        return x_posterior, P_posterior, 0


class LinearSys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, additional_noise=0):
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
        self.add_r = additional_noise

    # Generate noise lists.
    def noise_init(self, repeat=1):
        self.obs_noise = [np.mat(self.noise_r*randn(self.obs_dimension, self.N)
                                 + self.add_r*randn(self.obs_dimension, self.N)) for _ in range(repeat)]
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
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, additional_noise=0):
        self.func = self.func()
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
        self.add_r = additional_noise

    # Generate noise lists.
    def noise_init(self, repeat=1):
        self.obs_noise = [np.mat(self.noise_r*randn(self.obs_dimension, self.N)
                                 + self.add_r*randn(self.obs_dimension, self.N)) for _ in range(repeat)]
        return self.obs_noise

    def states_init(self, X0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.real_obs[:, 0] = self.func.observation_func(self.states[:, 0])

    def run(self):
        for i in range(1, self.N):
            self.states[:, i] = self.func.state_func(self.states[:, i-1], self.Ts, i) + self.state_noise[:, i]
            self.real_obs[:, i] = self.func.observation_func(self.states[:, i])
        return self.time_line, self.states, self.real_obs
