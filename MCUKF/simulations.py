import numpy as np
from numpy.random import randn
from filters import Mcukf
from filters import Ukf
import nonlinear_func as N_func


class FilterSim():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
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

    def noise_init(self, add_r=0, repeat=1):
        self.state_noise = np.mat(self.noise_q * randn(self.states_dimension, self.N))
        self.obs_noise = [np.mat(self.noise_r*randn(self.obs_dimension, self.N)
                                 + add_r*randn(self.obs_dimension, self.N)) for _ in range(repeat)]
        return self.obs_noise

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


class McukfSim(FilterSim):
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, alpha_, beta_, ki_, sigma_, eps_, q_, r_):
        FilterSim.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mcukf_init(sigma_, eps_, alpha_, beta_, ki_)

    # --------------------------------MCUKF init---------------------------------- #
    def mcukf_init(self, sigma_, eps_, alpha_, beta_, ki_):
        self.mcukf = Mcukf()
        self.mcukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.mcukf.mc_init(sigma_, eps_)
        self.mcukf.ut_init(alpha_, beta_, ki_)
        self.mcukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        mc_count = 0
        states_mean = 0
        mse1 = 0
        for j in range(repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P, count = \
                    self.mcukf.estimate(self.states[:, i-1],
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


class UkfSim(FilterSim):
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, alpha_, beta_, ki_, q_, r_):
        FilterSim.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.ukf_init(alpha_, beta_, ki_)

    # --------------------------------UKF init---------------------------------- #
    def ukf_init(self, alpha_, beta_, ki_):
        self.ukf = Ukf()
        self.ukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.ukf.ut_init(alpha_, beta_, ki_)
        self.ukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        states_mean = 0
        mse1 = 0
        for j in range(repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P = \
                    self.ukf.estimate(self.states[:, i-1],
                                      self.obs[:, i]+obs_noise[j][:, i],
                                      self.P, i)
            states_mean += self.states
            mse1 += self.MSE()
        states_mean /= repeat
        mse1 /= repeat
        mse = mse1.sum(axis=1)/self.N
        return self.time_line, states_mean, mse1, mse


class NonlinearSys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, additional_noise=0):
        self.sys_init(states_dimension, obs_dimension, t, Ts)
        self.noise_init(q_)

    def sys_init(self, states_dimension, obs_dimension, t, Ts):
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.real_obs = np.mat(np.zeros((obs_dimension, self.N)))

    def noise_init(self, q):
        self.state_noise = np.mat(q * randn(self.states_dimension, self.N))

    def states_init(self, X0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.real_obs[:, 0] = N_func.observation_func(self.states[:, 0])

    def run(self):
        for i in range(1, self.N):
            self.states[:, i] = N_func.state_func(self.states[:, i-1], self.Ts, i) + self.state_noise[:, i]
            self.real_obs[:, i] = N_func.observation_func(self.states[:, i])
        return self.time_line, self.states, self.real_obs
