import numpy as np
from numpy.random import randn
from functions import NonLinearFunc as Func
from filters import Mckf2 as Mckf
import matplotlib.pyplot as plt


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


class MckfSim(FilterSim):
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, sigma_, eps_, q_, r_):
        FilterSim.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mckf_init(Ts, sigma_, eps_)

    # --------------------------------mckf init---------------------------------- #
    def mckf_init(self, Ts, sigma_, eps_):
        self.func = Func()
        self.mckf = Mckf()
        self.mckf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r, Ts)
        self.mckf.mc_init(sigma_, eps_)

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        mc_count = 0
        states_mean = 0
        mse1 = 0
        for j in range(repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P, count = \
                    self.mckf.estimate(self.states[:, i-1],
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


class Sys():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_, additional_noise=0):
        self.sys_init(states_dimension, obs_dimension, t, Ts)
        self.noise_init(q_)
        self.func = Func()

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