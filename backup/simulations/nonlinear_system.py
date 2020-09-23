import numpy as np
from numpy.random import randn
from functions import nonlinear_func as N_func


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
