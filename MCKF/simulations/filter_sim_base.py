import numpy as np
from numpy.random import randn


class FilterSim():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, q_, r_):
        self.sys_init(states_dimension, obs_dimension, t, Ts)
        self.noise_init(q_, r_)

    def sys_init(self, states_dimension, obs_dimension, t, Ts):
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.mse = np.mat(np.zeros((states_dimension, self.N)))

    def noise_init(self, q, r):
        self.noise_q = q             # 系统噪音
        self.noise_r = r
        self.state_noise = np.mat(self.noise_q * randn(self.states_dimension, self.N))

    def states_init(self, init_parameters):
        ukf0, P0 = init_parameters
        self.states[:, 0] = np.array(ukf0).reshape(self.states_dimension, 1)
        self.P = np.diag(P0)

    def read_data(self, states, sensor):
        self.real_states = states
        self.sensor = sensor

    def MSE(self):
        for i in range(1, self.N):
            self.mse[:, i] = self.real_states[:, i] - self.states[:, i]
            self.mse[:, i] = np.power(self.mse[:, i], 2)
        return self.mse


if __name__ == "__main__":
    pass
