import numpy as np
from numpy.random import randn


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


if __name__ == "__main__":
    pass
