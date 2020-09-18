import numpy as np
from numpy.random import randn
from functions import nonlinear_func as N_func
from filters.ukf.ukf import UKF


class UKF_Sim():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, alpha_, beta_, ki_, q_, r_):
        self.sys_init(states_dimension, obs_dimension, t, Ts)
        self.noise_init(q_, r_)
        self.ukf_init(alpha_, beta_, ki_)

    def sys_init(self, states_dimension, obs_dimension, t, Ts):
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.ukf_states = np.mat(np.zeros((states_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.ukf_MSE = np.mat(np.zeros((states_dimension, self.N)))

    def noise_init(self, q, r):
        self.noise_q = q             # 系统噪音
        self.noise_r = r
        self.state_noise = np.mat(self.noise_q * randn(self.states_dimension, self.N))

    # --------------------------------UKF init---------------------------------- #
    def ukf_init(self, alpha_, beta_, ki_):
        self.ukf = UKF.UKF()
        self.ukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)
        self.ukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.ukf.ut_init(alpha_, beta_, ki_)

    def states_init(self, init_parameters):
        ukf0, P0 = init_parameters
        self.ukf_states[:, 0] = np.array(ukf0).reshape(self.states_dimension, 1)
        self.P = np.diag(P0)

    def read_data(self, states, sensor):
        self.states = states
        self.sensor = sensor

    def run(self):
        # --------------------------------main procedure---------------------------------- #
        for i in range(1, self.N):
            self.ukf_states[:, i], self.P = self.ukf.estimate(self.ukf_states[:, i-1], self.sensor[:, i], self.P, i)
        return self.time_line, self.ukf_states

    def MSE(self):
        for i in range(1, self.N):
            self.ukf_MSE[:, i] = self.states[:, i] - self.ukf_states[:, i]
            self.ukf_MSE[:, i] = np.power(self.ukf_MSE[:, i], 2)
        return self.ukf_MSE


if __name__ == "__main__":
    pass
