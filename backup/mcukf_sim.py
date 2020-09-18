import numpy as np
from numpy.random import randn
from functions import nonlinear_func as N_func
from filters.mcukf.mcukf import MCUKF


class MCUKF_Sim():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, alpha_, beta_, ki_, sigma_, eps_, q_, r_):
        self.sys_init(states_dimension, obs_dimension, t, Ts)
        self.noise_init(q_, r_)
        self.mcukf_init(sigma_, eps_, alpha_, beta_, ki_)

    def sys_init(self, states_dimension, obs_dimension, t, Ts):
        self.states_dimension = states_dimension
        self.obs_dimension = obs_dimension
        self.t = t
        self.Ts = Ts
        self.N = int(self.t/self.Ts)
        self.time_line = np.linspace(0, self.t, self.N)
        self.mcukf_states = np.mat(np.zeros((states_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.mcukf_MSE = np.mat(np.zeros((states_dimension, self.N)))

    def noise_init(self, q, r):
        self.noise_q = q             # 系统噪音
        self.noise_r = r
        self.state_noise = np.mat(self.noise_q * randn(self.states_dimension, self.N))

    # --------------------------------UKF init---------------------------------- #
    def mcukf_init(self, sigma_, eps_, alpha_, beta_, ki_):
        self.mcukf = MCUKF.MCUKF()
        self.mcukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.mcukf.mc_init(sigma_, eps_)
        self.mcukf.ut_init(alpha_, beta_, ki_)
        self.mcukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)

    def states_init(self, init_parameters):
        ukf0, P0 = init_parameters
        self.mcukf_states[:, 0] = np.array(ukf0).reshape(self.states_dimension, 1)
        self.P = np.diag(P0)

    def read_data(self, states, sensor):
        self.states = states
        self.sensor = sensor

    def run(self):
        # --------------------------------main procedure---------------------------------- #
        for i in range(1, self.N):
            self.mcukf_states[:, i], self.P = self.mcukf.estimate(self.mcukf_states[:, i-1], self.sensor[:, i], self.P, i)
        return self.time_line, self.mcukf_states

    def MSE(self):
        for i in range(1, self.N):
            self.mcukf_MSE[:, i] = self.states[:, i] - self.mcukf_states[:, i]
            self.mcukf_MSE[:, i] = np.power(self.mcukf_MSE[:, i], 2)
        return self.mcukf_MSE


if __name__ == "__main__":
    pass
