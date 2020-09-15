import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import functions.nonlinear_func as N_func
from . import mcukf3 as MCUKF


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
        self.states = np.mat(np.zeros((states_dimension, self.N)))
        self.sensor = np.mat(np.zeros((obs_dimension, self.N)))
        self.real_obs = np.mat(np.zeros((obs_dimension, self.N)))
        self.mcukf_states = np.mat(np.zeros((states_dimension, self.N)))
        self.ukf_obs = np.mat(np.zeros((obs_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.sensor_MSE = np.mat(np.zeros((obs_dimension, self.N)))
        self.mcukf_MSE = np.mat(np.zeros((obs_dimension, self.N)))

    def noise_init(self, q, r):
        self.noise_q = q             # 系统噪音
        self.noise_r = r
        self.state_noise = np.mat(self.noise_q * randn(self.states_dimension, self.N))
        self.observation_noise = np.mat(self.noise_r*randn(self.obs_dimension, self.N) + 0*randn(self.obs_dimension, self.N))

    # --------------------------------UKF init---------------------------------- #
    def mcukf_init(self, sigma_, eps_, alpha_, beta_, ki_):
        self.mcukf = MCUKF.MCUKF()
        self.mcukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.mcukf.mc_init(sigma_, eps_)
        self.mcukf.ut_init(alpha_, beta_, ki_)
        self.mcukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)

    def states_init(self, X0, ukf0, P0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.mcukf_states[:, 0] = np.array(ukf0).reshape(self.states_dimension, 1)
        self.real_obs[:, 0] = N_func.observation_func(self.states[:, 0])
        self.sensor[:, 0] = self.real_obs[:, 0] + self.observation_noise[:, 0]
        self.P = np.diag(P0)

    def read_data(self, states, obs):
        self.states = states
        self.real_obs = obs

    def run(self):
        # --------------------------------main procedure---------------------------------- #
        for i in range(1, self.N):
            self.sensor[:, i] = self.real_obs[:, i] + self.observation_noise[:, i]
            self.mcukf_states[:, i], self.P = self.mcukf.estimate(self.mcukf_states[:, i-1], self.sensor[:, i], self.P, i)
            self.mcukf_MSE[:, i] = abs(np.mean(self.states[0, i] - self.mcukf_states[0, i]))
        return self.time_line, self.mcukf_states, self.mcukf_MSE

    def plot(self):
        for i in range(self.states_dimension):
            plt.figure(1)
            plt.subplot(100*self.states_dimension+11+i)
            plt.plot(self.time_line, self.mcukf_states[i, :].A.reshape(self.N,), linewidth=1, linestyle="-", label="MCUKF")
            plt.plot(self.time_line, self.states[i, :].A.reshape(self.N,), linewidth=1, linestyle="-", label="Real State")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title("States")
        plt.figure(2)
        plt.plot(self.time_line, self.mcukf_MSE.A.reshape(self.N,), linewidth=1, linestyle="-", label="mcukf MSE")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.title("MSE")
        plt.figure(3)
        plt.plot(self.time_line, self.sensor.A.reshape(self.N,), linewidth=1, linestyle="-", label="Sensor")
        plt.plot(self.time_line, self.real_obs.A.reshape(self.N,), linewidth=1, linestyle="-", label="Real obs")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.title("Observation")
        plt.show()


if __name__ == "__main__":
    pass
