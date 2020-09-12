import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
# from scipy.stats import norm

import functions.nonlinear_func as N_func
# from mckf.mckf import MCKF
# from kf.kf import KF
from . import ukf_2


class UKF_Sim():
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t=10, Ts=0.01):
        self.sys_init(states_dimension, obs_dimension, t, Ts)
        self.noise_init()
        self.ukf_init()

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
        self.ukf_states = np.mat(np.zeros((states_dimension, self.N)))
        self.ukf_obs = np.mat(np.zeros((obs_dimension, self.N)))
        self.P = np.mat(np.identity(states_dimension))
        self.sensor_MSE = np.mat(np.zeros((obs_dimension, self.N)))
        self.ukf_MSE = np.mat(np.zeros((obs_dimension, self.N)))

    def noise_init(self, q=0, r=2):
        self.noise_q = q             # 系统噪音
        self.noise_r = r
        self.state_noise = np.mat(self.noise_q * randn(self.states_dimension, self.N))
        self.observation_noise = np.mat(self.noise_r*randn(self.obs_dimension, self.N) + 0*randn(self.obs_dimension, self.N))

    # --------------------------------UKF init---------------------------------- #
    def ukf_init(self, alpha_=1e-3, beta_=2, ki_=0):
        self.ukf = ukf_2.UKF()
        self.ukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)
        self.ukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.ukf.ut_init(alpha_, beta_, ki_)

    def states_init(self, X0, ukf0, P0):
        self.states[:, 0] = np.array(X0).reshape(self.states_dimension, 1)
        self.ukf_states[:, 0] = np.array(ukf0).reshape(self.states_dimension, 1)
        self.real_obs[:, 0] = N_func.observation_func(self.states[:, 0])
        self.sensor[:, 0] = self.real_obs[:, 0] + self.observation_noise[:, 0]
        self.P = np.diag(P0)

    def run(self):
        self.sys_only = False
        # --------------------------------main procedure---------------------------------- #
        for i in range(1, self.N):
            # 步进
            self.states[:, i] = N_func.state_func(self.states[:, i-1], self.Ts) + self.state_noise[:, i]
            self.real_obs[:, i] = N_func.observation_func(self.states[:, i])
            self.sensor[:, i] = self.real_obs[:, i] + self.observation_noise[:, i]
            self.ukf_states[:, i], self.P = self.ukf.estimate(self.ukf_states[:, i-1], self.sensor[:, i], self.P, i)
            self.ukf_MSE[:, i] = abs(np.mean(self.states[0, i] - self.ukf_states[0, i]))

    def system_only(self):
        self.sys_only = True
        for i in range(1, self.N):
            # 步进
            self.states[:, i] = N_func.state_func(self.states[:, i-1], self.Ts) + self.state_noise[:, i]
            self.real_obs[:, i] = N_func.observation_func(self.states[:, i])
            self.sensor[:, i] = self.real_obs[:, i] + self.observation_noise[:, i]

    def plot(self):
        if self.sys_only:
            for i in range(self.states_dimension):
                plt.figure(1)
                plt.subplot(100*self.states_dimension+11+i)
                plt.plot(self.time_line, self.states[i, :].A.reshape(self.N,), linewidth=1, linestyle="-", label="Real State")
                plt.grid(True)
                plt.legend(loc='upper left')
                plt.title("States")
            plt.figure(2)
            plt.plot(self.time_line, self.sensor.A.reshape(self.N,), linewidth=1, linestyle="-", label="Sensor")
            plt.plot(self.time_line, self.real_obs.A.reshape(self.N,), linewidth=1, linestyle="-", label="Real obs")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title("Observation")
        else:
            for i in range(self.states_dimension):
                plt.figure(1)
                plt.subplot(100*self.states_dimension+11+i)
                plt.plot(self.time_line, self.ukf_states[i, :].A.reshape(self.N,), linewidth=1, linestyle="-", label="UKF")
                plt.plot(self.time_line, self.states[i, :].A.reshape(self.N,), linewidth=1, linestyle="-", label="Real State")
                plt.grid(True)
                plt.legend(loc='upper left')
                plt.title("States")
            plt.figure(2)
            plt.plot(self.time_line, self.sensor.A.reshape(self.N,), linewidth=1, linestyle="-", label="Sensor")
            plt.plot(self.time_line, self.real_obs.A.reshape(self.N,), linewidth=1, linestyle="-", label="Real obs")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title("Observation")
            plt.figure(3)
            plt.plot(self.time_line, self.ukf_MSE.A.reshape(self.N,), linewidth=1, linestyle="-", label="ukf MSE")
            # plt.plot(self.time_line, self.sensor_MSE[i, :].A.reshape(self.N,), linewidth=1, linestyle="-", label="self.sensor MSE")
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title("MSE")
        plt.show()


if __name__ == "__main__":
    pass
