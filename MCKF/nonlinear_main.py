import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
# from scipy.stats import norm

import nonlinear_func as N_func
# from mckf.mckf import MCKF
# from kf.kf import KF
from ukf.ukf import UKF


t = 10
Ts = 0.1
states_dimension = 3
obs_dimension = 1
N = int(t/Ts)
time_line = np.linspace(0, t, N)
# --------------------------------init---------------------------------- #
states = np.mat(np.zeros((states_dimension, N)))
sensor = np.mat(np.zeros((obs_dimension, N)))
real_obs = np.mat(np.zeros((obs_dimension, N)))
ukf_states = np.mat(np.zeros((states_dimension, N)))
ukf_obs = np.mat(np.zeros((obs_dimension, N)))
P = np.mat(np.identity(states_dimension))
sensor_MSE = np.mat(np.zeros((obs_dimension, N)))
ukf_MSE = np.mat(np.zeros((obs_dimension, N)))

noise_q = 0             # 系统噪音
noise_r = 2
state_noise = np.mat(noise_q * randn(states_dimension, N))
observation_noise = np.mat(noise_r*randn(obs_dimension, N) + 0*randn(obs_dimension, N))

# --------------------------------UKF init---------------------------------- #
ukf = UKF()
ukf.state_func(N_func.state_func, N_func.observation_func, Ts)
ukf.filter_init(states_dimension, obs_dimension, noise_q, noise_r)
ukf.ut_init(1.1547e-3, 2, -1)

# --------------------------------main procedure---------------------------------- #
# system No.1
# states[:, 0] = np.array([0.3, 0.2, 1, 2]).reshape(states_dimension, 1)
# ukf_states[:, 0] = np.array([0.3, 0.2, 1, 2]).reshape(states_dimension, 1)
# P = np.diag([1, 1, 1, 1])

# system No.2
states[:, 0] = np.array([300000, -20000, 1/1000]).reshape(states_dimension, 1)
ukf_states[:, 0] = np.array([300000, -20000, 0.0009]).reshape(states_dimension, 1)
P = np.diag([1000000, 4000000, 1/1000000])
for i in range(1, N):
    # 步进
    real_obs[:, i] = N_func.observation_func(states[:, i-1])
    sensor[:, i] = real_obs[:, i] + observation_noise[:, i]
    ukf_states[:, i], P = ukf.estimate(ukf_states[:, i-1], sensor[:, i], P, i)
    states[:, i] = states[:, i-1] + N_func.state_func(states[:, i-1], Ts) * Ts + state_noise[:, i]
    ukf_MSE[:, i] = pow(np.mean(states[:, i] - ukf_obs[:, i]), 2)


for i in range(states_dimension):
    plt.figure(1)
    plt.subplot(100*states_dimension+11+i)
    # plt.plot(time_line, sensor[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Sensor")
    plt.plot(time_line, ukf_states[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="UKF")
    plt.plot(time_line, states[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Real State")
    plt.grid(True)
    plt.legend(loc='upper left')
    # plt MSE
plt.figure(2)
plt.plot(time_line, ukf_MSE.A.reshape(N,), linewidth=1, linestyle="-", label="ukf MSE")
# plt.plot(time_line, sensor_MSE[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="sensor MSE")
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
