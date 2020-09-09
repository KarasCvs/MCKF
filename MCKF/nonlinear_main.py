import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
# from scipy.stats import norm

import nonlinear_func as N_func
# from mckf.mckf import MCKF
# from kf.kf import KF
from ukf.ukf import UKF


t = 5
Ts = 0.1
states_dimension = 4
obs_dimension = 4
N = int(t/Ts)
time_line = np.linspace(0, t, N)
# --------------------------------init---------------------------------- #
states = np.mat(np.zeros((states_dimension, N)))
sensor = np.mat(np.zeros((obs_dimension, N)))
real_obs = np.mat(np.zeros((obs_dimension, N)))
ukf_states = np.mat(np.zeros((states_dimension, N)))
ukf_obs = np.mat(np.zeros((obs_dimension, N)))
P = np.mat(np.identity(states_dimension))

noise_q = 0             # 系统噪音
noise_r = 5
state_noise = np.mat(noise_q * randn(states_dimension, N))
observation_noise = np.mat(noise_r*randn(obs_dimension, N) + 0*randn(obs_dimension, N))

# --------------------------------UKF init---------------------------------- #
ukf = UKF()
ukf.state_func(N_func.state_func, N_func.observation_func, Ts)
ukf.filter_init(states_dimension, obs_dimension, noise_q, noise_r)
ukf.ut_init()

# --------------------------------main procedure---------------------------------- #
states[:, 0] = np.array([0.3, 0.2, 1, 2]).reshape(4, 1)
for i in range(1, N):
    # 步进
    states[:, i] = N_func.state_func(states[:, i-1], Ts) + state_noise[:, i]
    real_obs[:, i] = N_func.observation_func(states[:, i])
    sensor[:, i] = real_obs[:, i] + observation_noise[:, i]
    ukf_states[:, i], P = ukf.estimate(ukf_states[:, i-1], sensor[:, i], P, i)
    ukf_obs[:, i] = N_func.observation_func(ukf_states[:, i])
# MSE
sensor_error = real_obs - sensor
sensor_MSE = np.mean(sensor_error, axis=1)
# sensor_std = np.std(sensor_error)
# sensor_pdf = norm.pdf(time_line-t/2, 0, sensor_std)
ukf_error = ukf_obs - sensor
ukf_MSE = np.mean(ukf_error, axis=1)
print(f"Sensor MSE = {sensor_MSE}\n ukf MSE = {ukf_MSE}")

for i in range(obs_dimension):
    plt.subplot(411+i)
    plt.plot(time_line, sensor[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Sensor")
    plt.plot(time_line, ukf_obs[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="UKF")
    plt.plot(time_line, real_obs[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Real State")
    plt.grid(True)
    plt.legend(loc='upper left')
plt.show()
