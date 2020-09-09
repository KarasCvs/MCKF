import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.stats import norm

import nonlinear_func as N_func
# from mckf.mckf import MCKF
# from kf.kf import KF
from ukf.ukf import UKF


t = 5
Ts = 0.1
x_dimension = 4
y_dimension = 4
N = int(t/Ts)
time_line = np.linspace(0, t, N)

states = np.mat(np.zeros((x_dimension, N)))
sensor = np.mat(np.zeros((y_dimension, N)))
real_obs = np.mat(np.zeros((y_dimension, N)))
ukf_states = np.mat(np.zeros(x_dimension))
ukf_obs = np.mat(np.zeros((y_dimension, N)))

noise_q = 0             # 系统噪音
noise_r = 2
state_noise = np.mat(noise_q * randn(x_dimension, N))
observation_noise = np.mat(noise_r*randn(y_dimension, N) + 0*randn(y_dimension, N))
for i in range(N):
    if i == 0:
        states[:, 0] = np.array([0.3, 0.2, 1, 2])
    else:
        # 步进
        states[:, i] = N_func.state_func(states[:, i-1], Ts) + state_noise[:, i]
    real_obs[:, i] = N_func.observation_func(states[:, i])
    sensor[:, i] = real_obs[:, i] + observation_noise[:, i]

# Errors
sensor_error = real_obs - sensor
sensor_mean = np.mean(sensor_error)
sensor_std = np.std(sensor_error)
sensor_pdf = norm.pdf(time_line-t/2, 0, sensor_std)

# print(np.mean(mckf_error), np.mean(kf_error))

for i in range(y_dimension):
    plt.subplot(411+i)
    plt.plot(time_line, sensor[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Sensor")
    plt.plot(time_line, ukf_obs[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="MCKF")
    # plt.plot(np.array(KFObservation.T[:, 0]), np.array(KFObservation.T[:, 1]), linewidth=1, linestyle="-", label="KF ")
    plt.plot(time_line, real_obs[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Real State")
    plt.grid(True)
    plt.legend(loc='upper left')
plt.show()
