import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from mckf.mckf import MCKF
from kf.kf import KF
from pf.particlefilter import PF
from scipy.stats import norm
import math


t = 5
Ts = 0.1
x_dimension = 3
y_dimension = 3
N = int(t/Ts)
time_line = np.linspace(0, t, N)

X_real = np.asmatrix(np.zeros((x_dimension, N)))
sensor = np.asmatrix(np.zeros((y_dimension, N)))
RealOutput = np.asmatrix(np.zeros((y_dimension, N)))
MCKFObservation = np.zeros(N)
noise_q = 1             # 系统噪音
noise_r = np.matrix([2])
noise_state = np.asmatrix(np.sqrt(noise_q) * randn(x_dimension, N))
noise_observation = np.asmatrix(int(np.sqrt(noise_r)) * randn(y_dimension, N))

A = np.matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(x_dimension, x_dimension)
B = np.matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(x_dimension, y_dimension)
C = np.matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(y_dimension, x_dimension)

input = np.asmatrix(np.ones((y_dimension, N)))
r = 1
theta = 10
for i in range(N):
    input[0, i] = input[0, i-1] + r*math.cos(theta)
    input[1, i] = input[1, i-1] + r*math.sin(theta)
    input[2, i] = input[2, i-1] + theta
for i in range(N):
    if i == 0:
        X_real[:, 0] = np.matrix(([2], [1], [0]))
    else:
        # 欧拉法
        # X_real[:, i] = (A*X_real[:, i-1] + B*input[:, i-1])*Ts + X_real[:, i-1] + randn(x_dimension, 1)*np.sqrt(noise_q)
        # 塔斯汀法
        # X_real[:, i] = np.linalg.inv(np.eye(x_dimension, x_dimension) - Ts/2*A) *\
        #     (Ts/2*(A*X_real[:, i-1] + B*(input[:, i-1]+input[:, i]) +
        #      (randn(x_dimension, 1)*np.sqrt(noise_q)+randn(x_dimension, 1)*np.sqrt(noise_q))) + X_real[:, i-1])
        # 改进欧拉法
        X_temp = (A*X_real[:, i-1] + B*input[:, i-1])*Ts + X_real[:, i-1]
        X_real[:, i] = (A*X_real[:, i-1] + B*input[:, i-1] + X_temp + noise_state[:, i-1]+noise_state[:, i])*Ts/2 + X_real[:, i-1]
    sensor[:, i] = C*X_real[:, i] + noise_observation[:, i]
    RealOutput[:, i] = C*X_real[:, i]

mckf = MCKF()
mckf.read_data(x_dimension, y_dimension, sensor)
mckf.parmet(3, 0.2, noise_q, noise_r)
mckf.ss(A, B, C, 0, X_real[:, 0], input)
MCKFObservation = mckf.calculation()

kf = KF()
kf.read_data(x_dimension, y_dimension, sensor)
kf.parmet(noise_q, noise_r)
kf.ss(A, B, C, 0, X_real[:, 0], input)
KFObservation = kf.calculation()

# Errors
sensor_erros = RealOutput - sensor
sensor_mean = np.mean(sensor_erros)
sensor_std = np.std(sensor_erros)
sensor_pdf = norm.pdf(time_line-t/2, 0, sensor_std)

mckf_error = RealOutput - MCKFObservation
mckf_mean = np.mean(mckf_error)
mckf_std = np.std(mckf_error)
mckf_pdf = norm.pdf(time_line-t/2, 0, mckf_std)

kf_error = RealOutput - KFObservation
kf_mean = np.mean(kf_error)
kf_std = np.std(kf_error)
kf_pdf = norm.pdf(time_line-t/2, 0, kf_std)

# plt.subplot(221)
# plt.plot(np.linspace(0, t, N), np.array(sensor).reshape(N,), linewidth=1, linestyle="-", label="Sensor")
# plt.plot(np.linspace(0, t, N), np.array(MCKFObservation).reshape(N,), linewidth=1, linestyle="-", label="MCKF")
# plt.plot(np.linspace(0, t, N), np.array(KFObservation).reshape(N,), linewidth=1, linestyle="-", label="KF ")
# plt.plot(np.linspace(0, t, N), np.array(RealOutput).reshape(N,), linewidth=1, linestyle="-", label="Real State")
# plt.grid(True)
# plt.legend(loc='upper left')

plt.subplot(221)
# plt.plot(sensor.T[:, 0], sensor.T[:, 0], linewidth=1, linestyle=".", label="Sensor")
# plt.plot(MCKFObservation[0, :], MCKFObservation[1, :], linewidth=1, linestyle="-", label="MCKF")
# plt.plot(np.linspace(0, t, N), np.array(KFObservation).reshape(N,), linewidth=1, linestyle="-", label="KF ")
# plt.plot(RealOutput.T[:, 0], RealOutput.T[:, 0], linewidth=1, linestyle=".", label="Real State")
plt.scatter(np.array(RealOutput[0, :]), np.array(RealOutput[1, :]), label="Real State")
plt.grid(True)
plt.legend(loc='upper left')

# plt.subplot(222)
# plt.plot(time_line-t/2, sensor_pdf.reshape(N,), label="Sensor")
# plt.plot(time_line-t/2, mckf_pdf.reshape(N,), label="MCKF")
# plt.plot(time_line-t/2, kf_pdf.reshape(N,), label="KF")
# plt.grid(True)
# plt.legend(loc='upper left')

# 第二组图像, 用来测试各种效果
# data_number = 4
# mckf_t = MCKF()
# mckf_t.parmet(2, 0.002, noise_q, noise_r)
# mckf_t.ss(A, b, c, 0)
# for i in range(0, N):
#     if i == 0:
#         mckf_t.read_data(x_dimension, y_dimension, sensor[:, i])
#         MCKFObservation_test = mckf_t.calculation()
#     else:
#         mckf_t.read_data(x_dimension, y_dimension, np.c_[mckf_t.sensor, sensor[:, i]])
#         MCKFObservation_test = np.c_[MCKFObservation_test, mckf_t.calculation()[:, i]]

# PF test
# pf = PF()
# pf.parmet(noise_r)
# pf.read_data(y_dimension)
# for i in range(0, N):
#     if i == 0:
#         PFObservation_test = pf.calculation(MCKFObservation[:, i])
#     else:
#         PFObservation_test = np.c_[PFObservation_test, pf.calculation(MCKFObservation[:, i])]
# Error_1 = RealOutput - PFObservation_test

# for i in range(0, N):
#     if i == 0:
#         PFObservation_test2 = pf.calculation(sensor[:, i])
#     else:
#         PFObservation_test2 = np.c_[PFObservation_test2, pf.calculation(sensor[:, i])]
# Error_2 = RealOutput - PFObservation_test2

# plt.subplot(223)
# plt.plot(np.linspace(0, t, N), np.array(RealOutput[0, :]).reshape(100,), linewidth=1, linestyle="-", label="Real State")
# plt.plot(np.linspace(0, t, N), np.array(sensor[0, :]).reshape(100,), linewidth=1, linestyle="-", label="Sensor 1")
# plt.plot(np.linspace(0, t, N), np.array(PFObservation_test[0, :]).reshape(100,), linewidth=1, linestyle="-", label="PF_T 1")
# plt.grid(True)
# plt.legend(loc='upper left')

# plt.subplot(224)
# plt.plot(np.linspace(0, t, N), np.array(Error_1).reshape(100,), linewidth=1, linestyle="-", label="PF_T 1")
# plt.plot(np.linspace(0, t, N), np.array(Error_2).reshape(100,), linewidth=1, linestyle="-", label="PF_T 2")
# plt.plot(np.linspace(0, t, N), np.array(PFObservation_test2).reshape(100,), linewidth=1, linestyle="-", label="PF_T 2")
# plt.grid(True)
# plt.legend(loc='upper left')

plt.show()
