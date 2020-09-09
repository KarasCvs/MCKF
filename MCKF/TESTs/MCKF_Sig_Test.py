import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


t = 10
T = 0.1
x_dimension = 2
y_dimension = 1
N = int(t/T+1)

# 初始化数组
X_prior = np.matrix(np.zeros((x_dimension, N)))
X_posterior = np.matrix(np.zeros((x_dimension, N)))
X_real = np.matrix(np.zeros((x_dimension, N)))
g = np.matrix(np.zeros((x_dimension, N)))
sensor = np.zeros(N)
MCKFObservation = np.zeros(N)
err = np.zeros(N)
RealOutput = np.zeros(N)
P_prior = np.zeros((x_dimension, x_dimension, N))
P_posterior = np.zeros((x_dimension, x_dimension, N))
MCKF = [[], [], [], [], []]

# 初始化基本状态常量
A = np.matrix([0, -0.7, 1, -1.5]).reshape(x_dimension, x_dimension)
b = np.matrix([0.5, 1]).reshape(x_dimension, 1)
c = np.matrix([0, 1]).reshape(x_dimension, 1)
q_noise = 2                   # 系统噪音
r_noise = 3                    # 观测噪音


def G(sig, e):    # 用来判断熵的函数, 高斯核函数, sig越大, 对误差就越敏感.
    return float(np.exp(-((float(e)**2)/(2*(float(sig)**2)))))


for i in range(N):
    X_real[:, i] = A*X_real[:, i-1] + np.random.rand(x_dimension, 1)*np.sqrt(q_noise)
    sensor[i] = c.T*X_real[:, i] + np.random.rand()*np.sqrt(r_noise)
    RealOutput[i] = c.T*X_real[:, i]
sensor[10] = 12
sensor[50] = -15
sensor[80] = 20

# MCKF
sig = [0.2, 0.5, 2, 3, 10]
eps = 0.02
# H 就是c.T观测矩阵
for j in range(len(sig)):
    X_prior = np.matrix(np.zeros((x_dimension, N)))
    X_posterior = np.matrix(np.zeros((x_dimension, N)))
    MCKFObservation = np.zeros(N)
    g = np.matrix(np.zeros((x_dimension, N)))
    P_prior = np.zeros((x_dimension, x_dimension, N))
    P_posterior = np.zeros((x_dimension, x_dimension, N))
    for i in range(N):
        X_prior[:, i] = A*X_posterior[:, i-1]               # 先验估计 X(x|x-1)
        P_prior[:, :, i] = np.matrix(A*P_posterior[:, :, i-1]*A.T + q_noise*np.matrix(np.eye(x_dimension, x_dimension)))        # 先验增益 P(k|k-1)
        # if np.linalg.norm(P_prior[:, :, i]) == 2.8284271247461903:
        #     print("reset")
        try:
            B_p = np.linalg.cholesky(P_prior[:, :, i])              # B_p矩阵, 对先验增益的cholesky分解
        except BaseException as e:
            print(f"error: {e}, i={i}, P_prior={P_prior[:, :, i]}, P_posterior={P_posterior[:, :, i-1]}")
            break
        B_r = np.matrix(float(np.sqrt(r_noise)))                         # 这里应该是对观测噪音的cholesky 先用一阶的干扰开方将就一下
        B = np.hstack((np.vstack((P_prior[:, :, i], np.matrix([0, 0]))), np.vstack((np.matrix([0, 0]).reshape(x_dimension, 1), B_r))))   # 这里构建B矩阵
        D = np.linalg.inv(B) * np.vstack((X_prior[:, i], sensor[i]))        # 简化公式
        W = np.linalg.inv(B) * np.vstack((np.ones((x_dimension, x_dimension)), c.T))            # 简化公式
        e = D - W*X_posterior[:, i-1]
        X_posterior_temp_now = X_prior[:, 0]                                # 初始化X_t(1), 另其等于先验
        entropy_x = entropy_y = []                                          # 初始化熵列表, 用来计算C的角矩阵.
        EstimateRate = 1
        X_posterior_temp_before = np.matrix((1, 1))
        while EstimateRate > eps:                                           # 退出条件, 后验与上一时刻后验之比大于ep
            entropy_x = []
            entropy_y = []
            for k in range(x_dimension):                           # 计算熵和C_x,C_y矩阵  问题出在这, D如果是3+1维的话, L=4, 没有D(4)来计算, 也就没有C_y
                entropy_x.append(G(sig[j], e[k]))
            C_x = np.matrix(np.diag(entropy_x))
            if y_dimension == 1:
                C_y = G(sig[j], e[x_dimension])
            else:
                for k in range(x_dimension, x_dimension + y_dimension):
                    entropy_y.append(G(sig[j], e[k]))
                C_y = np.matrix(np.diag(entropy_y))  # L = 3+0 , D是3x1的.n=3 m=0                                       # 单纯计算, 目的是得到X_posterior B_r应当为矩阵.
            P = B_p*np.linalg.inv(C_x)*B_p.T
            R = B_r/C_y*B_r
            K = P*c/(c.T*P*c+R)
            X_posterior_temp_now = np.matrix(X_prior[:, i] + K*(sensor[i]-c.T*X_prior[:, i]), dtype="float")
            EstimateRate = np.linalg.norm(X_posterior_temp_now-X_posterior_temp_before) / np.linalg.norm(X_posterior_temp_before)
            X_posterior_temp_before = X_posterior_temp_now
        X_posterior[:, i] = X_posterior_temp_now
        X_posterior[:, i] = X_posterior_temp_now
        if R == np.inf:
            P_posterior[:, :, i] = (np.eye(x_dimension, x_dimension)-K*c.T)*P_prior[:, :, i]*(np.eye(x_dimension, x_dimension)-K*c.T).T         # 重定义R, 防止R=inf时造成计算出错. 因为R=inf的时候K=0, 
            print("set")
        else:
            P_posterior[:, :, i] = (np.eye(x_dimension, x_dimension)-K*c.T)*P_prior[:, :, i]*(np.eye(x_dimension, x_dimension)-K*c.T).T + K*R*K.T
        MCKFObservation[i] = float(c.T*X_posterior[:, i])
        MCKF[j].append(float(c.T*X_posterior[:, i]))


# plt.plot(np.linspace(0, t, N), RealOutput, linewidth=1, linestyle="-", label="Real State")
# plt.plot(np.linspace(0, t, N), sensor, linewidth=1, linestyle="-", label="Sensor")
for i in range(len(sig)):
    plt.plot(np.linspace(0, t, N), MCKF[i], linewidth=1, linestyle="-", label=f"sigm={sig[i]}")

plt.grid(True)
plt.legend(loc='upper left')
plt.show()