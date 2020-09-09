import numpy as np
import matplotlib.pyplot as plt
import cmath as math


t = 10
T = 0.1
N = int(t/T+1)
Xhat_o = np.zeros((2, N)).reshape(2, N)
Xhat_n = np.zeros((2, N)).reshape(2, N)
X = np.zeros((2, N)).reshape(2, N)
g = np.zeros((2, N)).reshape(2, N)
Z = np.zeros(N)
Y = np.zeros(N)
err = np.zeros(N)
Real = np.zeros(N)
P_o = np.zeros((2, 2, N)).reshape(2, 2, N)
P_n = np.zeros((2, 2, N)).reshape(2, 2, N)

A = np.array([0, -0.7, 1, -1.5]).reshape(2, 2)
b = np.array([0.5, 1]).reshape(2,)
c = np.array([0, 1]).reshape(2,)
v = np.array([0.2, 1, 5])
w = np.array([0.02, 0.1, 0.5])

for i in range(N):
    X[:, i] = np.dot(A, X[:, i-1]) + np.dot(b, np.random.rand()*math.sqrt(v[1]))
    Z[i] = np.dot(c.T, X[:, i]) + np.random.rand()*math.sqrt(w[1])
    Real[i] = np.dot(c.T, X[:, i])

for i in range(N):
    Xhat_o[:, i] = np.dot(A, Xhat_n[:, i-1])
    P_o[:, :, i] = A.dot(P_n[:, :, i-1]).dot(A.T) + v[1] * b.dot(b.T)
    g[:, i] = (P_o[:, :, i].dot(c)) / (c.T.dot(P_o[:, :, i]).dot(c) + (w[1]))
    Xhat_n[:, i] = Xhat_o[:, i] + g[:, i]*Z[i]-c.T.dot(Xhat_o[:, i])
    P_n[:, :, i] = (np.eye(2)-g[:, i].reshape(2, 1).dot(c.reshape(1, 2))).dot(P_o[:, :, i])
    Y[i] = np.dot(c.T, Xhat_n[:, i])


# plt.plot(np.linspace(0, t, N), Real, linewidth=1, linestyle="-", label="Real State")
plt.plot(np.linspace(0, t, N), Z, linewidth=1, linestyle="-", label="Observation")
plt.plot(np.linspace(0, t, N), Y, linewidth=1, linestyle="-", label="KF")

plt.grid(True)
plt.legend(loc='upper left')
plt.show()
