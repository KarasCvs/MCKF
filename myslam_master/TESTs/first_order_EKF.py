import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


t = 10
T = 0.1
N = int(t/T+1)
Xhat_o = np.zeros(N)
Xhat_n = np.zeros(N)
X = np.zeros(N)
g = np.zeros(N)
Z = np.zeros(N)
Y = np.zeros(N)
A = np.zeros(N)
c = np.zeros(N)
err = np.zeros(N)
err_EKF = np.zeros(N)
Real = np.zeros(N)
P_o = np.zeros(N)
P_n = np.zeros(N)


v = 1
w = 10
X[0] = Xhat_o[0] = 15


def motion_model(x, k):
    F = 0.2*x + 25*x / (1+x**2) + 8*sp.cos(1.2*k)
    b = 1
    return F, b


def Observation(x):
    H = 1/20 * x**2
    return H


def jacobF(j):
    # A = 0.2 + 25 * (1-x*x) / (1+x*x)*(1+x*x)
    x, k = sp.symbols('x k')
    F = 0.2*x + 25*x / (1+x**2) + 8*sp.cos(1.2*k)
    func = sp.diff(F, x)
    return func.evalf(subs={'x': j})


def jacobH(x):
    c = x/10
    return c


for i in range(N):
    if i <= 0:
        X[i], b = motion_model(X[0], 0)
    else:
        X[i], b = motion_model(X[i-1], i-1)
    X[i] = X[i] + np.random.rand()*sp.sqrt(v)
    Real[i] = Observation(X[i])
    Z[i] = Real[i] + np.random.rand()*sp.sqrt(w)

for i in range(N):
    if i <= 0:
        Xhat_o[i], b = motion_model(Xhat_n[0], 0)
        jF = jacobF(Xhat_o[i])
        P_o[i] = jF * P_n[0] * jF + v
    else:
        Xhat_o[i], b = motion_model(Xhat_n[i-1], i-1)
        jF = jacobF(Xhat_o[i])
        P_o[i] = jF*P_n[i-1]*jF + v
    jH = jacobH(Xhat_o[i])
    g[i] = (P_o[i]*jH) / (jH*P_o[i]*jH + w)
    Xhat_n[i] = Xhat_o[i] + g[i] * (Z[i] - Observation(Xhat_o[i]))
    P_n[i] = (np.eye(1) - g[i] * jH) * (P_o[i])
    Y[i] = Observation(Xhat_n[i])

# plt.plot(np.linspace(0, t, N), Real, linewidth=1, linestyle="-", label="Real State")
# plt.plot(np.linspace(0, t, N), Z, linewidth=1, linestyle="-", label="Observation")
# plt.plot(np.linspace(0, t, N), Y, linewidth=1, linestyle="-", label="KF")
# plt.plot(np.linspace(0, t, N), Z - Real, linewidth=1, linestyle="-", label="Observation")
# plt.plot(np.linspace(0, t, N), Y - Real, linewidth=1, linestyle="-", label="EKF")
plt.plot(np.linspace(0, t, N), (Z - Real) - (Y - Real), linewidth=1, linestyle="-", label="Relative Error")

plt.grid(True)
plt.legend(loc='upper left')
plt.show()
