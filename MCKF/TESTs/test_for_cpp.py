import numpy as np
from numpy.linalg import cholesky


def mc(sigma, E):
    kernel_G = lambda x: np.exp(-((x*x)/(2*sigma*sigma)))
    entropy_x = np.zeros(states_dimension)
    entropy_y = np.zeros(obs_dimension)
    for i in range(states_dimension):
        entropy_x[i] = kernel_G(E[i])
        if entropy_x[i] < 1e-9:
            entropy_x[i] = 1e-9
    for i in range(obs_dimension):
        entropy_y[i] = kernel_G(E[i+states_dimension])
        if entropy_y[i] < 1e-9:
            entropy_y[i] = 1e-9
    return np.diag(entropy_x), np.diag(entropy_y)


states_dimension = 3
obs_dimension = 2
# MC part
P_xx = np.asmatrix(np.diag((2, 2, 2)))
noise_R = np.asmatrix(np.diag((1, 1)))
P_xz = np.matrix(([1, 1, 1], [1, 1, 1])).reshape(3, 2)

x_mean = np.asmatrix(np.array((1, 1, 1))).reshape(3, 1)
obs_mean = np.asmatrix(np.array((2, 3))).reshape(2, 1)
sensor_data = np.asmatrix(np.array((4, 1))).reshape(2, 1)
eps = 1e-3
sigma = 6


Sp_mc = cholesky(P_xx)
Sr_mc = cholesky(noise_R)    # 这里是不是P_zz?
S_mc = np.hstack((np.vstack((Sp_mc, np.zeros((obs_dimension, states_dimension)))),
                  np.vstack((np.zeros((states_dimension, obs_dimension)), Sr_mc))))
S_mc_inv = np.linalg.inv(S_mc)
H_mc = (np.linalg.inv(P_xx) * P_xz).T
W_mc = S_mc_inv*np.vstack((np.identity(states_dimension), H_mc))
D_mc = S_mc_inv*np.vstack((x_mean, sensor_data - obs_mean + H_mc*x_mean))
x_init_mc = np.linalg.inv(W_mc.T*W_mc)*W_mc.T*D_mc
Evaluation = 1
count = 0
x_old_mc = x_init_mc
while Evaluation > eps:
    E_mc = D_mc - W_mc*x_old_mc
    Cx_mc, Cy_mc = mc(sigma, E_mc)
    P_mc = Sp_mc*np.linalg.inv(Cx_mc)*Sp_mc.T
    R_mc = Sr_mc*np.linalg.inv(Cy_mc)*Sr_mc.T
    K = P_mc*H_mc.T*np.linalg.inv(H_mc*P_mc*H_mc.T+R_mc)
    x_new_mc = x_mean + K*(sensor_data-obs_mean)
    Evaluation = np.linalg.norm(x_new_mc - x_old_mc)/np.linalg.norm(x_old_mc)
    x_old_mc = x_new_mc
    count += 1
x_posterior = x_new_mc
P_posterior = (np.eye(states_dimension)-K*H_mc)*P_xx*(np.eye(states_dimension)-K*H_mc).T \
    + K*noise_R*K.T
print(count)
print(f"{x_posterior}\n {P_posterior}")