import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from data_manager import Manager
import random


mg = Manager()
mg.read_data()
timeline = mg.find("time line")
states = mg.find("system states")[0]
N = len(timeline)
train_number = random.sample(range(N), int(N/10))
train_x = []
train_y = []
for i in train_number:
    train_x.append(timeline[i])
    train_y.append(states[i])
timeline = np.array(timeline).reshape(N, 1)
train_x = np.array(train_x).reshape(int(N/10), 1)
train_y = np.array(train_y).reshape(int(N/10), 1)
states = np.array(states).reshape(N, 1)

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(train_x, train_y)
y_pred, sigma = gp.predict(timeline, return_std=True)

plt.figure()
plt.plot(timeline, states, label='real', color='r')
plt.plot(timeline, y_pred, 'b-', label=u'Prediction')
# plt.fill(np.concatenate([timeline, timeline[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
plt.legend(loc='upper left')


plt.show()
