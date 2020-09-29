import sympy
import math
import numpy as np


class LinearFunc():
    def __init__(self):
        pass

    def state_matrix(self, states, Ts=0, k=0):
        self.F = np.matrix(([1, 1, 0.5, 0.5], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0.606],))
        return self.F

    def obs_matrix(self, states, Ts=0, k=0):
        self.H = np.matrix((1, 0, 0, 0))
        return self.H

    def state_func(self, states, Ts, k=0):
        states_ = self.F*states
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = self.H*states
        return observation


class NonLinearFunc():
    def __init__(self):
        pass

    def state_func(self, states, Ts, k=0):
        states_ = np.zeros(3).reshape(3, 1)
        states_[0] = states[0] + Ts*states[1]
        states_[1] = states[1] + Ts*(2*math.exp(-states[0]/2e4) * states[1] * states[1] * states[2]/2 - 32.2)
        states_[2] = states[2]
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = math.sqrt(1e5*1e5 + (states[0]-1e5) ** 2)
        return observation

    # Linear
    # def state_func(self, states, Ts, k=0):
    #     states_ = self.F*states
    #     return states_

    # def observation_func(self, states, Ts=0, k=0):
    #     observation = self.H*states
    #     return observation

    def state_matrix(self, states, Ts=0, k=0):
        self.F = np.matrix(([0, 1, 0], [0, float(2*math.exp(-states[0]/2e4) * states[1] * states[2]/2 - 32.2/states[1]), 0], [0, 0, 0]))
        self.F = np.eye(states.shape[0]) + Ts*self.F
        return self.F

    def obs_matrix(self, states, Ts=0, k=0):
        self.H = np.matrix((float(math.sqrt(1e5*1e5 + (states[0]-1e5)**2)/states[0]), 0, 0))
        return self.H

    def states_jacobian(self, states, Ts):
        x0 = float(states[0])
        x1 = float(states[1])
        x2 = float(states[2])
        states_jacobian = np.matrix(([1, 0.1, 0], [Ts * -5.0e-5*x1**2*x2*math.exp(-5.0e-5*x0),
                                     Ts * 2*x1*x2*math.exp(-5e-5*x0) + 1,
                                     Ts * x1**2*math.exp(-5e-5*x0)], [0, 0, 1]))
        return states_jacobian

    def obs_jacobian(self, states, Ts=0):
        x0 = float(states[0])
        obs_jacobian = np.matrix([(x0-1e5)/math.sqrt(1e5*1e5+(x0-1e5)**2), 0, 0])
        return obs_jacobian
