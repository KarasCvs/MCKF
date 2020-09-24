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

    def state_matrix(self, states, Ts=0, k=0):
        self.F = np.matrix(([0, 1, 0], [0, float(2*math.exp(-states[0]/2e4) * states[1] * states[2]/2 - 32.2/states[1]), 0], [0, 0, 1]))
        return self.F

    def obs_matrix(self, states, Ts=0, k=0):
        self.H = np.matrix((float(math.sqrt(1e5*1e5 + (states[0]-1e5)**2)/states[0]), 0, 0))
        return self.H

    def state_func(self, states, Ts, k=0):
        states_ = self.F*states
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = self.H*states
        return observation
