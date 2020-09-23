import math
import numpy as np


class LinearFunc():
    def __init__(self):
        self.F = np.matrix(([1, 1, 0.5, 0.5], [0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0.606],))
        self.H = np.matrix((1, 0, 0, 0))

    def state_func(self, states, Ts, k=0):
        states_ = self.F*states
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = self.H*states
        return observation
