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
        observation = math.sqrt(1e5*1e5 + pow((states[0]-1e5), 2))
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
        states_jac = np.matrix(([0, 1, 0], [1+Ts*(-2/2e4*math.exp(-states[0]/2e4) * states[1] * states[1] * states[2]/2),
                                            1+Ts*(4*math.exp(-states[0]/2e4) * states[1] * states[2]/2),
                                            1+Ts*(2*math.exp(-states[0]/2e4) * states[1] * states[1]/2)], [0, 0, 0]))
        return states_jac

    def obs_jacobian(self, states):
        obs_jac = np.matrix(([0, 1, 0], [-2/2e4*math.exp(-states[0]/2e4) * states[1] * states[1] * states[2]/2,
                                         4*math.exp(-states[0]/2e4) * states[1] * states[2]/2,
                                         2*math.exp(-states[0]/2e4) * states[1] * states[1]/2], [0, 0, 0]))
        return obs_jac

