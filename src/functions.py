# import sympy
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


# 2 states, step input system
class NonLinearFunc0():
    def __init__(self, u=0):
        self.u = u
        pass

    def state_func(self, states, Ts, k=0):
        states_ = np.zeros(2).reshape(2, 1)
        states_[0] = 0.5*states[0, 0] + states[1, 0]*self.u[k-1]
        states_[1] = -0.05*states[0, 0]*states[1, 0] + self.u[k-1]
        if states_[0] > 9999999:
            print('hit')
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = -states[0]*states[1]
        return observation

    def states_jacobian(self, states, Ts, k=0):
        states_jacobian = np.matrix(([0.5, self.u[k-1]], [-0.05*states[1, 0], -0.05*states[0, 0]])).reshape(2, 2)
        return states_jacobian

    def obs_jacobian(self, states, Ts=0):
        states[0]
        obs_jacobian = np.matrix([-states[1, 0], -states[0, 0]]).reshape(1, 2)
        return obs_jacobian


# 3 states, object falling simulation
class NonLinearFunc1():
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

    def state_matrix(self, states, Ts=0, k=0):
        self.F = np.matrix(([0, 1, 0], [0, float(2*math.exp(-states[0]/2e4) * states[1] * states[2]/2 - 32.2/states[1]), 0], [0, 0, 0]))
        self.F = np.eye(states.shape[0]) + Ts*self.F
        return self.F

    def obs_matrix(self, states, Ts=0, k=0):
        self.H = np.matrix((float(math.sqrt(1e5*1e5 + (states[0]-1e5)**2)/states[0]), 0, 0))
        return self.H

    def states_jacobian(self, states, Ts, k=0):
        x0 = float(states[0])
        x1 = float(states[1])
        x2 = float(states[2])
        states_jacobian = np.matrix(([1, Ts, 0], [Ts * -5.0e-5*x1**2*x2*math.exp(-5.0e-5*x0),
                                     Ts * 2*x1*x2*math.exp(-5e-5*x0) + 1,
                                     Ts * x1**2*math.exp(-5e-5*x0)], [0, 0, 1]))
        return states_jacobian

    def obs_jacobian(self, states, Ts=0):
        x0 = float(states[0])
        obs_jacobian = np.matrix([(x0-1e5)/math.sqrt(1e5*1e5+(x0-1e5)**2), 0, 0])
        return obs_jacobian


class NonLinearFunc2():
    def __init__(self):
        pass

    def state_func(self, states, Ts, k=0):
        states_ = np.zeros(1).reshape(1, 1)
        states_[0] = 0.5 * states[0] + (25 * states[0])/(1 + states[0]**2) + 8 * math.cos(1.2*k)
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = 0.05 * states[0]**2
        return observation

    def states_jacobian(self, states, Ts, k=0):
        states_jacobian = 0.5 + 25*((1-states[0]**2)/(1+states[0]**2)**2)
        return states_jacobian

    def obs_jacobian(self, states, Ts=0):
        x0 = float(states[0])
        obs_jacobian = np.matrix([0.1*x0])
        return obs_jacobian


class NonLinearFunc3():
    def __init__(self):
        pass

    def state_func(self, states, Ts, k=0):
        states_ = np.zeros(1).reshape(1, 1)
        states_[0] = states[0] + 3*math.cos((states[0]/10))
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = states[0] ** 3
        return observation

    def states_jacobian(self, states, Ts, k=0):
        states_jacobian = 1 - 3*math.sin((states[0]/10))
        return states_jacobian

    def obs_jacobian(self, states, Ts=0):
        states[0]
        obs_jacobian = 3*(states[0]**2)
        return obs_jacobian


class MoveSim():
    def __init__(self):
        self.r = 5
        self.omega = 5
        pass

    def state_func(self, states, Ts, k=0):
        states_ = np.zeros(2).reshape(2, 1)
        states_[0] = states[0] + Ts*self.r*math.cos(self.omega)
        states_[1] = states[1] + Ts*self.r*math.sin(self.omega)
        return states_

    def observation_func(self, states, Ts=0, k=0):
        observation = np.zeros(2).reshape(2, 1)
        observation[0] = states[0]
        observation[1] = states[1]
        return observation

    def states_jacobian(self, states, Ts, k=0):
        x0 = float(states[0])
        x1 = float(states[1])
        states_jacobian = np.matrix([-math.sin(x0), 0], [0, math.cos(x1)])
        return states_jacobian

    def obs_jacobian(self, states, Ts=0):
        return np.matrix([1, 0], [0, 1])
