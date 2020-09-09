import math
import numpy as np


def state_func(states, Ts):
    states_ = np.zeros(4).reshape(4,)
    states_[0] = states[0]+Ts*states[2]
    states_[1] = math.sin((states[1] + Ts*states[3]))
    states_[2] = states[2] + Ts*states[1]
    states_[3] = states[3] + Ts*states[0]
    return states_


def observation_func(states):
    observation = np.zeros(4).reshape(4,)
    observation[0] = (np.sqrt(states[0] + 1)).real
    observation[1] = 0.8*states[1] + 0.3*states[0]
    observation[2] = states[2]
    observation[3] = states[3]
    return observation
