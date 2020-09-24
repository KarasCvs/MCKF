import math
import numpy as np


# Nonlinear func NO.2, from paper[Maximum correntropy unscented filter]
def state_func(states, Ts, k=0):
    states_ = np.zeros(3).reshape(3, 1)
    states_[0] = states[0] + Ts*states[1]
    states_[1] = states[1] + Ts*(2*math.exp(-states[0]/2e4) * states[1] * states[1] * states[2]/2 - 32.2)
    states_[2] = states[2]
    return states_


def observation_func(states, Ts=0, k=0):
    observation = math.sqrt(1e5*1e5 + pow((states[0]-1e5), 2))
    return observation


if __name__ == "__main__":
    pass
