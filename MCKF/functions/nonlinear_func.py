import math
import numpy as np


# Nonlinear func No.1,  Default
# def state_func(states, k=0, Ts=0.2):
#     states_ = np.zeros(4).reshape(4, 1)
#     states_[0] = states[0]+Ts*states[2]
#     states_[1] = math.sin((states[1] + Ts*states[3]))
#     states_[2] = states[2] + Ts*states[1]
#     states_[3] = states[3] + Ts*states[0]
#     return states_


# def observation_func(states, k=0, Ts=0.1):
#     observation = np.zeros(4).reshape(4, 1)
#     observation[0] = math.sqrt(abs(states[0]+1))
#     observation[1] = 0.8*states[1] + 0.3*states[0]
#     observation[2] = states[2]
#     observation[3] = states[3]
#     return observation

# Nonlinear func NO.2, from paper[Maximum correntropy unscented filter]
# def state_func(states, k=0, Ts=0.1):
#     states_ = np.zeros(3).reshape(3, 1)
#     states_[0] = states[0] + Ts*states[1]
#     states_[1] = states[1] + Ts*math.exp(-states[0]*5e-5) * pow(states[1], 2)*states[2] - Ts*32.2
#     states_[2] = states[2]
#     return states_


# def observation_func(states, k=0, Ts=0.1):
#     observation = math.sqrt(pow(1e5, 2) + pow((states[0]-1e5), 2))
#     return observation

# No.3
def state_func(states, k=0, Ts=0.1):
    states_ = np.zeros(3).reshape(3, 1)
    states_[0] = states[0] + Ts*(-states[1])
    states_[1] = states[1] + Ts * (-math.exp(-states[0]*5e-5) * pow(states[1], 2)*states[2])
    states_[2] = states[2]
    return states_

def observation_func(states, k=0, Ts=0.1):
    observation = math.sqrt(pow(1e5, 2) + pow((states[0]-1e5), 2))
    return observation


if __name__ == "__main__":
    pass
