from filters import Mckf2 as Mckf
from filters import Sys
import numpy as np


def sim_run(sigma_, repeat_):
    # system part
    states_dimension = 3
    obs_dimension = 1
    repeat = repeat_
    t = 50
    Ts = 0.1
    N = int(t/Ts)
    # mckf part
    sigma = sigma_
    eps = 1e-6
    # noise
    q = 0
    r = 100
    add_r = 0
    additional_noise = add_r*np.random.randn(obs_dimension, N)

    # System initial
    sys = Sys(states_dimension, obs_dimension, t, Ts, q, r, add_r)
    # sys.states_init([1.25, 0.06, 0.01, -0.003])
    sys.states_init([3e5, -2e4, 1e-3])
    time_line, states, real_obs = sys.run()
    # sys.plot()
    # filter initial values
    # filter_init = ([0, 0, 0, 0], [10, 10, 10, 10])
    filter_init = ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
    # mckf initial, order: x dimension, y dimension, run time, time space, repeat, α, β, kappa, sigma, eps, q, r
    mckf_sim = Mckf(states_dimension, obs_dimension, t, Ts, q, r, sigma, eps)
    print("Simulation started.")
    mckf_sim.read_data(states, real_obs)
    obs_noise = mckf_sim.noise_init(additional_noise, repeat)
    _, mckf_states_mean, mckf_MSE1, mckf_MSE, mc_count = mckf_sim.run(filter_init, obs_noise, repeat)
    # build a data set
    description = "Mckf test."
    data_summarizes = {
                    'description': description,
                    'shapes': {'states dimension': states_dimension, 'obs dimension': states_dimension},
                    'parameters': {'repeat': repeat, 'time': t, 'ts': Ts, 'q': q, 'r': r, 'add noise': add_r,
                                   'sigma': sigma, 'eps': eps},
                    'mse': {'mckf mse': mckf_MSE.tolist()},
                    'mse1': {'mckf mse1': mckf_MSE1.tolist()},
                    'time line': time_line.tolist(), 'mc iteration': mc_count,
                    'states': {'system states': states.tolist(), 'mckf states': mckf_states_mean.tolist()},
                    'observations': {'noise_free observation': real_obs.tolist()}
                    }
    return data_summarizes

