from filters import Mcukf as Mcukf
from filters import Ukf as Ukf
from filters import Mckf2 as Mckf
from filters import Ekf as Ekf
from filters import Mcekf as Mcekf
from filters import NonlinearSys as Sys
import numpy as np


def sim_run(repeat_, sigma_=2, **kwargs):
    # system part
    states_dimension = 3
    obs_dimension = 1
    repeat = repeat_
    t = 50
    Ts = 0.1
    # MCUKF part
    sigma = sigma_
    eps = 1e-6
    # UKF part
    alpha = 1e-3
    beta = 2
    kappa = 0
    # noise
    q = 0
    r = 30
    # additional noise
    additional_noise = 20*np.random.randn(obs_dimension, int(t/Ts))
    for i in range(additional_noise.shape[1]):
        if np.random.randint(0, 100) < 5:
            additional_noise[:, i] = additional_noise[:, i]*np.random.randint(70, 90)
    # additional_noise = np.zeros((obs_dimension, int(t/Ts)))
    # System initial
    sys = Sys(states_dimension, obs_dimension, t, Ts, q, r)
    sys.states_init([3e5, -2e4, 1e-3])
    time_line, states, real_obs = sys.run()
    # Filter initial values
    filter_init = ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])  # default ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
    # Initial noise lists.
    obs_noise = sys.noise_init(repeat, additional_noise)

    print("Simulation started.")

    # Mckf part
    # Mckf initial, order: x dimension, y dimension, run time, time space, sigma, eps
    try:
        if kwargs['mckf']:
            print('Mckf runing.')
            mckf_sim = Mckf(states_dimension, obs_dimension, t, Ts, q, r, sigma, eps)
            mckf_sim.read_data(states, real_obs)
            mckf_states_mean, mckf_MSE1, mckf_MSE, _, _ = mckf_sim.run(filter_init, obs_noise, repeat)
    except KeyError:
        pass

    # Ekf part
    try:
        if kwargs['ekf']:
            print('Ekf running.')
            ekf_sim = Ekf(states_dimension, obs_dimension, t, Ts, q, r)
            ekf_sim.read_data(states, real_obs)
            ekf_states_mean, ekf_MSE1, ekf_MSE, _, _ = ekf_sim.run(filter_init, obs_noise, repeat)
    except KeyError:
        pass

    # Ukf part
    # Ukf initial, order: x dimension, y dimension, run time, time space, q, r, α, β, kappa
    try:
        if kwargs['ukf']:
            print('Ukf running.')
            ukf_sim = Ukf(states_dimension, obs_dimension, t, Ts, q, r, alpha, beta, kappa)
            ukf_sim.read_data(states, real_obs)
            ukf_states_mean, ukf_MSE1, ukf_MSE, _, _ = ukf_sim.run(filter_init, obs_noise, repeat)
    except KeyError:
        pass

    # Mcekf part
    try:
        if kwargs['mcekf']:
            print('Mcekf running.')
            mcekf_sim = Mcekf(states_dimension, obs_dimension, t, Ts, q, r, sigma)
            mcekf_sim.read_data(states, real_obs)
            mcekf_states_mean, mcekf_MSE1, mcekf_MSE, _, _ = mcekf_sim.run(filter_init, obs_noise, repeat)
    except KeyError:
        pass

    # Mcukf part
    # Mcukf initial, order: x dimension, y dimension, run time, time space, q, r, α, β, kappa, sigma, eps
    try:
        if kwargs['mcukf']:
            print('Mcukf running.')
            mcukf_sim = Mcukf(states_dimension, obs_dimension, t, Ts, q, r, alpha, beta, kappa, sigma, eps)
            mcukf_sim.read_data(states, real_obs)
            mcukf_states_mean, mcukf_MSE1, mcukf_MSE, mc_count, _ = mcukf_sim.run(filter_init, obs_noise, repeat)
    except KeyError:
        pass
    print("Simulation done.")

    # Build a data set
    description = "Non-Gaussian test."
    data_summarizes = {
                    'description': description,
                    'states dimension': states_dimension, 'obs dimension': states_dimension,
                    'repeat': repeat, 'time': t, 'ts': Ts, 'q': q, 'r': r,
                    'time line': time_line.tolist(),
                    'states': {'system states': states.tolist()},
                    'noises': {'obs noise': obs_noise[0].tolist(), 'add noise': additional_noise.tolist()},
                    'observations': {'noise_free observation': real_obs.tolist(), 'noise observation': (real_obs+obs_noise[0]).tolist()},
                    'mse': {}, 'mse1': {}, 'parameters': {}, 'run time': {}
                    }
    if "ukf_states_mean" in locals().keys():
        data_summarizes['states']['ukf states'] = ukf_states_mean.tolist()
        data_summarizes['mse']['ukf mse'] = ukf_MSE.tolist()
        data_summarizes['mse1']['ukf mse1'] = ukf_MSE1.tolist()
        data_summarizes['parameters']['ukf parameters'] = {'alpha': alpha, 'beta': beta, 'kappa': kappa}
        data_summarizes['run time']['ukf run time'] = ukf_sim.run_time
    if "mcukf_states_mean" in locals().keys():
        data_summarizes['states']['mcukf states'] = mcukf_states_mean.tolist()
        data_summarizes['mse']['mcukf mse'] = mcukf_MSE.tolist()
        data_summarizes['mse1']['mcukf mse1'] = mcukf_MSE1.tolist()
        data_summarizes['parameters']['mcukf parameters'] = {'alpha': alpha, 'beta': beta, 'kappa': kappa, 'sigma': sigma, 'eps': eps}
        data_summarizes['run time']['mcukf run time'] = mcukf_sim.run_time
    if "mckf_states_mean" in locals().keys():
        data_summarizes['parameters']['mckf parameters'] = {'sigma': sigma, 'eps': eps}
        data_summarizes['states']['mckf states'] = mckf_states_mean.tolist()
        data_summarizes['mse']['mckf mse'] = mckf_MSE.tolist()
        data_summarizes['mse1']['mckf mse1'] = mckf_MSE1.tolist()
        data_summarizes['run time']['mckf run time'] = mckf_sim.run_time
    if "ekf_states_mean" in locals().keys():
        data_summarizes['states']['ekf states'] = ekf_states_mean.tolist()
        data_summarizes['mse']['ekf mse'] = ekf_MSE.tolist()
        data_summarizes['mse1']['ekf mse1'] = ekf_MSE1.tolist()
        data_summarizes['run time']['ekf run time'] = ekf_sim.run_time
    if "mcekf_states_mean" in locals().keys():
        data_summarizes['parameters']['mcekf parameters'] = {'sigma': sigma, 'eps': eps}
        data_summarizes['states']['mcekf states'] = mcekf_states_mean.tolist()
        data_summarizes['mse']['mcekf mse'] = mcekf_MSE.tolist()
        data_summarizes['mse1']['mcekf mse1'] = mcekf_MSE1.tolist()
        data_summarizes['run time']['mcekf run time'] = mcekf_sim.run_time

    return data_summarizes
