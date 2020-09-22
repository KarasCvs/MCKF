from simulations.ukf_sim import UkfSim
from simulations.mcukf_sim import McukfSim
from simulations.nonlinear_system import NonlinearSys
from data_manager import Manager
import numpy as np


def sim_run(sigma_):
    mg = Manager()
    # system part
    states_dimension = 3
    obs_dimension = 1
    repeat = 100
    t = 50
    Ts = 0.1
    N = int(t/Ts)
    # MCUKF part
    sigma = sigma_
    eps = 1e-6
    # UKF part
    alpha = 1e-3
    beta = 2
    kappa = 0
    # noise
    q = 0
    r = 15.5
    add_r = 95
    additional_noise = add_r*np.random.randn(obs_dimension, N)

    # System initial
    sys = NonlinearSys(states_dimension, obs_dimension, t, Ts, q, r, add_r)
    sys.states_init([3e5, -2e4, 1e-3])
    time_line, states, real_obs = sys.run()
    # filter initial values
    filter_init = ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])

    # Ukf initial, order: x dimension, y dimension, run time, time space, α, β, kappa, q, r
    ukf_sim = UkfSim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, q, r)
    # Mcukf initial, order: x dimension, y dimension, run time, time space, repeat, α, β, kappa, sigma, eps, q, r
    mcukf_sim = McukfSim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, sigma, eps, q, r)
    print("Simulation started.")
    ukf_sim.read_data(states, real_obs)
    mcukf_sim.read_data(states, real_obs)
    obs_noise = ukf_sim.noise_init(additional_noise, repeat)
    _, ukf_states_mean, ukf_MSE1, ukf_MSE = ukf_sim.run(filter_init, obs_noise, repeat)
    _, mcukf_states_mean, mcukf_MSE1, mcukf_MSE, mc_count = mcukf_sim.run(filter_init, obs_noise, repeat)

    # build a data set
    description = "Non-Gaussian2."
    data_summarizes = {
                    'description': description,
                    'shapes': {'states dimension': states_dimension, 'obs dimension': states_dimension},
                    'parameters': {'repeat': repeat, 'time': t, 'ts': Ts, 'alpha': alpha,
                                   'beta': beta, 'kappa': kappa, 'q': q, 'r': r, 'add noise': add_r,
                                   'sigma': sigma, 'eps': eps},
                    'mse': {'ukf mse': ukf_MSE.tolist(), 'mcukf mse': mcukf_MSE.tolist()},
                    'mse1': {'ukf mse1': ukf_MSE1.tolist(), 'mcukf mse1': mcukf_MSE1.tolist()},
                    'time line': time_line.tolist(), 'mc iteration': mc_count,
                    'states': {'system states': states.tolist(), 'ukf states': ukf_states_mean.tolist(),
                               'mcukf states': mcukf_states_mean.tolist()},
                    'observations': {'noise_free observation': real_obs.tolist()}
                    }

    # data mananger
    mg.save_data(data_summarizes)
    # mg.view_data(data_summarizes)
    # mg.read_data()
    mg.plot_mse()
    # mg.plot_all()
    # mg.show()
