from simulations.ukf_sim import UkfSim
from simulations.mcukf_sim import McukfSim
from simulations.nonlinear_system import NonlinearSys
from data_manager import Manager
import numpy as np


# system part
states_dimension = 3
obs_dimension = 1
repeat = 100
t = 50
Ts = 0.1
N = int(t/Ts)
# MCUKF part
sigma = 2
eps = 1e-6
# UKF part
alpha = 1e-3
beta = 10
kappa = 0
# noise
q = 0
r = 22
add_r = 300
additional_noise = add_r*np.random.randn(obs_dimension, N)

# System initial
mean_ukf_states = 0
mean_mcukf_states = 0
sys = NonlinearSys(states_dimension, obs_dimension, t, Ts, q, r, add_r)

# system initial values
sys.states_init([3e5, -2e4, 1e-3])
# filter initial values
filter_init = ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
# order: x dimension, y dimension, run time, time space, α, β, kappa, q, r
ukf_sim = UkfSim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, q, r)
ukf_sim.states_init(filter_init)
# order: x dimension, y dimension, run time, time space, α, β, kappa, sigma, eps, q, r
mcukf_sim = McukfSim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, sigma, eps, q, r)
mcukf_sim.states_init(filter_init)
ukf_MSE_1, mcukf_MSE_1, mc_count = 0, 0, 0
time_line, states, real_obs, _ = sys.run()
for i in range(repeat):
    sensor = real_obs + r*np.random.randn(obs_dimension, N) + add_r*np.random.randn(obs_dimension, N)
    # Ukf run
    ukf_sim.states_init(filter_init)
    ukf_sim.read_data(states, sensor)
    _, ukf_states = ukf_sim.run()
    ukf_MSE_1 += ukf_sim.MSE()
    mean_ukf_states += ukf_states
    # Mcukf run
    mcukf_sim.states_init(filter_init)
    mcukf_sim.read_data(states, sensor)
    _, mcukf_states, count = mcukf_sim.run()
    mc_count += count
    mcukf_MSE_1 += mcukf_sim.MSE()
    mean_mcukf_states += mcukf_states
mc_count = mc_count/(repeat*N)
ukf_MSE_1 = ukf_MSE_1/repeat
ukf_MSE = ukf_MSE_1.sum(axis=1)/N
mean_ukf_states = mean_ukf_states/repeat
mcukf_MSE_1 = mcukf_MSE_1/repeat
mcukf_MSE = mcukf_MSE_1.sum(axis=1)/N
mean_mcukf_states = mean_mcukf_states/repeat

# build a data set
data_summarizes = {
                   'shapes': {'states dimension': states_dimension, 'obs dimension': states_dimension},
                   'parameters': {'repeat': repeat, 'time': t, 'ts': Ts, 'alpha': alpha,
                                  'beta': beta, 'kappa': kappa, 'q': q, 'r': r, 'add noise': add_r,
                                  'sigma': sigma, 'eps': eps},
                   'time line': time_line.tolist(), 'mcukf count': mc_count,
                   'states': {'system states': states.tolist(), 'ukf states': mean_ukf_states.tolist(),
                              'mcukf states': mean_mcukf_states.tolist()},
                   'observations': {'sensor observation': sensor.tolist(), 'noise_free observation': real_obs.tolist()},
                   'mse1': {'ukf mse1': ukf_MSE_1.tolist(), 'mcukf mse1': mcukf_MSE_1.tolist()},
                   'mse': {'ukf mse': ukf_MSE.tolist(), 'mcukf mse': mcukf_MSE.tolist()}
                   }
# data manager
manager = Manager()
manager.save_data(data_summarizes)
# manager.view_data(6data_summarizes)
manager.read_data()
print(manager.find("mcukf count"))
manager.plot_all()
manager.show()
