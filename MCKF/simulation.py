from ukf.ukf_sim import UKF_Sim
from mcukf.mcukf_sim import MCUKF_Sim
from nonlinear_system import NonlinearSys
from data_manager import Manager
import numpy as np


states_dimension = 3
obs_dimension = 1
repeat = 100
t = 50
Ts = 0.1
N = int(t/Ts)
alpha = 1e-3
beta = 10
kappa = 0
q = 0
r = 100
# MCUKF part
sigma = 6
eps = 1e-8
# testing parameters
add_r = 400
additional_noise = add_r*np.random.randn(obs_dimension, N)
filter_init = ([3e5, -2e4, 1e-3], [3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
# System
mean_ukf_states = 0
mean_mcukf_states = 0
sys = NonlinearSys(states_dimension, obs_dimension, t, Ts, q, r, additional_noise)
sys.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 1e-3], [1e6, 4e6, 1e-6])

time_line, states, real_obs, sensor = sys.run()
for i in range(repeat):
    # 顺序: x维度, y维度, 时间, 采样间隔, α, β, kappa, q, r
    ukf_sim = UKF_Sim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, q, r, additional_noise)
    ukf_sim.states_init(filter_init)
    ukf_sim.read_data(states, real_obs)
    _, ukf_states = ukf_sim.run()
    ukf_MSE_1 = ukf_sim.MSE()
    mean_ukf_states += ukf_states

    # 顺序: x维度, y维度, 时间, 采样间隔, α, β, kappa, sigma, eps, q, r
    mcukf_sim = MCUKF_Sim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, sigma, eps, q, r, additional_noise)
    mcukf_sim.states_init(filter_init)
    mcukf_sim.read_data(states, real_obs)
    _, mcukf_states = mcukf_sim.run()
    mcukf_MSE_1 = mcukf_sim.MSE()
    mean_mcukf_states += mcukf_states
    # MSE
    ukf_MSE_1 += ukf_MSE_1
    mcukf_MSE_1 += mcukf_MSE_1
ukf_MSE_1 = ukf_MSE_1/repeat
ukf_MSE = ukf_MSE_1.sum(axis=1)/N
mean_ukf_states = mean_ukf_states/repeat
mcukf_MSE_1 = mcukf_MSE_1/repeat
mcukf_MSE = mcukf_MSE_1.sum(axis=1)/N
mean_mcukf_states = mean_mcukf_states/repeat
# build data set
data_summarizes = {
                   'shapes': {'states dimension': states_dimension, 'obs dimension': states_dimension},
                   'parameters': {'repeat': repeat, 'time': t, 'ts': Ts, 'alpha': alpha,
                                  'beta': beta, 'kappa': kappa, 'q': q, 'r': r, 'add noise': add_r,
                                  'sigma': sigma, 'eps': eps},
                   'time line': time_line.tolist(),
                   'states': {'system states': states.tolist(), 'ukf states': mean_ukf_states.tolist(),
                              'mcukf states': mean_mcukf_states.tolist()},
                   'observations': {'sensor observation': sensor.tolist(), 'noise_free observation': real_obs.tolist()},
                   'mse1': {'ukf mse1': ukf_MSE_1.tolist(), 'mcukf mse1': mcukf_MSE_1.tolist()},
                   'mse': {'ukf mse': ukf_MSE.tolist(), 'mcukf mse': mcukf_MSE.tolist()}
                   }
# data manager
manager = Manager()
manager.save_data(data_summarizes)
manager.read_data()
manager.plot_all()
manager.show()
