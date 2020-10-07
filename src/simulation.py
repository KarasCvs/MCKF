# Every class with "1" mark means fixed-point iteration method. Without, means Dan.S's method
from filters import Mcukf as Mcukf
from filters import Ukf as Ukf
from filters import Mckf2 as Mckf
from filters import Mckf1 as Mckf1
from filters import Ekf as Ekf
from filters import Mcekf2 as Mcekf
from filters import Mcekf1 as Mcekf1
from filters import NonlinearSys as Sys
import numpy as np


class Simulation():
    def __init__(self, repeat_):
        # system part
        self.description = "non-Gaussian impulse, rocket simulation"
        self.states_dimension = 3
        self.obs_dimension = 1
        self.repeat = repeat_
        self.t = 50
        self.Ts = 0.1
        # noise
        self.q = 0
        self.r = 20

        # additional noise
        self.additional_noise = []
        for _ in range(self.repeat):
            additional_noise = np.zeros((self.obs_dimension, int(self.t/self.Ts)))
            for i in range(int(self.t/self.Ts)):
                if np.random.randint(0, 100) < 5:
                    additional_noise[:, i] = np.random.choice((-1, 1)) * np.random.randint(500, 700)
            self.additional_noise.append(additional_noise)

        # self.additional_noise = [20*np.random.randn(self.obs_dimension, int(self.t/self.Ts)) for _ in range(self.repeat)]
        # self.additional_noise = [np.zeros((self.obs_dimension, int(self.t/self.Ts))) for _ in range(self.repeat)]

    def sys_run(self):
        # System initial
        sys = Sys(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r)
        # Rocket system
        sys.states_init([3e5, -2e4, 1e-3])
        # Robot movement
        # sys.states_init([2, 1, 5, 3, 5, 12])
        self.time_line, self.states, self.real_obs = sys.run()
        # Initial noise lists.
        self.obs_noise = sys.noise_init(self.repeat, self.additional_noise)
        # sys.plot()

    def filter_run(self, sigma_=2, **kwargs):
        # MCUKF part
        self.sigma = sigma_
        self.eps = 1e-6
        # UKF part
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 3
        print("Simulation started.")
        # Filter initial values
        # Rocket system
        filter_init = ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])  # default ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
        # Robot movement
        # filter_init = ([0, 0, 2, 0, 0, 2], [1, 1, 5, 1, 1, 10])
        # Mckf part
        # Mckf initial, order: x dimension, y dimension, run time, time space, self.sigma, self.eps
        try:
            if kwargs['mckf']:
                print('Mckf runing.')
                mckf_sim = Mckf(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.sigma, self.eps)
                mckf_sim.read_data(self.states, self.real_obs)
                mckf_states_mean, mckf_MSE1, mckf_MSE, _, _ = mckf_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass

        try:
            if kwargs['mckf1']:
                print('Mckf1 runing.')
                mckf1_sim = Mckf1(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.sigma, self.eps)
                mckf1_sim.read_data(self.states, self.real_obs)
                mckf1_states_mean, mckf1_MSE1, mckf1_MSE, _, _ = mckf1_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass

        # Ekf part
        try:
            if kwargs['ekf']:
                print('Ekf running.')
                ekf_sim = Ekf(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r)
                ekf_sim.read_data(self.states, self.real_obs)
                ekf_states_mean, ekf_MSE1, ekf_MSE, _, _ = ekf_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass

        # Ukf part
        # Ukf initial, order: x dimension, y dimension, run time, time space, self.q, self.r, α, β, self.kappa
        try:
            if kwargs['ukf']:
                print('Ukf running.')
                ukf_sim = Ukf(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.alpha, self.beta, self.kappa)
                ukf_sim.read_data(self.states, self.real_obs)
                ukf_states_mean, ukf_MSE1, ukf_MSE, _, _ = ukf_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass

        # Mcekf part
        try:
            if kwargs['mcekf']:
                print('Mcekf running.')
                mcekf_sim = Mcekf(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.sigma)
                mcekf_sim.read_data(self.states, self.real_obs)
                mcekf_states_mean, mcekf_MSE1, mcekf_MSE, _, _ = mcekf_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass
        try:
            if kwargs['mcekf1']:
                print('Mcekf1 running.')
                mcekf1_sim = Mcekf1(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.sigma, self.eps)
                mcekf1_sim.read_data(self.states, self.real_obs)
                mcekf1_states_mean, mcekf1_MSE1, mcekf1_MSE, _, _ = mcekf1_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass

        # Mcukf part
        # Mcukf initial, order: x dimension, y dimension, run time, time space, self.q, self.r, α, β, self.kappa, self.sigma, self.eps
        try:
            if kwargs['mcukf']:
                print('Mcukf running.')
                mcukf_sim = Mcukf(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.alpha, self.beta, self.kappa, self.sigma, self.eps)
                mcukf_sim.read_data(self.states, self.real_obs)
                mcukf_states_mean, mcukf_MSE1, mcukf_MSE, mc_count, _ = mcukf_sim.run(filter_init, self.obs_noise, self.repeat)
        except KeyError:
            pass
        print("Simulation done.")
        # Build a data set
        data_summarizes = {
                        'description': self.description,
                        'states dimension': self.states_dimension, 'obs dimension': self.obs_dimension,
                        'repeat': self.repeat, 'time': self.t, 'ts': self.Ts, 'q': self.q, 'r': self.r,
                        'time line': self.time_line.tolist(),
                        'states': {'system states': self.states.tolist()},
                        'noises': {'obs noise': self.obs_noise[0].tolist(), 'add noise': self.additional_noise[0].tolist()},
                        'observations': {'noise_free observation': self.real_obs.tolist(), 'noise observation': (self.real_obs+self.obs_noise[0]).tolist()},
                        'mse': {}, 'mse1': {}, 'parameters': {}, 'run time': {}
                        }
        if "ukf_states_mean" in locals().keys():
            data_summarizes['states']['ukf states'] = ukf_states_mean.tolist()
            data_summarizes['mse']['ukf mse'] = ukf_MSE.tolist()
            data_summarizes['mse1']['ukf mse1'] = ukf_MSE1.tolist()
            data_summarizes['parameters']['ukf parameters'] = {'alpha': self.alpha, 'beta': self.beta, 'kappa': self.kappa}
            data_summarizes['run time']['ukf run time'] = ukf_sim.run_time
        if "mcukf_states_mean" in locals().keys():
            data_summarizes['states']['mcukf states'] = mcukf_states_mean.tolist()
            data_summarizes['mse']['mcukf mse'] = mcukf_MSE.tolist()
            data_summarizes['mse1']['mcukf mse1'] = mcukf_MSE1.tolist()
            data_summarizes['parameters']['mcukf parameters'] = {'alpha': self.alpha, 'beta': self.beta, 'kappa': self.kappa, 'sigma': self.sigma, 'eps': self.eps}
            data_summarizes['run time']['mcukf run time'] = mcukf_sim.run_time
        if "mckf_states_mean" in locals().keys():
            data_summarizes['parameters']['mckf parameters'] = {'sigma': self.sigma, 'eps': self.eps}
            data_summarizes['states']['mckf states'] = mckf_states_mean.tolist()
            data_summarizes['mse']['mckf mse'] = mckf_MSE.tolist()
            data_summarizes['mse1']['mckf mse1'] = mckf_MSE1.tolist()
            data_summarizes['run time']['mckf run time'] = mckf_sim.run_time
        if "mckf1_states_mean" in locals().keys():
            data_summarizes['parameters']['mckf1 parameters'] = {'sigma': self.sigma, 'eps': self.eps}
            data_summarizes['states']['mckf1 states'] = mckf1_states_mean.tolist()
            data_summarizes['mse']['mckf1 mse'] = mckf1_MSE.tolist()
            data_summarizes['mse1']['mckf1 mse1'] = mckf1_MSE1.tolist()
            data_summarizes['run time']['mckf1 run time'] = mckf1_sim.run_time
        if "ekf_states_mean" in locals().keys():
            data_summarizes['states']['ekf states'] = ekf_states_mean.tolist()
            data_summarizes['mse']['ekf mse'] = ekf_MSE.tolist()
            data_summarizes['mse1']['ekf mse1'] = ekf_MSE1.tolist()
            data_summarizes['run time']['ekf run time'] = ekf_sim.run_time
        if "mcekf_states_mean" in locals().keys():
            data_summarizes['parameters']['mcekf parameters'] = {'sigma': self.sigma, 'eps': self.eps}
            data_summarizes['states']['mcekf states'] = mcekf_states_mean.tolist()
            data_summarizes['mse']['mcekf mse'] = mcekf_MSE.tolist()
            data_summarizes['mse1']['mcekf mse1'] = mcekf_MSE1.tolist()
            data_summarizes['run time']['mcekf run time'] = mcekf_sim.run_time
        if "mcekf1_states_mean" in locals().keys():
            data_summarizes['parameters']['mcekf1 parameters'] = {'sigma': self.sigma, 'eps': self.eps}
            data_summarizes['states']['mcekf1 states'] = mcekf1_states_mean.tolist()
            data_summarizes['mse']['mcekf1 mse'] = mcekf1_MSE.tolist()
            data_summarizes['mse1']['mcekf1 mse1'] = mcekf1_MSE1.tolist()
            data_summarizes['run time']['mcekf1 run time'] = mcekf1_sim.run_time
        return data_summarizes
