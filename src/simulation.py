# Every class with "1" mark means fixed-point iteration method. Without, means Dan.S's method
from filters import MCUKF1 as MCUKF
from filters import UKF as UKF
from filters import EKF as EKF
# from filters import MCEKF2 as MCEKF
from filters import IMCEKF as IMCEKF
from filters import MCEKF1 as MCEKF
from filters import NonlinearSys as Sys
import numpy as np


class Simulation():
    def __init__(self, repeat_):
        # system part
        self.description = "non-Gaussian impulse, rocket simulation"
        self.states_dimension = 3
        self.obs_dimension = 1
        self.repeat = repeat_
        self.t = 30
        self.Ts = 0.1
        self.N = int(self.t/self.Ts)
        # noise
        self.q = np.diag([5, 0.2, 0])
        # np.array((1, 3, 1e-4, 2, 1, 1e-5)).reshape(self.states_dimension, 1)
        # np.array((1e-2, 1e-3, 0)).reshape(self.states_dimension, 1)
        self.r = 20        # 20 for non-Gaussian

        # Impulse noise
        self.additional_sys_noise = []
        for _ in range(self.repeat):
            additional_sys_noise = np.asmatrix(np.zeros((self.states_dimension, self.N)))
            for i in range(self.N):
                if np.random.randint(0, 100) < 5:
                    additional_sys_noise[:, i] = np.dot(
                                                        np.random.choice((-1, 1), 3) * np.diag([1, 1, 0]),
                                                        np.random.randint(100, 200, size=(self.states_dimension, 1)))
            self.additional_sys_noise.append(additional_sys_noise)

        self.additional_obs_noise = []
        for _ in range(self.repeat):
            additional_obs_noise = np.zeros((self.obs_dimension, self.N))
            for i in range(self.N):
                if np.random.randint(0, 100) < 5:
                    additional_obs_noise[:, i] = np.random.choice((-1, 1)) * np.random.randint(500, 700)
            self.additional_obs_noise.append(additional_obs_noise)
        # self.additional_obs_noise = np.zeros(self.repeat)
        # self.additional_sys_noise = np.zeros(self.repeat)

    def sys_run(self):
        # System initial
        sys = Sys(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.repeat)
        # Initial noise lists.
        self.obs_noise, self.sys_noise = sys.noise_init(self.additional_sys_noise, self.additional_obs_noise)
        # Rocket system
        sys.states_init([3e5, -2e4, 1e-3])
        self.time_line, self.states, self.real_obs, self.sensor = sys.run()
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
        filter_init = ([3e5, -2e4, 9e-4], [1e2, 4e2, 1e-6])  # default ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])

############################################################################### Run filters ######################################################################
        # EKF part
        try:
            if kwargs['ekf']:
                print('EKF running.')
                ekf_sim = EKF(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r)
                ekf_states_mean, ekf_MSE1, ekf_MSE, _, _ = ekf_sim.run(filter_init, self.states, self.sensor, self.repeat)
        except KeyError:
            pass

        # UKF part
        # UKF initial, order: x dimension, y dimension, run time, time space, self.q, self.r, α, β, self.kappa
        try:
            if kwargs['ukf']:
                print('UKF running.')
                ukf_sim = UKF(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.alpha, self.beta, self.kappa)
                ukf_states_mean, ukf_MSE1, ukf_MSE, _, _ = ukf_sim.run(filter_init, self.states, self.sensor, self.repeat)
        except KeyError:
            pass

        # IMCEKF part
        try:
            if kwargs['imcekf']:
                print('IMCEKF running.')
                imcekf_sim = IMCEKF(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.sigma, self.eps)
                imcekf_states_mean, imcekf_MSE1, imcekf_MSE, _, _ = imcekf_sim.run(filter_init, self.states, self.sensor, self.repeat)
        except KeyError:
            pass

        # MCEKF part
        try:
            if kwargs['mcekf']:
                print('MCEKF running.')
                mcekf_sim = MCEKF(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.sigma)
                mcekf_states_mean, mcekf_MSE1, mcekf_MSE, _, _ = mcekf_sim.run(filter_init, self.states, self.sensor, self.repeat)
        except KeyError:
            pass

        # MCUKF part
        # MCUKF initial, order: x dimension, y dimension, run time, time space, self.q, self.r, α, β, self.kappa, self.sigma, self.eps
        try:
            if kwargs['mcukf']:
                print('MCUKF running.')
                mcukf_sim = MCUKF(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q.tolist(), self.r, self.alpha, self.beta, self.kappa, self.sigma, self.eps)
                mcukf_states_mean, mcukf_MSE1, mcukf_MSE, mc_count, _ = mcukf_sim.run(filter_init, self.states, self.sensor, self.repeat)
        except KeyError:
            pass
        print("Simulation done.")

        # Build a data set
        data_summarizes = {
                        'description': self.description,
                        'states dimension': self.states_dimension, 'obs dimension': self.obs_dimension,
                        'repeat': self.repeat, 'time': self.t, 'ts': self.Ts, 'q': self.q.tolist(), 'r': self.r,
                        'time line': self.time_line.tolist(),
                        'states': {},  # {'system states': self.states},
                        'noises': {'obs noise': self.obs_noise[0].tolist(), 'add noise': self.additional_obs_noise[0].tolist()},
                        'observations': {},  # {'noise_free observation': self.real_obs, 'noise observation': self.sensor},
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
        if "imcekf_states_mean" in locals().keys():
            data_summarizes['parameters']['imcekf parameters'] = {'sigma': self.sigma, 'eps': self.eps}
            data_summarizes['states']['imcekf states'] = imcekf_states_mean.tolist()
            data_summarizes['mse']['imcekf mse'] = imcekf_MSE.tolist()
            data_summarizes['mse1']['imcekf mse1'] = imcekf_MSE1.tolist()
            data_summarizes['run time']['imcekf run time'] = imcekf_sim.run_time
        return data_summarizes
