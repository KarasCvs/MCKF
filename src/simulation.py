# Every class with "1" mark means fixed-point iteration method. Without, means Dan.S's method
from filters import MCUKF1 as MCUKF
from filters import UKF as UKF
from filters import EKF as EKF
from filters import MCEKF2 as MCEKF
from filters import IMCEKF as IMCEKF
from filters import IMCEKF3 as IMCEKF3
# from filters import MCEKF1 as MCEKF
from filters import NonlinearSys as Sys
import numpy as np
import math


class Simulation():
    def __init__(self, repeat_):
        # --------------------------------- System parameters --------------------------------- #
        self.description = "non-Gaussian impulse, rocket simulation"
        self.repeat = repeat_
        self.t = 30
        self.Ts = 0.1
        self.N = int(self.t/self.Ts)
        # noise
        self.q = np.diag([1, 1])
        self.r = 5                # 20 for non-Gaussian
        # Filter parameters
        self.q_filter = np.diag([np.sqrt(0.1), np.sqrt(0.1)])
        self.r_filter = 5         # 20 for non-Gaussian
        # MCUKF part
        self.eps = 1e-8
        # UKF part
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 3
        # System initial values
        # self.sys_init = [10]
        self.sys_init = [0, 0]   # [3e5, -2e4, 1e-3]
        # Filter initial values
        # self.filter_init = ([11], [1])
        self.filter_init = ([0, 0], [math.sqrt(0.1), math.sqrt(0.1)])  # default ([3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
        self.states_dimension = len(self.sys_init)
        self.obs_dimension = 1
        self.step()
        # --------------------------------------------------------------------------------- #
        # --------------------------------- Impulse noise --------------------------------- #
        impulse = 1
        self.additional_sys_noise = []
        for _ in range(self.repeat):
            additional_sys_noise = np.asmatrix(np.zeros((self.states_dimension, self.N)))
            for i in range(self.N):
                if np.random.randint(0, 100) < 5:
                    additional_sys_noise[:, i] = np.dot(
                                                        np.random.choice((-1, 1), 2) * np.diag([1, 1]),
                                                        np.random.randint(2, 5, size=(self.states_dimension, 1)))
            self.additional_sys_noise.append(additional_sys_noise)
        self.additional_obs_noise = []
        for _ in range(self.repeat):
            additional_obs_noise = np.zeros((self.obs_dimension, self.N))
            for i in range(self.N):
                if np.random.randint(0, 100) < 5:
                    additional_obs_noise[:, i] = np.random.choice((-1, 1)) * np.random.randint(200, 500)
            self.additional_obs_noise.append(additional_obs_noise)
        if not impulse:
            self.additional_sys_noise = np.zeros(self.repeat)
            self.additional_obs_noise = np.zeros(self.repeat)
        # --------------------------------------------------------------------------------- #

# --------------------------------- System --------------------------------- #
    def sys_run(self):
        # System initial
        sys = Sys(self.states_dimension, self.obs_dimension, self.t, self.Ts, self.q, self.r, self.input, self.repeat)
        # Initial noise lists.
        self.obs_noise, self.sys_noise = sys.noise_init(self.additional_sys_noise, self.additional_obs_noise)
        # Rocket system initiation
        sys.states_init(self.sys_init)
        self.time_line, self.states, self.real_obs, self.sensor = sys.run()
        # Define dataset
        self.data_summarizes = {
                'description': self.description,
                'states dimension': self.states_dimension, 'obs dimension': self.obs_dimension,
                'repeat': self.repeat, 'time': self.t, 'ts': self.Ts, 'q': self.q.tolist(), 'r': self.r,
                'time line': self.time_line.tolist(),
                'states': {},  # {'system states': self.states},
                'noises': {'obs noise': self.obs_noise[0].tolist(), 'add noise': self.additional_obs_noise[0].tolist()},
                'observations': {},  # {'noise_free observation': self.real_obs, 'noise observation': self.sensor},
                'ta_mse': {}, 'mse': {}, 'parameters': {}, 'run time': {}
                }
        # sys.plot()
# --------------------------------------------------------------------------------- #

    def filter_run(self, sigma_=2, **kwargs):
        self.sigma = sigma_
        filter_init = self.filter_init
        # --------------------------------- Filter parameters --------------------------------- #
        print("Simulation started.")
        # Rocket system
        self.parameters = (self.states_dimension, self.obs_dimension, self.t, self.Ts,
                           self.q_filter, self.r_filter, self.alpha, self.beta, self.kappa, self.sigma, self.eps,
                           self.data_summarizes, self.repeat, self.input)
        # --------------------------------------------------------------------------------- #

# --------------------------------- Run filter --------------------------------- #
        # EKF part
        try:
            if kwargs['ekf']:
                ekf_sim = EKF(self.parameters)
                self.data_summarizes = ekf_sim.run(filter_init, self.states, self.sensor)
        except KeyError:
            pass

        # UKF part
        # UKF initial, order: x dimension, y dimension, run time, time space, self.q_filter, self.r_filter, α, β, self.kappa
        try:
            if kwargs['ukf']:
                ukf_sim = UKF(self.parameters)
                self.data_summarizes = ukf_sim.run(filter_init, self.states, self.sensor)
        except KeyError:
            pass

        # IMCEKF part
        try:
            if kwargs['imcekf']:
                imcekf_sim = IMCEKF(self.parameters)
                self.data_summarizes = imcekf_sim.run(filter_init, self.states, self.sensor)
        except KeyError:
            pass
        # IMCEKF2 part
        try:
            if kwargs['imcekf3']:
                imcekf3_sim = IMCEKF3(self.parameters)
                self.data_summarizes = imcekf3_sim.run(filter_init, self.states, self.sensor, 1)
        except KeyError:
            pass

        # MCEKF part
        try:
            if kwargs['mcekf']:
                mcekf_sim = MCEKF(self.parameters)
                self.data_summarizes = mcekf_sim.run(filter_init, self.states, self.sensor)
        except KeyError:
            pass

        # MCUKF part
        # MCUKF initial, order: x dimension, y dimension, run time, time space, self.q_filter, self.r_filter, α, β, self.kappa, self.sigma, self.eps
        try:
            if kwargs['mcukf']:
                mcukf_sim = MCUKF(self.parameters)
                self.data_summarizes = mcukf_sim.run(filter_init, self.states, self.sensor)
        except KeyError:
            pass
        print("Simulation done.")
# --------------------------------------------------------------------------------- #
        return self.data_summarizes

        # Generate a step input
    def step(self):
        self.input = np.zeros(self.N)
        for i in range(self.N):
            if i > (5):
                self.input[i] = 1
                if i > (self.N/2):
                    self.input[i] = 2
        return self.input

