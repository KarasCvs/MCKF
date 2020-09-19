from functions import nonlinear_func as N_func
from simulations.filter_sim_base import FilterSim
from filters.ukf import Ukf


class UkfSim(FilterSim):
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, alpha_, beta_, ki_, q_, r_):
        FilterSim.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.ukf_init(alpha_, beta_, ki_)

    # --------------------------------UKF init---------------------------------- #
    def ukf_init(self, alpha_, beta_, ki_):
        self.ukf = Ukf()
        self.ukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.ukf.ut_init(alpha_, beta_, ki_)
        self.ukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        states_mean = 0
        mse1 = 0
        for j in range(repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P = \
                    self.ukf.estimate(self.states[:, i-1],
                                      self.obs[:, i]+obs_noise[j][:, i],
                                      self.P, i)
            states_mean += self.states
            mse1 += self.MSE()
        states_mean /= repeat
        mse1 /= repeat
        mse = mse1.sum(axis=1)/self.N
        return self.time_line, states_mean, mse1, mse
