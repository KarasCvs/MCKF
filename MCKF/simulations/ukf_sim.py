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

    def run(self):
        # --------------------------------main procedure---------------------------------- #
        for i in range(1, self.N):
            self.states[:, i], self.P = self.ukf.estimate(self.states[:, i-1], self.sensor[:, i], self.P, i)
        return self.time_line, self.states
