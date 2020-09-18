from functions import nonlinear_func as N_func
from filters.filter_sim_base import FilterSim
from filters.mcukf import Mcukf


class McukfSim(FilterSim):
    # --------------------------------init---------------------------------- #
    def __init__(self, states_dimension, obs_dimension, t, Ts, alpha_, beta_, ki_, sigma_, eps_, q_, r_):
        FilterSim.__init__(self, states_dimension, obs_dimension, t, Ts, q_, r_)
        self.mcukf_init(sigma_, eps_, alpha_, beta_, ki_)

    # --------------------------------MCUKF init---------------------------------- #
    def mcukf_init(self, sigma_, eps_, alpha_, beta_, ki_):
        self.mcukf = Mcukf()
        self.mcukf.filter_init(self.states_dimension, self.obs_dimension, self.noise_q, self.noise_r)
        self.mcukf.mc_init(sigma_, eps_)
        self.mcukf.ut_init(alpha_, beta_, ki_)
        self.mcukf.state_func(N_func.state_func, N_func.observation_func, self.Ts)

    def run(self):
        # --------------------------------main procedure---------------------------------- #
        for i in range(1, self.N):
            self.states[:, i], self.P = self.mcukf.estimate(self.states[:, i-1], self.sensor[:, i], self.P, i)
        return self.time_line, self.states
