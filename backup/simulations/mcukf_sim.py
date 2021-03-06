from functions import nonlinear_func as N_func
from simulations.filter_sim_base import FilterSim
from filters.mcukf2 import Mcukf


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

    def run(self, init_parameters, obs_noise, repeat=1):
        # --------------------------------main procedure---------------------------------- #
        mc_count = 0
        states_mean = 0
        mse1 = 0
        for j in range(repeat):
            self.states_init(init_parameters)
            for i in range(1, self.N):
                self.states[:, i], self.P, count = \
                    self.mcukf.estimate(self.states[:, i-1],
                                        self.obs[:, i]+obs_noise[j][:, i],
                                        self.P, i)
                mc_count += count
            states_mean += self.states
            mse1 += self.MSE()
        states_mean /= repeat
        mse1 /= repeat
        mse = mse1.sum(axis=1)/self.N
        mc_count /= self.N*repeat
        return self.time_line, states_mean, mse1, mse, mc_count
