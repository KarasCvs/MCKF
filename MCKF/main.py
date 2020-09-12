from ukf.ukf_sim import UKF_Sim


if __name__ == "__main__":
    ukf_sim = UKF_Sim(3, 1)
    ukf_sim.states_init([3e5, 2e4, 1e-3], [3e5, 2e4, 3e-5], [1e6, 4e6, 10])
    ukf_sim.run()
    ukf_sim.plot()
