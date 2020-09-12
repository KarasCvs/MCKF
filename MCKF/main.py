from ukf.ukf_sim import UKF_Sim


if __name__ == "__main__":
    sim = UKF_Sim(3, 1, 30, 0.1)
    sim.ukf_init(5e-1, 2, 0)
    sim.noise_init(0, 1e4)
    sim.states_init([3e5, -2e4, 1e-3], [3e5, 2e4, 3e-5], [1e6, 4e6, 10])
    sim.run()
    # sim.system_only()
    sim.plot()
