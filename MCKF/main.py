from ukf.ukf_sim import UKF_Sim


if __name__ == "__main__":
    sim = UKF_Sim(3, 1, 30, 0.01)
    sim.ukf_init(1e-3, 2, 0)
    sim.noise_init(0, 100)
    sim.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 1e-5], [1e6, 4e6, 10])
    # sim.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 1e-3], [1, 1, 1])
    sim.run()
    # sim.system_only()
    sim.plot()
