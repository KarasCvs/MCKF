from ukf.ukf_sim import UKF_Sim
from mcukf.mcukf_sim import MCUKF_Sim

if __name__ == "__main__":
    # 顺序: x维度, y维度, 时间, 采样间隔, α, β, ki, q, r
    ukf_sim = UKF_Sim(3, 1, 30, 0.01, 1e-3, 2, 0, 0, 100)
    ukf_sim.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 1e-5], [1e6, 4e6, 10])
    ukf_sim.run()
    # sim.system_only()
    ukf_sim.plot()

    # 顺序: x维度, y维度, 时间, 采样间隔, α, β, ki, sigma, eps, q, r
    # mcukf_sim = MCUKF_Sim(3, 1, 30, 0.1, 6, 1e-5, 1e-3, 2, 0, 0, 100)
    # mcukf_sim.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 1e-5], [1e6, 4e6, 10])
    # mcukf_sim.run()
    # # sim.system_only()
    # mcukf_sim.plot()
