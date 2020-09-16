from ukf.ukf_sim import UKF_Sim
from mcukf.mcukf_sim import MCUKF_Sim
import matplotlib.pyplot as plt
from nonlinear_system import NonlinearSys


if __name__ == "__main__":
    states_dimension = 3
    obs_dimension = 1
    repeat = 1
    t = 50
    Ts = 0.1
    N = int(t/Ts)
    alpha = 1e-3
    beta = 10
    kappa = 0
    q = 0
    r = 100
    # MCUKF part
    sigma = 2
    eps = 1e-6

    # System
    mean_ukf_MSE = 0
    mean_mcukf_MSE = 0
    sys = NonlinearSys(states_dimension, obs_dimension, t, Ts, q, r)
    sys.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 1e-3], [1e6, 4e6, 1e-6])
    time_line, states, real_obs, sensor = sys.run()
    for i in range(repeat):
        # 顺序: x维度, y维度, 时间, 采样间隔, α, β, kappa, q, r
        ukf_sim = UKF_Sim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, q, r)
        ukf_sim.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
        ukf_sim.read_data(states, real_obs)
        _, ukf_states = ukf_sim.run()
        ukf_MSE_1 = ukf_sim.MSE()

        # 顺序: x维度, y维度, 时间, 采样间隔, α, β, kappa, sigma, eps, q, r
        mcukf_sim = MCUKF_Sim(states_dimension, obs_dimension, t, Ts, alpha, beta, kappa, sigma, eps, q, r)
        mcukf_sim.states_init([3e5, -2e4, 1e-3], [3e5, -2e4, 9e-4], [1e6, 4e6, 1e-6])
        mcukf_sim.read_data(states, real_obs)
        _, mcukf_states = mcukf_sim.run()
        mcukf_MSE_1 = mcukf_sim.MSE()

        # MSE
        ukf_MSE_1 += ukf_MSE_1
        mcukf_MSE_1 += mcukf_MSE_1
    ukf_MSE_1 = ukf_MSE_1/repeat
    mcukf_MSE_1 = mcukf_MSE_1/repeat
    ukf_MSE = ukf_MSE_1.sum(axis=1)/N
    mcukf_MSE = mcukf_MSE_1.sum(axis=1)/N

    # Plot
    print(f"ukf_MSE =\n{ukf_MSE}\nmcukf_MSE =\n{mcukf_MSE}")
    for i in range(states_dimension):
        plt.figure(1)
        plt.subplot(100*states_dimension+11+i)
        plt.plot(time_line, ukf_states[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="UKF")
        plt.plot(time_line, mcukf_states[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="MCUKF")
        plt.plot(time_line, states[i, :].A.reshape(N,), linewidth=1, linestyle="-", label="Real State")
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.title("States")
    plt.figure(2)
    plt.plot(time_line, ukf_MSE_1[0, :].A.reshape(N,), linewidth=1, linestyle="-", label="ukf MSE")
    plt.plot(time_line, mcukf_MSE_1[0, :].A.reshape(N,), linewidth=1, linestyle="-", label="mcukf MSE")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title("MSE")
    plt.figure(3)
    plt.plot(time_line, sensor.A.reshape(N,), linewidth=1, linestyle="-", label="Sensor")
    plt.plot(time_line, real_obs.A.reshape(N,), linewidth=1, linestyle="-", label="Real obs")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.title("Observation")
plt.show()
