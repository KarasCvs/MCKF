from data_manager import Manager
import matplotlib.pyplot as plt


mg = Manager()
states = []


targets = mg.locate({"description": "Gaussian test.", "repeat": 100})
mg.read_data(targets[0])
timeline = mg.find("time line")
ekf_mse1 = mg.find('ekf mse1')
ukf_mse1 = mg.find('ukf mse1')
for i in range(3):
    plt.figure(i)
    plt.plot(timeline, ekf_mse1[i], label="ekf", color='r', linewidth=1, linestyle='-')
    plt.plot(timeline, ukf_mse1[i], label="ukf", color='blue', linewidth=1, linestyle='-')
    plt.xlabel('Time(s)')
    plt.ylabel('MSE')
    plt.title(f'Mean-Square Error of x{i+1}')
for target in targets:
    mg.read_data(target)
    sigma = mg.find('sigma')
    # print(f'sigma = {sigma}')
    state = mg.find("mcekf states")
    mse1 = mg.find("mcekf mse1")
    for i in range(3):
        plt.figure(i)
        plt.plot(timeline, mse1[i], label=f"sigma={sigma}", linewidth=1, linestyle='-.')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 15)
        plt.title(f'Mean-Square Error(MSE) of x{i+1}')
plt.show()